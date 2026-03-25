# ============================================================
#  server.py  –  Parameter server con FedAvg (PyTorch + CNN)
#  Dataset  : CIFAR-10  (32×32×3, 10 clases)
#  Protocolo: igual que la versión MNIST original
#             INIT → TRAIN×N_rondas → EXPERIMENT_END
# ============================================================

import socket
import pickle
import csv
import os
import time
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ──────────────────────────────────────────────────────────────
#  CONFIG  ← los únicos valores que hay que tocar
# ──────────────────────────────────────────────────────────────
N_WORKERS    = 2

# Cada valor es el número de RONDAS de comunicación a comparar.
# Una ronda = workers entrenan 1 epoch sobre su shard y devuelven pesos.
MALLA_RONDAS  = [30, 60, 100]

REPETICIONES        = 3
TASA_APRENDIZAJE    = 0.001   # Adam en los workers
BATCH_SIZE          = 128     # tamaño de mini-batch en los workers

HOST        = '0.0.0.0'
PORT        = 5000
BUFFER_SIZE = 4096 * 1024
DIRECTORIO_RESULTADOS = 'resultados'

DEVICE = torch.device('cpu')   # el servidor sólo promedia y evalúa


# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
#  ARQUITECTURA CNN (debe ser idéntica en worker.py)
#
#  Cambio respecto al diagrama: sin Softmax final.
#  CrossEntropyLoss ya incluye log-softmax; añadirlo explícitamente
#  causa inestabilidad numérica. Es la práctica estándar en PyTorch.
# ──────────────────────────────────────────────────────────────
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # ── Bloque conv 1  →  16×16×32 ──────────────────
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # ── Bloque conv 2  →  8×8×64 ────────────────────
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            # ── Bloque conv 3  →  4×4×128 ───────────────────
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),           # sin Softmax — ver nota arriba
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────
#  FedAvg: promedio de state_dicts
# ──────────────────────────────────────────────────────────────
def promediar_state_dicts(lista_sd):
    avg = copy.deepcopy(lista_sd[0])
    for key in avg:
        avg[key] = torch.stack([sd[key].float() for sd in lista_sd]).mean(dim=0)
    return avg


# ──────────────────────────────────────────────────────────────
#  EVALUACIÓN (servidor, sin gradientes)
# ──────────────────────────────────────────────────────────────
def calcular_precision(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / total


# ──────────────────────────────────────────────────────────────
#  COMUNICACIÓN
# ──────────────────────────────────────────────────────────────
def enviar_objeto(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(8, 'big'))
    sock.sendall(data)


def recibir_objeto(sock):
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    size = int.from_bytes(header, 'big')
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


# ──────────────────────────────────────────────────────────────
#  CSV
#
#  detalle_Nworkers.csv  — progresión ronda a ronda
#  resumen_Nworkers.csv  — una fila por repetición completada
# ──────────────────────────────────────────────────────────────
CABECERA_DETALLE = [
    'n_workers', 'n_rondas_config', 'repeticion', 'ronda',
    'perdida', 'precision_test', 'tiempo_ronda_seg',
]
CABECERA_RESUMEN = [
    'n_workers', 'n_rondas_config', 'repeticion',
    'precision_test_final', 'perdida_final',
    'tiempo_total_seg', 'tiempo_promedio_ronda_seg',
]


def inicializar_csv(ruta, cabecera):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', newline='') as f:
        csv.writer(f).writerow(cabecera)
    log(f"CSV creado: {ruta}")


def guardar_fila_detalle(ruta, n_rondas_config, rep, ronda, perdida, prec, t):
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, n_rondas_config, rep, ronda,
            round(perdida, 6), round(prec, 6), round(t, 4),
        ])


def guardar_fila_resumen(ruta, n_rondas_config, rep, historial):
    tiempo_total    = sum(e['t'] for e in historial)
    tiempo_promedio = tiempo_total / len(historial)
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, n_rondas_config, rep,
            round(historial[-1]['prec'],    6),
            round(historial[-1]['perdida'], 6),
            round(tiempo_total,             2),
            round(tiempo_promedio,          4),
        ])


# ──────────────────────────────────────────────────────────────
#  BUCLE DE ENTRENAMIENTO (una sola repetición)
# ──────────────────────────────────────────────────────────────
def ejecutar_entrenamiento(conexiones, n_rondas, rep,
                           model, test_loader, ruta_detalle):
    historial = []

    for ronda in range(n_rondas):
        t0 = time.perf_counter()

        # Extraer pesos actuales (CPU, serializable con pickle)
        sd_actual = {k: v.cpu() for k, v in model.state_dict().items()}

        # Enviar a todos los workers
        for c in conexiones:
            enviar_objeto(c['sock'], {
                'tipo':        'TRAIN',
                'state_dict':  sd_actual,
                'lr':          TASA_APRENDIZAJE,
                'ronda':       ronda,
            })

        # Recoger resultados
        resultados = []
        for c in conexiones:
            r = recibir_objeto(c['sock'])
            if r and r.get('tipo') == 'RESULT':
                resultados.append(r)

        t_ronda = time.perf_counter() - t0

        if resultados:
            avg_sd  = promediar_state_dicts([r['state_dict'] for r in resultados])
            model.load_state_dict(avg_sd)
            perdida = float(np.mean([r['perdida'] for r in resultados]))
            prec    = calcular_precision(model, test_loader)
        else:
            perdida, prec = -1.0, -1.0

        historial.append({'perdida': perdida, 'prec': prec, 't': t_ronda})
        guardar_fila_detalle(ruta_detalle, n_rondas, rep, ronda, perdida, prec, t_ronda)

        if ronda % 10 == 0 or ronda == n_rondas - 1:
            log(f"  [rondas={n_rondas:>4} rep={rep}] "
                f"{ronda+1:>4}/{n_rondas} | "
                f"loss={perdida:.4f} | "
                f"test={prec*100:.1f}% | "
                f"t={t_ronda:.2f}s")

    return historial


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

    # ── Cargar CIFAR-10 ──────────────────────────────────────
    log("Cargando CIFAR-10...")

    # Normalización estándar CIFAR-10
    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2470, 0.2435, 0.2616)

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_ds     = datasets.CIFAR10('data', train=False, download=True, transform=tf_test)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0,
    )

    # Dataset de entrenamiento en crudo (uint8): los workers normalizan + augmentan
    train_ds_raw = datasets.CIFAR10('data', train=True, download=True)
    X_train = train_ds_raw.data            # (50000, 32, 32, 3) uint8 numpy
    y_train = np.array(train_ds_raw.targets)

    # Mezcla aleatoria reproducible
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    log(f"Dataset: {len(X_train)} train · {len(test_ds)} test")

    # ── Esperar workers ──────────────────────────────────────
    log(f"Esperando {N_WORKERS} workers en {HOST}:{PORT}...")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(N_WORKERS)

    conexiones = []
    for i in range(N_WORKERS):
        conn, addr = srv.accept()
        hello = recibir_objeto(conn)
        name  = hello.get('name', str(addr)) if hello else str(addr)
        conexiones.append({'sock': conn, 'name': name})
        log(f"  Worker {i+1}/{N_WORKERS}: {name} ({addr})")
    srv.close()

    # ── Distribuir datos UNA sola vez ────────────────────────
    tam = len(X_train) // N_WORKERS
    for i, c in enumerate(conexiones):
        inicio = i * tam
        fin    = inicio + tam if i < N_WORKERS - 1 else len(X_train)
        enviar_objeto(c['sock'], {
            'tipo':       'INIT',
            'X':          X_train[inicio:fin],   # (shard, 32, 32, 3) uint8
            'y':          y_train[inicio:fin],
            'worker_idx': i,
            'batch_size': BATCH_SIZE,
            'mean':       MEAN,
            'std':        STD,
        })
        log(f"  Shard {i}: muestras {inicio}–{fin-1} → {c['name']}")
    log("Datos distribuidos. Iniciando malla.\n")

    # ── Crear CSVs ───────────────────────────────────────────
    ruta_detalle = os.path.join(DIRECTORIO_RESULTADOS, f'detalle_{N_WORKERS}workers.csv')
    ruta_resumen = os.path.join(DIRECTORIO_RESULTADOS, f'resumen_{N_WORKERS}workers.csv')
    inicializar_csv(ruta_detalle, CABECERA_DETALLE)
    inicializar_csv(ruta_resumen, CABECERA_RESUMEN)

    # ── Malla de experimentación ─────────────────────────────
    total_configs = len(MALLA_RONDAS)
    for cfg_idx, n_rondas in enumerate(MALLA_RONDAS, start=1):
        log("=" * 60)
        log(f"CONFIGURACIÓN {cfg_idx}/{total_configs}: {n_rondas} rondas  "
            f"({REPETICIONES} repeticiones)")
        log("=" * 60)

        for rep in range(1, REPETICIONES + 1):
            log(f"  ── Repetición {rep}/{REPETICIONES} ──")
            t_rep = time.perf_counter()

            # Reinicializar modelo en cada repetición
            model = CIFAR10CNN().to(DEVICE)

            historial = ejecutar_entrenamiento(
                conexiones, n_rondas, rep,
                model, test_loader, ruta_detalle,
            )
            guardar_fila_resumen(ruta_resumen, n_rondas, rep, historial)

            log(f"  Rep {rep} lista en {time.perf_counter()-t_rep:.1f}s | "
                f"acc final={historial[-1]['prec']*100:.2f}%\n")

    # ── Cerrar workers ───────────────────────────────────────
    for c in conexiones:
        try:
            enviar_objeto(c['sock'], {'tipo': 'EXPERIMENT_END'})
            c['sock'].close()
        except Exception:
            pass

    log("=" * 60)
    log("Malla completada.")
    log(f"  Detalle → {ruta_detalle}")
    log(f"  Resumen → {ruta_resumen}")
    log("=" * 60)
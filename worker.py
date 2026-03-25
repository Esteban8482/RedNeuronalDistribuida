# ============================================================
#  worker.py  –  Worker con CNN PyTorch para CIFAR-10
#
#  Uso:
#    python worker.py --host 127.0.0.1
#    python worker.py --host 127.0.0.1 --core 2   # anclar al núcleo 2
# ============================================================

import socket
import pickle
import time
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
#  ARGUMENTOS
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1',
                    help='IP del servidor')
parser.add_argument('--core', type=int, default=-1,
                    help='Núcleo físico al que anclar este proceso (-1 = sin afinidad)')
args = parser.parse_args()

HOST        = args.host
PORT        = 5000
BUFFER_SIZE = 4096 * 1024


# ──────────────────────────────────────────────────────────────
#  AFINIDAD DE CPU  (Linux · requiere permisos normales de usuario)
# ──────────────────────────────────────────────────────────────
if args.core >= 0:
    try:
        os.sched_setaffinity(0, {args.core})
        print(f"[AFFINITY] Proceso anclado al núcleo físico {args.core}", flush=True)
    except AttributeError:
        print("[AFFINITY] os.sched_setaffinity no está disponible en este SO "
              "(sólo Linux). El worker correrá sin afinidad fija.", flush=True)
    except OSError as e:
        print(f"[AFFINITY] No se pudo anclar al núcleo {args.core}: {e}", flush=True)


# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────
#  ARQUITECTURA CNN (idéntica a server.py — no modificar)
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
            nn.Linear(128, 10),   # sin Softmax — CrossEntropyLoss lo incluye
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────
#  DATASET CON DATA AUGMENTATION
#  Recibe numpy uint8 (N, 32, 32, 3), normaliza y aplica augmentation.
# ──────────────────────────────────────────────────────────────
class CIFARShard(Dataset):
    """Shard local del worker con augmentation en tiempo de entrenamiento."""

    def __init__(self, X_np: np.ndarray, y_np: np.ndarray,
                 mean: tuple, std: tuple):
        # HWC uint8 → NCHW float [0,1]
        self.X = (
            torch.from_numpy(X_np)           # (N, H, W, C)
            .permute(0, 3, 1, 2)             # (N, C, H, W)
            .float()
            .div(255.0)
        )
        self.y = torch.from_numpy(y_np).long()

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]


# ──────────────────────────────────────────────────────────────
#  ENTRENAMIENTO LOCAL (una ronda = un epoch completo sobre el shard)
# ──────────────────────────────────────────────────────────────
def entrenar_una_ronda(model: nn.Module, loader: DataLoader,
                       lr: float, device: torch.device) -> float:
    """
    Ejecuta un epoch completo con Adam + CrossEntropyLoss.
    Devuelve la pérdida media sobre todos los mini-batches.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


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
#  MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device   = torch.device('cpu')
    hostname = socket.gethostname()
    log(f"Worker iniciando: {hostname}")

    # ── Conectar al servidor con reintentos ──────────────────
    sock      = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    for attempt in range(30):
        try:
            sock.connect((HOST, PORT))
            connected = True
            log(f"Conectado al servidor {HOST}:{PORT}")
            break
        except Exception as e:
            log(f"Intento {attempt+1}/30 fallido: {e}")
            time.sleep(2)

    if not connected:
        log("No se pudo conectar. Saliendo.")
        raise SystemExit(1)

    # Enviar identificación
    enviar_objeto(sock, {'tipo': 'HELLO', 'name': hostname})

    # ── Estado local ─────────────────────────────────────────
    model:  nn.Module  = CIFAR10CNN().to(device)
    loader: DataLoader | None = None

    # ── Bucle principal ──────────────────────────────────────
    while True:
        msg = recibir_objeto(sock)

        if msg is None:
            log("Conexión cerrada por el servidor.")
            break

        tipo = msg.get('tipo', 'UNKNOWN')

        # ── INIT: recibir shard de datos ─────────────────────
        if tipo == 'INIT':
            X_np       = msg['X']           # (N, 32, 32, 3) uint8
            y_np       = msg['y']           # (N,) int
            batch_size = msg.get('batch_size', 128)
            mean       = msg.get('mean', (0.4914, 0.4822, 0.4465))
            std        = msg.get('std',  (0.2470, 0.2435, 0.2616))

            ds = CIFARShard(X_np, y_np, mean, std)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,       # un hilo por worker (ya está anclado a 1 núcleo)
                pin_memory=False,
            )
            log(f"Shard recibido: {len(ds)} muestras | "
                f"batch={batch_size} | "
                f"{len(loader)} pasos/epoch")

        # ── TRAIN: una ronda de FedAvg ───────────────────────
        elif tipo == 'TRAIN':
            if loader is None:
                log("ERROR: 'TRAIN' recibido antes de 'INIT'. Ignorando.")
                continue

            # Cargar pesos del servidor
            model.load_state_dict(msg['state_dict'])

            t0     = time.perf_counter()
            perdida = entrenar_una_ronda(model, loader, msg['lr'], device)
            t      = round(time.perf_counter() - t0, 4)

            log(f"Ronda {msg.get('ronda', '?'):>4} | "
                f"loss={perdida:.4f} | t={t}s")

            # Devolver pesos actualizados al servidor
            enviar_objeto(sock, {
                'tipo':       'RESULT',
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'perdida':    perdida,
                'tiempo':     t,
            })

        # ── EXPERIMENT_END ───────────────────────────────────
        elif tipo == 'EXPERIMENT_END':
            log("Experimento finalizado por el servidor.")
            break

        else:
            log(f"Tipo de mensaje desconocido: '{tipo}' — ignorando.")

    sock.close()
    log("Worker finalizado.")
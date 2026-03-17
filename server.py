import socket
import pickle
import numpy as np
import csv
import os
import time
from datetime import datetime
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# CONFIG  ← los únicos valores que hay que tocar
# --------------------------------------------------
N_WORKERS        = 2
MALLA_NEURONAS   = [100, 300, 500]   # configuraciones de la capa oculta
REPETICIONES     = 5                 # repeticiones por configuración
EPOCAS           = 100
TASA_APRENDIZAJE = 0.1
HOST             = '0.0.0.0'
PORT             = 5000
BUFFER_SIZE      = 4096 * 1024
DIRECTORIO_RESULTADOS = 'resultados'

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --------------------------------------------------
# RED NEURONAL
# --------------------------------------------------
def crear_params(n_neuronas):
    """Inicializa pesos para una red 784 → n_neuronas → 10."""
    return {
        'W1': np.random.randn(784, n_neuronas) * 0.05,
        'b1': np.full(n_neuronas, 0.01, dtype=float),
        'W2': np.random.randn(n_neuronas, 10) * 0.05,
        'b2': np.full(10, 0.01, dtype=float),
    }

def promediar_pesos(lista_params):
    n = len(lista_params)
    return {k: sum(p[k] for p in lista_params) / n for k in lista_params[0]}

def calcular_precision(X, y, params):
    z1 = np.dot(X, params['W1']) + params['b1']
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, params['W2']) + params['b2']
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return float(np.mean(np.argmax(a2, axis=1) == y))

def dividir_dataset(X, y, n_partes):
    tam = len(X) // n_partes
    return [(X[i*tam : (i+1)*tam if i < n_partes-1 else len(X)],
             y[i*tam : (i+1)*tam if i < n_partes-1 else len(y)])
            for i in range(n_partes)]

def one_hot(y, num_clases=10):
    oh = np.zeros((len(y), num_clases))
    oh[np.arange(len(y)), y] = 1
    return oh

# --------------------------------------------------
# COMUNICACIÓN
# --------------------------------------------------
def enviar_objeto(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(8, byteorder='big'))
    sock.sendall(data)

def recibir_objeto(sock):
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    size = int.from_bytes(header, byteorder='big')
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)

# --------------------------------------------------
# CSV
#
# Esquema diseñado para comparar entre experimentos:
#   n_workers y n_neuronas son columnas clave en ambos archivos,
#   lo que permite hacer JOIN/filter al combinar con otros experimentos.
#
# detalle_Nworkers.csv  → una fila por (n_neuronas × repeticion × epoca)
# resumen_Nworkers.csv  → una fila por (n_neuronas × epoca),
#                          con media y std sobre las REPETICIONES
# --------------------------------------------------
CABECERA_DETALLE = [
    'n_workers', 'n_neuronas', 'repeticion', 'epoca',
    'perdida', 'precision_test',
    'tiempo_epoca_seg', 't_computo_max', 't_computo_min', 't_computo_avg',
]

CABECERA_RESUMEN = [
    'n_workers', 'n_neuronas', 'epoca',
    'perdida_media', 'perdida_std',
    'precision_test_media', 'precision_test_std',
    'tiempo_epoca_media', 'tiempo_epoca_std',
]

def inicializar_csv(ruta, cabecera):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', newline='') as f:
        csv.writer(f).writerow(cabecera)
    log(f"CSV creado: {ruta}")

def guardar_fila_detalle(ruta, n_neuronas, rep, epoca,
                         perdida, precision, t_epoca, tiempos):
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, n_neuronas, rep, epoca,
            round(perdida,   6),
            round(precision, 6),
            round(t_epoca,   4),
            round(max(tiempos), 4) if tiempos else -1,
            round(min(tiempos), 4) if tiempos else -1,
            round(sum(tiempos) / len(tiempos), 4) if tiempos else -1,
        ])

def guardar_resumen_configuracion(ruta, n_neuronas, historia):
    """
    historia: lista de REPETICIONES listas, cada una con EPOCAS dicts
              {perdida, precision, t_epoca}.
    Escribe una fila por época con media y std entre repeticiones.
    """
    with open(ruta, 'a', newline='') as f:
        writer = csv.writer(f)
        for epoca in range(EPOCAS):
            perdidas    = [historia[r][epoca]['perdida']   for r in range(REPETICIONES)]
            precisiones = [historia[r][epoca]['precision'] for r in range(REPETICIONES)]
            tiempos     = [historia[r][epoca]['t_epoca']   for r in range(REPETICIONES)]
            writer.writerow([
                N_WORKERS,
                n_neuronas,
                epoca,
                round(float(np.mean(perdidas)),         6),
                round(float(np.std(perdidas,  ddof=1)), 6),
                round(float(np.mean(precisiones)),      6),
                round(float(np.std(precisiones, ddof=1)), 6),
                round(float(np.mean(tiempos)),          4),
                round(float(np.std(tiempos,   ddof=1)), 4),
            ])

# --------------------------------------------------
# BUCLE DE ENTRENAMIENTO (una sola repetición)
# --------------------------------------------------
def ejecutar_entrenamiento(conexiones, n_neuronas, rep,
                           X_test, y_test, ruta_detalle):
    """
    Entrena EPOCAS épocas con los workers ya inicializados.
    Devuelve lista de EPOCAS dicts {perdida, precision, t_epoca}.
    """
    params = crear_params(n_neuronas)
    historial = []

    for epoca in range(EPOCAS):
        t0 = time.perf_counter()

        for c in conexiones:
            enviar_objeto(c['sock'], {
                'tipo': 'TRAIN',
                'params': params,
                'tasa_aprendizaje': TASA_APRENDIZAJE,
                'epoca': epoca,
            })

        resultados = []
        for c in conexiones:
            resp = recibir_objeto(c['sock'])
            if resp and resp.get('tipo') == 'RESULT':
                resultados.append(resp)

        t_epoca = time.perf_counter() - t0

        if resultados:
            params    = promediar_pesos([r['params'] for r in resultados])
            perdida   = float(np.mean([r['perdida']  for r in resultados]))
            tiempos   = [r['tiempo_computo']          for r in resultados]
            precision = calcular_precision(X_test, y_test, params)
        else:
            perdida, precision, tiempos = -1.0, -1.0, []

        historial.append({'perdida': perdida, 'precision': precision, 't_epoca': t_epoca})
        guardar_fila_detalle(ruta_detalle, n_neuronas, rep,
                             epoca, perdida, precision, t_epoca, tiempos)

        if epoca % 20 == 0 or epoca == EPOCAS - 1:
            log(f"  [n={n_neuronas:>3} rep={rep}] "
                f"época {epoca:>3}/{EPOCAS} | "
                f"pérdida={perdida:.4f} | "
                f"test={precision*100:.1f}% | "
                f"t={t_epoca:.2f}s")

    return historial

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

    # ── Cargar dataset ────────────────────────────────────────────
    log("Cargando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32').to_numpy() / 255.0
    y = mnist.target.astype('int32').to_numpy()
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    y_train_oh = one_hot(y_train)
    log(f"Dataset: {len(X_train)} train · {len(X_test)} test")

    # ── Esperar workers ───────────────────────────────────────────
    log(f"Esperando {N_WORKERS} workers en {HOST}:{PORT}...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(N_WORKERS)

    conexiones = []
    for i in range(N_WORKERS):
        conn, addr = server_sock.accept()
        hello = recibir_objeto(conn)
        name = hello.get('name', str(addr)) if hello else str(addr)
        conexiones.append({'sock': conn, 'name': name})
        log(f"  Worker {i+1}/{N_WORKERS}: {name} ({addr})")
    server_sock.close()

    # ── Distribuir datos UNA sola vez ─────────────────────────────
    # Los datos no cambian entre configuraciones ni repeticiones;
    # solo cambia la forma de los parámetros (n_neuronas).
    partes = dividir_dataset(X_train, y_train_oh, N_WORKERS)
    for i, c in enumerate(conexiones):
        enviar_objeto(c['sock'], {
            'tipo': 'INIT',
            'X': partes[i][0],
            'y': partes[i][1],
            'worker_idx': i,
        })
    log("Datos distribuidos. Iniciando malla de experimentación.\n")

    # ── Crear CSVs ────────────────────────────────────────────────
    ruta_detalle = os.path.join(DIRECTORIO_RESULTADOS, f'detalle_{N_WORKERS}workers.csv')
    ruta_resumen = os.path.join(DIRECTORIO_RESULTADOS, f'resumen_{N_WORKERS}workers.csv')
    inicializar_csv(ruta_detalle, CABECERA_DETALLE)
    inicializar_csv(ruta_resumen, CABECERA_RESUMEN)

    # ── Malla de experimentación ──────────────────────────────────
    total_configs = len(MALLA_NEURONAS)
    for cfg_idx, n_neuronas in enumerate(MALLA_NEURONAS, start=1):
        log("=" * 60)
        log(f"CONFIGURACIÓN {cfg_idx}/{total_configs}: {n_neuronas} neuronas ocultas  "
            f"({REPETICIONES} repeticiones × {EPOCAS} épocas)")
        log("=" * 60)

        historia_config = []   # acumula REPETICIONES historiales

        for rep in range(1, REPETICIONES + 1):
            log(f"  ── Repetición {rep}/{REPETICIONES} ──")
            t_rep = time.perf_counter()

            historial = ejecutar_entrenamiento(
                conexiones, n_neuronas, rep,
                X_test, y_test, ruta_detalle,
            )
            historia_config.append(historial)

            log(f"  Repetición {rep} finalizada en "
                f"{time.perf_counter()-t_rep:.1f}s | "
                f"pérdida final={historial[-1]['perdida']:.4f} | "
                f"test final={historial[-1]['precision']*100:.2f}%\n")

        # ── Agregar y guardar resumen de la configuración ─────────
        guardar_resumen_configuracion(ruta_resumen, n_neuronas, historia_config)

        precisiones_finales = [historia_config[r][-1]['precision'] for r in range(REPETICIONES)]
        log(f"  RESUMEN {n_neuronas} neuronas | "
            f"precisión test → "
            f"media={np.mean(precisiones_finales)*100:.2f}% · "
            f"std={np.std(precisiones_finales, ddof=1)*100:.2f}%\n")

    # ── Cerrar workers ────────────────────────────────────────────
    for c in conexiones:
        try:
            enviar_objeto(c['sock'], {'tipo': 'EXPERIMENT_END'})
            c['sock'].close()
        except Exception:
            pass

    log("=" * 60)
    log("Malla completada.")
    log(f"  Detalle  → {ruta_detalle}")
    log(f"  Resumen  → {ruta_resumen}")
    log("=" * 60)
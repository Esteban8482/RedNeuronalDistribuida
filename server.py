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
N_NEURONAS       = 256               # neuronas de la capa oculta, fijo
MALLA_EPOCAS     = [100, 250, 500]  # configuraciones a comparar
REPETICIONES     = 5                 # repeticiones por configuración
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
def crear_params():
    return {
        'W1': np.random.randn(784, N_NEURONAS) * 0.05,
        'b1': np.full(N_NEURONAS, 0.01, dtype=float),
        'W2': np.random.randn(N_NEURONAS, 10) * 0.05,
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
# detalle_Nworkers.csv
#   Progresión época a época de cada corrida.
#   Útil para graficar curvas de aprendizaje.
#   Clave: (n_workers, n_neuronas, n_epocas_config, repeticion, epoca)
#
# resumen_Nworkers.csv
#   Una fila por repetición completada con sus métricas finales.
#   Diseñado para comparar entre experimentos con distinto n_workers:
#   al concatenar resumen_2workers.csv + resumen_4workers.csv + ...
#   se puede filtrar/agrupar por n_workers y comparar directamente
#   precisión final y tiempo de entrenamiento.
#   Clave: (n_workers, n_neuronas, n_epocas_config, repeticion)
# --------------------------------------------------
CABECERA_DETALLE = [
    'n_workers', 'n_neuronas', 'n_epocas_config', 'repeticion', 'epoca',
    'perdida', 'precision_test', 'tiempo_epoca_seg',
]

CABECERA_RESUMEN = [
    'n_workers', 'n_neuronas', 'n_epocas_config', 'repeticion',
    'precision_test_final', 'perdida_final',
    'tiempo_total_seg', 'tiempo_promedio_epoca_seg',
]

def inicializar_csv(ruta, cabecera):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', newline='') as f:
        csv.writer(f).writerow(cabecera)
    log(f"CSV creado: {ruta}")

def guardar_fila_detalle(ruta, n_epocas_config, rep, epoca,
                         perdida, precision, t_epoca):
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, N_NEURONAS, n_epocas_config, rep, epoca,
            round(perdida,   6),
            round(precision, 6),
            round(t_epoca,   4),
        ])

def guardar_fila_resumen(ruta, n_epocas_config, rep, historial):
    """Una fila por repetición completada con sus métricas finales."""
    tiempo_total   = sum(e['t_epoca']   for e in historial)
    tiempo_promedio = tiempo_total / len(historial)
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS,
            N_NEURONAS,
            n_epocas_config,
            rep,
            round(historial[-1]['precision'], 6),
            round(historial[-1]['perdida'],   6),
            round(tiempo_total,              2),
            round(tiempo_promedio,           4),
        ])

# --------------------------------------------------
# BUCLE DE ENTRENAMIENTO (una sola repetición)
# --------------------------------------------------
def ejecutar_entrenamiento(conexiones, n_epocas_config, rep,
                           X_test, y_test, ruta_detalle):
    params = crear_params()
    historial = []

    for epoca in range(n_epocas_config):
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
            precision = calcular_precision(X_test, y_test, params)
        else:
            perdida, precision = -1.0, -1.0

        historial.append({'perdida': perdida, 'precision': precision, 't_epoca': t_epoca})
        guardar_fila_detalle(ruta_detalle, n_epocas_config, rep,
                             epoca, perdida, precision, t_epoca)

        if epoca % 50 == 0 or epoca == n_epocas_config - 1:
            log(f"  [épocas={n_epocas_config:>4} rep={rep}] "
                f"época {epoca+1:>4}/{n_epocas_config} | "
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
    total_configs = len(MALLA_EPOCAS)
    for cfg_idx, n_epocas_config in enumerate(MALLA_EPOCAS, start=1):
        log("=" * 60)
        log(f"CONFIGURACIÓN {cfg_idx}/{total_configs}: {n_epocas_config} épocas  "
            f"({REPETICIONES} repeticiones · {N_NEURONAS} neuronas ocultas)")
        log("=" * 60)

        for rep in range(1, REPETICIONES + 1):
            log(f"  ── Repetición {rep}/{REPETICIONES} ──")
            t_rep = time.perf_counter()

            historial = ejecutar_entrenamiento(
                conexiones, n_epocas_config, rep,
                X_test, y_test, ruta_detalle,
            )

            # Guardar fila de resumen inmediatamente al terminar la repetición
            guardar_fila_resumen(ruta_resumen, n_epocas_config, rep, historial)

            log(f"  Repetición {rep} finalizada en "
                f"{time.perf_counter()-t_rep:.1f}s | "
                f"pérdida final={historial[-1]['perdida']:.4f} | "
                f"test final={historial[-1]['precision']*100:.2f}%\n")

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
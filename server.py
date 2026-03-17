import socket
import pickle
import numpy as np
import csv
import os
import time
from datetime import datetime
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
N_WORKERS   = 3          # número de workers a esperar antes de empezar
EPOCAS      = 100
TASA_APRENDIZAJE = 0.1
HOST        = '0.0.0.0'
PORT        = 5000
BUFFER_SIZE = 4096 * 1024
DIRECTORIO_RESULTADOS = 'resultados'

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --------------------------------------------------
# FUNCIONES RED NEURONAL
# --------------------------------------------------
def inicializar_pesos(n_entrada, n_salida):
    return np.random.randn(n_entrada, n_salida) * 0.05

def inicializar_sesgos(n_salida):
    return np.full(n_salida, 0.01, dtype=float)

def dividir_dataset(X, y, n_partes):
    n_total = len(X)
    tamano = n_total // n_partes
    partes = []
    for i in range(n_partes):
        inicio = i * tamano
        fin = inicio + tamano if i < n_partes - 1 else n_total
        partes.append((X[inicio:fin], y[inicio:fin]))
    return partes

def promediar_pesos(lista_params):
    n = len(lista_params)
    return {k: sum(p[k] for p in lista_params) / n for k in lista_params[0]}

def calcular_precision(X, y, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return np.mean(np.argmax(a2, axis=1) == y)

def one_hot(y, num_clases=10):
    y_oh = np.zeros((len(y), num_clases))
    y_oh[np.arange(len(y)), y] = 1
    return y_oh

# --------------------------------------------------
# COMUNICACIÓN
# --------------------------------------------------
def enviar_objeto(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(8, byteorder='big'))
    sock.sendall(data)

def recibir_objeto(sock):
    # Leer header de 8 bytes
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    size = int.from_bytes(header, byteorder='big')

    # Leer datos
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)

# --------------------------------------------------
# CSV
# --------------------------------------------------
def inicializar_csv(ruta):
    os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else '.', exist_ok=True)
    with open(ruta, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoca', 'perdida_promedio', 'precision_test',
            'tiempo_epoca_seg', 'tiempo_computo_max',
            'tiempo_computo_min', 'tiempo_computo_avg',
        ])

def registrar_epoca(ruta, epoca, perdida, precision, t_epoca, tiempos):
    with open(ruta, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoca,
            round(perdida, 6),
            round(precision, 6),
            round(t_epoca, 4),
            round(max(tiempos), 4) if tiempos else -1,
            round(min(tiempos), 4) if tiempos else -1,
            round(sum(tiempos) / len(tiempos), 4) if tiempos else -1,
        ])

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

    # Cargar dataset
    log("Cargando MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32').to_numpy() / 255.0
    y = mnist.target.astype('int32').to_numpy()

    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    y_train_oh = one_hot(y_train)
    log(f"Dataset listo: {len(X_train)} train, {len(X_test)} test")

    # Esperar conexiones
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
        log(f"  Worker {i+1}/{N_WORKERS} conectado: {name} ({addr})")

    server_sock.close()
    log("Todos los workers conectados.")

    # Distribuir datos
    partes = dividir_dataset(X_train, y_train_oh, N_WORKERS)
    for i, c in enumerate(conexiones):
        log(f"Enviando datos al worker {i+1} ({c['name']})...")
        enviar_objeto(c['sock'], {
            'tipo': 'INIT',
            'X': partes[i][0],
            'y': partes[i][1],
            'worker_idx': i,
        })
    log("Datos distribuidos.")

    # Inicializar parámetros y CSV
    params = {
        'W1': inicializar_pesos(784, 256),
        'b1': inicializar_sesgos(256),
        'W2': inicializar_pesos(256, 10),
        'b2': inicializar_sesgos(10),
    }

    ruta_csv = os.path.join(DIRECTORIO_RESULTADOS, f'entrenamiento_{N_WORKERS}workers.csv')
    inicializar_csv(ruta_csv)
    log(f"CSV de resultados: {ruta_csv}")

    # Bucle de entrenamiento
    log(f"Iniciando entrenamiento: {EPOCAS} épocas con {N_WORKERS} workers")
    for epoca in range(EPOCAS):
        t0 = time.perf_counter()

        # Enviar TRAIN a todos los workers
        for c in conexiones:
            enviar_objeto(c['sock'], {
                'tipo': 'TRAIN',
                'params': params,
                'tasa_aprendizaje': TASA_APRENDIZAJE,
                'epoca': epoca,
            })

        # Recoger resultados
        resultados = []
        for c in conexiones:
            resp = recibir_objeto(c['sock'])
            if resp and resp.get('tipo') == 'RESULT':
                resultados.append(resp)

        t_epoca = time.perf_counter() - t0

        if resultados:
            params = promediar_pesos([r['params'] for r in resultados])
            perdida = sum(r['perdida'] for r in resultados) / len(resultados)
            tiempos = [r['tiempo_computo'] for r in resultados]
            precision = calcular_precision(X_test, y_test, params)
        else:
            perdida, precision, tiempos = -1.0, -1.0, []

        registrar_epoca(ruta_csv, epoca, perdida, precision, t_epoca, tiempos)

        if epoca % 10 == 0 or epoca == EPOCAS - 1:
            log(f"Época {epoca:>3}/{EPOCAS} | Pérdida: {perdida:.4f} | "
                f"Test: {precision*100:.1f}% | T: {t_epoca:.2f}s")

    # Notificar fin y cerrar
    for c in conexiones:
        try:
            enviar_objeto(c['sock'], {'tipo': 'EXPERIMENT_END'})
            c['sock'].close()
        except Exception:
            pass

    log(f"Entrenamiento completado. Resultados guardados en: {ruta_csv}")
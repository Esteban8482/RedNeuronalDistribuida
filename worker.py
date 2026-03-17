import socket
import pickle
import numpy as np
import time
import argparse
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='172.178.121.181', help='IP del servidor')
args = parser.parse_args()

HOST        = args.host
PORT        = 5000
BUFFER_SIZE = 4096 * 1024

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# --------------------------------------------------
# RED NEURONAL
# --------------------------------------------------
def calcular_gradientes(X, y, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2

    # Softmax estable
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    y_pred = np.clip(a2, 1e-26, 1 - 1e-26)
    perdida = -np.sum(y * np.log(y_pred)) / len(y)

    m = X.shape[0]
    dz2 = (a2 - y) / m
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'perdida': perdida}

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
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    hostname = socket.gethostname()
    log(f"Worker iniciando: {hostname}")

    # Conectar al servidor con reintentos
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

    # Enviar HELLO
    enviar_objeto(sock, {'tipo': 'HELLO', 'name': hostname})

    # Variables de estado
    X, y = None, None

    # Bucle principal
    while True:
        msg = recibir_objeto(sock)

        if msg is None:
            log("Conexión cerrada por el servidor.")
            break

        tipo = msg.get('tipo', 'UNKNOWN')

        if tipo == 'INIT':
            X = msg['X']
            y = msg['y']
            log(f"Datos recibidos: {len(X)} muestras, forma X={X.shape}, forma y={y.shape}")

        elif tipo == 'TRAIN':
            epoca = msg.get('epoca', '?')
            t0 = time.perf_counter()

            grad = calcular_gradientes(X, y, msg['params'])

            p  = msg['params']
            lr = msg['tasa_aprendizaje']
            new_params = {
                'W1': p['W1'] - lr * grad['dW1'],
                'b1': p['b1'] - lr * grad['db1'],
                'W2': p['W2'] - lr * grad['dW2'],
                'b2': p['b2'] - lr * grad['db2'],
            }

            tiempo_computo = round(time.perf_counter() - t0, 4)
            log(f"Época {epoca} | loss={grad['perdida']:.4f} | t={tiempo_computo}s")

            enviar_objeto(sock, {
                'tipo': 'RESULT',
                'params': new_params,
                'perdida': grad['perdida'],
                'tiempo_computo': tiempo_computo,
            })

        elif tipo == 'EXPERIMENT_END':
            log("Entrenamiento finalizado por el servidor.")
            break

        else:
            log(f"Mensaje desconocido recibido: {tipo}")

    sock.close()
    log("Worker finalizado.")
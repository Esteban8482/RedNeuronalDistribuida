import socket, pickle, numpy as np, time, argparse, socket as s_mod
import os

parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default=None, help='region del worker (ej. westus2, eastus)')
args = parser.parse_args()

HOST = '172.178.121.181'
PORT = 5000
BUFFER_SIZE = 4096 * 1024
N_EPOCAS = 100

def calcular_gradientes(X, y, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    z_estable = z2 - np.max(z2, axis=1, keepdims=True)
    exp_z = np.exp(z_estable)
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
    return {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2,'perdida':perdida}

def enviar_objeto(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(8, 'big'))
    sock.sendall(data)

def recibir_objeto(sock):
    size_bytes = sock.recv(8)
    if not size_bytes:
        return None
    size = int.from_bytes(size_bytes, 'big')
    data = b''
    while len(data) < size:
        data += sock.recv(min(BUFFER_SIZE, size - len(data)))
    return pickle.loads(data)

if __name__ == '__main__':
    region = args.region or (os.environ.get('REGION') if 'os' in globals() else None) or 'unknown'
    hostname = s_mod.gethostname()
    print(f'Connecting to {HOST}:{PORT} from host={hostname} region={region}...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for i in range(30):
        try:
            sock.connect((HOST, PORT))
            print('Connected!')
            break
        except:
            time.sleep(2)

    # enviar HELLO con metadata
    try:
        enviar_objeto(sock, {'tipo': 'HELLO', 'region': region, 'name': hostname})
    except Exception as e:
        print('Error enviando HELLO:', e)

    X, y = None, None
    while True:
        msg = recibir_objeto(sock)
        if msg is None:
            break
        if msg.get('tipo') == 'INIT':
            X, y = msg['X'], msg['y']
            print(f'Received INIT with {len(X)} samples')
        elif msg.get('tipo') == 'TRAIN':
            t0 = time.perf_counter()
            grad = calcular_gradientes(X, y, msg['params'])
            p = msg['params']
            lr = msg['tasa_aprendizaje']
            new_params = {
                'W1': p['W1'] - lr * grad['dW1'],
                'b1': p['b1'] - lr * grad['db1'],
                'W2': p['W2'] - lr * grad['dW2'],
                'b2': p['b2'] - lr * grad['db2']
            }
            enviar_objeto(sock, {
                'tipo': 'RESULT', 'params': new_params,
                'perdida': grad['perdida'],
                'tiempo_computo': round(time.perf_counter()-t0, 4)
            })
        elif msg.get('tipo') == 'EXPERIMENT_END':
            print('Experiment completed (received EXPERIMENT_END)')
            break
    sock.close()
    print('Worker finished')

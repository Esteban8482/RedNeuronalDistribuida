# Modified worker.py - Single Process Version
import socket, pickle, numpy as np, time

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
    size = int.from_bytes(sock.recv(8), 'big')
    data = b''
    while len(data) < size:
        data += sock.recv(min(BUFFER_SIZE, size - len(data)))
    return pickle.loads(data)

if __name__ == '__main__':
    print(f'Connecting to {HOST}:{PORT}...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for i in range(30):
        try:
            sock.connect((HOST, PORT))
            print('Connected!')
            break
        except:
            time.sleep(2)
    X, y = None, None
    while True:
        msg = recibir_objeto(sock)
        if msg is None: break
        if msg['tipo'] == 'INIT':
            X, y = msg['X'], msg['y']
            print(f'Received {len(X)} samples')
        elif msg['tipo'] == 'TRAIN':
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
        elif msg['tipo'] == 'EXPERIMENT_END':
            print('Experiment completed')
            break
    sock.close()
    print('Worker finished')

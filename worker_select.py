# -*- coding: utf-8 -*-
"""
Worker - Proceso de trabajo que recibe tareas del server
Entrena en su porcion de datos y devuelve los resultados
Compatible con la version select() del server
"""

import socket
import pickle
import numpy as np
import time

# ============ CONFIGURACION ============
HOST = '127.0.0.1'
PORT = 5000
BUFFER_SIZE = 4096 * 1024  # 4MB

# ============ FUNCIONES DE RED NEURONAL ============
def relu(z):
    return np.maximum(0, z)

def derivada_relu(z):
    return (z > 0).astype(float)

def softmax(z):
    z_estable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_estable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

def calcular_perdida(y_pred, y_real):
    y_pred = np.clip(y_pred, 1e-26, 1 - 1e-26)
    return -np.sum(y_real * np.log(y_pred)) / len(y_real)

def backward_propagation(X, y, cache, W2):
    m = X.shape[0]
    a1 = cache['a1']
    a2 = cache['a2']
    z1 = cache['z1']
    
    dz2 = (a2 - y) / m
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * derivada_relu(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def actualizar_parametros(params, gradientes, tasa_aprendizaje):
    W1 = params['W1'] - tasa_aprendizaje * gradientes['dW1']
    b1 = params['b1'] - tasa_aprendizaje * gradientes['db1']
    W2 = params['W2'] - tasa_aprendizaje * gradientes['dW2']
    b2 = params['b2'] - tasa_aprendizaje * gradientes['db2']
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def entrenar_batch(X, y, params, tasa_aprendizaje):
    """Entrena un batch y devuelve los nuevos parametros y perdida"""
    y_pred, cache = forward_propagation(X, params['W1'], params['b1'], params['W2'], params['b2'])
    perdida = calcular_perdida(y_pred, y)
    gradientes = backward_propagation(X, y, cache, params['W2'])
    params_actualizados = actualizar_parametros(params, gradientes, tasa_aprendizaje)
    return params_actualizados, perdida

# ============ COMUNICACION SOCKET ============
def enviar_objeto(sock, obj):
    """Serializa y envia un objeto usando pickle"""
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(size.to_bytes(8, byteorder='big'))
    sock.sendall(data)

def recibir_objeto(sock):
    """Recibe y deserializa un objeto usando pickle"""
    size_bytes = sock.recv(8)
    if not size_bytes:
        return None
    size = int.from_bytes(size_bytes, byteorder='big')
    
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            return None
        data += chunk
    
    return pickle.loads(data)

# ============ WORKER PRINCIPAL ============
class TrainingWorker:
    def __init__(self):
        self.sock = None
        self.X = None
        self.y = None
        
    def conectar(self):
        """Conecta al server con sistema de reintentos"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((HOST, PORT))
                print(f"[Worker] Conectado a {HOST}:{PORT}")
                break
            except ConnectionRefusedError:
                print("[Worker] Servidor no listo. Reintentando en 3 segundos...")
                time.sleep(3)
        
    def ejecutar(self):
        """Bucle principal del worker - procesa mensajes secuencialmente"""
        try:
            while True:
                # Esperar mensaje del servidor (bloqueante)
                mensaje = recibir_objeto(self.sock)
                
                if mensaje is None:
                    print("[Worker] Conexion cerrada por el server")
                    break
                
                tipo = mensaje.get('tipo')
                
                if tipo == 'DONE':
                    print("[Worker] Recibido DONE. Finalizando...")
                    break
                
                elif tipo == 'INIT':
                    # Recibir dataset una sola vez
                    self.X = mensaje['X']
                    self.y = mensaje['y']
                    print(f"[Worker] Datos de entrenamiento recibidos: {self.X.shape[0]} muestras")

                elif tipo == 'TRAIN':
                    # Entrenar usando los datos cacheados
                    params = mensaje['params']
                    tasa = mensaje['tasa_aprendizaje']
                    epoca = mensaje['epoca']
                    
                    # Entrenar una epoca
                    nuevos_params, perdida = entrenar_batch(self.X, self.y, params, tasa)
                    
                    # Enviar resultado inmediatamente
                    respuesta = {
                        'tipo': 'RESULT',
                        'params': nuevos_params,
                        'perdida': perdida,
                        'epoca': epoca
                    }
                    enviar_objeto(self.sock, respuesta)
                    
        except ConnectionResetError:
            print("[Worker] Conexion reseteada por el server")
        except Exception as e:
            print(f"[Worker] Error: {e}")
        finally:
            self.sock.close()
            print("[Worker] Desconectado")

# ============ EJECUCION ============
if __name__ == '__main__':
    worker = TrainingWorker()
    worker.conectar()
    worker.ejecutar()

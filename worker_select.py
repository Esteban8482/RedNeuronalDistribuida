# -*- coding: utf-8 -*-
import socket
import pickle
import numpy as np
import time
import multiprocessing
import argparse

# ============ CONFIGURACION ============
HOST = '192.168.61.213'
PORT = 5000
BUFFER_SIZE = 4096 * 1024  # 4MB para el tráfico de arrays de pesos
N_PROCESOS_LOCALES = 2     # Valor por defecto (sobreescrito por argumento CLI)

# ============ FUNCIONES GLOBALES (Requeridas para Multiprocessing en Windows) ============

def calcular_gradientes_local(X_sub, y_sub, params):
    """
    Función que ejecutan los procesos hijos.
    Realiza forward y backward pass sobre una fracción del dataset.
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    # Forward Propagation
    z1 = np.dot(X_sub, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, W2) + b2

    # Softmax estable
    z_estable = z2 - np.max(z2, axis=1, keepdims=True)
    exp_z = np.exp(z_estable)
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Cálculo de Pérdida (Cross-Entropy)
    y_pred = np.clip(a2, 1e-26, 1 - 1e-26)
    perdida = -np.sum(y_sub * np.log(y_pred)) / len(y_sub)

    # Backward Propagation
    m = X_sub.shape[0]
    dz2 = (a2 - y_sub) / m
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0).astype(float)  # Derivada ReLU
    dW1 = np.dot(X_sub.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'perdida': perdida}

# ============ UTILIDADES DE RED ============

def enviar_objeto(sock, obj):
    """Serializa y envía un objeto usando pickle con encabezado de tamaño"""
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(size.to_bytes(8, byteorder='big'))
    sock.sendall(data)

def recibir_objeto(sock):
    """Recibe un objeto completo basándose en el tamaño indicado en el encabezado"""
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

# ============ CLASE PRINCIPAL DEL WORKER ============

class TrainingWorker:
    def __init__(self, n_procesos=N_PROCESOS_LOCALES):
        self.sock = None
        self.X = None
        self.y = None
        self.n_procesos = n_procesos

    def conectar(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((HOST, PORT))
                print(f"[Worker] Conectado exitosamente a {HOST}:{PORT}")
                print(f"[Worker] Procesos locales configurados: {self.n_procesos}")
                break
            except ConnectionRefusedError:
                print("[Worker] Esperando al servidor... (Reintento en 3s)")
                time.sleep(3)

    def ejecutar(self):
        try:
            with multiprocessing.Pool(processes=self.n_procesos) as pool:
                while True:
                    mensaje = recibir_objeto(self.sock)

                    if mensaje is None or mensaje.get('tipo') == 'DONE':
                        print("[Worker] Entrenamiento finalizado o servidor desconectado.")
                        break

                    if mensaje['tipo'] == 'INIT':
                        self.X = mensaje['X']
                        self.y = mensaje['y']
                        print(f"[Worker] Dataset recibido. Localmente usando {self.n_procesos} procesos.")

                    elif mensaje['tipo'] == 'TRAIN':
                        params = mensaje['params']
                        tasa = mensaje['tasa_aprendizaje']

                        # ── Medición de tiempo por época ──────────────────────
                        t_inicio = time.perf_counter()

                        # Subdivisión local del trabajo
                        X_split = np.array_split(self.X, self.n_procesos)
                        y_split = np.array_split(self.y, self.n_procesos)

                        # Paralelización local
                        tareas = [(X_s, y_s, params) for X_s, y_s in zip(X_split, y_split)]
                        resultados_hijos = pool.starmap(calcular_gradientes_local, tareas)

                        # Agregación local (Promedio de gradientes y pérdida)
                        grad_promedio = self._promediar_gradientes(resultados_hijos)
                        perdida_promedio = sum(r['perdida'] for r in resultados_hijos) / self.n_procesos

                        # Actualización local del modelo
                        nuevos_params = self._aplicar_gradientes(params, grad_promedio, tasa)

                        t_fin = time.perf_counter()
                        tiempo_epoca = round(t_fin - t_inicio, 4)
                        # ─────────────────────────────────────────────────────

                        # Enviar resultado al servidor central
                        enviar_objeto(self.sock, {
                            'tipo': 'RESULT',
                            'params': nuevos_params,
                            'perdida': perdida_promedio,
                            'epoca': mensaje['epoca'],
                            'tiempo_computo': tiempo_epoca,   # <-- nuevo campo de métricas
                            'n_procesos': self.n_procesos     # <-- trazabilidad del experimento
                        })

        except Exception as e:
            print(f"[Worker] Error durante la ejecución: {e}")
        finally:
            if self.sock:
                self.sock.close()

    def _promediar_gradientes(self, resultados):
        promedios = {}
        for llave in ['dW1', 'db1', 'dW2', 'db2']:
            promedios[llave] = sum(r[llave] for r in resultados) / len(resultados)
        return promedios

    def _aplicar_gradientes(self, p, g, lr):
        return {
            'W1': p['W1'] - lr * g['dW1'],
            'b1': p['b1'] - lr * g['db1'],
            'W2': p['W2'] - lr * g['dW2'],
            'b2': p['b2'] - lr * g['db2']
        }

# ============ PUNTO DE ENTRADA ============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker de entrenamiento distribuido',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--procesos', '-p',
        type=int,
        default=N_PROCESOS_LOCALES,
        choices=[2, 4, 6, 8, 10],
        help='Número de procesos locales del worker (experimento: 2|4|6|8|10)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=HOST,
        help='IP del servidor central'
    )
    args = parser.parse_args()

    # Sobreescribir HOST si se pasó por argumento
    HOST = args.host

    worker = TrainingWorker(n_procesos=args.procesos)
    worker.conectar()
    worker.ejecutar()

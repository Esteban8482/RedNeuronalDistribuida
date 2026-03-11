# -*- coding: utf-8 -*-
import socket
import pickle
import numpy as np
import time
import multiprocessing # <--- Nueva importación

# ============ CONFIGURACION ============
HOST = '127.0.0.1' # <--- RECUERDA: Cambiar por la IP del Server en la PC Worker
PORT = 5000
BUFFER_SIZE = 4096 * 1024
N_PROCESOS = 4  # <--- Cantidad de sub-procesos locales por worker

# ============ FUNCIONES DE APOYO PARA PARALELISMO ============

def calcular_gradientes_local(X_sub, y_sub, params):
    """
    Función que ejecutará cada proceso hijo.
    Calcula gradientes y pérdida sobre una fracción del dataset local.
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    
    # Forward
    z1 = np.dot(X_sub, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    
    # Softmax estable
    z_estable = z2 - np.max(z2, axis=1, keepdims=True)
    exp_z = np.exp(z_estable)
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Pérdida
    y_pred = np.clip(a2, 1e-26, 1 - 1e-26)
    perdida = -np.sum(y_sub * np.log(y_pred)) / len(y_sub)
    
    # Backward
    m = X_sub.shape[0]
    dz2 = (a2 - y_sub) / m
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = np.dot(X_sub.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'perdida': perdida}

# ============ WORKER PRINCIPAL ============
class TrainingWorker:
    def __init__(self, n_procesos=N_PROCESOS):
        self.sock = None
        self.X = None
        self.y = None
        self.n_procesos = n_procesos
        
    def conectar(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((HOST, PORT))
                print(f"[Worker] Conectado al server en {HOST}:{PORT}")
                break
            except ConnectionRefusedError:
                print("[Worker] Servidor no listo. Reintentando...")
                time.sleep(3)
        
    def ejecutar(self):
        try:
            # Crear el Pool de procesos una sola vez
            with multiprocessing.Pool(processes=self.n_procesos) as pool:
                while True:
                    mensaje = recibir_objeto(self.sock)
                    if mensaje is None or mensaje.get('tipo') == 'DONE':
                        break
                    
                    if mensaje['tipo'] == 'INIT':
                        self.X = mensaje['X']
                        self.y = mensaje['y']
                        print(f"[Worker] {self.n_procesos} procesos listos para {len(self.X)} muestras")

                    elif mensaje['tipo'] == 'TRAIN':
                        params = mensaje['params']
                        tasa = mensaje['tasa_aprendizaje']
                        
                        # 1. Dividir datos locales para los sub-procesos
                        X_split = np.array_split(self.X, self.n_procesos)
                        y_split = np.array_split(self.y, self.n_procesos)
                        
                        # 2. Mapear la tarea a los procesos hijos
                        tareas = [(X_s, y_s, params) for X_s, y_s in zip(X_split, y_split)]
                        resultados_locales = pool.starmap(calcular_gradientes_local, tareas)
                        
                        # 3. Reducir (promediar) gradientes y pérdida
                        grad_promedio = self._promediar_resultados(resultados_locales)
                        perdida_total = sum(r['perdida'] for r in resultados_locales) / self.n_procesos
                        
                        # 4. Actualizar parámetros localmente antes de enviar
                        nuevos_params = self._actualizar(params, grad_promedio, tasa)
                        
                        enviar_objeto(self.sock, {
                            'tipo': 'RESULT',
                            'params': nuevos_params,
                            'perdida': perdida_total,
                            'epoca': mensaje['epoca']
                        })
                        
        finally:
            self.sock.close()

    def _promediar_resultados(self, resultados):
        promedios = {}
        for clave in ['dW1', 'db1', 'dW2', 'db2']:
            promedios[clave] = sum(r[clave] for r in resultados) / len(resultados)
        return promedios

    def _actualizar(self, p, g, lr):
        return {
            'W1': p['W1'] - lr * g['dW1'],
            'b1': p['b1'] - lr * g['db1'],
            'W2': p['W2'] - lr * g['dW2'],
            'b2': p['b2'] - lr * g['db2']
        }

# ... (Mantener funciones enviar_objeto y recibir_objeto del original)

if __name__ == '__main__':
    worker = TrainingWorker()
    worker.conectar()
    worker.ejecutar()
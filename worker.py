# -*- coding: utf-8 -*-
"""
Worker Experimental - Participa en malla de experimentación distribuida.
Se ejecuta una sola vez y participa en todos los experimentos secuencialmente.

USO:
    python worker.py

CONFIGURACIÓN:
    Modificar HOST abajo para apuntar a la IP del servidor.
    Los experimentos se ejecutan automáticamente: 2, 4, 6, 8, 10 procesos.
"""

import socket
import pickle
import numpy as np
import time
import multiprocessing

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURACIÓN DEL WORKER                           ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  HOST: Dirección IP del servidor central.                                 ║
# ║         Cambiar según la configuración de tu laboratorio.                 ║
# ╠──────────────────────────────────────────────────────────────────────────╣
# ║  EXPERIMENTOS_PROCESOS: Debe coincidir con la configuración del server.   ║
# ║         El worker ejecutará cada experimento en secuencia.                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

HOST = '192.168.61.213'             # ← CAMBIAR ESTO a la IP del servidor
PORT = 5000
BUFFER_SIZE = 4096 * 1024           # 4MB para arrays de pesos
EXPERIMENTOS_PROCESOS = [2, 4, 6, 8, 10]  # Debe coincidir con el server


# ═══════════════════════════════════════════════════════════════════════════
#                    FUNCIONES GLOBALES (Multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════

def calcular_gradientes_local(X_sub, y_sub, params):
    """
    Función que ejecutan los procesos hijos.
    Realiza forward y backward pass sobre una fracción del dataset.
    
    Args:
        X_sub: Subconjunto de datos de entrada
        y_sub: Subconjunto de etiquetas (one-hot)
        params: Diccionario con parámetros del modelo (W1, b1, W2, b2)
    
    Returns:
        Dict con gradientes (dW1, db1, dW2, db2) y pérdida
    """
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']

    # Forward Propagation
    z1 = np.dot(X_sub, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, W2) + b2

    # Softmax estable numéricamente
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


# ═══════════════════════════════════════════════════════════════════════════
#                         COMUNICACIÓN SOCKET
# ═══════════════════════════════════════════════════════════════════════════

def enviar_objeto(sock, obj):
    """Serializa y envía un objeto usando pickle con encabezado de tamaño."""
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(size.to_bytes(8, byteorder='big'))
    sock.sendall(data)


def recibir_objeto(sock):
    """Recibe un objeto completo basándose en el tamaño del encabezado."""
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


# ═══════════════════════════════════════════════════════════════════════════
#                         CLASE DEL WORKER
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentWorker:
    """
    Worker que participa en múltiples experimentos secuenciales.
    Se conecta al servidor, ejecuta un experimento, espera la señal
    de finalización, y se reconecta para el siguiente experimento.
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.n_procesos_actual = 0

    def _conectar(self, n_procesos):
        """
        Intenta conectarse al servidor con reintentos.
        
        Args:
            n_procesos: Número de procesos para este experimento
            
        Returns:
            Socket conectado o None si falla
        """
        print(f"\n[Worker] Conectando al servidor {HOST}:{PORT}...")
        print(f"[Worker] Configuración: {n_procesos} procesos locales")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        max_intentos = 30

        for intento in range(max_intentos):
            try:
                sock.connect((HOST, PORT))
                print(f"[Worker] Conexión establecida!")
                self.n_procesos_actual = n_procesos
                return sock
            except ConnectionRefusedError:
                if intento < max_intentos - 1:
                    print(f"[Worker] Esperando al servidor... (intento {intento+1}/{max_intentos})")
                    time.sleep(2)
                else:
                    print(f"[Worker] No se pudo conectar después de {max_intentos} intentos")
                    return None
            except Exception as e:
                print(f"[Worker] Error de conexión: {e}")
                return None

        return None

    def _ejecutar_experimento(self, sock):
        """
        Ejecuta un experimento completo con el pool de procesos actual.
        
        Args:
            sock: Socket conectado al servidor
            
        Returns:
            True si el experimento terminó normalmente, False si hubo error
        """
        try:
            with multiprocessing.Pool(processes=self.n_procesos_actual) as pool:
                epocas_completadas = 0

                while True:
                    mensaje = recibir_objeto(sock)

                    if mensaje is None:
                        print("[Worker] Servidor desconectado inesperadamente.")
                        return False

                    # Mensaje de inicialización
                    if mensaje['tipo'] == 'INIT':
                        self.X = mensaje['X']
                        self.y = mensaje['y']
                        print(f"[Worker] Dataset recibido: {len(self.X)} muestras")
                        print(f"[Worker] Iniciando entrenamiento con {self.n_procesos_actual} procesos...")

                    # Mensaje de entrenamiento
                    elif mensaje['tipo'] == 'TRAIN':
                        params = mensaje['params']
                        tasa = mensaje['tasa_aprendizaje']

                        # Medición de tiempo por época
                        t_inicio = time.perf_counter()

                        # Subdivisión local del trabajo
                        X_split = np.array_split(self.X, self.n_procesos_actual)
                        y_split = np.array_split(self.y, self.n_procesos_actual)

                        # Paralelización local
                        tareas = [(X_s, y_s, params) for X_s, y_s in zip(X_split, y_split)]
                        resultados_hijos = pool.starmap(calcular_gradientes_local, tareas)

                        # Agregación local (promedio de gradientes y pérdida)
                        grad_promedio = self._promediar_gradientes(resultados_hijos)
                        perdida_promedio = sum(r['perdida'] for r in resultados_hijos) / len(resultados_hijos)

                        # Actualización local del modelo
                        nuevos_params = self._aplicar_gradientes(params, grad_promedio, tasa)

                        t_fin = time.perf_counter()
                        tiempo_epoca = round(t_fin - t_inicio, 4)

                        # Enviar resultado al servidor
                        enviar_objeto(sock, {
                            'tipo': 'RESULT',
                            'params': nuevos_params,
                            'perdida': perdida_promedio,
                            'epoca': mensaje['epoca'],
                            'tiempo_computo': tiempo_epoca,
                        })

                        epocas_completadas += 1
                        if epocas_completadas % 10 == 0:
                            print(f"[Worker] Época {epocas_completadas} completada (pérdida: {perdida_promedio:.4f})")

                    # Señal de fin de experimento
                    elif mensaje['tipo'] == 'EXPERIMENT_END':
                        print(f"[Worker] Experimento completado. Épocas: {epocas_completadas}")
                        return True

                    # Señal de fin definitivo (ya no hay más experimentos)
                    elif mensaje['tipo'] == 'DONE':
                        print("[Worker] Todos los experimentos finalizados por el servidor.")
                        return True

        except Exception as e:
            print(f"[Worker] Error durante el experimento: {e}")
            return False

    def _promediar_gradientes(self, resultados):
        """Promedia los gradientes de todos los procesos hijos."""
        promedios = {}
        for llave in ['dW1', 'db1', 'dW2', 'db2']:
            promedios[llave] = sum(r[llave] for r in resultados) / len(resultados)
        return promedios

    def _aplicar_gradientes(self, p, g, lr):
        """Aplica los gradientes a los parámetros con la tasa de aprendizaje."""
        return {
            'W1': p['W1'] - lr * g['dW1'],
            'b1': p['b1'] - lr * g['db1'],
            'W2': p['W2'] - lr * g['dW2'],
            'b2': p['b2'] - lr * g['db2']
        }

    def ejecutar_todos_experimentos(self):
        """
        Ciclo principal que ejecuta todos los experimentos en secuencia.
        
        Para cada configuración en EXPERIMENTOS_PROCESOS:
        1. Se conecta al servidor
        2. Ejecuta el experimento completo
        3. Espera señal de fin
        4. Se desconecta y pasa al siguiente
        """
        print("=" * 60)
        print("WORKER DE EXPERIMENTACIÓN DISTRIBUIDA")
        print("=" * 60)
        print(f"Servidor: {HOST}:{PORT}")
        print(f"Experimentos programados: {EXPERIMENTOS_PROCESOS}")
        print("=" * 60)

        experimentos_completados = 0
        experimentos_fallidos = 0

        for idx, n_procesos in enumerate(EXPERIMENTOS_PROCESOS):
            print("\n" + "─" * 60)
            print(f"EXPERIMENTO {idx+1}/{len(EXPERIMENTOS_PROCESOS)}: {n_procesos} PROCESOS")
            print("─" * 60)

            # Conectar al servidor
            sock = self._conectar(n_procesos)
            if sock is None:
                print(f"[Worker] ERROR: No se pudo conectar para experimento {n_procesos} proc")
                experimentos_fallidos += 1
                continue

            # Ejecutar experimento
            exito = self._ejecutar_experimento(sock)

            # Cerrar conexión
            try:
                sock.close()
            except Exception:
                pass

            if exito:
                experimentos_completados += 1
                print(f"[Worker] Experimento {n_procesos} proc: EXITOSO")
            else:
                experimentos_fallidos += 1
                print(f"[Worker] Experimento {n_procesos} proc: FALLIDO")

            # Pequeña pausa antes del siguiente experimento
            time.sleep(1)

        # Resumen final
        print("\n" + "=" * 60)
        print("RESUMEN DEL WORKER")
        print("=" * 60)
        print(f"Experimentos completados: {experimentos_completados}")
        print(f"Experimentos fallidos   : {experimentos_fallidos}")
        print("=" * 60)

        return experimentos_completados, experimentos_fallidos


# ═══════════════════════════════════════════════════════════════════════════
#                         PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Configuración de multiprocessing para Windows/Linux
    multiprocessing.set_start_method('spawn', force=True)

    worker = ExperimentWorker()
    worker.ejecutar_todos_experimentos()
    print("\n[Worker] Finalizando worker...")

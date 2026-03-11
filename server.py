# -*- coding: utf-8 -*-
"""
Server Experimental - Coordina entrenamiento distribuido con malla de experimentación.
Ejecuta automáticamente experimentos con 2, 4, 6, 8 y 10 procesos por worker.

USO:
    python server.py

CONFIGURACIÓN:
    Modificar N_WORKERS abajo para cambiar la cantidad de workers esperados.
"""

import socket
import pickle
import select
import numpy as np
import csv
import os
import time
from sklearn.datasets import fetch_openml

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURACIÓN DEL EXPERIMENTO                      ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  N_WORKERS: Número de workers que participarán en el experimento.         ║
# ║            Este valor determina cuántas máquinas worker deben conectarse. ║
# ║            IMPORTANTE: Debe coincidir con la cantidad de workers que      ║
# ║            ejecutes en tu laboratorio.                                    ║
# ╠──────────────────────────────────────────────────────────────────────────╣
# ║  EXPERIMENTOS_PROCESOS: Lista de configuraciones de procesos a probar.    ║
# ║            Cada valor representa el número de procesos locales que        ║
# ║            cada worker utilizará para el cómputo paralelo.                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

N_WORKERS = 2                      # ← CAMBIAR ESTO según tu laboratorio
EXPERIMENTOS_PROCESOS = [2, 4, 6, 8, 10]  # Configuraciones a probar
EPOCAS = 100
TASA_APRENDIZAJE = 0.1
HOST = '0.0.0.0'
PORT = 5000
BUFFER_SIZE = 4096 * 1024          # 4MB para arrays grandes
SELECT_TIMEOUT = None              # None = esperar indefinidamente
DIRECTORIO_RESULTADOS = 'resultados'


# ═══════════════════════════════════════════════════════════════════════════
#                         FUNCIONES DE RED NEURONAL
# ═══════════════════════════════════════════════════════════════════════════

def inicializar_pesos(n_entrada, n_salida):
    """Inicializa pesos con distribución normal escalada."""
    return np.random.randn(n_entrada, n_salida) * 0.05


def inicializar_sesgos(n_salida):
    """Inicializa sesgos con valor pequeño positivo."""
    return np.full(n_salida, 0.01, dtype=float)


def dividir_dataset(X, y, n_partes):
    """Divide el dataset en n_partes aproximadamente iguales."""
    n_total = len(X)
    tamano_parte = n_total // n_partes
    partes = []
    for i in range(n_partes):
        inicio = i * tamano_parte
        fin = inicio + tamano_parte if i < n_partes - 1 else n_total
        partes.append((X[inicio:fin], y[inicio:fin]))
    return partes


def promediar_pesos(lista_params):
    """Promedia los parámetros de múltiples workers (Federated Averaging)."""
    n = len(lista_params)
    params_promedio = {}
    for key in lista_params[0].keys():
        suma = sum(params[key] for params in lista_params)
        params_promedio[key] = suma / n
    return params_promedio


def calcular_precision(X, y, params):
    """Calcula la precisión del modelo sobre un conjunto de datos."""
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = np.dot(a1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    predicciones = np.argmax(a2, axis=1)
    return np.sum(predicciones == y) / len(y)


def one_hot(y, num_clases=10):
    """Convierte etiquetas a codificación one-hot."""
    n = len(y)
    y_one_hot = np.zeros((n, num_clases))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot


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
    """Recibe y deserializa un objeto usando pickle."""
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
#                         REGISTRO DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

def inicializar_csv(ruta_csv, n_workers, n_procesos):
    """
    Crea el archivo CSV de métricas para un experimento específico.
    
    Args:
        ruta_csv: Ruta del archivo CSV
        n_workers: Número de workers en el experimento
        n_procesos: Número de procesos por worker en este experimento
    """
    os.makedirs(os.path.dirname(ruta_csv) if os.path.dirname(ruta_csv) else '.', exist_ok=True)
    with open(ruta_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n_workers',
            'n_procesos_worker',
            'epoca',
            'perdida_promedio',
            'precision_test',
            'tiempo_epoca_seg',
            'tiempo_computo_worker_max',
            'tiempo_computo_worker_min',
            'tiempo_computo_worker_avg',
        ])
    print(f"[Server] Archivo de métricas creado: {ruta_csv}")


def registrar_epoca_csv(ruta_csv, n_workers, n_procesos, epoca, perdida, precision,
                        tiempo_epoca, tiempos_worker):
    """Registra una época de entrenamiento en el CSV."""
    with open(ruta_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            n_workers,
            n_procesos,
            epoca,
            round(perdida, 6),
            round(precision, 6),
            round(tiempo_epoca, 4),
            round(max(tiempos_worker), 4) if tiempos_worker else -1,
            round(min(tiempos_worker), 4) if tiempos_worker else -1,
            round(sum(tiempos_worker) / len(tiempos_worker), 4) if tiempos_worker else -1,
        ])


# ═══════════════════════════════════════════════════════════════════════════
#                         CLASE DEL SERVIDOR
# ═══════════════════════════════════════════════════════════════════════════

class ExperimentServer:
    """
    Servidor de experimentación que ejecuta múltiples configuraciones
    de procesos en secuencia automática.
    """

    def __init__(self):
        print("=" * 60)
        print("INICIANDO SERVIDOR DE EXPERIMENTACIÓN")
        print("=" * 60)
        print(f"Configuración del experimento:")
        print(f"  - Workers esperados     : {N_WORKERS}")
        print(f"  - Configuraciones       : {EXPERIMENTOS_PROCESOS} procesos")
        print(f"  - Épocas por experimento: {EPOCAS}")
        print(f"  - Tasa de aprendizaje   : {TASA_APRENDIZAJE}")
        print(f"  - Directorio resultados : {DIRECTORIO_RESULTADOS}/")
        print("=" * 60)

        # Cargar y preparar dataset MNIST (una sola vez)
        print("\n[Server] Cargando dataset MNIST...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32').to_numpy() / 255.0
        y = mnist.target.astype('int32').to_numpy()

        print("[Server] Preprocesando datos...")
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        self.X_train, self.X_test = X[:60000], X[60000:]
        self.y_train, self.y_test = y[:60000], y[60000:]
        self.y_train_one_hot = one_hot(self.y_train)

        print(f"[Server] Dataset listo: {len(self.X_train)} train, {len(self.X_test)} test")

        # Crear directorio de resultados
        os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

    def _crear_parametros_iniciales(self):
        """Crea nuevos parámetros iniciales para cada experimento."""
        return {
            'W1': inicializar_pesos(784, 256),
            'b1': inicializar_sesgos(256),
            'W2': inicializar_pesos(256, 10),
            'b2': inicializar_sesgos(10)
        }

    def _esperar_conexiones(self, server_socket):
        """Espera a que todos los workers se conecten."""
        print(f"\n[Server] Esperando {N_WORKERS} workers...")
        conexiones = []
        for i in range(N_WORKERS):
            conn, addr = server_socket.accept()
            print(f"[Server] Worker {i+1}/{N_WORKERS} conectado desde {addr}")
            conexiones.append(conn)
        return conexiones

    def _enviar_inicializacion(self, conexiones, partes):
        """Envía el dataset inicial a todos los workers."""
        print("[Server] Enviando particiones de dataset a workers...")
        for i, conn in enumerate(conexiones):
            mensaje_init = {
                'tipo': 'INIT',
                'X': partes[i][0],
                'y': partes[i][1]
            }
            enviar_objeto(conn, mensaje_init)
        print("[Server] Dataset distribuido exitosamente")

    def _ejecutar_entrenamiento(self, conexiones, params, ruta_csv, n_procesos):
        """
        Ejecuta el ciclo completo de entrenamiento para un experimento.
        
        Returns:
            historial_perdida: Lista con la pérdida por época
            tiempo_total: Tiempo total del entrenamiento
        """
        historial_perdida = []
        inicializar_csv(ruta_csv, N_WORKERS, n_procesos)

        tiempo_inicio_total = time.perf_counter()

        for epoca in range(EPOCAS):
            t_epoca_inicio = time.perf_counter()

            # PASO 1: Enviar parámetros a todos los workers
            for conn in conexiones:
                mensaje = {
                    'tipo': 'TRAIN',
                    'params': params,
                    'tasa_aprendizaje': TASA_APRENDIZAJE,
                    'epoca': epoca
                }
                enviar_objeto(conn, mensaje)

            # PASO 2: Recibir resultados usando select()
            resultados = []
            sockets_pendientes = list(conexiones)

            while sockets_pendientes:
                legibles, _, excepciones = select.select(
                    sockets_pendientes, [], sockets_pendientes, SELECT_TIMEOUT
                )

                for sock in excepciones:
                    if sock in sockets_pendientes:
                        sockets_pendientes.remove(sock)

                for sock in legibles:
                    try:
                        respuesta = recibir_objeto(sock)
                        if respuesta and respuesta['tipo'] == 'RESULT':
                            resultados.append({
                                'worker_id': conexiones.index(sock),
                                'params': respuesta['params'],
                                'perdida': respuesta['perdida'],
                                'tiempo_computo': respuesta.get('tiempo_computo', -1),
                            })
                            sockets_pendientes.remove(sock)
                        elif respuesta is None:
                            print(f"[Server] Worker desconectado inesperadamente")
                            sockets_pendientes.remove(sock)
                    except Exception as e:
                        print(f"[Server] Error recibiendo de worker: {e}")
                        sockets_pendientes.remove(sock)

            t_epoca_fin = time.perf_counter()
            tiempo_epoca = t_epoca_fin - t_epoca_inicio

            # PASO 3: Promediar pesos y registrar métricas
            if len(resultados) == N_WORKERS:
                params_parciales = [r['params'] for r in resultados]
                params = promediar_pesos(params_parciales)
                perdida_promedio = sum(r['perdida'] for r in resultados) / N_WORKERS
                tiempos_worker = [r['tiempo_computo'] for r in resultados]
                historial_perdida.append(perdida_promedio)

                # Calcular precisión cada 10 épocas
                if epoca % 10 == 0:
                    precision_test = calcular_precision(self.X_test, self.y_test, params)
                    print(f"  Época {epoca:>3}/{EPOCAS} | "
                          f"Pérdida: {perdida_promedio:.4f} | "
                          f"Test: {precision_test*100:.1f}% | "
                          f"T: {tiempo_epoca:.2f}s")
                else:
                    precision_test = -1.0

                # Registrar en CSV
                registrar_epoca_csv(
                    ruta_csv, N_WORKERS, n_procesos, epoca,
                    perdida_promedio, precision_test,
                    tiempo_epoca, tiempos_worker
                )
            else:
                print(f"[Server] ADVERTENCIA: Solo {len(resultados)}/{N_WORKERS} resultados")

        tiempo_total = time.perf_counter() - tiempo_inicio_total
        return historial_perdida, tiempo_total, params

    def _finalizar_experimento(self, conexiones):
        """Notifica a los workers que el experimento terminó y cierra conexiones."""
        print("[Server] Finalizando experimento, notificando workers...")
        for conn in conexiones:
            try:
                enviar_objeto(conn, {'tipo': 'EXPERIMENT_END'})
                conn.close()
            except Exception:
                pass

    def ejecutar_experimentos(self):
        """
        Ejecuta la malla completa de experimentos.
        
        Para cada configuración en EXPERIMENTOS_PROCESOS:
        1. Inicializa nuevos parámetros
        2. Espera conexiones de workers
        3. Ejecuta entrenamiento completo
        4. Guarda métricas
        5. Notifica workers para siguiente experimento
        """
        print("\n" + "=" * 60)
        print("INICIANDO MALLA DE EXPERIMENTACIÓN")
        print("=" * 60)

        resumen_experimentos = []

        for idx, n_procesos in enumerate(EXPERIMENTOS_PROCESOS):
            print("\n" + "─" * 60)
            print(f"EXPERIMENTO {idx+1}/{len(EXPERIMENTOS_PROCESOS)}: {n_procesos} PROCESOS POR WORKER")
            print("─" * 60)

            # Crear socket servidor para este experimento
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(N_WORKERS)

            # Preparar para este experimento
            params = self._crear_parametros_iniciales()
            partes = dividir_dataset(self.X_train, self.y_train_one_hot, N_WORKERS)
            ruta_csv = os.path.join(DIRECTORIO_RESULTADOS, f'experimento_{n_procesos}proc.csv')

            # Esperar conexiones
            conexiones = self._esperar_conexiones(server_socket)

            # Enviar inicialización
            self._enviar_inicializacion(conexiones, partes)

            # Ejecutar entrenamiento
            print(f"\n[Server] Iniciando entrenamiento con {n_procesos} procesos...")
            historial, tiempo_total, params_finales = self._ejecutar_entrenamiento(
                conexiones, params, ruta_csv, n_procesos
            )

            # Calcular métricas finales
            precision_train = calcular_precision(self.X_train, self.y_train, params_finales)
            precision_test = calcular_precision(self.X_test, self.y_test, params_finales)

            # Guardar resumen
            resumen_experimentos.append({
                'n_procesos': n_procesos,
                'perdida_final': historial[-1] if historial else -1,
                'precision_train': precision_train,
                'precision_test': precision_test,
                'tiempo_total': tiempo_total,
                'archivo_csv': ruta_csv
            })

            print(f"\n[Server] Experimento {n_procesos} proc completado:")
            print(f"  - Tiempo total     : {tiempo_total:.2f}s")
            print(f"  - Pérdida final    : {historial[-1]:.4f}")
            print(f"  - Precisión train  : {precision_train*100:.2f}%")
            print(f"  - Precisión test   : {precision_test*100:.2f}%")
            print(f"  - Resultados en    : {ruta_csv}")

            # Notificar workers y cerrar conexiones
            self._finalizar_experimento(conexiones)
            server_socket.close()

            # Pequeña pausa entre experimentos
            time.sleep(1)

        # Mostrar resumen final
        self._mostrar_resumen_final(resumen_experimentos)

    def _mostrar_resumen_final(self, resumen):
        """Muestra un resumen tabular de todos los experimentos."""
        print("\n" + "=" * 60)
        print("RESUMEN DE TODOS LOS EXPERIMENTOS")
        print("=" * 60)
        print(f"{'Proc':>6} | {'Tiempo':>10} | {'Pérdida':>10} | {'Train%':>8} | {'Test%':>8}")
        print("-" * 60)
        for exp in resumen:
            print(f"{exp['n_procesos']:>6} | {exp['tiempo_total']:>10.2f}s | "
                  f"{exp['perdida_final']:>10.4f} | {exp['precision_train']*100:>7.2f}% | "
                  f"{exp['precision_test']*100:>7.2f}%")
        print("=" * 60)
        print(f"\nTodos los archivos CSV guardados en: {DIRECTORIO_RESULTADOS}/")


# ═══════════════════════════════════════════════════════════════════════════
#                         PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    server = ExperimentServer()
    server.ejecutar_experimentos()
    print("\n[Server] Todos los experimentos completados. Finalizando...")

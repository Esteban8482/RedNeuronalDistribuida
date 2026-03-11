# -*- coding: utf-8 -*-
"""
Server - Coordina el entrenamiento distribuido.
Recibe conexiones de workers, distribuye trabajo y promedia resultados.

Modificado para experimento: acepta N_WORKERS por CLI y registra métricas
de tiempo, pérdida y precisión en un archivo CSV.
"""

import socket
import pickle
import select
import numpy as np
import argparse
import csv
import os
import time
from sklearn.datasets import fetch_openml

# ============ CONFIGURACION ============
HOST = '0.0.0.0'
PORT = 5000
N_WORKERS = 4               # Valor por defecto (sobreescrito por argumento CLI)
EPOCAS = 100
TASA_APRENDIZAJE = 0.1
BUFFER_SIZE = 4096 * 1024  # 4MB para manejar arrays grandes
SELECT_TIMEOUT = None       # None = esperar indefinidamente

# ============ FUNCIONES DE RED NEURONAL ============

def inicializar_pesos(n_entrada, n_salida):
    return np.random.randn(n_entrada, n_salida) * 0.05

def inicializar_sesgos(n_salida):
    return np.full(n_salida, 0.01, dtype=float)

def dividir_dataset(X, y, n_partes):
    n_total = len(X)
    tamano_parte = n_total // n_partes
    partes = []
    for i in range(n_partes):
        inicio = i * tamano_parte
        fin = inicio + tamano_parte if i < n_partes - 1 else n_total
        partes.append((X[inicio:fin], y[inicio:fin]))
    return partes

def promediar_pesos(lista_params):
    n = len(lista_params)
    params_promedio = {}
    for key in lista_params[0].keys():
        suma = sum(params[key] for params in lista_params)
        params_promedio[key] = suma / n
    return params_promedio

def calcular_precision(X, y, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    predicciones = np.argmax(a2, axis=1)
    return np.sum(predicciones == y) / len(y)

def one_hot(y, num_clases=10):
    n = len(y)
    y_one_hot = np.zeros((n, num_clases))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot

# ============ COMUNICACION SOCKET ============

def enviar_objeto(sock, obj):
    """Serializa y envía un objeto usando pickle"""
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

def enviar_objeto_nonblocking(sock, obj):
    """
    Envía un objeto de forma no bloqueante.
    Retorna True si el envío fue completo, False si el socket no está listo.
    """
    try:
        _, writable, _ = select.select([], [sock], [], 0)
        if sock in writable:
            enviar_objeto(sock, obj)
            return True
        return False
    except Exception as e:
        print(f"[Server] Error enviando: {e}")
        return False

# ============ REGISTRO DE MÉTRICAS (nuevo) ============

def inicializar_csv(ruta_csv, n_workers):
    """Crea (o sobreescribe) el archivo CSV de métricas con cabecera."""
    os.makedirs(os.path.dirname(ruta_csv) if os.path.dirname(ruta_csv) else '.', exist_ok=True)
    with open(ruta_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'n_workers',
            'n_procesos_worker',   # reportado por cada worker
            'epoca',
            'perdida_promedio',
            'precision_test',
            'tiempo_epoca_seg',    # tiempo total de la época (server-side)
            'tiempo_computo_worker_max',  # máximo tiempo de cómputo reportado por workers
            'tiempo_computo_worker_min',
            'tiempo_computo_worker_avg',
        ])
    print(f"[Server] Archivo de métricas: {ruta_csv}")

def registrar_epoca_csv(ruta_csv, n_workers, epoca, perdida, precision,
                        tiempo_epoca, tiempos_worker, n_procesos_workers):
    """Añade una fila de métricas al CSV."""
    # n_procesos_worker: tomar el valor del primer worker (todos deben ser iguales en el experimento)
    n_proc = n_procesos_workers[0] if n_procesos_workers else -1
    with open(ruta_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            n_workers,
            n_proc,
            epoca,
            round(perdida, 6),
            round(precision, 6),
            round(tiempo_epoca, 4),
            round(max(tiempos_worker), 4) if tiempos_worker else -1,
            round(min(tiempos_worker), 4) if tiempos_worker else -1,
            round(sum(tiempos_worker) / len(tiempos_worker), 4) if tiempos_worker else -1,
        ])

# ============ SERVER PRINCIPAL ============

class TrainingServer:
    def __init__(self, n_workers, ruta_csv):
        self.n_workers = n_workers
        self.ruta_csv = ruta_csv

        print("Cargando dataset MNIST...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32').to_numpy() / 255.0
        y = mnist.target.astype('int32').to_numpy()

        print("Preprocesando datos...")
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        self.X_train, self.X_test = X[:60000], X[60000:]
        self.y_train, self.y_test = y[:60000], y[60000:]
        self.y_train_one_hot = one_hot(self.y_train)

        print("Inicializando pesos y sesgos...")
        self.params = {
            'W1': inicializar_pesos(784, 256),
            'b1': inicializar_sesgos(256),
            'W2': inicializar_pesos(256, 10),
            'b2': inicializar_sesgos(10)
        }

        print(f"Dividiendo dataset entre {self.n_workers} workers...")
        self.partes = dividir_dataset(self.X_train, self.y_train_one_hot, self.n_workers)

    def entrenar(self):
        """Inicia el server y coordina el entrenamiento usando select()"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(self.n_workers)

        print(f"Server escuchando en {HOST}:{PORT}")
        print(f"Esperando {self.n_workers} workers...")
        print(f"Épocas: {EPOCAS}, Workers: {self.n_workers}, LR: {TASA_APRENDIZAJE}")
        print("-" * 50)

        # Aceptar conexiones de todos los workers
        conexiones = []
        for i in range(self.n_workers):
            conn, addr = server.accept()
            print(f"Worker {i} conectado desde {addr}")
            conexiones.append(conn)

        print("-" * 50)
        print("Enviando dataset inicial a los workers (Esto puede tomar unos segundos)...")

        for i, conn in enumerate(conexiones):
            mensaje_init = {
                'tipo': 'INIT',
                'X': self.partes[i][0],
                'y': self.partes[i][1]
            }
            enviar_objeto(conn, mensaje_init)

        print("Distribución inicial del dataset exitosa. Iniciando entrenamiento...")
        print("-" * 50)

        historial_perdida = []
        inicializar_csv(self.ruta_csv, self.n_workers)

        for epoca in range(EPOCAS):
            t_epoca_inicio = time.perf_counter()

            # PASO 1: Enviar parámetros a TODOS los workers
            for i, conn in enumerate(conexiones):
                mensaje = {
                    'tipo': 'TRAIN',
                    'params': self.params,
                    'tasa_aprendizaje': TASA_APRENDIZAJE,
                    'epoca': epoca
                }
                enviar_objeto(conn, mensaje)

            # PASO 2: Recibir resultados usando select()
            resultados = []
            sockets_pendientes = list(conexiones)

            while sockets_pendientes:
                legibles, _, excepciones = select.select(
                    sockets_pendientes,
                    [],
                    sockets_pendientes,
                    SELECT_TIMEOUT
                )

                for sock in excepciones:
                    print(f"[Server] Error en socket, removiendo...")
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
                                # Campos nuevos — con .get() para compatibilidad
                                'tiempo_computo': respuesta.get('tiempo_computo', -1),
                                'n_procesos':     respuesta.get('n_procesos', -1),
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

            # PASO 3: Promediar pesos
            if len(resultados) == self.n_workers:
                params_parciales   = [r['params'] for r in resultados]
                self.params        = promediar_pesos(params_parciales)
                perdida_promedio   = sum(r['perdida'] for r in resultados) / self.n_workers
                tiempos_worker     = [r['tiempo_computo'] for r in resultados]
                n_procs_workers    = [r['n_procesos']     for r in resultados]
                historial_perdida.append(perdida_promedio)

                # Calcular precisión en test cada 10 épocas (costoso, no hacerlo cada época)
                if epoca % 10 == 0:
                    precision_test = calcular_precision(self.X_test, self.y_test, self.params)
                    print(f"\n[Server] Época {epoca:>3} | "
                          f"Pérdida: {perdida_promedio:.4f} | "
                          f"Precisión test: {precision_test*100:.2f}% | "
                          f"T_época: {tiempo_epoca:.2f}s")
                    print("-" * 50)
                else:
                    precision_test = -1.0   # No calculada esta época

                # Registrar en CSV
                registrar_epoca_csv(
                    self.ruta_csv, self.n_workers, epoca,
                    perdida_promedio, precision_test,
                    tiempo_epoca, tiempos_worker, n_procs_workers
                )
            else:
                print(f"[Server] Advertencia: Solo se recibieron "
                      f"{len(resultados)} de {self.n_workers} resultados")

        # Cerrar conexiones
        for conn in conexiones:
            try:
                enviar_objeto(conn, {'tipo': 'DONE'})
                conn.close()
            except Exception:
                pass

        server.close()
        return historial_perdida

# ============ EJECUCION ============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Servidor de entrenamiento distribuido',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=N_WORKERS,
        help='Número de workers a esperar'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='resultados/metricas.csv',
        help='Ruta del archivo CSV donde se guardan las métricas'
    )
    args = parser.parse_args()

    # Sobreescribir constante global para que dividir_dataset y demás la usen
    N_WORKERS = args.workers

    print("Creando servidor de entrenamiento...")
    print(f"Workers esperados: {N_WORKERS} | CSV: {args.csv}")
    print("=" * 50)

    server = TrainingServer(n_workers=N_WORKERS, ruta_csv=args.csv)

    print("Iniciando entrenamiento distribuido...")
    historial = server.entrenar()

    print("\n" + "=" * 50)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 50)

    precision_train = calcular_precision(server.X_train, server.y_train, server.params)
    precision_test  = calcular_precision(server.X_test,  server.y_test,  server.params)

    print(f"Precisión en entrenamiento : {precision_train:.4f} ({precision_train*100:.2f}%)")
    print(f"Precisión en prueba        : {precision_test:.4f}  ({precision_test*100:.2f}%)")
    print(f"Pérdida final              : {historial[-1]:.4f}")
    print(f"Métricas guardadas en      : {args.csv}")

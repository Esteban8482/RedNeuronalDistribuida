import socket
import pickle
import select
import numpy as np
import csv
import os
import time
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_WORKERS = 7                     # número total de workers que esperamos conecten
N_WORKERS = MAX_WORKERS             # para compatibilidad con código previo
EXPERIMENTOS_PROCESOS = list(range(1, MAX_WORKERS + 1))  # 1..7 workers
EPOCAS = 100
TASA_APRENDIZAJE = 0.1
HOST = '0.0.0.0'
PORT = 5000
BUFFER_SIZE = 4096 * 1024
SELECT_TIMEOUT = None
DIRECTORIO_RESULTADOS = 'resultados'

# --------------------------------------------------
# FUNCIONES RED NEURONAL
# --------------------------------------------------
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

# --------------------------------------------------
# COMUNICACIÓN: enviar/recibir con header de 8 bytes
# --------------------------------------------------
def enviar_objeto(sock, obj):
    data = pickle.dumps(obj)
    size = len(data)
    sock.sendall(size.to_bytes(8, byteorder='big'))
    sock.sendall(data)

def recibir_objeto(sock):
    try:
        size_bytes = sock.recv(8)
    except Exception:
        return None
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

# --------------------------------------------------
# LOG / CSV
# --------------------------------------------------
def inicializar_csv(ruta_csv, n_workers, n_procesos):
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

# --------------------------------------------------
# SERVIDOR DE EXPERIMENTOS
# --------------------------------------------------
class ExperimentServer:
    def __init__(self):
        print("=" * 60)
        print("INICIANDO SERVIDOR DE EXPERIMENTACIÓN")
        print("=" * 60)
        print(f"Configuración del experimento:")
        print(f"  - Workers esperados     : {N_WORKERS}")
        print(f"  - Configuraciones       : {EXPERIMENTOS_PROCESOS}")
        print(f"  - Épocas por experimento: {EPOCAS}")
        print(f"  - Tasa de aprendizaje   : {TASA_APRENDIZAJE}")
        print(f"  - Directorio resultados : {DIRECTORIO_RESULTADOS}/")
        print("=" * 60)

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
        os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

    def _crear_parametros_iniciales(self):
        return {
            'W1': inicializar_pesos(784, 256),
            'b1': inicializar_sesgos(256),
            'W2': inicializar_pesos(256, 10),
            'b2': inicializar_sesgos(10)
        }

    def _esperar_conexiones(self, server_socket):
        """
        Espera MAX_WORKERS conexiones y lee la primera mensaje HELLO de cada worker
        que debe contener {'tipo':'HELLO','region':...,'name':...}
        Retorna lista de dicts: {'sock':conn,'addr':addr,'region':region,'name':name}
        """
        print(f"\n[Server] Esperando {MAX_WORKERS} workers (aceptar conexiones)...")
        conexiones = []
        for i in range(MAX_WORKERS):
            conn, addr = server_socket.accept()
            print(f"[Server] Conexión entrante {i+1}/{MAX_WORKERS} desde {addr}")
            # leer mensaje HELLO
            hello = recibir_objeto(conn)
            region = 'unknown'
            name = ''
            if hello and isinstance(hello, dict) and hello.get('tipo') == 'HELLO':
                region = hello.get('region', 'unknown')
                name = hello.get('name', '')
            else:
                print("[Server] Advertencia: no se recibió HELLO formal del worker; asignando region='unknown'")
            conexiones.append({'sock': conn, 'addr': addr, 'region': region, 'name': name})
            print(f"  -> Worker registrado: name={name} region={region}")
        print("[Server] Todos los workers han sido aceptados.")
        return conexiones

    def _enviar_inicializacion(self, conexiones_seleccionadas, partes):
        print("[Server] Enviando particiones de dataset a workers activos...")
        for i, entry in enumerate(conexiones_seleccionadas):
            conn = entry['sock']
            mensaje_init = {
                'tipo': 'INIT',
                'X': partes[i][0],
                'y': partes[i][1]
            }
            enviar_objeto(conn, mensaje_init)
        print("[Server] Dataset distribuido a workers activos exitosamente")

    def _select_participantes(self, conexiones, n_active):
        """
        Selecciona n_active workers asegurando incluir al menos uno con region 'westus2'
        """
        # normalizar region strings
        for c in conexiones:
            c['region_norm'] = c['region'].lower() if isinstance(c['region'], str) else 'unknown'
        west = [c for c in conexiones if c['region_norm'] == 'westus2']
        others = [c for c in conexiones if c['region_norm'] != 'westus2']
        selected = []

        if len(conexiones) < n_active:
            print(f"[Server] ADVERTENCIA: solo {len(conexiones)} workers conectados, requerido {n_active}. Se usará lo disponible.")
            n_active = len(conexiones)

        # siempre incluir uno westus2 si existe
        if west:
            selected.append(west[0])

        # completar hasta n_active por orden: preferir otros no seleccionados
        for c in conexiones:
            if c in selected:
                continue
            selected.append(c)
            if len(selected) >= n_active:
                break

        # si no alcanzó (caso raro), reintentar llenando con others
        if len(selected) < n_active:
            for c in others:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= n_active:
                        break

        # mostrar selección
        print(f"[Server] Seleccionados {len(selected)}/{n_active} workers para este experimento:")
        for idx, s in enumerate(selected):
            print(f"   #{idx+1} -> name={s.get('name')} region={s.get('region')} addr={s.get('addr')}")
        return selected

    def _ejecutar_entrenamiento(self, conexiones_seleccionadas, params, ruta_csv, n_procesos):
        historial_perdida = []
        n_activos = len(conexiones_seleccionadas)
        inicializar_csv(ruta_csv, n_activos, n_procesos)
        tiempo_inicio_total = time.perf_counter()

        # sockets list
        sockets = [c['sock'] for c in conexiones_seleccionadas]

        for epoca in range(EPOCAS):
            t_epoca_inicio = time.perf_counter()

            # Enviar parámetros a todos los workers activos
            for conn in sockets:
                mensaje = {
                    'tipo': 'TRAIN',
                    'params': params,
                    'tasa_aprendizaje': TASA_APRENDIZAJE,
                    'epoca': epoca
                }
                enviar_objeto(conn, mensaje)

            # Recibir resultados (select)
            resultados = []
            sockets_pendientes = list(sockets)

            while sockets_pendientes:
                legibles, _, excepciones = select.select(sockets_pendientes, [], sockets_pendientes, SELECT_TIMEOUT)

                for sock in excepciones:
                    if sock in sockets_pendientes:
                        sockets_pendientes.remove(sock)

                for sock in legibles:
                    try:
                        respuesta = recibir_objeto(sock)
                        if respuesta and respuesta.get('tipo') == 'RESULT':
                            # map sock -> index of conexiones_seleccionadas
                            try:
                                worker_idx = sockets.index(sock)
                            except ValueError:
                                worker_idx = -1
                            resultados.append({
                                'worker_idx': worker_idx,
                                'params': respuesta['params'],
                                'perdida': respuesta['perdida'],
                                'tiempo_computo': respuesta.get('tiempo_computo', -1),
                            })
                            if sock in sockets_pendientes:
                                sockets_pendientes.remove(sock)
                        elif respuesta is None:
                            print(f"[Server] Worker desconectado inesperadamente durante epoca {epoca}")
                            if sock in sockets_pendientes:
                                sockets_pendientes.remove(sock)
                    except Exception as e:
                        print(f"[Server] Error recibiendo de worker: {e}")
                        if sock in sockets_pendientes:
                            sockets_pendientes.remove(sock)

            t_epoca_fin = time.perf_counter()
            tiempo_epoca = t_epoca_fin - t_epoca_inicio

            # si recibimos resultados de todos los activos -> promediar
            if len(resultados) == n_activos:
                params_parciales = [r['params'] for r in resultados]
                params = promediar_pesos(params_parciales)
                perdida_promedio = sum(r['perdida'] for r in resultados) / n_activos
                tiempos_worker = [r['tiempo_computo'] for r in resultados]
                historial_perdida.append(perdida_promedio)

                if epoca % 10 == 0:
                    precision_test = calcular_precision(self.X_test, self.y_test, params)
                    print(f"  Época {epoca:>3}/{EPOCAS} | Pérdida: {perdida_promedio:.4f} | Test: {precision_test*100:.1f}% | T: {tiempo_epoca:.2f}s")
                else:
                    precision_test = -1.0

                registrar_epoca_csv(ruta_csv, n_activos, n_procesos, epoca, perdida_promedio, precision_test, tiempo_epoca, tiempos_worker)
            else:
                print(f"[Server] ADVERTENCIA: Solo {len(resultados)}/{n_activos} resultados en la epoca {epoca}")
                # registrar igualmente con valores -1
                registrar_epoca_csv(ruta_csv, n_activos, n_procesos, epoca, -1.0, -1.0, tiempo_epoca, [])

        tiempo_total = time.perf_counter() - tiempo_inicio_total
        return historial_perdida, tiempo_total, params

    def ejecutar_experimentos(self):
        print("\n" + "=" * 60)
        print("INICIANDO MALLA DE EXPERIMENTACIÓN")
        print("=" * 60)

        resumen_experimentos = []

        # Crear socket servidor y aceptar conexiones una vez
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(MAX_WORKERS)
        conexiones = self._esperar_conexiones(server_socket)
        server_socket.close()

        # ejecutar experimentos 1..MAX_WORKERS
        for idx, n_procesos in enumerate(EXPERIMENTOS_PROCESOS):
            print("\n" + "─" * 60)
            print(f"EXPERIMENTO {idx+1}/{len(EXPERIMENTOS_PROCESOS)}: USANDO {n_procesos} WORKER(S)")
            print("─" * 60)

            # seleccionar subset que garantice al menos un westus2
            seleccion = self._select_participantes(conexiones, n_procesos)

            # preparar parametros y dataset dividido en n_procesos
            params = self._crear_parametros_iniciales()
            partes = dividir_dataset(self.X_train, self.y_train_one_hot, len(seleccion))
            ruta_csv = os.path.join(DIRECTORIO_RESULTADOS, f'experimento_{n_procesos}workers.csv')

            # enviar init sólo a los seleccionados
            self._enviar_inicializacion(seleccion, partes)

            print(f"\n[Server] Iniciando entrenamiento con {len(seleccion)} workers...")
            historial, tiempo_total, params_finales = self._ejecutar_entrenamiento(seleccion, params, ruta_csv, n_procesos)

            # métricas finales
            precision_train = calcular_precision(self.X_train, self.y_train, params_finales)
            precision_test = calcular_precision(self.X_test, self.y_test, params_finales)

            resumen_experimentos.append({
                'n_procesos': n_procesos,
                'perdida_final': historial[-1] if historial else -1,
                'precision_train': precision_train,
                'precision_test': precision_test,
                'tiempo_total': tiempo_total,
                'archivo_csv': ruta_csv
            })

            print(f"\n[Server] Experimento {n_procesos} workers completado:")
            print(f"  - Tiempo total     : {tiempo_total:.2f}s")
            print(f"  - Pérdida final    : {historial[-1] if historial else -1:.4f}")
            print(f"  - Precisión train  : {precision_train*100:.2f}%")
            print(f"  - Precisión test   : {precision_test*100:.2f}%")
            print(f"  - Resultados en    : {ruta_csv}")

            # pequeña pausa
            time.sleep(1)

        # Al finalizar TODOS los experimentos notificamos a todos los workers y cerramos
        print("\n[Server] Notificando a todos los workers que los experimentos finalizaron...")
        for c in conexiones:
            try:
                enviar_objeto(c['sock'], {'tipo': 'EXPERIMENT_END'})
                c['sock'].close()
            except Exception:
                pass

        self._mostrar_resumen_final(resumen_experimentos)

    def _mostrar_resumen_final(self, resumen):
        print("\n" + "=" * 60)
        print("RESUMEN DE TODOS LOS EXPERIMENTOS")
        print("=" * 60)
        print(f"{'Workers':>7} | {'Tiempo':>10} | {'Pérdida':>10} | {'Train%':>8} | {'Test%':>8}")
        print("-" * 60)
        for exp in resumen:
            print(f"{exp['n_procesos']:>7} | {exp['tiempo_total']:>10.2f}s | "
                  f"{exp['perdida_final']:>10.4f} | {exp['precision_train']*100:>7.2f}% | "
                  f"{exp['precision_test']*100:>7.2f}%")
        print("=" * 60)
        print(f"\nTodos los archivos CSV guardados en: {DIRECTORIO_RESULTADOS}/")

if __name__ == '__main__':
    server = ExperimentServer()
    server.ejecutar_experimentos()
    print("\n[Server] Todos los experimentos completados. Finalizando...")
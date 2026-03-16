import socket
import pickle
import select
import numpy as np
import csv
import os
import time
import traceback
import sys
from datetime import datetime
from sklearn.datasets import fetch_openml

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_WORKERS = 7
N_WORKERS = MAX_WORKERS
EXPERIMENTOS_PROCESOS = list(range(1, MAX_WORKERS + 1))
EPOCAS = 100
TASA_APRENDIZAJE = 0.1
HOST = '0.0.0.0'
PORT = 5000
BUFFER_SIZE = 4096 * 1024
SELECT_TIMEOUT = None
DIRECTORIO_RESULTADOS = 'resultados'

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
LOG_FILE = None

def log(msg, level="INFO"):
    """Log con timestamp y nivel"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    caller = sys._getframe(1).f_code.co_name
    line = sys._getframe(1).f_lineno
    log_msg = f"[{timestamp}] [{level}] [{caller}:{line}] {msg}"
    print(log_msg)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(log_msg + "\n")

def log_worker_state(worker_id, state, extra=""):
    """Log específico para estado de workers"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    msg = f"[{timestamp}] [WORKER_STATE] Worker {worker_id}: {state} {extra}"
    print(msg)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")

def log_experiment_state(exp_num, phase, details=""):
    """Log específico para estado de experimentos"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    msg = f"[{timestamp}] [EXP_STATE] Experiment {exp_num} | {phase} | {details}"
    print(msg)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")

def log_socket_state(sock, label=""):
    """Log estado del socket"""
    try:
        fileno = sock.fileno()
        local_addr = sock.getsockname() if sock.fileno() != -1 else "CLOSED"
        peer_addr = sock.getpeername() if sock.fileno() != -1 else "N/A"
        log(f"Socket {label}: fd={fileno}, local={local_addr}, peer={peer_addr}")
    except Exception as e:
        log(f"Socket {label}: ERROR getting state - {e}", "ERROR")

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
def enviar_objeto(sock, obj, context=""):
    """Envía un objeto serializado por el socket con timeout"""
    try:
        data = pickle.dumps(obj)
        size = len(data)
        log(f"Enviando objeto ({context}): tipo={obj.get('tipo', 'N/A')}, size={size} bytes")
        
        # Enviar tamaño (8 bytes)
        size_bytes = size.to_bytes(8, byteorder='big')
        sock.sendall(size_bytes)
        log(f"  Header enviado: 8 bytes")
        
        # Enviar datos
        sock.sendall(data)
        log(f"  Datos enviados: {size} bytes")
        return True
    except Exception as e:
        log(f"ERROR enviando objeto ({context}): {e}", "ERROR")
        return False

def enviar_objeto_con_timeout(sock, obj, context="", timeout=60.0):
    """Envía un objeto con timeout para evitar bloqueos indefinidos"""
    old_timeout = None
    try:
        old_timeout = sock.gettimeout()
        sock.settimeout(timeout)
        result = enviar_objeto(sock, obj, context)
        return result
    except Exception as e:
        log(f"ERROR enviando objeto con timeout ({context}): {e}", "ERROR")
        return False
    finally:
        try:
            if old_timeout is not None:
                sock.settimeout(old_timeout)
        except:
            pass

def recibir_objeto(sock, context=""):
    try:
        log(f"Esperando recibir objeto ({context})...")
        
        # Recibir tamaño (8 bytes)
        size_bytes = b''
        while len(size_bytes) < 8:
            chunk = sock.recv(8 - len(size_bytes))
            if not chunk:
                log(f"  Conexión cerrada mientras leía header ({context})", "WARN")
                return None
            size_bytes += chunk
        
        size = int.from_bytes(size_bytes, byteorder='big')
        log(f"  Header recibido: objeto de {size} bytes")
        
        # Recibir datos
        data = b''
        bytes_recibidos = 0
        while len(data) < size:
            chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
            if not chunk:
                log(f"  Conexión cerrada mientras leía datos ({context}). Recibidos {len(data)}/{size} bytes", "WARN")
                return None
            data += chunk
            bytes_recibidos += len(chunk)
            if bytes_recibidos % (1024*1024) == 0:
                log(f"  Progreso: {bytes_recibidos}/{size} bytes ({100*bytes_recibidos/size:.1f}%)")
        
        obj = pickle.loads(data)
        log(f"  Objeto recibido completamente ({context}): tipo={obj.get('tipo', 'N/A') if isinstance(obj, dict) else type(obj).__name__}")
        return obj
    except Exception as e:
        log(f"ERROR recibiendo objeto ({context}): {e}", "ERROR")
        return None

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
    log(f"Archivo de métricas creado: {ruta_csv}")

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
        global LOG_FILE
        LOG_FILE = os.path.join(DIRECTORIO_RESULTADOS, f"server_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)
        
        log("=" * 80)
        log("INICIANDO SERVIDOR DE EXPERIMENTACIÓN (DEBUG MODE)")
        log("=" * 80)
        log(f"Configuración del experimento:")
        log(f"  - Workers esperados     : {N_WORKERS}")
        log(f"  - Configuraciones       : {EXPERIMENTOS_PROCESOS}")
        log(f"  - Épocas por experimento: {EPOCAS}")
        log(f"  - Tasa de aprendizaje   : {TASA_APRENDIZAJE}")
        log(f"  - Directorio resultados : {DIRECTORIO_RESULTADOS}/")
        log(f"  - Log file              : {LOG_FILE}")
        log("=" * 80)

        log("Cargando dataset MNIST...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32').to_numpy() / 255.0
        y = mnist.target.astype('int32').to_numpy()

        log("Preprocesando datos...")
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        self.X_train, self.X_test = X[:60000], X[60000:]
        self.y_train, self.y_test = y[:60000], y[60000:]
        self.y_train_one_hot = one_hot(self.y_train)

        log(f"Dataset listo: {len(self.X_train)} train, {len(self.X_test)} test")
        os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)
        
        # Almacenar datos particionados para cada worker
        self.datos_por_worker = {}

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
        """
        log_experiment_state(0, "CONNECTION_PHASE", f"Esperando {MAX_WORKERS} workers")
        conexiones = []
        
        for i in range(MAX_WORKERS):
            log(f"Esperando conexión {i+1}/{MAX_WORKERS}...")
            try:
                conn, addr = server_socket.accept()
                log(f"Conexión entrante {i+1}/{MAX_WORKERS} desde {addr}")
                log_socket_state(conn, f"new_conn_{i+1}")
                
                # leer mensaje HELLO
                log(f"  Esperando mensaje HELLO de worker {i+1}...")
                hello = recibir_objeto(conn, f"HELLO_worker_{i+1}")
                
                region = 'unknown'
                name = ''
                if hello and isinstance(hello, dict) and hello.get('tipo') == 'HELLO':
                    region = hello.get('region', 'unknown')
                    name = hello.get('name', '')
                    log(f"  HELLO recibido correctamente: name={name}, region={region}")
                else:
                    log(f"  ADVERTENCIA: HELLO inválido recibido: {hello}", "WARN")
                
                worker_info = {
                    'sock': conn, 
                    'addr': addr, 
                    'region': region, 
                    'name': name,
                    'worker_id': i + 1,
                    'connected_at': time.time(),
                    'initialized': False  # Track initialization status
                }
                conexiones.append(worker_info)
                log_worker_state(i+1, "CONNECTED", f"name={name}, region={region}, addr={addr}")
                
            except Exception as e:
                log(f"ERROR aceptando conexión {i+1}: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")
        
        log(f"Todas las conexiones aceptadas. Total workers: {len(conexiones)}")
        for c in conexiones:
            log(f"  Worker {c['worker_id']}: {c['name']} ({c['region']}) from {c['addr']}")
        
        return conexiones

    def _inicializar_todos_los_workers(self, conexiones):
        """
        CRITICAL FIX: Initialize ALL workers immediately after connection.
        This ensures every worker is ready to participate in any experiment.
        """
        log("=" * 80)
        log("INICIALIZANDO TODOS LOS WORKERS (CRITICAL FIX)")
        log("=" * 80)
        
        # Dividir el dataset en MAX_WORKERS partes (una para cada worker)
        partes = dividir_dataset(self.X_train, self.y_train_one_hot, MAX_WORKERS)
        
        # Enviar INIT a cada worker con su porción de datos
        for i, worker_info in enumerate(conexiones):
            conn = worker_info['sock']
            worker_id = worker_info['worker_id']
            
            log(f"Enviando INIT a worker {worker_id} ({worker_info['name']})...")
            log_socket_state(conn, f"worker_{worker_id}_init_all")
            
            try:
                # Establecer timeout para evitar bloqueos indefinidos
                conn.settimeout(120.0)  # 2 minutos para INIT (datos grandes)
                
                mensaje_init = {
                    'tipo': 'INIT',
                    'X': partes[i][0],
                    'y': partes[i][1],
                    'exp_num': 0,  # exp_num=0 indica inicialización global
                    'worker_idx': i
                }
                
                success = enviar_objeto(conn, mensaje_init, f"GLOBAL_INIT_to_worker_{worker_id}")
                
                if success:
                    worker_info['initialized'] = True
                    log(f"  INIT enviado exitosamente a worker {worker_id}")
                else:
                    log(f"  FALLO al enviar INIT a worker {worker_id}", "ERROR")
                    
            except socket.timeout:
                log(f"  TIMEOUT enviando INIT a worker {worker_id}", "ERROR")
            except Exception as e:
                log(f"  ERROR enviando INIT a worker {worker_id}: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")
        
        # Contar cuántos workers se inicializaron correctamente
        initialized_count = sum(1 for c in conexiones if c.get('initialized', False))
        log(f"Workers inicializados: {initialized_count}/{len(conexiones)}")
        
        if initialized_count < MAX_WORKERS:
            log("ADVERTENCIA: No todos los workers se inicializaron correctamente", "WARN")
        
        log("=" * 80)
        
        # Guardar las particiones de datos para usarlas en los experimentos
        self.datos_por_worker = {i: partes[i] for i in range(MAX_WORKERS)}

    def _enviar_inicializacion_experimento(self, conexiones_seleccionadas, exp_num):
        """
        Envía INIT a los workers seleccionados para un experimento específico.
        Como los workers ya están inicializados, esto es más un "re-init" para
        sincronizar el número de experimento.
        """
        log_experiment_state(exp_num, "INIT_PHASE", f"Enviando INIT a {len(conexiones_seleccionadas)} workers")
        
        for i, worker_info in enumerate(conexiones_seleccionadas):
            conn = worker_info['sock']
            worker_id = worker_info['worker_id']
            worker_idx_global = worker_info['worker_id'] - 1  # 0-based index
            
            log(f"Enviando INIT a worker {worker_id} ({worker_info['name']})...")
            log_socket_state(conn, f"worker_{worker_id}_pre_init")
            
            try:
                # Usar los datos pre-particionados
                X_worker, y_worker = self.datos_por_worker[worker_idx_global]
                
                mensaje_init = {
                    'tipo': 'INIT',
                    'X': X_worker,
                    'y': y_worker,
                    'exp_num': exp_num,
                    'worker_idx': i  # índice dentro del experimento actual
                }
                
                success = enviar_objeto_con_timeout(conn, mensaje_init, f"INIT_to_worker_{worker_id}", timeout=120.0)
                
                if success:
                    log(f"  INIT enviado exitosamente a worker {worker_id}")
                else:
                    log(f"  FALLO al enviar INIT a worker {worker_id}", "ERROR")
                    
            except Exception as e:
                log(f"  ERROR enviando INIT a worker {worker_id}: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")
        
        log("Dataset distribuido a workers activos")

    def _select_participantes(self, conexiones, n_active, exp_num):
        """
        Selecciona n_active workers asegurando incluir al menos uno con region 'westus2'.
        """
        log_experiment_state(exp_num, "SELECTION_PHASE", f"Seleccionando {n_active} workers de {len(conexiones)} disponibles")
        
        # normalizar region strings
        for c in conexiones:
            c['region_norm'] = c['region'].lower() if isinstance(c['region'], str) else 'unknown'
        
        west = [c for c in conexiones if c['region_norm'] == 'westus2']
        
        if len(conexiones) < n_active:
            log(f"ADVERTENCIA: solo {len(conexiones)} workers conectados, requerido {n_active}. Se usará lo disponible.", "WARN")
            n_active = len(conexiones)
        
        # Seleccionar los primeros N workers
        selected = conexiones[:n_active]
        log(f"  Selección inicial (primeros {n_active}): {[c['worker_id'] for c in selected]}")
        
        # Verificar si hay al menos un westus2
        has_west = any(c['region_norm'] == 'westus2' for c in selected)
        
        if not has_west and west:
            # Reemplazar el último worker seleccionado con el primer westus2
            west_worker = west[0]
            if west_worker not in selected:
                removed = selected.pop()
                selected.append(west_worker)
                log(f"  Reemplazado worker {removed['worker_id']} con westus2 worker {west_worker['worker_id']}")
        
        # Filtrar workers que no están inicializados (fallaron en la inicialización global)
        selected_inicializados = [c for c in selected if c.get('initialized', False)]
        if len(selected_inicializados) < len(selected):
            log(f"ADVERTENCIA: {len(selected) - len(selected_inicializados)} workers seleccionados no están inicializados", "WARN")
            selected = selected_inicializados
        
        log(f"Selección final: {len(selected)} workers")
        for idx, s in enumerate(selected):
            log(f"   #{idx+1} -> worker_id={s['worker_id']}, name={s.get('name')}, region={s.get('region')}, addr={s.get('addr')}")
        
        return selected

    def _ejecutar_entrenamiento(self, conexiones_seleccionadas, params, ruta_csv, n_procesos, exp_num):
        log_experiment_state(exp_num, "TRAINING_START", f"Iniciando entrenamiento con {len(conexiones_seleccionadas)} workers")
        
        historial_perdida = []
        n_activos = len(conexiones_seleccionadas)
        
        if n_activos == 0:
            log("ERROR: No hay workers activos para el entrenamiento", "ERROR")
            return [], 0, params
        
        inicializar_csv(ruta_csv, n_activos, n_procesos)
        tiempo_inicio_total = time.perf_counter()

        sockets = [c['sock'] for c in conexiones_seleccionadas]
        
        log("Estado inicial de sockets:")
        for i, c in enumerate(conexiones_seleccionadas):
            log_socket_state(c['sock'], f"worker_{c['worker_id']}_start")

        for epoca in range(EPOCAS):
            log_experiment_state(exp_num, f"EPOCH_{epoca}", "Iniciando época")
            t_epoca_inicio = time.perf_counter()

            # Enviar parámetros a todos los workers activos
            log(f"Enviando TRAIN a {n_activos} workers para época {epoca}...")
            for i, conn in enumerate(sockets):
                worker_id = conexiones_seleccionadas[i]['worker_id']
                try:
                    mensaje = {
                        'tipo': 'TRAIN',
                        'params': params,
                        'tasa_aprendizaje': TASA_APRENDIZAJE,
                        'epoca': epoca,
                        'exp_num': exp_num
                    }
                    success = enviar_objeto_con_timeout(conn, mensaje, f"TRAIN_exp{exp_num}_ep{epoca}_worker{worker_id}", timeout=30.0)
                    if not success:
                        log(f"  FALLO enviando TRAIN a worker {worker_id}", "ERROR")
                except Exception as e:
                    log(f"  ERROR enviando TRAIN a worker {worker_id}: {e}", "ERROR")

            # Recibir resultados (select)
            resultados = []
            sockets_pendientes = list(sockets)
            workers_reportados = set()
            
            log(f"Esperando resultados de {len(sockets_pendientes)} workers...")

            while sockets_pendientes:
                log(f"  Select esperando en {len(sockets_pendientes)} sockets...")
                for i, s in enumerate(sockets_pendientes):
                    worker_idx = sockets.index(s)
                    worker_id = conexiones_seleccionadas[worker_idx]['worker_id']
                    log(f"    Socket pendiente {i}: worker {worker_id}")
                
                try:
                    legibles, _, excepciones = select.select(sockets_pendientes, [], sockets_pendientes, 60.0)
                    log(f"  Select retornó: {len(legibles)} legibles, {len(excepciones)} excepciones")
                except Exception as e:
                    log(f"  ERROR en select: {e}", "ERROR")
                    break

                for sock in excepciones:
                    worker_idx = sockets.index(sock) if sock in sockets else -1
                    worker_id = conexiones_seleccionadas[worker_idx]['worker_id'] if worker_idx >= 0 else "UNKNOWN"
                    log(f"  Excepción en socket de worker {worker_id}", "ERROR")
                    if sock in sockets_pendientes:
                        sockets_pendientes.remove(sock)

                for sock in legibles:
                    try:
                        worker_idx = sockets.index(sock)
                        worker_id = conexiones_seleccionadas[worker_idx]['worker_id']
                        
                        log(f"  Recibiendo resultado de worker {worker_id}...")
                        respuesta = recibir_objeto(sock, f"RESULT_exp{exp_num}_ep{epoca}_worker{worker_id}")
                        
                        if respuesta and respuesta.get('tipo') == 'RESULT':
                            resultados.append({
                                'worker_idx': worker_idx,
                                'worker_id': worker_id,
                                'params': respuesta['params'],
                                'perdida': respuesta['perdida'],
                                'tiempo_computo': respuesta.get('tiempo_computo', -1),
                            })
                            workers_reportados.add(worker_id)
                            log(f"    Resultado recibido de worker {worker_id}: loss={respuesta['perdida']:.4f}")
                            if sock in sockets_pendientes:
                                sockets_pendientes.remove(sock)
                        elif respuesta is None:
                            log(f"    Worker {worker_id} desconectado (respuesta=None)", "WARN")
                            if sock in sockets_pendientes:
                                sockets_pendientes.remove(sock)
                        else:
                            log(f"    Respuesta inesperada de worker {worker_id}: {respuesta}", "WARN")
                    except Exception as e:
                        log(f"    ERROR recibiendo de worker: {e}", "ERROR")
                        if sock in sockets_pendientes:
                            sockets_pendientes.remove(sock)

            t_epoca_fin = time.perf_counter()
            tiempo_epoca = t_epoca_fin - t_epoca_inicio
            
            log(f"Época {epoca} completada en {tiempo_epoca:.2f}s. Resultados: {len(resultados)}/{n_activos}")
            log(f"  Workers que reportaron: {sorted(workers_reportados)}")
            if len(resultados) < n_activos:
                faltantes = set(c['worker_id'] for c in conexiones_seleccionadas) - workers_reportados
                log(f"  Workers que NO reportaron: {sorted(faltantes)}", "WARN")

            # si recibimos resultados de todos los activos -> promediar
            if len(resultados) == n_activos:
                params_parciales = [r['params'] for r in resultados]
                params = promediar_pesos(params_parciales)
                perdida_promedio = sum(r['perdida'] for r in resultados) / n_activos
                tiempos_worker = [r['tiempo_computo'] for r in resultados]
                historial_perdida.append(perdida_promedio)

                if epoca % 10 == 0:
                    precision_test = calcular_precision(self.X_test, self.y_test, params)
                    log(f"  Época {epoca:>3}/{EPOCAS} | Pérdida: {perdida_promedio:.4f} | Test: {precision_test*100:.1f}% | T: {tiempo_epoca:.2f}s")
                else:
                    precision_test = -1.0

                registrar_epoca_csv(ruta_csv, n_activos, n_procesos, epoca, perdida_promedio, precision_test, tiempo_epoca, tiempos_worker)
            else:
                log(f"ADVERTENCIA: Solo {len(resultados)}/{n_activos} resultados en la epoca {epoca}", "WARN")
                registrar_epoca_csv(ruta_csv, n_activos, n_procesos, epoca, -1.0, -1.0, tiempo_epoca, [])

        tiempo_total = time.perf_counter() - tiempo_inicio_total
        log_experiment_state(exp_num, "TRAINING_END", f"Tiempo total: {tiempo_total:.2f}s")
        return historial_perdida, tiempo_total, params

    def ejecutar_experimentos(self):
        log("=" * 80)
        log("INICIANDO MALLA DE EXPERIMENTACIÓN")
        log("=" * 80)

        resumen_experimentos = []

        # Crear socket servidor y aceptar conexiones una vez
        log("Creando socket servidor...")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(MAX_WORKERS)
        log(f"Servidor escuchando en {HOST}:{PORT}")
        
        conexiones = self._esperar_conexiones(server_socket)
        server_socket.close()
        log("Socket servidor cerrado después de aceptar conexiones")
        
        # CRITICAL FIX: Inicializar TODOS los workers inmediatamente
        self._inicializar_todos_los_workers(conexiones)

        # ejecutar experimentos 1..MAX_WORKERS
        for idx, n_procesos in enumerate(EXPERIMENTOS_PROCESOS):
            exp_num = idx + 1
            log("=" * 80)
            log_experiment_state(exp_num, "EXPERIMENT_START", f"USANDO {n_procesos} WORKER(S)")
            log("=" * 80)

            # Verificar estado de conexiones antes de seleccionar
            log("Estado de conexiones antes de selección:")
            for c in conexiones:
                log_socket_state(c['sock'], f"worker_{c['worker_id']}_pre_exp")

            # seleccionar subset que garantice al menos un westus2
            seleccion = self._select_participantes(conexiones, n_procesos, exp_num)
            
            if len(seleccion) == 0:
                log("ERROR: No hay workers disponibles para este experimento", "ERROR")
                continue

            # preparar parametros
            params = self._crear_parametros_iniciales()
            ruta_csv = os.path.join(DIRECTORIO_RESULTADOS, f'experimento_{n_procesos}workers.csv')

            # enviar init a los seleccionados (re-init para sincronizar exp_num)
            self._enviar_inicializacion_experimento(seleccion, exp_num)

            log(f"Iniciando entrenamiento con {len(seleccion)} workers...")
            historial, tiempo_total, params_finales = self._ejecutar_entrenamiento(seleccion, params, ruta_csv, n_procesos, exp_num)

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

            log(f"Experimento {n_procesos} workers completado:")
            log(f"  - Tiempo total     : {tiempo_total:.2f}s")
            log(f"  - Pérdida final    : {historial[-1] if historial else -1:.4f}")
            log(f"  - Precisión train  : {precision_train*100:.2f}%")
            log(f"  - Precisión test   : {precision_test*100:.2f}%")
            log(f"  - Resultados en    : {ruta_csv}")

            # pequeña pausa entre experimentos
            log("Pausa de 2 segundos entre experimentos...")
            time.sleep(2)
            
            # Verificar estado de todas las conexiones después del experimento
            log("Estado de conexiones después del experimento:")
            for c in conexiones:
                log_socket_state(c['sock'], f"worker_{c['worker_id']}_post_exp")

        # Al finalizar TODOS los experimentos notificamos a todos los workers y cerramos
        log("Notificando a todos los workers que los experimentos finalizaron...")
        for c in conexiones:
            try:
                log(f"  Enviando EXPERIMENT_END a worker {c['worker_id']} ({c['name']})...")
                enviar_objeto_con_timeout(c['sock'], {'tipo': 'EXPERIMENT_END'}, f"END_to_worker_{c['worker_id']}", timeout=30.0)
                c['sock'].close()
                log(f"    Worker {c['worker_id']} notificado y desconectado")
            except Exception as e:
                log(f"    Error notificando a worker {c['worker_id']}: {e}", "ERROR")

        self._mostrar_resumen_final(resumen_experimentos)

    def _mostrar_resumen_final(self, resumen):
        log("=" * 80)
        log("RESUMEN DE TODOS LOS EXPERIMENTOS")
        log("=" * 80)
        log(f"{'Workers':>7} | {'Tiempo':>10} | {'Pérdida':>10} | {'Train%':>8} | {'Test%':>8}")
        log("-" * 60)
        for exp in resumen:
            log(f"{exp['n_procesos']:>7} | {exp['tiempo_total']:>10.2f}s | "
                  f"{exp['perdida_final']:>10.4f} | {exp['precision_train']*100:>7.2f}% | "
                  f"{exp['precision_test']*100:>7.2f}%")
        log("=" * 80)
        log(f"Todos los archivos CSV guardados en: {DIRECTORIO_RESULTADOS}/")

if __name__ == '__main__':
    server = ExperimentServer()
    server.ejecutar_experimentos()
    log("Todos los experimentos completados. Finalizando...")
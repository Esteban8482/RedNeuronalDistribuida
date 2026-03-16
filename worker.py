import socket
import pickle
import numpy as np
import time
import argparse
import socket as s_mod
import os
import sys
import traceback
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--region', type=str, default=None, help='region del worker (ej. westus2, eastus)')
parser.add_argument('--log-dir', type=str, default='worker_logs', help='directorio para logs')
args = parser.parse_args()

HOST = '172.178.121.181'
PORT = 5000
BUFFER_SIZE = 4096 * 1024
N_EPOCAS = 100

# --------------------------------------------------
# LOGGING DEBUG
# --------------------------------------------------
LOG_FILE = None
WORKER_ID = None

def init_logging():
    """Inicializa el archivo de log"""
    global LOG_FILE
    hostname = s_mod.gethostname()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILE = os.path.join(log_dir, f"worker_{hostname}_{timestamp}.log")
    log("=" * 80)
    log("WORKER INICIANDO (DEBUG MODE)")
    log("=" * 80)
    log(f"Hostname: {hostname}")
    log(f"Region: {get_region()}")
    log(f"Server: {HOST}:{PORT}")
    log(f"Log file: {LOG_FILE}")

def log(msg, level="INFO"):
    """Log con timestamp y nivel"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    caller = sys._getframe(1).f_code.co_name
    line = sys._getframe(1).f_lineno
    worker_tag = f"[W{WORKER_ID}]" if WORKER_ID else "[W?]"
    log_msg = f"[{timestamp}] {worker_tag} [{level}] [{caller}:{line}] {msg}"
    print(log_msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(log_msg + "\n")

def log_state(state, details=""):
    """Log específico para estados del worker"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    worker_tag = f"[W{WORKER_ID}]" if WORKER_ID else "[W?]"
    msg = f"[{timestamp}] {worker_tag} [STATE] {state} {details}"
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")

def log_socket_state(sock, label=""):
    """Log estado del socket"""
    try:
        fileno = sock.fileno()
        local_addr = sock.getsockname() if sock.fileno() != -1 else "CLOSED"
        try:
            peer_addr = sock.getpeername() if sock.fileno() != -1 else "N/A"
        except:
            peer_addr = "NOT_CONNECTED"
        log(f"Socket {label}: fd={fileno}, local={local_addr}, peer={peer_addr}")
    except Exception as e:
        log(f"Socket {label}: ERROR getting state - {e}", "ERROR")

def get_region():
    """Obtiene la región del worker"""
    return args.region or os.environ.get('REGION', 'unknown')

# --------------------------------------------------
# FUNCIONES DE RED NEURONAL
# --------------------------------------------------
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
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'perdida': perdida}

# --------------------------------------------------
# COMUNICACIÓN
# --------------------------------------------------
def enviar_objeto(sock, obj, context=""):
    """Envía un objeto serializado por el socket"""
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
        log(traceback.format_exc(), "ERROR")
        return False

def recibir_objeto(sock, context=""):
    """Recibe un objeto serializado del socket"""
    try:
        log(f"Esperando recibir objeto ({context})...")
        
        # Recibir tamaño (8 bytes)
        size_bytes = b''
        bytes_leidos = 0
        while bytes_leidos < 8:
            chunk = sock.recv(8 - bytes_leidos)
            if not chunk:
                log(f"  Conexión cerrada mientras leía header ({context}). Leídos {bytes_leidos}/8 bytes", "WARN")
                return None
            size_bytes += chunk
            bytes_leidos += len(chunk)
        
        size = int.from_bytes(size_bytes, byteorder='big')
        log(f"  Header recibido: objeto de {size} bytes")
        
        # Recibir datos
        data = b''
        bytes_recibidos = 0
        last_progress = 0
        while len(data) < size:
            chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
            if not chunk:
                log(f"  Conexión cerrada mientras leía datos ({context}). Recibidos {len(data)}/{size} bytes", "WARN")
                return None
            data += chunk
            bytes_recibidos += len(chunk)
            
            # Log progreso cada 10%
            progress = int(100 * bytes_recibidos / size)
            if progress >= last_progress + 10:
                log(f"  Progreso recepción: {bytes_recibidos}/{size} bytes ({progress}%)")
                last_progress = progress
        
        obj = pickle.loads(data)
        tipo = obj.get('tipo', 'N/A') if isinstance(obj, dict) else type(obj).__name__
        log(f"  Objeto recibido completamente ({context}): tipo={tipo}")
        return obj
        
    except Exception as e:
        log(f"ERROR recibiendo objeto ({context}): {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return None

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    init_logging()
    
    region = get_region()
    hostname = s_mod.gethostname()
    WORKER_ID = hostname  # Temporal hasta recibir asignación del servidor
    
    log_state("STARTUP", f"hostname={hostname}, region={region}")
    
    # Conectar al servidor
    log(f"Conectando a {HOST}:{PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Intentar conectar con reintentos
    connected = False
    for attempt in range(30):
        try:
            log(f"  Intento de conexión {attempt + 1}/30...")
            sock.connect((HOST, PORT))
            log("Conexión establecida!")
            log_socket_state(sock, "post_connect")
            connected = True
            break
        except Exception as e:
            log(f"  Fallo intento {attempt + 1}: {e}")
            time.sleep(2)
    
    if not connected:
        log("No se pudo conectar después de 30 intentos. Saliendo.", "ERROR")
        sys.exit(1)

    # Enviar HELLO con metadata
    try:
        log("Enviando mensaje HELLO al servidor...")
        hello_msg = {'tipo': 'HELLO', 'region': region, 'name': hostname}
        success = enviar_objeto(sock, hello_msg, "HELLO")
        if success:
            log("HELLO enviado exitosamente")
        else:
            log("FALLO al enviar HELLO", "ERROR")
    except Exception as e:
        log(f"Error enviando HELLO: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")

    # Variables de estado
    X, y = None, None
    exp_num = None
    worker_idx = None
    epoch_count = 0
    
    log_state("WAITING_FOR_INIT", "Esperando mensaje INIT del servidor...")

    # Bucle principal
    while True:
        log_socket_state(sock, "pre_receive")
        log("Esperando mensaje del servidor...")
        
        msg = recibir_objeto(sock, "main_loop")
        
        if msg is None:
            log("Mensaje recibido es None. Conexión probablemente cerrada.", "WARN")
            log_socket_state(sock, "after_none_msg")
            break
        
        msg_type = msg.get('tipo', 'UNKNOWN')
        log(f"Mensaje recibido: tipo={msg_type}")
        
        if msg_type == 'INIT':
            log_state("RECEIVED_INIT")
            X = msg['X']
            y = msg['y']
            exp_num = msg.get('exp_num', 'unknown')
            worker_idx = msg.get('worker_idx', 'unknown')
            
            # Actualizar WORKER_ID para logging
            WORKER_ID = f"{hostname}_E{exp_num}_W{worker_idx}"
            
            log(f"  Datos recibidos: {len(X)} muestras")
            log(f"  Experimento: {exp_num}")
            log(f"  Worker index: {worker_idx}")
            log(f"  Forma X: {X.shape if hasattr(X, 'shape') else 'N/A'}")
            log(f"  Forma y: {y.shape if hasattr(y, 'shape') else 'N/A'}")
            log_state("READY_FOR_TRAINING")
            
        elif msg_type == 'TRAIN':
            epoch_count += 1
            epoca = msg.get('epoca', 'unknown')
            exp_num_msg = msg.get('exp_num', 'unknown')
            
            log_state("RECEIVED_TRAIN", f"exp={exp_num_msg}, epoca={epoca}, total_epochs_processed={epoch_count}")
            
            t0 = time.perf_counter()
            
            try:
                log("  Calculando gradientes...")
                grad = calcular_gradientes(X, y, msg['params'])
                log(f"  Gradientes calculados: loss={grad['perdida']:.4f}")
                
                p = msg['params']
                lr = msg['tasa_aprendizaje']
                
                log("  Actualizando parámetros...")
                new_params = {
                    'W1': p['W1'] - lr * grad['dW1'],
                    'b1': p['b1'] - lr * grad['db1'],
                    'W2': p['W2'] - lr * grad['dW2'],
                    'b2': p['b2'] - lr * grad['db2']
                }
                
                tiempo_computo = round(time.perf_counter() - t0, 4)
                log(f"  Computación completada en {tiempo_computo}s")
                
                log("  Enviando RESULT al servidor...")
                result_msg = {
                    'tipo': 'RESULT',
                    'params': new_params,
                    'perdida': grad['perdida'],
                    'tiempo_computo': tiempo_computo
                }
                
                success = enviar_objeto(sock, result_msg, f"RESULT_ep{epoca}")
                if success:
                    log(f"  RESULT enviado exitosamente")
                else:
                    log(f"  FALLO al enviar RESULT", "ERROR")
                    
            except Exception as e:
                log(f"ERROR procesando TRAIN: {e}", "ERROR")
                log(traceback.format_exc(), "ERROR")
                
        elif msg_type == 'EXPERIMENT_END':
            log_state("RECEIVED_EXPERIMENT_END", f"Experimento {exp_num} completado. Total epochs procesadas: {epoch_count}")
            break
            
        else:
            log(f"Mensaje de tipo desconocido recibido: {msg_type}", "WARN")
    
    # Cierre
    log_state("SHUTDOWN", "Cerrando conexión...")
    try:
        sock.close()
        log("Socket cerrado exitosamente")
    except Exception as e:
        log(f"Error cerrando socket: {e}", "ERROR")
    
    log_state("FINISHED", f"Worker finalizado. Total epochs procesadas: {epoch_count}")
    log("=" * 80)

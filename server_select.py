# -*- coding: utf-8 -*-
"""
Server - Coordina el entrenamiento distribuido.
Recibe conexiones de workers, distribuye trabajo y promedia resultados.
"""

import socket
import pickle
import select
import numpy as np
from sklearn.datasets import fetch_openml

# ============ CONFIGURACION ============
HOST = '0.0.0.0'
PORT = 5000
N_WORKERS = 4
EPOCAS = 100
TASA_APRENDIZAJE = 0.1
BUFFER_SIZE = 4096 * 1024  # 4MB para manejar arrays grandes
SELECT_TIMEOUT = None  # None = esperar indefinidamente

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
    """Serializa y envia un objeto usando pickle"""
    data = pickle.dumps(obj)
    size = len(data)
    # Avisar del tamaño en un paquete de 8 bytes
    sock.sendall(size.to_bytes(8, byteorder='big'))
    # Enviar datos
    sock.sendall(data)

def recibir_objeto(sock):
    """Recibe y deserializa un objeto usando pickle"""
    # Recibir tamaño
    size_bytes = sock.recv(8)
    if not size_bytes:
        return None
    size = int.from_bytes(size_bytes, byteorder='big')
    
    # Recibir datos completos
    data = b''
    
    # Mientras que no hayamos recibido todo el objeto, seguir recibiendo
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data))) # Recibe como maximo el tamaño del buffer, si no recibe lo que falta
        if not chunk:
            return None
        data += chunk
    
    return pickle.loads(data)

def enviar_objeto_nonblocking(sock, obj):
    """
    Envia un objeto de forma no bloqueante.
    Retorna True si el envio fue completo, False si el socket no esta listo.
    """
    try:
        # Usar select para verificar si el socket esta listo para escribir
        _, writable, _ = select.select([], [sock], [], 0)
        if sock in writable:
            enviar_objeto(sock, obj)
            return True
        return False
    except Exception as e:
        print(f"[Server] Error enviando: {e}")
        return False

# ============ SERVER PRINCIPAL ============
class TrainingServer:
    def __init__(self):
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
        
        print("Dividiendo dataset entre workers...")
        self.partes = dividir_dataset(self.X_train, self.y_train_one_hot, N_WORKERS)
        
    def entrenar(self):
        """Inicia el server y coordina el entrenamiento usando select()"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Definir protocolos IPv4 y TCP
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Permitir reutilizar direcciones en caso de reinicios rapidos
        server.bind((HOST, PORT)) # Asociar el socket a la direccion y puerto especificados
        server.listen(N_WORKERS) # Escuchar, el argumento es el maximo de conexiones 
        
        print(f"Server escuchando en {HOST}:{PORT}")
        print(f"Esperando {N_WORKERS} workers...")
        print(f"Epocas: {EPOCAS}, Workers: {N_WORKERS}, LR: {TASA_APRENDIZAJE}")
        print("-" * 50)
        
        # Aceptar conexiones de todos los workers
        conexiones = []
        for i in range(N_WORKERS):
            conn, addr = server.accept() # Esperar a que un worker se conecte, retorna un nuevo socket para comunicarse con ese worker y su direccion   
            print(f"Worker {i} conectado desde {addr}")
            conexiones.append(conn) # Guardar el socket de cada worker para comunicarse con ellos
        
        print("-" * 50)
        print("Enviando dataset inicial a los workers (Esto puede tomar unos segundos)...")
        
        # Enviar el dataset una sola vez antes de iniciar las épocas
        for i, conn in enumerate(conexiones):
            mensaje_init = {
                'tipo': 'INIT',
                'X': self.partes[i][0],
                'y': self.partes[i][1]
            }
            enviar_objeto(conn, mensaje_init)
            
        print("Distribucion inicial del dataset exitosa. Iniciando entrenamiento...")
        print("-" * 50)
        
        # Entrenamiento epoca por epoca
        historial_perdida = []
        
        for epoca in range(EPOCAS):
            # PASO 1: Enviar parametros a TODOS los workers primero
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
                # select() espera hasta que uno o mas sockets tengan datos listos
                # Retorna tres listas: sockets legibles, escribibles, con errores
                legibles, _, excepciones = select.select(
                    sockets_pendientes,  # sockets a monitorear para lectura
                    [],                   # sockets a monitorear para escritura (no necesitamos)
                    sockets_pendientes,   # sockets a monitorear para errores
                    SELECT_TIMEOUT        # timeout: None = esperar indefinidamente
                )
                
                # Manejar sockets con errores
                for sock in excepciones:
                    print(f"[Server] Error en socket, removiendo...")
                    if sock in sockets_pendientes:
                        sockets_pendientes.remove(sock)
                
                # Manejar sockets con datos listos para leer
                for sock in legibles:
                    try:
                        respuesta = recibir_objeto(sock)
                        if respuesta and respuesta['tipo'] == 'RESULT':
                            resultados.append({
                                'worker_id': conexiones.index(sock),
                                'params': respuesta['params'],
                                'perdida': respuesta['perdida']
                            })
                            sockets_pendientes.remove(sock)
                        elif respuesta is None:
                            # Conexion cerrada
                            print(f"[Server] Worker desconectado inesperadamente")
                            sockets_pendientes.remove(sock)
                    except Exception as e:
                        print(f"[Server] Error recibiendo de worker: {e}")
                        sockets_pendientes.remove(sock)
            
            # PASO 3: Promediar pesos de esta epoca
            if len(resultados) == N_WORKERS:
                params_parciales = [r['params'] for r in resultados]
                self.params = promediar_pesos(params_parciales)
                perdida_promedio = sum(r['perdida'] for r in resultados) / N_WORKERS
                historial_perdida.append(perdida_promedio)
                
                if epoca % 10 == 0:
                    print(f"\n[Server] Epoca {epoca}: Perdida Promedio = {perdida_promedio:.4f}")
                    print("-" * 50)
            else:
                print(f"[Server] Advertencia: Solo se recibieron {len(resultados)} de {N_WORKERS} resultados")
        
        # Cerrar conexiones enviando señal DONE
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
    print("Creando servidor de entrenamiento...")
    
    server = TrainingServer()
    
    print("Iniciando entrenamiento distribuido...")
    historial = server.entrenar()
    
    print("\n" + "=" * 50)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 50)
    
    # Evaluacion final
    precision_train = calcular_precision(server.X_train, server.y_train, server.params)
    precision_test = calcular_precision(server.X_test, server.y_test, server.params)
    
    print(f"Precision en entrenamiento: {precision_train:.4f} ({precision_train*100:.2f}%)")
    print(f"Precision en prueba: {precision_test:.4f} ({precision_test*100:.2f}%)")
    print(f"Perdida final: {historial[-1]:.4f}")

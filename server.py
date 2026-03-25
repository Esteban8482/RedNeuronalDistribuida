import socket
import pickle
import csv
import os
import time
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

N_WORKERS = 1

EPOCHS_GRID = [50] 

REPETITIONS = 3
LEARNING_RATE = 0.001 
BATCH_SIZE = 128

HOST = '0.0.0.0'
PORT = 5000
BUFFER_SIZE = 4096 * 1024
RESULTS_DIR = 'resultados'

DEVICE = torch.device('cpu')

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ──────────────────────────────────────────────────────────────
#  SHARED COMPONENTS (duplicated in worker.py)
# ──────────────────────────────────────────────────────────────
def log(msg):
    """Timestamped logging."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class CIFAR10CNN(nn.Module):
    """CNN architecture for CIFAR-10 classification.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def send_object(sock, obj):
    """Send a pickled object over socket with 8-byte size header."""
    data = pickle.dumps(obj)
    sock.sendall(len(data).to_bytes(8, 'big'))
    sock.sendall(data)


def receive_object(sock):
    """Receive a pickled object from socket with 8-byte size header."""
    header = b''
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    size = int.from_bytes(header, 'big')
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(BUFFER_SIZE, size - len(data)))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


# ──────────────────────────────────────────────────────────────
#  FEDERATED LEARNING OPERATIONS
# ──────────────────────────────────────────────────────────────
def average_state_dicts(state_dicts):
    """FedAvg: compute element-wise average of model state dicts."""
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    return avg


def evaluate_model(model, loader):
    """Evaluate model accuracy on test set (no gradients)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
#  CSV LOGGING
# ──────────────────────────────────────────────────────────────
DETAIL_HEADER = [
    'n_workers', 'n_epochs_config', 'repetition', 'epoch',
    'loss', 'test_accuracy', 'epoch_time_sec',
]
SUMMARY_HEADER = [
    'n_workers', 'n_epochs_config', 'repetition',
    'final_test_accuracy', 'final_loss',
    'total_time_sec', 'avg_epoch_time_sec',
]


def init_csv(path, header):
    """Initialize CSV file with header row."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(header)
    log(f"CSV created: {path}")


def save_detail_row(path, n_epochs, rep, epoch, loss, acc, t):
    """Append detailed epoch-level result to CSV."""
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, n_epochs, rep, epoch,
            round(loss, 6), round(acc, 6), round(t, 4),
        ])


def save_summary_row(path, n_epochs, rep, history):
    """Append experiment summary row to CSV."""
    total_time = sum(e['t'] for e in history)
    avg_time = total_time / len(history)
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            N_WORKERS, n_epochs, rep,
            round(history[-1]['acc'], 6),
            round(history[-1]['loss'], 6),
            round(total_time, 2),
            round(avg_time, 4),
        ])

# ──────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def run_training(connections, n_epochs, rep, model, test_loader, detail_path):
    """Execute federated training for specified number of epochs."""
    history = []

    for epoch in range(n_epochs):
        t0 = time.perf_counter()

        # Extract current weights (CPU, pickle-serializable)
        current_sd = {k: v.cpu() for k, v in model.state_dict().items()}

        # Broadcast to all workers
        for conn in connections:
            send_object(conn['sock'], {
                'type': 'TRAIN',
                'state_dict': current_sd,
                'lr': LEARNING_RATE,
                'epoch': epoch,
            })

        # Collect results from workers
        results = []
        for conn in connections:
            r = receive_object(conn['sock'])
            if r and r.get('type') == 'RESULT':
                results.append(r)

        epoch_time = time.perf_counter() - t0

        if results:
            avg_sd = average_state_dicts([r['state_dict'] for r in results])
            model.load_state_dict(avg_sd)
            loss = float(np.mean([r['loss'] for r in results]))
            acc = evaluate_model(model, test_loader)
        else:
            loss, acc = -1.0, -1.0

        history.append({'loss': loss, 'acc': acc, 't': epoch_time})
        save_detail_row(detail_path, n_epochs, rep, epoch, loss, acc, epoch_time)

        if epoch % 5 == 0 or epoch == n_epochs - 1:  # More frequent logging
            log(f"  [epochs={n_epochs:>3} rep={rep}] "
                f"{epoch+1:>3}/{n_epochs} | "
                f"loss={loss:.4f} | "
                f"test={acc*100:.1f}% | "
                f"t={epoch_time:.2f}s")

    return history

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log("Loading CIFAR-10...")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0,
    )

    train_raw = datasets.CIFAR10('data', train=True, download=True)
    X_train = train_raw.data
    y_train = np.array(train_raw.targets)

    # Reproducible shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    log(f"Dataset: {len(X_train)} train · {len(test_dataset)} test")

    log(f"Waiting for {N_WORKERS} workers on {HOST}:{PORT}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(N_WORKERS)

    connections = []
    for i in range(N_WORKERS):
        conn, addr = server.accept()
        hello = receive_object(conn)
        name = hello.get('name', str(addr)) if hello else str(addr)
        connections.append({'sock': conn, 'name': name})
        log(f"  Worker {i+1}/{N_WORKERS}: {name} ({addr})")
    server.close()

    shard_size = len(X_train) // N_WORKERS
    for i, conn in enumerate(connections):
        start = i * shard_size
        end = start + shard_size if i < N_WORKERS - 1 else len(X_train)
        send_object(conn['sock'], {
            'type': 'INIT',
            'X': X_train[start:end],
            'y': y_train[start:end],
            'worker_idx': i,
            'batch_size': BATCH_SIZE,
            'mean': CIFAR10_MEAN,
            'std': CIFAR10_STD,
        })
        log(f"  Shard {i}: samples {start}-{end-1} → {conn['name']}")
    log("Data distributed. Starting experiment grid.\n")

    detail_path = os.path.join(RESULTS_DIR, f'detail_{N_WORKERS}workers.csv')
    summary_path = os.path.join(RESULTS_DIR, f'summary_{N_WORKERS}workers.csv')
    init_csv(detail_path, DETAIL_HEADER)
    init_csv(summary_path, SUMMARY_HEADER)

    total_configs = len(EPOCHS_GRID)
    for cfg_idx, n_epochs in enumerate(EPOCHS_GRID, start=1):
        log("=" * 60)
        log(f"CONFIGURATION {cfg_idx}/{total_configs}: {n_epochs} epochs "
            f"({REPETITIONS} repetitions)")
        log("=" * 60)

        for rep in range(1, REPETITIONS + 1):
            log(f"  ── Repetition {rep}/{REPETITIONS} ──")
            t_rep = time.perf_counter()

            model = CIFAR10CNN().to(DEVICE)

            history = run_training(
                connections, n_epochs, rep,
                model, test_loader, detail_path,
            )
            save_summary_row(summary_path, n_epochs, rep, history)

            log(f"  Rep {rep} completed in {time.perf_counter()-t_rep:.1f}s | "
                f"final acc={history[-1]['acc']*100:.2f}%\n")

    for conn in connections:
        try:
            send_object(conn['sock'], {'type': 'EXPERIMENT_END'})
            conn['sock'].close()
        except Exception:
            pass

    log("=" * 60)
    log("Experiment grid completed.")
    log(f"  Details → {detail_path}")
    log(f"  Summary → {summary_path}")
    log("=" * 60)
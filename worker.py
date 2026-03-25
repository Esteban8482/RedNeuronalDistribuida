import socket
import pickle
import time
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='127.0.0.1',
                    help='Server IP address')
parser.add_argument('--core', type=int, default=-1,
                    help='Physical CPU core to bind (-1 = no affinity)')
args = parser.parse_args()

HOST = args.host
PORT = 5000
BUFFER_SIZE = 4096 * 1024

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ──────────────────────────────────────────────────────────────
#  CPU AFFINITY
# ──────────────────────────────────────────────────────────────
if args.core >= 0:
    try:
        os.sched_setaffinity(0, {args.core})
        print(f"[AFFINITY] Process bound to core {args.core}", flush=True)
    except AttributeError:
        print("[AFFINITY] os.sched_setaffinity not available on this OS", flush=True)
    except OSError as e:
        print(f"[AFFINITY] Could not bind to core {args.core}: {e}", flush=True)


# ──────────────────────────────────────────────────────────────
#  SHARED COMPONENTS (duplicated in server.py)
# ──────────────────────────────────────────────────────────────
def log(msg):
    """Timestamped logging."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class CIFAR10CNN(nn.Module):
    """CNN architecture for CIFAR-10 classification."""
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
#  DATASET WITH AUGMENTATION
# ──────────────────────────────────────────────────────────────
class CIFARShard(Dataset):
    """Local worker shard with training-time augmentation."""

    def __init__(self, X_np: np.ndarray, y_np: np.ndarray,
                 mean: tuple = CIFAR10_MEAN, std: tuple = CIFAR10_STD):
        self.X = (
            torch.from_numpy(X_np)
            .permute(0, 3, 1, 2)
            .float()
            .div(255.0)
        )
        self.y = torch.from_numpy(y_np).long()

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    lr: float, device: torch.device) -> float:
    """Train for one epoch with Adam + CrossEntropyLoss.
    
    Returns average loss over all mini-batches.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

if __name__ == '__main__':
    device = torch.device('cpu')
    hostname = socket.gethostname()
    log(f"Worker starting: {hostname}")

    # Connect to server with retries
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    for attempt in range(30):
        try:
            sock.connect((HOST, PORT))
            connected = True
            log(f"Connected to server {HOST}:{PORT}")
            break
        except Exception as e:
            log(f"Attempt {attempt+1}/30 failed: {e}")
            time.sleep(2)

    if not connected:
        log("Could not connect. Exiting.")
        raise SystemExit(1)

    # Send identification
    send_object(sock, {'type': 'HELLO', 'name': hostname})

    # Local state
    model: nn.Module = CIFAR10CNN().to(device)
    loader: DataLoader | None = None

    # Main loop
    while True:
        msg = receive_object(sock)

        if msg is None:
            log("Connection closed by server.")
            break

        msg_type = msg.get('type', 'UNKNOWN')

        if msg_type == 'INIT':
            # Receive data shard
            X_np = msg['X']
            y_np = msg['y']
            batch_size = msg.get('batch_size', 128)
            mean = msg.get('mean', CIFAR10_MEAN)
            std = msg.get('std', CIFAR10_STD)

            dataset = CIFARShard(X_np, y_np, mean, std)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            log(f"Shard received: {len(dataset)} samples | "
                f"batch={batch_size} | {len(loader)} steps/epoch")

        elif msg_type == 'TRAIN':
            if loader is None:
                log("ERROR: 'TRAIN' received before 'INIT'. Ignoring.")
                continue

            # Load server weights
            model.load_state_dict(msg['state_dict'])

            t0 = time.perf_counter()
            loss = train_one_epoch(model, loader, msg['lr'], device)
            t = round(time.perf_counter() - t0, 4)

            log(f"Epoch {msg.get('epoch', '?'):>3} | loss={loss:.4f} | t={t}s")

            # Return updated weights to server
            send_object(sock, {
                'type': 'RESULT',
                'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'loss': loss,
                'time': t,
            })

        elif msg_type == 'EXPERIMENT_END':
            log("Experiment ended by server.")
            break

        else:
            log(f"Unknown message type: '{msg_type}' — ignoring.")

    sock.close()
    log("Worker finished.")
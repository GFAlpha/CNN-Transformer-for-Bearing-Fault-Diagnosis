import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# 参数
# =========================
DATA_DIR = "data/splits"
RESULT_DIR = "results/cnn_bilstm"

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
RUNS = 5
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CNN + BiLSTM 模型
# =========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # x: [B, 1024] or [B, 1, 1024]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.cnn(x)            # [B, 32, T]
        x = x.permute(0, 2, 1)     # [B, T, 32]

        out, _ = self.lstm(x)      # [B, T, 128]
        out = out[:, -1, :]        # last timestep

        return self.fc(out)

# =========================
# 训练与评估函数
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            labels.append(y)

    return accuracy_score(
        torch.cat(labels),
        torch.cat(preds)
    )

def evaluate_with_preds(model, loader):
    """用于保存 y_true / y_pred"""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.numpy())

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)

    return y_true, y_pred

def measure_inference_time(model, loader, repeat=20):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        for _ in range(5):  # warm-up
            _ = model(x)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(repeat):
            _ = model(x)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    return (end - start) / repeat

# =========================
# 主流程
# =========================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ===== 加载数据 =====
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    test_accs = []
    train_times = []
    infer_times = []

    final_y_true, final_y_pred = None, None

    for run in range(RUNS):
        print(f"\n=== Run {run + 1}/{RUNS} ===")

        model = CNN_BiLSTM(NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train).float(),
                          torch.tensor(y_train).long()),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val).float(),
                          torch.tensor(y_val).long()),
            batch_size=BATCH_SIZE
        )

        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test).float(),
                          torch.tensor(y_test).long()),
            batch_size=BATCH_SIZE
        )

        # ===== 训练 =====
        best_val = 0.0
        best_state = None

        start_time = time.time()
        for _ in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate(model, val_loader)
            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        train_time = time.time() - start_time
        train_times.append(train_time)

        # ===== 测试 =====
        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)

        infer_time = measure_inference_time(model, test_loader)
        infer_times.append(infer_time)

        # ===== 仅最后一轮保存 y_true / y_pred =====
        if run == RUNS - 1:
            final_y_true, final_y_pred = evaluate_with_preds(model, test_loader)

        print(f"Test Acc: {test_acc:.4f}")
        print(f"Train Time: {train_time:.2f}s | Inference Time: {infer_time:.4f}s")

    # ===== 保存结果 =====
    np.save(f"{RESULT_DIR}/test_accs.npy", np.array(test_accs))
    np.save(f"{RESULT_DIR}/train_times.npy", np.array(train_times))
    np.save(f"{RESULT_DIR}/infer_times.npy", np.array(infer_times))
    np.save(f"{RESULT_DIR}/y_true.npy", final_y_true)
    np.save(f"{RESULT_DIR}/y_pred.npy", final_y_pred)

    meta = {
        "model": "CNN+BiLSTM",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "runs": RUNS,
        "num_classes": NUM_CLASSES
    }
    np.save(f"{RESULT_DIR}/meta.npy", meta)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {np.mean(test_accs):.4f}")
    print(f"Std  Acc: {np.std(test_accs):.4f}")
    print(f"Avg Train Time: {np.mean(train_times):.2f}s")
    print(f"Avg Inference Time: {np.mean(infer_times):.4f}s")

if __name__ == "__main__":
    main()

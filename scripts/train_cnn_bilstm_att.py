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
RESULT_DIR = "results/cnn_bilstm_att"

EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
RUNS = 5
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Attention 机制
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [B, T, 2H]
        scores = self.attn(x)              # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


# =========================
# CNN + BiLSTM + Attention 模型
# =========================
class CNN_BiLSTM_Att(nn.Module):
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

        self.att = TemporalAttention(64)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)        # [B, 1, L]

        x = self.cnn(x)               # [B, 32, T]
        x = x.permute(0, 2, 1)        # [B, T, 32]

        x, _ = self.lstm(x)           # [B, T, 128]
        x = self.att(x)               # [B, 128]

        return self.fc(x)


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
    return accuracy_score(torch.cat(labels), torch.cat(preds))


def inference_time(model, loader):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            _ = model(x)
    total_time = time.time() - start
    return total_time / len(loader.dataset)


def collect_predictions(model, loader):
    """用于混淆矩阵的 y_true / y_pred"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


# =========================
# 主函数
# =========================
def main():
    # 加载数据
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_test.npy")

    test_accs = []
    train_times = []
    infer_times = []

    final_y_true = None
    final_y_pred = None

    for run in range(RUNS):
        print(f"\n=== Run {run + 1}/{RUNS} ===")

        model = CNN_BiLSTM_Att(NUM_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train).float(),
                          torch.tensor(y_train).long()),
            batch_size=BATCH_SIZE, shuffle=True
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

        # -------- 训练 --------
        start_train = time.time()
        best_val = 0.0
        best_state = None

        for _ in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)
            val_acc = evaluate(model, val_loader)
            if val_acc > best_val:
                best_val = val_acc
                best_state = model.state_dict()

        train_times.append(time.time() - start_train)

        model.load_state_dict(best_state)

        # -------- 测试 --------
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)

        # -------- 推理时间 --------
        infer_times.append(inference_time(model, test_loader))

        # 👉 只保存最后一次 run 的预测（固定 Test 集）
        if run == RUNS - 1:
            final_y_true, final_y_pred = collect_predictions(model, test_loader)

        print(f"Test Acc: {test_acc:.4f}")

    # =========================
    # 保存结果
    # =========================
    np.save(f"{RESULT_DIR}/test_accs.npy", np.array(test_accs))
    np.save(f"{RESULT_DIR}/train_times.npy", np.array(train_times))
    np.save(f"{RESULT_DIR}/inference_times.npy", np.array(infer_times))
    np.save(f"{RESULT_DIR}/y_true.npy", final_y_true)
    np.save(f"{RESULT_DIR}/y_pred.npy", final_y_pred)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {np.mean(test_accs):.4f}")
    print(f"Std Acc : {np.std(test_accs):.4f}")
    print(f"Avg Train Time: {np.mean(train_times):.2f}s")
    print(f"Avg Inference Time: {np.mean(infer_times):.6f}s")


if __name__ == "__main__":
    main()

# 这其实是v6版本，但是懒得重命名或者新建文件夹了（v2版本的代码应该在项目根文件里，是.txt格式的）
# v1版本就是项目里的train_cnn.py
# v2版本是每次训练都随机划分数据集
# v3版本是固定划分数据集，多次训练取平均
# v4版本是将v3版本稍作修改，实现保存结果供后续画图
# v5版本是将v4版本稍作修改，实现运行后记录用时等指标
# v6版本是将v5版本稍作修改，实现只保存第一次run的y_true和y_pred，减少存储空间
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# =========================
# 参数
# =========================
NUM_RUNS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULT_DIR = "results/cnn"
MODEL_DIR = "models/cnn"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 固定随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 数据集
# =========================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # CNN 输入：[B, 1, L]
        return self.X[idx].unsqueeze(0), self.y[idx]

# =========================
# CNN 模型
# =========================
class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# =========================
# 单次训练 + 测试
# =========================
def train_and_test(run_id, train_loader, val_loader, test_loader):
    print(f"\n===== Run {run_id + 1}/{NUM_RUNS} =====")

    model = CNN1D(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = f"{MODEL_DIR}/best_model_run{run_id+1}.pth"

    # ---------- 训练计时 ----------
    train_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # ---------- 验证 ----------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    train_time = time.time() - train_start

    # ---------- 测试 + 推理计时 ----------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds, test_labels = [], []
    infer_start = time.time()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    infer_time = (time.time() - infer_start) / len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test Acc (Run {run_id+1}): {test_acc:.4f}")

    return test_acc, train_time, infer_time, test_labels, test_preds

# =========================
# 主函数
# =========================
def main():
    # 加载数据
    X_train = np.load("data/splits/X_train.npy")
    y_train = np.load("data/splits/y_train.npy")
    X_val = np.load("data/splits/X_val.npy")
    y_val = np.load("data/splits/y_val.npy")
    X_test = np.load("data/splits/X_test.npy")
    y_test = np.load("data/splits/y_test.npy")

    train_loader = DataLoader(BearingDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BearingDataset(X_val, y_val), BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(BearingDataset(X_test, y_test), BATCH_SIZE, shuffle=False)

    test_accs, train_times, infer_times = [], [], []

    y_true_saved, y_pred_saved = None, None

    for run in range(NUM_RUNS):
        set_seed(3000 + run)
        acc, t_time, i_time, y_true, y_pred = train_and_test(
            run, train_loader, val_loader, test_loader
        )
        test_accs.append(acc)
        train_times.append(t_time)
        infer_times.append(i_time)

        # ✅ 只保存第 1 次 run 的 y_true / y_pred（用于混淆矩阵）
        if run == 0:
            y_true_saved = np.array(y_true)
            y_pred_saved = np.array(y_pred)

    test_accs = np.array(test_accs)
    train_times = np.array(train_times)
    infer_times = np.array(infer_times)

    print("\n===== Final Statistics =====")
    print("Test Accuracies:", test_accs)
    print(f"Mean Acc: {test_accs.mean():.4f}")
    print(f"Std Acc : {test_accs.std():.4f}")
    print(f"Avg Train Time: {train_times.mean():.2f}s")
    print(f"Avg Inference Time: {infer_times.mean():.4f}s")

    # ---------- 保存 ----------
    np.save(f"{RESULT_DIR}/test_accs.npy", test_accs)
    np.save(f"{RESULT_DIR}/train_times.npy", train_times)
    np.save(f"{RESULT_DIR}/infer_times.npy", infer_times)
    np.save(f"{RESULT_DIR}/y_true.npy", y_true_saved)
    np.save(f"{RESULT_DIR}/y_pred.npy", y_pred_saved)

if __name__ == "__main__":
    main()

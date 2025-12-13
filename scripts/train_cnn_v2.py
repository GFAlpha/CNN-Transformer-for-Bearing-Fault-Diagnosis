import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================
# 参数区
# =====================
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 数据集类
# =====================
class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx]


# =====================
# CNN 模型
# =====================
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
            nn.Linear(32 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =====================
# 主流程
# =====================
def main():
    print(f"使用设备: {DEVICE}")

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    # 先分出 Test（10%）
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 再从剩余 90% 里分 Train / Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2222, random_state=42, stratify=y_tmp
        # 0.2222 × 90% ≈ 20%
    )

    train_loader = DataLoader(BearingDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BearingDataset(X_val, y_val),
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(BearingDataset(X_test, y_test),
                             batch_size=BATCH_SIZE)

    model = CNN1D(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    # ========= 训练阶段 =========
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # ========= 验证阶段 =========
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Val Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/cnn1d_best.pth")
            print("  🔥 保存新的最优模型")

    print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")

    # ========= 测试阶段（只做一次） =========
    print("\n开始测试集评估...")
    model.load_state_dict(torch.load("models/cnn1d_best.pth"))
    model.eval()

    test_preds, test_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"✅ 测试集准确率: {test_acc:.4f}")


if __name__ == "__main__":
    main()

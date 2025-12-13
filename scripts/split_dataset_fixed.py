import os
import numpy as np
from sklearn.model_selection import train_test_split

# =====================
# 固定随机种子（非常重要）
# =====================
SEED = 42

# =====================
# 读取完整数据
# =====================
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

os.makedirs("data/splits", exist_ok=True)

# =====================
# 1️⃣ 先固定 Test（10%）
# =====================
X_remain, X_test, y_remain, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=SEED,
    stratify=y
)

# =====================
# 2️⃣ 再分 Train / Val（7 : 2）
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X_remain, y_remain,
    test_size=2/9,   # 在剩余 90% 里切 20%
    random_state=SEED,
    stratify=y_remain
)

# =====================
# 保存（物理固定）
# =====================
np.save("data/splits/X_train.npy", X_train)
np.save("data/splits/y_train.npy", y_train)

np.save("data/splits/X_val.npy", X_val)
np.save("data/splits/y_val.npy", y_val)

np.save("data/splits/X_test.npy", X_test)
np.save("data/splits/y_test.npy", y_test)

print("✅ 数据集已物理固定完成")
print(f"Train: {len(X_train)}")
print(f"Val  : {len(X_val)}")
print(f"Test : {len(X_test)}")

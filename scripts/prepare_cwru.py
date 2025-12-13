import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# =====================
# 参数区
# 原始数据路径和保存路径
# 滑窗大小和步长
# =====================
RAW_DATA_DIR = "data/raw/CWRU_12K_DE"
SAVE_DIR = "data/processed"

WINDOW_SIZE = 1024
STEP_SIZE = 1024

LABEL_MAP = {
    "Ball": 1,
    "Inner Race": 2,
    "Outer Race": 3,
}

NORMAL_LABEL = 0

# =====================
# 工具函数
# =====================
def extract_de_signal(mat_path):
    """
    从 CWRU 的 mat 文件中提取 Drive End 信号
    """
    mat = loadmat(mat_path)
    for key in mat.keys():
        if "DE_time" in key:
            return mat[key].squeeze()
    raise ValueError(f"DE signal not found in {mat_path}")


def slice_signal(signal, window_size, step_size):
    """
    滑窗切片
    """
    slices = []
    for start in range(0, len(signal) - window_size, step_size):
        slices.append(signal[start:start + window_size])
    return slices


# =====================
# 主流程
# =====================
X, y = [], []

print("=== 开始处理 CWRU 数据集 ===")

for root, dirs, files in os.walk(RAW_DATA_DIR):
    for file in files:
        if not file.endswith(".mat"):
            continue

        mat_path = os.path.join(root, file)

        # 判断类别
        label = None
        for k in LABEL_MAP:
            if k in mat_path:
                label = LABEL_MAP[k]
                break

        if label is None:
            label = NORMAL_LABEL

        # 读取信号
        try:
            signal = extract_de_signal(mat_path)
        except Exception as e:
            print(f"跳过文件 {file}: {e}")
            continue

        segments = slice_signal(signal, WINDOW_SIZE, STEP_SIZE)

        for seg in segments:
            X.append(seg)
            y.append(label)

print(f"切片完成，共 {len(X)} 个样本")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

print("数据已保存至 data/processed/")

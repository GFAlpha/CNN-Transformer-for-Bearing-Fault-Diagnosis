import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

RAW_DIR = "data/raw/CWRU_12K_DE"
SAVE_DIR = "data/processed"

# 信号长度（可以调整）
SIGNAL_LENGTH = 2048


def load_mat_file(path):
    """加载 .mat 文件并提取 DE 时间序列数据"""
    data = loadmat(path)
    # mat 文件中关键字段名称可能不同，这里做通用兼容
    for key in data.keys():
        if "DE" in key:        # 例如：DE_time
            return data[key].squeeze()
    raise ValueError(f"未找到 DE 通道: {path}")


def pad_or_trim(signal, length=SIGNAL_LENGTH):
    """统一长度：过长截断，过短补零"""
    if len(signal) > length:
        return signal[:length]
    elif len(signal) < length:
        padded = np.zeros(length)
        padded[:len(signal)] = signal
        return padded
    return signal


def label_fault(folder_name, subfolder):
    """
    folder_name: Ball / Inner Race / Outer Race
    subfolder  : 0007 / 0014 / ...
    返回整数标签 y
    """

    fault_type_map = {
        "Normal": 0,
        "Ball": 1,
        "Inner Race": 2,
        "Outer Race": 3,
    }

    fault_level = int(subfolder)   # 0007 → 7

    base_label = fault_type_map.get(folder_name, -1)

    # Outer Race 需要加上方向编码
    # Centered = 0, Orthogonal = 1, Opposite = 2
    def outer_offset(name):
        if name == "Centered":
            return 0
        if name == "Orthogonal":
            return 1
        if name == "Opposite":
            return 2
        return 0

    if folder_name == "Outer Race":
        return base_label * 10 + outer_offset(subfolder)

    return base_label


def main():
    X_all = []
    y_all = []

    print("\n=== 开始处理 CWRU 数据集 ===")

    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if not file.endswith(".mat"):
                continue

            file_path = os.path.join(root, file)

            # 解析路径，例如：
            # Ball/0007/B007_0.mat
            parts = file_path.replace("\\", "/").split("/")

            # parts[-3] = Ball / Inner Race / Outer Race
            # parts[-2] = 0007 / 0014 / 0021 / 0028
            folder_name = parts[-3]
            subfolder = parts[-2]

            # 加载信号
            try:
                sig = load_mat_file(file_path)
            except Exception as e:
                print(f"跳过文件 {file_path}: {e}")
                continue

            sig = pad_or_trim(sig)

            label = label_fault(folder_name, subfolder)

            X_all.append(sig)
            y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    os.makedirs(SAVE_DIR, exist_ok=True)

    np.save(os.path.join(SAVE_DIR, "X.npy"), X_all)
    np.save(os.path.join(SAVE_DIR, "y.npy"), y_all)

    print(f"\n处理完成！共加载 {len(X_all)} 个样本")
    print(f"保存到：{SAVE_DIR}/X.npy 和 y.npy\n")


if __name__ == "__main__":
    main()

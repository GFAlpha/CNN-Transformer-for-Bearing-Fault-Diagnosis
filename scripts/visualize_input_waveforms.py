# scripts/visualize_input_waveforms.py
# -*- coding: utf-8 -*-
"""
输入数据可视化脚本，用于软硬件验收展示。

功能：
1. 从 data/raw 中读取 CWRU 原始 .mat 文件，绘制四类原始时域波形；
2. 从 data/processed/X.npy 和 y.npy 中读取模型实际训练输入，绘制四类 1024 点切片波形；
3. 输出图片到 analysis_results/input_visualization/。

数据流：
data/raw 原始 CWRU .mat 文件
        ↓ prepare_cwru.py
data/processed/X.npy, y.npy
        ↓ 训练脚本
模型训练与测试
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# =========================
# 1. 基础路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results" / "input_visualization"

RAW_SHOW_LENGTH = 4096
WINDOW_LENGTH = 1024
FS = 12000


# =========================
# 2. 类别配置
# 注意：
# 这里 class_id 必须和 y.npy 中的标签保持一致
# =========================
CLASS_INFO = {
    0: {
        "name_cn": "正常状态",
        "name_en": "Normal",
        "raw_subdir": ["CWRU_Normal"],
    },
    1: {
        "name_cn": "滚动体故障",
        "name_en": "Ball Fault",
        "raw_subdir": ["CWRU_12K_DE", "Ball"],
    },
    2: {
        "name_cn": "内圈故障",
        "name_en": "Inner Race Fault",
        "raw_subdir": ["CWRU_12K_DE", "Inner Race"],
    },
    3: {
        "name_cn": "外圈故障",
        "name_en": "Outer Race Fault",
        "raw_subdir": ["CWRU_12K_DE", "Outer Race"],
    },
}


def set_chinese_font():
    """设置中文字体，避免图片标题乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_npy(path):
    return np.load(path, allow_pickle=True)


def find_signal_from_mat(mat_dict, min_len=2048):
    """
    从 .mat 文件中自动寻找一条有效的一维振动信号。

    CWRU 的 mat 文件中通常包含类似：
    - X097_DE_time
    - X105_DE_time
    这样的变量。
    """
    candidate_keys = []

    for key in mat_dict.keys():
        if key.startswith("__"):
            continue

        lower_key = key.lower()

        # 优先找驱动端 DE 信号
        if "de_time" in lower_key:
            candidate_keys.insert(0, key)
        elif "time" in lower_key:
            candidate_keys.append(key)

    # 先按 key 名查找
    for key in candidate_keys:
        value = np.squeeze(mat_dict[key])
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size >= min_len:
            return value.astype(np.float32), key

    # 如果 key 名没有匹配成功，就兜底找任意足够长的一维数组
    for key, value in mat_dict.items():
        if key.startswith("__"):
            continue

        value = np.squeeze(value)
        if isinstance(value, np.ndarray) and value.ndim == 1 and value.size >= min_len:
            return value.astype(np.float32), key

    return None, None


def find_raw_mat_file_by_class(class_id):
    """按类别在 data/raw 中寻找原始 .mat 文件。"""
    cfg = CLASS_INFO[class_id]

    search_dir = RAW_DATA_DIR
    for part in cfg["raw_subdir"]:
        search_dir = search_dir / part

    if not search_dir.exists():
        print(f"[警告] 原始数据目录不存在：{search_dir}")
        return None

    mat_files = sorted(list(search_dir.rglob("*.mat")))

    if not mat_files:
        print(f"[警告] 目录下未找到 .mat 文件：{search_dir}")
        return None

    return mat_files[0]


def find_processed_xy_files():
    """
    寻找模型训练输入 X.npy 和标签 y.npy。
    优先使用你的项目当前结构：
    data/processed/X.npy
    data/processed/y.npy
    """
    x_path = PROCESSED_DATA_DIR / "X.npy"
    y_path = PROCESSED_DATA_DIR / "y.npy"

    if x_path.exists() and y_path.exists():
        return x_path, y_path

    npy_files = list(PROCESSED_DATA_DIR.rglob("*.npy"))

    x_candidates = []
    y_candidates = []

    for file in npy_files:
        lower = file.name.lower()

        if lower in ["x.npy", "x_train.npy", "train_x.npy"]:
            x_candidates.append(file)

        if lower in ["y.npy", "y_train.npy", "train_y.npy", "label.npy", "labels.npy"]:
            y_candidates.append(file)

    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]

    print("[错误] 未找到处理后的 X/y 数据。")
    print(f"[检查目录] {PROCESSED_DATA_DIR}")
    print("[当前发现的 npy 文件]")
    for file in npy_files:
        print(" -", file)

    raise FileNotFoundError("未找到 X.npy 和 y.npy，请检查 data/processed 目录。")


def select_one_sample_per_class(X, y):
    """从 X.npy/y.npy 中每个类别选一个样本。"""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    selected = {}

    for class_id in CLASS_INFO.keys():
        idxs = np.where(y == class_id)[0]

        if len(idxs) == 0:
            print(f"[警告] y.npy 中没有找到类别 {class_id}：{CLASS_INFO[class_id]['name_cn']}")
            continue

        sample = np.asarray(X[idxs[0]]).reshape(-1)
        selected[class_id] = sample

    return selected


def plot_raw_waveforms():
    """绘制 CWRU 原始 .mat 信号的四类时域波形。"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)

    found_any = False

    for row, (class_id, cfg) in enumerate(CLASS_INFO.items()):
        ax = axes[row]

        mat_file = find_raw_mat_file_by_class(class_id)

        if mat_file is None:
            ax.text(
                0.5,
                0.5,
                f"未找到 {cfg['name_cn']} 原始 .mat 文件",
                ha="center",
                va="center",
            )
            ax.set_title(f"{cfg['name_cn']}（未找到原始文件）")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue

        mat_data = sio.loadmat(mat_file)
        signal, signal_key = find_signal_from_mat(mat_data)

        if signal is None:
            ax.text(
                0.5,
                0.5,
                f"{mat_file.name} 中未找到有效时域信号",
                ha="center",
                va="center",
            )
            ax.set_title(f"{cfg['name_cn']}（信号读取失败）")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue

        signal = signal[:RAW_SHOW_LENGTH]
        t = np.arange(len(signal)) / FS

        ax.plot(t, signal, linewidth=0.8)
        ax.set_title(
            f"{cfg['name_cn']}原始时域波形（{mat_file.name}，变量：{signal_key}，前 {len(signal)} 点）"
        )
        ax.set_ylabel("幅值")
        ax.grid(True, linestyle="--", alpha=0.4)

        found_any = True

    axes[-1].set_xlabel("时间 / s")
    fig.suptitle("CWRU 原始振动信号时域波形示例", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUT_DIR / "raw_cwru_waveforms_4classes.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    if found_any:
        print(f"[完成] 原始信号波形图已保存：{out_path}")
    else:
        print("[警告] 没有找到任何可用的原始 .mat 信号。")


def plot_train_window_waveforms():
    """绘制模型实际训练输入的 1024 点切片波形。"""
    x_file, y_file = find_processed_xy_files()

    X = load_npy(x_file)
    y = load_npy(y_file)

    print(f"[信息] 训练输入文件：{x_file}")
    print(f"[信息] 训练标签文件：{y_file}")
    print(f"[信息] X shape = {X.shape}")
    print(f"[信息] y shape = {y.shape}")

    selected = select_one_sample_per_class(X, y)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    for row, (class_id, cfg) in enumerate(CLASS_INFO.items()):
        ax = axes[row]

        if class_id not in selected:
            ax.text(
                0.5,
                0.5,
                f"未找到 {cfg['name_cn']} 训练样本",
                ha="center",
                va="center",
            )
            ax.set_title(f"{cfg['name_cn']}（未找到训练样本）")
            ax.grid(True, linestyle="--", alpha=0.4)
            continue

        signal = selected[class_id][:WINDOW_LENGTH]
        sample_points = np.arange(len(signal))

        ax.plot(sample_points, signal, linewidth=0.8)
        ax.set_title(f"{cfg['name_cn']}模型输入切片波形（窗口长度={len(signal)}）")
        ax.set_ylabel("幅值")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("采样点")
    fig.suptitle("模型实际训练输入样本时域波形示例", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUTPUT_DIR / "train_input_windows_4classes.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[完成] 训练输入切片波形图已保存：{out_path}")


def print_data_flow_summary():
    """打印验收时可说明的数据流。"""
    print("\n" + "=" * 70)
    print("数据流说明")
    print("=" * 70)
    print(f"原始数据目录：{RAW_DATA_DIR}")
    print(f"处理后数据目录：{PROCESSED_DATA_DIR}")
    print(f"可视化输出目录：{OUTPUT_DIR}")
    print()
    print("data/raw 原始 CWRU .mat 文件")
    print("        ↓ prepare_cwru.py 读取原始信号并按 1024 点窗口切片")
    print("data/processed/X.npy, y.npy")
    print("        ↓ 训练脚本读取 X/y")
    print("CNN、LSTM、Transformer、CNN+Transformer 等模型训练与测试")
    print("=" * 70)


def main():
    set_chinese_font()
    ensure_output_dir()

    print("=" * 70)
    print("输入数据可视化：原始 CWRU 信号 + 模型训练输入切片")
    print("=" * 70)

    print("\n[步骤1] 绘制原始 CWRU 四类时域波形")
    plot_raw_waveforms()

    print("\n[步骤2] 绘制模型实际训练输入 1024 点切片波形")
    plot_train_window_waveforms()

    print_data_flow_summary()

    print("\n全部完成。")


if __name__ == "__main__":
    main()
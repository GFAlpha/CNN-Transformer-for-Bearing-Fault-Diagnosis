# -*- coding: utf-8 -*-
"""
将原始时域切片数据转换为 FFT 幅值谱数据
用于补充“输入形式对比实验”：
1. Time-domain input：原始 1024 点时域信号
2. FFT input：1024 点信号经过 rFFT 后得到的幅值谱

输出位置：
data/splits_fft/
"""

import os
import numpy as np

SRC_DIR = "data/splits"
DST_DIR = "data/splits_fft"
os.makedirs(DST_DIR, exist_ok=True)


def time_to_fft_mag(X):
    """
    X: [N, 1024]
    返回: [N, 1024]
    说明：
    rFFT 对实数信号输出 513 个频率点；
    为了不改 CNN+Transformer 模型结构，这里补零到 1024 维。
    """
    X = X.astype(np.float32)

    # 去直流分量，避免 0Hz 分量过大影响频谱
    X = X - np.mean(X, axis=1, keepdims=True)

    # 加汉宁窗，减少频谱泄漏
    window = np.hanning(X.shape[1]).astype(np.float32)
    X_win = X * window

    # rFFT 幅值谱
    fft_mag = np.abs(np.fft.rfft(X_win, axis=1)).astype(np.float32)

    # log 压缩，避免幅值差异过大
    fft_mag = np.log1p(fft_mag)

    # 每个样本单独标准化
    mean = np.mean(fft_mag, axis=1, keepdims=True)
    std = np.std(fft_mag, axis=1, keepdims=True) + 1e-8
    fft_mag = (fft_mag - mean) / std

    # 补零到 1024，保证可以直接复用原 CNN+Transformer
    pad_len = X.shape[1] - fft_mag.shape[1]
    fft_mag = np.pad(fft_mag, ((0, 0), (0, pad_len)), mode="constant")

    return fft_mag.astype(np.float32)


def convert_one(name):
    X_path = os.path.join(SRC_DIR, f"X_{name}.npy")
    y_path = os.path.join(SRC_DIR, f"y_{name}.npy")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"[{name}] 原始 X shape: {X.shape}")
    X_fft = time_to_fft_mag(X)
    print(f"[{name}] FFT X shape: {X_fft.shape}")

    np.save(os.path.join(DST_DIR, f"X_{name}.npy"), X_fft)
    np.save(os.path.join(DST_DIR, f"y_{name}.npy"), y)


def main():
    for name in ["train", "val", "test"]:
        convert_one(name)

    print("\nFFT 输入数据生成完成！")
    print(f"保存位置：{DST_DIR}")


if __name__ == "__main__":
    main()
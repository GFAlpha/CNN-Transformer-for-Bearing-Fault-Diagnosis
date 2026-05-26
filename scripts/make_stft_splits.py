# -*- coding: utf-8 -*-
"""
生成 STFT 输入数据集
用于输入形式对比实验：
1. 时域原始信号
2. FFT幅值谱
3. STFT时频表示

说明：
为了快速复用现有 CNN+Transformer 模型，
这里将 STFT 时频矩阵展平后截断/补零到 1024 维。
"""

import os
import numpy as np

SRC_DIR = "data/splits"
DST_DIR = "data/splits_stft"

os.makedirs(DST_DIR, exist_ok=True)


def compute_stft_feature(
        x,
        n_fft=128,
        hop_length=64,
        target_len=1024
):
    """
    x: 单个 1024 点时域信号
    返回: 1024维 STFT 特征
    """

    x = x.astype(np.float32)

    # 去直流分量
    x = x - np.mean(x)

    # 汉宁窗
    window = np.hanning(n_fft).astype(np.float32)

    frames = []

    for start in range(0, len(x) - n_fft + 1, hop_length):
        frame = x[start:start + n_fft]
        frame = frame * window

        # rFFT 幅值谱
        spec = np.abs(np.fft.rfft(frame))

        # log压缩
        spec = np.log1p(spec)

        frames.append(spec)

    stft = np.stack(frames, axis=1)   # [freq_bins, time_frames]

    # 标准化
    stft = (stft - np.mean(stft)) / (np.std(stft) + 1e-8)

    # 展平为一维
    feature = stft.flatten()

    # 截断或补零到 1024
    if len(feature) >= target_len:
        feature = feature[:target_len]
    else:
        pad_len = target_len - len(feature)
        feature = np.pad(feature, (0, pad_len), mode="constant")

    return feature.astype(np.float32)


def convert_one(name):
    X_path = os.path.join(SRC_DIR, f"X_{name}.npy")
    y_path = os.path.join(SRC_DIR, f"y_{name}.npy")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"[{name}] 原始 X shape: {X.shape}")

    X_stft = []

    for i in range(len(X)):
        feature = compute_stft_feature(X[i])
        X_stft.append(feature)

        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(X)}")

    X_stft = np.array(X_stft, dtype=np.float32)

    print(f"[{name}] STFT X shape: {X_stft.shape}")

    np.save(os.path.join(DST_DIR, f"X_{name}.npy"), X_stft)
    np.save(os.path.join(DST_DIR, f"y_{name}.npy"), y)


def main():
    for name in ["train", "val", "test"]:
        convert_one(name)

    print("\nSTFT 输入数据生成完成！")
    print(f"保存位置：{DST_DIR}")


if __name__ == "__main__":
    main()
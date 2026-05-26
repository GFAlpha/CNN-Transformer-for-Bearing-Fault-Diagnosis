# -*- coding: utf-8 -*-
"""
时域信号与FFT频谱对比可视化
用于答辩PPT展示“信号处理思路”
"""

import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/splits"
OUT_DIR = "analysis_results/input_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_fft(x):
    x = x.astype(np.float32)
    x = x - np.mean(x)

    window = np.hanning(len(x))
    x_win = x * window

    fft_mag = np.abs(np.fft.rfft(x_win))
    fft_mag = np.log1p(fft_mag)

    return fft_mag


def main():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    classes = sorted(list(set(y_test.tolist())))

    for cls in classes:
        idx = np.where(y_test == cls)[0][0]
        x = X_test[idx]
        fft_mag = compute_fft(x)

        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(x)
        plt.title(f"Class {cls} - Time Domain Signal")
        plt.xlabel("Sampling Point")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(fft_mag)
        plt.title(f"Class {cls} - FFT Magnitude Spectrum")
        plt.xlabel("Frequency Bin")
        plt.ylabel("Log Magnitude")

        plt.tight_layout()

        save_path = os.path.join(
            OUT_DIR,
            f"class_{cls}_time_fft_compare.png"
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"已保存：{save_path}")

    print("\n时域-频域对比图生成完成！")


if __name__ == "__main__":
    main()
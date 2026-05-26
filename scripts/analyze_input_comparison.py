# -*- coding: utf-8 -*-
"""
输入形式对比实验结果汇总

对比：
1. 原始时域信号
2. FFT 幅值谱输入

输出：
results/input_comparison/input_comparison_summary.txt
"""

import os
import numpy as np

# ==========================
# 路径配置
# ==========================

RESULT_DIR = "results"

OUT_DIR = os.path.join(
    RESULT_DIR,
    "input_comparison"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ==========================
# 加载结果
# ==========================

def load_accs(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n找不到文件：\n{path}"
        )

    return np.load(path)


# ==========================
# 主程序
# ==========================

def main():

    print("=" * 60)
    print("开始分析输入形式对比实验")
    print("=" * 60)

    # 原始时域模型结果
    time_acc_path = os.path.join(
        RESULT_DIR,
        "cnn_transformer_test_accs.npy"
    )

# FFT结果
    fft_acc_path = os.path.join(
        RESULT_DIR,
        "cnn_transformer_fft",
        "test_accs.npy"
    ) 

    print("\n读取文件：")

    print("Time:")
    print(time_acc_path)

    print("\nFFT:")
    print(fft_acc_path)

    time_accs = load_accs(time_acc_path)
    fft_accs = load_accs(fft_acc_path)

    # ======================
    # 计算指标
    # ======================

    time_mean = np.mean(time_accs)
    time_std = np.std(time_accs)

    fft_mean = np.mean(fft_accs)
    fft_std = np.std(fft_accs)

    rows = [

        [
            "时域原始信号",
            "1024点原始切片",
            time_mean,
            time_std
        ],

        [
            "FFT幅值谱",
            "去均值+汉宁窗+rFFT+log幅值谱",
            fft_mean,
            fft_std
        ]

    ]

    # ======================
    # 保存txt
    # ======================

    out_txt = os.path.join(
        OUT_DIR,
        "input_comparison_summary.txt"
    )

    with open(
            out_txt,
            "w",
            encoding="utf-8"
    ) as f:

        f.write("输入形式对比实验结果\n")
        f.write("=" * 80 + "\n\n")

        header = "{:<18}{:<35}{:<12}{:<12}\n"

        f.write(
            header.format(
                "输入形式",
                "处理方式",
                "MeanAcc",
                "Std"
            )
        )

        f.write("-" * 80 + "\n")

        for row in rows:

            name = row[0]
            method = row[1]
            mean_acc = row[2]
            std_acc = row[3]

            line = "{:<18}{:<35}{:<12.4f}{:<12.4f}\n"

            f.write(
                line.format(
                    name,
                    method,
                    mean_acc,
                    std_acc
                )
            )

    # ======================
    # 屏幕输出
    # ======================

    print("\n")
    print("=" * 60)

    print("输入形式对比结果：")

    print("\n时域输入")
    print(f"Mean Acc: {time_mean:.4f}")
    print(f"Std      : {time_std:.4f}")

    print("\nFFT输入")
    print(f"Mean Acc: {fft_mean:.4f}")
    print(f"Std      : {fft_std:.4f}")

    print("=" * 60)

    print("\n结果已保存：")
    print(out_txt)


if __name__ == "__main__":
    main()
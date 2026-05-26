# -*- coding: utf-8 -*-
"""
输入形式对比实验结果汇总

对比：
1. 原始时域信号
2. FFT 幅值谱输入
3. STFT 时频特征输入

输出：
analysis_results/input_comparison/input_comparison_summary.txt
analysis_results/input_comparison/input_comparison_bar_en.png
analysis_results/input_comparison/input_comparison_bar_cn.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

RESULT_DIR = "results"
OUT_DIR = os.path.join("analysis_results", "input_comparison")
os.makedirs(OUT_DIR, exist_ok=True)


def setup_chinese_font():
    """
    尝试设置中文字体，避免中文显示成方块
    """
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Arial Unicode MS"
    ]

    available_fonts = set(
        f.name for f in font_manager.fontManager.ttflist
    )

    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"已使用中文字体：{font_name}")
            return True

    print("警告：未找到常见中文字体，将额外输出英文版图片。")
    return False


def load_accs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"\n找不到文件：\n{path}")
    return np.load(path)


def save_summary(rows):
    out_txt = os.path.join(
        OUT_DIR,
        "input_comparison_summary.txt"
    )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("输入形式对比实验结果\n")
        f.write("=" * 90 + "\n\n")
        f.write("{:<18}{:<38}{:<12}{:<12}\n".format(
            "输入形式",
            "处理方式",
            "MeanAcc",
            "Std"
        ))
        f.write("-" * 90 + "\n")

        for name, method, mean_acc, std_acc in rows:
            f.write("{:<18}{:<38}{:<12.4f}{:<12.4f}\n".format(
                name,
                method,
                mean_acc,
                std_acc
            ))

    return out_txt


def plot_bar(
        labels,
        means,
        stds,
        title,
        ylabel,
        out_path,
        ylim=(0.96, 1.005)
):
    """
    绘制输入形式对比柱状图
    """
    x = np.arange(len(labels))

    plt.figure(figsize=(8.5, 5.2))

    bars = plt.bar(
        x,
        means,
        width=0.55,
        yerr=stds,
        capsize=6
    )

    plt.xticks(x, labels, fontsize=11)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.ylim(*ylim)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("开始分析输入形式对比实验")
    print("=" * 60)

    setup_chinese_font()

    time_acc_path = os.path.join(
        RESULT_DIR,
        "cnn_transformer_test_accs.npy"
    )

    fft_acc_path = os.path.join(
        RESULT_DIR,
        "cnn_transformer_fft",
        "test_accs.npy"
    )

    stft_acc_path = os.path.join(
        RESULT_DIR,
        "cnn_transformer_stft",
        "test_accs.npy"
    )

    print("\n读取文件：")
    print("Time:", time_acc_path)
    print("FFT :", fft_acc_path)
    print("STFT:", stft_acc_path)

    time_accs = load_accs(time_acc_path)
    fft_accs = load_accs(fft_acc_path)
    stft_accs = load_accs(stft_acc_path)

    rows = [
        [
            "时域原始信号",
            "1024点原始切片",
            float(np.mean(time_accs)),
            float(np.std(time_accs)),
        ],
        [
            "FFT幅值谱",
            "去均值+汉宁窗+rFFT+log幅值谱",
            float(np.mean(fft_accs)),
            float(np.std(fft_accs)),
        ],
        [
            "STFT时频特征",
            "短时傅里叶变换+log幅值谱+展平",
            float(np.mean(stft_accs)),
            float(np.std(stft_accs)),
        ],
    ]

    out_txt = save_summary(rows)

    means = [row[2] for row in rows]
    stds = [row[3] for row in rows]

    # 英文版：推荐放PPT，最稳，不会乱码
    labels_en = [
        "Time-domain",
        "FFT spectrum",
        "STFT feature"
    ]

    out_png_en = os.path.join(
        OUT_DIR,
        "input_comparison_bar_en.png"
    )

    plot_bar(
        labels=labels_en,
        means=means,
        stds=stds,
        title="Comparison of Different Input Representations",
        ylabel="Mean Accuracy",
        out_path=out_png_en,
        ylim=(0.96, 1.005)
    )

    # 中文版：如果系统有中文字体，会正常显示
    labels_cn = [
        "时域信号",
        "FFT幅值谱",
        "STFT时频特征"
    ]

    out_png_cn = os.path.join(
        OUT_DIR,
        "input_comparison_bar_cn.png"
    )

    plot_bar(
        labels=labels_cn,
        means=means,
        stds=stds,
        title="不同输入信号表示方式对比",
        ylabel="平均准确率",
        out_path=out_png_cn,
        ylim=(0.96, 1.005)
    )

    print("\n" + "=" * 60)
    print("输入形式对比结果：")

    for name, method, mean_acc, std_acc in rows:
        print(f"\n{name}")
        print(f"处理方式 : {method}")
        print(f"Mean Acc: {mean_acc:.4f}")
        print(f"Std     : {std_acc:.4f}")

    print("=" * 60)

    print("\n结果已保存：")
    print(out_txt)
    print(out_png_en)
    print(out_png_cn)


if __name__ == "__main__":
    main()
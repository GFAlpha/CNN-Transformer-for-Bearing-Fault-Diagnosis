import os
import sys
import subprocess
import argparse
from datetime import datetime


# =========================================================
# 取 scripts 的上一级作为项目根目录
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(py_exe: str, script_path: str, title: str):
    """运行一个脚本步骤，失败则直接退出（防止后续产物基于错误结果继续跑）"""
    print("\n" + "=" * 80)
    print(f"[STEP] {title}")
    print(f"[RUN ] {py_exe} {script_path}")
    print("=" * 80)

    start = datetime.now()
    r = subprocess.run([py_exe, script_path], cwd=PROJECT_ROOT)
    end = datetime.now()

    if r.returncode != 0:
        print(f"\n[ERROR] 步骤失败：{title}")
        print("        请查看上方报错信息（通常是缺文件/路径不对/依赖缺失/未激活env）。")
        sys.exit(r.returncode)

    print(f"[OK]  步骤完成：{title}  (耗时 {(end - start)})")


def _fmt_path(p: str) -> str:
    """复制方便，统一输出成相对路径"""
    try:
        rel = os.path.relpath(p, PROJECT_ROOT)
        return rel.replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def _exists_mark(p: str) -> str:
    return "✅" if os.path.exists(p) else "⚠️"


def print_key_artifacts():
    """
    在最后输出最有用的关键产物清单
    说明：
      - ✅ 表示文件存在
      - ⚠️ 表示文件不存在（可能被 skip 了，或某步运行失败）
    """
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis_results")
    noise_dir = os.path.join(analysis_dir, "noise")
    cm_dir = os.path.join(analysis_dir, "confusion_matrices")
    dl_vs_ml_dir = os.path.join(analysis_dir, "dl_vs_ml")
    comp_dir = os.path.join(analysis_dir, "complexity")

    # 关键文件
    key_files = [
        # ---- 干净数据（全模型对比）----
        ("干净数据总表", os.path.join(analysis_dir, "summary.txt")),
        ("干净数据 CSV", os.path.join(analysis_dir, "summary.csv")),
        ("干净数据：准确率对比图", os.path.join(analysis_dir, "accuracy_comparison.png")),
        ("干净数据：训练时间对比图(log)", os.path.join(analysis_dir, "train_time_comparison_log.png")),
        ("干净数据：推理时间对比图(log)", os.path.join(analysis_dir, "infer_time_comparison_log.png")),

        # ---- 噪声鲁棒性（DL 全模型）----
        ("噪声鲁棒性总表", os.path.join(noise_dir, "noise_summary.txt")),
        ("噪声鲁棒性 CSV", os.path.join(noise_dir, "noise_summary_table.csv")),
        ("噪声鲁棒性曲线图", os.path.join(noise_dir, "acc_vs_snr.png")),

        # ---- DL vs ML----
        ("DL vs ML：干净对比（txt）", os.path.join(dl_vs_ml_dir, "clean_summary_dl_vs_ml.txt")),
        ("DL vs ML：噪声对比（txt）", os.path.join(dl_vs_ml_dir, "noise_summary_dl_vs_ml.txt")),
        ("DL vs ML：噪声曲线图（png）", os.path.join(dl_vs_ml_dir, "acc_vs_snr_dl_vs_ml.png")),

        # ---- 复杂度统计（参数量/FLOPs）----
        ("模型复杂度：参数量/FLOPs（txt）", os.path.join(comp_dir, "model_complexity.txt")),
        ("模型复杂度：参数量/FLOPs（csv）", os.path.join(comp_dir, "model_complexity.csv")),

        # ---- 混淆矩阵目录 ----
        ("混淆矩阵目录（所有模型）", cm_dir),
    ]

    # 混淆矩阵常看的几张
    cm_files = [
        ("混淆矩阵：CNN", os.path.join(cm_dir, "cm_cnn.png")),
        ("混淆矩阵：LSTM", os.path.join(cm_dir, "cm_rnn_lstm.png")),
        ("混淆矩阵：CNN+BiLSTM", os.path.join(cm_dir, "cm_cnn_bilstm.png")),
        ("混淆矩阵：CNN+BiLSTM+Att", os.path.join(cm_dir, "cm_cnn_bilstm_att.png")),
        ("混淆矩阵：Transformer", os.path.join(cm_dir, "cm_transformer.png")),
        ("混淆矩阵：CNN+Transformer", os.path.join(cm_dir, "cm_cnn_transformer.png")),
        ("混淆矩阵：CNN+Transformer(NoiseAug)", os.path.join(cm_dir, "cm_cnn_transformer_noiseaug.png")),
    ]

    print("\n" + "=" * 80)
    print("【关键产物清单】")
    print("=" * 80)

    for name, p in key_files:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("\n【混淆矩阵】")
    for name, p in cm_files:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("=" * 80)


def print_final_reminder():
    """
    在最后输出一些建议
    """
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis_results")
    noise_dir = os.path.join(analysis_dir, "noise")
    dl_vs_ml_dir = os.path.join(analysis_dir, "dl_vs_ml")
    comp_dir = os.path.join(analysis_dir, "complexity")

    must_watch = [
        ("【干净数据总表】(全模型 Acc/Std/Train/Infer)", os.path.join(analysis_dir, "summary.txt")),
        ("【噪声鲁棒性总表】(Clean→9→6→3→0 + drop%)", os.path.join(noise_dir, "noise_summary.txt")),
        ("【DL vs ML 噪声曲线图】(论文主图之一)", os.path.join(dl_vs_ml_dir, "acc_vs_snr_dl_vs_ml.png")),
        ("【模型复杂度表】(Params/FLOPs)", os.path.join(comp_dir, "model_complexity.txt")),
    ]

    print("\n" + "=" * 80)
    print("【提醒】跑完以后最优先检查的输出文件：")
    print("=" * 80)

    for name, p in must_watch:
        print(f"  {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="一键运行：噪声评估 + 汇总分析 + 画图 + 混淆矩阵 + DLvsML + 复杂度统计"
    )

    # 原有步骤控制开关
    parser.add_argument("--skip_make_noisy", action="store_true", help="跳过生成带噪测试集 make_noisy_testset.py")
    parser.add_argument("--skip_noise_eval", action="store_true", help="跳过噪声评估 noise_test_all_models.py")
    parser.add_argument("--skip_noise_analyze", action="store_true", help="跳过噪声汇总分析 analyze_noise_results.py")
    parser.add_argument("--skip_noise_plot", action="store_true", help="跳过噪声曲线绘制 plot_noise_curve.py")
    parser.add_argument("--skip_clean_analyze", action="store_true", help="跳过干净数据汇总分析 analyze_all_results_v4.py")
    parser.add_argument("--skip_confusion", action="store_true", help="跳过混淆矩阵绘制 plot_confusion_matrices_all.py")

    # DL vs ML 主对比
    parser.add_argument("--skip_dl_vs_ml", action="store_true", help="跳过 DL vs ML 对比 analyze_dl_vs_ml.py")

    # 复杂度统计（参数量/FLOPs）
    parser.add_argument("--skip_complexity", action="store_true", help="跳过 模型复杂度统计 analyze_model_complexity.py")

    args = parser.parse_args()

    py_exe = sys.executable  # 使用当前终端环境下的 python（建议在 env 中运行）
    print("[INFO] Using python:", py_exe)
    print("[INFO] Project root:", PROJECT_ROOT)

    # 1) 生成带噪测试集
    if not args.skip_make_noisy:
        run_step(py_exe, os.path.join("scripts", "make_noisy_testset.py"),
                 "生成带噪测试集（data/noise_test/）")

    # 2) 噪声评估（DL）
    if not args.skip_noise_eval:
        run_step(py_exe, os.path.join("scripts", "noise_test_all_models.py"),
                 "噪声鲁棒性评估（DL，输出到 noise_results/）")

    # 3) 噪声结果汇总（DL）
    if not args.skip_noise_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_noise_results.py"),
                 "噪声结果汇总（analysis_results/noise/）")

    # 4) 噪声曲线图（DL）
    if not args.skip_noise_plot:
        run_step(py_exe, os.path.join("scripts", "plot_noise_curve.py"),
                 "绘制噪声曲线图（analysis_results/noise/acc_vs_snr.png）")

    # 5) 干净数据汇总 + 对比图（DL 全模型）
    if not args.skip_clean_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_all_results_v4.py"),
                 "干净数据汇总分析（analysis_results/summary.csv + 图）")

    # 6) 混淆矩阵
    if not args.skip_confusion:
        run_step(py_exe, os.path.join("scripts", "plot_confusion_matrices_all.py"),
                 "绘制混淆矩阵（analysis_results/confusion_matrices/）")

    # 7) DL vs ML 主对比（建议放后面，论文主线更直接）
    if not args.skip_dl_vs_ml:
        run_step(py_exe, os.path.join("scripts", "analyze_dl_vs_ml.py"),
                 "DL vs ML 主对比分析（analysis_results/dl_vs_ml/）")

    # 8) 复杂度统计（参数量/FLOPs）
    if not args.skip_complexity:
        run_step(py_exe, os.path.join("scripts", "analyze_model_complexity.py"),
                 "模型复杂度统计（参数量/FLOPs，analysis_results/complexity/）")

    print("\n" + "=" * 80)
    print("[DONE] analysis_all.py 全流程执行完成 ✅")
    print("=" * 80)

    # 最后输出：关键产物清单 
    print_key_artifacts()
    print_final_reminder()


if __name__ == "__main__":
    main()
import os
import sys
import subprocess
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(py_exe: str, script_path: str, title: str):
    print("\n" + "=" * 80)
    print(f"[STEP] {title}")
    print(f"[RUN ] {py_exe} {script_path}")
    print("=" * 80)

    start = datetime.now()
    r = subprocess.run([py_exe, script_path], cwd=PROJECT_ROOT)
    end = datetime.now()

    if r.returncode != 0:
        print(f"\n[ERROR] 步骤失败：{title}")
        print("请查看上方报错信息。")
        sys.exit(r.returncode)

    print(f"[OK] 步骤完成：{title} (耗时 {(end - start)})")


def _fmt_path(p: str) -> str:
    try:
        rel = os.path.relpath(p, PROJECT_ROOT)
        return rel.replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def _exists_mark(p: str) -> str:
    return "✅" if os.path.exists(p) else "⚠️"


def print_key_artifacts():
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis_results")
    noise_dir = os.path.join(analysis_dir, "noise")
    cm_dir = os.path.join(analysis_dir, "confusion_matrices")
    dl_vs_ml_dir = os.path.join(analysis_dir, "dl_vs_ml")
    comp_dir = os.path.join(analysis_dir, "complexity")
    input_dir = os.path.join(analysis_dir, "input_comparison")

    key_files = [
        ("干净数据总表", os.path.join(analysis_dir, "summary.txt")),
        ("干净数据 CSV", os.path.join(analysis_dir, "summary.csv")),
        ("干净数据：准确率对比图", os.path.join(analysis_dir, "accuracy_comparison.png")),
        ("干净数据：训练时间对比图(log)", os.path.join(analysis_dir, "train_time_comparison_log.png")),
        ("干净数据：推理时间对比图(log)", os.path.join(analysis_dir, "infer_time_comparison_log.png")),

        ("噪声鲁棒性总表", os.path.join(noise_dir, "noise_summary.txt")),
        ("噪声鲁棒性 CSV", os.path.join(noise_dir, "noise_summary_table.csv")),
        ("噪声鲁棒性曲线图", os.path.join(noise_dir, "acc_vs_snr.png")),

        ("DL vs ML：干净对比（txt）", os.path.join(dl_vs_ml_dir, "clean_summary_dl_vs_ml.txt")),
        ("DL vs ML：噪声对比（txt）", os.path.join(dl_vs_ml_dir, "noise_summary_dl_vs_ml.txt")),
        ("DL vs ML：噪声曲线图（png）", os.path.join(dl_vs_ml_dir, "acc_vs_snr_dl_vs_ml.png")),

        ("模型复杂度：参数量/FLOPs（txt）", os.path.join(comp_dir, "model_complexity.txt")),
        ("模型复杂度：参数量/FLOPs（csv）", os.path.join(comp_dir, "model_complexity.csv")),

        ("输入形式对比总表", os.path.join(input_dir, "input_comparison_summary.txt")),
        ("输入形式对比柱状图（英文）", os.path.join(input_dir, "input_comparison_bar_en.png")),
        ("输入形式对比柱状图（中文）", os.path.join(input_dir, "input_comparison_bar_cn.png")),

        ("混淆矩阵目录（所有模型）", cm_dir),
    ]

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
    print("〖关键产物清单〗")
    print("=" * 80)

    for name, p in key_files:
        print(f" {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("\n〖混淆矩阵〗")
    for name, p in cm_files:
        print(f" {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("=" * 80)


def print_final_reminder():
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis_results")
    noise_dir = os.path.join(analysis_dir, "noise")
    dl_vs_ml_dir = os.path.join(analysis_dir, "dl_vs_ml")
    comp_dir = os.path.join(analysis_dir, "complexity")
    input_dir = os.path.join(analysis_dir, "input_comparison")

    must_watch = [
        ("〖干净数据总表〗全模型 Acc/Std/Train/Infer", os.path.join(analysis_dir, "summary.txt")),
        ("〖噪声鲁棒性总表〗Clean→9→6→3→0", os.path.join(noise_dir, "noise_summary.txt")),
        ("〖DL vs ML 噪声曲线图〗", os.path.join(dl_vs_ml_dir, "acc_vs_snr_dl_vs_ml.png")),
        ("〖模型复杂度表〗Params/FLOPs", os.path.join(comp_dir, "model_complexity.txt")),
        ("〖输入形式对比总表〗Time/FFT/STFT", os.path.join(input_dir, "input_comparison_summary.txt")),
        ("〖输入形式对比柱状图〗PPT推荐图", os.path.join(input_dir, "input_comparison_bar_en.png")),
    ]

    print("\n" + "=" * 80)
    print("〖提醒〗跑完以后最优先检查的输出文件：")
    print("=" * 80)

    for name, p in must_watch:
        print(f" {_exists_mark(p)} {name}: {_fmt_path(p)}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="一键运行：噪声评估 + 汇总分析 + 画图 + 混淆矩阵 + DLvsML + 复杂度统计 + 输入形式对比"
    )

    parser.add_argument("--skip_make_noisy", action="store_true", help="跳过生成带噪测试集 make_noisy_testset.py")
    parser.add_argument("--skip_noise_eval", action="store_true", help="跳过噪声评估 noise_test_all_models.py")
    parser.add_argument("--skip_noise_analyze", action="store_true", help="跳过噪声汇总分析 analyze_noise_results.py")
    parser.add_argument("--skip_noise_plot", action="store_true", help="跳过噪声曲线绘制 plot_noise_curve.py")
    parser.add_argument("--skip_clean_analyze", action="store_true", help="跳过干净数据汇总分析 analyze_all_results_v4.py")
    parser.add_argument("--skip_confusion", action="store_true", help="跳过混淆矩阵绘制 plot_confusion_matrices_all.py")
    parser.add_argument("--skip_dl_vs_ml", action="store_true", help="跳过 DL vs ML 对比 analyze_dl_vs_ml.py")
    parser.add_argument("--skip_complexity", action="store_true", help="跳过模型复杂度统计 analyze_model_complexity.py")
    parser.add_argument("--skip_input_comparison", action="store_true", help="跳过输入形式对比 analyze_input_comparison.py")

    args = parser.parse_args()
    py_exe = sys.executable

    print("[INFO] Using python:", py_exe)
    print("[INFO] Project root:", PROJECT_ROOT)

    if not args.skip_make_noisy:
        run_step(py_exe, os.path.join("scripts", "make_noisy_testset.py"), "生成带噪测试集（data/noise_test/）")

    if not args.skip_noise_eval:
        run_step(py_exe, os.path.join("scripts", "noise_test_all_models.py"), "噪声鲁棒性评估（DL，输出到 noise_results/）")

    if not args.skip_noise_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_noise_results.py"), "噪声结果汇总（analysis_results/noise/）")

    if not args.skip_noise_plot:
        run_step(py_exe, os.path.join("scripts", "plot_noise_curve.py"), "绘制噪声曲线图（analysis_results/noise/acc_vs_snr.png）")

    if not args.skip_clean_analyze:
        run_step(py_exe, os.path.join("scripts", "analyze_all_results_v4.py"), "干净数据汇总分析（analysis_results/summary.csv + 图）")

    if not args.skip_confusion:
        run_step(py_exe, os.path.join("scripts", "plot_confusion_matrices_all.py"), "绘制混淆矩阵（analysis_results/confusion_matrices/）")

    if not args.skip_dl_vs_ml:
        run_step(py_exe, os.path.join("scripts", "analyze_dl_vs_ml.py"), "DL vs ML 主对比分析（analysis_results/dl_vs_ml/）")

    if not args.skip_complexity:
        run_step(py_exe, os.path.join("scripts", "analyze_model_complexity.py"), "模型复杂度统计（analysis_results/complexity/）")

    if not args.skip_input_comparison:
        run_step(py_exe, os.path.join("scripts", "analyze_input_comparison.py"), "输入形式对比实验分析（Time / FFT / STFT）")

    print("\n" + "=" * 80)
    print("[DONE] analysis_all.py 全流程执行完成 ✅")
    print("=" * 80)

    print_key_artifacts()
    print_final_reminder()


if __name__ == "__main__":
    main()
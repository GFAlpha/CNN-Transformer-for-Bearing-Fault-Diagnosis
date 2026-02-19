import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 配置区
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NOISE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "noise_results")

OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "dl_vs_ml")
os.makedirs(OUT_DIR, exist_ok=True)

# 目标模型 + 三个传统ML
MODELS = [
    ("cnn_transformer_noiseaug", "CNN+Transformer(NoiseAug)"),
    ("ml_svm", "SVM(RBF)"),
    ("ml_rf", "RandomForest"),
    ("ml_knn", "KNN"),
]

# 横轴顺序（干净 -> 9 -> 6 -> 3 -> 0），标签显示：Clean, 9, 6, 3, 0
SNR_ORDER = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]
SNR_LABELS = ["Clean", "9", "6", "3", "0"]


# =========================
# 工具函数
# =========================
def _safe_load_npy(path: str):
    if not os.path.exists(path):
        return None
    return np.load(path)


def _load_infer_times(folder_path: str):
    """
    兼容项目里可能存在的两种命名：
    infer_times.npy / inference_times.npy
    """
    for name in ["infer_times.npy", "inference_times.npy"]:
        p = os.path.join(folder_path, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def _fmt(val, width=12, nd=4):
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return f"{'NaN':>{width}}"
    return f"{val:>{width}.{nd}f}"


# =========================
# ① 干净数据：Acc/Std/Time 汇总
# =========================
def summarize_clean():
    rows = []

    for folder, disp in MODELS:
        p = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(p):
            print(f"[WARN] clean: {folder} not found, skipped.")
            continue

        accs = _safe_load_npy(os.path.join(p, "test_accs.npy"))
        train_times = _safe_load_npy(os.path.join(p, "train_times.npy"))
        infer_times = _load_infer_times(p)

        if accs is None or train_times is None or infer_times is None:
            print(f"[WARN] clean: {folder} missing npy files, skipped.")
            continue

        rows.append({
            "model_key": folder,
            "model": disp,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "train_s": float(np.mean(train_times)),
            "infer_ms": float(np.mean(infer_times) * 1000.0),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "clean_summary_dl_vs_ml.csv")
    out_txt = os.path.join(OUT_DIR, "clean_summary_dl_vs_ml.txt")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Clean Data Summary (DL vs ML)\n")
        f.write("Models: CNN+Transformer(NoiseAug) vs SVM/RF/KNN\n")
        f.write("=" * 92 + "\n")
        f.write(f"{'Model':<28}{'Acc Mean':>10}{'Acc Std':>10}{'Train(s)':>12}{'Infer(ms)':>12}\n")
        f.write("-" * 92 + "\n")
        for _, r in df.iterrows():
            f.write(f"{r['model']:<28}"
                    f"{r['acc_mean']:>10.4f}"
                    f"{r['acc_std']:>10.4f}"
                    f"{r['train_s']:>12.2f}"
                    f"{r['infer_ms']:>12.3f}\n")

    print("[OK] clean summary saved:")
    print(" -", out_csv)
    print(" -", out_txt)

    return df


# =========================
# ② 噪声：鲁棒性汇总（Acc + drop%）
# =========================
def summarize_noise():
    rows = []

    for model_key, disp in MODELS:
        for tag in SNR_ORDER:
            d = os.path.join(NOISE_RESULTS_DIR, model_key, tag)
            acc_path = os.path.join(d, "acc.npy")
            if not os.path.exists(acc_path):
                # 不强制报错（有时候还没跑ML噪声）
                rows.append({
                    "model_key": model_key,
                    "model": disp,
                    "snr_tag": tag,
                    "acc": np.nan,
                })
                continue

            acc = float(np.load(acc_path).reshape(-1)[0])
            rows.append({
                "model_key": model_key,
                "model": disp,
                "snr_tag": tag,
                "acc": acc,
            })

    df_long = pd.DataFrame(rows)

    # 宽表：model_key 为 index
    wide = df_long.pivot_table(index=["model_key", "model"], columns="snr_tag", values="acc", aggfunc="mean")
    wide = wide.reindex(columns=SNR_ORDER)

    # drop%
    if "clean" in wide.columns:
        for tag in ["snr_9", "snr_6", "snr_3", "snr_0"]:
            wide[f"drop_{tag}"] = (wide["clean"] - wide[tag]) / wide["clean"] * 100.0

    # 输出
    out_csv = os.path.join(OUT_DIR, "noise_summary_dl_vs_ml.csv")
    out_txt = os.path.join(OUT_DIR, "noise_summary_dl_vs_ml.txt")

    wide.to_csv(out_csv, encoding="utf-8-sig")

    # txt：列顺序按照 Clean,9,6,3,0，再接 drop
    cols = SNR_ORDER + [f"drop_{t}" for t in ["snr_9", "snr_6", "snr_3", "snr_0"]]

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Noise Robustness Summary (DL vs ML, AWGN on Test Set)\n")
        f.write("SNR order: Clean -> 9 -> 6 -> 3 -> 0\n")
        f.write("=" * 140 + "\n")
        f.write(f"{'Model':<28}")
        for c in cols:
            f.write(f"{c:>12}")
        f.write("\n")
        f.write("-" * 140 + "\n")

        for (model_key, model_name), row in wide.iterrows():
            f.write(f"{model_name:<28}")
            for c in cols:
                v = row[c] if c in row.index else np.nan
                f.write(_fmt(float(v) if pd.notna(v) else np.nan, width=12, nd=4))
            f.write("\n")

    print("[OK] noise summary saved:")
    print(" -", out_csv)
    print(" -", out_txt)

    return df_long, wide


# =========================
# ③ 噪声折线图（DL vs ML）
# =========================
def plot_noise_curve(wide_df):
    out_png = os.path.join(OUT_DIR, "acc_vs_snr_dl_vs_ml.png")

    plt.figure(figsize=(10, 5))

    # wide_df index: (model_key, model)
    for (model_key, model_name), row in wide_df.iterrows():
        y = []
        for tag in SNR_ORDER:
            v = row[tag] if tag in row.index else np.nan
            y.append(float(v) if pd.notna(v) else np.nan)

        plt.plot(SNR_LABELS, y, marker="o", label=model_name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR (DL vs ML)")
    plt.ylim(0.0, 1.01)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("[OK] noise curve saved:")
    print(" -", out_png)


def main():
    print("=" * 70)
    print("[STEP] Clean summary (DL vs ML)")
    clean_df = summarize_clean()

    print("=" * 70)
    print("[STEP] Noise summary (DL vs ML)")
    df_long, wide = summarize_noise()

    # 如果噪声结果都缺失，就不画图
    has_any_noise = False
    for tag in SNR_ORDER:
        if tag in wide.columns and wide[tag].notna().any():
            has_any_noise = True
            break

    if has_any_noise:
        print("=" * 70)
        print("[STEP] Plot noise curve (DL vs ML)")
        plot_noise_curve(wide)
    else:
        print("[WARN] 当前没有可用噪声数据（尤其是ML噪声）。已跳过噪声曲线绘制。")
        print("      请先运行：python scripts/noise_test_ml_models.py")

    print("=" * 70)
    print("[DONE] analyze_dl_vs_ml.py finished.")
    print("Outputs saved to:")
    print(" -", OUT_DIR)
    print("Key files (for thesis writing):")
    print(" - clean_summary_dl_vs_ml.txt / .csv")
    print(" - noise_summary_dl_vs_ml.txt / .csv")
    print(" - acc_vs_snr_dl_vs_ml.png")


if __name__ == "__main__":
    main()
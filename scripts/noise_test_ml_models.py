import os
import json
import time
import numpy as np
import joblib
from sklearn.metrics import accuracy_score


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "noise_test")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NOISE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "noise_results")

# 三个 ML 都评估
ML_MODELS = [
    ("ml_svm", "SVM(RBF)"),
    ("ml_rf", "RandomForest"),
    ("ml_knn", "KNN"),
]

# 横轴顺序：Clean → 9 → 6 → 3 → 0
TAGS = ["clean", "snr_9", "snr_6", "snr_3", "snr_0"]

X_FILE_MAP = {
    "clean": "X_test_clean.npy",
    "snr_9": "X_test_snr_9.npy",
    "snr_6": "X_test_snr_6.npy",
    "snr_3": "X_test_snr_3.npy",
    "snr_0": "X_test_snr_0.npy",
}
Y_FILE = "y_test.npy"


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _load_X(tag: str) -> np.ndarray:
    x_path = os.path.join(DATA_DIR, X_FILE_MAP[tag])
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"找不到 {x_path}，请先运行 make_noisy_testset.py 生成噪声测试集。")
    X = np.load(x_path)
    # 统一展平到 [N, D]
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim != 2:
        raise ValueError(f"不支持的 X 维度：{X.shape}")
    return X


def _load_y() -> np.ndarray:
    y_path = os.path.join(DATA_DIR, Y_FILE)
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"找不到 {y_path}，请确认 data/noise_test 下有 y_test.npy")
    return np.load(y_path)


def _save_one(model_key: str, tag: str, acc: float, infer_ms_per_sample: float, meta: dict):
    out_dir = os.path.join(NOISE_RESULTS_DIR, model_key, tag)
    _ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "acc.npy"), np.array([acc], dtype=np.float64))

    # 这里存 “每样本推理时间(秒)” 的形式，后续分析脚本会乘 1000 变成 ms
    infer_s_per_sample = infer_ms_per_sample / 1000.0
    np.save(os.path.join(out_dir, "infer_times.npy"), np.array([infer_s_per_sample], dtype=np.float64))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"找不到 {DATA_DIR}，请先生成 data/noise_test/")

    y = _load_y()
    _ensure_dir(NOISE_RESULTS_DIR)

    for model_key, display_name in ML_MODELS:
        model_path = os.path.join(RESULTS_DIR, model_key, "model.joblib")
        if not os.path.exists(model_path):
            print(f"[WARN] {model_key} 缺少模型文件：{model_path}，跳过")
            continue

        clf = joblib.load(model_path)

        for tag in TAGS:
            X = _load_X(tag)

            # 推理计时：用 predict(X) 的总耗时 / 样本数
            t0 = time.perf_counter()
            y_pred = clf.predict(X)
            t1 = time.perf_counter()

            acc = float(accuracy_score(y, y_pred))
            infer_ms_per_sample = float((t1 - t0) / max(len(y), 1) * 1000.0)

            meta = {
                "model_key": model_key,
                "display_name": display_name,
                "tag": tag,
                "infer_ms_per_sample": infer_ms_per_sample,
                "note": "ML baseline noise evaluation; infer_times.npy stores per-sample seconds.",
            }

            _save_one(model_key, tag, acc, infer_ms_per_sample, meta)

            # 终端输出保持和你原本 noise_test_all_models.py 类似的风格
            print(f"[OK] {model_key:<18} | {tag:<6} | acc={acc:.4f} | infer={infer_ms_per_sample:.3f}ms/sample")

    print(f"\n[DONE] ML noise evaluation finished. 输出目录: noise_results/")


if __name__ == "__main__":
    main()
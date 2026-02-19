import os
import time
import json
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =========================
# 配置区
# =========================
SEED = 42
RUNS = 5  # 和深度学习脚本保持一致，便于统计 mean/std
RESULTS_ROOT = "results"

# 数据集路径
DATA_CANDIDATES = [
    os.path.join("data", "splits"),
    os.path.join("data", "splits", "CWRU_12K_DE"),
    os.path.join("data", "splits", "cwru"),  # 兜底
]

# 三个传统模型，保证baseline简单可靠
MODEL_ZOO = {
    "ml_svm": {
        "display_name": "SVM(RBF)",
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", probability=False, random_state=SEED)),
        ]),
        "notes": "SVM 对尺度敏感，需标准化；RBF 核通常作为强基线。",
    },
    "ml_rf": {
        "display_name": "RandomForest",
        "estimator": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=SEED,
            n_jobs=-1
        ),
        "notes": "RF 对尺度不敏感，鲁棒易用；训练/推理速度通常不错。",
    },
    "ml_knn": {
        "display_name": "KNN",
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)),
        ]),
        "notes": "KNN 非参数模型，训练快；推理慢（要算距离），但很好解释。",
    },
}


def _find_splits_dir():
    for d in DATA_CANDIDATES:
        if os.path.isdir(d) and \
           os.path.exists(os.path.join(d, "X_train.npy")) and \
           os.path.exists(os.path.join(d, "y_train.npy")):
            return d
    raise FileNotFoundError(
        "找不到数据切分文件：请确认 data/splits/ 下存在 X_train.npy, y_train.npy 等文件。"
    )


def _load_splits():
    d = _find_splits_dir()
    X_train = np.load(os.path.join(d, "X_train.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    X_val = np.load(os.path.join(d, "X_val.npy"))
    y_val = np.load(os.path.join(d, "y_val.npy"))
    X_test = np.load(os.path.join(d, "X_test.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    return d, X_train, y_train, X_val, y_val, X_test, y_test


def _to_2d_features(X: np.ndarray) -> np.ndarray:
    """
    将数据一般统一展平到 [N, 1024]的特征矩阵
    """
    if X.ndim == 3:
        # [N, C, L] -> [N, C*L]
        return X.reshape(X.shape[0], -1)
    if X.ndim == 2:
        return X
    raise ValueError(f"不支持的 X 维度：{X.shape}")


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_results(out_dir: str,
                  test_accs: np.ndarray,
                  train_times: np.ndarray,
                  infer_times: np.ndarray,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  meta: dict,
                  fitted_model):
    _ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "test_accs.npy"), test_accs)
    np.save(os.path.join(out_dir, "train_times.npy"), train_times)
    np.save(os.path.join(out_dir, "infer_times.npy"), infer_times)
    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 保存模型
    joblib.dump(fitted_model, os.path.join(out_dir, "model.joblib"))


def train_one(model_key: str, model_cfg: dict, X_train, y_train, X_val, y_val, X_test, y_test):
    out_dir = os.path.join(RESULTS_ROOT, model_key)
    _ensure_dir(out_dir)

    test_accs = []
    train_times = []
    infer_times = []

    # 记录 best（按 val acc）
    best_val_acc = -1.0
    best_model = None
    best_test_pred = None

    # 为了跟 DL 的“多次 run”一致：这里每次 run 都改随机种子
    for run in range(1, RUNS + 1):
        est = model_cfg["estimator"]

        # 有 random_state 的模型尽量设置一下
        if hasattr(est, "random_state"):
            try:
                est.set_params(random_state=SEED + run)
            except Exception:
                pass

        # 训练计时
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        t1 = time.perf_counter()
        train_times.append(t1 - t0)

        # 验证集评估
        y_val_pred = est.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        # 测试集推理计时（按“每步/每次 predict”粗略估计）
        t2 = time.perf_counter()
        y_test_pred = est.predict(X_test)
        t3 = time.perf_counter()
        infer_times.append((t3 - t2))  # 这里是整段 predict 的总时间

        test_acc = accuracy_score(y_test, y_test_pred)
        test_accs.append(test_acc)

        print(f"[OK] {model_key:<10} | Run {run}/{RUNS} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = est
            best_test_pred = y_test_pred

    test_accs = np.array(test_accs, dtype=np.float64)
    train_times = np.array(train_times, dtype=np.float64)
    infer_times = np.array(infer_times, dtype=np.float64)

    meta = {
        "model_key": model_key,
        "display_name": model_cfg.get("display_name", model_key),
        "notes": model_cfg.get("notes", ""),
        "seed_base": SEED,
        "runs": RUNS,
        "best_val_acc": float(best_val_acc),
        "test_acc_mean": float(test_accs.mean()),
        "test_acc_std": float(test_accs.std()),
        "avg_train_time_s": float(train_times.mean()),
        "avg_infer_time_s_total_predict": float(infer_times.mean()),
        "feature_type": "raw_flatten_1024",
    }

    # 兼容分析脚本的“推理时间是 per step(per batch)”概念：
    # 简单把一次 predict(X_test) 的总耗时除以测试样本数，作为“每个样本的平均推理耗时”
    infer_per_sample = infer_times.mean() / max(len(y_test), 1)
    meta["avg_infer_time_s_per_sample"] = float(infer_per_sample)

    _save_results(
        out_dir=out_dir,
        test_accs=test_accs,
        train_times=train_times,
        infer_times=np.array([infer_per_sample], dtype=np.float64),  # 用 per-sample 形式，便于跟 DL 更可比
        y_true=y_test,
        y_pred=best_test_pred,
        meta=meta,
        fitted_model=best_model
    )

    print(f"[DONE] Saved ML results to: {out_dir}")
    print(f"       Mean Acc: {test_accs.mean():.4f} | Std: {test_accs.std():.4f} | "
          f"Train(s): {train_times.mean():.2f} | Infer(ms/sample): {infer_per_sample*1000:.3f}")


def main():
    splits_dir, X_train, y_train, X_val, y_val, X_test, y_test = _load_splits()

    print("[INFO] Found splits at:", splits_dir.replace("\\", "/"))
    print("[INFO] Shapes:",
          "X_train", X_train.shape,
          "X_val", X_val.shape,
          "X_test", X_test.shape)

    # 转成 2D 特征
    X_train_f = _to_2d_features(X_train)
    X_val_f = _to_2d_features(X_val)
    X_test_f = _to_2d_features(X_test)

    # 固定随机种子（影响 numpy 的部分）
    np.random.seed(SEED)

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    for k, cfg in MODEL_ZOO.items():
        train_one(k, cfg, X_train_f, y_train, X_val_f, y_val, X_test_f, y_test)


if __name__ == "__main__":
    main()
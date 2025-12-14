import numpy as np
import matplotlib.pyplot as plt
import os

# ========================
# 1. 明确：模型名 -> 对应文件路径
# ========================
MODEL_FILES = {
    "CNN": "results/cnn_multi_run/test_accs.npy",
    "LSTM": "results/rnn_multi_run/test_accs.npy",
    "CNN+BiLSTM": "results/cnn_bilstm_test_accs.npy",
    "CNN+BiLSTM+Att": "results/cnn_bilstm_att_test_accs.npy",
    "Transformer": "results/transformer_test_accs.npy",
    "CNN+Transformer": "results/cnn_transformer_test_accs.npy",
}

stats = {}

# ========================
# 2. 读取 & 统计
# ========================
for model, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"[跳过] {model} - 找不到文件: {path}")
        continue

    accs = np.load(path)
    stats[model] = {
        "accs": accs,
        "mean": accs.mean(),
        "std": accs.std()
    }

# ========================
# 3. 打印论文级表格
# ========================
print("\n===== Model Performance Summary =====")
print("{:<22} {:>10} {:>10}".format("Model", "Mean Acc", "Std"))
print("-" * 46)

for model, s in stats.items():
    print("{:<22} {:>10.4f} {:>10.4f}".format(
        model, s["mean"], s["std"]
    ))

# ========================
# 4. 画统一对比图
# ========================
models = list(stats.keys())
means = [stats[m]["mean"] for m in models]
stds  = [stats[m]["std"] for m in models]

plt.figure(figsize=(11, 5))
plt.bar(models, means, yerr=stds, capsize=6)
plt.ylabel("Test Accuracy")
plt.title("Performance Comparison of Different Models")
plt.ylim(0.6, 1.01)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("results/all_model_comparison.png", dpi=300)
plt.show()

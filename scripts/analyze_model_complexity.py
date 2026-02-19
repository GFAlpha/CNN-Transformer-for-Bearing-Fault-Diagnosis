import os
import sys
import math
import json
import numpy as np
import pandas as pd
import torch

from datetime import datetime

# =========================================================
# 说明：
# 1) 统计 7 个 DL 模型的参数量 + FLOPs（尽力）
# 2) FLOPs 通过 torch.profiler(with_flops=True) 做一次前向推断统计
#    - 注意：PyTorch 对部分算子/模块可能无法给出 flops（会出现 N/A）
# 3) 输出 csv + txt，方便论文引用
# =========================================================


# =========================================================
# 项目根目录 & 导入训练脚本中的模型类
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# 从你现有训练脚本中导入模型
from train_cnn_v2 import CNN1D
from train_rnn_v4 import RNNLSTM, SEQ_LEN, FEAT_DIM
from train_cnn_bilstm import CNN_BiLSTM
from train_cnn_bilstm_att import CNN_BiLSTM_Att
from train_transformer_v2 import TransformerClassifier
from train_cnn_transformer import CNNTransformer


# =========================================================
# 输出目录
# =========================================================
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis_results", "complexity")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "model_complexity.csv")
OUT_TXT = os.path.join(OUT_DIR, "model_complexity.txt")


# =========================================================
# 工具函数：参数统计
# =========================================================
def count_params(model: torch.nn.Module):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return int(total), int(trainable)


def fmt_million(n: int) -> str:
    """把参数量显示成 M（百万）"""
    return f"{n / 1e6:.4f}M"


def fmt_giga(n: float) -> str:
    """把 FLOPs 显示成 GFLOPs"""
    return f"{n / 1e9:.4f}G"


# =========================================================
# 工具函数：用 profiler 尽力统计 FLOPs
# =========================================================
def try_profile_flops(model: torch.nn.Module, example_input: torch.Tensor):
    """
    返回：flops(float) 或 None
    说明：
    - profiler 的 flops 是对一些算子可用，不保证所有算子都能统计到
    - 所以如果得到 0 或者都 None，就返回 None
    """
    # 强制 CPU 统计，避免 GPU 异步导致混乱，也避免必须有 CUDA
    model = model.to("cpu").eval()
    x = example_input.to("cpu")

    # 有的版本 torch.profiler 在 Windows 上可能偶发报错，做 try/except
    try:
        import torch.profiler as profiler
        activities = [profiler.ProfilerActivity.CPU]

        with torch.no_grad():
            with profiler.profile(
                activities=activities,
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                with_flops=True,
            ) as prof:
                _ = model(x)

        # 累加所有 event 的 flops
        flops_sum = 0
        has_any = False
        for evt in prof.key_averages():
            f = getattr(evt, "flops", None)
            if f is not None:
                has_any = True
                flops_sum += float(f)

        # 如果统计不到或为 0，则认为不可用
        if (not has_any) or flops_sum <= 0:
            return None

        return flops_sum

    except Exception:
        return None


# =========================================================
# 7 个模型的构建 & 输入适配
# =========================================================
def build_model_and_input(key: str):
    """
    返回：
    - display_name: 显示名
    - model: torch.nn.Module
    - example_input: 用于 FLOPs 统计的 dummy 输入（batch=1）
    - input_desc: 记录输入形状说明（写入 txt，论文可解释）
    """
    # 统一分类数=4
    num_classes = 4

    if key == "cnn":
        model = CNN1D(num_classes=num_classes)
        x = torch.randn(1, 1, 1024)  # [B,1,L]
        return "CNN", model, x, "[1, 1, 1024]"

    if key == "rnn_lstm":
        model = RNNLSTM(input_dim=FEAT_DIM, hidden_dim=128, num_layers=2, num_classes=num_classes)
        x = torch.randn(1, SEQ_LEN, FEAT_DIM)  # [B,SEQ_LEN,FEAT_DIM] = [1,32,32]
        return "LSTM", model, x, f"[1, {SEQ_LEN}, {FEAT_DIM}]"

    if key == "cnn_bilstm":
        model = CNN_BiLSTM(num_classes=num_classes)
        x = torch.randn(1, 1024)  # 模型内部会兼容 [B,L] 或 [B,1,L]
        return "CNN+BiLSTM", model, x, "[1, 1024]"

    if key == "cnn_bilstm_att":
        model = CNN_BiLSTM_Att(num_classes=num_classes)
        x = torch.randn(1, 1024)
        return "CNN+BiLSTM+Att", model, x, "[1, 1024]"

    if key == "transformer":
        model = TransformerClassifier()  # 你脚本默认 input_dim=1, d_model=64...
        x = torch.randn(1, 1024, 1)  # [B,L,1]
        return "Transformer", model, x, "[1, 1024, 1]"

    if key == "cnn_transformer":
        model = CNNTransformer(num_classes=num_classes)
        x = torch.randn(1, 1, 1024)
        return "CNN+Transformer", model, x, "[1, 1, 1024]"

    if key == "cnn_transformer_noiseaug":
        # 注意：NoiseAug 的复杂度和 CNN+Transformer 结构一致（训练策略不同）
        model = CNNTransformer(num_classes=num_classes)
        x = torch.randn(1, 1, 1024)
        return "CNN+Transformer(NoiseAug)", model, x, "[1, 1, 1024]"

    raise ValueError(f"Unknown model key: {key}")


MODEL_KEYS = [
    "cnn",
    "rnn_lstm",
    "cnn_bilstm",
    "cnn_bilstm_att",
    "transformer",
    "cnn_transformer",
    "cnn_transformer_noiseaug",
]


def main():
    print("[INFO] Project root:", PROJECT_ROOT)
    print("[INFO] Output dir  :", OUT_DIR.replace("\\", "/"))

    records = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for k in MODEL_KEYS:
        display_name, model, x, input_desc = build_model_and_input(k)

        # 参数量
        total_params, trainable_params = count_params(model)

        # FLOPs（尽力）
        flops = try_profile_flops(model, x)  # 可能为 None

        rec = {
            "model_key": k,
            "model": display_name,
            "input_shape(batch=1)": input_desc,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6,
            "flops": float(flops) if flops is not None else np.nan,
            "gflops": float(flops) / 1e9 if flops is not None else np.nan,
        }
        records.append(rec)

        # 终端输出（方便你快速看）
        if flops is None:
            print(f"[OK] {display_name:<24} | Params={fmt_million(total_params):>10} | FLOPs=N/A")
        else:
            print(f"[OK] {display_name:<24} | Params={fmt_million(total_params):>10} | FLOPs={fmt_giga(flops):>10}")

    df = pd.DataFrame(records)

    # 保存 CSV
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 保存 TXT（更像论文里直接贴表格）
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Model Complexity Summary (Params & FLOPs)\n")
        f.write(f"Generated at: {now}\n")
        f.write("Notes:\n")
        f.write(" - Params: total/trainable parameters.\n")
        f.write(" - FLOPs: measured by torch.profiler(with_flops=True) with batch=1.\n")
        f.write(" - Some ops may not report FLOPs in PyTorch -> shown as N/A.\n")
        f.write("=" * 110 + "\n")

        f.write(f"{'Model':<26}"
                f"{'Input':>16}"
                f"{'Params(M)':>14}"
                f"{'Trainable(M)':>14}"
                f"{'GFLOPs':>14}\n")
        f.write("-" * 110 + "\n")

        for _, r in df.iterrows():
            model_name = r["model"]
            inp = r["input_shape(batch=1)"]
            pm = r["total_params_M"]
            tm = r["trainable_params_M"]
            gf = r["gflops"]

            gf_str = "N/A" if (isinstance(gf, float) and (math.isnan(gf) or math.isinf(gf))) else f"{gf:.4f}"
            f.write(f"{model_name:<26}"
                    f"{str(inp):>16}"
                    f"{pm:>14.4f}"
                    f"{tm:>14.4f}"
                    f"{gf_str:>14}\n")

        f.write("=" * 110 + "\n")
        f.write("Tips for thesis:\n")
        f.write(" - Use Params + Inference Time as 'efficiency' indicators.\n")
        f.write(" - NoiseAug has same Params/FLOPs as CNN+Transformer; improvement comes from training strategy.\n")

    print("\n[DONE] Complexity analysis finished.")
    print(" - CSV:", OUT_CSV.replace("\\", "/"))
    print(" - TXT:", OUT_TXT.replace("\\", "/"))


if __name__ == "__main__":
    main()
# Bearing_Diagnosis

<div align="center">

# Bearing Fault Diagnosis Based on Deep Learning

本科毕业设计项目 / Undergraduate Graduation Project

Author: GFAlpha

</div>

---

# English Version

## Introduction

This repository contains the undergraduate graduation project of **GFAlpha**.

The project focuses on intelligent bearing fault diagnosis using deep learning methods based on the:

```text
Case Western Reserve University (CWRU) Bearing Dataset
```

The project implements:

- Data preprocessing and slicing
- Deep learning model training
- Traditional machine learning baselines
- Noise robustness testing
- Feature visualization
- Confusion matrix analysis
- Model complexity analysis
- Deep Learning vs Machine Learning comparison

The final proposed model is:

```text
CNN + Transformer + NoiseAug
```

Core ideas:

- CNN extracts local fault features
- Transformer models global dependencies
- Noise augmentation improves robustness under noisy environments

---

# Project Structure

```text
Bearing_Diagnosis/
│
├── configs/                         # Configuration files
│   └── default.yaml
│
├── data/
│   ├── raw/                         # Original CWRU .mat files
│   │
│   ├── processed/                   # Processed full dataset
│   │   ├── X.npy
│   │   └── y.npy
│   │
│   ├── splits/                      # Train / val / test split
│   │
│   └── noise_test/                  # Noisy test datasets
│
├── models/                          # Saved model weights
│
├── results/                         # Training/testing results
│
├── noise_results/                   # Noise robustness results
│
├── analysis_results/                # Analysis figures and summary tables
│
├── scripts/                         # Main executable scripts
│
├── src/                             # Core modules
│
└── README.md
```

---

# Dataset

This project uses:

```text
Case Western Reserve University (CWRU) Bearing Dataset
```

The original vibration signals are stored under:

```text
data/raw/
```

Directory example:

```text
data/raw/CWRU_Normal/
data/raw/CWRU_12K_DE/Ball/
data/raw/CWRU_12K_DE/Inner Race/
data/raw/CWRU_12K_DE/Outer Race/
```

Mainly used:

- 12k Drive End vibration signals
- Normal condition
- Ball fault
- Inner race fault
- Outer race fault

---

# Data Processing Pipeline

Complete workflow:

```text
Original CWRU .mat signals
        ↓
prepare_cwru.py
        ↓
1024-point window slicing
        ↓
X.npy / y.npy
        ↓
Fixed train/validation/test split
        ↓
Model training
        ↓
Noise robustness testing
        ↓
Analysis and visualization
```

---

# Environment & Dependencies

## PyTorch Environment

Recommended environment:

```text
Python 3.10+
PyTorch 2.9.1+cu130
TorchVision 0.24.1+cu130
TorchAudio 2.9.1+cu130
CUDA 13.0
Windows 11
```

GPU acceleration is recommended.

---

## Install PyTorch

CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

CPU version:

```bash
pip install torch torchvision torchaudio
```

---

## Install Required Packages

```bash
pip install numpy scipy matplotlib pandas scikit-learn seaborn joblib thop tqdm pyyaml opencv-python
```

---

## Verify Environment

```bash
python scripts/test_torch.py
python scripts/test_cuda.py
```

These scripts verify:

- PyTorch installation
- CUDA availability
- GPU information

---

# Main Scripts

# 1. Data Preparation

## prepare_cwru.py

Location:

```text
scripts/prepare_cwru.py
```

Functions:

- Load original CWRU `.mat` files
- Slice signals using:
  - Window length = 1024
  - Step size = 1024
- Generate:
  - `X.npy`
  - `y.npy`

Run:

```bash
python scripts/prepare_cwru.py
```

---

## split_dataset_fixed.py

Location:

```text
scripts/split_dataset_fixed.py
```

Functions:

- Fixed train/validation/test split
- Random seed = 42

Run:

```bash
python scripts/split_dataset_fixed.py
```

---

# 2. Deep Learning Model Training

## CNN

```bash
python scripts/train_cnn_v2.py
```

## LSTM

```bash
python scripts/train_rnn_v4.py
```

## CNN + BiLSTM

```bash
python scripts/train_cnn_bilstm.py
```

## CNN + BiLSTM + Attention

```bash
python scripts/train_cnn_bilstm_att.py
```

## Transformer

```bash
python scripts/train_transformer_v2.py
```

## CNN + Transformer

```bash
python scripts/train_cnn_transformer.py
```

## CNN + Transformer + NoiseAug

```bash
python scripts/train_cnn_transformer_noiseaug.py
```

---

# 3. Traditional Machine Learning Baselines

```bash
python scripts/train_ml_baselines.py
```

Includes:

- SVM (RBF)
- Random Forest
- KNN

---

# 4. Noise Robustness Testing

## Generate noisy test datasets

```bash
python scripts/make_noisy_testset.py
```

Generated SNR levels:

```text
9 dB
6 dB
3 dB
0 dB
```

---

## Test DL models under noise

```bash
python scripts/noise_test_all_models.py
```

---

## Test ML models under noise

```bash
python scripts/noise_test_ml_models.py
```

---

# 5. Analysis Scripts

## Overall result analysis

```bash
python scripts/analyze_all_results_v4.py
```

Outputs:

- Accuracy comparison
- Training time comparison
- Inference time comparison

---

## Noise robustness analysis

```bash
python scripts/analyze_noise_results.py
```

---

## Noise curve plotting

```bash
python scripts/plot_noise_curve.py
```

---

## Confusion matrix plotting

```bash
python scripts/plot_confusion_matrices_all.py
```

---

## Feature visualization

```bash
python scripts/analyze_feature_visualization.py
```

Includes:

- t-SNE
- PCA

---

## Model complexity analysis

```bash
python scripts/analyze_model_complexity.py
```

Includes:

- Parameter count
- FLOPs

---

## DL vs ML comparison

```bash
python scripts/analyze_dl_vs_ml.py
```

---

# 6. One-Click Analysis Pipeline

```bash
python scripts/analysis_all.py
```

Automatically runs:

- Noise dataset generation
- Noise testing
- Result analysis
- Confusion matrix plotting
- Complexity analysis
- DL vs ML comparison

---

# Visualization Script

```bash
python scripts/visualize_input_waveforms.py
```

Functions:

- Visualize original CWRU signals
- Visualize 1024-point model inputs
- Visualize noisy test samples

---

# Proposed Model

Final proposed architecture:

```text
Input Signal
      ↓
CNN Feature Extraction
      ↓
Transformer Global Modeling
      ↓
Classification Head
```

Noise augmentation is additionally introduced during training to improve robustness under low-SNR conditions.

---

# Author

GFAlpha

Undergraduate Graduation Project

Bearing Fault Diagnosis Based on Deep Learning

---

# 中文版本

# 项目简介

本仓库为：

```text
GFAlpha 本科毕业设计项目
```

项目主题为：

```text
基于深度学习的轴承故障诊断
```

项目基于：

```text
凯斯西储大学（CWRU）轴承数据集
```

实现了：

- 数据预处理与切片
- 深度学习模型训练
- 传统机器学习基线对比
- 噪声鲁棒性测试
- 特征可视化
- 混淆矩阵分析
- 模型复杂度分析
- 深度学习与传统机器学习对比

最终提出模型：

```text
CNN + Transformer + NoiseAug
```

核心思想：

- CNN 提取局部特征
- Transformer 建模全局依赖
- NoiseAug 提升低信噪比环境下鲁棒性

---

# 项目结构

```text
Bearing_Diagnosis/
│
├── configs/                         # 配置文件
├── data/                            # 数据集目录
├── models/                          # 模型权重
├── results/                         # 实验结果
├── noise_results/                   # 噪声测试结果
├── analysis_results/                # 分析结果与图表
├── scripts/                         # 核心脚本
├── src/                             # 核心模块
└── README.md
```

---

# 数据集

项目使用：

```text
CWRU（Case Western Reserve University）轴承数据集
```

原始 `.mat` 数据位置：

```text
data/raw/
```

目录示例：

```text
data/raw/CWRU_Normal/
data/raw/CWRU_12K_DE/Ball/
data/raw/CWRU_12K_DE/Inner Race/
data/raw/CWRU_12K_DE/Outer Race/
```

主要使用：

- 12k Drive End 信号
- 正常状态
- 滚动体故障
- 内圈故障
- 外圈故障

---

# 数据处理流程

完整数据流：

```text
原始 CWRU .mat 信号
        ↓
prepare_cwru.py
        ↓
1024 点窗口切片
        ↓
X.npy / y.npy
        ↓
固定训练/验证/测试集划分
        ↓
模型训练
        ↓
噪声鲁棒性测试
        ↓
分析与可视化
```

---

# 环境与依赖

## PyTorch 环境

推荐环境：

```text
Python 3.10+
PyTorch 2.9.1+cu130
TorchVision 0.24.1+cu130
TorchAudio 2.9.1+cu130
CUDA 13.0
Windows 11
```

推荐使用 NVIDIA GPU 训练。

---

## 安装 PyTorch

CUDA 版本：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

CPU 版本：

```bash
pip install torch torchvision torchaudio
```

---

## 安装项目依赖

```bash
pip install numpy scipy matplotlib pandas scikit-learn seaborn joblib thop tqdm pyyaml opencv-python
```

---

## 验证环境

```bash
python scripts/test_torch.py
python scripts/test_cuda.py
```

可验证：

- PyTorch 是否安装成功
- CUDA 是否可用
- GPU 信息

---

# 主要脚本说明

# 1. 数据准备

## prepare_cwru.py

功能：

- 读取原始 CWRU `.mat`
- 进行 1024 点无重叠切片
- 生成：
  - `X.npy`
  - `y.npy`

运行：

```bash
python scripts/prepare_cwru.py
```

---

## split_dataset_fixed.py

功能：

- 固定随机种子划分训练集
- 生成 train/val/test

运行：

```bash
python scripts/split_dataset_fixed.py
```

---

# 2. 深度学习模型训练

## CNN

```bash
python scripts/train_cnn_v2.py
```

## LSTM

```bash
python scripts/train_rnn_v4.py
```

## CNN+BiLSTM

```bash
python scripts/train_cnn_bilstm.py
```

## CNN+BiLSTM+Attention

```bash
python scripts/train_cnn_bilstm_att.py
```

## Transformer

```bash
python scripts/train_transformer_v2.py
```

## CNN+Transformer

```bash
python scripts/train_cnn_transformer.py
```

## CNN+Transformer(NoiseAug)

```bash
python scripts/train_cnn_transformer_noiseaug.py
```

---

# 3. 传统机器学习基线

```bash
python scripts/train_ml_baselines.py
```

包括：

- SVM
- RandomForest
- KNN

---

# 4. 噪声鲁棒性测试

## 生成噪声测试集

```bash
python scripts/make_noisy_testset.py
```

SNR：

```text
9 dB
6 dB
3 dB
0 dB
```

---

## 深度学习模型噪声测试

```bash
python scripts/noise_test_all_models.py
```

---

## 传统机器学习模型噪声测试

```bash
python scripts/noise_test_ml_models.py
```

---

# 5. 分析脚本

## 综合结果分析

```bash
python scripts/analyze_all_results_v4.py
```

---

## 噪声鲁棒性分析

```bash
python scripts/analyze_noise_results.py
```

---

## 噪声曲线绘制

```bash
python scripts/plot_noise_curve.py
```

---

## 混淆矩阵绘制

```bash
python scripts/plot_confusion_matrices_all.py
```

---

## 特征可视化

```bash
python scripts/analyze_feature_visualization.py
```

包括：

- t-SNE
- PCA

---

## 模型复杂度分析

```bash
python scripts/analyze_model_complexity.py
```

包括：

- 参数量
- FLOPs

---

## 深度学习 vs 传统机器学习

```bash
python scripts/analyze_dl_vs_ml.py
```

---

# 一键分析总脚本

```bash
python scripts/analysis_all.py
```

自动完成：

- 噪声数据生成
- 噪声测试
- 结果汇总
- 混淆矩阵绘制
- 模型复杂度分析
- DL vs ML 对比

---

# 输入数据可视化脚本

```bash
python scripts/visualize_input_waveforms.py
```

功能：

- 原始 CWRU 波形可视化
- 1024 点模型输入可视化
- 噪声测试集可视化

---

# 提出模型结构

最终提出模型结构：

```text
输入信号
    ↓
CNN 局部特征提取
    ↓
Transformer 全局依赖建模
    ↓
分类输出
```

并在训练阶段引入 NoiseAug 噪声增强策略，以提高低信噪比环境下的鲁棒性。

---

# 作者

GFAlpha

本科毕业设计项目

基于深度学习的轴承故障诊断
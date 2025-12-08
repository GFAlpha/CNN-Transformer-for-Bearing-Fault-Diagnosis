# Bearing_Diagnosis（轴承故障诊断系统）

本项目用于实现基于深度学习的轴承故障诊断，包括数据预处理、模型构建、训练与评估流程。  
使用 **PyTorch** 实现，并支持 **GPU 加速训练（CUDA 13.0）**。

本文档将详细介绍项目结构、各文件与文件夹的用途，以及项目的安装与运行方式。

---

## 📁 项目结构

BEARING_Diagnosis/
├─ configs/
│ └─ default.yaml
│
├─ data/
│ ├─ raw/
│ └─ processed/
│
├─ env/
│
├─ experiments/
│ └─ exp1/
│ └─ logs/
│
├─ logs/
│
├─ models/
│
├─ notebooks/
│ └─ EDA.ipynb
│
├─ scripts/
│ ├─ test_torch.py
│ └─ test_cuda.py
│
├─ src/
│ ├─ init.py
│ ├─ data.py
│ ├─ eval.py
│ ├─ model.py
│ ├─ train.py
│
├─ .gitignore
├─ README.md
└─ requirements.txt

---

## 📂 各文件夹与文件说明

### ### 1. `configs/`
存放模型训练与实验用的配置文件（YAML）。

- **default.yaml**  
  - 保存训练参数（batch size / lr / epochs 等）
  - 保存模型结构超参（层数、隐藏维度等）
  - 保存数据路径  
  可在此文件中切换不同模型或超参组合。

---

### ### 2. `data/`
存放项目所有数据。

#### `data/raw/`
- 原始数据  
- 推荐放 CWRU、MFPT 等数据集的原始下载文件  
- **不会上传到 GitHub（已添加到 .gitignore）**

#### `data/processed/`
- 存放预处理完的数据，如：  
  - 分段后的数据  
  - FFT 数据  
  - npy/tensor 格式数据  
- 可在训练中直接加载，提高速度。

---

### ### 3. `env/`
Python 虚拟环境目录。  
用于隔离依赖，不会上传到 GitHub（.gitignore 已排除）。

---

### ### 4. `experiments/`
用于保存不同实验（exp1, exp2, …）的运行结果和日志。

#### `experiments/exp1/logs/`
- TensorBoard 日志文件  
- 训练时 loss/accuracy 曲线  
- 可视化训练过程

未来可以加入：

- `metrics.json`
- `config.yaml`
- `results/`

---

### ### 5. `logs/`
通用日志文件目录。  
存放训练过程中的打印日志或系统运行信息。

---

### ### 6. `models/`
用于保存训练得到的模型权重（.pth/.pt 文件）。  
大文件不会上传至 GitHub。

---

### ### 7. `notebooks/`
存放 Jupyter Notebook，主要用于：

- 数据可视化  
- EDA（探索性数据分析）  
- 原型调试  
- 绘制波形图、FFT 图、特征分布图

当前包含：

- **EDA.ipynb**

---

### ### 8. `scripts/`
存放可直接运行的脚本，如：

- **test_torch.py**  
  检查 PyTorch 是否安装成功

- **test_cuda.py**  
  检查 CUDA 是否可用、GPU 是否被识别

未来可加入：

- `download_data.py`
- `run_training.ps1`（一键训练）

---

### ### 9. `src/`（核心代码目录）
所有项目的核心代码均放在此目录。

#### `data.py`
- Dataset 类定义  
- 数据加载  
- 滑窗切片 / FFT 等预处理逻辑  
- DataLoader 构建

#### `model.py`
- 深度学习模型定义（如 CNN / LSTM / Transformer）  
- 可以在这里扩展不同模型用于对比

#### `train.py`
- 训练主循环  
- 日志记录  
- 训练过程可视化数据生成  
- 模型保存  
- 配置加载（YAML）

常用命令：
bash
python src/train.py
eval.py
加载训练好的权重
在测试集上评估性能
输出 Accuracy / F1 Score / Confusion Matrix
保存评估结果
__init__.py
标记 src 为 Python 包，使其可被 import。


### ### 10. `.gitignore`

用于忽略以下内容：

虚拟环境 env/

数据集 data/raw/

模型权重 models/

日志 logs/

缓存文件、系统文件

保持 GitHub 仓库干净。

### ### 11. `requirements.txt`
项目依赖记录文件
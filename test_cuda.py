# test_cuda.py
import time
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check nvidia-smi and driver / wheel compatibility.")

# 基本张量搬运测试
a = torch.randn(1024, 1024)
t0 = time.time()
a_gpu = a.cuda()
t1 = time.time()
# 做一次简单运算并返回到 CPU
res = (a_gpu @ a_gpu).sum()
res_cpu = res.cpu()
t2 = time.time()

print("transfer to gpu time: {:.4f} s".format(t1-t0))
print("matmul+sum time (on gpu): {:.4f} s".format(t2-t1))
print("result (cpu):", res_cpu.item())

# 更深入：简单模型单步训练
from torch import nn, optim
model = nn.Linear(1024, 10).cuda()
opt = optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(8, 1024).cuda()
y = torch.randint(0, 10, (8,)).cuda()
criterion = nn.CrossEntropyLoss()
opt.zero_grad()
out = model(x)
loss = criterion(out, y)
loss.backward()
opt.step()
print("single training step loss:", loss.item())

我们后面的 Tensor 默认采用：
[B,L,d]

其中：
B：batch size
L：序列长度
d：特征维度

比如：

X∈R
2×4×8

表示：

2 个句子,每个句子 4 个 token,每个 token 8 维

项目的实现进展、验证状态和后续事项见 [WORKLOG.md](WORKLOG.md)。

代码实现说明(模块职责、函数实现、张量形状约定)见
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。

每次训练的完整参数与结果记录见 [TRAINING_LOG.md](TRAINING_LOG.md)。

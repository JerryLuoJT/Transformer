# Transformer 项目工作日志

本文件记录项目已经完成的工作、验证状态和下一步事项。后续每次开发都在“更新记录”顶部追加一条带日期的记录。

## 截至 2026-09-11 的实现

### 数据约定与说明

- 约定张量布局为 `[B, L, d]`：批量大小、序列长度、特征维度。
- `readme.md` 中记录了该约定及一个 `2 × 4 × 8` 的示例。

### 注意力模块

- `modulus/attention/attention.py`
  - 实现因果掩码 `create_causal_mask`。
  - 编写单头 Q/K/V 投影原型 `QKVProjection`。
  - 实现多头注意力 `MultiHeadAttention`，包含 Q/K/V/O 线性映射、多头拆分与合并、缩放点积、掩码和 softmax。
  - 支持通过 `context` 参数表达解码器中的交叉注意力。

### Transformer 基础组件

- `modulus/attention/position.py`
  - 实现正弦/余弦位置编码 `PositionalEncoding`。
  - 位置编码通过 buffer 保存，不参与训练。
- `modulus/attention/FFN.py`
  - 实现两层前馈网络 `FeedForward`，中间使用 ReLU。

### 编码器与解码器

- `modulus/attention/Encoder.py`
  - 实现 `EncoderLayer`：自注意力、残差连接、LayerNorm、Dropout 和前馈网络。
  - 实现 `Encoder`：词嵌入、位置编码和多层编码器堆叠。
- `modulus/attention/Decoder`
  - 实现 `DecoderLayer`：带掩码的自注意力、编码器—解码器交叉注意力、前馈网络、残差连接、LayerNorm 和 Dropout。
  - 实现 `Decoder`：词嵌入、位置编码和多层解码器堆叠。

### 完整模型与测试

- `modulus/attention/Transformer.py`
  - 组合 Encoder、Decoder 和输出线性层。
  - `forward` 接收源序列、目标序列及相应掩码，返回目标词表 logits。
- `modulus/attention/test.py`
  - 固定随机种子。
  - 对多头自注意力和因果掩码进行一次基础运行测试。

## 当前验证状态

2026-09-11 在本机使用 Python 3 运行：

- `python3 test.py`：通过。
  - 输出形状：`[2, 4, 8]`。
  - 注意力权重形状：`[2, 2, 4, 4]`。
  - 打印结果显示未来位置被因果掩码置零。
- 导入完整模型：未通过。
  - `Transformer.py` 使用 `from Decoder import Decoder`，但当前文件名是 `Decoder`，没有 `.py` 后缀，因此触发 `ModuleNotFoundError`。

## 已知待办

- [ ] 将 `Decoder` 调整为可导入的 Python 模块，并恢复完整模型导入。
- [ ] 修正 `EncoderLayer` 向 `MultiHeadAttention` 传递 mask 的方式；当前位置参数会被当作 `context`。
- [ ] 让交叉注意力分别使用查询长度和 context 长度，支持源序列与目标序列长度不同。
- [ ] 修复并验证 `QKVProjection` 原型中在定义 `score` 前读取它、以及缩放因子类型的问题。
- [ ] 增加完整 Encoder–Decoder 前向传播测试、不同长度测试和 mask 测试。
- [ ] 增加项目依赖与运行方式说明。
- [ ] 将源码整理成标准 Python package，并统一模块/文件命名。

## 更新记录

### 2026-09-11

- 盘点并记录现有实现。
- 运行基础注意力测试并记录结果。
- 检查完整模型导入并记录当前阻塞问题。
- 添加 `.gitignore`，排除 Python 缓存、虚拟环境、模型权重和实验输出。
- 初始化本地 Git 仓库，默认分支为 `main`。
- 创建 GitHub 私有仓库 [`JerryLuoJT/Transformer`](https://github.com/JerryLuoJT/Transformer)，并推送 `main` 分支。

### 2026-08-21 至 2026-08-28（根据现有文件修改时间整理）

- 建立张量维度约定。
- 编写位置编码、前馈网络、注意力、Encoder、Decoder、完整 Transformer 组合及基础测试。

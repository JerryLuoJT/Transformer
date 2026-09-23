# 代码实现文档

本文档描述仓库中每个文件的**实现方式**:类与函数怎么组织、参数怎么传递、张量形状怎么变化。
不涉及注意力/Transformer 的原理,只回答「代码是怎么写出来的」。

## 1. 目录结构

```text
Transformer/
├── modulus/
│   ├── __init__.py                 # 标记 modulus 为包
│   └── attention/
│       ├── __init__.py             # 标记 attention 为包
│       ├── attention.py            # 因果掩码、单头原型、MultiHeadAttention
│       ├── position.py             # PositionalEncoding 正弦位置编码
│       ├── FFN.py                  # FeedForward 两层前馈网络
│       ├── Encoder.py              # EncoderLayer / Encoder
│       ├── Decoder.py              # DecoderLayer / Decoder
│       ├── Transformer.py          # Encoder + Decoder + 输出层组合
│       └── test.py                 # 注意力与因果掩码的最小验证
├── train/
│   ├── __init__.py
│   ├── data.py                     # 语料读取、字符级词表、Dataset、批处理
│   ├── train.py                    # 训练/验证/贪心解码三个过程函数
│   └── main.py                     # 训练入口与全部超参数
├── data/cmn-eng/
│   ├── cmn.txt                     # Tatoeba 中英句对(32,028 行)
│   └── cmn-eng.zip                 # 原始压缩包
├── docs/IMPLEMENTATION.md          # 本文档
├── TRAINING_LOG.md                 # 训练日志(每次运行自动追加)
├── outputs/                        # 训练产物(checkpoint 被 .gitignore 排除)
├── readme.md                       # 数据约定 [B, L, d]
└── WORKLOG.md                      # 工作日志
```

所有模块统一使用**相对导入**(`from .attention import ...`),因此必须用
`python3 -m 包.模块` 的方式从仓库根目录启动:

```bash
python3 -m train.main               # 训练
python3 -m modulus.attention.test   # 注意力测试
```

直接 `python3 train/main.py` 或 `python3 test.py` 会因包路径缺失/无父包上下文而报
`ModuleNotFoundError` / `ImportError`。

## 2. 训练数据流

```text
cmn.txt(英语\t中文\t版权)
  └─ load_pairs       读文件、按 \t 切分、取前两列、洗牌、截断
       └─ CharVocab ×2 从采样句对统计字符,构建 src/tgt 两个独立词表
            └─ TranslationDataset  每个句对编码成 (src_ids, tgt_ids)
                 └─ collate_fn     变长序列 pad 成 [B, L]
                      └─ train_one_epoch  teacher forcing 前向 + 反向
                           └─ evaluate        验证集损失
                           └─ greedy_decode   训练后贪心生成一段译文
```

## 3. 数据模块 `train/data.py`

**固定 token 索引**(`train/data.py:4`)

| id | 符号 |
|---|---|
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<eos>` |
| 3 | `<unk>` |

`SPECIALS` 列表与这些 id 一一对应,普通字符的 id 从 4 开始。

### `load_pairs(path, max_pairs=None, seed=42)`

- 逐行读取 `cmn.txt`,按 `\t` 切分,`len(parts) >= 2` 时取第 0、1 列并 `strip()`。
- 用 `random.Random(seed)` 洗牌(不影响全局随机状态),可选截断前 `max_pairs` 个。
- 返回 `(英语, 中文)` 元组列表。

### `CharVocab(sentences)`

- 构造:对传入的所有句子取字符集合并排序,`itos = SPECIALS + chars`,
  `stoi` 为反向字典;`__len__` 返回词表大小。
- `encode(text, bos=False, eos=False)`:逐字符查表,查不到用 `UNK_IDX`;
  可选在开头插 `<bos>`、结尾插 `<eos>`。

### `TranslationDataset(pairs, src_vocab, tgt_vocab)`

- `__init__` 中一次性把所有句对编码好,存入 `self.samples`:
  - 源:`src_vocab.encode(src, eos=True)`(句尾一个 `<eos>`);
  - 目标:`tgt_vocab.encode(tgt, bos=True, eos=True)`(首尾各一个)。
- `__getitem__` 直接返回 `(src_ids, tgt_ids)` 两个 list。

### `collate_fn(batch)`

- 用 `zip(*batch)` 把 batch 内的源、目标分开;
- 分别过 `torch.nn.utils.rnn.pad_sequence(batch_first=True, padding_value=PAD_IDX)`,
  补齐到 batch 内最长长度,返回两个 `[B, L]` 的 LongTensor。

## 4. 训练过程 `train/train.py`

### `create_padding_mask(tokens, pad_idx)`

- 输入 `tokens: [B, L]`,输出 `(tokens != pad_idx)` 的 bool 张量,
  连续两次 `unsqueeze(1).unsqueeze(2)` 变成 `[B, 1, 1, L]`。
- 这个形状是为了和注意力分数 `[B, h, Lq, Lk]` 广播:batch/头/查询维为 1,
  只约束 key 维。

### `train_one_epoch(model, dataloader, optimizer, criterion, device, src_pad_idx, tgt_pad_idx)`

- 每个 batch 的实现顺序:
  1. `src`、`tgt` 移到 device;
  2. teacher forcing 切片:`decoder_input = tgt[:, :-1]`,`labels = tgt[:, 1:]`,
     即用 `<bos> + 前 n-1 个词` 预测 `后 n-1 个词 + <eos>`;
  3. `src_mask`、`tgt_mask` 都只是 padding 掩码(因果掩码在模型内部合成);
  4. `optimizer.zero_grad()` → `logits = model(src, decoder_input, src_mask, tgt_mask)`
     → `loss = criterion(logits.reshape(-1, V), labels.reshape(-1))` → `backward()` → `step()`;
     `backward()` 与 `step()` 之间插入
     `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)`
     (默认 `clip_norm=1.0`)做梯度裁剪;
  5. 累加 `loss.item()`,最后返回 `total_loss / len(dataloader)`。
- 注意损失处用的是 `.reshape()` 而不是 `.view()`:`logits` 经
  `output_layer` 后内存不连续,`.view()` 会直接报错。

### `evaluate(...)`

- 结构与 `train_one_epoch` 相同,但包在 `@torch.no_grad()` 里、`model.eval()`、
  没有 `optimizer` 步骤,只返回验证集平均损失。

### `greedy_decode(model, src, src_mask, tgt_vocab, max_len, device)`

- 实现上**拆开调用模型的三个子模块**,而不是走 `model.forward`:
  1. `encoder_output = model.encoder(src, src_mask)` 只编码一次;
  2. `decoder_input = [[BOS_IDX]]` 起步;
  3. 循环最多 `max_len` 步:用 `create_padding_mask` 生成当前目标掩码,
     `model.decoder(...)` 后接 `model.output_layer(...)`,
     取 `logits[:, -1, :].argmax(dim=-1)` 得到下一个 token;
  4. 遇到 `EOS_IDX` 提前结束,否则把 token 用 `torch.cat` 追加到 `decoder_input`。
- 返回不含 `<bos>/<eos>` 的 token id 列表。

### `beam_search_decode(model, src, src_mask, max_len, device, beam_width=4)`

- 与 `greedy_decode` 的区别:每一步对当前所有候选做
  `log_softmax` + `topk(beam_width)`,展开出 `beam_width` 条新候选,
  按累计 log 概率排序后只保留前 `beam_width` 条;
- 遇到 `EOS_IDX` 的候选移入 `finished` 列表,不再继续展开;
  `finished` 满 `beam_width` 条时提前结束;
- 最后从 `finished`(否则 `beams`)里取累计分数最高的一条,去掉末尾的
  `<eos>` 与开头的 `<bos>` 后返回 id 列表。

### `sentence_bleu(ref_tokens, hyp_tokens, max_n=4)`

- 字符级 BLEU:统计 1~4 gram 的命中比例(参考 n-gram 与候选 n-gram 的交集 /
  候选 n-gram 总数),过滤掉为 0 的精度后取几何平均,再乘长度惩罚
  (`min(1, exp(1 - ref_len/hyp_len))`),返回 0~100。
- `eval_bleu_sample(model, dataset, device, n_samples=50, beam_width=4, max_len=30)`:
  在数据集前 `n_samples` 条上做束搜索解码,返回平均 BLEU。

## 5. 训练入口 `train/main.py`

### 可调配置(文件顶部常量)

| 常量 | 当前值 | 含义 |
|---|---|---|
| `MAX_PAIRS` | 30000 | 采样句对数 |
| `MAX_LEN` | 40 | 单侧超过 40 字符丢弃 |
| `VAL_RATIO` | 0.1 | 验证集比例 |
| `BATCH_SIZE` | 32 | 批大小 |
| `EPOCHS` | 20 | 训练轮数 |
| `D_MODEL` | 128 | 隐层维度 |
| `NUM_HEADS` | 4 | 注意力头数 |
| `D_FF` | 256 | 前馈网络中间维度 |
| `NUM_LAYERS` | 2 | Encoder/Decoder 层数 |
| `DROPOUT` | 0.3 | Dropout 概率 |
| `LEARNING_RATE` | 1e-4 | Adam 学习率 |
| `LABEL_SMOOTHING` | 0.1 | 标签平滑系数 |
| `WEIGHT_DECAY` | 1e-4 | Adam 权重衰减 |
| `GRAD_CLIP` | 1.0 | 梯度裁剪阈值 |
| `LR_FACTOR` / `LR_PATIENCE` | 0.5 / 3 | ReduceLROnPlateau 衰减系数与耐心 |
| `EARLY_STOP_PATIENCE` | 8 | 验证损失连续不改善多少轮后早停 |
| `BEAM_WIDTH` | 4 | 束搜索宽度 |
| `BLEU_SAMPLE` | 50 | BLEU 评测采样条数 |

### `build_dataloaders()`

按顺序执行:读全量句对 → 按 `MAX_LEN` 过滤 → 截取 `MAX_PAIRS` 个 →
用采样后的句对分别构建 `src_vocab` / `tgt_vocab` → 构建 `TranslationDataset`
→ `random_split` 切 90/10(seed 42)→ 返回两个 `DataLoader`(都带 `collate_fn`)。

### `main()`

- `device` 优先 cuda,否则 cpu;
- 用两个词表大小实例化 `Transformer`;
- 损失 `CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)`,
  优化器 `Adam(..., weight_decay=1e-4)`,外加 `ReduceLROnPlateau`
  (mode=min、factor=0.5、patience=3)按验证损失调学习率;
- 每轮依次调 `train_one_epoch`、`evaluate`,打印两个损失与当前学习率;
- 验证损失创新低时把 `state_dict` 存到 `outputs/transformer_zh_en_best.pt`;
  连续 `EARLY_STOP_PATIENCE` 轮未改善则提前结束;
- 训练结束后 `load_state_dict` 恢复最优 checkpoint,从验证集取第 0 条做
  束搜索演示,再用 `eval_bleu_sample` 打印字符级 BLEU。
- `log_run(...)` 会把本次运行的全部超参数、best epoch/loss、BLEU 与 demo
  翻译追加到仓库根目录的 `TRAINING_LOG.md`。

## 6. 模型模块 `modulus/attention/`

### `attention.py`

- 模块顶部遗留一个未使用的常量 `d_model = 8`。
- `create_causal_mask(seq_len, device)`:返回 `torch.tril(torch.ones(...))`,
  形状 `[seq_len, seq_len]` 的下三角 0/1 掩码。
- `QKVProjection`:单头注意力原型。三个无偏置 `Linear(d_model, d_model)`;
  forward 顺序:投影 Q/K/V → `K.transpose(-2,-1)` → `Q @ K^T` → 除以
  `math.sqrt(dk)` → 建下三角掩码、`masked_fill(mask==0, -inf)` → softmax 最后一维 → 乘 V。
  返回 `(output, attention_weights)`。
- `MultiHeadAttention(d_model, num_heads)`:
  - `__init__` 先 `assert d_model % num_heads == 0`,`d_k = d_model // num_heads`;
    持有 `W_q/W_k/W_v/W_o` 四个无偏置 `Linear`。
  - `forward(X, context=None, mask=None)` 六步:
    1. `Q = W_q(X)`;`context` 为 `None` 时 K/V 由 X 投影(自注意力),
       否则由 `context` 投影(交叉注意力);
    2. 取 `Lk = K.size(1)`,Q 用查询长度 L、K/V 用 Lk,
       分别 `view(B, L/Lk, num_heads, d_k)`;
    3. `transpose(1, 2)` 换成 `[B, h, L, d_k]`;
    4. `scores = Q @ K^T / math.sqrt(d_k)`,若传了 `mask` 则
       `masked_fill(mask == 0, -inf)`,再对最后一维 softmax;
    5. 乘 V 后 `transpose(1,2)` 回 `[B, L, h, d_k]`,
       `contiguous().view(B, L, d_model)` 合并多头;
    6. 过 `W_o`。
  - 返回 `(output, attention_weights)`,权重形状 `[B, h, Lq, Lk]`。

### `position.py`

- `PositionalEncoding(d_model, max_len=5000)`:
  - `__init__` 预计算 `pe: [max_len, d_model]`,偶数维填 sin、奇数维填 cos,
    最后 `unsqueeze(0)` 成 `[1, max_len, d_model]` 并
    `register_buffer("pe", pe)`(不参与反向传播);
  - `forward(X)`:取 `X.size(1)` 作为长度,把 `pe[:, :L]` 直接加到输入上。
- 注意:长度超过 `max_len` 时会被静默截断,不报错。

### `FFN.py`

- `FeedForward(d_model, d_ff)`:两个 `Linear`,`d_model -> d_ff -> GELU -> d_model`。

### `Encoder.py`

- `EncoderLayer(d_model, d_ff, num_heads, dropout=0.1)`:
  持有 `self_attention`、`ffn`、两个 LayerNorm、两个 Dropout;
  forward:`self_attention(X, mask=mask)`(mask 必须用关键字传)→
  残差 + norm1 + dropout1 → `ffn` → 残差 + norm2 + dropout2;
  返回 `(X, attn_weight)`。
- `Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1)`:
  持有 `embedding`、`positional_encoding`、`ModuleList` 的 N 个 `EncoderLayer`;
  forward:`embedding * sqrt(d_model) → positional_encoding → dropout →`
  逐层 `X = layer(X, mask)[0]`,
  即丢弃每层的注意力权重,只保留隐状态。

### `Decoder.py`

- `DecoderLayer(d_model, d_ff, num_heads, dropout=0.1)`:
  持有 `self_attention`、`cross_attention`、`ffn`、三个 LayerNorm、三个 Dropout;
  forward 接收 `(X, encoder_output, self_mask=None, cross_mask=None)`:
  1. 自注意力之前,先用 `X.size(1)` 构造下三角 bool 因果掩码
     `[L, L]`,`self_mask` 存在时做 `self_mask & causal_mask`(广播成
     `[B, 1, L, L]`),不存在时直接用因果掩码;
  2. `self_attention(X, mask=self_mask)` → 残差 + norm1;
  3. `cross_attention(X, context=encoder_output, mask=cross_mask)` → 残差 + norm2;
  4. `ffn` → 残差 + norm3;只返回 `X`,不返回注意力权重。
- `Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1)`:
  结构与 Encoder 对应(embedding + positional_encoding + ModuleList);
  forward:`embedding * sqrt(d_model) → positional_encoding → dropout →`
  `X = layer(X, encoder_output, self_mask, cross_mask)`,
  `encoder_output` 在每一层复用、不参与迭代。

### `Transformer.py`

- `Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1)`:
  持有 `self.encoder`、`self.decoder`、`self.output_layer = Linear(d_model, tgt_vocab_size)`。
- `forward(src, tgt, src_mask=None, tgt_mask=None)` 的参数顺序是固定的:
  1. `encoder_output = self.encoder(src, src_mask)`;
  2. `self.decoder(tgt, encoder_output, tgt_mask, src_mask)`
     ——注意 Decoder 的 `self_mask` 收到的是 `tgt_mask`,`cross_mask` 收到的是 `src_mask`;
  3. `self.output_layer(...)` 输出 `[B, Lt, tgt_vocab_size]` 的 logits。

### `test.py`

- 固定 `torch.manual_seed(42)`;构造 `[2, 4, 8]` 随机输入、
  `d_model=8, num_heads=2` 的 MHA、长度 4 的因果掩码,打印输出形状、
  权重形状和第一个头第 0 个查询的权重行(用于人工确认未来位置被置零)。
- 作为模块运行:`python3 -m modulus.attention.test`。

## 7. 关键约定速查

- 张量布局:`[B, L, d]`(batch、序列长、特征维);token id 张量 `[B, L]` 为 LongTensor。
- 掩码形状:padding 掩码统一 `[B, 1, 1, L]`;因果掩码 `[L, L]`;
  注意力内部以 `mask == 0` 的位置填 `-inf`。
- 掩码职责划分:padding 掩码由训练/推理代码(`create_padding_mask`)构建;
  因果掩码由 `DecoderLayer` 自动合成,调用方不需要传。
- 目标序列编码约定:源句尾加一个 `<eos>`,目标句首尾各加 `<bos>/<eos>`。
- 运行入口:一律 `python3 -m 模块` 从仓库根目录启动。

## 8. 实现注意点(与原理无关的事实)

- `train_one_epoch` 中 logits 必须用 `.reshape()`(`.view()` 要求连续内存)。
- `attention.py` 顶部 `d_model = 8` 是遗留常量,未被任何代码引用。
- `EncoderLayer` 返回注意力权重但 `Encoder` 丢弃;`DecoderLayer` 不返回权重,
  因此当前代码无法直接做注意力可视化(需要改返回值)。
- 位置编码对超过 `max_len` 的序列静默截断。
- src 与 tgt 使用**两个独立词表**,英文源词表约几十个字符、中文目标词表上千个字符。
- 已实现贪心与束搜索两种解码、字符级 BLEU;尚未实现:checkpoint 断点续训、
  学习率 warmup、标签平滑之外的更多正则(如 R-Drop)。

## 9. 数据来源

- `data/cmn-eng/cmn.txt`:Tatoeba 中英句对,来源
  <https://www.manythings.org/anki/>,许可 CC-BY 2.0(每行第三列含署名信息)。
- 共 32,028 句对,格式 `英语 \t 中文 \t 版权信息`。

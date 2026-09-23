# 训练日志

记录每次完整训练的:超参数、best epoch、best val loss、字符级 BLEU 与一条 demo 翻译。
每次 `python3 -m train.main` 结束后,`train/main.py` 会自动在本文件末尾追加一节。

## 汇总

| Run | 日期 | 数据 | DROPOUT | best epoch | best val | BLEU | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | 2026-09-22 | 3,000 句(MAX_LEN=30) | 0.1 | —(保存末轮) | 4.5645(末轮) | — | 首次跑通,贪心解码 |
| 2 | 2026-09-23 | 30,000 上限(过滤后约 17,170) | 0.1 | —(保存末轮) | 2.6785(末轮) | — | 明显过拟合 |
| 3 | 2026-09-23 | 30,000 上限(过滤后约 17,170) | 0.3 | 50 | 3.5664 | 25.20 | 正则化后过拟合消除;beam 输出含 `<eos>` bug |

## Run 1 — 2026-09-22

- 数据:`MAX_PAIRS=3000`,`MAX_LEN=30`,`VAL_RATIO=0.1`,`BATCH_SIZE=32`
- 模型:`D_MODEL=128`,`NUM_HEADS=4`,`D_FF=256`,`NUM_LAYERS=2`,`DROPOUT=0.1`
- 训练:`EPOCHS=5`,`LEARNING_RATE=1e-4`(Adam,无 label smoothing / weight decay / 梯度裁剪 / 学习率调度)
- 解码:贪心,无 BLEU 评测
- 结果:保存末轮,`epoch 05 | train=4.5512 / val=4.5645`
- demo:
  - src : She is at work right now.
  - ref : 她现在正在工作。
  - pred: 我們不是我的。
- 备注:首个可运行版本,验证整条数据管线;demo 内容不对但已是合法中文,符合 3,000 句 × 5 轮的预期。

## Run 2 — 2026-09-23

- 数据:`MAX_PAIRS=30000`(MAX_LEN=30 过滤后实际约 17,170 句),`VAL_RATIO=0.1`,`BATCH_SIZE=32`
- 模型:`D_MODEL=128`,`NUM_HEADS=4`,`D_FF=256`,`NUM_LAYERS=2`,`DROPOUT=0.1`
- 训练:`EPOCHS=50`,`LEARNING_RATE=1e-4`(Adam,无正则化 / 调度 / 早停)
- 解码:贪心,无 BLEU 评测
- 结果:保存末轮,`epoch 50 | train=1.2151 / val=2.6785`
- demo:
  - src : My mother is a very good cook.
  - ref : 我媽媽的廚藝很好。
  - pred: 我的媽媽是一個好石的廚師。
- 备注:train/val 差距一倍以上,典型过拟合;且 Encoder/Decoder 的 embedding dropout 当时未生效。
  由此触发后续正则化优化。

## Run 3 — 2026-09-23

- 数据:`MAX_PAIRS=30000`(过滤后约 17,170 句),`MAX_LEN=30`,`VAL_RATIO=0.1`,`BATCH_SIZE=32`
- 模型:`D_MODEL=128`,`NUM_HEADS=4`,`D_FF=256`,`NUM_LAYERS=2`,`DROPOUT=0.3`
- 训练:`EPOCHS=50`,`LEARNING_RATE=1e-4`,`LABEL_SMOOTHING=0.1`,`WEIGHT_DECAY=1e-4`,
  `GRAD_CLIP=1.0`,`ReduceLROnPlateau(0.5/3)`,`EARLY_STOP_PATIENCE=8`
- 解码:`BEAM_WIDTH=4`,`BLEU_SAMPLE=50`
- 结果:`best_epoch=50`,`best_val=3.5664`,末轮 `train=3.2620 / val=3.5664`,`BLEU=25.20`
- demo:
  - src : My mother is a very good cook.
  - ref : 我媽媽的廚藝很好。
  - pred: 我爸爸爸爸很好。`<eos>`
- 备注:过拟合基本消除(train/val 差距约 0.3);beam 输出末尾残留 `<eos>`(bug,已修复);
  字符级 BLEU 受数据简繁体混杂影响被系统性压低。

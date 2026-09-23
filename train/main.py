"""端到端训练入口:数据 -> 词表 -> DataLoader -> 模型 -> 训练 -> 保存。

用法(必须在仓库根目录运行):
    python3 -m train.main
"""

from pathlib import Path
import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from modulus.attention.Transformer import Transformer
from train.data import (
    PAD_IDX,
    BOS_IDX,
    CharVocab,
    TranslationDataset,
    collate_fn,
    load_pairs,
)
from train.train import beam_search_decode, evaluate, eval_bleu_sample, train_one_epoch

# ---------------------- 可调配置 ----------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cmn-eng" / "cmn.txt"
MAX_PAIRS = 30000      # 采样句对数(教学用小样本)
MAX_LEN = 40          # 单侧超过 40 个字符的句对丢弃
VAL_RATIO = 0.1       # 验证集比例
BATCH_SIZE = 32
EPOCHS = 50

D_MODEL = 512
NUM_HEADS = 8
D_FF = 1024
NUM_LAYERS = 6
DROPOUT = 0.3
LEARNING_RATE = 1e-4
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 8
BEAM_WIDTH = 4
BLEU_SAMPLE = 50

LOG_PATH = Path(__file__).resolve().parent.parent / "TRAINING_LOG.md"
CKPT_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "transformer_zh_en_best.pt"
)


def build_dataloaders():
    pairs = load_pairs(DATA_PATH)
    pairs = [
        (src, tgt)
        for src, tgt in pairs
        if len(src) <= MAX_LEN and len(tgt) <= MAX_LEN
    ][:MAX_PAIRS]

    src_vocab = CharVocab([src for src, _ in pairs])
    tgt_vocab = CharVocab([tgt for _, tgt in pairs])
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab)

    n_val = int(len(dataset) * VAL_RATIO)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    return src_vocab, tgt_vocab, train_loader, val_loader, val_ds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    src_vocab, tgt_vocab, train_loader, val_loader, val_ds = build_dataloaders()
    print(
        f"src vocab: {len(src_vocab)} | tgt vocab: {len(tgt_vocab)} | "
        f"train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)}"
    )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    CKPT_PATH.parent.mkdir(exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    patience_left = EARLY_STOP_PATIENCE
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PAD_IDX, PAD_IDX, clip_norm=GRAD_CLIP,
        )
        val_loss = evaluate(model, val_loader, criterion, device, PAD_IDX, PAD_IDX)
        scheduler.step(val_loss)
        print(
            f"epoch {epoch:02d} | train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), CKPT_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"early stop: val_loss 连续 {EARLY_STOP_PATIENCE} 轮未改善")
                break

    model.load_state_dict(torch.load(CKPT_PATH))
    print(f"best epoch: {best_epoch} | best val_loss: {best_val:.4f}")
    print(f"checkpoint saved: {CKPT_PATH}")

    demo_src_ids, demo_tgt_ids = val_ds[0]
    demo_src = torch.tensor(demo_src_ids).unsqueeze(0).to(device)
    src_mask = (demo_src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        out_ids = beam_search_decode(
            model, demo_src, src_mask, MAX_LEN, device, BEAM_WIDTH
        )
    src_text = "".join(src_vocab.itos[i] for i in demo_src_ids)
    ref_text = "".join(tgt_vocab.itos[i] for i in demo_tgt_ids[1:-1])
    pred_text = "".join(tgt_vocab.itos[i] for i in out_ids)
    print("demo src :", src_text)
    print("demo ref :", ref_text)
    print("demo pred:", pred_text)
    bleu = eval_bleu_sample(
        model, val_ds, device, BLEU_SAMPLE, BEAM_WIDTH, MAX_LEN
    )
    print(f"val BLEU (char-level, {BLEU_SAMPLE} samples, beam={BEAM_WIDTH}): {bleu:.2f}")
    log_run(
        src_text, ref_text, pred_text,
        best_epoch, best_val, train_loss, val_loss, bleu,
    )
    print(f"training log appended: {LOG_PATH}")


def log_run(src_text, ref_text, pred_text, best_epoch, best_val, final_train, final_val, bleu):
    """把本次运行的参数与结果追加到 TRAINING_LOG.md。"""
    run_no = 1
    if LOG_PATH.exists():
        run_no = LOG_PATH.read_text(encoding="utf-8").count("## Run") + 1
    entry = f"""
## Run {run_no} — {datetime.date.today()}

- 数据:`MAX_PAIRS={MAX_PAIRS}`,`MAX_LEN={MAX_LEN}`,`VAL_RATIO={VAL_RATIO}`,`BATCH_SIZE={BATCH_SIZE}`
- 模型:`D_MODEL={D_MODEL}`,`NUM_HEADS={NUM_HEADS}`,`D_FF={D_FF}`,`NUM_LAYERS={NUM_LAYERS}`,`DROPOUT={DROPOUT}`
- 训练:`EPOCHS={EPOCHS}`,`LEARNING_RATE={LEARNING_RATE:g}`,`LABEL_SMOOTHING={LABEL_SMOOTHING}`,
  `WEIGHT_DECAY={WEIGHT_DECAY}`,`GRAD_CLIP={GRAD_CLIP}`,`LR_FACTOR={LR_FACTOR}`,
  `LR_PATIENCE={LR_PATIENCE}`,`EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}`
- 解码:`BEAM_WIDTH={BEAM_WIDTH}`,`BLEU_SAMPLE={BLEU_SAMPLE}`
- 结果:`best_epoch={best_epoch}`,`best_val={best_val:.4f}`,末轮 `train={final_train:.4f} / val={final_val:.4f}`,
  `BLEU={bleu:.2f}`
- demo:
  - src : {src_text}
  - ref : {ref_text}
  - pred: {pred_text}
"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


if __name__ == "__main__":
    main()

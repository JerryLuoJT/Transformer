"""中英平行语料的加载、字符级词表与批处理。"""

import random

import torch
from torch.utils.data import Dataset

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_pairs(path, max_pairs=None, seed=42):
    """读取 cmn.txt。每行格式:英语\t中文\t版权信息,只保留前两列。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


class CharVocab:
    """字符级词表:每个中文字符/英文字母是一个 token。"""

    def __init__(self, sentences):
        chars = sorted({c for s in sentences for c in s})
        self.itos = SPECIALS + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, bos=False, eos=False):
        ids = [self.stoi.get(c, UNK_IDX) for c in text]
        if bos:
            ids = [BOS_IDX] + ids
        if eos:
            ids = ids + [EOS_IDX]
        return ids


class TranslationDataset(Dataset):
    """把句对编码为 id 序列;目标序列首尾带 <bos>/<eos>。"""

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.samples = []
        for src, tgt in pairs:
            self.samples.append(
                (src_vocab.encode(src, eos=True), tgt_vocab.encode(tgt, bos=True, eos=True))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """变长序列补齐成 [B, L],不足的位置填 <pad>。"""
    src_batch, tgt_batch = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s, dtype=torch.long) for s in src_batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    tgt = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tgt_batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    return src, tgt

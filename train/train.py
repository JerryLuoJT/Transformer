import torch
import math
from collections import Counter

from train.data import BOS_IDX, EOS_IDX, PAD_IDX


def create_padding_mask(tokens, pad_idx):
    """
    tokens: [B, L]

    return:
        mask: [B, 1, 1, L]
    """
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)

def train_one_epoch(model,
    dataloader,
    optimizer,
    criterion,
    device,
    src_pad_idx,
    tgt_pad_idx,
    clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for src, tgt in dataloader:
        # move to device
        src = src.to(device)
        tgt = tgt.to(device)
        # prepare input and labels
        # target:
        # <BOS> I love NLP <EOS>
        #
        # decoder_input:
        # <BOS> I love NLP
        #
        # labels:
        # I love NLP <EOS>
        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]
        # create masks
        src_mask = create_padding_mask(src, src_pad_idx)
        tgt_mask = create_padding_mask(decoder_input, tgt_pad_idx)
        # 解码器自注意力的因果掩码已在 DecoderLayer 内自动合成，
        # 训练代码只需传入 padding 掩码。
        # clear gradients
        optimizer.zero_grad()
        # forward
        logits = model(src, decoder_input, src_mask, tgt_mask)
        # loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, src_pad_idx, tgt_pad_idx):
    """验证集平均损失。只做前向,不更新参数。"""
    model.eval()
    total_loss = 0.0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]
        src_mask = create_padding_mask(src, src_pad_idx)
        tgt_mask = create_padding_mask(decoder_input, tgt_pad_idx)
        logits = model(src, decoder_input, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)


def greedy_decode(model, src, src_mask, tgt_vocab, max_len, device):
    """推理:逐 token 贪心生成目标序列。

    与训练不同,推理时未来的目标 token 还不存在,所以每步只把
    已生成的序列交给 decoder,取最后一个位置的 logits 做 argmax;
    DecoderLayer 内的因果掩码保证位置 i 看不到未来。
    """
    model.eval()
    encoder_output = model.encoder(src, src_mask)
    decoder_input = torch.tensor([[BOS_IDX]], device=device)
    generated = []
    for _ in range(max_len):
        tgt_mask = create_padding_mask(decoder_input, PAD_IDX)
        decoder_output = model.decoder(decoder_input, encoder_output, tgt_mask, src_mask)
        logits = model.output_layer(decoder_output)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        if next_token == EOS_IDX:
            break
        generated.append(next_token)
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], device=device)], dim=1
        )
    return generated


def beam_search_decode(model, src, src_mask, max_len, device, beam_width=4):
    """推理:束搜索。每一步保留概率最高的 beam_width 条候选,
    而不是只留 argmax 的一条,可修正贪心解码一步走错的连锁错误。
    """
    model.eval()
    encoder_output = model.encoder(src, src_mask)
    beams = [([BOS_IDX], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for ids, score in beams:
            decoder_input = torch.tensor([ids], device=device)
            tgt_mask = create_padding_mask(decoder_input, PAD_IDX)
            decoder_output = model.decoder(decoder_input, encoder_output, tgt_mask, src_mask)
            logits = model.output_layer(decoder_output)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(log_probs, beam_width, dim=-1)
            for prob, tok in zip(topk_probs[0], topk_ids[0]):
                new_ids = ids + [tok.item()]
                new_score = score + prob.item()
                if tok.item() == EOS_IDX:
                    finished.append((new_ids, new_score))
                else:
                    candidates.append((new_ids, new_score))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_width]
        if len(finished) >= beam_width:
            break
    if finished:
        finished.sort(key=lambda item: item[1], reverse=True)
        best = finished[0][0]
    else:
        best = beams[0][0]
    if best and best[-1] == EOS_IDX:
        best = best[:-1]
    return best[1:]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(ref_tokens, hyp_tokens, max_n=4):
    """单句 BLEU(1-4 gram 几何平均 + 长度惩罚),按 token 列表计算。"""
    if not hyp_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        hyp_ngrams = _ngrams(hyp_tokens, n)
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = max(sum(hyp_ngrams.values()), 1)
        precisions.append(overlap / total)
    precisions = [p for p in precisions if p > 0]
    if not precisions:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    brevity = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return brevity * geo_mean * 100


@torch.no_grad()
def eval_bleu_sample(model, dataset, device, n_samples=50, beam_width=4, max_len=30):
    """在数据集前 n_samples 条上做束搜索解码,返回平均字符级 BLEU。"""
    model.eval()
    scores = []
    for idx in range(min(n_samples, len(dataset))):
        src_ids, tgt_ids = dataset[idx]
        src = torch.tensor(src_ids).unsqueeze(0).to(device)
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        hyp = beam_search_decode(model, src, src_mask, max_len, device, beam_width)
        scores.append(sentence_bleu(tgt_ids[1:-1], hyp))
    return sum(scores) / len(scores)



        

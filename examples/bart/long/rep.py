from nltk.tokenize import word_tokenize
from nltk import ngrams
import nltk
import argparse
import torch
from collections import Counter
import math


def read_lines(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(line.rstrip())
    return data


def build_dict(texts):
    t2i = {}
    for text in texts:
        for tok in text:
            if tok not in t2i:
                t2i[tok] = len(t2i)
    print("Build vocab size: {}".format(len(t2i)))
    return t2i
 
def tok_repeat_l(hypo_toks, context_len=16):
    hypo = torch.tensor(hypo_toks).long()
    T = hypo.size(0)
    
    # prev_hypo[t, :] = [y_1, y_2, ... , y_t-1, -1 ,-1, ... , -1]
    prev_hypo = hypo.expand(T, T).masked_fill(torch.ones(T, T).triu().bool(), -1)

    # prev_hypo[t, :] = [-1, ... , -1, y_t-k-1, ..., y_t-1, -1 ,-1, ... , -1]
    prev_hypo = prev_hypo.masked_fill(torch.ones(T, T).tril(-context_len).bool(), -1)

    repeat = (hypo[:, None] == prev_hypo)
    has_repeat = repeat.sum(1).gt(0)
    total_repeat = has_repeat.sum()

    return total_repeat * 1.0 / T 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=str, required=True, help="hypothesis")
    parser.add_argument("--max-context-len", type=int, default=64)
    parser.add_argument("--lower", action="store_true")
    args = parser.parse_args()

    CONTEXT_LENS = [4, 8, 16, 32, 64]
    metrics = {}
    for c_len in CONTEXT_LENS[:int(math.log2(args.max_context_len))]:
        metrics.update({f"tok_repeat_{c_len}": 0.0})

    preds = read_lines(args.h)
    
    tok_preds = [[t.lower() if args.lower else t for t in word_tokenize(x)] for x in preds]

    dictionary = build_dict(tok_preds)

    hyp_ids = []
    for hyp in tok_preds:
        hyp_id = []
        for tok in hyp:
            hyp_id.append(dictionary[tok])
        
        
        for c_len in CONTEXT_LENS[:int(math.log2(args.max_context_len))]:
            metrics[f"tok_repeat_{c_len}"] += tok_repeat_l(hyp_id, context_len=c_len)
        

    for k, v in metrics.items():
        metrics[k] = v * 1.0 / len(tok_preds)

    print(metrics)

main()
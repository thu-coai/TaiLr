import torch
import os
import argparse
import random
from tqdm import tqdm

def read_text(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(line.rstrip().split())
    return data

def read_vocab(filename):
    vocab = []
    with open(filename, "r") as f:
        for line in f.readlines():
            vocab.append(line.rstrip().split()[0])
    return vocab


def repeat(tokens):
    tok_ids = list(range(len(tokens)))
    rep_id = random.choice(tok_ids)
    rep_tokens = tokens.copy()
    rep_tokens.insert(rep_id, rep_tokens[rep_id])
    return rep_tokens

def delete(tokens):
    del_tokens = tokens.copy()
    del_tokens.pop()
    return del_tokens


def substitute(tokens, vocab):
    tok_ids = list(range(len(tokens)))
    sub_tokens = tokens.copy()
    sub_tok = random.choice(vocab)
    sub_pos = random.choice(tok_ids)
    sub_tokens[sub_pos] = sub_tok
    return sub_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", type=str, default="data/coco_pseudo/dev.tgt")
    parser.add_argument("--vocab-dir", type=str, default="data/coco-bin/dict.tgt.txt")
    parser.add_argument("--re", action="store_true", help="repeat one word")
    parser.add_argument("--de", action="store_true", help="delete one random word")
    parser.add_argument("--su", action="store_true", help="substitute one word from vocab")
    parser.add_argument("--repeat", type=int, default=20, help="repeat corruption N times")
    args = parser.parse_args()

    random.seed(42)

    perturb_types = []
    if args.re:
        perturb_types.append("repeat")
    if args.de:
        perturb_types.append("delete")
    if args.su:
        perturb_types.append("substitute")

    texts = read_text(args.text_dir)
    vocab = read_vocab(args.vocab_dir)
    perturb_texts = []
    for text in tqdm(texts):
        cur_perturb_text = []
        valid = True
        cur_text = text
        cur_perturb_text.append(cur_text)
        for r in range(args.repeat):
            print(cur_text)
            if len(cur_text) == 0:
                valid = False
                break
            cur_type = random.choice(perturb_types)
            if cur_type == "repeat":
                cur_text = repeat(cur_text)
            if cur_type == "delete":
                cur_text = delete(cur_text)
            if cur_type == "substitute":
                cur_text = substitute(cur_text, vocab)
            cur_perturb_text.append(cur_text)
        if valid:
            perturb_texts.extend(cur_perturb_text)
    
    save_dir = "data/coco_pseudo-"
    if args.re:
        save_dir += "re-"
    if args.de:
        save_dir += "de-"
    if args.su:
        save_dir += "su-"
    save_dir += "repeat-{}/".format(args.repeat)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(save_dir + "test.tgt", "w") as f:
        for line in perturb_texts:
            f.write(" ".join(line) + "\n")
    
    with open(save_dir + "test.src", "w") as f:
        for i in range(len(perturb_texts)):
            f.write("<go>\n")


if __name__ == "__main__":
    main()

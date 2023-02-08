from nltk.tokenize import word_tokenize
import nltk
import argparse

def corpus_bleu(refs, hypos, order=None):
    refs = [[x] for x in refs]
    hypos = [x for x in hypos]
    weight = [0, 0, 0, 0]
    if order == None or order == 4:
        weight = [0.25, 0.25, 0.25, 0.25]
    elif order == 1:
        weight = [1.0, 0.0, 0.0, 0.0]
    elif order == 2:
        weight = [0.5, 0.5, 0.0, 0.0]
    elif order == 3:
        weight = [0.33, 0.33, 0.33, 0.0]
    return nltk.translate.bleu_score.corpus_bleu(refs, hypos, weights=weight)

def sentence_bleu(refs, hypos, order=None):
    bleu = 0.0
    for r,h in zip(refs, hypos):
        if order == None or order == 4:
            weight = [0.25, 0.25, 0.25, 0.25]
        elif order == 1:
            weight = [1.0, 0.0, 0.0, 0.0]
        elif order == 2:
            weight = [0.5, 0.5, 0.0, 0.0]
        elif order == 3:
            weight = [0.33, 0.33, 0.33, 0.0]
        bleu += nltk.translate.bleu_score.sentence_bleu([r], h, weights=weight)
    return bleu / len(hypos)

def read_lines(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(line.rstrip())
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=str, required=True, help="reference")
    parser.add_argument("--h", type=str, required=True, help="hypothesis")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--corpus", action="store_true")
    parser.add_argument("--replace_newline", action="store_true")
    args = parser.parse_args()

    preds = read_lines(args.h)
    refs = read_lines(args.r)

    if args.replace_newline:
        preds = [x.replace("<newline>", "") for x in preds]
        refs = [x.replace("<newline>", "") for x in refs]

    tok_preds = [word_tokenize(x) for x in preds]
    tok_refs = [word_tokenize(x) for x in refs]

    fmt = ""
    if args.corpus:
        for order in range(1, args.order + 1):
            print("Evaluating order: ", order)
            fmt += "corpus-BLEU{}: ".format(order)
            score = corpus_bleu(tok_refs, tok_preds, order = order)
            fmt += str(score) + " | "
    else:
        for order in range(1, args.order + 1):
            print("Evaluating order: ", order)
            fmt += "sentence-BLEU{}: ".format(order)
            score = sentence_bleu(tok_refs, tok_preds, order = order)
            fmt += str(score) + " | "
    
    print(fmt)

main()

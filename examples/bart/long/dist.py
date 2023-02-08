from nltk.tokenize import word_tokenize
import nltk
import argparse



def distinct_ngrams(inputs, n, vocabs=None):
    output = {}
    for input in inputs:
        for i in range(len(input)-n+1):
            g = ' '.join(input[i:i+n])
            valid = True
            if vocabs is not None:
                for tok in g.split():
                    if tok not in vocabs:
                        valid = False
                        break
            if valid:
                output.setdefault(g, 0)
                output[g] += 1

    if sum(output.values())==0:
        ratio = 0
    else:
        ratio = float(len(output.keys()))/ sum(output.values())

    return ratio

def read_lines(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(line.rstrip())
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=str, required=True, help="hypothesis")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--replace_newline", action="store_true")
    args = parser.parse_args()

    preds = read_lines(args.h)
    
    if args.replace_newline:
        preds = [x.replace("<newline>", "") for x in preds]

    
    tok_preds = [[t.lower() if args.lower else t for t in word_tokenize(x)] for x in preds]

    vocabs = []
    for pred in tok_preds:
        vocabs.extend([x.lower() if args.lower else x for x in pred])
    
    vocabs = set(vocabs)

    fmt = ""
    for order in range(1, args.order + 1):
        print("Evaluating order: ", order)
        fmt += "Dist{}: ".format(order)
        score = distinct_ngrams(tok_preds, order, vocabs)
        fmt += str(score) + " | "
    
    print(fmt)

main()
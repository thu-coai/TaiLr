import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from abc import abstractmethod
import argparse


class Metrics:
    def __init__(self):
        self.name = 'Metric'

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    @abstractmethod
    def get_score(self):
        pass

class Bleu(Metrics):
    def __init__(self, test_text='', real_text='', gram=3):
        super().__init__()
        self.name = 'Bleu'
        self.test_data = test_text
        self.real_data = real_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.real_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_hypothesis(self):
        hypothesis = list()
        with open(self.test_data) as test_data:
            for text in test_data:
                text = nltk.word_tokenize(text)
                hypothesis.append(text)
        return hypothesis


    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        hypothesis = self.get_hypothesis()
        weight = tuple((1. / ngram for _ in range(ngram)))
        #with open(self.test_data) as test_data:
        for hypo in hypothesis:
            #hypo = nltk.word_tokenize(hypo)
            bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypo, weight,
                                                                smoothing_function=SmoothingFunction().method1))
        return sum(bleu) / len(bleu)

    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        hypothesis = self.get_hypothesis()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        
        for index in range(len(hypothesis)):
            result.append(pool.apply_async(self.calc_bleu, args=(reference, hypothesis[index], weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str)
    parser.add_argument("--r", type=str)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    bleu = Bleu(args.s, args.r, args.order)
    print("bleu{}: {}".format(args.order, bleu.get_bleu_fast()))

if __name__ == "__main__":
    main()


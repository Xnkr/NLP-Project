import sys
import math
from collections import defaultdict

input_sentence = "<s> BMW produces luxury vehicles and automobiles </s>"
# input_sentence = "<s> plan a panama </s>"
input_sentence_tokens = input_sentence.lower().split()

with open("NLP6320_POSTaggedTrainingSet-Windows.txt") as f:
    lines = f.readlines()
    # lines = ['a man a man a man a plan a plan a canal panama panama']
    unigram_frequencies = defaultdict(int)
    unigram_tokens = 0
    bigram_frequencies = defaultdict(int)
    bigram_tokens = 0
    for sentence in lines:
        words = [word.lower().split('_')[0] for word in sentence.split() if word]
        words.insert(0, '<s>')
        words.append('</s>')
        for word in words:
            unigram_frequencies[word] += 1

        for unigram in unigram_frequencies:
            unigram_tokens += unigram_frequencies[unigram]

        word_pairs = zip(*[words[i:] for i in range(2)])
        bigrams = [' '.join(bigram) for bigram in word_pairs]
        for bigram in bigrams:
            bigram_frequencies[bigram] += 1

        for bigram in bigram_frequencies:
            bigram_tokens += bigram_frequencies[bigram]
    V = len(unigram_frequencies)


def no_smoothing():
    prob = 1
    for i in range(1, len(input_sentence_tokens)):
        try:
            prob *= bigram_frequencies[input_sentence_tokens[i - 1] + " " + input_sentence_tokens[i]] / \
                    unigram_frequencies[
                        input_sentence_tokens[i - 1]]
        except ZeroDivisionError:
            print("Cannot compute probabilities of unseen words with no smoothing; Unseen word:",
                  input_sentence_tokens[i - 1])
            return
    print("No smoothing; P(" + input_sentence + ") = ", prob)


def add_one_smoothing():
    prob = 1
    for i in range(1, len(input_sentence_tokens)):
        prob *= (bigram_frequencies[input_sentence_tokens[i - 1] + " " + input_sentence_tokens[i]] + 1) / (
                unigram_frequencies[input_sentence_tokens[i - 1]] + V)
    print("Add one smoothing; P(" + input_sentence + ") = ", prob)


def good_turing_smoothing():
    N = defaultdict(int)
    for bigram in bigram_frequencies:
        N[bigram_frequencies[bigram]] += 1
    c_star = defaultdict(float)
    N_ = 0
    for c, N_c in N.copy().items():
        c_star[c] = 0.0 if N[c + 1] == 0 else ((c + 1) * N[c + 1]) / N_c
        N_ += (c * N_c)

    prob = 1
    for i in range(1, len(input_sentence_tokens)):
        c = bigram_frequencies[input_sentence_tokens[i - 1] + " " + input_sentence_tokens[i]]
        if c == 0:
            prob *= N[1] / N_
        else:
            prob *= c_star[c] / N_

    print("Good turing discounting based smoothing; P(" + input_sentence + ") = ", prob)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == '1':
            no_smoothing()
        elif sys.argv[1] == '2':
            add_one_smoothing()
        elif sys.argv[1] == '3':
            good_turing_smoothing()
        else:
            print("Invalid command")
    else:
        no_smoothing()
        add_one_smoothing()
        good_turing_smoothing()


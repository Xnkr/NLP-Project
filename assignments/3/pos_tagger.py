import math
import sys
from collections import Counter, defaultdict
from typing import List

START_TAG = "<s>"
END_TAG = "</s>"

input_file = "NLP6320_POSTaggedTrainingSet-Windows.txt"
with open(input_file) as f:
    lines = f.readlines()
    tag_bigrams = Counter()
    tag_unigrams = Counter()
    word_tag_freq = Counter()

    for sentence in lines:
        sentence = f"{START_TAG} {sentence} {END_TAG}"
        tags = []
        for word in sentence.split():
            if word:
                sp = word.split('_')
                tags.append(sp[0] if len(sp) == 1 else sp[1])
        
        unigrams = zip(*[tags[_i:] for _i in range(1)])
        tag_unigram_list = [" ".join(unigram) for unigram in unigrams]

        bigrams = zip(*[tags[_i:] for _i in range(2)])
        tag_bigram_list = [" ".join(bigram) for bigram in bigrams]
        
        word_tag = [word for word in sentence.split() if word]

        tag_bigrams += Counter(tag_bigram_list)
        tag_unigrams += Counter(tag_unigram_list)
        word_tag_freq += Counter(word_tag)

bigram_prob = defaultdict(float)
for word_tag in word_tag_freq:
    s = word_tag.split("_")
    if len(s) == 1:
        tag = s[0]
    else:
        tag = s[1]
    bigram_prob[word_tag] = word_tag_freq[word_tag] / tag_unigrams[tag]

def pos_tag(input_str):
    tagged_input = ''
    max_prob = 0
    max_tag = ''
    prev_max_tag = START_TAG
    prob = 1
    for input_word in input_str.split()[:-1]:
        for tag in tag_unigrams:
            tag_prob = bigram_prob[f"{input_word}_{tag}"] * tag_bigrams[f"{prev_max_tag} {tag}"]
            if tag_prob > max_prob:
                max_prob = tag_prob
                max_tag = tag
        prob *= max_prob
        tagged_input += f"{input_word}_{max_tag} "
        prev_max_tag = max_tag
        max_prob = 0
    input_word = input_str.split()[-1]
    for tag in tag_unigrams:
        tag_prob = bigram_prob[f"{input_word}_{tag}"] * tag_bigrams[f"{prev_max_tag} {tag}"] * tag_bigrams[f"{tag} </s>"]
        if tag_prob > max_prob:
            max_prob = tag_prob
            max_tag = tag
    prob *= max_prob 
    tagged_input += f"{input_word}_{max_tag}"
    print("Given Sentence: ", input_str)
    print("POS Tagged: ", tagged_input)

input_line = input("Enter the test sentence: ")
pos_tag(input_line)

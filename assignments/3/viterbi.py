from collections import defaultdict
import csv

pi = defaultdict(float)

with open('transition.csv') as f:
    dr = csv.DictReader(f)
    next_tags = dr.fieldnames[1:]
    for row in dr:
        current_tag = row['']
        for next_tag in next_tags:
            pi[f"{current_tag} {next_tag}"] = float(row[next_tag])

observation_likelihood = defaultdict(float)

with open('observation.csv') as f:
    dr = csv.DictReader(f)
    words = dr.fieldnames[1:]
    for row in dr:
        current_tag = row['']
        for word in words:
            observation_likelihood[f"{word} {current_tag}"] = float(row[word])

def get_max_prob(current_word, current_tag):
    max_tag_prob = 0
    max_tag_i = 0
    for previous_index, previous_tag in enumerate(next_tags):
        p = pi[f"{previous_tag} {current_tag}"] * observation_likelihood[f"{current_word} {current_tag}"]
        if p > max_tag_prob:
            max_tag_prob = p
            max_tag_i = previous_index
    return max_tag_prob, max_tag_i


def compute_trellis(input_words, trellis, backtrack):
        for i in range(len(input_words)):
            for tag_i, current_tag in enumerate(next_tags):
                max_tag_prob, max_tag_i = get_max_prob(input_words[i], current_tag)
                trellis[tag_i][i] = max_tag_prob
                backtrack[tag_i][i] = max_tag_i 

def get_final_tags(n, trellis, backtrack):
    final_tags = [''] * n
    last_level = n - 1
    last_level_trellis = trellis[:][last_level]
    prev = last_level_trellis.index(max(last_level_trellis))
    final_tags[last_level] = next_tags[prev]
    while last_level > 0:
        prev = backtrack[prev][last_level]
        last_level -= 1
        final_tags[last_level] = next_tags[prev]
    return final_tags

def pos_tag_viterbi(input_sentence):
    input_words = input_sentence.split()

    trellis = [[0.0 for _ in range(len(input_words))] for _ in range(len(next_tags))]
    backtrack = [[0 for _ in range(len(input_words))] for _ in range(len(next_tags))]
    
    compute_trellis(input_words, trellis, backtrack)
    final_tags = get_final_tags(len(input_words), trellis, backtrack)

    print("Most likely tag sequence:")
    print(" ".join(word + "_" + tag for word, tag in zip(input_words, final_tags)))

input_sentence = input("Test sentence: ") # "Janet will back the bill"
pos_tag_viterbi(input_sentence)
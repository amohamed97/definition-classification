from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from pathlib import Path
import sys
import fasttext
import os
import math


# Naive Bayes With Unigrams
positive_count = 0
negative_count = 0
positive_words = 0
negative_words = 0
vocab = dict()


for child in Path("./train/").iterdir():
        if child.suffix == '.deft':
            with open(child, encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    splits = line.split('"')
                    text = splits[1]
                    text = text.lower()
                    has_def = int(splits[-2])
                    tokens = word_tokenize(text)
                    # tokens = list(bigrams(text.split()))
                    # for i in range(len(tokens)):
                    #     tokens[i] = ' '.join(tokens[i])
                    if has_def:
                        positive_count += 1
                        positive_words += len(tokens)
                    else:
                        negative_count += 1
                        negative_words += len(tokens)
                    for token in tokens:
                        if token in vocab:
                            if has_def:
                                vocab[token][0] += 1
                            else:
                                vocab[token][1] += 1
                        else:
                            if has_def:
                                vocab[token] = [1,0]
                            else:
                                vocab[token] = [0,1]

true = 0
false = 0
positive_prob = positive_count/(positive_count+negative_count)
negative_prob = negative_count/(positive_count+negative_count)
vocab_len = len(vocab)
unknown_prob = 1/vocab_len
for child in Path("./test/").iterdir():
    if child.suffix == '.deft':
            with open(child,encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    classified = -1
                    splits = line.split('"')
                    text = splits[1]
                    text = text.lower()
                    has_def = int(splits[-2])
                    tokens = word_tokenize(text)
                    # tokens = list(bigrams(text.split()))
                    # for i in range(len(tokens)):
                    #     tokens[i] = ' '.join(tokens[i])
                    p_prob = positive_prob
                    n_prob = negative_prob
                    for token in tokens:    
                        if token in vocab:
                            p_prob *= (1+vocab[token][0])/(positive_words+vocab_len)
                            n_prob *= (1+vocab[token][1])/(negative_words+vocab_len)
                        else:
                            p_prob *= unknown_prob
                            n_prob *= unknown_prob
                    if p_prob > n_prob:
                        classified = 1
                    else:
                        classified = 0
                    if classified == has_def:
                        true += 1
                    else:
                        false += 1
print("NAIVE BAYES (Unigrams) : Accuracy = {:0.2f}% \n".format((true/(true+false))*100))


# Naive Bayes With Bigrams
positive_count = 0
negative_count = 0
positive_words = 0
negative_words = 0
vocab = dict()


for child in Path("./train/").iterdir():
        if child.suffix == '.deft':
            with open(child, encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    splits = line.split('"')
                    text = splits[1]
                    text = text.lower()
                    has_def = int(splits[-2])
                    # tokens = word_tokenize(text)
                    tokens = list(bigrams(text.split()))
                    for i in range(len(tokens)):
                        tokens[i] = ' '.join(tokens[i])
                    if has_def:
                        positive_count += 1
                        positive_words += len(tokens)
                    else:
                        negative_count += 1
                        negative_words += len(tokens)
                    for token in tokens:
                        if token in vocab:
                            if has_def:
                                vocab[token][0] += 1
                            else:
                                vocab[token][1] += 1
                        else:
                            if has_def:
                                vocab[token] = [1,0]
                            else:
                                vocab[token] = [0,1]

true = 0
false = 0
positive_prob = positive_count/(positive_count+negative_count)
negative_prob = negative_count/(positive_count+negative_count)
vocab_len = len(vocab)
unknown_prob = 1/vocab_len
for child in Path("./test/").iterdir():
    if child.suffix == '.deft':
            with open(child,encoding="utf8") as f:
                lines = f.readlines()
                for line in lines:
                    classified = -1
                    splits = line.split('"')
                    text = splits[1]
                    text = text.lower()
                    has_def = int(splits[-2])
                    # tokens = word_tokenize(text)
                    tokens = list(bigrams(text.split()))
                    for i in range(len(tokens)):
                        tokens[i] = ' '.join(tokens[i])
                    p_prob = positive_prob
                    n_prob = negative_prob
                    for token in tokens:    
                        if token in vocab:
                            p_prob *= (1+vocab[token][0])/(positive_words+vocab_len)
                            n_prob *= (1+vocab[token][1])/(negative_words+vocab_len)
                        else:
                            p_prob *= unknown_prob
                            n_prob *= unknown_prob
                    if p_prob > n_prob:
                        classified = 1
                    else:
                        classified = 0
                    if classified == has_def:
                        true += 1
                    else:
                        false += 1
print("NAIVE BAYES (Bigrams) : Accuracy = {:0.2f}% \n".format((true/(true+false))*100))




# USING fastTEXT
sys.stdout = open(os.devnull, "w")
model = fasttext.train_supervised(input="output.train",wordNgrams=2,epoch=5)
sys.stdout = sys.__stdout__
percent = model.test("output.test")[1]*100
# formatted = str(math.ceil((percent*100)/100))
print("fastText : Accuracy = {:0.2f}%".format(percent))


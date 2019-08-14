# Larry Liu
# Aletheia is a chatbot that focuses on greek mythology for discussion

import numpy
import tflearn
import random
import json
import nltk
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

if type(tf.contrib) != type(tf):
    tf.contrib._warning = None

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


with open("intents.json") as file:
    data = json.load(file)

word = []
classes = []
patA = []
patB = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        word.append(wrds)
        patA.append(pattern)
        patB.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent("tags"))

word = [stemmer.stem(w.lower()) for w in word]
word = sorted(list(set(newClass)))

classes = sorted(classes)

training = []
result = []
zero = [0 for _ in range(len(classes))]

for x, pat in enumerate(patA):
    bag = []
    letter = [stemmer.stem(w) for w in pat]

    for w in word:
        if w in letter:
            bag.append(1)
        else:
            bag.append(0)

        result_row = zero[:]
        result_row[classes.index(patB[x])] = 1

        training.append(bag)
        result.append(result_row)


training = numpy.array(training)
output = numpy.array(result)

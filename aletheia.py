# aletheia.py

import os
import random

import discord
from dotenv import load_dotenv

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn

import random
import json
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        word, classes, training, result = pickle.load(f)
except:
    word = []
    classes = []
    patA = []
    patB = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            word.extend(wrds)
            patA.append(wrds)
            patB.append(intent["tag"])

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    word = [stemmer.stem(w.lower()) for w in word if w != "?"]
    word = sorted(list(set(word)))

    classes = sorted(classes)

    training = []
    result = []

    zero = [0 for _ in range(len(classes))]

    for x, doc in enumerate(patA):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in word:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        result_row = zero[:]
        result_row[classes.index(patB[x])] = 1

        training.append(bag)
        result.append(result_row)

    training = numpy.array(training)
    result = numpy.array(result)

    with open("data.pickle", "wb") as f:
        pickle.dump((word, classes, training, result), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(result[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


try:
    model.load("model.tflearn")
except:
    model.fit(training, result, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, word):
    bag = [0 for _ in range(len(word))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(word):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(message):
    results = model.predict([bag_of_words(message, word)])
    results_index = numpy.argmax(results)
    tag = classes[results_index]
    if results[0][results_index] > 0.7:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        return random.choice(responses)
    else:
        return "Truth is only given for what I know, please try again mortal."


load_dotenv()
token = os.getenv("DISCORD_TOKEN")

client = discord.Client()


@client.event
async def on_ready():
    print(f"{client.user.name} has connected to Discord!")


@client.event
async def on_member_join(member):
    await member.create_dm()
    await member.dm_channel.send(f"Hi {member.name}, welcome to my Discord server!")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    typed = chat(message.content)
    await message.channel.send(typed)


print(token)
client.run(token)

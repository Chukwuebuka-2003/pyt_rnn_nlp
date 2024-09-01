from io import open
import glob
import os


def file_path(path):
    return glob.glob(path)


file_path("data/names/*.txt")

import unicodedata
import string

all_char = string.ascii_letters + " .,;'"
n_char = len(all_char)


# turn a unicode string to plain ascii


def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_char
    )


#print(unicode_to_ascii("Sérgio"))

# lists of names per language
category_line = {}
all_categories = []


# read and split files into lines
def line_read(file_name):
    lines = open(file_name, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


for file_name in file_path("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    lines = line_read(file_name)
    category_line[category] = lines


n_categories = len(all_categories)


# turning names into tensor
import torch


def char_to_index(char):
    return all_char.find(char)


def char_to_tensor(char):
    tensor = torch.zeros(1, n_char)
    tensor[0][char_to_index(char)] = 1
    return tensor


def line_to_tensor(line):
    tensor = torch.zeros(len(line), n_char)
    for i, char in enumerate(line):
        tensor[i][char_to_index(char)] = 1
    return tensor


# RNN module
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h20 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.relu(self.i2h(input) + self.h2h(hidden))
        output = self.softmax(self.h20(hidden))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_char, n_hidden, n_categories)

# testing import it
input = line_to_tensor("Ebuka")
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)

# Training the model


def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


print(category_from_output(output))

import random


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrain_example():
    category = random_choice(all_categories)
    line = random_choice(category_line[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


for i in range(5):
    category, line, category_tensor, line_tensor = randomTrain_example()
    print("category =", category, "/ line =", line)




criterion = nn.NLLLoss() 

learning_rate = 0.005 

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()



# run/test it 

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrain_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess = category_from_output(output)
        correct = "✓" if guess == category else "✗ (%s)" % category
        print(
            "%d %d%% (%s) %.4f %s / %s %s"
            % (
                iter,
                iter / n_iters * 100,
                timeSince(start),
                loss,
                line,
                guess,
                correct,
            )
        )

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.figure()
plt.plot(all_losses)


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=4):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))

        topv, top1 = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = top1[0][i].item()

            print('(%.2f) %s' % (value, all_categories[category_index]))

            predictions.append([value, all_categories[category_index]])


predict('Dylan')
predict('Micheal')
predict('Catherine')

import string
import sys
import json

with open(sys.argv[1], 'r', encoding='utf8') as modelFile:
    data = modelFile.readlines()
    weights = json.loads(data[0])
    b = json.loads(data[1])

outputFile = open("percepoutput.txt", "w", encoding='utf8')

table = str.maketrans({key: None for key in string.punctuation})


def tokenize(s):
    return s.translate(table).rstrip()


inputFile = open(sys.argv[2], "r", encoding='utf8')
input = {}
id = ""
line_token_count = 0

# parse test data to input vectors
for line in inputFile:
    line = tokenize(line)
    tokens = line.split(" ")

    for token in tokens:
        if line_token_count == 0:
            id = token
            input[id] = {}
        else:
            token = token.lower()
            if token not in input[id]:
                input[id][token] = 1
            else:
                input[id][token] += 1

        line_token_count += 1

# classify inputs
output = {}
for id in input:
    a = 0
    for f in input[id]:
        if f in weights:
            a += weights[f] * input[id][f]
    a += b
    if a >= 0:
        output[id] = ["True"]
    else:
        output[id] = ["Fake"]

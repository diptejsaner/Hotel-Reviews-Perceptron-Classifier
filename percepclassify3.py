import string
import sys
import json

with open(sys.argv[1], 'r', encoding='utf8') as modelFile:
    data = modelFile.readlines()
    weightsTF = json.loads(data[0])
    bTF = json.loads(data[1])
    weightsPN = json.loads(data[2])
    bPN = json.loads(data[3])

outputFile = open("percepoutput.txt", "w", encoding='utf8')

table = str.maketrans({key: None for key in string.punctuation})


def tokenize(s):
    return s.translate(table).rstrip()


inputFile = open(sys.argv[2], "r", encoding='utf8')
input = {}
id = ""

# parse test data to input vectors
for line in inputFile:
    line = tokenize(line)
    tokens = line.split(" ")
    line_token_count = 0

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
for id in input:
    aTF = 0
    aPN = 0

    for f in input[id]:
        if f in weightsTF:
            aTF += weightsTF[f] * input[id][f]
        if f in weightsPN:
            aPN += weightsPN[f] * input[id][f]

    aTF += bTF
    aPN += bPN

    outputFile.write(id)
    if aTF >= 0:
        outputFile.write(" True")
    else:
        outputFile.write(" Fake")

    if aPN >= 0:
        outputFile.write(" Pos")
    else:
        outputFile.write(" Neg")

    outputFile.write("\n")

outputFile.close()

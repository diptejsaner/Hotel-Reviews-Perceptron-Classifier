import string
import sys
import json

table = str.maketrans({key: None for key in string.punctuation})


def tokenize(s):
    return s.translate(table).rstrip()


file = open(sys.argv[1], "r", encoding='utf8')

classes_map = {
    "True": 1,
    "Fake": -1,
    "Pos": 1,
    "Neg": -1
}

# create feature vectors
feature_vectors = {}

line_count = 0
for line in file:
    line = tokenize(line)
    tokens = line.split(" ")

    line_token_count = 0
    class1 = ""
    class2 = ""
    id = ""
    y = 0
    list = []
    feature_count = {}

    for token in tokens:
        if line_token_count == 0:
            id = token

        if line_token_count == 1:
            class1 = token
            y = classes_map[class1]
            list.append(y)

        if line_token_count == 2:
            class2 = token
            y = classes_map[class2]
            list.append(y)

        elif line_token_count > 2:
            token = token.lower()
            if token not in feature_count:
                feature_count[token] = 1
            else:
                feature_count[token] += 1
        line_token_count += 1

    list.append(feature_count)
    feature_vectors[id] = list
    line_count += 1

# start training
weightsTF = {}
weightsPN = {}
bTF = 0
bPN = 0
iterations = 1

for id in feature_vectors:
    aTF = 0
    aPN = 0
    x = feature_vectors[id][2]

    for f in x:
        if f in weightsTF:
            aTF += weightsTF[f] * x[f]
        aTF += bTF

    yTF = feature_vectors[id][0]
    yPN = feature_vectors[id][1]

    if aTF * yTF <= 0:
        # update weights
        x = feature_vectors[id][2]

        for f in x:
            if f in weightsTF:
                weightsTF[f] += yTF * x[f]
            else:
                weightsTF[f] = yTF * x[f]
        bTF += y

file.close()

with open('vanillamodel.txt', 'w', encoding='utf8') as fp:
    fp.write(json.dumps(weightsTF))
    fp.write("\n")
    fp.write(json.dumps(bTF))


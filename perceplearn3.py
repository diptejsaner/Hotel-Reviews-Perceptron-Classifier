import string
import sys
import json
import random

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

# create stop words list
stop_words = set()
with open('stop-words.txt', 'r') as sw_file:
    for word in sw_file:
        stop_words.add(word.rstrip())

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
    f_list = []
    feature_count = {}

    for token in tokens:
        if line_token_count == 0:
            id = token

        if line_token_count == 1:
            class1 = token
            y = classes_map[class1]
            f_list.append(y)

        if line_token_count == 2:
            class2 = token
            y = classes_map[class2]
            f_list.append(y)

        elif line_token_count > 2:
            token = token.lower()
            if token not in stop_words:
                if token not in feature_count:
                    feature_count[token] = 1
                else:
                    feature_count[token] += 1
        line_token_count += 1

    f_list.append(feature_count)
    feature_vectors[id] = f_list
    line_count += 1

# start training
v_weightsTF = {}
v_weightsPN = {}
v_bTF = 0
v_bPN = 0

avg_weightsTF = {}
avg_weightsPN = {}
avg_bTF = 0
avg_bPN = 0

iterations = 20
c = 1

for iter in range(iterations):
    for id in feature_vectors:
        aTF = 0
        aPN = 0
        x = feature_vectors[id][2]

        for f in x:
            if f in v_weightsTF:
                aTF += v_weightsTF[f] * x[f]

            if f in v_weightsPN:
                aPN += v_weightsPN[f] * x[f]

        aTF += v_bTF
        aPN += v_bPN

        yTF = feature_vectors[id][0]
        yPN = feature_vectors[id][1]

        if aTF * yTF <= 0:
            for f in x:
                if f in v_weightsTF:
                    v_weightsTF[f] += yTF * x[f]
                    avg_weightsTF[f] += yTF * x[f] * c
                else:
                    v_weightsTF[f] = yTF * x[f]
                    avg_weightsTF[f] = yTF * x[f] * c
            v_bTF += yTF
            avg_bTF += yTF * c

        if aPN * yPN <= 0:
            for f in x:
                if f in v_weightsPN:
                    v_weightsPN[f] += yPN * x[f]
                    avg_weightsPN[f] += yPN * x[f] * c
                else:
                    v_weightsPN[f] = yPN * x[f]
                    avg_weightsPN[f] = yPN * x[f] * c
            v_bPN += yPN
            avg_bPN += yPN * c

        c += 1
    random.shuffle(list(feature_vectors))
file.close()

for f in avg_weightsTF:
    avg_weightsTF[f] = v_weightsTF[f] - (avg_weightsTF[f] / c)

avg_bTF = v_bTF - (avg_bTF / c)

for f in avg_weightsPN:
    avg_weightsPN[f] = v_weightsPN[f] - (avg_weightsPN[f] / c)

avg_bPN = v_bPN - (avg_bPN / c)

with open('vanillamodel.txt', 'w') as fp:
    fp.write(json.dumps(v_weightsTF))
    fp.write("\n")
    fp.write(json.dumps(v_bTF))
    fp.write("\n")
    fp.write(json.dumps(v_weightsPN))
    fp.write("\n")
    fp.write(json.dumps(v_bPN))

with open('averagedmodel.txt', 'w') as fp:
    fp.write(json.dumps(avg_weightsTF))
    fp.write("\n")
    fp.write(json.dumps(avg_bTF))
    fp.write("\n")
    fp.write(json.dumps(avg_weightsPN))
    fp.write("\n")
    fp.write(json.dumps(avg_bPN))


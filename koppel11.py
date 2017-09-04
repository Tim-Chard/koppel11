'''-- Jakob Koehler & Tolga Buz
Reproduction of Koppel11
'''
#--- Parameters:
# n-gram size
n = 4
# length of feature list
featureLength = 20000
# Score threshold (needed for open set)
threshold = 0
# number of k repetitions
repetitions = 100
# minimum size of document
# (increases precision, but deteriorates recall,
# if there are many small documents)
minlen = 0
# candidates with less than this amount of words in trainingdata are not
# attributed to
mintrainlen = 500

#--- Imports:
from math import sqrt
import jsonhandler
import random
import argparse
import numpy as np

from collections import Counter

#--- Methods:
'''- create dict:
gets a string (e.g. Book), splits it into and returns a dict
with all possible n-grams/features'''


def create_dict(s): 
    dict = Counter([word[i:i + n] for word in s.split() for i in range(max(1,len(word) - n + 1))]) 
    return dict

def tokenize(s):
    return [word[i:i + n] for word in s.split() for i in range(max(1,len(word) - n + 1))]


def create_vector(tokens, corpus):
    vec = np.zeros((len(corpus,)))
    for i in tokens:
        vec[corpus[i]] += 1
    return vec


'''- selectFeatures:
selects the x most frequent n-grams/features (x=featureLength)
to avoid a (possibly) too big featurelist'''


def selectFeatures(word_counts):
    return sorted(word_counts, key=word_counts.get, reverse=True)[:min(len(word_counts), featureLength)]

'''- createFeatureMap:
creates Feature Map that only saves
the features that actually appear more frequently than 0.
Thus, the featurelist needs less memory and can work faster'''


def createFeatureMap(s, features):
    fmap = {}
    vec = create_dict(s)
    for ngram in features:
        if ngram in vec:
            fmap[ngram] = vec[ngram]
    return fmap


def create_feature_map_from_tokens(tokens, features):
    vec = Counter(tokens)
    fmap = {ngram : vec[ngram] for ngram in features if ngram in vec}
    return fmap



def cosMetric(v1, v2):
    return 1 - cosSim(v1, v2)

def minmaxMetric(v1, v2):
    return 1 - minmax(v1, v2)

'''- cosSim:
calculates cosine similarity of two dicts v1 and v2.
-> cosine(X, Y) = (X * Y)/(|X|*|Y|)
'''


def cosSim(v1, v2):
    sp = float(0)
    len1 = 0
    len2 = 0
    for ngram in v1:
        len1 += v1[ngram] ** 2
    for ngram in v2:
        len2 += v2[ngram] ** 2
    len1 = sqrt(len1)
    len2 = sqrt(len2)
    for ngram in v1:
        if ngram in v2:
            sp += v1[ngram] * v2[ngram]
    return sp / (len1 * len2)

'''- minmax:
calculates minmax similarity of two dicts v1 and v2.
-> minmax(X, Y) = sum(min(Xi, Yi))/sum(max(Xi, Yi))

This baseline method will be used for further evaluation.
'''


def minmax(v1, v2):
    minsum = 0
    maxsum = 0
    for ngram in v1:
        if ngram in v2:
            # ngram is in both dicts
            minsum += min(v1[ngram], v2[ngram])
            maxsum += max(v1[ngram], v2[ngram])
        else:
            # ngram only in v1
            maxsum += v1[ngram]
    for ngram in v2:
        if ngram not in v1:
            # ngram only in v2
            maxsum += v2[ngram]
    if maxsum == 0:
        return 0
    return float(minsum) / maxsum

'''- training:
Turns a given string into a n-gram dict
and returns its feature list.
'''


def training(s):
    print("training...")
    vec = create_dict(s)
    print("selecting features...")
    fl = selectFeatures(vec)
    print("done")
    return fl

'''- testSim:
args: two dicts, a featurelist
and func(to decide whether to use cosine or minmax similarity).

uses createFeatureMap and cosSim or minmax
and returns the similarity value of the two dicts
'''


def testSim(x, y, fl, func):
    fx = createFeatureMap(x, fl)
    fy = createFeatureMap(y, fl)
    if func == 0:
        return cosSim(fx, fy)
    else:
        return minmax(fx, fy)


'''- getRandomString:
Returns a random part of a string s
that has a given length
'''


def getRandomString(s, length):
    words = s.split()
    r = random.randint(0, len(words) - length)
    return "".join(words[r:r + length])


def get_random_tokens(tokens, length):
    r = random.randint(0, len(tokens) - length)
    return tokens[r:r + length]


#--- main:

def main(corpusdir, outputdir):
    
    candidates = jsonhandler.candidates
    unknowns = jsonhandler.unknowns
    jsonhandler.loadJson(corpusdir)
    jsonhandler.loadTraining()

    texts = {}
    # texts = frozenset() would this work??
    corpus = ""
    print("loading texts for training")
    deletes = []
    for cand in candidates:
        texts[cand] = ""
        for file in jsonhandler.trainings[cand]:
            texts[cand] += jsonhandler.getTrainingText(cand, file)
            # if frozenset() is used:
            # texts.add(jsonhandler.getTrainingText(cand, file))
            print("text " + file + " read")
        if len(texts[cand].split()) < mintrainlen:
            del texts[cand]
            deletes.append(cand)
        else:
            corpus += texts[cand]

    newcands = []
    for cand in candidates:
        if cand not in deletes:
            newcands.append(cand)
    candidates = newcands


    words_counts = [len(texts[cand].split()) for cand in texts]
    minwords = min(words_counts)

    tokens = {cand : tokenize(texts[cand]) for cand in texts}
    token_counts = [len(tokens[cand]) for cand in tokens]
    min_tokens = min(token_counts)

    print(minwords, min_tokens)

    base_features = training(corpus)
    feature_corpus = {base_features[i] : i for i in range(len(base_features))}
    authors = []
    scores = []

    for file in unknowns:
        print("testing " + file)

        unknown_text = jsonhandler.getUnknownText(file)
        unknown_len = len(unknown_text.split())
        textlen = min(unknown_len, minwords)

        #unknown_string = "".join(unknown_text.split()[:textlen])

        unknown_tokens = tokenize(unknown_text)
        token_len = min(len(unknown_tokens), min_tokens)

        unknown_tokens = unknown_tokens[:token_len]
        
        if unknown_len < minlen:
            authors.append("None")
            scores.append(0)
        else:
            wins = [0] * len(candidates)

            print(textlen)
            
            
            for i in range(repetitions):
                sample_features = random.sample(base_features, len(base_features) // 2)
                unknown_features = create_feature_map_from_tokens(unknown_tokens, sample_features)
                
                distances = []
                for cand in candidates:
                    cand_tokens = get_random_tokens(tokens[cand], textlen)
                    cand_features = create_feature_map_from_tokens(cand_tokens, sample_features)

                    dist = minmaxMetric(cand_features, unknown_features)
                    distances.append(dist)

                wins[distances.index(min(distances))] += 1
            score = max(wins) / float(repetitions)
            if score >= threshold:
                authors.append(candidates[wins.index(max(wins))])
                scores.append(score)
            else:
                authors.append("None")
                scores.append(score)

    print("storing answers")
    jsonhandler.storeJson(outputdir, unknowns, authors, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tira submission for PPM approach (koppel11)")
    parser.add_argument("-i", action="store", help="path to corpus directory")
    parser.add_argument("-o", action="store", help="path to output directory")
    args = vars(parser.parse_args())

    corpusdir = args["i"]
    outputdir = args["o"]

    if corpusdir == None or outputdir == None:
        parser.print_help()

    main(corpusdir, outputdir)

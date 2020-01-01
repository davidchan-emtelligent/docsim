"""
Expand norms with 3 tokens which has similary vector
"""
import os
import sys
import json
import string
import pickle
import gensim
import numpy as np
from segtok_spans.tokenizer import med_tokenizer

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("curr_dir")
from _extract_helper import multi_processor_works

TOP_N = 3

with open(os.path.join(curr_dir, "resources/stopwords.txt"), 'r') as fd:
	stopwords = set(fd.read().lower().split() + ['/']) - set(['no', 'not', 'nor', 'without'])

def work(word):
    if word in vocabs:
        return  [x[0] for x in model.most_similar(word, topn=TOP_N)]
    else:
        return []


def expanded_by_similarity(words):
    expanded_lst = []
    for lst in multi_processor_works(work, words, display_every_batches=10, batch_size=10):
        if lst:
            expanded_lst += lst

    return expanded_lst


clean_lst = list(string.punctuation) + ['-rrb-', '-lrb-', '-rsb-', '-lsb-']
def tokens_to_vec(toks, w2v, clean=clean_lst):
    def run():
        for tok in toks:
            tok = tok.lower()
            if tok in clean:
                continue
            if tok in w2v:
                yield w2v[tok]

    arr = list(run())
    if arr:
        return np.mean(arr, axis=0)
    else:
        return np.array([]) 


def save_word2vec(w2v_data, output_pickle):
    with open(output_pickle, "wb") as fj:
        pickle.dump(w2v_data, fj)
    print("save to: %s"%output_pickle, len(w2v_data))



import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input_path', dest='input_path', type=str, default="", help="input path")
argparser.add_argument('-o', '--output_pickle', dest='output_pickle', type=str, default=os.path.join(curr_dir, "resources/w2v.pickle"), help="output path")
argparser.add_argument('-m', '--model_path', dest='model_path', type=str, default="", help="model path")
args = argparser.parse_args()

with open(args.input_path, "r") as fd:
    lines = fd.read().split("\n")
keys = []
for line in lines:
    tokens, _ = zip(*med_tokenizer(line))
    keys += tokens
keys = [k for k in set(keys) if k not in stopwords]
print("keys:", len(keys), keys[:3]);keys=keys[:3]

if os.path.isdir(args.model_path):
    fs = [f for f in os.listdir(args.model_path) if f.endswith(".model")]
    argsorted = np.argsort([int(f.split('.')[-2].split('-')[-1]) for f in fs])
    fname = fs[argsorted[-1]]
    model_path = os.path.join(args.model_path, fname)
elif os.path.exists(args.model_path):
    model_path = args.model_path
else:
    print("ERROR: Invalid model_dir")
    sys.exit(0)

print("loading w2v model: %s ....."%model_path)
model = gensim.models.Word2Vec.load(model_path)
vocabs = [v for v in model.wv.vocab.keys() if v not in stopwords and v.isalpha()]
print("vocabs:", len(vocabs), vocabs[:3])

words = list(set(expanded_by_similarity(keys)));print(len(words))
w2v_data = dict([(word, model.wv[word]) for word in words])
save_word2vec(w2v_data, args.output_pickle)

import sys
import os
import re
import json
import gensim
import string
import math
import numpy as np
import pickle
from multiprocessing import Pool, Value
from segtok_spans.segmenter import split_multi
from segtok_spans.tokenizer import med_tokenizer
import pdftotext

markers = {"blue": ":blue:`", "red": ":red:`", "bold": ":bold:`", "end": "`"}

curr_dir = os.path.dirname(os.path.realpath(__file__))
default_config_path = os.path.join(curr_dir, "resources/default.config.json")
config_path = os.path.join(curr_dir, "resources/config.json")
if not os.path.exists(config_path):
	with open(default_config_path, 'r') as fd:
		config = json.load(fd)
else:
	with open(config_path, 'r') as fd:
		config = json.load(fd)
with open(os.path.join(curr_dir, "resources/stopwords.txt"), 'r') as fd:
	stopwords = set(fd.read().lower().split() + ['/']) - set(['no', 'not', 'nor', 'without'])


def read_file(f, fm):
	text = ""
	if fm == 'txt':
		with open(f, 'r') as fd:
			text += fd.read()
	elif fm == 'pdf':
		with open(f, 'rb') as fd:
			pdf = pdftotext.PDF(fd)
			text += "\n\n".join(pdf) + "\n\n"
	else:
		print("ERROR: Invalid input file type:", fm)
		sys.exit(0)

	return text


def parse_keys(key_path):
	with open(key_path, "r") as fd:
		terms = fd.read().split()

	def run():
		for t in terms:
			toks, _ = zip(*med_tokenizer(t.lower()))
			for tok in toks:
				if not tok.isdigit() and tok not in stopwords:
					yield tok

	return list(set(list(run())))


def get_words_keys(tokens):
	ws, ks = [], []
	for tok in tokens:
		if tok in global_keys:
			ks += [tok]
		if tok not in stopwords and not tok.isdigit():
			ws += [tok]
	return ws, ks


def work(data):
	f_id, text = data
	
	spans = []
	start = 0
	for m in re.finditer("( {0,3}\n {0,3}){2,10}", text):
		s, e = m.start(), m.end()
		spans += [(start, s)]
		start = e
	end = len(text)
	if start < end:
		spans += [(start, end)]

	paras = []
	p_id = 0
	for s, e in spans:
		sentences = []
		toks = []
		for sent_str, span in split_multi(text[s:e].lower()):
			if re.findall(r"[A-Za-z]+", sent_str):
				sent_str = sent_str.replace(".", " . ").replace("  ", " ")
				tokens, _ = zip(*med_tokenizer(sent_str))
				tokenized_text = " ".join(tokens)
				toks += tokens
				sentences += [{"tokenized_text": tokenized_text, "sentence_span": span}]
		ws, ks = get_words_keys(toks)
		if not ks:
			continue
		paras += [{"sentences": sentences, "words": ws, "keys": ks, "file_para_id": (f_id, p_id), "para_span": (s, e)}]
		p_id += 1
		
	if counter.value % 10 == 0:
		print ("\rprocessing files: %3d"%(counter.value), end='	')
		sys.stdout.flush()

	with counter.get_lock():
		counter.value += 1

	return paras


def parse_paragraphs(data, keys):
	global counter
	global global_keys
	counter = Value('i', 0)
	global_keys = keys

	pool = Pool()
	paras = []
	for para_lst in pool.map_async(work, data).get():
		paras += para_lst
	pool.close()
	pool.join()

	for i, para in enumerate(paras):
		para.update({'para_id':i})
 
	return paras


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

		return np.zeros(100)


def calculate_distances(vec1, vec2s):
	vec_sqr = vec1.dot(vec1)
	for vec2 in vec2s: 
		
		yield vec1.dot(vec2)/math.sqrt(vec_sqr*vec2.dot(vec2))


def main():
	import argparse

	argparser = argparse.ArgumentParser()
	argparser.add_argument("-i", "--input_dir", dest="input_dir", type=str, default="", help="input_dir")
	argparser.add_argument("-f", "--format", dest="format", type=str, default="", help="input format: pdf/txt")
	argparser.add_argument("-o", "--output_path", dest="output_path", type=str, default="", help="output_path")
	argparser.add_argument("-m", "--model_dir", dest="model_dir", type=str, default="", help="model_dir")
	argparser.add_argument("-k", "--keys_path", dest="keys_path", type=str, default="", help="keys_path")
	argparser.add_argument("--sources_para_idxs", dest="sources_para_idxs", type=str, default="0,1", help="source paragraph idxs")
	argparser.add_argument("-t", "--topn", dest="topn", type=int, default=None, help="top n similar")
	argparser.add_argument("-r", "--resume", dest="resume", default=False, action='store_true', help="resume")
	args = argparser.parse_args()

	config['resume'] = args.resume | config['resume']
	if args.input_dir:
		config['input_dir'] = args.input_dir
		config['resume'] = False
	if args.format:
		config['format'] = args.format
	if args.output_path:
		config['output_path'] = args.output_path
	if args.model_dir:
		config['model_dir'] = args.model_dir
	if args.keys_path:
		config['keys_path'] = args.keys_path
	if args.topn != None:
		config['topn'] = args.topn

	if config['resume']:
		with open(config['tmp_path'], "rb") as fp:
			pick = pickle.load(fp)
		print("load from : %s"%config['tmp_path'], [(k, len(val)) for k, val in pick.items()])
		keys, files, paras, para_vecs = pick['keys'], pick['files'], pick['paragraphs'], pick['para_vecs']
	else:
		if not config['input_dir']:
			print("ERROR: No input_dir.")
			sys.exit(0)
		if os.path.isdir(config['model_dir']):
			fs = [f for f in os.listdir(config['model_dir']) if f.endswith(".model")]
			argsorted = np.argsort([int(f.split('.')[-2].split('-')[-1]) for f in fs])
			fname = fs[argsorted[-1]]
			model_path = os.path.join(config['model_dir'], fname)
		elif os.path.exists(config['model_dir']):
			model_path = config['model_dir']
		else:
			print("ERROR: Invalid model_dir")
			sys.exit(0)
		print ("loading model:", model_path, " .....")
		model = gensim.models.Word2Vec.load(model_path)
	
		keys = parse_keys(config['keys_path'])
		data = []
		files = []
		fs = [os.path.join(config['input_dir'], f) for f in os.listdir(config['input_dir']) if f.endswith(config['format'])]
		print("read files:", len(fs))
		for f_id, f in enumerate(fs):
			text = read_file(f, config['format'])	
			files += [{"path": f, "text": text}] 
			data += [(f_id, text)]
		if not files:
			print("ERROR: No input file.")
			sys.exit(0)

		paras = parse_paragraphs(data, keys)
		para_vecs = [tokens_to_vec(para['keys'], model.wv) for para in paras]
		pick = {"keys":keys,"files":files,"paragraphs":paras, "para_vecs":para_vecs}

		resume_dir = "/".join(config['tmp_path'].split('/')[:1])
		os.system("rm -rf %s"%resume_dir)
		os.system("mkdir -p %s"%resume_dir)

		with open(config['tmp_path'], "wb") as fp:
			pickle.dump(pick, fp)
		print("save to : %s"%config['tmp_path'], [(k, len(val)) for k, val in pick.items()])

	sources_para_idxs = [int(idx) for idx in args.sources_para_idxs.split(',')]
	sources_vec = np.mean([para_vecs[idx] for idx in sources_para_idxs], axis=0)

	distances = list(calculate_distances(sources_vec, para_vecs))
	idxs = np.argsort(distances)[::-1]
	similarity_idx_scores = [(idx, distances[idx]) for idx in idxs[:config['topn']]]
	print("most similar files", idxs[:10])
	save_result(paras, keys, files, similarity_idx_scores, config)
	with open(config_path, 'w') as fj:
		json.dump(config, fj)


def save_result(paras, keys, files, similarity_idx_scores, config):
	new_line = ""
	out_file = sys.stdout
	if config['output_path'].endswith(".html"):
		out_file = open("html.txt", "w")
		new_line = "| "
		print(\
""".. pip install sphinx
.. rst2html.py html.txt html.html
.. raw:: html

	<style> .bold {font-weight:bold} </style>

	<style> .blue {color:blue} </style>

	<style> .red {color:red} </style>

.. role:: bold

.. role:: blue

.. role:: red
""", file=out_file)

	keys_regex = re.compile("(" + "|".join(keys) + ")", re.IGNORECASE)
	ret_str = []
	for para_id, score in similarity_idx_scores:
		para = paras[para_id]
		s, e = para['para_span']
		f_id, p_id = para["file_para_id"]
		text = files[f_id]['text']
		ret_str += [to_ref(files[f_id]['path']) + " (paragraph:%d  score:%.2f)"%(para_id, score)]
		ret_str += [to_marked(text[s:e].replace("\n", " "), keys_regex)]
	ret = new_line + ("\n\n"+new_line).join(ret_str)
	print(ret, file=out_file)

	if config['output_path'].endswith(".html"):
		out_file.close()
		os.system("rst2html.py html.txt > %s"%config['output_path'])
		print("save to : html.txt and %s"%config['output_path'], file=sys.stdout)
	else:
		print("save to : html.txt", file=sys.stdout)


def to_ref(file_name):
	return "`ref: " + file_name + " <" + file_name + ">`__"


def to_marked(text, regex):
	text_len = len(text)
	ret_str = ""
	start = 0
	first = True
	for match in regex.finditer(text):
		s, e = match.start(), match.end() 
		if s > 0:
			if text[s - 1].isalpha():
				continue
		if e < len(text):
			if text[e].isalpha():
				continue
		marker_end = markers["end"]
		ret_str += text[start:s] + markers["bold"] + text[s:e] + markers["end"]
		start = e
	if start < text_len:
		ret_str += text[start:text_len]

	return ret_str

	
if __name__ == '__main__':
	main()

import sys
import os
import re
import time
from multiprocessing import Pool, Value

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, 'resources/stopwords.txt'), 'r') as fd:
    stopwords = set([x for x in fd.read().lower().split() if len(x) > 2])

def is_valid_word(word):
	if len(word) > 2:
		if not word[:2].isupper():
			lower = word.lower()
			if lower[0] != lower[1]:
				if not re.findall('([^a-z]|\s)', lower) \
				and re.findall(r'[aeiou]', lower) \
				and lower not in stopwords:
					return True
	return False


counter = None
total = None
t0 = None
b_size = None
display_everys = None
verbose = None

def display():
	global counter
	if counter.value%display_every.value == 0:
		t_diff = time.time() - t0.value
		sub_total = counter.value*b_size.value
		print("\rworking: %d/%d	time: %.2f "%(sub_total, total.value, t_diff), end="    ")
		sys.stdout.flush()
	with counter.get_lock():
		counter.value += 1

def batch_work(i_func_batch):
	i, func, xs = i_func_batch
	if vbose.value:
		display()

	return i, [func(x) for x in xs]

def multi_processor_works(func, n_data, batch_size=1, display_every_batches=100, verbose=True):
	global counter
	global b_size
	global total
	global display_every
	global t0
	global vbose

	counter = Value('i', 0)
	b_size = Value('i', batch_size)
	total = Value('i', len(n_data))
	display_every = Value('i', display_every_batches)
	t0 = Value('d', time.time())
	vbose = Value('b', verbose)

	i_func_batches = []
	for i in range(0, len(n_data), batch_size):
		i_func_batches += [(i, func, n_data[i:i+batch_size])]

	pool = Pool()
	result = []
	for i, ret in sorted(list(pool.imap_unordered(batch_work, i_func_batches)), key=lambda x: x[0]):
		result += ret
	print()
	pool.close()
	pool.join()

	return result

#run testing
if __name__=="__main__":


	n_data = [(x, x) for x in range(1000000)]
	def funct(x):
		return x[0], x[1]**2.3

	t1 = time.time()
	ret = multi_processor_works(funct, n_data, batch_size=1, display_every_batches=100, verbose=True)
	print ("time:%.2f sec"%(time.time() - t1))
	print (len(ret), ret[:5], ret[-5:]);print("")

	t1 = time.time()
	ret = [funct(r) for r in n_data]
	print ("time:%.2f sec"%(time.time() - t1))
	print (len(ret), ret[:5], ret[-5:])

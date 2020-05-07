from os import makedirs
from os.path import splitext, join
from ntpath import basename
from Bio import SeqIO
from itertools import combinations
import random
from math import factorial
import numpy as np


def batch_iterator(iterator, batch_size):
	"""
	Groups sequences in batches with specified batch_size.

	This can be used on any iterator, for example to batch up
	SeqRecord objects from Bio.SeqIO.parse(...), or to batch
	Alignment objects from Bio.AlignIO.parse(...), or simply
	lines from a file handle.
	This is a generator function, and it returns lists of the
	entries from the supplied iterator.  Each list will have
	batch_size entries, although the final list may be shorter.
	#Credits: https://biopython.org/wiki/Split_large_file

	Parameters
	----------
	iterator : list of SeqIO
		list containing of SeqIO records

	Returns
	-------
	list of str
		list of batches of sequences
	"""
	entry = True  # Make sure we loop once
	while entry:
		batch = []
		while len(batch) < batch_size:
			try:
				# entry = iterator.next() #Python 2
				entry = next(iterator)
			except StopIteration:
				entry = None
			if entry is None:
				# End of file
				break
			batch.append(entry)
		if batch:
			yield batch


def sec2hour_min_sec(seconds):
	"""
	Convert elapsed seconds to hours, minutes and seconds
	#Credits: https://codereview.stackexchange.com/questions/174796/convert-seconds-to-hours-minutes-seconds-and-pretty-print

	Parameters
	----------
	seconds : long int
		elapsed seconds

	Returns
	-------
	str
		string with converted hours, minutes, seconds and microseconds
	"""
	microseconds = int(seconds * 1000000)

	if microseconds != 0:
		seconds, microseconds = divmod(microseconds, 1000000)
		minutes, seconds = divmod(seconds, 60)
		hours, minutes = divmod(minutes, 60)
		periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds), ('microseconds', microseconds)]
		return ', '.join('{} {}'.format(value, name) for name, value in periods if value)
	else:
		return str(microseconds) + ' microseconds'


def create_dir(base_path):
	"""
	Create directory specified by base path, if does not exist

	Parameters
	----------
	base_path : str
		full path of base directory

	Returns
	-------
	None
	"""
	makedirs(base_path, exist_ok=True)


def get_base_name(file):
	"""
	Get file base name

	Parameters
	----------
	file : str
		file name

	Returns
	-------
	str
		file base name
	"""
	return splitext(basename(file))[0]


def choose_combos(n, r, n_chosen):
	"""
	Choose n_chosen combination from (n choose r) total combinations
	# Credits: https://stackoverflow.com/questions/9874887/calculating-and-putting-all-possibilities-of-36-ncr-10-in-a-list-in-python

	Parameters
	----------
	n : int
		number of total elements (proteins domains)
	r : int
		coupling r elements to create a combination
	n_chosen : int
		number of chosen combinations

	Returns
	-------
	list of int
		list of chosen combinations
	"""

	total_combs = int(factorial(n) / (factorial(n - r) * factorial(r)))
	combos = combinations(range(n), r)
	chosen_indexes = random.sample(list(range(total_combs)), n_chosen)
	random_combos = []

	for i in list(range(total_combs)):
		ele = next(combos)
		if i in chosen_indexes:
			random_combos.append(ele)
	return random_combos


def split_fasta(base_path, base_path_out, fasta_name, split_size):
	"""
	Function to split fasta into sub files each containing up to split_size sequences

	Parameters
	----------
	base_path : str
		input base path where fasta with all proteins is located
	base_path_out : str
		output base path to save sub files
	fasta_name : str
		name of fasta file with all proteins
	split_size : int
		maximum number of sequences that the subfile can contain

	Returns
	-------
	None
	"""
	print("Splitting fasta into subfasta files.")
	fasta_name_prefix = get_base_name(fasta_name)
	record_iter = SeqIO.parse(open(join(base_path, fasta_name)), "fasta")
	for i, batch in enumerate(batch_iterator(record_iter, split_size)):
		batch_file_name = fasta_name_prefix + "_" + str(i + 1) + ".fasta"
		with open(join(base_path_out, batch_file_name), "w") as handle:
			count = SeqIO.write(batch, handle, "fasta")
		print("{} records written to {}".format(count, batch_file_name))


def write_random_vectors(emb_path, emb_file):
	"""
	Save random vectors to embedding file

	Parameters
	----------
	emb_path : str
		embedding path
	emb_file: str
		embedding file name

	Returns
	-------
	None
	"""
	assert emb_file[-4:] == ".txt", "AssertError: embedding file should be txt file"
	print("Creating random vectors for {}".format(emb_file))
	rand_vec_file = get_base_name(emb_file) + "_rand.txt"

	with open(emb_file, 'r') as file_in, open(join(emb_path, rand_vec_file), 'w') as file_out:
		rand_vecs_lines = []
		for i, line in enumerate(file_in):
			if i == 0:  # save header with number of domains and emb dim
				emb_dim = int(line.strip().split(" ")[1])
				rand_vecs_lines.append(line.strip())
			else:  # split on space to get domain id
				token_id = line.strip().split(" ")[0]
				token_rand_vec = np.random.random(emb_dim)
				rand_vecs_lines.append(" ".join([token_id] + [str(x) for x in token_rand_vec.tolist()]))

		file_out.write("\n".join(rand_vecs_lines))  # save random vector file


def get_freq_as_keys(freqs_dict):
	"""
	Function with input a {word:word_freq} dictionary and output {word_freq: freq_occurrence}

	Parameters
	----------
	freqs_dict: dict
		input dictionary {word: word_freq}

	Returns
	-------
	freq2freq_occurence : dict
		output dictionary {word_freq : freq_occurrence}
	"""
	freq2freq_occurence = {}
	for _, freq in freqs_dict.items():
		if freq not in freq2freq_occurence:
			freq2freq_occurence[freq] = 1
		else:
			freq2freq_occurence[freq] += 1
	return freq2freq_occurence


def is_interpro_domain(domain):
	"""
	Function to check if the input domain is Interpro or GAP or unknown domain (unk)

	Parameters
	----------
	domain : str
		domain string name to check if it is Interpro or not

	Returns
	-------
	bool
		True if the domain is an Interpro domain or not
	"""
	if domain[0:3] == "IPR" or domain[0:3] == "GAP":
		return True
	else:
		if len(domain.split("_")) > 1 and domain.split("_")[1] == "unk":
			return True
		else:
			return False

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from math import ceil, log10
from collections import deque, Counter

plt.style.use('seaborn-paper')


class Corpus:
	def __init__(self,data_path,file_in):
		self.data_path = data_path
		self.file_in = file_in

	def __iter__(self):
		with open(os.path.join(self.data_path,self.file_in),'r') as corpus_file:
			for line in corpus_file:
				yield line.strip().split(" ")

	def plot_proteins_len_hist(self):
		"""
		Plot histogram of proteins length
		Credits: https://stackoverflow.com/questions/47705972/how-to-plot-keys-and-values-from-dictionary-in-histogram
		Parameters
		----------
		self : object
			Corpus object set up for this analysis

		Returns
		-------
		None
		"""
		fig = plt.figure()
		proteins_length = self.calculate_line_histogram()
		protein_len = proteins_length.keys()
		protein_len_freq = proteins_length.values()

		plt.bar(protein_len, np.divide(list(protein_len), sum(protein_len_freq)), color='k')

		plt.ylim(0, 1)
		plt.ylabel('Proteins fraction', fontsize=14)
		plt.xlabel('Domains number', fontsize=14)
		plt.xticks(list(protein_len))

		hist_name = self.file_in.split(".")[0] + "_hist" + ".png"
		fig.savefig(os.path.join(self.data_path, hist_name), bbox_inches='tight', dpi=600)

	def plot_line_histogram(self):
		### credits ###
		# https://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
		###         ###
		line_word_length = self.calculate_line_histogram()
		fig = plt.figure()
		lines_length = np.asarray(list(line_word_length.values()), dtype=np.int64)

		plt.hist(lines_length,bins='auto', color='k', align='left', edgecolor='k', alpha=0.8,density=True, histtype='bar', cumulative=True)
		plt.xticks(rotation=90)
		plt.yscale('log')
		#plt.xscale('log',xbase=2)
		#plt.title('Histogram of domains per protein', fontsize=14)
		plt.xlabel('Number of domains', fontsize=14)
		plt.ylabel('Number of proteins', fontsize=14)

		hist_name = self.file_in.split(".")[0] + "_hist" + ".png"
		fig.savefig(os.path.join(self.data_path,hist_name),bbox_inches='tight',dpi=600)

	def plot_line_histogram_log_axes(self):
		"""
		Plot hist with log axes (worked best compared to bins='auto' in Jan 2020)
		Parameters
		----------
		self : object
			Corpus object for this analysis

		Returns
		-------
		None
		"""
		line_word_length = self.calculate_line_histogram()
		fig = plt.figure()
		plt.bar(list(line_word_length.keys()), line_word_length.values(), color='g')
		plt.xticks(rotation=90)
		plt.yscale('log')
		plt.xscale('log', basex=2)
		plt.title('Histogram of domains per protein', fontsize=14)
		plt.xlabel('Number of domains ($log_2$)', fontsize=14)
		plt.ylabel('Number of proteins ($log_10$)', fontsize=14)

		hist_name = self.file_in.split(".")[0] + "_logx" + ".png"
		fig.savefig(os.path.join(self.data_path, hist_name), bbox_inches='tight', dpi=600)

	def create_domain_counts_steps(self,max_domain_count):
		if max_domain_count < 10:
			print("<10")
			steps = np.arange(1, max_domain_count)
		elif max_domain_count < 100:
			print("<100")
			linear_bins = np.arange(1, 10)
			bins_10 = np.arange(10, max_domain_count, 10)
			steps = np.concatenate((linear_bins, bins_10), axis=0)
		else:
			print("<1000000")
			linear_bins = np.arange(1, 10)
			bins_10 = np.arange(10, 100, 10)
			exp_bins = np.logspace(100, ceil(log10(max_domain_count)), num=ceil(log10(max_domain_count)) + 1, base=10,
			                       dtype='int')
			steps = np.concatenate((linear_bins, bins_10, exp_bins), axis=0)
		return steps

	def plot_domains_count_histogram(self):
		"""
		Function to plot domains count histogram
		Credits:https://stackoverflow.com/questions/35596128/how-to-generate-a-word-frequency-histogram-where-bars-are-ordered-according-to#35603850

		Parameters
		----------
		self : object
			Corpus object for this analysis

		Returns
		-------
		None
		"""
		counts = Counter(self.get_domains_count())

		labels, values = zip(*counts.items())

		# sort your values in descending order
		indSort = np.argsort(labels)[::-1]

		# rearrange your data
		labels = np.array(labels)[indSort]
		values = np.array(values)[indSort]

		indexes = np.arange(len(labels))
		# indexes = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100])
		# indexes = self.create_domain_counts_steps(max(labels))
		bar_width = 0.35

		plt.bar(indexes, values)

		# add labels
		steps = self.create_domain_counts_steps(max(labels))
		plt.xticks(steps + bar_width, labels)
		plt.show()

	def get_domains_count(self):
		all_proteins_domains_count = []
		with open(os.path.join(self.data_path,self.file_in),'r') as corpus_file:
			for protein in corpus_file:
				domains_count = len(protein.strip().split(" "))
				all_proteins_domains_count.append(domains_count)
		return all_proteins_domains_count

	def calculate_line_histogram(self):
		line_word_length = {}
		with open(os.path.join(self.data_path,self.file_in),'r') as corpus_file:
			for line in corpus_file:
				num_words = len(line.strip().split(" "))
				if num_words not in line_word_length:
					line_word_length[num_words] = 1
				else:
					line_word_length[num_words] += 1

		return line_word_length

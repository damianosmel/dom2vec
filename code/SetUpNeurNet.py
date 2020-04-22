import torch
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from gensim.models import KeyedVectors
from torchtext import data
from RNN import RNN
from CNN import CNN
from CNN1d import CNN1d
from SeqVecNet import SeqVecNet
from SeqVecCharNet import SeqVecCharNet
from FastText import FastText

from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path

import os
import random
from utils import create_dir, basename
from math import exp
import pandas as pd


class SetUpNeurNet:
	"""
	Class to set up the network in order to allow multiple runs with the same set-up.
	Credits: https://github.com/bentrevett/pytorch-sentiment-analysis
	"""

	SEED = 1234
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	def __init__(self, data_path, data_train_file, data_test_file, out_path, label_name, emb_name, use_emb, emb_file,
	             is_emb_bin, freeze_emb, emb_dim, normalize_emb, is_binary_class, k_fold):
		self.data_path = data_path
		self.data_train_file = data_train_file
		self.data_test_file = data_test_file
		self.total_data_file = None  # only for DeepLoc to load the embedding representations for words both in train and test
		self.out_path = out_path
		self.label_name = label_name
		self.models = []
		self.optimizers = []
		self.criteria = []
		self.emb_name = emb_name
		self.use_emb = use_emb
		self.emb_file = emb_file
		self.is_emb_bin = is_emb_bin
		self.freeze_emb = freeze_emb
		self.emb_dim = emb_dim
		self.to_normalize_emb = normalize_emb
		self.is_binary_class = is_binary_class
		self.k_fold = k_fold
		self.best_model_name = ""
		self.label_weights = []
		create_dir(out_path)

	def count_parameters(self, model_idx):
		return sum(p.numel() for p in self.models[model_idx].parameters() if p.requires_grad)

	def normalize_emb(self, emb_file):
		print("Normalizing embeddings")
		txt_name = os.path.basename(emb_file).split(".txt")[0] + "_norm" + ".txt"
		out_path = os.path.dirname(emb_file)
		emb_txt_file = os.path.join(out_path, txt_name)
		emb_model = KeyedVectors.load_word2vec_format(emb_file)
		emb_model.init_sims(replace=True)
		emb_model.save_word2vec_format(emb_txt_file, binary=False)
		return emb_txt_file

	def convert_bin_emb_txt(self, emb_file):
		"""
		Use gensim util to convert binary embeddings file to txt

		:param emb_file: full path of embeddings file in binary format
		:return: full path of embeddings file in txt format
		"""
		txt_name = os.path.basename(emb_file) + ".txt"
		out_path = os.path.dirname(emb_file)
		emb_txt_file = os.path.join(out_path, txt_name)
		emb_model = KeyedVectors.load_word2vec_format(emb_file, binary=True)
		emb_model.save_word2vec_format(emb_txt_file, binary=False)
		return emb_txt_file

	def generate_bigrams(self, x):
		# Credits: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb
		n_grams = set(zip(*[x[i:] for i in range(2)]))
		for n_gram in n_grams:
			x.append(' '.join(n_gram))
		return x

	def generate_trimers(self, x):
		# Generate all trigrams of amino sequence
		# Following Figure 1 of
		# Asgari, Ehsaneddin, and Mohammad RK Mofrad. "Continuous distributed representation of biological
		# sequences for deep proteomics and genomics." PloS one 10.11 (2015): e0141287.
		# Example: MNTPA -> MNT, NTP, TPA, NTP, TPA, TPA
		all_trigrams = []
		x = x[0]
		if len(x) < 3:
			all_trigrams.append(x)
		else:
			for start_idx in range(3):
				for mer_idx in range(len(x[start_idx::]) - 3 + 1):
					temp = x[start_idx::]
					all_trigrams.append(temp[mer_idx:mer_idx + 3])
		# print(' '.join(all_trigrams))
		return ' '.join(all_trigrams)

	@staticmethod
	def generate_residues(x):
		# Generate all residues
		# For example: x = 'MNTPA' --> 'M N T P A'
		return ' '.join(list(x))

	def merge_train_test(self):
		print("To load quicker the {} embeddings, merge {} and {} and load embeddings only for them.".format(
			self.emb_name, self.data_train_file, self.data_test_file))
		df_train = pd.read_csv(os.path.join(self.data_path, self.data_train_file), sep=",", header=0)
		df_test = pd.read_csv(os.path.join(self.data_path, self.data_test_file), sep=",", header=0)
		df_train_test = pd.concat([df_train, df_test], axis=0, ignore_index=True)
		train_name = self.data_train_file.split(".csv")[0]
		train_test_name = train_name + "_test.csv"
		df_train_test.to_csv(os.path.join(self.data_path, train_test_name), sep=",", index=False)
		print("Saving train+test with shape: {}".format(df_train_test.shape))
		self.total_data_file = train_test_name

	def config_train_val_test(self, dataset_name, model_type, use_uni_bi):
		# SeqVec takes time to load all embeddings for the whole data set
		# so load only the needed ones
		if self.k_fold == -1 and self.emb_name == "seqvec":  # and self.use_emb == 1
			self.merge_train_test()
		if dataset_name == "DeepLoc":
			"""
			###############
			### DeepLoc ###
			###############
			##
			# csv: [ 'train_test'   'cellular_location' 'membrane_soluble' 'seq' 'interpro_domains']
			# use: [  train_test     cel_loc (label)      mem_sol (label)   seq   domains]
			##
			"""
			if self.label_name == "cellular_location":
				self.TRAIN_TEST = data.Field(dtype=torch.float)
				self.LABEL = data.LabelField()  # multi-class
				self.MEMBR_SOL = data.Field(dtype=torch.float)
				if self.emb_name == "protvec":
					print("Processing amino acid sequence to get trimers")
					self.SEQUENCE = data.Field(sequential=True, include_lengths=True,
					                           preprocessing=self.generate_trimers)
					self.fields = [('train_test', self.TRAIN_TEST), ('label', self.LABEL), ('mem_sol', self.MEMBR_SOL),
					               ('seq', self.SEQUENCE), (None, None)]
				elif self.emb_name == "seqvec":
					print("Processing amino acid sequence to get residues")
					self.SEQUENCE = data.Field(sequential=True, include_lengths=True)
					self.fields = [('train_test', self.TRAIN_TEST), ('label', self.LABEL), ('mem_sol', self.MEMBR_SOL),
					               ('seq', self.SEQUENCE), (None, None)]
				elif self.emb_name == "dom2vec":
					print("Processing domains")
					if model_type == "FastText":
						if use_uni_bi:
							print("Generate uni- and bi-gram domains for FastText")
							self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True,
							                                   preprocessing=self.generate_bigrams)
						else:
							print("Generate uni-gram domains for FastText")
							self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)
					else:
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
					self.fields = [('train_test', self.TRAIN_TEST), ('label', self.LABEL), ('mem_sol', self.MEMBR_SOL),
					               (None, None),
					               ('domains', self.INTERPRO_DOMAINS)]
			else:  # binary
				self.TRAIN_TEST = data.Field(dtype=torch.float)
				self.CELLULAR_LOC = data.Field(dtype=torch.float)
				self.LABEL = data.LabelField(dtype=torch.float)  # binary: membrane or soluble
				if self.emb_name == "protvec":
					print("Processing amino acid sequence to get trimers")
					self.SEQUENCE = data.Field(sequential=True, include_lengths=True,
					                           preprocessing=self.generate_trimers)
					self.fields = [('train_test', self.TRAIN_TEST), ('cel_loc', self.CELLULAR_LOC),
					               ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
				elif self.emb_name == "seqvec":
					print("Processing amino acid sequence to get residues")
					# generate_residues
					self.SEQUENCE = data.Field(sequential=True,
					                           include_lengths=True)
					self.fields = [('train_test', self.TRAIN_TEST), ('cel_loc', self.CELLULAR_LOC),
					               ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
				elif self.emb_name == "dom2vec":
					print("Processing domains")
					if model_type == "FastText":
						if use_uni_bi:
							print("Generate uni and bi-gram domains for FastText")
							self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True,
							                                   preprocessing=self.generate_bigrams)
						else:
							print("Generate uni-gram domains for FastText")
							self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)
					else:
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
					self.fields = [('train_test', self.TRAIN_TEST), ('cel_loc', self.CELLULAR_LOC),
					               ('label', self.LABEL), (None, None),
					               ('domains', self.INTERPRO_DOMAINS)]

			# if SeqVec not selected
			# to build embeddings, get all words of train + test
			if self.emb_name != "seqvec":
				if len(basename(self.data_test_file).split("_test")) > 1:  # get all test OOV splits
					# the train file is given, so get the full data set file name
					self.total_data_file = basename(self.data_test_file).split("_test")[0] + ".csv"
			self.total_data, self.total_train_data, self.test_data = data.TabularDataset.splits(
				path=self.data_path,
				train=self.total_data_file,
				validation=self.data_train_file,
				test=self.data_test_file,
				format='csv',
				fields=self.fields,
				skip_header=True)

			if self.k_fold == -1:  # train, test
				self.train_data, self.valid_data = self.total_train_data.split(random_state=random.seed(self.SEED),
				                                                               split_ratio=0.8)  # get train and val splits
			elif self.k_fold == 0:  # train, val and test
				self.train_data, self.valid_data = self.total_train_data.split(random_state=random.seed(self.SEED),
				                                                               split_ratio=0.8)
			else:  # k-fold inner cross validation
				# Read each train and validation sets of a fold and save to pytorch data set
				self.fold_sets = []
				for i in range(0, self.k_fold):
					train_data, valid_data = data.TabularDataset.splits(
						path=self.data_path,
						train='train_fold_' + str(i) + '.csv',
						validation='val_fold_' + str(i) + '.csv',
						format='csv',
						fields=self.fields,
						skip_header=True)
					self.fold_sets.append((train_data, valid_data))

		elif dataset_name == "TargetP":
			"""
			###############
			### TargetP ###
			###############
			##
			# csv: [ 'uniprot_id' 'cellular_location' 'seq' 'interpro_domains']
			# use: [  None         cel_loc (label)     seq   domains]
			##
			"""
			assert self.label_name == "cellular_location", "AssertionError: this data set has only cellular location as label."  # this data set has only one label
			self.LABEL = data.LabelField()  # multi-class
			if self.emb_name == "protvec":
				print("Processing amino acid sequence to get trimers")
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True,
				                           preprocessing=self.generate_trimers)
				self.fields = [(None, None), ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "seqvec":
				print("Processing amino acid sequence to get residues")
				# generate_residues
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True)
				self.fields = [(None, None),('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "dom2vec":
				print("Processing domains")
				if model_type == "FastText":
					if use_uni_bi:
						print("Generate uni- and bi-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True,
						                                   preprocessing=self.generate_bigrams)
					else:
						print("Generate uni-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)
				else:
					self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
				self.fields = [(None, None), ('label', self.LABEL), (None, None), ('domains', self.INTERPRO_DOMAINS)]

			# if SeqVec embeddings are not selected
			# to build embeddings, get all words of train + test
			if self.emb_name != "seqvec":
				# if len(basename(self.data_test_file).split("_test.csv")) > 1:
				if len(basename(self.data_test_file).split("_test")) > 1:  # get all test OOV splits
					# the train file is given, so get the full data set file name
					# self.total_data_file = basename(self.data_test_file).split("_test.csv")[0] + ".csv"
					self.total_data_file = basename(self.data_test_file).split("_test")[0] + ".csv"
			self.total_data, self.total_train_data, self.test_data = data.TabularDataset.splits(
				path=self.data_path,
				train=self.total_data_file,
				validation=self.data_train_file,
				test=self.data_test_file,
				format='csv',
				fields=self.fields,
				skip_header=True)

			if self.k_fold == -1:  # train, test
				self.train_data, self.valid_data = self.total_train_data.split(random_state=random.seed(self.SEED),
				                            split_ratio=0.8) #get train and val splits
			else:  # k-fold inner cross validation
				assert self.k_fold == 3, "AssertionError: TargetP data set has only 3-fold splits."
				# Read each train and validation sets of a fold and save to pytorch data set
				self.fold_sets = []
				for i in range(0, self.k_fold):
					train_data, valid_data = data.TabularDataset.splits(
						path=self.data_path,
						train='train_fold_' + str(i) + '.csv',
						validation='val_fold_' + str(i) + '.csv',
						format='csv',
						fields=self.fields,
						skip_header=True)
					self.fold_sets.append((train_data, valid_data))
		elif dataset_name == "Toxin":
			"""
			#############
			### Toxin ###
			#############
			##
			# csv: [ 'uniprot_id'  'toxin'     'seq' 'interpro_domains']
			# use: [  None       tox(label)  seq   domains]
			##
			"""
			assert self.label_name == "toxin", "AssertionError: this data set has only toxin as label."  # this data set has only one label
			self.LABEL = data.LabelField(dtype=torch.float)  # binary: membrane or soluble
			if self.emb_name == "protvec":
				print("Processing amino acid sequence to get trimers")
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True,
				                           preprocessing=self.generate_trimers)
				self.fields = [(None, None), ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "seqvec":
				print("Processing amino acid sequence to get residues")
				# generate_residues
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True)
				self.fields = [(None, None), ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "dom2vec":
				print("Processing domains")
				if model_type == "FastText":
					if use_uni_bi:
						print("Generate uni- and bi-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True,
						                                   preprocessing=self.generate_bigrams)
					else:
						print("Generate uni-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)
				else:
					self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
				self.fields = [(None, None), ('label', self.LABEL), (None, None), ('domains', self.INTERPRO_DOMAINS)]

			# if SeqVec embeddings are not selected
			# to build embeddings, get all words of train + test
			if self.emb_name != "seqvec":
				if len(basename(self.data_test_file).split("_test.csv")) > 1:
					self.total_data_file = basename(self.data_test_file).split("_test.csv")[0] + ".csv"

			self.total_data, self.total_train_data, self.test_data = data.TabularDataset.splits(
				path=self.data_path,
				train=self.total_data_file,
				validation=self.data_train_file,
				test=self.data_test_file,
				format='csv',
				fields=self.fields,
				skip_header=True)

			if self.k_fold == -1:  # train,val /test splits
				self.train_data, self.valid_data = self.total_train_data.split(random_state=random.seed(self.SEED),
				                                                               split_ratio=0.8)  # get train and val splits
			else:  # 3-fold inner cross validation
				assert self.k_fold == 3, "AssertionError: Toxin data set has only 3-fold splits."
				# Read each train and validation sets of a fold and save to pytorch data set
				self.fold_sets = []
				for i in range(0, self.k_fold):
					train_data, valid_data = data.TabularDataset.splits(
						path=self.data_path,
						train='train_fold_' + str(i) + '.csv',
						validation='val_fold_' + str(i) + '.csv',
						format='csv',
						fields=self.fields,
						skip_header=True)
					self.fold_sets.append((train_data, valid_data))
		elif dataset_name == "NEW":
			"""
			###########
			### NEW ###
			###########
			enzyme class interpro_domains
			##
			# csv: [ 'id'    'ec'        'seq' 'interpro_domains']
			# use: [  None    ec (label)  seq   domains]
			##
			"""
			assert self.label_name == "ec", "AssertionError: this data set has only ec as label."  # this data set has only one label
			self.LABEL = data.LabelField()  # multi-class

			if self.emb_name == "protvec":  # process sequence
				print("Processing amino acid sequence to get trimers")
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True, preprocessing=self.generate_trimers)
				self.fields = [(None, None), ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "seqvec":
				print("Processing amino acid sequence to get residues")
				# generate_residues
				self.SEQUENCE = data.Field(sequential=True, include_lengths=True)
				self.fields = [(None, None), ('label', self.LABEL), ('seq', self.SEQUENCE), (None, None)]
			elif self.emb_name == "dom2vec":
				print("Processing domains")
				if model_type == "FastText":
					if use_uni_bi:
						print("Generate uni- and bi-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True,
						                                   preprocessing=self.generate_bigrams)
					else:
						print("Generate uni-gram domains for FastText")
						self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)
				else:
					self.INTERPRO_DOMAINS = data.Field(sequential=True, include_lengths=True)  # split on spaces
				self.fields = [(None, None), ('label', self.LABEL), (None, None), ('domains', self.INTERPRO_DOMAINS)]

			# if SeqVec embeddings are not selected
			# to build embeddings, get all words of train + test
			if self.emb_name != "seqvec":
				if len(basename(self.data_test_file).split("_test.csv")) > 1:
					self.total_data_file = basename(self.data_test_file).split("_test.csv")[0] + ".csv"

			self.total_data, self.total_train_data, self.test_data = data.TabularDataset.splits(
				path=self.data_path,
				train=self.total_data_file,
				validation=self.data_train_file,
				test=self.data_test_file,
				format='csv',
				fields=self.fields,
				skip_header=True)

			if self.k_fold == -1:  # train/test split
				self.train_data, self.valid_data = self.total_train_data.split(random_state=random.seed(self.SEED),
				                                                               split_ratio=0.8)  # get train and val splits
			else:  # 3-fold inner cross validation
				assert self.k_fold == 3, "AssertionError: NEW data set has only 3-fold splits."
				# Read each train and validation sets of a fold and save to pytorch data set
				self.fold_sets = []
				for i in range(0, self.k_fold):
					train_data, valid_data = data.TabularDataset.splits(
						path=self.data_path,
						train='train_fold_' + str(i) + '.csv',
						validation='val_fold_' + str(i) + '.csv',
						format='csv',
						fields=self.fields,
						skip_header=True)
					self.fold_sets.append((train_data, valid_data))
		print("=== Stats of examples ===")
		print("Data set first instance: {}".format(vars(self.total_data[0])))
		if self.k_fold == -1:
			print('Number of training examples: {}'.format(len(self.train_data)))
			print('Number of validation examples: {}'.format(len(self.valid_data)))
		elif self.k_fold == 0:
			print('Number of training examples: {}'.format(len(self.train_data)))
			print('Number of validation examples: {}'.format(len(self.valid_data)))
		else:
			for i in range(0, self.k_fold):
				print("=== Fold {} ===".format(i))
				print("Number of training examples: {}".format(len(self.fold_sets[i][0])))
				print("Number of validation examples: {}".format(len(self.fold_sets[i][1])))
		print("=== Test ===")
		print('Number of testing examples: {}'.format(len(self.test_data)))
		print("=== --- ===")

	def build_domain_vocs(self):
		print("Building domain vocabulary")
		MAX_VOCAB_SIZE = 50_000  # for interpro
		if self.use_emb:
			if self.is_emb_bin:
				self.emb_file = self.convert_bin_emb_txt(self.emb_file)
			if self.to_normalize_emb:
				self.emb_file = self.normalize_emb(self.emb_file)
			print("Loading embeddings from: {}".format(self.emb_file))
			custom_embeddings = vocab.Vectors(name=self.emb_file,
			                                  cache=os.path.join(self.out_path, "custom_embeddings"),
			                                  unk_init=torch.Tensor.normal_)

			print("Building vocabulary for protein domains in the whole data set from " + self.total_data_file)
			if self.k_fold == -1:
				self.INTERPRO_DOMAINS.build_vocab(self.total_data,
			                                    max_size=MAX_VOCAB_SIZE,
			                                    vectors=custom_embeddings,
			                                    unk_init=torch.Tensor.normal_)
			elif self.k_fold == 3:
				self.INTERPRO_DOMAINS.build_vocab(self.total_data,
				                                  max_size=MAX_VOCAB_SIZE,
				                                  vectors=custom_embeddings,
				                                  unk_init=torch.Tensor.normal_)
		else:
			print("Do not load embeddings, convert 1-hot to dimensional vectors for protein domain in whole data set.")
			if self.k_fold == -1:
				self.INTERPRO_DOMAINS.build_vocab(self.total_data, max_size=MAX_VOCAB_SIZE)
			elif self.k_fold == 3:
				self.INTERPRO_DOMAINS.build_vocab(self.total_data, max_size=MAX_VOCAB_SIZE)
		return len(self.INTERPRO_DOMAINS.vocab)

	def build_trigram_vocs(self):
		print("Building trigram vocabulary")
		MAX_VOCAB_SIZE = 100_000  # for protvec
		if self.use_emb:
			print("Loading embeddings from: {}".format(self.emb_file))
			custom_embeddings = vocab.Vectors(name=self.emb_file,
			                                  cache=os.path.join(self.out_path, "custom_embeddings"),
			                                  unk_init=torch.Tensor.normal_)
			print("Building vocabulary for all trimers of proteins in the whole data set from " + self.total_data_file)
			self.SEQUENCE.build_vocab(self.total_data,
			                          max_size=MAX_VOCAB_SIZE,
			                          vectors=custom_embeddings,
			                          unk_init=torch.Tensor.normal_)
		else:
			print(
				"Do not load embeddings, convert 1-hot to dimensional vectors for trigrams of protein sequence in whole data set.")
			self.SEQUENCE.build_vocab(self.total_data, max_size=MAX_VOCAB_SIZE)
		return len(self.SEQUENCE.vocab)

	def build_residue_vocs(self):
		print("Building residue vocabulary")
		MAX_VOCAB_SIZE = 50_000  # each protein instance as a voc word

		print("Converting 1-hot dimensional vectors for all proteins in the whole data set from " + self.total_data_file)
		self.SEQUENCE.build_vocab(self.total_data, max_size=MAX_VOCAB_SIZE)
		return len(self.SEQUENCE.vocab)

	def build_vocs(self, dataset_name):
		# build voc for X matrix of features
		voc_size = 0
		if self.emb_name == "protvec":
			voc_size = self.build_trigram_vocs()
		elif self.emb_name == "seqvec":
			voc_size = self.build_residue_vocs()
		elif self.emb_name == "dom2vec":
			voc_size = self.build_domain_vocs()

		# build voc for extra column found only in the DeepLoc data sets
		if dataset_name == "DeepLoc":
			if self.label_name == "cellular_location":
				self.MEMBR_SOL.build_vocab(self.total_data)
			else:
				self.CELLULAR_LOC.build_vocab(self.total_data)
			self.TRAIN_TEST.build_vocab(self.total_data)  # build vocabulary for TRAIN_TEST

		self.LABEL.build_vocab(self.total_train_data)
		print("=== Stats for variables ===")
		if self.emb_name == "protvec":
			print("Number of unique tri-grams: {}".format(voc_size))
		elif self.emb_name == "seqvec":
			print("Number of unique proteins: {}".format(voc_size))
		elif self.emb_name == "dom2vec":
			print("Number of unique domains: {}".format(voc_size))
		self.print_labels_info()
		print("=== --- ===")

	def print_labels_info(self):
		print("=== Labels ===")
		print("Number of labels: {}".format(len(self.LABEL.vocab)))
		for label_index, label in enumerate(self.LABEL.vocab.itos):
			print("{} -> {}".format(label_index, label))

	def calculate_label_weights(self, scaling):
		print("Calculate label weights")
		label_freqs = []
		for label_name in self.LABEL.vocab.itos:
			label_freqs.append(self.LABEL.vocab.freqs[label_name])
			print("label: {}, freq: {}".format(label_name, self.LABEL.vocab.freqs[label_name]))

		p = [round(freq_i / sum(label_freqs), 2) for freq_i in label_freqs]
		assert 0.99999 <= round(sum(p),
		                        1) <= 1.0, "AssertionError: Sum of label probabilities should be 0.99999 <= p <= 1.0."
		p_compl = [1 - p_i for p_i in p]
		if scaling == "linear":
			self.label_weights = torch.FloatTensor(p_compl)
		elif scaling == "exponential":
			self.label_weights = torch.FloatTensor([exp(p_compl_i) for p_compl_i in p_compl])
		elif scaling == "1over":
			self.label_weights = torch.FloatTensor([1 / p_i for p_i in p])
		elif scaling == "over_average":
			"""
			Inspired by "Logistic Regression in RareEvents Data" by Gary King and Langche Zeng.
			"""
			avg = sum(label_freqs) / len(label_freqs)
			self.label_weights = torch.FloatTensor([avg / freq_i for freq_i in label_freqs])
		elif scaling == "norm":
			max_p_compl = max(p_compl)
			min_p_compl = min(p_compl)
			max_min_range = max_p_compl - min_p_compl
			weights = [((p_compl_i - min_p_compl) / max_min_range) for p_compl_i in p_compl]
			self.label_weights = torch.FloatTensor([0.01 if weight == 0.0 else weight for weight in weights])  # 0.001
		elif scaling == "default":
			self.label_weights = torch.FloatTensor([1.0 for i in range(len(p_compl))])
		elif scaling == "balanced_scikit":
			self.label_weights = torch.FloatTensor(
				[sum(label_freqs) / (len(label_freqs) * label_freq) for label_freq in label_freqs])
		elif scaling == "custom":
			self.label_weights = torch.FloatTensor([1.0, 4.0]) #for toxin does not work well.
		else:
			raise NotImplementedError
		for i in range(len(self.LABEL.vocab.itos)):
			print("label: {} -> weight: {}".format(self.LABEL.vocab.itos[i], self.label_weights[i]))
		print("-----")

	def build_sequence_iterators(self, batch_size, dataset_name):
		print("Building sequence iterators")
		BATCH_SIZE = batch_size
		if dataset_name == "DeepLoc":
			if self.k_fold == -1:  # train and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.seq,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			elif self.k_fold == 0:  # train, val and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.seq,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:  # k-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: DeepLoc has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.seq,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "TargetP":
			if self.k_fold == -1:  # train and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.seq,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:  # 3-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: TargetP has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.seq,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "Toxin":
			if self.k_fold == -1:  # train, val & test
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.seq,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:  # 3-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: Toxin has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.seq,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "NEW":
			if self.k_fold == -1:  # train/test
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.seq,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:  # 3-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: NEW has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.seq,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

	def build_domain_iterators(self, batch_size, dataset_name):
		print("Buidling domain iterators")
		BATCH_SIZE = batch_size
		if dataset_name == "DeepLoc":
			if self.k_fold == -1:  # train and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.domains,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			elif self.k_fold == 0:  # train, val and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.domains,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:  # k-fold cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: DeepLoc has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.domains,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "TargetP":
			if self.k_fold == -1:  # train, val and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.domains,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:
				assert self.k_fold == 3, "AssertionError: TargetP has only 3-fold splits."
				# 3-fold inner cross validation
				# Add train and validation iterators
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.domains,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "Toxin":
			if self.k_fold == -1:  # train, val and test iter
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.domains,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:
				# 3-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: Toxin has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.domains,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

		elif dataset_name == "NEW":
			if self.k_fold == -1:
				self.train_iterator, self.valid_iterator, self.test_iterator = data.BucketIterator.splits(
					(self.train_data, self.valid_data, self.test_data),
					sort_key=lambda x: x.domains,
					batch_size=BATCH_SIZE,
					sort_within_batch=True,
					device=self.device)
			else:
				# 3-fold inner cross validation
				# Add train and validation iterators
				assert self.k_fold == 3, "AssertionError: NEW has only 3-fold splits."
				self.fold_iterators = []
				for i in range(0, self.k_fold):
					train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
						(self.fold_sets[i][0], self.fold_sets[i][1], self.test_data),
						sort_key=lambda x: x.domains,
						batch_size=BATCH_SIZE,
						sort_within_batch=True,
						device=self.device)
					self.fold_iterators.append((train_iterator, valid_iterator))
				self.test_iterator = test_iterator

	def build_iterators(self, batch_size, dataset_name):
		self.batch_size = batch_size
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if self.emb_name == "protvec" or self.emb_name == "seqvec":
			self.build_sequence_iterators(batch_size, dataset_name)
		elif self.emb_name == "dom2vec":
			self.build_domain_iterators(batch_size, dataset_name)

	def create_best_model_name(self, epoch):
		"""
		Assuming the embedding file is on the dir hierarchy
		~/overlap or graph/embeddings_dir/embeddings_file.bin or txt
		:return: best model name, should be like:
		LSTM: overlap_dom2vec_w2_cbow0_hierSoft0_dim50_e5_LSTM_bi1_hid256
		CNN: overlap_dom2vec_w2_cbow0_hierSoft0_dim50_e5_CNN_num100_sizes_1_3_5
		"""
		if self.use_emb:
			way_emb_trained = os.path.basename(os.path.dirname(os.path.dirname(self.emb_file)))
			emb_parameters = os.path.basename(self.emb_file).split(".")[0]
			emb_prefix = way_emb_trained + "_" + emb_parameters
		else:
			emb_prefix = "1hot"
		if self.model_type == "LSTM":
			if self.is_bidirectional:
				bi = "bi"
			else:
				bi = "uni"
			hid_dim = "hid" + str(self.hid_dim)
			model_suffix = "_".join([bi, hid_dim])
		elif self.model_type == "CNN":
			num_filters = "num" + str(self.num_filters)
			filters_size = "sizes_" + "_".join([str(filter_size) for filter_size in self.filter_sizes])
			model_suffix = "_".join([num_filters, filters_size])
		elif self.model_type == "FastText":
			model_suffix = ""
		model_suffix = model_suffix + "_e" + str(epoch)
		return "_".join([emb_prefix, self.model_type, model_suffix]) + ".pt"

	def config_RNN_model(self, model_type, hid_dim, is_bidirectional, n_layers, dropout):
		print("Configuring RNN model.")
		assert self.emb_name == "dom2vec", "AssertionError: RNN is implemented only for domains"
		self.model_type = model_type
		self.hid_dim = hid_dim
		self.is_bidirectional = is_bidirectional
		INPUT_DIM = len(self.INTERPRO_DOMAINS.vocab)
		EMBEDDING_DIM = self.emb_dim
		HIDDEN_DIM = hid_dim
		if self.is_binary_class:
			OUTPUT_DIM = 1
		else:
			OUTPUT_DIM = len(self.LABEL.vocab)
		N_LAYERS = n_layers
		BIDIRECTIONAL = self.is_bidirectional
		if N_LAYERS == 1:
			DROPOUT = dropout
		else:
			DROPOUT = dropout
		PAD_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.pad_token]
		if self.k_fold == -1 or self.k_fold == 0:  # train/test, train/val
			model = RNN(INPUT_DIM,
			            EMBEDDING_DIM,
			            HIDDEN_DIM,
			            OUTPUT_DIM,
			            N_LAYERS,
			            BIDIRECTIONAL,
			            DROPOUT,
			            PAD_IDX)
			model.embedding.weight.requires_grad = not self.freeze_emb
			self.models.append(model)
		else:  # k-fold
			for model_idx in range(self.k_fold):
				model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
				model.embedding.weight.requires_grad = not self.freeze_emb
				self.models.append(model)
		# init embeddings, init pad and unk, print model info
		for model_idx in range(len(self.models)):
			print("=== Model {} ===".format(model_idx))
			if self.use_emb:
				self.init_emb(model_idx)
			self.init_pad_unk(model_idx)
			self.print_model_config_stats(model_idx)
			print("=== --- ===")

	def config_CNN_model(self, model_type, num_filters, filter_sizes, dropout):
		print("Configuring CNN model.")
		assert self.emb_name == "dom2vec", "AssertionError: CNN is implemented only for domains"
		self.model_type = model_type
		self.num_filters = num_filters
		self.filter_sizes = filter_sizes
		INPUT_DIM = len(self.INTERPRO_DOMAINS.vocab)
		EMBEDDING_DIM = self.emb_dim
		N_FILTERS = num_filters
		FILTER_SIZES = filter_sizes
		if self.is_binary_class:
			OUTPUT_DIM = 1
		else:
			OUTPUT_DIM = len(self.LABEL.vocab)
		DROPOUT = dropout
		PAD_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.pad_token]
		if self.k_fold == -1 or self.k_fold == 0:
			model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
			# model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
			model.embedding.weight.requires_grad = not self.freeze_emb
			self.models.append(model)
		else:  # k-fold
			for model_idx in range(self.k_fold):
				model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
				# model = CNN1d(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
				model.embedding.weight.requires_grad = not self.freeze_emb
				self.models.append(model)

		# init embeddings, init pad and unk, print model info
		for model_idx in range(len(self.models)):
			print("=== Model {} ===".format(model_idx))
			if self.use_emb:
				self.init_emb(model_idx)
			self.init_pad_unk(model_idx)
			self.print_model_config_stats(model_idx)
			print("=== --- ===")

	def config_SeqVecCharNet_model(self, model_type, hid_dim, dropout):
		print("Configuring SeqVecCharNet model.")
		# common init for all models
		self.model_type = model_type
		INPUT_DIM = len(self.SEQUENCE.vocab)
		EMBEDDING_DIM = self.emb_dim
		if self.is_binary_class:
			OUTPUT_DIM = 1
		else:
			OUTPUT_DIM = len(self.LABEL.vocab)
		HIDDEN_DIM = hid_dim
		DROPOUT = dropout
		PAD_IDX = self.SEQUENCE.vocab.stoi[self.SEQUENCE.pad_token]

		# for each fold create a model
		if self.k_fold == -1 or self.k_fold == 0:
			model = SeqVecCharNet(INPUT_DIM,
			                      EMBEDDING_DIM,
			                      HIDDEN_DIM,
			                      OUTPUT_DIM,
			                      DROPOUT,
			                      PAD_IDX)
			model.embedding.weight.requires_grad = not self.freeze_emb
			self.models.append(model)
		else:  # k-fold
			for model_idx in range(self.k_fold):
				model = SeqVecCharNet(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, PAD_IDX)
				model.embedding.weight.requires_grad = not self.freeze_emb
				self.models.append(model)

		# init embeddings, init pad and unk, print model info
		for model_idx in range(len(self.models)):
			print("=== Model {} ===".format(model_idx))
			if self.use_emb:
				self.init_emb(model_idx)
			self.init_pad_unk(model_idx)
			self.print_model_config_stats(model_idx)
			print("=== --- ===")

	def config_SeqVecNet_model(self, model_type, hid_dim, dropout):
		print("Configuring SeqVecNet model.")
		# common init for all models
		self.model_type = model_type
		if self.emb_name == "protvec":
			INPUT_DIM = len(self.SEQUENCE.vocab)
			PAD_IDX = self.SEQUENCE.vocab.stoi[self.SEQUENCE.pad_token]
		elif self.emb_name == "dom2vec":
			INPUT_DIM = len(self.INTERPRO_DOMAINS.vocab)
			PAD_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.pad_token]
		else:
			print("Not implemented for {} embeddings".format(self.emb_name))
		EMBEDDING_DIM = self.emb_dim
		if self.is_binary_class:
			OUTPUT_DIM = 1
		else:
			OUTPUT_DIM = len(self.LABEL.vocab)
		HIDDEN_DIM = hid_dim
		DROPOUT = dropout

		# for each fold create a model
		if self.k_fold == -1 or self.k_fold == 0:
			model = SeqVecNet(INPUT_DIM,
			                  EMBEDDING_DIM,
			                  HIDDEN_DIM,
			                  OUTPUT_DIM,
			                  DROPOUT,
			                  PAD_IDX)
			model.embedding.weight.requires_grad = not self.freeze_emb
			self.models.append(model)
		else:  # k-fold
			for model_idx in range(self.k_fold):
				model = SeqVecNet(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT, PAD_IDX)
				model.embedding.weight.requires_grad = not self.freeze_emb
				self.models.append(model)

		# init embeddings, init pad and unk, print model info
		for model_idx in range(len(self.models)):
			print("=== Model {} ===".format(model_idx))
			if self.use_emb:
				self.init_emb(model_idx)
			self.init_pad_unk(model_idx)
			self.print_model_config_stats(model_idx)
			print("=== --- ===")

	def config_FastText_model(self, model_type, dropout):
		print("Configuring FastText model.")
		assert self.emb_name == "dom2vec", "AssertionError: FastText is implemented only for domains"
		# common init for all models
		self.model_type = model_type
		INPUT_DIM = len(self.INTERPRO_DOMAINS.vocab)
		EMBEDDING_DIM = self.emb_dim
		if self.is_binary_class:
			OUTPUT_DIM = 1
		else:
			OUTPUT_DIM = len(self.LABEL.vocab)
		DROPOUT = dropout
		PAD_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.pad_token]

		# for each fold create a model
		if self.k_fold == -1 or self.k_fold == 0:
			model = FastText(INPUT_DIM,
			                 EMBEDDING_DIM,
			                 OUTPUT_DIM,
			                 DROPOUT,
			                 PAD_IDX)
			model.embedding.weight.requires_grad = not self.freeze_emb
			self.models.append(model)
		else:  # k-fold
			for model_idx in range(self.k_fold):
				model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT, PAD_IDX)
				model.embedding.weight.requires_grad = not self.freeze_emb
				self.models.append(model)

		# init embeddings, init pad and unk, print model info
		for model_idx in range(len(self.models)):
			print("=== Model {} ===".format(model_idx))
			if self.use_emb:
				self.init_emb(model_idx)
			self.init_pad_unk(model_idx)
			self.print_model_config_stats(model_idx)
			print("=== --- ===")

	def print_model_config_stats(self, model_idx):
		print("=== Model Stats ===")
		print(self.models[model_idx])
		print("Number of trainable parameters= {}.".format(self.count_parameters(model_idx)))
		if self.emb_name == "protvec" or self.emb_name == "dom2vec":
			print("Embeddings:")
			print(self.models[model_idx].embedding.weight.data)
		print("=== ===")

	def get_SeqVec_embs(self):
		# for each protein get the residue embeddings through seqvec
		# then concatenate the embeddings tensors to initialize the embeddings layer
		# credits: https://stackoverflow.com/questions/55050717/converting-list-of-tensors-to-tensors-pytorch

		print("=== Load SeqVec ===")
		model_dir = Path(self.emb_file)
		weights = model_dir / 'weights.hdf5'
		options = model_dir / 'options.json'
		# seqvec = ElmoEmbedder(options,weights, cuda_device=-1) #local pc, no cuda
		seqvec = ElmoEmbedder(options, weights, cuda_device=0)  # server with cuda
		# get SeqVec residue embeddings
		seqvec_embeddings = []
		count_progress = 0
		for seq in self.SEQUENCE.vocab.itos:
			if seq == "<unk>" or seq == "<pad>":
				seqvec_embeddings.append(torch.zeros(self.emb_dim))
			else:
				elmo_layers = seqvec.embed_sentence(list(seq))

				# elmo_layers = [3,L,1024]
				residue_emb = torch.tensor(elmo_layers).sum(dim=0)

				# residue_emb = [L,1024]
				prot_emb = residue_emb.mean(dim=0)

				# prot_emb = [1024]
				seqvec_embeddings.append(prot_emb)
			count_progress += 1
			if (count_progress + 1) % 250 == 0:
				print("Retrieved SeqVec embeddings for the first {} proteins".format(count_progress + 1))

		print("=== ===")
		return torch.stack(seqvec_embeddings)

	def init_emb(self, model_idx):
		if self.emb_name == "protvec":
			print("Init pre-trained tri-mer embeddings")
			embeddings = self.SEQUENCE.vocab.vectors
		elif self.emb_name == "seqvec":
			print("Init residue embeddings with SeqVec pre-trained ones")
			if model_idx == 0:
				self.seqvec_emb = self.get_SeqVec_embs()
			embeddings = self.seqvec_emb
		elif self.emb_name == "dom2vec":
			print("Init pre-trained domain embeddigns")
			embeddings = self.INTERPRO_DOMAINS.vocab.vectors

		print("Embeddings shape: {}".format(embeddings.shape))
		self.models[model_idx].embedding.weight.data.copy_(embeddings)

	def init_pad_unk(self, model_idx):
		if self.emb_name == "protvec" or self.emb_name == "seqvec":
			print("Set up sequence embeddings of <unk> and <pad> to 0.")
			UNK_IDX = self.SEQUENCE.vocab.stoi[self.SEQUENCE.unk_token]
			PAD_IDX = self.SEQUENCE.vocab.stoi[self.SEQUENCE.pad_token]
		elif self.emb_name == "dom2vec":
			print("Set up domain embeddings of <unk> and <pad> to 0.")
			UNK_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.unk_token]
			PAD_IDX = self.INTERPRO_DOMAINS.vocab.stoi[self.INTERPRO_DOMAINS.pad_token]

		# if self.emb_name == "protvec" or self.emb_name == "dom2vec":
		self.models[model_idx].embedding.weight.data[UNK_IDX] = torch.zeros(self.emb_dim)
		self.models[model_idx].embedding.weight.data[PAD_IDX] = torch.zeros(self.emb_dim)

	def config_criterion_optimizer(self, learning_rate, decay):
		###
		# Optimizer
		# Credits: https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_optim.html
		# Setting parameters to optimize per layer:
		# https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch
		###
		self.learning_rate = learning_rate
		for model_idx in range(len(self.models)):
			print("Setting up optimizer.")
			if self.model_type == "FastText":
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].fc.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))
			elif self.model_type == "CNN":
				"""
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].convs.parameters()},
				                                   {"params": self.models[model_idx].fc.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))
				"""
				#CNN2d
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].convs.parameters()},
				                                   {"params": self.models[model_idx].fc.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].fc2.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].fc3.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))

			elif self.model_type == "LSTM":
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].rnn.parameters()},
				                                   {"params": self.models[model_idx].fc.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))
			elif self.model_type == "SeqVecNet":
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].fc1.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].fc2.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))
			elif self.model_type == "SeqVecCharNet":
				self.optimizers.append(optim.Adam([{"params": self.models[model_idx].embedding.parameters()},
				                                   {"params": self.models[model_idx].fc1.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].fc2.parameters(),
				                                    "weight_decay": decay},
				                                   {"params": self.models[model_idx].bn.parameters()},
				                                   {"params": self.models[model_idx].dropout.parameters()}
				                                   ], lr=learning_rate))
			if self.is_binary_class:
				print("Setting up criterion for binary class.")
				self.criteria.append(nn.BCEWithLogitsLoss(pos_weight=self.label_weights[0]/self.label_weights[1]))
				# self.criteria.append(nn.BCEWithLogitsLoss(pos_weight=self.label_weights[1]))
			else:  # multi-class
				print("Setting up criterion for multi-class.")
				self.criteria.append(nn.CrossEntropyLoss(weight=self.label_weights))
			self.optimizers[model_idx].zero_grad()
			self.models[model_idx] = self.models[model_idx].to(self.device)
			self.criteria[model_idx] = self.criteria[model_idx].to(self.device)

		assert len(self.models) == len(self.optimizers) == len(
			self.criteria), "AssertionError: Each model has one optimizer and one criterion."

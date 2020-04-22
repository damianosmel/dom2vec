from nltk.corpus import wordnet as wn
import nltk
from utils import create_dir
import os
from gensim.models import KeyedVectors

"""
Class to create data set to map a word to its domain
"""


class WordNetDomain:

	def __init__(self, domains_file_path, out_path, out_name, emb_path, is_emb_bin, is_wordNet_installed):
		self.domains_file_path = domains_file_path
		self.out_path = out_path
		create_dir(self.out_path)
		self.out_name = out_name
		self.emb_path = emb_path
		self.is_emb_bin = is_emb_bin
		self.words2topics = {}  # dictionary to save {..,"code": computer_science,..}
		if is_wordNet_installed == False:
			print("Installing wordnet, please wait..")
			nltk.download('wordnet')

	# nltk.download('dict')

	def load_word_embs(self):
		print("Loading word embeddings from {}".format(self.emb_path))
		return KeyedVectors.load_word2vec_format(self.emb_path, binary=self.is_emb_bin)  # C bin format

	def create_dataset(self):
		self.parse_wordnet_topics()
		self.save_word_topics()

	@staticmethod
	def get_wn_pos(pos):
		if pos == "NOUN":
			return wn.NOUN
		elif pos == "VERB":
			return wn.VERB
		elif pos == "ADJ":
			return wn.ADJ
		elif pos == "ADV":
			return wn.ADV
		elif pos == "ADJ_SAT":
			return wn.ADJ_SAT
		else:
			return None

	def parse_wordnet_topics(self):
		# credits: http://www.nltk.org/howto/wordnet.html, issue 541
		print("Parsing words topic domains for WordNet version {}".format(wn.get_version()))
		words2vecs = self.load_word_embs()
		count = 0
		voc_len = len(words2vecs.wv.vocab)
		for token in words2vecs.wv.vocab:
			if (count + 1) % 1000 == 0:
				print("Processed {} tokens".format(count + 1))
			assert len(token.split("_")) >= 1, "AssertError: The token {} is shorter than word_POS format".format(token)
			word = token.split("_")[0]
			pos = token.split("_")[1]
			# print(pos)
			# print(WordNetDomain.get_wn_pos(pos))
			if token not in self.words2topics:
				unique_topics = set()
				self.words2topics[token] = unique_topics
			for synset in wn.synsets(word, WordNetDomain.get_wn_pos(pos)):
				if len(synset.topic_domains()) > 0:
					self.words2topics[token].add(synset.topic_domains()[0].name().split(".")[0])
			count += 1

	def tab_word_topics(self, word):
		return word + "\t" + " ".join(topic for topic in self.words2topics[word])

	def save_word_topics(self):
		print("Saving word topics tabular file {}".format(os.path.join(self.out_path, self.out_name)))
		with open(os.path.join(self.out_path, self.out_name), 'w') as out_file:
			out_file.write("word\ttopics\n")
			for word, topics in self.words2topics.items():
				if len(topics) > 0:
					out_file.write(self.tab_word_topics(word) + "\n")

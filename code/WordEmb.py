import os
from gensim.models import Word2Vec
from timeit import default_timer as timer
from utils import sec2hour_min_sec
from Corpus import Corpus
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

class WordEmb:
	"""
	Class to train word2vec for protein domains using gensim
	Credits: https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
	"""

	def __init__(self,data_path,corpus_file):
		self.data_path = data_path
		self.corpus_file = corpus_file
		self.Corpus = Corpus(self.data_path,self.corpus_file)
		self.w2v_model = "none"
		self.w2v_file_out = ""

	def create_file_out_name(self,window,use_skipgram,use_hier_soft, vec_dim):
		self.w2v_file_out = "dom2vec_" + "_".join(["w"+str(window),"sg"+str(use_skipgram),"hierSoft"+str(use_hier_soft),"dim"+str(vec_dim)])

	def set_up(self,window,use_skipgram,use_hier_soft,vec_dim,cores):
		print("=====")
		print("1) Set up word2vec model.")
		#self.w2v_model = Word2Vec(min_count=5,window=window,sg=use_cbow,hs=use_hier_soft,size=vec_dim,sample=6e-5,alpha=0.03,min_alpha=0.0007,negative=20,workers=cores)
		self.w2v_model = Word2Vec(window=window,
		                          sg=use_skipgram,
		                          hs=use_hier_soft,
		                          size=vec_dim,
		                          min_count=1,
		                          workers=cores,
		                          compute_loss=True)
		self.create_file_out_name(window,use_skipgram,use_hier_soft,vec_dim)

	def build_voc(self):
		"""
		Build the vocabulary
		"""
		print("\n=====")
		print("2) Build the voc table.")

		time_start=timer()
		self.w2v_model.build_vocab(self.Corpus, progress_per=10000)
		time_end=timer()
		print("Elapsed CPU time for building vocabulary: {}.".format(sec2hour_min_sec(time_end - time_start)))

	def train_stepwise(self,max_epochs,epochs_step):
		"""
		Train model and save it iteratively.

		:param max_epochs: maximum number of epochs to train the model
		:param epochs_step: train in a step-wise manner
		:return:
		"""
		for current_epochs in range(0,max_epochs,epochs_step):
			print("Training for {} of {}.".format(current_epochs+epochs_step,max_epochs))
			self.train(epochs_step,finalize_model=False)
			current_w2v_file_out = self.w2v_file_out + "_e" + str(current_epochs + epochs_step) + ".txt"#".bin"
			self.save(current_w2v_file_out,save_bin_format=False)

	def train(self,epochs,finalize_model):
		"""
		Train model
		"""
		print("\n=====")
		print("3) Train word2vec model.")
		time_start = timer()
		self.w2v_model.train(self.Corpus, total_examples=self.w2v_model.corpus_count, epochs=epochs,report_delay=1)
		time_end = timer()
		print("Elapsed CPU time for model training: {}.".format(sec2hour_min_sec(time_end - time_start)))

		if(finalize_model):
			print("Finalizing model.")
			self.w2v_model.init_sims(replace=True)

	def save(self,file_out,save_bin_format):
		"""
		Save trained embeddings into file.

		:param file_out: name of output file
		:return: None
		"""
		self.w2v_model.wv.save_word2vec_format(os.path.join(self.data_path, file_out), binary=save_bin_format)
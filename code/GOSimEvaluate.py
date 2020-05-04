from gensim.models import KeyedVectors
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import join
import ntpath
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from sklearn.metrics.pairwise import pairwise_kernels
import random


###                               ###
### Deprecated Class              ###
### please see GOEvaluate instead ###
###                               ###

class GOSimEvaluate:
	"""
	Class to apply Spearman correlation on the GO similarity of domains
	and the distance of domains calculated in the embeddings space
	1) Load embeddings space
	2) Load GO similarity
	3) Scatter plot of embeddings spaces vs GO similarity
	4) Calculate Spearman correlation and check significance
	"""

	def __init__(self, out_path, go_sim_path, emb_path, emb_method_name, is_model_binary):
		print("GOSimEvaluate")
		self.out_path = out_path
		self.go_sim_path = go_sim_path
		self.emb_path = emb_path
		self.emb_method_name = emb_method_name
		self.is_model_binary = is_model_binary
		self.emb_model = None
		self.domains_sim = None
		self.SEED = 1234

	def get_model_name(self):
		return self.emb_method_name + "_" + ntpath.basename(self.emb_path).split(".")[0]

	def normalize_cosine(self, cos_sim):
		"""
		Normalize cosince similarity value to [0,1] range
		norm_cos_sim = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min)
		:param cos_sim: cosine similarity value
		:return: norm_cos_sim
		"""
		norm_cos_sim = (cos_sim + 1) / 2
		assert norm_cos_sim >= 0 and norm_cos_sim <= 1, "AssertionError normalized cosine should be 0 <= norm_cos <= 1."
		return norm_cos_sim

	def cos_sim2angular_distance(self, cos_sim):
		# Convert cosine similarity to angular distance
		# Based on https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
		return np.arccos(cos_sim) / np.pi

	def compute_angular_similarity(self, cos_sim):
		# Convert cosine similarity to angular similarity
		# Based on https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
		return 1.0 - (np.arccos(cos_sim) / np.pi)

	def normalize_values(self, similarities):
		max_sim = max(similarities)
		min_sim = min(similarities)
		return [(x - min_sim) / (max_sim - min_sim) for x in similarities]

	def load_emb_model(self):
		self.emb_model = KeyedVectors.load_word2vec_format(self.emb_path, binary=self.is_model_binary)

	def load_GO_similarity(self):
		return read_csv(self.go_sim_path, sep="\t", header=0)

	def compute_emb_sim(self, dom_vec_X, dom_vec_Y):
		# Convert euclidean distance to similarity
		# Credits: https://stats.stackexchange.com/questions/158279/how-i-can-convert-distance-euclidean-to-similarity-score
		distance = np.linalg.norm(dom_vec_X - dom_vec_Y)
		similarity = 1.0 / (distance + 1.0)
		return similarity

	def extract_similarity_common_pairs(self, go_sim_all):
		go_sim_values = []
		emb_sim_values = []
		num_go_sim = 0
		num_emb_sim = 0
		for index, domain_pair in go_sim_all.iterrows():
			if self.emb_model.__contains__(domain_pair["interpro_id1"]) and self.emb_model.__contains__(
					domain_pair["interpro_id2"]):
				num_go_sim = num_go_sim + 1
				num_emb_sim = num_emb_sim + 1
				# go_sim_values.append(round(domain_pair["domain_similarity"], 2))
				go_sim_values.append(domain_pair["domain_similarity"])
				# normalize cosine to [0,1] using (x-min_value)/(max_value-min_value) = (x+1)/2
				# Euclidean distance
				# emb_sim = self.compute_emb_sim(self.emb_model.word_vec(domain_pair["interpro_id1"]), self.emb_model.word_vec(domain_pair["interpro_id2"]))

				# cos_sim = self.emb_model.similarity(domain_pair["interpro_id1"], domain_pair["interpro_id2"])
				emb_sim = pairwise_kernels(self.emb_model.word_vec(domain_pair["interpro_id1"]).reshape(1, -1),
				                           self.emb_model.word_vec(domain_pair["interpro_id2"]).reshape(1, -1),
				                           metric="sigmoid")
				emb_sim_values.append(emb_sim[0][0])

		assert num_go_sim == len(
			go_sim_values), "AssertionError: number of GO similarity values should be equal to total number of common GO similarity values."
		assert num_emb_sim == len(
			emb_sim_values), "AssertionError: number of angular distances should be equal to total number of found angular distances."
		print("Common pairs: #Go scores= {}, emb scores= {}".format(num_go_sim, num_emb_sim))

		# return go_sim_values, emb_sim_values
		return go_sim_values, emb_sim_values

	def sample_values(self, go_sim, emb_sim, sample_num):
		random.seed(self.SEED)
		go_emb_sim = list(zip(go_sim, emb_sim))
		sample_go_sim = []
		sample_emb_sim = []
		for go, emb in random.sample(go_emb_sim, sample_num):
			sample_go_sim.append(go)
			sample_emb_sim.append(emb)
		return sample_go_sim, sample_emb_sim

	def scatter_plot(self, go_sim_values, emb_sim_values):
		sample_go_sim, sample_emb_sim = self.sample_values(go_sim_values, emb_sim_values, sample_num=10000)
		plt.clf()  # clear figure
		fig = plt.figure()
		sns.set_context("paper")
		# Set the font to be serif, rather than sans
		sns.set(font='serif')

		# Make the background white, and specify the
		# specific font family
		sns.set_style("white", {
			"font.family": "serif",
			"font.serif": ["Times", "Palatino", "serif"]
		})

		sns.set(font_scale=1.5)

		g = sns.jointplot(x=go_sim_values, y=emb_sim_values, kind="reg",
		                  joint_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.5, 'edgecolor': 'k'}})
		g.set_axis_labels("GO similarity", "Embedding sigmoid similarity", fontsize=18)
		fig = g.fig
		fig_name = "go_sim_vs_" + self.get_model_name() + ".png"
		fig.savefig(join(self.out_path, fig_name), bbox_inches='tight', dpi=600)
		plt.close("all")

	def calculate_spearman(self, go_sim_values, emb_sim_values):
		# Credits: https://machinelearningmastery.com/how-to-calculate-nonparametric-rank-correlation-in-python/
		return spearmanr(go_sim_values, emb_sim_values)

	def print_domains_labels(self):
		print("=== domains_yX ===")
		print(self.domains_sim.head())
		print("=== ===")

	def get_emb_num_dim(self):
		return self.emb_model[self.emb_model.index2entity[0]].shape[0]

	def domains2vectors(self, domains, id_col):
		if domains[id_col] in self.emb_model.wv.vocab:  # check if interpro id exists in word model
			return self.emb_model[domains[id_col]]
		else:
			return [0] * self.get_emb_num_dim()

	def create_Xy(self, go_sim_df):
		# get vectors for the two domains of each pair
		# remove domains ids
		self.domains_sim = go_sim_df
		self.domains_sim["emb_id1"] = self.domains_sim.apply(self.domains2vectors, id_col="interpro_id1", axis=1)
		self.domains_sim["emb_id2"] = self.domains_sim.apply(self.domains2vectors, id_col="interpro_id2", axis=1)
		# convert vector to one column per dimension
		self.domains_sim[["x1_" + str(i) for i in range(self.get_emb_num_dim())]] = pd.DataFrame(
			self.domains_sim.emb_id1.tolist(), index=self.domains_sim.index)
		self.domains_sim[["x2_" + str(i) for i in range(self.get_emb_num_dim())]] = pd.DataFrame(
			self.domains_sim.emb_id2.tolist(), index=self.domains_sim.index)
		self.domains_sim.drop(columns=["interpro_id1", "interpro_id2", "emb_id1", "emb_id2"], inplace=True)

	# self.print_domains_labels()

	def get_x(self, x_name):
		if x_name == "concat_vec":
			x1_cols = ["x1_" + str(i) for i in range(self.get_emb_num_dim())]
			x2_cols = ["x2_" + str(i) for i in range(self.get_emb_num_dim())]
			x_cols = x1_cols + x2_cols
		else:
			x_cols = [x_name + "_" + str(i) for i in range(self.get_emb_num_dim())]

		return self.domains_sim[[col_name for col_name in x_cols]]

	def transform2normal(self, y):
		y_ar = np.array(y)
		y_ar = np.exp((y_ar + abs(y_ar.min())) / 200)
		return np.log1p(y_ar)

	def mean_absolute_percentage_error(self, y_true, y_pred):
		# Credits: https://stats.stackexchange.com/questions/58391/mean-absolute-percentage-error-mape-in-scikit-learn
		y_true, y_pred = np.array(y_true), np.array(y_pred)
		return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

	def run_linear_reg(self, run_linear):
		# get X
		x1 = self.get_x("x1").values
		x2 = self.get_x("x2").values
		x = x1 - x2
		# x = self.get_x("concat_vec").values
		x_min, x_max = np.min(x, 0), np.max(x, 0)
		X = (x - x_min) / (x_max - x_min)

		y = self.domains_sim["domain_similarity"].values
		# convert domains similarity to normal distribution
		# credits: https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py
		y_trans = quantile_transform(y.reshape(-1, 1), n_quantiles=100,
		                             output_distribution='normal',
		                             copy=True).squeeze()

		kf = KFold(n_splits=5, random_state=self.SEED)
		fold_idx = 0
		mse_folds, r2_score_folds, mae_folds, mape_folds = [], [], [], []

		for train_index, test_index in iter(kf.split(X, y_trans)):
			fold_idx += 1
			print("=== Fold {} ===".format(fold_idx))
			X_train = X[train_index]
			y_train = y_trans[train_index]
			X_test = X[test_index]
			y_test = y_trans[test_index]
			# print(X_test.shape)
			# print(y_test.shape)
			if run_linear:
				# Create linear regression object
				regr = linear_model.LinearRegression()
			else:
				# Section 1.4.2: https://scikit-learn.org/stable/modules/svm.html
				regr = svm.SVR(gamma="scale")
			# Train the model using the training sets
			regr.fit(X_train, y_train)
			# Make predictions using the testing set
			y_pred = regr.predict(X_test)
			# The mean squared error
			mse = mean_squared_error(y_test, y_pred)
			print("Mean squared error: {:.2f}".format(mse))
			# Explained variance score: 1 is perfect prediction
			r2 = r2_score(y_test, y_pred)
			print("Variance score: {:.2f}".format(r2))
			mae = median_absolute_error(y_test, y_pred)
			print("Median absolute error: {:.2f}".format(mae))
			mape = self.mean_absolute_percentage_error(y_test, y_pred)
			print("Mean absolute percentage error: {:.2f}".format(mape))
			mse_folds.append(mse)
			r2_score_folds.append(r2)
			mae_folds.append(mae)
			mape_folds.append(mape)

		print("---")
		print("Average MSE: {:.3f}".format(sum(mse_folds) / len(mse_folds)))
		print("Average variance score: {:.3f}".format(sum(r2_score_folds) / len(r2_score_folds)))
		print("Average MAE: {:.3f}".format(sum(mae_folds) / len(mae_folds)))
		print("Average MAPE: {:.3f}".format(sum(mape_folds) / len(mape_folds)))

	def evaluate_by_regression(self, run_linear):
		# load embeddings space
		self.load_emb_model()
		# load GO similarity
		go_sim_all = self.load_GO_similarity()

		# Regression
		self.create_Xy(go_sim_all)
		self.run_linear_reg(run_linear)

	def evaluate(self):
		# load embeddings space
		self.load_emb_model()
		# load GO similarity
		go_sim_all = self.load_GO_similarity()

		# get similarities for all pairs that exist in the embeddings space
		go_sim_values, emb_sim_values = self.extract_similarity_common_pairs(go_sim_all)

		# plot scatter plot
		self.scatter_plot(go_sim_values, emb_sim_values)

		# calculate Spearman
		coef, p = self.calculate_spearman(go_sim_values, emb_sim_values)
		print("Spearmans correlation coefficient: {:.3f}".format(coef))
		# interpret the significance
		alpha = 0.05
		if p > alpha:
			print("Samples are uncorrelated (fail to reject H0) p={:.3f}.".format(p))
		else:
			print("Samples are correlated (reject H0) p={:.3f}.".format(p))

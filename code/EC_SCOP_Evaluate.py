from os.path import join
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, OPTICS
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture
import sklearn.utils.validation
from sklearn import neighbors
from gensim.models import KeyedVectors


class EC_SCOP_Evaluate:
	"""
	Class to cluster domain embeddings and evaluate them based on EC or SCOPe labeling
	"""

	def __init__(self, data_path, id_ec_scop_file, use_ec, out_name):
		"""
		EC_SCOP_Evaluate class init

		Parameters
		----------
		data_path : str
			data full path
		id_ec_scop_file : str
			id ec scop file name
		use_ec : bool
			use ec label (True), use SCOP otherwise
		out_name :
			output file name

		Returns
		-------
		None
		"""
		self.data_path = data_path
		self.id_ec_scop_file = id_ec_scop_file
		self.use_ec = use_ec
		self.out_name = out_name
		self.num2name = None  # dictionary to hold labels
		self.domains_labels = None
		self.counter_multilabel = {"no_label": 0, "one_label": 0, "multi_label": 0}
		self.emb_model = None
		self.Xy = None
		self.random_state = 314

		"""
		### EC ###
		Convert EC number to seven principal enzyme classes
		based on https://www.ebi.ac.uk/intenz/browse.jsp
		"""
		self.ec_num2name = {1: "Oxidoreductases", 2: "Transferases", 3: "Hydrolases", 4: "Lyases", 5: "Isomerases",
		                    6: "Ligases", 7: "Translocases"}
		"""
		### SCOP ###
		Convert to seven 
		based on the naming: http://scop.berkeley.edu/statistics/ver=2.06
		and on the stable identifiers: http://scop.berkeley.edu/help/ver=2.06
		"""
		self.scop_letter2num = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}
		self.scop_num2name = {1: "All alpha", 2: "All beta", 3: "Alpha & beta (a|b)", 4: "Alpha & beta (a+b)",
		                      5: "Multi-domain", 6: "Membrane, cell surface", 7: "Small proteins"}

	def get_emb_num_dim(self):
		"""
		Get number of embedding dimensions

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		Returns
		-------
		int
		number of embedding dimensions
		"""
		return self.emb_model[self.emb_model.index2entity[0]].shape[0]

	def set_counter_label_zero(self):
		"""
		Set label counter to zero

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None
		"""
		self.counter_multilabel = {"no_label": 0, "one_label": 0, "multi_label": 0}

	def create_Xy(self):
		"""
		Create X and y for each domain embedding and its EC or SCOP label

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None
		"""
		# remove domains with unknown label
		# remove not used label
		if self.use_ec:
			self.domains_labels = self.domains_labels[self.domains_labels.ECs != "unknown"]
			self.domains_labels.drop(columns=["SCOPs"], inplace=True)
		else:
			self.domains_labels = self.domains_labels[self.domains_labels.SCOPs != "unknown"]
			self.domains_labels.drop(columns=["ECs"], inplace=True)
		self.domains_labels["vector"] = self.domains_labels.apply(self.domains2vectors, axis=1)

		# convert vector to one column per dimension
		self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = pd.DataFrame(
			self.domains_labels.vector.tolist(), index=self.domains_labels.index)
		self.domains_labels.drop(columns=["vector"], inplace=True)

	def domains2vectors(self, domains):
		"""
		Map domain id to domain embeddings

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		self : pandas.Series
			domain row

		Returns
		-------
		numpy.array
			domain embedding vector
		"""
		if domains["interpro_id"] in self.emb_model.wv.vocab:  # check if interpro id exists in word model
			return self.emb_model[domains["interpro_id"]]
		else:  # interpro with no embedding vector
			return [0] * self.get_emb_num_dim()

	def load_emb2domains(self, model_file, is_model_bin):
		"""
		Load embeddings from model file

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		model_file : str
			embedding model file path
		is_model_bin : bool
			binary format for model (True), integer format otherwise

		Returns
		-------
		None
		"""
		print("Load embeddings")
		self.emb_model = KeyedVectors.load_word2vec_format(model_file, binary=is_model_bin)

	def read_id_ec_scop(self):
		"""
		Read dataframe with columns id,ec,scop for all domains

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None
		"""
		self.domains_labels = pd.read_csv(join(self.data_path, self.id_ec_scop_file), sep="\t",
		                                  header=0)  # , dtype={"interpro_id":str,"ECs":str,"SCOPs":str}

	def calc_label_stats(self, domains):
		"""
		(Apply function) calculate label statistics in domain dataframe

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		domains : pandas.Series
			domains row

		Returns
		-------
		None
		"""
		if self.use_ec:
			col = str(domains["ECs"])
		else:
			col = str(domains["SCOPs"])

		if col == "unknown":
			self.counter_multilabel["no_label"] += 1
		elif len(set(col.split(","))) == 1:
			self.counter_multilabel["one_label"] += 1
		elif len(set(col.split(","))) > 1:
			self.counter_multilabel["multi_label"] += 1

	def get_scop_class(self, domains, remove_multilabeled):
		"""
		Get SCOPe class based on description of "stable identifiers" at http://scop.berkeley.edu/help/ver=2.06
		For example map a.39.1.1 to 1 (meaning a)

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		domains : pandas.Series
			domains dataframe row
		remove_multilabeled : bool
			remove domain row with multi-labeled SCOP (True) otherwise keep (False)

		Returns
		-------
		pandas.Series
			updated domains row with deleted label (if needed)
		"""
		label = str(domains["SCOPs"])
		uniq_SCOP = set()
		if label == "nan":
			# print("Nan scope")
			uniq_SCOP.add("unknown")
		else:
			multilabel = label.split(" ")
			for single_label in multilabel:
				first_letter = single_label.split(".")[0]
				if first_letter in self.scop_letter2num:
					uniq_SCOP.add(str(self.scop_letter2num[first_letter]))
				else:
					uniq_SCOP.add("unknown")
			assert len(uniq_SCOP) > 0, "AssertionError: unique SCOPe class should be at least one."

			if remove_multilabeled and len(uniq_SCOP) > 1:
				uniq_SCOP = set()
				uniq_SCOP.add("unknown")

		domains["SCOPs"] = ",".join(uniq_SCOP)
		return domains

	def get_principalEC(self, domains, remove_multilabeled):
		"""
		Convert specific EC code to the principal EC that belongs to
		For example map 3.1.3.48 to 3 (Protein-tyrosine-phosphatase -> Hydrolases)
		link: https://www.ebi.ac.uk/intenz/query?cmd=SearchEC&ec=3.1.3.48

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		domains : pandas.Series
			domains dataframe row
		remove_multilabeled: bool
			remove row with multi-labeled EC number (True) otherwise keep (False)

		Returns
		-------
		pandas.Series
			updated domains row with deleted label (if needed)
		"""
		label = str(domains["ECs"])  # convert to string
		uniq_EC = set()
		if label == "nan":  # isnan(label): #if no label
			uniq_EC.add("unknown")
		else:
			multilabel = label.split(" ")
			for single_label in multilabel:
				first_num = single_label.split(".")[0]
				# assert first_num in num2name, "{} does not belong to known EC numbers".format(first_num)
				uniq_EC.add(first_num)
			assert len(uniq_EC) > 0, "AssertionError: unique EC principal class should be at least one."

			if remove_multilabeled and len(uniq_EC) > 1:
				uniq_EC = set()
				uniq_EC.add("unknown")

		domains["ECs"] = ",".join(uniq_EC)
		return domains

	def parse_ec_labels(self, remove_multilabeled):
		"""
		Convert EC number to seven principal enzyme classes
		based on https://www.ebi.ac.uk/intenz/browse.jsp
		1. -. -.-  Oxidoreductases.
		2. -. -.-  Transferases.
		3. -. -.-  Hydrolases.
		4. -. -.-  Lyases.
		5. -. -.-  Isomerases.
		6. -. -.-  Ligases.
		7. -. -.-  Translocases.

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		remove_multilabeled: bool
			remove row with multi-labeled EC number (True) otherwise keep (False)

		Returns
		-------
		None
		"""
		self.domains_labels.apply(self.get_principalEC, remove_multilabeled=remove_multilabeled, axis=1)

	def get_label_names(self, remove_multilabeled):
		"""
		Get label names

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		remove_multilabeled: bool
			remove row with multi-labeled SCOP/EC number (True) otherwise keep (False)

		Returns
		-------
		"""
		print("Get label names")
		self.read_id_ec_scop()
		# print initial counts for multi-label
		self.set_counter_label_zero()
		self.domains_labels.apply(self.calc_label_stats, axis=1)
		print(self.counter_multilabel)

		# get requested column as label and optionally remove multi-label domains
		if self.use_ec:
			self.domains_labels.apply(self.get_principalEC, remove_multilabeled=remove_multilabeled, axis=1)
		else:
			self.domains_labels.apply(self.get_scop_class, remove_multilabeled=remove_multilabeled, axis=1)
		self.set_counter_label_zero()
		self.domains_labels.apply(self.calc_label_stats, axis=1)
		print(self.counter_multilabel)
		assert self.domains_labels.shape[0] == sum(
			self.counter_multilabel.values()), "AssertionError: the number of rows should be equal with the number of sum counts of (no EC, single EC, multi EC)"
		print("===")

	def pred_by_center(self, cluster_centers, labels):
		"""
		Predicy by the center element
		for each label
		   get its cluster center
		   then get the closest domain of that cluster center
		   get the true label of the cluster center

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		cluster_centers:
			coordinates of each cluster center
		labels: list of str
			assigned label of each point (cluster index)

		Returns
		-------
		:return: predicted label by the true class of center where the point is clustered to
		"""
		labels = labels.tolist()
		y_pred = [-1] * len(labels)
		print(len(labels))
		for i in range(len(labels)):
			cluster_center = cluster_centers[labels[i], :]
			word_closest_center = self.emb_model.similar_by_vector(cluster_center)[0][0]
			if word_closest_center in self.domains_labels["interpro_id"]:
				y_pred[i] = self.domains_labels.loc[self.domains_labels["interpro_id"] == word_closest_center]["ECs"]
		return np.array(y_pred)

	def cluster_Xy(self, are_dim_reduced, cluster_algo):
		"""
		Cluster domain vectors (X) and then get label based on clusters
		# Credits: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
		# Credits: https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		are_dim_reduced : bool
			X dimensions are reduced (True), otherwise are with no change (False)
		cluster_algo : str
			clustering algorithm name to be used

		Returns
		-------

		"""
		y_pred = None
		num_uniq_classes = 0
		if self.use_ec:
			num_uniq_classes = len(self.ec_num2name)
		else:
			num_uniq_classes = len(self.scop_num2name)

		if cluster_algo == "k-means":
			print("Running k-means")
			if are_dim_reduced:
				y_pred = KMeans(n_clusters=num_uniq_classes, random_state=self.random_state).fit_predict(self.X_low)
			else:
				y_pred = KMeans(n_clusters=num_uniq_classes, random_state=self.random_state).fit_predict(
					self.get_x().values)

		elif cluster_algo == "OPTICS":
			print("Running OPTICS")

			optics = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05, metric="cosine")
			optics.fit(self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]])
			y_pred = optics.labels_[optics.ordering_]
		elif cluster_algo == "GMM":
			print("Running GMM")

		return y_pred

	def purity_score(self, y_true, y_pred):
		"""
		Compute purity score as in https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
		# Credits: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
		# purity measure: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		y_true : numpy.array
			array of ground truth y labels
		y_pred : numpy.array
			array of predicted y labels

		Returns
		-------
		float
			purity score
		"""

		# compute contingency matrix (also called confusion matrix)
		contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
		print(contingency_matrix)
		# return purity
		return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

	def get_x(self):
		"""
		Get the X (domains vectors)

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		Returns
		-------
		pandas.DataFrame
			all domain vectors
		"""
		return self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]]

	def run_NN(self, are_dim_reduced):
		"""
		run k-NN algorithm

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		are_dim_reduced : bool
			X dimensions are reduced (True), otherwise are of no change (False)

		Returns
		-------
		None
		"""
		print("Run kNN")
		if are_dim_reduced:
			x_min, x_max = np.min(self.X_low, 0), np.max(self.X_low, 0)
			X = (self.X_low - x_min) / (x_max - x_min)
		else:
			x_min, x_max = np.min(self.get_x().values, 0), np.max(self.get_x().values, 0)
			X = (self.get_x().values - x_min) / (x_max - x_min)

		if self.use_ec:
			y = self.domains_labels["ECs"].values
		else:
			y = self.domains_labels["SCOPs"].values

		sklearn.utils.validation._assert_all_finite(X)
		skf = StratifiedKFold(n_splits=5, random_state=self.random_state)

		fold_idx = 0
		# train_acc_models = {2: [], 5: [], 10: [], 20: []}
		# test_acc_models = {2: [], 5: [], 10: [], 20: [], 40: []} #for k-NN we can only assess the test acc
		test_acc_models = {2: [], 5: [], 20: [], 40: []}

		for train_index, test_index in iter(skf.split(X, y)):
			fold_idx += 1
			# print("=== Fold {} ===".format(fold_idx))
			X_train = X[train_index]
			y_train = y[train_index] - 1
			X_test = X[test_index]
			y_test = y[test_index] - 1
			n_classes = len(np.unique(y_train))
			estimators = {nn_num: neighbors.KNeighborsClassifier(nn_num, weights='distance')
			              for nn_num in [2, 5, 20, 40]}
			n_estimators = len(estimators)
			for index, (nn_num, estimator) in enumerate(estimators.items()):
				estimator.fit(X_train, y_train)
				y_test_pred = estimator.predict(X_test)
				test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
				test_acc_models[nn_num].append(test_accuracy)

		for nn_num in [2, 5, 20, 40]:
			print("=== {}-NN ===".format(nn_num))
			print(
				"max test acc: {}, min test acc: {}".format(max(test_acc_models[nn_num]), min(test_acc_models[nn_num])))
			print("avg test acc: {}".format(sum(test_acc_models[nn_num]) / float(len(test_acc_models[nn_num]))))

	def run_GMM(self, are_dim_reduced):
		"""
		run GMM algorithm
		# Credits: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		are_dim_reduced : bool
			the X dimensions are reduced (True), otherwise are of no change (False)

		Returns
		-------
		None
		"""
		print("Run GMM")
		# Break up the dataset into non-overlapping training (75%) and testing
		# (25%) sets.
		if are_dim_reduced:
			x_min, x_max = np.min(self.X_low, 0), np.max(self.X_low, 0)
			X = (self.X_low - x_min) / (x_max - x_min)
		else:
			x_min, x_max = np.min(self.get_x().values, 0), np.max(self.get_x().values, 0)
			X = (self.get_x().values - x_min) / (x_max - x_min)
		if self.use_ec:
			y = self.domains_labels["ECs"].values
		else:
			y = self.domains_labels["SCOPs"].values
		sklearn.utils.validation._assert_all_finite(X)
		skf = StratifiedKFold(n_splits=5)

		fold_idx = 0

		train_acc_models = {"spherical": [], "diag": [], "tied": [], "full": []}
		test_acc_models = {"spherical": [], "diag": [], "tied": [], "full": []}

		for train_index, test_index in iter(skf.split(X, y)):
			fold_idx += 1
			print("=== Fold {} ===".format(fold_idx))
			X_train = X[train_index]
			y_train = y[train_index] - 1  # both EC and SCOP class numbers are 1-index so subtract 1
			X_test = X[test_index]
			y_test = y[test_index] - 1  # both EC and SCOP class numbers are 1-index so subtract 1
			n_classes = len(np.unique(y_train))
			# Try GMMs using different types of covariances.
			estimators = {cov_type: GaussianMixture(n_components=n_classes,
			                                        covariance_type=cov_type, max_iter=200,
			                                        random_state=self.random_state)
			              for cov_type in ['spherical', 'diag', 'tied', 'full']}
			n_estimators = len(estimators)
			for index, (name, estimator) in enumerate(estimators.items()):
				# Since we have class labels for the training data, we can
				# initialize the GMM parameters in a supervised manner.
				sklearn.utils.validation._assert_all_finite(
					np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)]))
				estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])
				# Train the other parameters using the EM algorithm.
				estimator.fit(X_train)
				y_train_pred = estimator.predict(X_train)
				train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
				train_acc_models[name].append(train_accuracy)
				print("GMM with covariance type {} has train accuracy: {:.3f}".format(name, train_accuracy))
				y_test_pred = estimator.predict(X_test)
				test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
				test_acc_models[name].append(test_accuracy)
				print("GMM with covariance type {} has test accuracy: {:.3f}".format(name, test_accuracy))
				print("---")

		for model_name in ['spherical', 'diag', 'tied', 'full']:
			print("=== {} ===".format(model_name))
			print("max train acc: {}, min test acc: {}".format(max(train_acc_models[model_name]),
			                                                   min(train_acc_models[model_name])))
			print("avg train acc: {}".format(
				sum(train_acc_models[model_name]) / float(len(train_acc_models[model_name]))))
			print("max test acc: {}, min test acc: {}".format(max(test_acc_models[model_name]),
			                                                  min(test_acc_models[model_name])))
			print("avg test acc: {}".format(sum(test_acc_models[model_name]) / float(len(test_acc_models[model_name]))))

	def run_pca(self, low_dim_size):
		"""
		Run PCA algorithm

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		low_dim_size : int
			resulted number of dimensions after PCA

		Returns
		-------
		None
		"""
		print("Run PCA")
		pca = PCA(n_components=low_dim_size)
		self.X_low = pca.fit_transform(self.get_x().values)
		print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))

	def run_isomap(self, n_neighbors, low_dim_size):
		"""
		Run isomap algorithm

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		n_neighbors : int
			number of neighbors using for the isomap run
		low_dim_size : int
			resulted number of dimensions after isomap

		Returns
		-------
		None
		"""
		print("Run isomap")
		isomap = Isomap(n_neighbors=n_neighbors, n_components=low_dim_size)
		self.X_low = isomap.fit_transform(self.get_x().values)
		print("Done. Reconstruction error: {:.3f}".format(isomap.reconstruction_error()))

	def run_LLE(self, n_neighbors, low_dim_size):
		"""
		Run LLE algorithm

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		n_neighbors : int
			number of neighbors using for the isomap run
		low_dim_size : int
			resulted number of dimensions after isomap

		Returns
		-------
		None
		"""
		print("Run Locally Linear Embeddings")
		lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=low_dim_size, method='modified')
		self.X_low = lle.fit_transform(self.get_x().values)
		print("Done. Reconstruction error: {:.3f}".format(lle.reconstruction_error_))

	def run_tsne(self, perplexity, low_dim_size):
		"""
		Run t-sne algorithm

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		perplexity : float
			selected perplexity used for the t-sne run
		low_dim_size : int
			resulted number of dimensions after isomap

		Returns
		-------
		None
		"""
		print("Run t-SNE")
		tsne = TSNE(perplexity=perplexity, n_components=low_dim_size, init='pca', n_iter=2500,
		            random_state=self.random_state)
		self.X_low = tsne.fit_transform(self.get_x().values)
		print("Done. Reconstruction error: {:.3f}".format(tsne.kl_divergence_))

	def get_colors(self):
		"""
		Get colors for plotting

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		cmap : matplotlib.pyplot.cmap
			color map
		norm : matplotlib.colors
			normalized colours
		bounds : numpy.array
			colors boundaries
		"""
		if self.use_ec:
			N = len(self.ec_num2name)
		else:
			N = len(self.scop_num2name)
		# define the colormap
		cmap = plt.cm.nipy_spectral  # plt.cm.jet
		# extract all colors from the .jet map
		cmaplist = [cmap(i) for i in range(cmap.N)]
		# create the new map
		cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

		# define the bins and normalize
		bounds = np.linspace(0, N, N + 1)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		return cmap, norm, bounds

	def plot_low(self, low_method, low_dim_size):
		"""
		Plot X after dimensionality reduction
		# Credits: https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		low_method : str
			which method to use for dimensionality reduction
		low_dim_size: int
			number of produced dimensional reduced space

		Returns
		-------
		None
		"""
		print("Plot low dimensional representation")
		x_min, x_max = np.min(self.X_low, 0), np.max(self.X_low, 0)
		X_low = (self.X_low - x_min) / (x_max - x_min)
		cmap, norm, bounds = self.get_colors()

		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		if self.use_ec:
			ax.scatter(X_low[:, 0], X_low[:, 1], c=[int(label) - 1 for label in self.domains_labels["ECs"].values],
			           cmap=cmap, norm=norm, edgecolor='k')
			# Credits: https://stackoverflow.com/questions/31303912/matplotlib-pyplot-scatterplot-legend-from-color-dictionary
			markers = [plt.Line2D([0, 0], [0, 0], color=cmap(norm(ec_num - 1)), marker='o', linestyle='') for ec_num, _
			           in self.ec_num2name.items()]
			plt.legend(markers, [ec_name for _, ec_name in self.ec_num2name.items()], numpoints=1, loc=9,
			           bbox_to_anchor=(0.5, -0.1), ncol=2)
		else:
			ax.scatter(X_low[:, 0], X_low[:, 1], c=[int(label) - 1 for label in self.domains_labels["SCOPs"].values],
			           cmap=cmap, norm=norm, edgecolor='k')
			markers = [plt.Line2D([0, 0], [0, 0], color=cmap(norm(scop_num - 1)), marker='o', linestyle='') for
			           scop_num, _ in self.scop_num2name.items()]
			plt.legend(markers, [scop_name for _, scop_name in self.scop_num2name.items()], numpoints=1, loc=9,
			           bbox_to_anchor=(0.5, -0.1), ncol=2)

		if self.use_ec:
			plt.title("Enzyme Commission class (E.C.)", fontsize=18)
			label_name = "EC"
		else:
			plt.title("SCOPe Structural class", fontsize=18)
			label_name = "SCOP"
		plot_name = low_method + "_" + label_name + ".png"
		fig.savefig(join(self.data_path, plot_name), bbox_inches='tight', dpi=600)

	def print_domains_labels(self):
		"""
		Print domain label info

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None
		"""
		print("=== domains_yX ===")
		print(self.domains_labels.head())

		print("Class distribution")
		if self.use_ec:
			print(self.domains_labels.ECs.value_counts())
		else:
			print(self.domains_labels.SCOPs.value_counts())
		print("=== ===")

	def convert_dom2Xy(self, model_file, is_model_bin):
		"""
		Convert domains to X and y

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		model_file : str
			model file name
		is_model_bin : bool
			model is in binary format (True), otherwise model is not in binary format (False)

		Returns
		-------
		None
		"""
		# preprocess steps
		print("Convert dom_EC_SCOP.tab -> (X,y)")
		# dom_EC_SCOP.tab -> (domain,y)
		remove_multilabeled = True
		self.get_label_names(remove_multilabeled)
		if self.use_ec:
			num_uniq_classes = len(self.ec_num2name)
		else:
			num_uniq_classes = len(self.scop_num2name)
		print("Num of unique classes: {}".format(num_uniq_classes))

		# load word embeddings
		self.load_emb2domains(model_file, is_model_bin)
		self.create_Xy()

		if self.use_ec:
			self.domains_labels.sort_values(by="ECs", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_id": str, "ECs": int})
		else:
			self.domains_labels.sort_values(by="SCOPs", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_id": str, "SCOPs": int})
		self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = self.domains_labels[
			["x_" + str(i) for i in range(self.get_emb_num_dim())]].astype(float)
		self.print_domains_labels()
		sklearn.utils.validation._assert_all_finite(self.get_x().values)

	def prepare_labels(self):
		"""
		Prepare labels

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None

		"""
		# Clean: a) get all not unknown labels, b) remove not used columns
		if self.use_ec:
			self.domains_labels = self.domains_labels[self.domains_labels.ECs != "unknown"]
			self.domains_labels.drop(columns=["SCOPs"], inplace=True)
		else:
			self.domains_labels = self.domains_labels[self.domains_labels.SCOPs != "unknown"]
			self.domains_labels.drop(columns=["ECs"], inplace=True)

	def map_dom2label(self):
		"""
		Map domains to EC/SCOP label

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis

		Returns
		-------
		None
		"""
		if self.use_ec:
			print("interpro2EC_SCOPe.tab -> interpro_id, EC class")
		else:
			print("interpro2EC_SCOPe.tab -> interpro_id, SCOPe class")
		remove_multilabeled = True
		self.get_label_names(remove_multilabeled)
		self.prepare_labels()
		self.print_domains_labels()

		if self.use_ec:
			print("--- interpro_id, EC ---")
			print(self.domains_labels.shape)
			print(list(self.ec_num2name.values()))
		else:
			print("--- interpro_id, SCOPe ---")
			print(self.domains_labels.shape)
			print(list(self.scop_num2name.values()))

		if self.use_ec:
			self.domains_labels.astype({"interpro_id": str, "ECs": int})
		else:
			self.domains_labels.astype({"interpro_id": str, "SCOPs": int})

	def calculate_precision_k(self, model_file, is_model_bin):
		"""
		Calculate precision @ k

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		model_file : str
			model file name
		is_model_bin : bool
			model is in binary format (True), otherwise (False)

		Returns
		-------
		NNs_num : list of int
			list of used NNs
		avg_precision_k : list of float
			list of average precision achieved per used NN
		"""
		# load emb model
		self.load_emb2domains(model_file, is_model_bin)

		NNs_num = [i * 10 for i in range(1, 11)]
		# NNs_num = [i * 10 for i in range(1, 3)]

		avg_precision_k = [0.0] * len(NNs_num)
		for i in range(len(NNs_num)):
			num_examined_interpro = 0
			for _, row in self.domains_labels.iterrows():
				# get NN for embedding space, for current interpro id
				if self.emb_model.__contains__(row["interpro_id"]):
					# get the neighbors of the interpro domain
					neighbors_emb = set(
						[nn[0] for nn in self.emb_model.most_similar(positive=row["interpro_id"], topn=NNs_num[i])])

					# calculate the precision for domains with known SCOPe or EC label
					is_any_neighbor_with_known_label = False
					retrieved = 0  # retrieved (true_positive + false positive)
					true_positive = 0  # true_positive
					for neighbor in neighbors_emb:
						subset_domains = self.domains_labels[self.domains_labels["interpro_id"] == neighbor]
						if subset_domains.empty == False:
							is_any_neighbor_with_known_label = True
							if self.use_ec:
								if self.subset_domains.ECs.iloc[0] == row["ECs"]:
									true_positive = true_positive + 1
							else:
								# print("true:{},predicted:{}".format(row["one_level_parents"], subset_domains.one_level_parents.iloc[0]))
								if subset_domains.SCOPs.iloc[0] == row["SCOPs"]:
									true_positive = true_positive + 1
							retrieved = retrieved + 1
					if is_any_neighbor_with_known_label:
						assert true_positive <= retrieved, "AssertError: for current domain the true positives are more than the retrieved ones."
						avg_precision_k[i] = avg_precision_k[i] + (true_positive / retrieved)
						num_examined_interpro = num_examined_interpro + 1
			# calculate the average precision for all domains at current k
			if num_examined_interpro != 0:
				avg_precision_k[i] = round(avg_precision_k[i] / num_examined_interpro, 5)
			print("Average precision@{} = {}".format(NNs_num[i], avg_precision_k[i]))
		return NNs_num, avg_precision_k

	def run_precision(self, model_file, is_model_bin):
		"""
		Run precision experiments

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		model_file : str
			model file name
		is_model_bin : bool
			model is in binary format (True), otherwise (False)

		Returns
		-------
		NNs_num : list of int
			list of used NNs
		avg_precision_k : list of float
			list of average precision achieved per used NN
		"""
		self.map_dom2label()
		return self.calculate_precision_k(model_file, is_model_bin)

	def run_classification(self, model_file, is_model_bin, dim_reduction_algo, low_dim_size, classifier_name):
		"""
		Run classification experiments

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		model_file : str
			model file name
		is_model_bin : bool
			model is in binary format (True), otherwise (False)
		dim_reduction_algo : str
			used dimensionality reduction algorithm name
		low_dim_size : int
			number of dimensions for dimensionality reduced space
		classifier_name : str
			used classifier name

		Returns
		-------
			classification results
		"""
		self.convert_dom2Xy(model_file, is_model_bin)
		are_dim_reduced = True
		if dim_reduction_algo == "":  # no dimensionality reduction
			self.X_low = None
			are_dim_reduced = False
		elif dim_reduction_algo == "lle":  # Locally Linear Embedding
			self.run_LLE(n_neighbors=30, low_dim_size=low_dim_size)
		elif dim_reduction_algo == "pca":  # PCA
			self.run_pca(low_dim_size=low_dim_size)
		elif dim_reduction_algo == "tsne":  # t-SNE
			self.run_tsne(perplexity=30, low_dim_size=low_dim_size)
		elif dim_reduction_algo == "isomap":  # isomap
			self.run_isomap(n_neighbors=30, low_dim_size=low_dim_size)

		if dim_reduction_algo != "":
			# if dimension reduction is used and size of low dimension is 2, then plot the 2-d space
			if low_dim_size == 2:
				self.plot_low(dim_reduction_algo, low_dim_size)
		# find classifier performance
		if classifier_name == "GMM":
			self.run_GMM(are_dim_reduced)
		elif classifier_name == "NN":
			self.run_NN(are_dim_reduced)

	def cluster_and_eval(self, are_dim_reduced, cluster_algo):
		"""
		Cluster data and evaluate purity of resulted clusters

		Parameters
		----------
		self : object
			EC_SCOP_Evaluate object setup for this analysis
		are_dim_reduced : bool
			dimensions were reduced before applying clustering (True), otherwise (False)
		cluster_algo : str
			cluster algorithm name

		Returns
		-------
		None
		"""
		# cluster data
		# cluster_algo = "k-means"#"OPTICS"#

		y_pred = self.cluster_Xy(are_dim_reduced, cluster_algo)
		if self.use_ec:
			y_true = self.domains_labels["ECs"]
		else:
			y_true = self.domains_labels["SCOPs"]

		y_pred_l = y_pred.tolist()
		y_true_l = y_true.tolist()

		for i in range(100):
			print("y_true: {}\ty_pred: {}".format(y_true_l[i], y_pred_l[i]))
		purity_score = self.purity_score(y_true, y_pred)
		print("{} has purity of {:.3f}".format(cluster_algo, purity_score))

import pandas as pd
from os.path import join
from gensim.models import KeyedVectors
import numpy as np
from sklearn import neighbors
import sklearn.utils.validation
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target


class GOEvaluate:
	"""
	Class to evaluate
	"""

	def __init__(self, data_path, id2label_file):
		"""
		GOEvaluate class init

		Parameters
		----------
		data_path : str
			data full path
		id2label_file : str
			interpro to GO id mapping file name

		Returns
		-------
		None
		"""
		self.data_path = data_path
		self.id2label_file = id2label_file
		self.emb_model = None
		self.domains_labels = None
		self.random_state = 1234
		self.unique_labels = set()
		self.counter_multilabel = {"no_label": 0, "one_label": 0, "multi_label": 0}

	def get_emb_num_dim(self):
		"""
		Get number of dimensions of embedding vector

		Parameters
		----------

		Returns
		-------
		int
			number of dimensions of embedding vector
		"""
		return self.emb_model[self.emb_model.index2entity[0]].shape[0]

	def load_emb2domains(self, model_file, is_model_bin):
		"""
		Load embedding domains from model file

		Parameters
		----------
		model_file : str
			model file name
		is_model_bin : bool
			model is in binary format (True), otherwise (False)

		Returns
		-------
		None
		"""
		print("Load embeddings")
		self.emb_model = KeyedVectors.load_word2vec_format(model_file, binary=is_model_bin)

	def read_id_go(self):
		"""
		Read csv file containing domain id and its mapping to GO label

		Parameters
		----------

		Returns
		-------
		None
		"""
		self.domains_labels = pd.read_csv(join(self.data_path, self.id2label_file), sep=",", header=0)

	def set_counter_label_zero(self):
		"""
		Set label-type counter to zero

		Parameters
		----------

		Returns
		-------
		"""
		self.counter_multilabel = {"no_label": 0, "one_label": 0, "multi_label": 0}

	def get_go_label(self, domains_go, remove_multilabel, use_shortest):
		"""
		Get GO molecular function label for given GO id, using two ways:
		1) short_parent: use the GO annotation extracted from the shortest parent having as child starting GO id
		2) one_level_parents: use the GO annotation extracted from all one level parents containing GO id as sub-child
		Parameters
		----------
		domains_go : pandas.DataFrame row
			dataframe row containing info for single domain
		remove_multilabel : bool
			remove domain/row that containing more than one GO id (True), otherwise keep multilabel rows (False)
		use_shortest : bool
			use the shortest parent "way" (True),
			use the one level parents "way" (False)

		Returns
		-------
		str
			domain GO id
		"""
		if use_shortest:
			label = str(domains_go["short_parent"])
		else:
			label = str(domains_go["one_level_parents"])
		labels = label.split(";")
		if remove_multilabel and len(labels) > 1:
			if use_shortest:
				domains_go["short_parent"] = "unknown"
			else:
				domains_go["one_level_parents"] = "unknown"

		# keep track of unique labels
		if remove_multilabel and len(labels) == 1:
			self.unique_labels.add(label)
		elif remove_multilabel == False:
			for label in labels:
				self.unique_labels.add(label)
		return domains_go

	def calc_label_stats(self, domains, use_shortest):
		"""
		Calculate label statistics

		Parameters
		----------
		domains : pandas.DataFrame row
			domain info row
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		if use_shortest:
			col = str(domains["short_parent"])
		else:
			col = str(domains["one_level_parents"])

		if col == "unknown":
			self.counter_multilabel["no_label"] += 1
		elif len(set(col.split(";"))) == 1:
			self.counter_multilabel["one_label"] += 1
		elif len(set(col.split(";"))) > 1:
			self.counter_multilabel["multi_label"] += 1

	def get_label_names(self, remove_multilabel, use_shortest):
		"""
		Get GO label statistics

		Parameters
		----------
		remove_multilabel : bool
			remove domains with multi-label GO annotation (True), otherwise (False)

		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		print("Get label names")
		# read interpro_go csv file and get statistics for multilabel domains
		self.read_id_go()
		# print the initial counts for multi-label
		self.set_counter_label_zero()
		self.domains_labels.apply(self.calc_label_stats, use_shortest=use_shortest, axis=1)
		print(self.counter_multilabel)

		# get requested column as label and optionally remove multi-label domains
		self.domains_labels.apply(self.get_go_label, remove_multilabel=remove_multilabel, use_shortest=use_shortest,
		                          axis=1)
		self.set_counter_label_zero()
		self.domains_labels.apply(self.calc_label_stats, use_shortest=use_shortest, axis=1)
		print(self.counter_multilabel)
		assert self.domains_labels.shape[0] == sum(
			self.counter_multilabel.values()), "AssertionError: the number of rows should be equal with the number of sum counts of (no EC, single EC, multi EC)"

	def domains2vectors(self, domains):
		"""
		Convert domain id to embedding vector

		Parameters
		----------
		domains : pandas.DataFrame
			domain row info

		Returns
		-------
		numpy.array
			embedding vector
		"""
		if domains["interpro_ids"] in self.emb_model.wv.vocab:  # check if interpro id exists in word model
			return self.emb_model[domains["interpro_ids"]]
		else:
			# print("intepro_id with no embedding vector")
			return [0] * self.get_emb_num_dim()

	def convert_label2int(self, domains, use_shortest):
		"""
		Convert GO label to int value

		Parameters
		----------
		domains : pandas.DataFrame row
			domain row
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		pandas.DataFrame
			domain row with GO label in integer format
		"""
		if "unknown" in self.unique_labels:
			self.unique_labels.remove("unknown")
		name2num = {label: num for num, label in enumerate(self.unique_labels)}
		if use_shortest:
			label = str(domains["short_parent"])
			if label in name2num:
				domains["short_parent"] = name2num[label]
			else:
				print("Unknown label: {}".format(label))
				domains["short_parent"] = "unknown"
		else:
			label = str(domains["one_level_parents"])
			if label in name2num:
				domains["one_level_parents"] = name2num[label]
			else:
				print("Unknown label: {}".format(label))
				domains["one_level_parents"] = "unknown"
		return domains

	def create_Xy(self, use_shortest):
		"""
		Convert domains->GO dataframe to vectors and labels

		Parameters
		----------
		use_shortest : bool
		use the annotation extracted using the short parent node information (True),
		use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		# Clean: a) get all not unknown labels, b) remove not used columns
		if use_shortest:
			self.domains_labels = self.domains_labels[self.domains_labels.short_parent != "unknown"]
			self.domains_labels.drop(columns=["GO_terms", "one_level_parents"], inplace=True)
		else:
			self.domains_labels = self.domains_labels[self.domains_labels.one_level_parents != "unknown"]
			self.domains_labels.drop(columns=["GO_terms", "short_parent"], inplace=True)

		# Convert label to integer
		self.domains_labels.apply(self.convert_label2int, use_shortest=use_shortest, axis=1)

		# add the embedding vectors
		self.domains_labels["vector"] = self.domains_labels.apply(self.domains2vectors, axis=1)
		self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = pd.DataFrame(
			self.domains_labels.vector.tolist(), index=self.domains_labels.index)
		self.domains_labels.drop(columns=["vector"], inplace=True)

	def print_domains_labels(self, use_shortest):
		"""
		Print unique GO labels distribution

		Parameters
		----------
		use_shortest : bool
		use the annotation extracted using the short parent node information (True),
		use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		print("---")
		print(self.domains_labels.head())

		print("Class distribution")
		if use_shortest:
			print(self.domains_labels.short_parent.value_counts())
		else:
			print(self.domains_labels.one_level_parents.value_counts())
		print("---")

	def prepare_GO_labels(self, use_shortest):
		"""
		Convert GO label to homogeneous labels for learning experiments

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		# Clean: a) get all not unknown labels, b) remove not used columns
		if use_shortest:
			self.domains_labels = self.domains_labels[self.domains_labels.short_parent != "unknown"]
			self.domains_labels.drop(columns=["GO_terms", "one_level_parents"], inplace=True)
		else:
			self.domains_labels = self.domains_labels[self.domains_labels.one_level_parents != "unknown"]
			self.domains_labels.drop(columns=["GO_terms", "short_parent"], inplace=True)

		# Convert label to integer
		self.domains_labels.apply(self.convert_label2int, use_shortest=use_shortest, axis=1)

	def map_dom2GO(self, use_shortest):
		"""
		Map domain id to GO

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)

		Returns
		-------
		None
		"""
		print("go_labels.csv -> interpro_id, GO class")
		remove_multilabeled = True
		self.get_label_names(remove_multilabeled, use_shortest)
		self.print_domains_labels(use_shortest)
		self.prepare_GO_labels(use_shortest)

		print("--- interpro_id, GO ---")
		print(self.domains_labels.shape)
		print(self.unique_labels)

		if use_shortest:
			# self.domains_labels.sort_values(by="", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_ids": str, "short_parent": int})
		else:
			# self.domains_labels.sort_values(by="SCOPs", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_ids": str, "one_level_parents": int})

	def convert_dom2Xy(self, use_shortest, model_file, is_model_bin):
		"""
		Convert domains->GO labels to embedding vector and labels

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)
		model_file : str
			domains embedding file name
		is_model_bin : bool
			the embedding model is in binary format (True), otherwise (False)

		Returns
		-------
		None
		"""
		print("Convert go_labels.csv -> (X,y)")
		remove_multilabeled = True
		self.get_label_names(remove_multilabeled, use_shortest)
		self.print_domains_labels(use_shortest)
		# load word embeddings
		self.load_emb2domains(model_file, is_model_bin)
		self.create_Xy(use_shortest)
		print("--- Xy ---")
		print(self.domains_labels.shape)
		print(self.unique_labels)

		if use_shortest:
			# self.domains_labels.sort_values(by="", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_ids": str, "short_parent": int})
		else:
			# self.domains_labels.sort_values(by="SCOPs", inplace=True)
			self.domains_labels = self.domains_labels.astype({"interpro_ids": str, "one_level_parents": int})
		self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = self.domains_labels[
			["x_" + str(i) for i in range(self.get_emb_num_dim())]].astype(float)

	def get_x(self):
		"""
		Get embedding vector values

		Parameters
		----------

		Returns
		-------
		numpy.array
			embedding vector
		"""
		return self.domains_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]]

	def run_NN(self, use_shortest):
		"""
		Run k-NN algorithm

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)
		:return:
		"""
		print("Run kNN")

		x_min, x_max = np.min(self.get_x().values, 0), np.max(self.get_x().values, 0)
		X = (self.get_x().values - x_min) / (x_max - x_min)

		if use_shortest:
			y = self.domains_labels["short_parent"].values
		else:
			y = self.domains_labels["one_level_parents"].values
		# print(type_of_target(y))
		sklearn.utils.validation._assert_all_finite(X)
		skf = StratifiedKFold(n_splits=5, random_state=self.random_state)

		fold_idx = 0
		test_acc_models = {2: [], 5: [], 20: [], 40: []}

		for train_index, test_index in iter(skf.split(X, y)):
			fold_idx += 1
			# print("=== Fold {} ===".format(fold_idx))
			X_train = X[train_index]
			y_train = y[train_index]
			X_test = X[test_index]
			y_test = y[test_index]
			estimators = {nn_num: neighbors.KNeighborsClassifier(nn_num, weights='distance')
			              for nn_num in [2, 5, 20, 40]}

			for index, (nn_num, estimator) in enumerate(estimators.items()):
				estimator.fit(X_train, y_train)
				y_test_pred = estimator.predict(X_test)
				test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
				test_acc_models[nn_num].append(test_accuracy)
		# print("k-NN with k={} test accuracy: {:.3f}".format(nn_num, test_accuracy))
		# print("---")

		for nn_num in [2, 5, 20, 40]:
			print("=== {}-NN ===".format(nn_num))
			print(
				"max test acc: {}, min test acc: {}".format(max(test_acc_models[nn_num]), min(test_acc_models[nn_num])))
			print("avg test acc: {}".format(sum(test_acc_models[nn_num]) / float(len(test_acc_models[nn_num]))))

	def calculate_precision_k(self, use_shortest, model_file, is_model_bin):
		"""
		Calculate precision at k

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)
		model_file : str
			model file name
		is_model_bin : bool
			model is binary format (True), otherwise (False)
		Returns
		-------
		list of int, list of float
			number of used k's
			average precision achieved for each k
		"""
		# load emb model
		self.load_emb2domains(model_file, is_model_bin)

		NNs_num = [i * 10 for i in range(1, 11)]
		avg_precision_k = [0.0] * len(NNs_num)
		for i in range(len(NNs_num)):
			num_examined_interpro = 0
			for _, row in self.domains_labels.iterrows():
				# get NN for embedding space, for current interpro id
				if self.emb_model.__contains__(row["interpro_ids"]):
					# get the neighbors of the interpro domain
					neighbors_emb = set(
						[nn[0] for nn in self.emb_model.most_similar(positive=row["interpro_ids"], topn=NNs_num[i])])

					# calculate the precision for domains with known GO
					is_any_neighbor_with_GO = False
					retrieved = 0  # retrieved (true_positive + false positive)
					true_positive = 0  # true_positive
					for neighbor in neighbors_emb:
						subset_domains = self.domains_labels[self.domains_labels["interpro_ids"] == neighbor]
						if subset_domains.empty == False:
							is_any_neighbor_with_GO = True

							if use_shortest:
								if self.subset_domains.short_parent.iloc[0] == row["short_parent"]:
									true_positive = true_positive + 1
							else:
								# print("true:{},predicted:{}".format(row["one_level_parents"], subset_domains.one_level_parents.iloc[0]))
								if subset_domains.one_level_parents.iloc[0] == row["one_level_parents"]:
									true_positive = true_positive + 1
							retrieved = retrieved + 1
					if is_any_neighbor_with_GO:
						assert true_positive <= retrieved, "AssertError: for current domain the true positives are more than the retrieved ones."
						avg_precision_k[i] = avg_precision_k[i] + (true_positive / retrieved)
						num_examined_interpro = num_examined_interpro + 1
			# calculate the average precision for all domains at current k
			if num_examined_interpro != 0:
				avg_precision_k[i] = round(avg_precision_k[i] / num_examined_interpro, 5)
			print("For average precision@{} = {}".format(NNs_num[i], avg_precision_k[i]))
		return NNs_num, avg_precision_k

	def run_precision(self, use_shortest, model_file, is_model_bin):
		"""
		Run precision@K experiment

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)
		model_file : str
			model file name
		is_model_bin : bool
			is model in binary format (True), otherwise (False)

		Returns
		-------
		list of int, list of float
			number of used k's
			average precision achieved for each k
		"""
		self.map_dom2GO(use_shortest)
		return self.calculate_precision_k(use_shortest, model_file, is_model_bin)

	def run_classification(self, use_shortest, model_file, is_model_bin):
		"""
		Run k-NN classification

		Parameters
		----------
		use_shortest : bool
			use the annotation extracted using the short parent node information (True),
			use the annotation extracted using the one level parents node information (False)
		model_file : str
			model file name
		is_model_bin : bool
			is model in binary format (True), otherwise (False)

		Returns
		-------
		None
		"""
		self.convert_dom2Xy(use_shortest, model_file, is_model_bin)
		self.run_NN(use_shortest)

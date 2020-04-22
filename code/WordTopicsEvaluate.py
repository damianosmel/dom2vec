import pandas as pd
import numpy as np
from os.path import join
from gensim.models import KeyedVectors
import sklearn.utils.validation as sk_validation
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

"""
Class to evaluate word embeddings using wordnet topics
"""


class WordTopicsEvaluate:

	def __init__(self, data_path, word_topics_file):
		self.data_path = data_path
		self.word_topics_file = word_topics_file
		self.words_labels = None
		self.topics2num = {}
		self.counter_multilabel = {"no_label": 0, "one_label": 0, "multi_label": 0}
		self.random_state = 314

	@staticmethod
	def get_basic_domain(specific_domain):
		# get the basic domain from WordNet domain file: wn-domains-3.2/WDH-to-DDC-map.pdf
		specific_domain = specific_domain.lower()
		if specific_domain in ["history", "archeology", "heraldry"]:
			return "history"
		elif specific_domain in ["lingustics", "grammar"]:
			return "linguistics"
		elif specific_domain in ["literature", "philology"]:
			return "literature"
		elif specific_domain in ["psychology", "psychoanalyis"]:
			return "psychology"
		elif specific_domain in ["art", "poetry", "fine_arts", "dancing", "singing", "film", "graphic_arts", "dance",
		                         "drawing", "painting", "music", "photography", "plastic_Arts", "theatre", "cinema",
		                         "jewellery", "numismatics", "sculpture", "philately", "ceramics"]:
			return "art"
		elif specific_domain in ["paranormal", "occultism", "astrology", "alchemy", "voodo"]:
			return "paranormal"
		elif specific_domain in ["religion", "church_service", "christianity", "new_testament", "old_testament",
		                         "roman_catholic", "judaism", "islam", "hinduism", "buddhism", "bible", "sikhism",
		                         "anglican_church", "christian_theology", "taoism", "genesis", "theology",
		                         "zoroastrianism", "roman_Catholic", "mythology", "roman_mythology", "greek_mythology",
		                         "norse_mythology", "classical_mythology"]:
			return "religion"
		elif specific_domain in ["play", "betting", "chess", "game", "poker", "card_game"]:
			return "play"
		elif specific_domain in ["sport", "riding", "water_sport", "rugby", "ice_hockey", "gymnastics",
		                         "mountain_climbing", "polo", "croquet", "ball_game", "badminton", "baseball",
		                         "basketball", "cricket", "football", "golf", "rugdy", "soccer", "table_tennis",
		                         "tennis", "volleyball", "cycling", "skating", "skiing", "hockey", "mountaineering",
		                         "rowing", "swimming", "sub", "diving", "racing", "athletics", "wrestling", "boxing",
		                         "fencing", "archery", "fishing", "hunting", "bowling"]:
			return "sport"
		elif specific_domain in ["agriculture", "animal_husbandry", "veterinary", "plant", "farming", "botany",
		                         "gardening"]:
			return "agriculture"
		elif specific_domain in ["food", "gastronomy"]:
			return "food"
		elif specific_domain in ["architecture", "town_planning", "buildings", "furniture", "classical_architecture"]:
			return "architecture"
		elif specific_domain in ["engineering", "mechanics", "astronautics", "electrotechnology", "hydraulics",
		                         "construction", "bridge"]:
			return "engineering"
		elif specific_domain in ["telecommunication", "post", "telegraphy", "telephony", "telephone", "television",
		                         "broadcast_medium"]:
			return "telecommunication"
		elif specific_domain in ["medicine", "medicine", "narcotic", "drug", "veterinary_medicine", "dentistry",
		                         "pharmacy", "pharmacology", "physchiatry", "radiology", "surgery", "otology"]:
			return "medicine"
		elif specific_domain in ["biology", "cytology", "histology", "neuroscience", "vertebrate", "ecology",
		                         "microbiology", "immunology", "organism", "anatomy", "virology", "biochemistry",
		                         "physiology", "genetics", "epidemiology", "microorganism", "bacteria"]:
			return "biology"
		elif specific_domain in ["animals", "entomology", "ornithology", "falconry"]:
			return "animals"
		elif specific_domain in ["earth", "mining", "lake", "mineralogy", "geology", "meteorology", "oceanography",
		                         "paleontology", "geography", "topography", "tectonics"]:
			return "earth"
		elif specific_domain in ["computer_science", "information_science", "computer_technology", "programming",
		                         "computer"]:
			return "computer_science"
		elif specific_domain in ["mathematics", "geometry", "statistics", "arithmetic", "numeration_system"]:
			return "mathematics"
		elif specific_domain in ["physics", "astronomy", "thermodynamics", "particle_physics", "physical_chemistry",
		                         "nuclear_physics", "acoustics", "atomic_physic", "electricity", "electronics", "gas",
		                         "optics", "microscopy"]:
			return "physics"
		elif specific_domain in ["anthropology", "ethnology", "folklore"]:
			return "anthropology"
		elif specific_domain in ["health", "body_care"]:
			return "health"
		elif specific_domain in ["pedagogy", "school", "university", "academia", "education", "philosophy", "ethics"]:
			return "pedagogy"
		elif specific_domain in ["transport", "train", "airplane", "aviation", "vehicles", "nautical", "railway",
		                         "water_travel", "car", "sailing_vessel", "boat", "aircraft", "ship"]:
			return "transport"
		elif specific_domain in ["economy", "commerce", "commercial_enterprise", "investing", "investment",
		                         "accounting", "corporate_finance", "stock_exchange", "business", "economics",
		                         "enterprise", "corporation", "book_keeping", "bookkeeping", "finance", "banking",
		                         "money", "exchange", "insurance", "tax"]:
			return "economy"
		elif specific_domain in ["politics", "diplomacy"]:
			return "politics"
		elif specific_domain in ["law", "civil_law", "contract_law", "criminal_law", "roman_law", "justice"]:
			return "law"
		elif specific_domain in ["persian", "west_indies", "ethiopia", "jamaica", "latin", "brazil", "italian"]:
			return "ethnicities"
		else:
			return specific_domain

	def read_word_topics(self):
		# Read dataframe with columns word, topic for all domains
		self.words_labels = pd.read_csv(join(self.data_path, self.word_topics_file), sep="\t", header=0)

	def load_emb2domains(self, model_file, is_model_bin):
		print("Load embeddings")
		self.emb_model = KeyedVectors.load_word2vec_format(model_file, binary=is_model_bin)

	def get_emb_num_dim(self):
		return self.emb_model[self.emb_model.index2entity[0]].shape[0]

	def get_x(self):
		return self.words_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]]

	def get_topic_id(self, words, remove_multilabel, use_basic_label):
		label = str(words["topics"])  # convert to string
		uniq_topics_id = set()
		uniq_topics = set()
		if label == "nan":
			uniq_topics_id.add("unknown")
			uniq_topics.add("unknown")
		else:  # get all labels and save them in a total dictionary
			multilabel = label.split(" ")
			for label in multilabel:
				if use_basic_label:
					basic_label = WordTopicsEvaluate.get_basic_domain(label)
				else:
					basic_label = label
				if basic_label not in self.topics2num:
					self.topics2num[basic_label] = len(self.topics2num)
				uniq_topics_id.add(str(self.topics2num[basic_label]))
				uniq_topics.add(basic_label)
			# remove multilabel instances
			if remove_multilabel and len(uniq_topics_id) > 1:
				uniq_topics_id = set(["unknown"])
				uniq_topics = set(["unknown"])

		words["topics"] = ",".join(uniq_topics)
		words["topics_id"] = ",".join(uniq_topics_id)
		return words

	def get_label_names(self, remove_multilabeled, use_basic_label):
		# read word topics file
		# then convert the labels from categorical to number
		self.read_word_topics()
		self.words_labels = self.words_labels.apply(self.get_topic_id, remove_multilabel=remove_multilabeled,
		                                            use_basic_label=use_basic_label, axis=1)

	def calc_label_stats(self, words):
		col = str(words["topics_id"])
		if col == "unknown":
			self.counter_multilabel["no_label"] += 1
		elif len(set(col.split(","))) == 1:
			self.counter_multilabel["one_label"] += 1
		elif len(set(col.split(","))) > 1:
			self.counter_multilabel["multi_label"] += 1

	def compute_label_stats(self):
		print("=== Labels stats ===")
		print(self.words_labels.shape)
		self.words_labels.apply(self.calc_label_stats, axis=1)
		print(self.counter_multilabel)
		assert self.words_labels.shape[0] == sum(
			self.counter_multilabel.values()), "AssertionError: the number of rows should be equal with the number of sum counts of (no EC, single EC, multi EC)"
		print("===")

	def words2vectors(self, words_topics, create_rand_vec):
		if create_rand_vec:
			return np.random.random(self.get_emb_num_dim())
		else:
			if words_topics["word"] in self.emb_model.wv.vocab:  # check if word exists in emb model
				return self.emb_model[words_topics["word"]]
			else:
				# print("intepro_id with no embedding vector")
				return [0] * self.get_emb_num_dim()

	def create_Xy(self, use_rand_vec):
		print("=== Create Xy ===")
		if use_rand_vec:
			print("X will be random vectors")
		# remove words with unknown label
		self.words_labels = self.words_labels[self.words_labels.topics_id != "unknown"]
		self.words_labels["vector"] = self.words_labels.apply(self.words2vectors, create_rand_vec=use_rand_vec, axis=1)

		# convert vector to one column per dimension
		self.words_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = pd.DataFrame(
			self.words_labels.vector.tolist(), index=self.words_labels.index)
		self.words_labels.drop(columns=["vector"], inplace=True)

	def print_words_labels(self):
		print("=== words_yX ===")
		print(self.words_labels.head())
		print("Topics counts saved at {}".format(join(self.data_path, "topics_counts.csv")))
		self.words_labels.topics.value_counts().to_csv(join(self.data_path, "topics_counts.csv"), sep=",", index=True,
		                                               header=True)
		print("=== ===")

	def convert_word2Xy(self, model_file, is_model_bin, use_basic_label, use_rand_vec):
		# preprocess steps
		print("Convert word_topics.tab -> (X,y)")
		if (use_basic_label):
			print("Get basic labels from given ones")
		remove_multilabeled = True

		self.get_label_names(remove_multilabeled, use_basic_label)

		# get the number of unique classes and get the count of them
		# load word embeddings
		self.load_emb2domains(model_file, is_model_bin)
		self.create_Xy(use_rand_vec)
		# print(self.words_labels.head())
		self.words_labels.sort_values(by="topics_id", inplace=True)
		self.words_labels = self.words_labels.astype({"word": str, "topics": str, "topics_id": int})

		self.words_labels[["x_" + str(i) for i in range(self.get_emb_num_dim())]] = self.words_labels[
			["x_" + str(i) for i in range(self.get_emb_num_dim())]].astype(float)
		self.print_words_labels()
		sk_validation._assert_all_finite(self.get_x().values)

	def run_NN(self, are_dim_reduced):
		print("Run kNN")
		if are_dim_reduced:
			x_min, x_max = np.min(self.X_low, 0), np.max(self.X_low, 0)
			X = (self.X_low - x_min) / (x_max - x_min)
		else:
			x_min, x_max = np.min(self.get_x().values, 0), np.max(self.get_x().values, 0)
			X = (self.get_x().values - x_min) / (x_max - x_min)

		y = self.words_labels["topics_id"].values

		sk_validation._assert_all_finite(X)
		skf = StratifiedKFold(n_splits=5, random_state=self.random_state)

		fold_idx = 0
		test_acc_models = {2: [], 5: [], 20: [], 40: []}  # for k-NN we can only assess the test acc

		for train_index, test_index in iter(skf.split(X, y)):
			fold_idx += 1
			# print("=== Fold {} ===".format(fold_idx))
			X_train = X[train_index]
			y_train = y[train_index]
			X_test = X[test_index]
			y_test = y[test_index]
			n_classes = len(np.unique(y_train))
			estimators = {nn_num: neighbors.KNeighborsClassifier(nn_num, weights='distance')
			              for nn_num in [2, 5, 20, 40]}
			n_estimators = len(estimators)
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

	def run(self, model_file, is_model_bin, use_basic_label, use_rand_vec, classifier_name):
		self.convert_word2Xy(model_file, is_model_bin, use_basic_label, use_rand_vec)
		are_dim_reduced = False

		# find classifier performance
		if classifier_name == "NN":
			self.run_NN(are_dim_reduced)

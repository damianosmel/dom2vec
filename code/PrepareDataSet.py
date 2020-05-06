import os
import numpy as np
from pandas import read_csv, concat
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from utils import create_dir, is_interpro_domain
from math import ceil, log10


class PrepareDataSet:
	"""
	Class for common utilities to prepare data set
	1) Spliting in train and test:
	dataset.csv -> dataset_train.csv dataset_test.csv
	"""

	def __init__(self, dataset_path, output_path, dataset_name):
		"""
		PrepareDataSet Class init

		Parameters
		----------
		dataset_path : str
			input data set full path
		output_path : str
			output path
		dataset_name : str
			data set name

		Returns
		-------
		None
		"""
		self.dataset_path = dataset_path
		self.output_path = output_path
		create_dir(self.output_path)
		self.dataset = None
		self.dataset_train = None
		self.dataset_name = dataset_name
		self.SEED = 1234

	def read_dataset(self, use_columns):
		"""
		Read initial data csv
		if a directory is given read all the csvs file and create the merged data set
		if only one csv is given load it to create the merged data set

		Parameters
		----------
		use_columns : list of str
			list of columns names to use for data set creation

		Returns
		-------
		None
		"""
		print("Loading csv(s) -> data set.")
		if len(use_columns) > 1:
			print("Using columns {}".format(use_columns))
		if os.path.isdir(
				self.dataset_path):  # if a directory is given read all the csvs file and create the merged data set
			dfs = []
			for file_name in os.listdir(self.dataset_path):
				if file_name.endswith(".csv"):
					if len(use_columns) > 0:
						df = read_csv(os.path.join(self.dataset_path, file_name), sep=",", header=0,
						              usecols=use_columns)
					else:
						df = read_csv(os.path.join(self.dataset_path, file_name), sep=",", header=0)
					print("Current df shape: {}".format(df.shape))
					dfs.append(df)
			self.dataset = concat(dfs, axis=0, ignore_index=True)
			print("Data set shape: {}".format(self.dataset.shape))

		else:  # if only one csv is given load it to create the merged data set
			if len(use_columns) > 0:
				self.dataset = read_csv(self.dataset_path, sep=",", header=0, usecols=use_columns)
			else:
				self.dataset = read_csv(self.dataset_path, sep=",", header=0)

	def remove_duplicate_instances(self, x_col_name, y_col_name):
		"""
		Remove duplicate instances

		Parameters
		----------
		x_col_name : name of columns defining X
		y_col_name : name of columns defining Y

		Returns
		-------
		None
		"""
		print("Removing duplicate instances joining x:{} and y:{} columns".format(x_col_name, y_col_name))
		if y_col_name != None:
			self.dataset = self.dataset.drop_duplicates(subset=[x_col_name, y_col_name], keep='first')
		else:
			self.dataset = self.dataset.drop_duplicates(subset=[x_col_name], keep='first')
		print("Data set shape: {}".format(self.dataset.shape))

	def save_dataset(self, remove_ids):
		"""
		Function to save data frame containing preprocessed data set

		Parameters
		----------
		remove_ids : bool
		if True, remove ids columns before saving

		Returns
		-------
		None
		"""
		if remove_ids:
			self.dataset.drop(self.dataset.columns[[0]], axis=1, inplace=True)
		print("Saving dataset -> csv.")
		self.dataset.to_csv(os.path.join(self.output_path, self.dataset_name + ".csv"), sep=",", index=False)

	def split_train_test(self, train_test_col):
		"""
		Split data into train and test

		Parameters
		----------
		train_test_col : str
			name of column indicating train/test instance

		Returns
		-------
		None
		"""
		train_file_name = self.dataset_name + "_train.csv"
		test_file_name = self.dataset_name + "_test.csv"
		with open(os.path.join(self.output_path, train_file_name), 'w') as train_file, open(
				os.path.join(self.output_path, test_file_name), 'w') as test_file:
			if train_test_col in self.dataset.columns.values:
				print("Train/Test split is done already, creating train and test.")
				dataset_train = self.dataset.loc[self.dataset.train_test == 'train']
				dataset_test = self.dataset.loc[self.dataset.train_test == 'test']
			else:  # split on train/test -> 70/30
				dataset_train, dataset_test = np.split(self.dataset, [int(.7 * len(self.dataset))])
			print("Writing: ")
			print("dataset_train: {}".format(dataset_train.shape))
			print("dataset test: {}".format(dataset_test.shape))
			dataset_train.to_csv(train_file, sep=",", index=False)
			dataset_test.to_csv(test_file, sep=",", index=False)

	def split_train_test_stratified(self, x_columns, y_name, test_portion):
		"""
		Split data into train and test using stratified split based on Y label

		Parameters
		----------
		x_columns : list of str
			list of columns defining X
		y_name : str
			column name defining Y
		test_portion : float
			split portion of test

		Returns
		-------
		None
		"""
		print("Performing train/test split stratified for {}, test size: {}.".format(y_name, test_portion))
		train_file_name = self.dataset_name + "_train.csv"
		test_file_name = self.dataset_name + "_test.csv"
		Xy = self.dataset[x_columns]
		self.dataset_train, dataset_test = train_test_split(Xy, test_size=test_portion, stratify=Xy[y_name])
		print("Writing: ")
		print("dataset_train: {}".format(self.dataset_train.shape))
		print("dataset_test: {}".format(dataset_test.shape))
		self.dataset_train.to_csv(os.path.join(self.output_path, train_file_name), sep=",", index=False)
		dataset_test.to_csv(os.path.join(self.output_path, test_file_name), sep=",", index=False)
		print("---")

	def split_inner_stratified_kfold(self, k, x_columns, y_column):
		"""
		Split train set in k folds using stratified splits, to run cross validation

		Parameters
		----------
		k : int
			number of folds
		x_columns : list of str
			list of columns defining X
		y_column : str
			column name defining Y

		Returns
		-------
		None
		"""
		print("Splitting training set with inner {}-fold stratified on {}.".format(k, y_column))
		X = self.dataset_train[x_columns]
		y = self.dataset_train[y_column]
		skf = StratifiedKFold(n_splits=k)

		fold_index = 0
		for train_index, test_index in skf.split(X, y):
			print("Writing fold {}".format(fold_index + 1))
			train_data = self.dataset_train.iloc[train_index]
			val_data = self.dataset_train.iloc[test_index]
			train_file_name = "train_fold_" + str(fold_index) + ".csv"
			val_file_name = "val_fold_" + str(fold_index) + ".csv"
			print("train data: {}".format(train_data.shape))
			print("validation data: {}".format(val_data.shape))
			train_data.to_csv(os.path.join(self.output_path, train_file_name), sep=",", index=False)
			val_data.to_csv(os.path.join(self.output_path, val_file_name), sep=",", index=False)
			fold_index += 1
		print("---")

	def subset_train(self, percentages, num_picks):
		"""
		Subset train set by given percentage, perform subset num_picks times randomly

		Parameters
		----------
		percentages : list of float
			list of percentages to subset train set
		num_picks : int
			number of random picks for each subset

		Returns
		-------
		None
		"""
		print("Subseting training set in {} realizations for each percentage in {}".format(num_picks, percentages))
		for percentage in percentages:
			print("---")
			for num_pick in range(num_picks):
				print("{} instance of percentage {}".format(num_pick, percentage))
				train_subset = self.dataset_train.sample(frac=percentage, random_state=num_pick)
				print("subset of train: {}".format(train_subset.shape))
				subset_file_name = "train_subset" + "_p" + str(percentage) + "_" + str(num_pick) + ".csv"
				train_subset.to_csv(os.path.join(self.output_path, subset_file_name), sep=",", index=False)
		print("===")

	def split_train_kfold(self, k, x_columns):
		"""
		Split train set in k folds
		Credits: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
		Parameters
		----------
		k : int
			number of of folds
		x_columns : list of str
			list of columns defining X

		Returns
		-------
		None
		"""
		print("=== K-fold ===")
		if "train_test" in x_columns:  # for DeepLoc there exists a column to show training or testing instance
			dataset_train = self.dataset.loc[self.dataset.train_test == 'train']
		else:  # for targetP take the original dataset shuffled
			dataset_train = read_csv(os.path.join(self.dataset_path, self.dataset_name + ".csv"), sep=",", header=0)

		num_instances = len(dataset_train)
		Xy = dataset_train[x_columns]
		kf = KFold(n_splits=k, shuffle=True, random_state=self.SEED)

		for train_index, test_index in kf.split(Xy):
			print("Fold {}".format(fold))
			Xy_train, Xy_test = Xy.iloc[train_index], Xy.iloc[test_index]
			train_data = Xy_train.reset_index(drop=True)
			test_data = Xy_test.reset_index(drop=True)
			train_file_name = os.path.join(self.output_path, "train_data_" + str(fold) + ".csv")
			test_file_name = os.path.join(self.output_path, "test_data_" + str(fold) + ".csv")
			with open(train_file_name, "w") as file_train, open(test_file_name, "w") as file_test:
				print("train set: {}".format(train_data.shape))
				print("validation set: {}".format(test_data.shape))
				train_data.to_csv(file_train, sep=",", index=False)
				test_data.to_csv(file_test, sep=",", index=False)
			fold = fold + 1

		print("---")

	def shuffle_dataset(self):
		"""
		Shuffle data set

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Shuffling data set..")
		self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

	def find_common_and_oov_domains(self, train_file, test_file):
		"""
		Find common and out of vocabulary (oov) domains in train and test

		Parameters
		----------
		self : object
			PrepareDataSet object set up for analysis
		train_file : str
			train data set file name
		test_file : str
			test data set file name

		Returns
		-------
		unique_dom_train : list of str
			set of unique domains in train set
		unique_dom_test : list of str
			set of unique domains in test set
		common_dom : list of str
			set of commons domains
		"""
		train_set = read_csv(os.path.join(self.dataset_path, train_file), sep=",", header=0)
		test_set = read_csv(os.path.join(self.dataset_path, test_file), sep=",", header=0)

		dom_train = set()
		dom_test = set()
		for _, row in train_set.iterrows():
			for dom in row["interpro_domains"].split(" "):
				if is_interpro_domain(dom):
					dom_train.add(dom)
		for _, row in test_set.iterrows():
			for dom in row["interpro_domains"].split(" "):
				if is_interpro_domain(dom):
					dom_test.add(dom)

		common_dom = dom_train.intersection(dom_test)
		unique_dom_train = dom_train - common_dom
		unique_dom_test = dom_test - common_dom
		print("=== Stats ===")
		print("num of train unique domains: {}".format(len(unique_dom_train)))
		print("num of test unique domains: {}".format(len(unique_dom_test)))
		print("num of common unique domains: {}".format(len(common_dom)))
		print("=== === ===")
		return unique_dom_train, unique_dom_test, common_dom

	def calculate_domains_freq(self, data_set_file):
		"""
		Calculate the frequency of each domain in given data set

		Parameters
		----------
		self : object
			PrepareDataSet object set up for this analysis
		data_set_file : str
			data set file name

		Returns
		-------
		domains_freq : dict
			dictionary with domains as keys and their frequency as value
		"""
		domains_freq = {}
		data_set = read_csv(os.path.join(self.dataset_path, data_set_file), sep=",", header=0)
		for _, row in data_set.iterrows():
			for dom in row["interpro_domains"].split(" "):
				if is_interpro_domain(dom):
					if dom not in domains_freq:
						domains_freq[dom] = 1
					else:
						domains_freq[dom] += 1
		return domains_freq

	def plot_oov_common_domains(self, domains_freq, oov_domains, train_test_name):
		"""
		Plot the domains frequency of all domains and the oov domains in (test set)

		Parameters
		----------
		self : object
			PrepareDataSet object set up for this analysis
		domains_freq : dict
			dictionary of domain as key and their frequency as value
		oov_domains : list of str
			domains found only in the data set (next parameter)
		train_test_name : str
			train or test set name

		Returns
		-------
		None
		"""
		oov_domains_freq = {}
		common_domains_freq = {}
		for dom, freq in domains_freq.items():
			if dom in oov_domains:
				oov_domains_freq[dom] = freq
			else:
				common_domains_freq[dom] = freq

		max_x = max(max(common_domains_freq.values()), max(oov_domains_freq.values()))
		bins = np.logspace(0, ceil(log10(max_x)), num=ceil(log10(max_x)) + 1, base=10, dtype='int')
		fig, axs = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		fig.suptitle("Cumulative histograms of common & OOV domains frequencies " + train_test_name)
		axs[0].hist(list(common_domains_freq.values()), bins, density=True, histtype='bar', cumulative=True,
		            color='tab:green', edgecolor='k', alpha=0.8)
		axs[0].set(ylabel="Common", xscale='log')
		axs[0].set_xlim(1, max_x)
		axs[1].hist(list(oov_domains_freq.values()), bins, density=True, histtype='bar', cumulative=True,
		            color='tab:orange', edgecolor='k', alpha=0.8)
		axs[1].set(xlabel="Distinct occurrence of a domain", ylabel="OOV", xscale='log')
		axs[1].set_xlim(1, max_x)
		for ax in axs:
			ax.label_outer()
		hist_name = self.dataset_name + "_common_oov_domains_hist.png"
		fig.savefig(os.path.join(self.dataset_path, hist_name), bbox_inches='tight', dpi=600)

	def plot_oov_percentages_hist(self, oov_percentages, train_test_name):
		"""
		Plot the histogram of OOV domain percentage

		Parameters
		----------
		oov_percentages : list of float
			list of oov percentage per protein
		train_test_name : str
			train or test set name

		Returns
		-------
		None
		"""
		fig = plt.figure()
		n_bins = 10
		plt.hist(oov_percentages, n_bins, density=True, histtype='bar', cumulative=True, color='tab:orange',
		         edgecolor='k', alpha=0.8)
		plt.title("Cumulative histogram of OOV % per protein in " + train_test_name)
		plt.xlabel("OOV %", fontsize=14)
		plt.ylabel("Num. of proteins", fontsize=14)
		plt.xticks(np.arange(0, 1.1, 0.1))
		hist_name = self.dataset_name + "_" + train_test_name.split()[0] + "_oov_percentage_hist" + ".png"
		fig.savefig(os.path.join(self.dataset_path, hist_name), bbox_inches='tight', dpi=600)

	@staticmethod
	def calculate_oov_percentage(df, oov_domains):
		"""
		Apply function to calculate the OOV percentage per protein sample

		Parameters
		----------
			df : pandas.DataFrame to calculate OOV for
				dataframe to calculate OOV for
			oov_domains : list of str
				list of oov domains
		Returns
		-------
			oov_percentage : float
				percentage of OOV for current protein sample
		"""
		domains = str(df["interpro_domains"]).split(" ")
		oov_sum = 0
		for dom in domains:
			if dom in oov_domains:
				oov_sum += 1.0
		assert oov_sum >= 0 and oov_sum <= len(
			domains), "AssertionError: total oov should be in [0, num of all domains]"
		return oov_sum / len(domains)

	def calculate_oov_per_protein(self, data_set_file, oov_domains):
		"""
		Calculate the percentage of OOV domains per protein sample

		Parameters
		----------
		data_set_file : str
			data set file name to read protein data set
		oov_domains : list of str
			list of oov domains

		Returns
		-------
		oov_percentage : list of float
			list of oov percentage per proteins
		"""

		data_set = read_csv(os.path.join(self.dataset_path, data_set_file), sep=",", header=0)
		oov_percentages = []
		for _, protein in data_set.iterrows():
			oov_sum = 0.0
			for dom in protein["interpro_domains"].split(" "):
				if is_interpro_domain(dom):
					if dom in oov_domains:
						oov_sum += 1.0
			oov_percentage = oov_sum / len(protein["interpro_domains"].split(" "))
			oov_percentages.append(oov_percentage)
		return oov_percentages

	def count_num_proteins_per_dom_type(self):
		"""
		Count number of proteins per domain type

		Parameters
		----------
			self : object
			PrepareDataSet object set up for this analysis

		Returns
		-------
		None
		"""
		unk_count, gap_count, gap_unk_count, interpro_count = 0.0, 0.0, 0.0, 0.0

		print("Count proteins per domain type:")
		for _, protein in self.dataset.iterrows():
			domains = protein["interpro_domains"].split(" ")
			unique_domains = list(set(domains))

			if len(unique_domains) == 1:
				if len(unique_domains[0].split("_")) > 1:
					if unique_domains[0].split("_")[1] == "unk":
						unk_count += 1.0
				elif unique_domains[0] == "GAP":
					gap_count += 1.0
				else:
					interpro_count += 1.0
			elif len(unique_domains) == 2:
				if "GAP" in unique_domains:
					unique_domains.remove("GAP")
					if len(unique_domains[0].split("_")) > 1:
						if unique_domains[0].split("_")[1] == "unk":
							gap_unk_count += 1.0
					else:
						interpro_count += 1.0
				else:
					interpro_count += 1.0
			else:
				interpro_count += 1.0

		assert unk_count + gap_count + gap_unk_count + interpro_count == self.dataset.shape[
			0], "AssertionError: the total number of counts should be equal the number of protein instances."

		print("Protein instances with ONLY unknown domains: {}% ".format((unk_count / self.dataset.shape[0]) * 100.0))
		print("Protein instances with ONLY gap domains: {}%".format((gap_count / self.dataset.shape[0]) * 100.0))
		print("Protein instances with ONLY gap and unknown domains: {}%".format(
			(gap_unk_count / self.dataset.shape[0]) * 100.0))
		print("Protein instances with ONLY Interpro domains: {}%".format(
			(interpro_count / self.dataset.shape[0]) * 100.0))

	@staticmethod
	def count_unique_domains(df):
		"""
		Apply function to get the count of unique domains

		Parameters
		----------
			df : pandas.dataframe
			Data frame to get number of unique domains per protein row

		Returns
		-------
			int
			count of unique domains containing in the current protein row
		"""

		unique_domains = set(str(df["interpro_domains"]).split(" "))
		assert len(unique_domains) > 0, "AssertionError: number of unique domains should be at least 1"
		return len(unique_domains)

	def remove_unk_gap_proteins(self):
		"""
		Remove all protein instances that contain
		only unknown full length domain or only gap domain (or gap domains)

		Parameters
		----------
		self : object
			PrepareDataSet object set up for this analysis

		Returns
		-------
		None
		"""
		print("Removing protein instances with only full length unknown or only GAP domains")
		print("Starting data set shape: {}".format(self.dataset.shape))
		self.dataset["num_unique_doms"] = self.dataset.apply(self.count_unique_domains, axis=1)

		one_uniq_domain = self.dataset['num_unique_doms'] == 1
		unk_domain = self.dataset['interpro_domains'].str.contains("_unk_dom")
		gap_domain = self.dataset['interpro_domains'].str.contains("GAP")
		unk_dom_df = self.dataset.loc[one_uniq_domain & unk_domain]
		gap_dom_df = self.dataset.loc[one_uniq_domain & gap_domain]
		removed_proteins_df = concat([unk_dom_df, gap_dom_df])

		removed_proteins_df.drop("num_unique_doms", axis=1, inplace=True)
		self.dataset.drop(self.dataset[one_uniq_domain & unk_domain].index, inplace=True)
		self.dataset.drop(self.dataset[one_uniq_domain & gap_domain].index, inplace=True)
		self.dataset.drop("num_unique_doms", axis=1, inplace=True)
		print("Removed instances data frame shape: {}".format(removed_proteins_df.shape))
		print("Final data set data frame shape: {}".format(self.dataset.shape))
		removed_proteins_df.to_csv(os.path.join(self.output_path, self.dataset_name + "_removed.csv"), sep=",",
		                           index=False)

	def plot_oov_per_class_hist(self, oov_unknown_per_class, oov_interpro_per_class, train_test_name):
		"""
		Plot OOV per class in the train or test set
		# Credits: https://stackoverflow.com/questions/18449602/matplotlib-creating-stacked-histogram-from-three-unequal-length-arrays

		Parameters
		----------
		oov_unknown_per_class : dict
			dictionary with key the class name and value the OOV percentage (OOV can be unknown full protein)
		oov_interpro_per_class : dict
			dictionary with key the class name and value the OOV percentage (OOV can be Interpro domain)
		train_test_name : str
			train or test set name

		Returns
		-------
		None
		"""
		bins = np.arange(0, 1.1, 0.1)
		fig, axs = plt.subplots(len(oov_interpro_per_class), sharex=True, sharey=True)
		fig.suptitle("Histograms of OOV for unknown full-length and Interpro domains in " + train_test_name)
		for ax_idx, label_class in enumerate(oov_interpro_per_class.keys()):
			axs[ax_idx].hist([oov_interpro_per_class[label_class], oov_unknown_per_class[label_class]], bins,
			                 stacked=True, density=False, histtype='bar', cumulative=False,
			                 color=['tab:green', 'tab:orange'], edgecolor='k', alpha=0.8)
			axs[ax_idx].set_ylabel(label_class, rotation=75)
		axs[ax_idx].legend({"Interpro": 'tab:green', "Unknown full-length": 'tab:orange'})
		axs[ax_idx].set(xlabel="OOV %")

		for ax in axs:
			ax.label_outer()
		hist_name = self.dataset_name + "_" + train_test_name.split()[0] + "_oov_interpro_unk_per_class_hist.png"
		fig.savefig(os.path.join(self.dataset_path, hist_name), bbox_inches='tight', dpi=600)

	def calculate_oov_per_class(self, data_set_file, oov_domains):
		"""
		Calculate the percentage of OOV per class

		Parameters
		----------
		data_set_file : str
			data set file to read protein domains and their class
		oov_domains : list of str
			list of oov domains

		Returns
		-------
			oov_unknown_per_class : dict
				dictionary with key the class name and value the OOV percentage (OOV can be unknown full protein)
			oov_interpro_per_class : dict
				dictionary with key the class name and value the OOV percentage (OOV can be Interpro domain)
		"""
		oov_unknown_per_class = {}
		oov_interpro_per_class = {}
		data_set = read_csv(os.path.join(self.dataset_path, data_set_file), sep=",", header=0)
		for _, protein in data_set.iterrows():
			# for each protein first compute the OOV percentage of unknown full-length domains and interpro domains
			# then assign the percentages
			oov_unk = 0.0
			oov_interpro = 0.0
			no_oov = 0.0
			for dom in protein["interpro_domains"].split(" "):
				if dom in oov_domains:
					if dom[0:3] == "IPR" or dom[0:3] == "GAP":
						oov_interpro += 1
					else:
						oov_unk += 1
				else:
					no_oov += 1
			assert no_oov + oov_interpro + oov_unk == len(protein["interpro_domains"].split(
				" ")), "AssertionError: For current protein total oov and no oov does not sum up to the total number of domains."
			if protein.iloc[0] not in oov_unknown_per_class:
				oov_unknown_per_class[protein.iloc[0]] = [oov_unk / len(protein["interpro_domains"].split(" "))]
			else:
				oov_unknown_per_class[protein.iloc[0]].append(oov_unk / len(protein["interpro_domains"].split(" ")))
			if protein.iloc[0] not in oov_interpro_per_class:
				oov_interpro_per_class[protein.iloc[0]] = [oov_interpro / len(protein["interpro_domains"].split(" "))]
			else:
				oov_interpro_per_class[protein.iloc[0]].append(
					oov_interpro / len(protein["interpro_domains"].split(" ")))

		return oov_unknown_per_class, oov_interpro_per_class

	def diagnose_oov_domains(self, train_file, test_file, use_test4analysis, unk_domains_exist):
		"""
		Diagnose out of vocabulary (oov) domains

		Parameters
		----------
		self : object
			PrepareDataSet object set up for this analysis
		train_file : str
			train data set file name
		test_file : str
			test data set file name
		use_test4analysis : bool
			True if test file is used for the OOV specific plots, False if train is used
		unk_domains_exist : bool
			True if unknown domains exist in the protein instances so respective analysis will be used,
			False otherwise
		Returns
		-------
		None
		"""
		if use_test4analysis:
			file_name4analysis = "test set"
		else:
			file_name4analysis = "train set"

		print("Plot common and OOV domains occurrence histograms in " + file_name4analysis)
		unique_domains_train, unique_domains_test, common_domains = self.find_common_and_oov_domains(train_file,
		                                                                                             test_file)
		domains_freq = self.calculate_domains_freq(test_file)
		self.plot_oov_common_domains(domains_freq, unique_domains_test, file_name4analysis)

		print("Plot OOV percentage per protein in " + file_name4analysis)
		if use_test4analysis:
			oov_percentages = self.calculate_oov_per_protein(test_file, unique_domains_test)
		else:
			oov_percentages = self.calculate_oov_per_protein(train_file, unique_domains_train)
		self.plot_oov_percentages_hist(oov_percentages, file_name4analysis)

		if unk_domains_exist:
			print("Plot OOV percentage per protein per class in " + file_name4analysis)
			if use_test4analysis:
				oov_unknown_per_class, oov_interpro_per_class = self.calculate_oov_per_class(test_file,
				                                                                             unique_domains_test)
			else:
				oov_unknown_per_class, oov_interpro_per_class = self.calculate_oov_per_class(train_file,
				                                                                             unique_domains_train)
			self.plot_oov_per_class_hist(oov_unknown_per_class, oov_interpro_per_class, file_name4analysis)

	def split_test_per_oov_percentage(self, train_file, test_file, oov_percentages, used_columns):
		"""
		Split test per oov percentage

		Parameters
		----------
		train_file : str
			train data set file name
		test_file : str
			test data set file name
		oov_percentages : list of int
			list OOV percentages to split the data set
		used_columns : list of str
			columns to be loaded/used for the test data set

		Returns
		-------
		None
		"""
		print("Splitting test based on OOV")
		unique_domains_train, unique_domains_test, common_domains = self.find_common_and_oov_domains(train_file,
		                                                                                             test_file)
		print("Adding OOV to test set: ")
		test_df = read_csv(os.path.join(self.dataset_path, test_file), sep=",", header=0, usecols=used_columns)
		test_df["oov"] = test_df.apply(self.calculate_oov_percentage, axis=1, oov_domains=unique_domains_test)
		print(test_df.head())

		for percentage in oov_percentages:
			print("---")
			print("Split test for OOV <= {}".format(percentage))
			oov_percentage_lim = test_df['oov'] <= percentage
			test_oov_df = test_df.loc[oov_percentage_lim]
			print("split shape: {}".format(test_oov_df.shape))
			test_oov_file = test_file.split(".csv")[0] + "_" + str(percentage) + ".csv"
			test_oov_df.to_csv(os.path.join(self.output_path, test_oov_file), sep=",", index=False,
			                   columns=used_columns)

	@staticmethod
	def replace_domains_df(data_df, preprocess_df):
		"""
		Apply function to dataframe to replace domains with domains from preprocess_df

		Parameters
		----------
		data_df : pandas.DataFrame
			dataframe from whose domains will be replaced
		preprocess_df : pandas.DataFrame
			dataframe original after all preprocessing, it contains the new kind of domains
			used to new kind of domains
		Returns
		-------
		data_df : pandas.DataFrame
			dataframe with replaced domains
		"""
		all_domains = str(data_df["interpro_domains"]).split(" ")
		prot_seq = str(data_df["seq"])
		new_domains = preprocess_df.loc[preprocess_df["seq"] == prot_seq]["interpro_domains"].iloc[0]
		print("current: {}".format(all_domains))
		print("new: {}".format(new_domains))
		data_df["interpro_domains"] = new_domains
		return data_df

	@staticmethod
	def remove_no_interpro(data_df, id_col_exists):
		"""
		Apply function to dataframe to remove no interpro domains from respective column

		Parameters
		----------
		data_df : pandas.DataFrame
			dataframe from which to remove all non interpro domains
		id_col_exists : bool
			flag to show if id column exists in the dataset dataframe (default is True)

		Returns
		-------
		data_df : pandas.DataFrame
			dataframe with removed non interpro domains
		"""
		all_domains = str(data_df["interpro_domains"]).split(" ")

		interpro_domains = [dom for dom in all_domains if is_interpro_domain(dom)]
		if len(interpro_domains) == 0:
			# if all domain list is empty, after removal of non interpro domains,
			# add a full length unk domain
			if id_col_exists:
				interpro_domains.append(str(data_df.iloc[0]).strip(";") + "_unk_dom")  # the strip only for TargetP id
			else:
				interpro_domains.append(str(data_df.name) + "_unk_dom")
		data_df["interpro_domains"] = " ".join(interpro_domains)
		return data_df

	def remove_no_interpro_domains(self, use_columns, id_col_exists=True):
		"""
		Remove no Interpro domains (GAP and unknown full-length protein are accepted) from data set
		if a directory is given read all the csvs file and clean each csv
		if only one csv is given then clean it

		Parameters
		----------
		preprocess_out_csv : pandas.DataFrame
			initial csv after preprocessing
		use_columns: list of str
			columns names to use
		id_col_exists : bool
			flag to show if id column exists in the dataset dataframe (default is True)

		Returns
		-------
		None
		"""
		print("Cleaning rows of csv(s) from no Interpro domains.")
		if len(use_columns) > 1:
			print("Using columns {}".format(use_columns))
		self.dataset = self.dataset.apply(self.remove_no_interpro, axis=1, id_col_exists=id_col_exists)

	def replace_domains(self, preprocess_out_csv, use_columns, clean_after):
		"""
		Replace domains of each csv in self.dataset_path with domains found in preprocess_out_csv
		if a directory is given read all the csvs file and replace domains for each one
		if only one csv is given then replace domains for this only

		Parameters
		----------
		preprocess_out_csv : pandas.DataFrame
			initial csv after preprocessing (containing the new kind of domains)
		use_columns : list of str
			columns names to use
		clean_after : bool
			True if clean no Interpro after replacing domains, False otherwise

		Returns
		-------
		None
		"""
		print("Replacing domains in csv(s).")
		if len(use_columns) > 1:
			print("Using columns {}".format(use_columns))
		preprocess_df = read_csv(preprocess_out_csv, sep=",", header=0)
		print(preprocess_df.head())
		# if a directory is given read all the csvs and remove from each of them the non Interpro domains
		if os.path.isdir(self.dataset_path):
			for file_name in os.listdir(self.dataset_path):
				if file_name.endswith(".csv"):
					if len(use_columns) > 0:
						df = read_csv(os.path.join(self.dataset_path, file_name), sep=",", header=0,
						              usecols=use_columns)
					else:
						df = read_csv(os.path.join(self.dataset_path, file_name), sep=",", header=0)
					df.apply(self.replace_domains_df, preprocess_df=preprocess_df, axis=1)
					if clean_after:
						df.apply(self.remove_no_interpro, preprocess_df=preprocess_df, axis=1)
					print("Saving dataframe with only Interpro domains at {}".format(file_name))
					df.to_csv(os.path.join(self.dataset_path, file_name), sep=",", index=False)

	@staticmethod
	def calculate_oov(data_df, oov_domains):
		"""
		Apply function to get the oov ratio for protein instance

		Parameters
		----------
		data_df : pandas.DataFrame
			dataframe to read domains from
		oov_domains : list of str
			list of OOV domains

		Returns
		-------
		oov_ratio : float
			OOV ratio for current protein
		"""
		domains = str(data_df["interpro_domains"]).split(" ")
		oov_count = 0.0
		for domain in domains:
			if domain in oov_domains:
				oov_count += 1.0
		oov_ratio = oov_count / len(domains)
		return oov_ratio

	def swap_instances2increase_common_domains(self, train_file, test_file):
		"""
		Swap instances between train and test to increase common domains

		Parameters
		---------
		self : object
			PrepareDataSet object set up for this analysis
		train_file : str
			train file name
		test_file : str
			test file name

		Returns
		-------
		None
		"""

		unique_domains_train, unique_domains_test, common_domains = self.find_common_and_oov_domains(train_file,
		                                                                                             test_file)
		# find the frequencies of unique domains in train
		domains_freq_train = self.calculate_domains_freq(train_file)
		# find oov per protein in test
		print("Find OOV for test set:")
		test_df = read_csv(os.path.join(self.dataset_path, test_file), sep=",", header=0)
		test_df["oov_ratio_test"] = test_df.apply(self.calculate_oov, oov_domains=unique_domains_test, axis=1)
		test_df["oov_ratio_train"] = -1.0
		print(test_df.head())
		print("Find OOV for train set:")
		train_df = read_csv(os.path.join(self.dataset_path, train_file), sep=",", header=0)
		train_df["oov_ratio_train"] = train_df.apply(self.calculate_oov, oov_domains=unique_domains_train, axis=1)
		train_df["oov_ratio_test"] = -1.0
		print(train_df.head())

		print("Test high OOV")
		low_oov_test = test_df['oov_ratio_test'] >= 0.4
		max_oov_test = test_df['oov_ratio_test'] == 1.0
		print(test_df.loc[max_oov_test]['interpro_domains'])
		print("with shape: {}".format(test_df[low_oov_test & max_oov_test].shape))

		print("Train intermediate OOV")
		low_oov_train = train_df['oov_ratio_train'] >= 0.1
		intermediate_oov_train = train_df['oov_ratio_train'] <= 0.5
		print(train_df[low_oov_train & intermediate_oov_train].head())
		print("with shape: {}".format(train_df[low_oov_train & intermediate_oov_train].shape))

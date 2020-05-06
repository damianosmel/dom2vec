import os, ntpath
from treelib import Tree
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

import numpy as np

plt.style.use('seaborn-paper')


class ParentChildEvaluate:
	"""
	Class to perform intrinsic evaluation of embeddings using the hierarchical relation of parent/child domains

	1) parse ParendChildTreeFile.txt from interpro
	2)	for each child of root
			nn = ask embeddings model to give M nearest neighbors
		calculate_precision_atM(child.descendants, nn)
		calculate_recall_atN(child.descendants, nn)
	3) plot histogram of precision and recall

	#Credits: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
	"""

	def __init__(self, data_path):
		"""
		ParentChildEvaluate class init

		Parameters
		----------
		data_path : str
			full data path

		Returns
		-------
		None
		"""
		print("ParentChildEvaluate")
		self.data_path = data_path
		self.tree = Tree()

	def get_model_name(self):
		"""
		Get embedding model name

		Parameters
		----------

		Returns
		-------
		str
			embedding model name
		"""
		return ntpath.basename(self.model_file)

	def load_emb_model(self, model_file, is_model_binary):
		"""
		Load embedding model

		Parameters
		----------
		model_file : str
			model file name
		is_model_binary : bool
			model is saved in binary format (True), otherwise (False)

		Returns
		-------
		None
		"""
		self.model_file = model_file
		self.emb_model = KeyedVectors.load_word2vec_format(model_file, binary=is_model_binary)

	def parse_parent_child_file(self, parent_child_file_name, out_path, output_file_name, save_parsed_tree=False):
		"""
		Parse the parent child file

		Parameters
		----------
		parent_child_file_name : str
			parent child file name
		out_path : str
			output data path
		output_file_name : str
			output file name
		save_parsed_tree : bool
			after parsing save parsed tree (True), otherwise (False)

		Returns
		-------
		None
		"""
		previous_num_minus_signs = 0
		last_interpro_id = None

		self.tree.create_node("INTERPRO", "INTERPRO")
		current_parent = "INTERPRO"
		with open(parent_child_file_name, 'r') as parent_child_file:
			for line in parent_child_file:
				line = line.strip()
				current_num_minus_signs = line[0:line.find("IPR")].count("--")
				double_colon_split = line.strip("--").split("::")
				interpro_id = double_colon_split[0]
				assert interpro_id[
				       0:3] == "IPR", "AssertionError: {} \n interpro id should start with IPR and has length of 9.".format(
					interpro_id)
				if current_num_minus_signs == 0:
					# assert child not in the tree
					current_parent = "INTERPRO"
					self.tree.create_node(interpro_id, interpro_id, parent=current_parent)
				else:
					# check if you are still with current parent or you need to create a new one
					if current_num_minus_signs == previous_num_minus_signs:  # same level as last parent
						self.tree.create_node(interpro_id, interpro_id, parent=current_parent)
					elif current_num_minus_signs > previous_num_minus_signs:  # one level down from last parent -> create new parent
						current_parent = last_interpro_id
						self.tree.create_node(interpro_id, interpro_id, parent=current_parent)
					else:  # one level up from last parent -> get parent of the current parent
						if current_parent == "INTERPRO":  # if one level up is the root then your papa is the root
							papa = "INTERPRO"
						else:  # if one level up is not the root then get the parent of your parent (papa)
							papa = self.tree[current_parent].bpointer
						self.tree.create_node(interpro_id, interpro_id, parent=papa)
						current_parent = papa
				previous_num_minus_signs = current_num_minus_signs
				last_interpro_id = interpro_id

		# quick test
		# for interpro_node in self.tree.children("IPR000549"):
		#	print(interpro_node.identifier)
		# self.tree.show()
		if save_parsed_tree:
			self.tree.save2file(filename=os.path.join(out_path, output_file_name))

	def get_nn_calculate_precision_recall_atN(self, N, plot_histograms, save_diagnostics):
		"""
		Get nearest domain vector for each domains and calculate recall based on the ground truth (parsed tree)

		Parameters
		----------
		N : int
			number of nearest domain vector,
			if N==100 then retrieve as many as the children of a domain in the parsed tree
		plot_histograms : bool
			plot histograms for performance metrics (True), otherwise (False)
		save_diagnostics : bool
			save diagnostic plots for domain with low recall

		Returns
		-------
		None
		"""
		print("Get NN and calculate precision and recall at {}".format(N))
		recalls_n = []
		precisions_n = []
		interpros_recall0 = []
		interpros_num_children_recall0 = []

		if N == 100:
			retrieve_all_children = True
		else:
			retrieve_all_children = False

		for interpro_node in self.tree.children("INTERPRO"):
			recall_n = 0.0
			precision_n = 0.0
			all_children = self.tree.subtree(interpro_node.identifier).all_nodes()
			assert interpro_node in all_children, "AssertionError: parent {} is not in the set of all children.".format(
				interpro_node.identifier)
			all_children.remove(interpro_node)
			if retrieve_all_children:
				N = len(all_children)
			if self.emb_model.__contains__(interpro_node.identifier):
				nearest_neighbor_ids = set(
					[nn[0] for nn in self.emb_model.most_similar(positive=interpro_node.identifier, topn=N)])
			else:
				print("Model does not contain this id.")
				continue
			true_positives = set([child.identifier for child in all_children]).intersection(nearest_neighbor_ids)
			assert len(all_children) > 0 and len(
				nearest_neighbor_ids) == N, "AssertionError: For parent {} all children should be > 0 and nearest neighbors should be equal to N.".format(
				interpro_node.identifier)
			recall_n = len(true_positives) / len(all_children)
			precision_n = len(true_positives) / len(nearest_neighbor_ids)
			assert 0.0 <= recall_n <= 1.0 and 0.0 <= precision_n <= 1.0, "AssertionError: For parent {} recall or precision is not at (0,1]".format(
				interpro_node.identifier)
			recalls_n.append(recall_n)
			precisions_n.append(precision_n)
			if recall_n == 0.0:
				interpros_recall0.append(interpro_node.identifier)
				interpros_num_children_recall0.append(len(all_children))
		if retrieve_all_children:  # for printing in title
			N = 100
		if plot_histograms:
			if retrieve_all_children:
				self.plot_histogram(recalls_n, "Recall", "Recall", "Number of Interpro domains", "recall")
			else:
				self.plot_histogram(recalls_n, "Recall@{}".format(N), "Recall", "Number of Interpro domains",
				                    "recall_{}".format(N))
				self.plot_histogram(precisions_n, "Precision@{}".format(N), "Precision", "Number of Interpro domains",
				                    "precision_{}".format(N))
		if retrieve_all_children:
			avg_recall = sum(recalls_n) / len(recalls_n)
			print("Average recall at 100: {:.3f}".format(avg_recall))
		if save_diagnostics:
			self.save_diagnostics_recall0(interpros_recall0, interpros_num_children_recall0)

	def save_diagnostics_recall0(self, interpros_recall0, interpros_num_children_recall0):
		"""
		Save diagnostics histogram for domains with recall of 0

		Parameters
		----------
		interpros_recall0 : list of str
			interpro ids with recall 0
		interpros_num_children_recall0 : list of str
			number of children of each interpro id, found from the parsed tree, with recall 0
		Returns
		-------
		None
		"""
		print("Saving diagnostics for intepro domains with recall 0")
		with open(os.path.join(self.data_path, self.get_model_name() + "_interpros_recall0" + ".txt"),
		          "w") as interpros_recall0_file:
			# write file with names of interpro having recall 0
			interpros_recall0_file.write("\n".join(interpros_recall0))
		# plot histogram of number of children for interpro parents with recall 0
		self.plot_histogram(interpros_num_children_recall0, None,
		                    "Number of Intepro domains", "Number of children", "hist")

	def plot_histogram(self, performance_N, title, xlabel, ylabel, out_suffix):
		"""
		Plot histogram for performance metric and also for the number of children

		Parameters
		----------
		performance_N : list of float
			performance metric value per parent domain
		title : str
			histogram title (if not None)
		xlabel : str
			label x
		ylabel : str
			label y
		out_suffix : str
			histogram output file name suffix

		Returns
		-------
		None
		"""
		# plot the histogram of lengths
		fig = plt.figure()
		plt.hist(performance_N, color='g', align='left', edgecolor='k', alpha=0.8)
		plt.xlabel(xlabel, fontsize=14)
		plt.ylabel(ylabel, fontsize=14)
		if title is not None:
			plt.title(title, fontsize=14)
		plt.xticks(np.arange(0, 1.1, 0.1))
		hist_name = self.get_model_name() + "_" + out_suffix + ".png"
		fig.savefig(os.path.join(self.data_path, hist_name), bbox_inches='tight', dpi=600)

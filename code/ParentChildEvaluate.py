import os, ntpath
from treelib import Tree
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
plt.style.use('seaborn-paper')


class ParentChildEvaluate:
	"""
	Intrinsic evaluation of embeddings
	1) parse ParendChildTreeFile.txt from interpro
	2)
		for each child of root
			nn = ask embeddings model to give M nearest neighbors
		calculate_precision_atM(child.descendants, nn)
		calculate_recall_atN(child.descendants, nn)
	3) plot histogram of precision and recall

	#Credits: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
	"""

	def __init__(self, data_path):
		print("ParentChildEvaluate")
		self.data_path = data_path
		# self.model_file = model_file
		# self.is_model_binary = is_model_binary
		# self.emb_model = self.load_emb_model()
		self.tree = Tree()

	def get_model_name(self):
		return ntpath.basename(self.model_file)

	def load_emb_model(self, model_file, is_model_binary):
		self.model_file = model_file
		self.emb_model = KeyedVectors.load_word2vec_format(model_file, binary=is_model_binary)

	def parse_parentChildFile(self, parent_child_file_name, out_path, output_file_name, save_parsed_tree=False):

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
				# print("processing interpro_id:{}".format(interpro_id))

				if current_num_minus_signs == 0:
					# assert child not in the tree
					current_parent = "INTERPRO"
					self.tree.create_node(interpro_id, interpro_id, parent=current_parent)
				# current_parent = interpro_id
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

		# for interpro_node in self.tree.children("IPR000549"):
		#	print(interpro_node.identifier)
		# self.tree.show()
		if save_parsed_tree:
			self.tree.save2file(filename=os.path.join(out_path, output_file_name))

	def get_nn_calculate_precision_recall_atN(self, N, plot_histograms, save_diagnostics):
		print("Get NN and calculate precision and recall at {}".format(N))
		recalls_n = []
		precisions_n = []
		interpros_recall0 = []
		interpros_num_children_recall0 = []


		if N == 100:
			retrieve_all_children = True
		else:
			retrieve_all_children = False
		# assert self.tree.get_node("INTERPRO") self.tree.root.identifier == "INTERPRO","AssertionError root should be interpro."
		for interpro_node in self.tree.children("INTERPRO"):
			# print("evaluating for child:{}".format(interpro_node.identifier))
			recall_n = 0.0
			precision_n = 0.0
			all_children = self.tree.subtree(interpro_node.identifier).all_nodes()
			assert interpro_node in all_children, "AssertionError: parent {} is not in the set of all children.".format(
				interpro_node.identifier)
			all_children.remove(interpro_node)
			# all_children = self.tree.children(interpro_node.identifier)
			if retrieve_all_children:
				N = len(all_children)
			if self.emb_model.__contains__(interpro_node.identifier):
				nearest_neighbor_ids = set(
					[nn[0] for nn in self.emb_model.most_similar(positive=interpro_node.identifier, topn=N)])
			else:
				print("Model does not contain this id.")
				# nearest_neighbor_ids = ["example" for i in range(0,N)]
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
		print("Saving diagnostics for intepro domains with recall 0")
		with open(os.path.join(self.data_path,self.get_model_name() + "_interpros_recall0" + ".txt"),"w") as interpros_recall0_file:
			interpros_recall0_file.write(
				"\n".join(interpros_recall0))  # write file with names of interpro having recall 0
		# plot histogram of number of children for interpro parents with recall 0
		self.plot_histogram(interpros_num_children_recall0, None,
		                    "Number of Intepro domains", "Number of children", "hist")

	def plot_histogram(self, performance_N, title, xlabel, ylabel, out_suffix):
		# plot the histogram of lengths
		fig = plt.figure()
		plt.hist(performance_N, color='g', align='left',edgecolor='k',alpha=0.8)
		plt.xlabel(xlabel, fontsize=14)
		plt.ylabel(ylabel, fontsize=14)
		plt.xticks(np.arange(0, 1.1, 0.1))
		hist_name = self.get_model_name() + "_" + out_suffix + ".png"
		fig.savefig(os.path.join(self.data_path, hist_name), bbox_inches='tight', dpi=600)

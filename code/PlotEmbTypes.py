from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from pandas import read_csv
import ntpath
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PlotEmbTypes:
	"""
	Class to plot the domains embeddings and color them based on the types classified by Interpro
	Credits: https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
	"""

	def __init__(self, dom_types_file, data_path, model_file, is_model_binary, dim_size):
		"""
		PlotEmbTypes class init
		dom_types_file : str
			domain types file name
		data_path : str
			input data path
		model_file : str
			model file name
		is_model_binary : bool
			is model saved in binary format (True), otherwise (False)
		dim_size : int
			dimension size

		Returns
		-------
		None
		"""
		print("PlotEmbTypes")
		self.dom_types_file = dom_types_file
		self.data_path = data_path
		self.model_file = model_file
		self.is_model_binary = is_model_binary
		self.domain_types = read_csv(dom_types_file, sep="\t", header=0)
		self.types2colors = {"PMP-22/EMP/MP20/Claudin superfamily": "red", "small GTPase superfamily": "blue",
		                     "Kinase-pyrophosphorylase": "green", "Exonuclease, RNase T/DNA polymerase III": "cyan",
		                     "SH2 domain": "magenta"}
		self.vocab_colors = []  # list to hold interpro id: color of domain type
		self.dim_size = dim_size

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
		return ntpath.basename(self.model_file).split(".")[0]

	def load_selected_domains_emb(self):
		"""
		Load embedding vectors for selected domains

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Load embeddings")
		emb_model = KeyedVectors.load_word2vec_format(self.model_file, binary=self.is_model_binary)
		self.vocab = []
		for interpro_id in self.domain_types["interpro_id"]:
			if interpro_id in emb_model.wv.vocab:
				self.vocab.append(interpro_id)
			else:
				print("{} does not exist in the model".format(interpro_id))
		self.X = emb_model[self.vocab]
		print("Loaded {} interpros ".format(len(self.vocab)))

	def load_emb(self):
		"""
		Load embedding vector for all domains

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Load embeddings")
		emb_model = KeyedVectors.load_word2vec_format(self.model_file, binary=self.is_model_binary)
		self.vocab = list(emb_model.wv.vocab)
		if "GAP" in self.vocab:
			self.types2colors["GAP"] = "black"
		self.vocab = self.vocab
		self.X = emb_model[self.vocab]

	def run_pca(self):
		"""
		Run PCA for embedding vector space

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Run PCA")
		self.X_low = PCA().fit_transform(self.X)[:, :self.dim_size]

	def run_isomap(self):
		"""
		Run Isomap for embedding vector space

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Run isomap")
		isomap = Isomap(n_neighbors=40, n_components=self.dim_size)
		self.X_low = isomap.fit_transform(self.X)

	def run_tsne(self):
		"""
		Run t-sne for embedding vector space

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Run t-SNE")
		tsne = TSNE(perplexity=40, n_components=self.dim_size, init='pca', n_iter=2500, random_state=23)
		self.X_low = tsne.fit_transform(self.X)

	def map_superfamily2color(self):
		"""
		Get color of domain superfamily

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Map intepro id -> superfamily -> color")
		for interpro_id in self.vocab:
			assert interpro_id in self.domain_types['interpro_id'].values, "AssertionError interpro id not in the tsv"
			superfamily_type = \
			self.domain_types.loc[self.domain_types['interpro_id'] == interpro_id, 'superfamily'].iloc[0]
			assert superfamily_type in self.types2colors, "AssertionError: domain superfamily {} not found".format(
				superfamily_type)
			self.vocab_colors.append(self.types2colors[superfamily_type])

	def inter2color(self):
		"""
		Map interpro id to a color

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Map interpro ids to colors")
		for interpro_id in self.vocab:
			if interpro_id in self.domain_types['ENTRY_AC'].values:
				domain_type = self.domain_types.loc[self.domain_types['ENTRY_AC'] == interpro_id, "ENTRY_TYPE"].iloc[0]
				assert domain_type in self.types2colors, "AssertionError: domain type {} not found.".format(domain_type)
				# if domain_type in self.types2colors:
				self.vocab_colors.append(self.types2colors[domain_type])
			elif interpro_id == "GAP":
				print("annotating GAP")
				self.vocab_colors.append(self.types2colors["GAP"])

	def plot_low(self, low_method, title_msg):
		"""
		Plot low dimensional space
		Credits: https://stackoverflow.com/questions/31303912/matplotlib-pyplot-scatterplot-legend-from-color-dictionary

		Parameters
		----------
		low_method : str
			name of dimensionality reduction algorithm
		title_msg : str
			title of the plot

		Returns
		-------
		None
		"""
		print("Plot Low dimensional representation")
		if self.dim_size == 2:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			ax.scatter(self.X_low[:, 0], self.X_low[:, 1], c=self.vocab_colors, alpha=0.4)
			markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
			           self.types2colors.values()]
			plt.legend(markers, [key.replace("_", " ") for key in self.types2colors.keys()], numpoints=1, loc=9,
			           bbox_to_anchor=(0.5, -0.1), ncol=2)
		else:
			fig = plt.figure(1, figsize=(4, 3))
			plt.clf()
			ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
			plt.cla()
			ax.scatter(self.X_low[:, 0], self.X_low[:, 1], self.X_low[:, 2], c=self.vocab_colors)

		plt.title(title_msg, fontsize=18)
		plot_name = self.get_model_name() + "_types" + "_" + low_method + ".png"
		fig.savefig(join(self.data_path, plot_name), bbox_inches='tight', dpi=600)


"""
### Entry visualization ###
print("Entry.list -> Visualize embeddings with color the domain type")

dom_superfamilies_file = "/home/damian/Desktop/domains/5interpro_superfamily.tsv"
model_path = "/home/damian/Documents/L3S/projects/linear_gap/no_red_gap"
model_file_name = "dom2vec_w5_sg1_hierSoft0_dim50_e5.txt"
model_file = join(model_path,model_file_name)
data_path = model_path
is_model_bin = False
dim_size = 2
PlotEmbTypes_no_red_skip_w5_dim50 = PlotEmbTypes(dom_superfamilies_file,data_path,model_file,is_model_bin,dim_size)
PlotEmbTypes_no_red_skip_w5_dim50.load_selected_domains_emb()
PlotEmbTypes_no_red_skip_w5_dim50.map_superfamily2color()
PlotEmbTypes_no_red_skip_w5_dim50.run_pca()
PlotEmbTypes_no_red_skip_w5_dim50.plot_low("pca", "5 Interpro superfamilies")
"""

from os.path import join, isfile
import wget
from utils import get_base_name, create_dir, choose_combos
from goatools import obo_parser
from pandas import read_csv
import csv

class ParseInterPro2GO:
	"""
	Class to read all interpro domains GO annotation and write them back in a tab file
	"""

	def __init__(self, data_path, interpro2go, species_domains, species_name):
		print("~~~ Init ParseInterPro2GO ~~~")
		self.data_path = data_path
		self.interpro2go = interpro2go
		self.species_domains = species_domains
		self.species_name = species_name
		self.go_obo = None
		self.go_db = None
		self.get_GO_data()

	def get_GO_data(self):
		print("Getting GO data folder.")
		#Credits: https://nbviewer.jupyter.org/urls/dessimozlab.github.io/go-handbook/GO%20Tutorial%20in%20Python%20-%20Solutions.ipynb
		go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
		go_data_folder = join(self.data_path, "data")
		create_dir(go_data_folder)

		# Check if the file exists already
		if not isfile(join(go_data_folder,"go-basic.obo")):
			self.go_obo = wget.download(go_obo_url, join(go_data_folder, "go-basic.obo"))
		else:
			self.go_obo = join(go_data_folder, "go-basic.obo")
		self.go_db = obo_parser.GODag(self.go_obo)

	def read_species_domains(self):
		print("Reading species domains.")
		self.species_domains_dict = {}
		with open(join(self.data_path,self.species_domains), 'r') as species_domains_file:
			for line in species_domains_file:
				domain = line.strip()
				if domain not in self.species_domains_dict:
					self.species_domains_dict[domain] = 1
		print("Loaded {} domains of {} species.".format(len(self.species_domains_dict),self.species_name))

	def save_rand_comb(self, num_comb, uniq_dom2go):
		"""
		Pick num_comb random combinations from the domains column of uniq_dom2go dataframe
		:param num_comb: number of combination to pick
		:param uniq_dom2go: dataframe of domains with unique GO terms
		:return: None
		"""

		num_uniq_dom = uniq_dom2go.shape[0]
		print("Pick {} random combinations of the {} domains and save them.".format(num_comb, num_uniq_dom))
		rand_combos = choose_combos(num_uniq_dom, 2, num_comb)
		#save dataframe with the combinations
		rand_comb_name = get_base_name(self.interpro2go_tab) + "_rand_comb.csv"
		with open(join(self.data_path, rand_comb_name), 'w') as rand_comb_file:
			combo_domains_header = ["interpro_id1", "interpro_id2", "gos_id1", "gos_id2"]
			writer = csv.writer(rand_comb_file, delimiter=',')
			writer.writerow(combo_domains_header)
			for rand_combo in rand_combos:
				dom_combo = [str(uniq_dom2go.iloc[rand_combo[0]].interpro_ids), str(uniq_dom2go.iloc[rand_combo[1]].interpro_ids), str(uniq_dom2go.iloc[rand_combo[0]].GO_terms), str(uniq_dom2go.iloc[rand_combo[1]].GO_terms)]
				writer.writerow(dom_combo)

	def get_shortest_parent(self, child):
		"""
		Find recursively the shortest parent and get its description
		:param child: current go node
		:return: shortest parent (with level = 1) description
		"""
		parents = [parent for parent in child.parents]

		if len(parents) == 1 and parents[0].level == 0 and parents[0].depth == 0:
			# if you reach the root then return the child
			return child.name
		else:# find the parent of the child with lowest level and ask for his parents (recursion)
			parents.sort(key=lambda x: x.level, reverse=False)
			next_child = parents[0]
			return self.get_shortest_parent(next_child)

	def get_all_one_level_parents(self, dom_go):
		"""
		Get the annotation of all parents with level 1
		:param dom_go: go annotation of domain
		:return:
		"""
		parents_level_one = []
		parents = dom_go.get_all_parents()
		for parent in parents:
			if self.go_db[parent].level == 1:
				parents_level_one.append(self.go_db[parent].name)
		if len(parents_level_one) == 0:  # check there is no parents of level one, then get the child as parent
			parents_level_one.append(dom_go.name)

		return parents_level_one

	def extract_go_labels(self, domains2go):
		gos = domains2go["GO_terms"]
		gos_list = gos.split(" ")
		labels_shortest_parent = set()
		labels_all_one_level_parents = set()
		for go in gos_list:
			if go in self.go_db:
				# get labels from the name of the shortest parent
				labels_shortest_parent.add(self.get_shortest_parent(self.go_db[go]))
				# get labels from the name of all one level parents
				for parent_label in self.get_all_one_level_parents(self.go_db[go]):
					labels_all_one_level_parents.add(parent_label)
		if len(labels_shortest_parent) == 0:
			labels_shortest_parent.add("unknown")
		if len(labels_all_one_level_parents) == 0:
			labels_all_one_level_parents.add("unknown")
		domains2go["short_parent"] = ";".join(labels_shortest_parent)
		domains2go["one_level_parents"] = ";".join(labels_all_one_level_parents)
		return domains2go

	def get_go_labels(self):
		print("Get labels for GOs.")
		dom2go = read_csv(join(self.data_path, self.interpro2go_tab), sep="\t", header=0)
		dom2go_labels = dom2go.apply(self.extract_go_labels, axis=1)
		domains_with_labels = get_base_name(self.interpro2go_tab) + "_labels.csv"
		# print(dom2go_labels.head)
		dom2go_labels.to_csv(join(self.data_path, domains_with_labels), sep=",", index=False)

	def traverse_domains(self):
		print("Traversing domains.")
		dom2go = read_csv(join(self.data_path, self.interpro2go_tab), sep="\t", header=0)
		num_dom = dom2go.shape[0]
		unique_domains_name = get_base_name(self.interpro2go_tab) + "_unique.tab"

	def remove_duplicate_domains(self):
		"""
		Filter out redundant domains, that is remove all but the first of the following domains:
		IPR001433	GO:0016491
		IPR001709	GO:0016491
		IPR001834	GO:0016491
		:return:
		"""
		print("Filtering out duplicate domains.")
		dom2go = read_csv(join(self.data_path, self.interpro2go_tab), sep="\t", header=0)
		num_dom = dom2go.shape[0]
		unique_domains_name = get_base_name(self.interpro2go_tab) +"_unique.tab"
		uniq_dom2go = dom2go.drop_duplicates(["GO_terms"])
		uniq_dom2go.to_csv(join(self.data_path, unique_domains_name), sep="\t", index=False)
		num_uniq_dom = uniq_dom2go.shape[0]
		print("Reducing domains from {} to {}.".format(num_dom,num_uniq_dom))
		return uniq_dom2go

	def convert_to_tab(self, keep_only_MF):
		"""
		Convert mapping of interpro to GOs into tabular file
		For each interpro domain in species file, read all GOs and arrange them as the column of the row
		:return: None
		"""
		print("Converting to tabs.")
		self.read_species_domains()
		interpro2go_tab = get_base_name(self.interpro2go) + "_" + self.species_name + "_MF.tab" if keep_only_MF else ".tab"
		self.interpro2go_tab = interpro2go_tab
		num_written_lines = 0
		with open(self.interpro2go, 'r') as interpro2go_file, open(join(self.data_path, interpro2go_tab), 'w') as interpro2go_tab_file:
			interpro2go_tab_file.write("interpro_ids\tGO_terms\n")
			previous_id = " "
			previous_go_terms = []
			for interpro2go_line in interpro2go_file:
				if interpro2go_line[0] != "!":
					current_id = interpro2go_line.strip().split("InterPro:")[1].split(" ")[0]
					assert current_id[:3] == "IPR", "AssertionError: interpro id must start with IPR.\n line: {}".format(interpro2go_line)
					current_go_term = interpro2go_line.strip().split(" ; ")[-1]
					if keep_only_MF and (current_go_term in self.go_db and self.go_db[current_go_term].namespace != "molecular_function"):
						# print("Skipping GO as it's not a no molecular function annotation.")
						continue
					if previous_id == " ": #init
						previous_go_terms.append(current_go_term)
						previous_id = current_id
					else:
						if current_id == previous_id:#still in the same interpro domain
							previous_go_terms.append(current_go_term)
						else: #on another interpro domain
							assert previous_id != " ", "AssertionError: id must not be null.\n line: {}".format(interpro2go_line)
							assert len(previous_go_terms) > 0, "AssertionError: each interpro should have at least one GO.\n line:{}".format(interpro2go_line)
							if previous_id in self.species_domains_dict:
								interpro2go_tab_file.write(previous_id + '\t' + " ".join(previous_go_terms)+"\n")
								num_written_lines = num_written_lines + 1
							previous_id = current_id
							previous_go_terms = [current_go_term]

		print("Saved {} interpro2GO tabs in {}.".format(num_written_lines, interpro2go_tab))

	def convert_go_labels(self, keep_only_MF):
		# convert to tabs
		self.convert_to_tab(keep_only_MF)
		self.get_go_labels()

	def run(self, keep_only_MF, num_comb):
		#convert to tabs
		self.convert_to_tab(keep_only_MF)
		#remove domains if they have the same GO
		uniq_dom2go = self.remove_duplicate_domains()
		self.save_rand_comb(num_comb, uniq_dom2go)
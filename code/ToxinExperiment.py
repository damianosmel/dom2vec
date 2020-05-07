from Bio import SeqIO
from os import listdir
from os.path import join, basename, splitext, isfile
import csv
from utils import batch_iterator


class ToxinExperiment:
	"""
	Class to set up Toxin experiment:
	1) convert fasta to csv for Pytorch
	2) get labels from fasta
	"""

	def __init__(self, fasta_dir_path, domains_path, output_path):
		"""
		ToxinExperiment class init

		Parameters
		----------
		fasta_dir_path : str
			full path to fasta file directory
		domains_path : str
		output_path : str

		Returns
		-------
		None
		"""
		self.fasta_dir_path = fasta_dir_path
		self.domains_path = domains_path
		self.output_path = output_path

	@staticmethod
	def get_labels(fasta_file):
		"""
		Extract label from fasta file name
		Please see paper:
		Gacesa, Ranko, David J. Barlow, and Paul F. Long.
		"Machine learning can differentiate venom toxins from other proteins having non-toxic physiological functions."
		PeerJ Computer Science 2 (2016): e90.
		Section: Methods - Datasets.

		Parameters
		----------
		fasta_file : str
			full path of fasta file

		Returns
		-------
		str
			String of label name(Toxin (pos), No_toxin (hard))
		"""
		base_name = basename(fasta_file)
		name = splitext(base_name)[0]
		label = name.split("_")[-1]
		assert label == "pos" or label == "hard", "AssertionError: label {} not found, possible labels pos, hard."
		if label == "pos":
			return "Toxin"
		elif label == "hard":
			return "No_toxin"

	def extract_uniprot4protein_keys(self, proteins_dict):
		"""
		Extract only the uniprot id as key of the proteins_dict
		Example: 'sp_tox|F8QN53|PA2A2_VIPRE' -> F8QN53

		Parameters
		----------
		proteins_dict : dict
			protein record dict as provided by SeqIO

		Returns
		-------
		str
			uniprot id
		"""
		return {key.split("|")[1]: value for (key, value) in proteins_dict.items()}

	def fasta2csv(self, is_local_interpro):
		"""
		Convert fasta file to csv

		Parameters
		----------
		is_local_interpro : bool
			the input fasta file is created by running local Interproscan (True), otherwise (False)

		Returns
		-------

		"""
		print("Creating row for each protein with domain, please wait..")
		dataset_name = "toxin_dataset.csv"
		num_all_proteins = 0
		num_proteins_with_domains = 0
		num_remain_proteins = 0
		csv_already_exists = True
		if not isfile(join(self.output_path, dataset_name)):  # if csv not exists then firstly write header
			csv_already_exists = False
		for fasta_file in listdir(self.fasta_dir_path):
			short_label = splitext(basename(fasta_file))[0].split(".")[0]
			with open(join(self.fasta_dir_path, fasta_file), 'r') as fasta_data, open(self.domains_path,
			                                                                          'r') as domains_data, open(
					join(self.output_path, dataset_name), 'a') as dataset_csv, open(
					join(self.output_path, "targetp_remaining_seq" + "_" + short_label + ".fasta"),
					'a') as remain_seqs_file:
				proteins_dict = SeqIO.to_dict(SeqIO.parse(fasta_data, "fasta"))
				num_all_proteins += len(proteins_dict)
				uniprot2prot = self.extract_uniprot4protein_keys(proteins_dict)
				writer = csv.writer(dataset_csv, delimiter=',')
				if not csv_already_exists:  # if csv not exists then firstly write header
					proteins_domains_header = ["uniprot_id", "toxin", "seq", "seq_len", "interpro_domains",
					                           "evidence_db_domains"]
					writer.writerow(proteins_domains_header)
					csv_already_exists = True
				batch_num_lines = 10000

				for i, batch in enumerate(batch_iterator(domains_data, batch_num_lines)):
					for line in batch:
						line_split = line.strip().split("\t")
						assert len(line_split) == 3, "AssertionError: {} does not have 3 tabs.".format(line)
						uniprot_id = line_split[0]
						if uniprot_id == "uniprot_id":
							print("Skipping first line")
							continue
						if is_local_interpro:
							uniprot_id = uniprot_id.split("|")[1]
						if uniprot_id in uniprot2prot:
							interpro_ids = line_split[1]
							evidence_db_ids = line_split[2]
							label = self.get_labels(fasta_file)
							# make the row of the current protein
							protein_row = [uniprot_id, label, str(uniprot2prot[uniprot_id].seq),
							               len(str(uniprot2prot[uniprot_id].seq)), interpro_ids, evidence_db_ids]
							writer.writerow(protein_row)
							num_proteins_with_domains += 1
							# remove found protein from the dictionary, to keep track of the remaining proteins
							uniprot2prot.pop(uniprot_id)

				num_remain_proteins += len(uniprot2prot)  # update num of remain proteins
				SeqIO.write(uniprot2prot.values(), remain_seqs_file, "fasta")  # append remaining proteins to fasta
				print("num of remaining proteins for {} label: {} saved on remaining fasta".format(
					self.get_labels(fasta_file), len(uniprot2prot)))
		assert num_all_proteins == num_proteins_with_domains + num_remain_proteins, "AssertionError: total num of proteins should be equal to proteins with domains + proteins without domains."
		print("num of Toxin proteins: {}".format(num_all_proteins))
		print("num of Toxin proteins with found domains: {}".format(num_proteins_with_domains))
		print("num of remaining proteins with not found domains: {}".format(num_remain_proteins))

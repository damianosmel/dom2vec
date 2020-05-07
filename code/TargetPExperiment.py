from Bio import SeqIO
from os import listdir
from os.path import join,basename,splitext,isfile
import csv
from utils import batch_iterator


class TargetPExperiment:
	"""
	Class to set up TargetP experiment:
	1) Convert fasta to csv for Pytorch
	2) get labels from fasta
	"""
	def __init__(self, fasta_dir_path, domains_path, output_path):
		"""
		TargetPExperiment class init

		Parameters
		----------
		fasta_dir_path : str
			full path to directory with input fasta files
		domains_path : str
			domains full path
		output_path : str
			output full path

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
		Extract label from the fasta file name
		For example: String of label name (Mitochondrial (mTP), Secretory Pathway/Signal Peptide (SP), Nuclear sequences (nuc), Cytosolic (cyt))
		based on: http://www.cbs.dtu.dk/services/TargetP-1.1/datasets/datasets.php
		Parameters
		----------
		fasta_file : str
			full path of fasta file

		Returns
		-------
		str
			label name
		"""
		base_name = basename(fasta_file)
		name = splitext(base_name)[0]
		label = name.split(".")[0]

		assert label == "cyt" or label == "mTP" or label == "nuc" or label == "SP", "AssertionError: label {} not found, possible labels: cyt, mTP, nuc, SP"
		if label == "cyt":
			return "Cytosole"
		elif label == "mTP":
			return "Mitochondrial"
		elif label == "SP":
			return "PathwaySignal"
		elif label == "nuc":
			return "Nuclear"

	def fasta2csv(self):
		"""
		Convert a directory of fasta files to data csv

		Parameters
		----------

		Returns
		-------
		None
		"""
		print("Creating row for each protein with domain, please wait..")
		dataset_name = "targetp_dataset.csv"
		num_all_proteins = 0
		num_proteins_with_domains = 0
		num_remain_proteins = 0
		csv_already_exists = True
		if not isfile(join(self.output_path, dataset_name)):  # if csv not exists then firstly write header
			csv_already_exists = False
		for fasta_file in listdir(self.fasta_dir_path):
			short_label = splitext(basename(fasta_file))[0].split(".")[0]
			with open(join(self.fasta_dir_path, fasta_file),'r') as fasta_data, open(self.domains_path,'r') as domains_data, open(join(self.output_path,dataset_name),'a') as dataset_csv, open(join(self.output_path, short_label+"."+"targetp_remaining_seq.fasta"),'a') as remain_seqs_file:
				proteins_dict = SeqIO.to_dict(SeqIO.parse(fasta_data,"fasta"))
				num_all_proteins += len(proteins_dict)
				writer = csv.writer(dataset_csv, delimiter=',')
				if not csv_already_exists: #if csv not exists then firstly write header
					proteins_domains_header = ["uniprot_id","cellular_location","seq","seq_len","interpro_domains","evidence_db_domains"]
					writer.writerow(proteins_domains_header)
					csv_already_exists = True

				batch_num_lines = 10000
				for i, batch in enumerate(batch_iterator(domains_data, batch_num_lines)):
					for line in batch:
						line_split = line.strip().split("\t")
						assert len(line_split) == 3, "AssertionError: {} does not have 3 tabs.".format(line)
						uniprot_id = line_split[0]
						if uniprot_id in proteins_dict:
							interpro_ids = line_split[1]
							evidence_db_ids = line_split[2]
							label = self.get_labels(fasta_file)
							# make the row of the current protein
							protein_row = [uniprot_id, label, str(proteins_dict[uniprot_id].seq),len(str(proteins_dict[uniprot_id].seq)), interpro_ids, evidence_db_ids]
							writer.writerow(protein_row)
							num_proteins_with_domains += 1
							# remove found protein from the dictionary, to keep track of the remaining proteins
							proteins_dict.pop(uniprot_id)

				num_remain_proteins += len(proteins_dict) # update num of remain proteins
				SeqIO.write(proteins_dict.values(), remain_seqs_file, "fasta") # append remaining proteins to fasta
				print("num of remaining proteins for {} label: {} saved on remaining fasta".format(self.get_labels(fasta_file), len(proteins_dict)))

		### processed proteins stats ###
		assert num_all_proteins == num_proteins_with_domains + num_remain_proteins, "AssertionError: total num of proteins should be equal to proteins with domains + proteins without domains."
		print("num of TargetP proteins: {}".format(num_all_proteins))
		print("num of TargetP proteins with found domains: {}".format(num_proteins_with_domains))
		print("num of remaining proteins with not found domains: {}".format(num_remain_proteins))
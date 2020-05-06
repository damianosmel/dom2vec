import os
import gzip
from Protein import Protein
from utils import batch_iterator
from Bio import SeqIO


class Preprocess:
	"""
	Class to preprocess file from interpro database, found at:
	https://www.ebi.ac.uk/interpro/beta/download/protein2ipr.dat.gz
	"""

	def __init__(self, data_path, prot_len_file_name, with_overlap, with_redundant, with_gap, interpro_local_format):
		"""
		Preprocess class init

		Parameters
		----------
		data_path : str
			full data path
		prot_len_file_name : str
			file name containing protein length information
		with_overlap : bool
			output overlapping domain annotation (True), otherwise not overlapping domain annotation will be created (False)
		with_redundant : bool
			if with_overlap is False then create non overlapping (but possibly redundant) domains (True),
			otherwise create non overlapping and non redundant domain annotation (False)
		with_gap : bool
			add GAP domain for each protein subsequence >30 amino acids without domain hit (True),
			otherwise don't add GAP domain (False)
		interpro_local_format : bool
			preprocess output format produced by local interproscan run (True),
			otherwise preprocess Interpro downloaded protein2ipr format (False)

		Returns
		-------
		None
		"""
		self.data_path = data_path
		self.prot_len_file_name = prot_len_file_name
		self.with_overlap = with_overlap
		self.with_redundant = with_redundant
		self.with_gap = with_gap
		self.last_protein = Protein(self.with_overlap, self.with_redundant, self.with_gap)
		self.proteins = []
		self.interpro_local_format = interpro_local_format
		self.num_prot_with_no_interpro = 0

	def update_no_intepro(self):
		"""
		Update statistic count for proteins without interpro domain

		Parameters
		----------

		Returns
		-------
		None
		"""
		# check how many interpro ids exist for domains of proteins
		for protein in self.proteins:
			if sum(protein.interpro_exist_all_domains) == 0:
				self.num_prot_with_no_interpro = self.num_prot_with_no_interpro + 1

	def update_output(self, file_out):
		"""
		Update output tabular file

		Parameters
		----------
		file_out : str
			output file name

		Returns
		-------
		None
		"""
		for protein in self.proteins:
			file_out.write(protein.to_tabs())

	def create_file_out_name(self):
		"""
		Create output file name based on the type of domain annotation that was selected

		Parameters
		----------

		Returns
		-------
		str
			created output file name
		"""
		file_out_name = "id_domains"

		if self.with_overlap:
			file_out_name = file_out_name + "_overlap"
		elif self.with_redundant is False:
			file_out_name = file_out_name + "_no_overlap"
		else:
			file_out_name = file_out_name + "_no_redundant"
		if self.with_gap:
			file_out_name = file_out_name + "_gap"
		else:
			file_out_name = file_out_name + "_no_gap"
		return file_out_name + ".tab"

	def parse_prot2in(self, file_in_name, batch_num_lines, batch_num_prot):
		"""
		Parse protein domain hits to create tabular formatted file relating each protein to its domains

		Parameters
		----------
		file_in_name : str
			input file name
		batch_num_lines : int
			number of lines to be parsed per batch
		batch_num_prot : int
			number of proteins to be processed per batch

		Returns
		-------
		None
		"""
		file_out_name = self.create_file_out_name()
		total_out_prot = 0
		if self.prot_len_file_name != "":
			prot_file = open(os.path.join(self.data_path, self.prot_len_file_name), 'r')
		else:
			prot_file = ""

		# check if output tabular file already exists, if yes then don't add header
		output_exists_already = False
		if os.path.isfile(os.path.join(self.data_path, file_out_name)):
			output_exists_already = True

		with gzip.open(os.path.join(self.data_path, file_in_name), 'rt') as file_in, open(
				os.path.join(self.data_path, file_out_name), 'a') as file_out:
			if not output_exists_already:
				# write the header of the output file
				file_out.write("uniprot_id\tinterpro_ids\tevidence_db_ids\n")
			line_count = 0
			for i, batch in enumerate(batch_iterator(file_in, batch_num_lines)):
				for hit_line in batch:
					hit_line = hit_line.strip()
					hit_tabs = hit_line.split("\t")
					if self.interpro_local_format:
						assert len(hit_tabs) >= 11, "AssertionError: " + hit_line + "has less than 11 tabs."
					else:
						assert len(hit_tabs) == 6, "AssertionError: " + hit_line + " has more than 6 tabs."
					if self.last_protein.uniprot_id == "":
						# initialize protein list
						protein = Protein(self.with_overlap, self.with_redundant, self.with_gap, hit_line, prot_file,
						                  self.interpro_local_format)
						self.last_protein = protein
						self.proteins.append(protein)
					else:
						if Protein.get_prot_id(hit_line) == self.last_protein.uniprot_id:
							# update last created protein
							self.last_protein.add_domain(hit_line)
						else:
							# write to file complete proteins
							if len(self.proteins) == batch_num_prot:
								self.update_output(file_out)
								total_out_prot = total_out_prot + len(self.proteins)
								self.update_no_intepro()
								del self.proteins[:]
							# create new protein and append it to proteins
							protein = Protein(self.with_overlap, self.with_redundant, self.with_gap, hit_line,
							                  prot_file, self.interpro_local_format)
							self.last_protein = protein
							self.proteins.append(protein)
					line_count = line_count + 1
				# save last proteins
				self.update_output(file_out)
				total_out_prot = total_out_prot + len(self.proteins)
				self.update_no_intepro()
				del self.proteins[:]
		if self.prot_len_file_name != "":
			prot_file.close()
		print("Successfully parsed {} lines.".format(line_count))
		print("Successfully created {} proteins.".format(total_out_prot))
		print("Number of proteins without any interpro annotation: {}.".format(self.num_prot_with_no_interpro))

	def create_domains_corpus(self, file_in_name, file_out_name, batch_num_lines):
		"""
		Create domain corpus from protein domains tabular file

		Parameters
		----------
		file_in_name : str
			input file name
		file_out_name : str
			output file name
		batch_num_lines : int
			number of lines to be processed per batch

		Returns
		-------
		None
		"""
		total_out_lines = 0
		with open(os.path.join(self.data_path, file_in_name), 'r') as file_in, open(
				os.path.join(self.data_path, file_out_name), 'a') as file_out:
			for i, batch in enumerate(batch_iterator(file_in, batch_num_lines)):
				for line in batch:
					line_tabs = line.split("\t")
					assert len(line_tabs) == 3, "AssertionError: line should have only three tabs."
					protein_domains = line_tabs[1]
					if protein_domains.strip() != "interpro_ids":
						file_out.write(protein_domains + "\n")
						total_out_lines = total_out_lines + 1
		print("Successfully written {} proteins in domains representation.".format(total_out_lines))

	def fasta2default_domains(self, fasta_name, data_id_format):
		"""
		Convert a fasta file containing proteins without any interproscan domain hit
		(mainly for prediction tasks)

		Parameters
		----------
		fasta_name : str
			input fasta name
		data_id_format : int
			data set contains id format in following types: protein ids (0), protein ids but remove ending ";" (1),
			protein ids can be extracted by splitting at "|"

		Returns
		-------
		None
		"""
		file_out_name = "default_domains.tab"
		with open(os.path.join(self.data_path, fasta_name), "r") as fasta_file, open(
				os.path.join(self.data_path, file_out_name), "w") as file_out:
			file_out.write("uniprot_id\tinterpro_ids\tevidence_db_ids\n")
			for protein in SeqIO.parse(fasta_file, "fasta"):
				if data_id_format == 0:
					# DeepLoc
					domain_annot = protein.id + "_unk_dom"
					evid_annot = protein.id + "_unk_evid"
				elif data_id_format == 1:
					# for targetP remove ending ;
					domain_annot = protein.id.strip(";") + "_unk_dom"
					evid_annot = protein.id.strip(";") + "_unk_evid"
				elif data_id_format == 2:
					# Toxin
					domain_annot = protein.id.split("|")[1] + "_unk_dom"
					evid_annot = protein.id.split("|")[1] + "_unk_evid"
				file_out.write("\t".join([protein.id, domain_annot, evid_annot]) + "\n")

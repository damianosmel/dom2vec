from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import csv
from os.path import join, isfile
from utils import split_fasta, batch_iterator

class NEWExperiment:
	"""
	Class to read NEW dataset for enzyme commission:
	1) convert txt to fasta
	2) get labels from fasta
	"""
	def __init__(self, input_path, domains_path, output_path):
		self.input_path = input_path
		self.domains_path = domains_path
		self.output_path = output_path

	def get_label(self, prot_description):
		#convert description of protein to EC primary class
		#e.g. >6634 2.5.1.16 -desc-> 2.5.1.16 -> 2
		#Follow the introduction of
		#to get the mapping of EC primary class to Enzyme class names
		#(i) oxidoreductases, (ii) transferases, (iii) hydrolases, (iv) lyases, (v) isomerases and (vi) ligases, represented by the first digit.
		ec_num2name = {"1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases", "4": "Lyases", "5": "Isomerases", "6": "Ligases"}
		ec_num = prot_description.split(".")[0][-1]
		assert ec_num in ec_num2name, "AssertionError: current description {} is not one of the six known EC classes".format(prot_description)

		return ec_num2name[ec_num]

	def txt2fastas(self, txt_name):
		"""
		Read line by line and save the sequence of each protein along with an incremental id and its label
		from ^1.1.1.100>VYEQVSIEVPQSVEAPVVIITGASEIEASTIQALSFGPDVXKEADVEAMIKAVDAWGQVDVLINNAGITRAGVIGLQKNINVNAIAPGFIASDMTAKILETIPLGR
		to >id_num 1.1.1.100 \n VYEQVSIEVPQSVEAPVVIITGASEIEASTIQALSFGPDVXKEADVEAMIKAVDAWGQVDVLINNAGITRAGVIGLQKNINVNAIAPGFIASDMTAKILETIPLGR
		Following the documentation at the original data file: https://www.cbrc.kaust.edu.sa/DEEPre/dataset.html
		:return:
		"""
		#create the fasta output name
		fasta_name = "new_dataset.fasta"
		prot_records = []
		line_count = 0
		with open(join(self.input_path, txt_name), 'r') as txt_in, open(join(self.output_path, fasta_name), 'w') as fasta_out:
			for data_line in txt_in:
				line_splits = data_line.lstrip("^").strip().split(">")
				assert len(line_splits) == 2, "AsseertionError: All data lines of NEW data set should separated in two parts by the character >"
				line_count += 1
				prot_records.append(SeqRecord(Seq(line_splits[1], IUPAC.protein), id=str(line_count), name="", description=line_splits[0]))
			print("Parsed {} proteins from txt file, now saving them to {}.".format(line_count,fasta_name))
			SeqIO.write(prot_records, fasta_out, "fasta")

		#split fasta to make faster the local interpro run (22,168 into 3 fastas of at most 8000 sequences)
		split_fasta(self.output_path,self.output_path,fasta_name,8000)

	def fasta2csv(self, fasta_name):
		print("Creating row for each protein with domain, please wait..")
		dataset_name = "new_dataset.csv"
		num_all_proteins = 0
		num_proteins_with_domains = 0
		num_remain_proteins = 0
		csv_already_exists = True
		if not isfile(join(self.output_path, dataset_name)): #if csv does not exist write header
			csv_already_exists = False
		with open(join(self.input_path, fasta_name), 'r') as fasta_data, open(self.domains_path, 'r') as domains_data, \
			open(join(self.output_path, dataset_name), 'a') as dataset_csv,\
			open(join(self.output_path, "new_remaining_seq.fasta"), 'w') as remaining_seq_file:
			proteins_dict = SeqIO.to_dict(SeqIO.parse(fasta_data,"fasta"))
			num_all_proteins = len(proteins_dict)
			writer = csv.writer(dataset_csv, delimiter=',')
			proteins_domains_header = ["id","ec","seq","seq_len","interpro_domains","evidence_db_domains"]
			if not csv_already_exists:
				writer.writerow(proteins_domains_header)
				csv_already_exists = True
			batch_num_lines = 10000

			for i, batch in enumerate(batch_iterator(domains_data,batch_num_lines)):
				for line in batch:
					line_split = line.strip().split("\t")
					assert len(line_split) == 3, "AssertionError: {} does not have 3 tabs.".format(line)
					prot_id = line_split[0]
					if prot_id == "uniprot_id":
						print("Skipping first line")
						continue
					else:
						if prot_id in proteins_dict:
							# print("Writing row for prot id {}".format(prot_id))
							interpro_ids = line_split[1]
							evidence_db_ids = line_split[2]
							label = self.get_label(proteins_dict[prot_id].description)
							# make the row of current protein
							protein_row = [prot_id, label, str(proteins_dict[prot_id].seq), len(str(proteins_dict[prot_id].seq)), interpro_ids, evidence_db_ids]
							writer.writerow(protein_row)
							num_proteins_with_domains += 1
							proteins_dict.pop(prot_id) #remove found protein from whole proteins dictionary
			num_remain_proteins = len(proteins_dict)
			assert num_all_proteins == num_proteins_with_domains + num_remain_proteins, "AssertionError: total num of proteins should be equal to proteins with domains + proteins without domains."
			SeqIO.write(proteins_dict.values(),remaining_seq_file,"fasta")
			print("num of NEW proteins: {}".format(num_all_proteins))
			print("num of NEW proteins with found domains: {}".format(num_proteins_with_domains))
			print("num of remaining proteins with not found domains: {}".format(len(proteins_dict)))
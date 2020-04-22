from Bio import SeqIO
import os
from utils import batch_iterator,create_dir
import csv
import collections

class DeepLocExperiment:
	"""
	Class to set up DeepLoc experiments:

	1) convert fasta -> csv for Pytorch
	2) split csv -> train and test
	"""

	def __init__(self,fasta_path,domains_path,output_path,label_name):
		self.fasta_path = fasta_path
		self.domains_path = domains_path
		self.output_path = output_path
		self.label_name = label_name
		create_dir(self.output_path)

	def get_labels(self,prot_description):
		"""
		Get labels from protein description for info please see: http://www.cbs.dtu.dk/services/DeepLoc/data.php
		:param prot_description: string of a DeepLoc protein like "Q9H400 Cell.membrane-M test"
		:return: labels.loc -> location, labels.soluble -> membrane or soluble, labels.train -> train or test
		"""
		labels = collections.namedtuple('LocSolTest', ["loc", "sol", "train"])
		descr_split = prot_description.strip().split(" ")
		assert len(descr_split) >= 2, "Protein description: {} has less information than usual.".format(prot_description)
		descr_label = descr_split[1].split("-")
		if len(descr_label) == 2:
			labels.loc = descr_label[0]
			labels.sol = descr_label[1]
		else:#case like A1L020 Cytoplasm-Nucleus-U
			# in such cases keep the first annotation as for A1L020 the uniprot says that "predominantly expressed in Cytoplasm and shuttles.."
			# https://www.uniprot.org/uniprot/A1L020
			# labels.loc = "-".join([descr_label[0],descr_label[1]])
			labels.loc = descr_label[0]
			labels.sol = descr_label[2]
		if len(descr_split) == 3:#if there is third part, then it is test instance
			labels.train = descr_split[2]
		else:
			labels.train = "train"
		return labels

	def fasta2csv(self,value2remove):
		print("Creating row for each protein with domains, please wait..")
		dataset_name = "deeploc_dataset_" + self.label_name + ".csv"
		with open(self.fasta_path,'r') as fasta_data, open(self.domains_path,'r') as domains_data, open(os.path.join(self.output_path,dataset_name),'w') as dataset_csv,open(os.path.join(self.output_path,"deeploc_remaining_seq.fasta"),'w') as remain_seqs_file:
			proteins_dict = SeqIO.to_dict(SeqIO.parse(fasta_data, "fasta"))
			num_all_proteins = len(proteins_dict)
			proteins_domains_header = ["uniprot_id", "train_test", "cellular_location", "membrane_soluble", "seq", "seq_len", "interpro_domains", "evidence_db_domains"]
			writer = csv.writer(dataset_csv, delimiter=',')
			writer.writerow(proteins_domains_header)
			batch_num_lines = 10000
			num_proteins_with_domains = 0
			for i, batch in enumerate(batch_iterator(domains_data,batch_num_lines)):
				for line in batch:
					line_split = line.strip().split("\t")
					assert len(line_split) == 3, "AssertionError: {} does not have 3 tabs.".format(line)
					uniprot_id = line_split[0]
					if uniprot_id in proteins_dict:
						print("Writing row for {}".format(uniprot_id))
						#print(proteins_dict[uniprot_id])
						interpro_ids = line_split[1]
						evidence_db_ids = line_split[2]
						labels = self.get_labels(proteins_dict[uniprot_id].description)
						#make the row of current protein
						protein_row = [uniprot_id,labels.train,labels.loc,labels.sol,str(proteins_dict[uniprot_id].seq),len(str(proteins_dict[uniprot_id].seq)),interpro_ids,evidence_db_ids]
						if(value2remove!=""):
							if labels.sol == value2remove:
								print("Skipping protein {} having membrane_soluble as {}".format(uniprot_id,labels.sol))
							else:
								writer.writerow(protein_row)
						else:
							writer.writerow(protein_row)
						num_proteins_with_domains = num_proteins_with_domains + 1
						proteins_dict.pop(uniprot_id)#remove found protein from the dictionary, to keep track of the remaining proteins

			SeqIO.write(proteins_dict.values(), remain_seqs_file, "fasta")
		print("num of DeepLoc proteins: {}".format(num_all_proteins))
		print("num of DeepLoc proteins with found domains: {}".format(num_proteins_with_domains))
		print("num of remaining proteins with not found domains: {}".format(len(proteins_dict)))
		return os.path.join(self.output_path, dataset_name)
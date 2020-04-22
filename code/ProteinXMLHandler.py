import xml.sax
import os
""""
Class to read all proteins from match_complete.xml.gz from InterPro, with the aim to retrieve the protein length
#Credits: https://www.tutorialspoint.com/python3/python_xml_processing.htm
"""


class ProteinXMLHandler(xml.sax.ContentHandler):

	def __init__(self):
		self.CurrentData = ""
		self.id = ""
		self.seq_len = ""
		self.data_path = "/home/damian/Documents/L3S/projects"
		self.prot_len_file = "prot_id_len.tab"
		self.file_out = open(os.path.join(self.data_path,self.prot_len_file),'a')
		self.file_out.write("uniprot_id\tseq_len\n")

	# Call when an element starts
	def startElement(self, tag, attributes):
		self.CurrentData = tag
		if tag == "protein":
			self.id = attributes["id"]
			self.seq_len = attributes["length"]
			self.file_out.write(self.id+"\t"+self.seq_len+"\n")

	# Call when an elements ends
	def endElement(self, tag):
		self.CurrentData = ""

	# Call when a character is read
	def characters(self, content):
		self.currentData = ""

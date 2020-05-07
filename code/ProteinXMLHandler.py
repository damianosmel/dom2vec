import xml.sax
import os


class ProteinXMLHandler(xml.sax.ContentHandler):
	""""
	Class to read all proteins from match_complete.xml.gz from InterPro, with the aim to retrieve the protein length
	#Credits: https://www.tutorialspoint.com/python3/python_xml_processing.htm
	"""

	def __init__(self):
		"""
		ProteinXMLHandler class init

		Parameters
		----------

		Returns
		-------
		None
		"""
		self.CurrentData = ""
		self.id = ""
		self.seq_len = ""
		self.data_path = "/home/damian/Documents/L3S/projects"
		self.prot_len_file = "prot_id_len.tab"
		self.file_out = open(os.path.join(self.data_path, self.prot_len_file), 'a')
		self.file_out.write("uniprot_id\tseq_len\n")

	def startElement(self, tag, attributes):
		"""
		Process XML element when it starts

		Parameters
		----------
		tag : str
			XML element tag
		attributes : dict
			attributes of XML element

		Returns
		-------
		None
		"""
		self.CurrentData = tag
		if tag == "protein":
			self.id = attributes["id"]
			self.seq_len = attributes["length"]
			self.file_out.write(self.id + "\t" + self.seq_len + "\n")

	def endElement(self, tag):
		"""
		Process XML element when it ends

		Parameters
		----------
		tag : str
			XML element tag

		Returns
		-------
		None
		"""
		self.CurrentData = ""

	# Call when a character is read
	def characters(self, content):
		"""
		Process read characters

		Parameters
		----------
		content : str
			read characters

		Returns
		-------
		None
		"""
		self.currentData = ""

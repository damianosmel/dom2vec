import xml.etree.ElementTree as ET
import os
from collections import namedtuple

"""
Class to read all domains and save EC and SCOPe labels per interpro domain (if present)
"""


class DomainXMLParser:

	def __init__(self, data_path, interpro_xml, out_name):
		"""
		DomainXMLParser class init
		Parameters
		----------
		data_path : str
			data root path
		interpro_xml : str
			interpro xml file name
		out_name : str
			output file name

		Returns
		-------
		None
		"""
		self.data_path = data_path
		self.interpro_xml = interpro_xml
		self.out_name = out_name

	def create_id2ECSCOP(self):
		"""
		Create a named tuple to keep mapping from a interpro domain to ECs and SCOPs
		Parameters
		----------
		self : object
			DomainXMLParser object setup for this analysis

		Returns
		-------
		ID2EC_SCOP : namedtuple
			tuple mapping interpro domain to EC and SCOP ids
		"""
		ID2EC_SCOP = namedtuple('ID2EC_SCOP', 'id ecs scopes')
		ID2EC_SCOP.id = None
		ID2EC_SCOP.ecs = []  # potentially list of EC ids
		ID2EC_SCOP.scops = []  # potentrially list of SCOPe ids
		return ID2EC_SCOP

	def tab_id2ECSCOP(self, id2ECSCOP):
		"""
		Convert named tuple of id->EC,SCOP to tabular string

		Parameters
		----------
		id2ECSCOP : namedtuple
			tuple mapping interpro domain to EC and SCOP ids

		Returns
		-------
		str
			tabulated info for the domain id and its EC and SCOP ids
		"""
		ecs = " ".join(id2ECSCOP.ecs)
		scops = " ".join(id2ECSCOP.scops)
		return "\t".join([id2ECSCOP.id, ecs, scops])

	def parse2get_EC_SCOP(self):
		"""
		Parse interpro xml to get available EC and SCOP info for domains

		Parameters
		----------
		self : object
			DomainXMLParser object setup for this analysis

		Returns
		-------
		None
		"""
		tree = ET.parse(os.path.join(self.data_path, self.interpro_xml))
		root = tree.getroot()
		id2EC_SCOP_all = []
		for interpro_dom in root.findall("interpro"):
			id2EC_SCOP = self.create_id2ECSCOP()
			interpro_id = interpro_dom.attrib['id']
			id2EC_SCOP.id = interpro_id
			ext_doc_list = interpro_dom.find('external_doc_list')
			str_doc_list = interpro_dom.find('structure_db_links')
			if ext_doc_list is not None:
				for ext_doc in ext_doc_list:  # parse external document list for EC annotation
					if ext_doc.attrib['db'] == 'EC':
						id2EC_SCOP.ecs.append(ext_doc.attrib['dbkey'])
			if str_doc_list is not None:
				for str_doc in str_doc_list:  # parse external document list for SCOPe annotation
					if str_doc.attrib['db'] == 'SCOP':
						id2EC_SCOP.scops.append(str_doc.attrib['dbkey'])
			id2EC_SCOP_all.append(id2EC_SCOP)
		return id2EC_SCOP_all

	def save_EC_SCOP(self, ids2EC_SCOP):
		"""
		Save EC SCOP information to csv file

		Parameters
		----------
		self : object
			DomainXMLParser object setup for this analysis
		ids2EC_SCOP : namedtuple
			tuple containing domain id and its found EC and SCOP class

		Returns
		-------
		None
		"""
		with open(os.path.join(self.data_path, self.out_name), "w") as out_file:
			out_file.write("interpro_id\tECs\tSCOPs\n")
			for id2EC_SCOP in ids2EC_SCOP:
				out_file.write(self.tab_id2ECSCOP(id2EC_SCOP) + "\n")

	def parse_and_save_EC_SCOP(self):
		"""
		Parse interpro xml and save domains and their found EC and SCOP

		Parameters
		----------
		self : object
			DomainXMLParser object setup for this analysis

		Returns
		-------
		None
		"""
		ids2EC_SCOP = self.parse2get_EC_SCOP()
		self.save_EC_SCOP(ids2EC_SCOP)

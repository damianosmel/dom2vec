class Domain:
	"""
	We will represent a domain as:
	interpro id
	name
	evidence_db_id
	start_pos
	end_pos
	"""

	def __init__(self,hit_line):
		assert isinstance(hit_line,str), "AssertionError: Input of domain should a String line."
		self.interpro_id = ""
		self.name = ""
		self.evidence_db_id = ""
		self.start_pos = ""
		self.end_pos = ""
		self.length=0
		self.interpro_id_exists = 1
		self.line2domain(hit_line)

	def line2domain(self,hit_line):
		"""
		:param hit_line: line containing all the information of the hit
		 e.g.  A0A000\tIPR004839\tAminotransferase, class I/classII\tPF00155\t41\t381\n
		:return: none
		"""
		hit_tabs = hit_line.split("\t")
		#Choose between pre-calculated interpro tab format and local interpro format
		if len(hit_tabs) == 6:
			self.interpro_id = hit_tabs[1]
			self.name = hit_tabs[2]
			self.evidence_db_id = hit_tabs[3]
			self.start_pos = int(hit_tabs[4])
			self.end_pos = int(hit_tabs[5])

			assert self.end_pos >= self.start_pos, "AssertionError: " + self.interpro_id + "has end position before start position."
			self.length = self.end_pos - self.start_pos + 1
		elif len(hit_tabs) >= 11:
			# get the interpro annotation of protein line based on:
			# https://github.com/ebi-pf-team/interproscan/wiki/OutputFormats

			# Check if intepro id exists
			if len(hit_tabs) >= 12:  # Get existing interpro id
				#print("Exists intepro id!")
				self.interpro_id = hit_tabs[11]
				self.name = hit_tabs[12]
			else:  # create unk id: uniprot_sign_accession
				uniprot = hit_tabs[0]
				sign_accession = hit_tabs[4]
				#self.interpro_id = uniprot + "_" + sign_accession
				self.interpro_id = sign_accession
				self.name = hit_tabs[5]  # signature description
				self.interpro_id_exists = 0

			self.evidence_db_id = hit_tabs[4]
			self.start_pos = int(hit_tabs[6])
			self.end_pos = int(hit_tabs[7])

			assert self.end_pos >= self.start_pos, "AssertionError: " + self.interpro_id + "has end position before start position."
		else:
			print("Unknown format line")
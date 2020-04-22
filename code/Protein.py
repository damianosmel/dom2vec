from Domain import Domain
from intervaltree import IntervalTree
import os
class Protein:


	"""
	We will represent a protein as its domains
	3 ways:
	1) protein can have overlapping domains
	2) protein has only no-overlapping domains
	3) protein has known length so gap domain can be added
	"""
	def __init__(self,with_overlap,with_redundant,with_gap,hit_line="",proteins_id_len="",interpro_local_format=False):
		self.with_overlap = with_overlap
		self.with_redundant = with_redundant
		self.with_gap = with_gap
		self.domain_interval_tree = IntervalTree()
		self.domains_with_gaps = []
		self.gap_min_size = 30
		self.length = 0
		self.interpro_exist_all_domains = []
		if hit_line != "":
			if interpro_local_format:#interpro local run format
				#get the interpro annotation of protein line based on:
				#https://github.com/ebi-pf-team/interproscan/wiki/OutputFormats
				assert len(hit_line.split("\t")) >= 11, "AssertionError: line: {} has less than 11 tabs.".format(hit_line)
				self.uniprot_id = self.get_uniprot_id(hit_line)
				self.domains = {}
				self.add_domain(hit_line)
				if with_gap:
					self.length = int(hit_line.split("\t")[2])
					assert self.length > 0, "AssertionError: protein with id {} has length <=0.".format(self.length)
			else:#prot2ipr format
				assert isinstance(hit_line,str), "AssertionError: Input of protein should be a String line."
				hit_line = hit_line.strip()
				self.uniprot_id = self.get_uniprot_id(hit_line)
				self.domains = {}
				self.add_domain(hit_line)
				if with_gap:
					self.length = self.get_prot_length(proteins_id_len)
					assert self.length > 0, "AssertionError: protein with id {} has length <= 0.".format(self.length)
		else:
			self.uniprot_id = ""

	def get_prot_length(self,proteins_id_len):
		prot_len = -1
		prot_found = False
		try:
			while prot_found == False:
				prot_id_len = next(proteins_id_len)
				#print("current len:{}".format(prot_id_len))
				if prot_id_len.strip().split("\t")[0] == self.uniprot_id:
					prot_len = int(prot_id_len.strip().split("\t")[1])
					prot_found = True

		except(StopIteration):
			print("EOF")
		return prot_len

	@staticmethod
	def get_prot_id(hit_line):
		return hit_line.split("\t")[0]

	def get_uniprot_id(self,hit_line):
		return hit_line.split("\t")[0]

	def add_domain(self,hit_line):
		if self.with_overlap:
			self.add_overlap(hit_line)
		elif self.with_overlap is False or self.with_redundant:
			self.add_no_overlap(hit_line)

	def add_overlap(self, hit_line):
		domain = Domain(hit_line)
		self.interpro_exist_all_domains.append(domain.interpro_id_exists)
		if domain.end_pos > domain.start_pos:

			#construct start_stop index
			start_stop = str(domain.start_pos) + str(domain.end_pos)
			start_stop = float(start_stop)
			if start_stop not in self.domains:
				self.domains[start_stop] = domain
			else:
				#allow for 100 domain annotations to have the same start and end
				start_stop = start_stop + 0.01
				self.domains[start_stop] = domain

	def add_no_overlap(self,hit_line):
		domain = Domain(hit_line)
		self.interpro_exist_all_domains.append(domain.interpro_id_exists)
		if domain.end_pos > domain.start_pos:
			self.domain_interval_tree.addi(domain.start_pos,domain.end_pos,domain)

	def to_tabs(self):
		if self.with_redundant == False:
			return self.to_tabs_no_redundant()
		elif self.with_overlap:
			return self.to_tabs_overlap()
		else:
			return self.to_tabs_no_overlap()

	def to_tabs(self):
		if self.with_overlap:
			#print("Overlap")
			return self.to_tabs_overlap()
		elif self.with_redundant is False:
			#print("No overlap")
			return self.to_tabs_no_overlap()
		else:
			#print("No redundant")
			return self.to_tabs_no_redundant()

	def find_strong_no_overlap_domains(self,parent_domain,already_resolved):
		"""
		Find all no strong overlap domains with maximum length
		1) Resolve overlapping domains that overlap for less than 0.99% of their length
		to no strong overlap domains
		No strong overlap: |-----"--|-----"
		Strong overlap: |----"--"--|
		2) Find enveloppe domains
		3) From the rest of the domains, find the one with maximum length
		:param parent_domain: anchor domain to start overlaping search
		:param candidate_overlap_domains: list of overlapping domains
		:return strong_overlap_domains, no_strong_overlap_domains: lists of strong overlapping domains (resolved) and no strong overlapping (not (yet) resolved)
		"""
		envelopped_domains = self.domain_interval_tree.envelop(parent_domain.begin,parent_domain.end)
		overlapping_domains = self.domain_interval_tree.overlap(parent_domain.begin,parent_domain.end)
		candidate_domains = overlapping_domains - envelopped_domains - already_resolved

		strong_overlap_domains = set()
		no_strong_overlap_domains = set()
		for candidate_domain in list(candidate_domains):
			# As parent has the maximum length, there are two choices:
			#1) candidate domain is strongly overlapping with the parent => add it to strong_overlap_domains (resolved)
			#2) candidate domain is no strongly overlapping so => add it to no_strong_overlap_domains (not_resolved)
			candidate_domain_len = candidate_domain.end - candidate_domain.begin + 1
			if candidate_domain.begin >= parent_domain.begin:
				# |---parent---|
				#          |---child---|
				overlap_len = parent_domain.end - candidate_domain.begin + 1
			else:
				#    |---parent---|
				# |---child---|
				overlap_len = candidate_domain.end - parent_domain.begin + 1

			if float(overlap_len)/candidate_domain_len >= 0.8:#Strong overlap
				#print("candidate id: {}".format(candidate_domain.data.evidence_db_id))
				strong_overlap_domains.add(candidate_domain)
				assert candidate_domain.data.length <= parent_domain.data.length, "AssertionError: prot:{} candidate domain {} is longer than parent domain {}".format(
					self.uniprot_id,candidate_domain.data.evidence_db_id, parent_domain.data.evidence_db_id)
			else:#No strong overlap
				if candidate_domain.data.interpro_id == parent_domain.data.interpro_id:#if no strong overlap but the same interpro id take the longest one
					assert candidate_domain.data.length <= parent_domain.data.length, "AssertionError: prot:{} candidate domain {} is longer than parent domain {}".format(
						self.uniprot_id, candidate_domain.data.evidence_db_id, parent_domain.data.evidence_db_id)
					strong_overlap_domains.add(candidate_domain)
				else:
					no_strong_overlap_domains.add(candidate_domain)

		strong_overlap_domains.update(envelopped_domains) #add envelopped domains to strong_overlap domains
		return strong_overlap_domains,no_strong_overlap_domains

	def find_no_redundant_domains(self,parent_domain,already_resolved):
		overlapping_domains = self.domain_interval_tree.overlap(parent_domain.begin, parent_domain.end)
		candidate_domains = overlapping_domains - already_resolved
		redundant_domains = set()
		no_redundant_domains = set()

		for candidate_domain in list(candidate_domains):

			# As parent has the maximum length, there are two choices:
			# 1) candidate domain has the same interpro id => add it to redundant (resolved)
			# 2) candidate domain has not the same interpro id => add it to no redundant (not_resolved)
			if candidate_domain.data.interpro_id == parent_domain.data.interpro_id:
				#print("cand id: {} redundant".format(candidate_domain.data.evidence_db_id))
				redundant_domains.add(candidate_domain)
			else:
				#print("cand id: {} no redundant".format(candidate_domain.data.evidence_db_id))
				no_redundant_domains.add(candidate_domain)

		return redundant_domains,no_redundant_domains

	def find_no_redundant_max_len(self):
		"""
		Find all domains that are not redundant (having unique interpro id) and are maximally long
		:return: list[Interval] list of IntervalTree nodes as the no redundant maximum length domains
		"""
		resolved = set()
		domains_no_redundant_max = []

		domains_len_srt = [domain for domain in self.domain_interval_tree]
		domains_len_srt.sort(key=lambda dom_node:dom_node.data.length,reverse=True)

		for domain_node in domains_len_srt:
			if domain_node not in resolved:
				#print("---")
				#print("parent id:{}".format(domain_node.data.evidence_db_id))
				redundant_domains, no_redundant_domains = self.find_no_redundant_domains(domain_node,resolved)
				#domains_no_redundant_max.append(domain_node.data)
				domains_no_redundant_max.append(domain_node)
				resolved.update(redundant_domains)
		return domains_no_redundant_max

	def find_no_overlap_max_len(self):
		"""
		Find all domains that are not overlapping and are maximally long
		:return: list of not overlapping maximum length domains
		"""
		resolved = set()
		domains_no_overlap_max = []

		domains_len_srt = [domain for domain in self.domain_interval_tree]
		domains_len_srt.sort(key=lambda dom_node:dom_node.data.length,reverse=True)

		"""
		Idea: After sorting the domains by length in descending order, then
		pick each domain and check for 
		envelopped domains -> resolved
		strong overlap domains -> resolved
		no strong overlap domains -> not resolved, the for loop will either add it as max no overlap or as resolved
		"""
		for domain_node in domains_len_srt:
			if domain_node not in resolved:
				#print("--- ---")
				#print("parent id:{}".format(domain_node.data.evidence_db_id))
				strong_overlap_domains,strong_no_overlap_domains = self.find_strong_no_overlap_domains(domain_node,resolved)
				domains_no_overlap_max.append(domain_node.data)

				resolved.update(strong_overlap_domains)
				#print("resolving: ")
				#for str_overlap_dom in strong_overlap_domains:
				#	print("resolved prot id:{}".format(str_overlap_dom.data.evidence_db_id))
		return domains_no_overlap_max

	def construct_gap_hitline(self,gap_start,gap_stop):
		return "\t".join([self.uniprot_id,"GAP","gap","gap_no_evid",str(gap_start),str(gap_stop)])

	def add_gaps_no_redundant(self,domains_srt):
		"""
		Update the self.domains_with_gaps
		:param domains_srt:

		"""
		start_gap = 1
		previous_domain = None #interval tree node
		is_first_domain = True
		for domain_interval in domains_srt:
			if is_first_domain:#first domain
				if domain_interval.begin - start_gap + 1 > self.gap_min_size:#add start GAP
					#print("start GAP")
					assert domain_interval.begin > 1, "AssertionError: Start gap can be added if the very first domain is not starting at 1."
					self.domains_with_gaps.append(Domain(self.construct_gap_hitline(start_gap,domain_interval.begin-1)))
					start_gap = domain_interval.end + 1
				is_first_domain = False
			else:
				#check if the current domain and the previous are overlaping if yes then you can't add a gap
				#if no check the space between them
				overlap_domains = self.domain_interval_tree.overlap(domain_interval.begin, domain_interval.end)
				no_redundant_overlap_domains = overlap_domains.intersection(set(domains_srt))

				if previous_domain not in no_redundant_overlap_domains:#not overlapping domains => check for space to add a GAP
					if domain_interval.begin - start_gap + 1 > self.gap_min_size:#add middle GAP
						#print("middle GAP")
						self.domains_with_gaps.append(Domain(self.construct_gap_hitline(start_gap,domain_interval.begin-1)))
			#adding gap or no append current domain interval and update start_gap
			self.domains_with_gaps.append(domain_interval.data)
			start_gap = domain_interval.end + 1
			previous_domain = domain_interval

		#To check for end GAP, you should get the maximum end_pos of non redundant domain
		max_end_pos = max([dom.end for dom in domains_srt])
		max_end_pos = max_end_pos + 1
		if self.length - max_end_pos + 1 > self.gap_min_size:
			#print("end GAP")
			self.domains_with_gaps.append(Domain(self.construct_gap_hitline(start_gap,self.length)))

	def add_gaps(self,domains_srt):
		start_gap = 1
		for domain in domains_srt:#check for GAP in the start and middle of the protein
			# |--- --- protein --- ---|
			#     |--dom1--| |--dom2--|
			# |GAP|
			if domain.start_pos - start_gap + 1 > self.gap_min_size:
				self.domains_with_gaps.append(Domain(self.construct_gap_hitline(start_gap,domain.start_pos)))
			start_gap = domain.end_pos + 1
			self.domains_with_gaps.append(domain)
		
		# check for gap in the end of the protein seq
		# |--- --- protein --- ---|
		# |--dom1--| |--dom2--|
		#                     |GAP|
		if self.length - domain.end_pos + 1 > self.gap_min_size:
			self.domains_with_gaps.append(Domain(self.construct_gap_hitline(domain.end_pos+1,self.length)))

	def to_tabs_no_redundant(self):
		#find no redundant domains with maximum length
		domains_no_redundant_max_len = self.find_no_redundant_max_len()
		#sort by start position
		domains_no_redundant_max_len.sort(key=lambda domain:domain.begin,reverse=False)
		if self.with_gap:
			self.add_gaps_no_redundant(domains_no_redundant_max_len)
			self.domains_with_gaps.sort(key=lambda domain:domain.start_pos,reverse=False) #sort by start position
			domains_no_redundant = " ".join([domain.interpro_id for domain in self.domains_with_gaps])
			domains_evidence_db_ids = " ".join([domain.evidence_db_id for domain in self.domains_with_gaps])
		else:
			domains_no_redundant = " ".join([domain.data.interpro_id for domain in domains_no_redundant_max_len])
			domains_evidence_db_ids = " ".join([domain.data.evidence_db_id for domain in domains_no_redundant_max_len])

		return self.uniprot_id+"\t"+domains_no_redundant+"\t"+domains_evidence_db_ids+"\n"

	def to_tabs_no_overlap(self):
		#find non overlaping domains with maximum length
		domains_no_overlap_max_len = self.find_no_overlap_max_len()
		#sort by start position
		domains_no_overlap_max_len.sort(key=lambda domain:domain.start_pos,reverse=False)
		if self.with_gap:
			self.add_gaps(domains_no_overlap_max_len)
			domains_no_overlap = " ".join([domain.interpro_id for domain in self.domains_with_gaps])
			domains_evidence_db_ids = " ".join([domain.evidence_db_id for domain in self.domains_with_gaps])
		else:
			domains_no_overlap = " ".join([domain.interpro_id for domain in domains_no_overlap_max_len])
			domains_evidence_db_ids = " ".join([domain.evidence_db_id for domain in domains_no_overlap_max_len])
		return self.uniprot_id+"\t"+domains_no_overlap+"\t"+domains_evidence_db_ids+"\n"

	def to_tabs_overlap(self):
		#for gaps you shall give a list out of the sorted dictionary sorted(self.domains)
		if self.with_gap:
			self.add_gaps([self.domains[start_stop] for start_stop in sorted(self.domains.keys())])
			domains_overlap = " ".join([domain.interpro_id for domain in self.domains_with_gaps])
			domains_evid_db_ids = " ".join([domain.evidence_db_id for domain in self.domains_with_gaps])
		else:
			domains_overlap = " ".join([self.domains[start_stop].interpro_id for start_stop in self.domains])
			domains_evid_db_ids = " ".join([self.domains[start_stop].evidence_db_id for start_stop in self.domains])
		return self.uniprot_id+"\t"+domains_overlap+"\t"+domains_evid_db_ids+"\n"
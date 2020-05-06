from timeit import default_timer as timer
from Preprocess import Preprocess
from Corpus import Corpus
from utils import sec2hour_min_sec

###                                ###
###          Deprecated            ###
### please see main instead        ###
###                                ###


"""
File to run all pre-processing steps
"""

# Server test
data_path = "/data2/melidis"  # "/home/damian/Documents/L3S/projects"#
with_overlap = False
with_gap = True
preprocess_protein2ipr = Preprocess(data_path, with_overlap, with_gap)

print("=====")
print("1) Parsing protein2ipr -> protein_id tab domains")

file_name = "protein2ipr.dat.gz"

# Tests: each one one protein
# A0A00 - PASS
# P64826 - PASS
# P77334 - PASS!
# A0A009E9Q4 - PASS (assertError should be <=)!

# file_name = "prot6.tab"#"prot_A0A009F0U6.tab" #"prot_A0A009E9Q4.tab"#"prot_A0A009HCC6.tab"#"proteins3.tab.gz"#"prot_P77334.tab"#"prot_A0A000.tab"#"prot_P64826.tab"
batch_num_lines = 1000000
batch_out_prot = 10000
# credits: https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
time_start = timer()
preprocess_protein2ipr.parse_prot2in(file_name, batch_num_lines, batch_out_prot)
time_end = timer()
print("Elapsed CPU time for parsing: {}.".format(sec2hour_min_sec(time_end - time_start)))

print("\n=====")
print("2) Parse id_domains.tab -> domains_corpus.txt")

file_in_name = "id_domains_no_overlap_gap.tab"  # "id_domains_overlap.tab"
file_out_name = "domains_corpus_no_overlap_gap.txt"  # "domains_corpus_overlap.txt"
batch_num_lines = 100000
time_start = timer()
preprocess_protein2ipr.create_domains_corpus(file_in_name, file_out_name, batch_num_lines)
time_end = timer()
print("Elapsed CPU time for creating corpus: {}.".format(sec2hour_min_sec(time_end - time_start)))

print("\n=====")
print("3) Plot corpus histogram")
file_in = "domains_corpus_no_overlap_gap.txt"  # "domains_corpus_example.txt"#
domains_corpus = Corpus(data_path, file_in)
domains_corpus.plot_line_histogram()

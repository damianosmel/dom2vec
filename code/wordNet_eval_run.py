from WordNetDomain import WordNetDomain
from WordTopicsEvaluate import WordTopicsEvaluate
print("Word embeddings evaluation")

###                 ###
### WordNet domains ###
###                 ###
print("=== WordNet domains k-NN--> accuracy ===")

###                                 ###
### Preprocess and create data set  ###
###                                 ###
# print("0. Create data set")
# domains_file_path = "/home/damian/Desktop/domains/wordnet/wn-domains-3.2/wn-domains-3.2-20070223"
# out_path = "/home/damian/Desktop/domains/wordnet/wn-domains-3.2/dataset"
# out_name = "word_topics.tab"
# is_wordNet_installed = True #False
# emb_path = "/home/damian/Documents/L3S/projects/nlp_emb/google_news_2013/model.bin"
# is_emb_bin = True
# wordNetDomain = WordNetDomain(domains_file_path,out_path,out_name,emb_path,is_emb_bin,is_wordNet_installed)
# wordNetDomain.create_dataset()

###                                ###
### Evaluate word emb using topics ###
###                                ###
data_path = "/home/damian/Desktop/domains/wordnet/wn-domains-3.2/dataset"
word_topics_file = "word_topics.tab"
model_path = "/home/damian/Documents/L3S/projects/nlp_emb/google_news_2013/model.bin"
is_emb_bin = True

print("1. word topics -> k-NN")
google_emb2013_eval = WordTopicsEvaluate(data_path, word_topics_file)
use_basic_label = True
use_rand_vec = True
google_emb2013_eval.run(model_path, is_emb_bin, use_basic_label, use_rand_vec, classifier_name="NN")
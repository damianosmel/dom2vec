#!/bin/bash

source /etc/profile

echo "=== Overlap ==="
echo "CBOW"
echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg0_vec50_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg0_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg0_vec100_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg0_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg0_vec200_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg0_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg0_vec50_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg0_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg0_vec100_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg0_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg0_vec200_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg0_vec200_max_ep50.error

echo "Skip-gram"
echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg1_vec50_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg1_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg1_vec100_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg1_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w2_sg1_vec200_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w2_sg1_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg1_vec50_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg1_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg1_vec100_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg1_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/overlap \
--file_in domains_corpus_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/overlap/overlap_gap_w5_sg1_vec200_max_ep50.out 2> /data2/melidis/overlap/overlap_gap_w5_sg1_vec200_max_ep50.error

echo "=== No Overlap ==="
echo "CBOW"
echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec50_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec100_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec200_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg0_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec50_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec100_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec200_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg0_vec200_max_ep50.error

echo "Skip-gram"
echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec50_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec100_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec200_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w2_sg1_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec50_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec100_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_overlap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec200_max_ep50.out 2> /data2/melidis/no_overlap/no_overlap_gap_w5_sg1_vec200_max_ep50.error

echo "=== No redundant ==="
echo "CBOW"

echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec50_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec100_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 2 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec200_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg0_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec50_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec100_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_redundant_gap.txt \
--window 5 \
--use_skipgram 0 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec200_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg0_vec200_max_ep50.error

echo "Skip-gram"
echo "window 2, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec50_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec50_max_ep50.error

echo "window 2, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec100_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec100_max_ep50.error

echo "window 2, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 2 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec200_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w2_sg1_vec200_max_ep50.error

echo "window 5, dim 50"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 50 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec50_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec50_max_ep50.error

echo "window 5, dim 100"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 100 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec100_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec100_max_ep50.error

echo "window 5, dim 200"
python /home/melidis/dom2vec/code/word2vec_run.py \
--data_path /data2/melidis/no_red_gap \
--file_in domains_corpus_no_overlap_gap.txt \
--window 5 \
--use_skipgram 1 \
--vec_dim 200 \
--cores 8 \
--max_epoches 50 \
--epoches_step 5 > /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec200_max_ep50.out 2> /data2/melidis/no_red_gap/no_red_gap_w5_sg1_vec200_max_ep50.error

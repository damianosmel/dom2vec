#!/bin/bash
source /etc/profile

###	    	     ###
### dom2vec (random) ###
###	    	     ###
echo "Toxin dom2vec random"
echo "=== SeqVecNet hid_dim=1024, drop 0.25, lr=0.0003 - dom2vec update ==="

echo "10%"
for i in {0..19}; do
echo "Running train_subset_p0.1_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file train_subset_p0.1_$i.csv \
--test_file toxin_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 50 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 1024 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/dom2vec_rand/p10/sample_$i.out 2>/home/melidis/toxin/dom2vec_rand/p10/sample_$i.error
done

echo "20%"
for i in {0..19}; do
echo "Running train_subset_p0.2_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file train_subset_p0.2_$i.csv \
--test_file toxin_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 50 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 1024 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/dom2vec_rand/p20/sample_$i.out 2>/home/melidis/toxin/dom2vec_rand/p20/sample_$i.error
done

echo "50%"
for i in {0..19}; do
echo "Running train_subset_p0.5_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file train_subset_p0.5_$i.csv \
--test_file toxin_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 50 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 1024 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/dom2vec_rand/p50/sample_$i.out 2>/home/melidis/toxin/dom2vec_rand/p50/sample_$i.error
done

echo "100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file toxin_dataset_train.csv \
--test_file toxin_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 50 \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--normalize_emb 0 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 1024 \
--dropout 0.25 \
--save_predictions 1 >/home/melidis/toxin/dom2vec_rand/p100/sample_0.out 2>/home/melidis/toxin/dom2vec_rand/p100/sample_0.error

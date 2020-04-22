#!/bin/bash
source /etc/profile


echo "NEW - dom2vec update"
echo "CNN 1x200, dropout 0.5,"
#used emb:  /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt

echo "10%"
for i in {0..9}; do
echo "Running train_subset_p0.1_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file train_subset_p0.1_$i.csv \
--test_file new_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1 \
--dropout 0.5 \
--save_predictions 0 >/home/melidis/new/dom2vec/p10/sample_$i.out 2>/home/melidis/new/dom2vec/p10/sample_$i.error
done

echo "20%"
for i in {0..9}; do
echo "Running train_subset_p0.2_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file train_subset_p0.2_$i.csv \
--test_file new_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1 \
--dropout 0.5 \
--save_predictions 0 >/home/melidis/new/dom2vec/p20/sample_$i.out 2>/home/melidis/new/dom2vec/p20/sample_$i.error
done

echo "50%"
for i in {0..9}; do
echo "Running train_subset_p0.5_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file train_subset_p0.5_$i.csv \
--test_file new_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1 \
--dropout 0.5 \
--save_predictions 0 >/home/melidis/new/dom2vec/p50/sample_$i.out 2>/home/melidis/new/dom2vec/p50/sample_$i.error
done

echo "100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file new_dataset_train.csv \
--test_file new_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1 \
--dropout 0.5 \
--save_predictions 1 >/home/melidis/new/dom2vec/p100/sample_0.out 2>/home/melidis/new/dom2vec/p100/sample_0.error


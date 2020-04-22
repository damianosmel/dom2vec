#!/bin/bash
source /etc/profile


echo "NEW - dom2vec (random) update"
echo "CNN 1x200, dropout 0.5"
#used emb:  dom2vec random vectors of dimension 200

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
--use_emb 0 \
--emb_file 200 \
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
--save_predictions 0 >/home/melidis/new/dom2vec_rand/p10/sample_$i.out 2>/home/melidis/new/dom2vec_rand/p10/sample_$i.error
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
--use_emb 0 \
--emb_file 200 \
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
--save_predictions 0 >/home/melidis/new/dom2vec_rand/p20/sample_$i.out 2>/home/melidis/new/dom2vec_rand/p20/sample_$i.error
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
--use_emb 0 \
--emb_file 200 \
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
--save_predictions 0 >/home/melidis/new/dom2vec_rand/p50/sample_$i.out 2>/home/melidis/new/dom2vec_rand/p50/sample_$i.error
done

echo "100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file new_dataset_train.csv \
--test_file new_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
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
--save_predictions 1 >/home/melidis/new/dom2vec_rand/p100/sample_0.out 2>/home/melidis/new/dom2vec_rand/p100/sample_0.error


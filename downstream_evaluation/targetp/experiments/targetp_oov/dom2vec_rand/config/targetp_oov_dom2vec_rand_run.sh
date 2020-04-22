#!/bin/bash
source /etc/profile

###	                      ###
### dom2vec (random) - update ###
###	                      ###
echo "TargetP dom2vec OOV split"
echo "=== CNN 200x1_2 >FC(128x128) >FC(128xPrediction), drop 0.7, lr=0.0005 - dom2vec update ==="
#used emb: random dom2vec vectors of size 200

echo "OOV 0%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/oov/dom2vec_rand/p0/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p0/sample_0.error

echo "OOV 10%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.1.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/oov/dom2vec_rand/p10/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p10/sample_0.error

echo "OOV 30%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.3.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/oov/dom2vec_rand/p30/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p30/sample_0.error

echo "OOV 50%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.5.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/oov/dom2vec_rand/p50/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p50/sample_0.error

echo "OOV 70%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.7.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/oov/dom2vec_rand/p70/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p70/sample_0.error

echo "OOV 100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 0 \
--emb_file 200 \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 1 >/home/melidis/targetp/oov/dom2vec_rand/p100/sample_0.out 2>/home/melidis/targetp/oov/dom2vec_rand/p100/sample_0.error

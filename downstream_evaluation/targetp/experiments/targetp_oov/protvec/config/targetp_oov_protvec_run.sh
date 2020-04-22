#!/bin/bash
source /etc/profile

###	    ###
### ProtVec ###
###	    ###
echo "TargetP protvec OOV split"
echo "=== SeqVecNet drop 0.25, lr=0.0003 - protvec update ==="

echo "OOV 0%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--normalize_emb 0 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/oov/protvec/p0/sample_0.out 2>/home/melidis/targetp/oov/protvec/p0/sample_0.error

echo "OOV 10%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.1.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/oov/protvec/p10/sample_0.out 2>/home/melidis/targetp/oov/protvec/p10/sample_0.error

echo "OOV 30%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.3.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/oov/protvec/p30/sample_0.out 2>/home/melidis/targetp/oov/protvec/p30/sample_0.error

echo "OOV 50%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.5.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/oov/protvec/p50/sample_0.out 2>/home/melidis/targetp/oov/protvec/p50/sample_0.error

echo "OOV 70%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.7.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/oov/protvec/p70/sample_0.out 2>/home/melidis/targetp/oov/protvec/p70/sample_0.error

echo "OOV 100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 1 >/home/melidis/targetp/oov/protvec/p100/sample_0.out 2>/home/melidis/targetp/oov/protvec/p100/sample_0.error

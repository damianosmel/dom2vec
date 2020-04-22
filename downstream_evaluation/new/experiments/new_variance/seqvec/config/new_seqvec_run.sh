#!/bin/bash
source /etc/profile

###	   ###
### SeqVec ###
###	   ###
### seqvec averaging to 1024 vector per protein  ###
echo "NEW Seqvec"
echo "=== SeqVecNet drop 0.25, lr=0.0003 - seqvec update ==="
#echo "1%"
#for i in {0..19}; do
#echo "Running train_subset_p0.01_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file train_subset_p0.01_$i.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/new/seqvec/p1/sample_$i.out 2>/home/melidis/new/seqvec/p1/sample_$i.error
#done

#echo "2%"
#for i in {0..19}; do
#echo "Running train_subset_p0.02_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file train_subset_p0.02_$i.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/new/seqvec/p2/sample_$i.out 2>/home/melidis/new/seqvec/p2/sample_$i.error
#done

#echo "5%"
#for i in {0..19}; do
#echo "Running train_subset_p0.05_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file train_subset_p0.05_$i.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/new/seqvec/p5/sample_$i.out 2>/home/melidis/new/seqvec/p5/sample_$i.error
#done

#echo "10%"
#for i in {0..19}; do
#echo "Running train_subset_p0.1_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file train_subset_p0.1_$i.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/new/seqvec/p10/sample_$i.out 2>/home/melidis/new/seqvec/p10/sample_$i.error
#done

echo "20%"
for i in {1..10}; do
echo "Running train_subset_p0.2_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name NEW \
--data_path /home/melidis/datasets/new/ec \
--label_name ec \
--train_file train_subset_p0.2_$i.csv \
--test_file new_dataset_test.csv \
--emb_name seqvec \
--use_emb 1 \
--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
--emb_bin 0 \
--normalize_emb 0 \
--freeze_emb 0 \
--k_fold -1 \
--model_type SeqVecCharNet \
--batch_size 64 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/new/seqvec/p20/sample_$i.out 2>/home/melidis/new/seqvec/p20/sample_$i.error
done

#echo "50%"
#for i in {0..19}; do
#echo "Running train_subset_p0.5_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file train_subset_p0.5_$i.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/new/seqvec/p50/sample_$i.out 2>/home/melidis/new/seqvec/p50/sample_$i.error
#done

#echo "100%"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name NEW \
#--data_path /home/melidis/datasets/new/ec \
#--label_name ec \
#--train_file new_dataset_train.csv \
#--test_file new_dataset_test.csv \
#--emb_name seqvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/seqvec/uniref50_v2 \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecCharNet \
#--batch_size 64 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 1 >/home/melidis/new/seqvec/p100/sample_0.out 2>/home/melidis/new/seqvec/p100/sample_0.error

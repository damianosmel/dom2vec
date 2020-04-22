#!/bin/bash
source /etc/profile

echo "==== ==== ==== ==== ==== ==== === ="
echo "       TargetP 3-fold              "
echo "==== ==== ==== ==== ==== ==== === ="
#used emb: /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt

echo "==== ==== ==== ===="
echo "  dom2vec update   "
echo "==== ==== ==== ===="


echo "CNN 1x128 0.25"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type CNN \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--num_filters 128 \
--filter_sizes 1 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/cnn_128_1.out 2>/home/melidis/targetp/3fold/dom2vec/cnn_128_1.error

echo "CNN 1_2x200 0.7"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test_0.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type CNN \
--batch_size 2048 \
--learning_rate 0.0005 \
--epoches 300 \
--weight_decay 0 \
--num_filters 200 \
--filter_sizes 1_2 \
--dropout 0.7 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/cnn_200_1_2.out 2>/home/melidis/targetp/3fold/dom2vec/cnn_200_1_2.error

echo "LSTM hid_dim=512, num_layers=1, uni-directional"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type LSTM \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 512 \
--is_bidirectional 0 \
--num_layers 1 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/lstm_uni.out 2>/home/melidis/targetp/3fold/dom2vec/lstm_uni.error

echo "LSTM hid_dim=512, num_layers=1, bi-directional"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type LSTM \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 512 \
--is_bidirectional 1 \
--num_layers 1 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/lstm_bi.out 2>/home/melidis/targetp/3fold/dom2vec/lstm_bi.error

echo "FastText uni-gram"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type FastText \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--use_uni_bi 0 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/fastText_uni.out 2>/home/melidis/targetp/3fold/dom2vec/fastText_uni.error

echo "SeqVecNet 32"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/seqvecnet_32.out 2>/home/melidis/targetp/3fold/dom2vec/seqvecnet_32.error

echo "SeqVecNet 128"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 128 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/seqvecnet_128.out 2>/home/melidis/targetp/3fold/dom2vec/seqvecnet_128.error

echo "SeqVecNet 512"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name TargetP \
--data_path /home/melidis/datasets/targetp/cellular_location \
--label_name cellular_location \
--train_file targetp_dataset_train.csv \
--test_file targetp_dataset_test.csv \
--emb_name dom2vec \
--use_emb 1 \
--emb_file /home/melidis/emb/no_red_gap/dom2vec_w2_sg1_hierSoft0_dim200_e50_norm.txt \
--emb_bin 0 \
--freeze_emb 0 \
--normalize_emb 0 \
--k_fold 3 \
--model_type SeqVecNet \
--batch_size 512 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 512 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/targetp/3fold/dom2vec/seqvecnet_512.out 2>/home/melidis/targetp/3fold/dom2vec/seqvecnet_512.error

#!/bin/bash

source /etc/profile

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/melidis/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/melidis/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/melidis/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/melidis/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate gensim

###	    ###
### ProtVec ###
###	    ###
echo "Toxin protvec"
echo "=== SeqVecNet drop 0.25, lr=0.0003 - protvec update ==="

#echo "1%"
#for i in {0..19}; do
#echo "Running train_subset_p0.01_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name Toxin \
#--data_path /home/melidis/datasets/toxin \
#--label_name toxin \
#--train_file train_subset_p0.01_$i.csv \
#--test_file toxin_dataset_test.csv \
#--emb_name protvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecNet \
#--batch_size 32 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/toxin/protvec/p1/sample_$i.out 2>/home/melidis/toxin/protvec/p1/sample_$i.error
#done

#echo "2%"
#for i in {0..19}; do
#echo "Running train_subset_p0.02_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name Toxin \
#--data_path /home/melidis/datasets/toxin \
#--label_name toxin \
#--train_file train_subset_p0.02_$i.csv \
#--test_file toxin_dataset_test.csv \
#--emb_name protvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecNet \
#--batch_size 32 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/toxin/protvec/p2/sample_$i.out 2>/home/melidis/toxin/protvec/p2/sample_$i.error
#done

#echo "5%"
#for i in {0..19}; do
#echo "Running train_subset_p0.05_$i.csv"
#python3 /home/melidis/dom2vec/code/neurNet_run.py \
#--dataset_name Toxin \
#--data_path /home/melidis/datasets/toxin \
#--label_name toxin \
#--train_file train_subset_p0.05_$i.csv \
#--test_file toxin_dataset_test.csv \
#--emb_name protvec \
#--use_emb 1 \
#--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
#--emb_bin 0 \
#--freeze_emb 0 \
#--k_fold -1 \
#--model_type SeqVecNet \
#--batch_size 32 \
#--learning_rate 0.0003 \
#--epoches 300 \
#--weight_decay 0 \
#--hid_dim 32 \
#--dropout 0.25 \
#--save_predictions 0 >/home/melidis/toxin/protvec/p5/sample_$i.out 2>/home/melidis/toxin/protvec/p5/sample_$i.error
#done

echo "10%"
for i in {0..19}; do
echo "Running train_subset_p0.1_$i.csv"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file train_subset_p0.1_$i.csv \
--test_file toxin_dataset_test.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/protvec/p10/sample_$i.out 2>/home/melidis/toxin/protvec/p10/sample_$i.error
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
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/protvec/p20/sample_$i.out 2>/home/melidis/toxin/protvec/p20/sample_$i.error
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
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 0 >/home/melidis/toxin/protvec/p50/sample_$i.out 2>/home/melidis/toxin/protvec/p50/sample_$i.error
done

echo "100%"
python3 /home/melidis/dom2vec/code/neurNet_run.py \
--dataset_name Toxin \
--data_path /home/melidis/datasets/toxin \
--label_name toxin \
--train_file toxin_dataset_train.csv \
--test_file toxin_dataset_test.csv \
--emb_name protvec \
--use_emb 1 \
--emb_file /home/melidis/emb/protvec/protvec_100d_3grams.txt \
--emb_bin 0 \
--freeze_emb 0 \
--k_fold -1 \
--model_type SeqVecNet \
--batch_size 32 \
--learning_rate 0.0003 \
--epoches 300 \
--weight_decay 0 \
--hid_dim 32 \
--dropout 0.25 \
--save_predictions 1 >/home/melidis/toxin/protvec/p100/sample_0.out 2>/home/melidis/toxin/protvec/p100/sample_0.error


conda deactivate

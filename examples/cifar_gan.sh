#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate global_workspace

ad_heads=$1
topk=$2
ad_key=$3
ad_value=$4
add_name='POSWITHCONCAT_CIFAR_NFL_'$ad_heads'_topk_'$topk'_key_'$ad_key'_val_'$ad_value

python /home/anirudh/nips2020/mimicry/examples/infomax_example.py --n_heads $ad_heads --topk $topk --key_size $ad_key --val_size $ad_value --name $add_name

#Normal NFL, 1, 3, 32, 32
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0908 sec/batch)INFO: FID Score: 12.137045100044816 [Time Taken: 338.2048 secs]
#INFO: FID (step 100000) [seed 0]: 12.137045100044816
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0962 sec/batch)INFO: FID Score: 12.174696541604249 [Time Taken: 326.0470 secs]
#INFO: FID (step 100000) [seed 1]: 12.174696541604249
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0927 sec/batch)INFO: FID Score: 12.725213195003505 [Time Taken: 328.3336 secs]
#INFO: FID (step 100000) [seed 2]: 12.725213195003505
#INFO: FID (step 100000): 12.345651612217523 (± 0.26883037292206613)
#INFO: FID Evaluation completed!

#anirudh@dc1-wks-07:~/nips2020/mimicry/examples$ borgy logs 377b9f27-0f8a-4673-98de-80d1a72f63c4 | grep 'Inception'
#Downloading Inception model
#INFO: Computing Inception Score in memory...
##INFO: Inception Score (step 100000) [seed 0]: 8.461987495422363
#INFO: Computing Inception Score in memory...
#INFO: Inception Score (step 100000) [seed 1]: 8.433279037475586
#INFO: Computing Inception Score in memory...
#INFO: Inception Score (step 100000) [seed 2]: 8.477009773254395
#INFO: Inception Score (step 100000): 8.457425435384115 (± 0.01814209849647006)
#INFO: Inception Score Evaluation completed!

###############################################################3

#Normal, NFL, 4, 4, 32, 32


#anirudh@dc1-wks-07:~/nips2020/mimicry/examples$ borgy logs 7ed5f518-1bf1-4899-a4be-c54671d62b5d | grep 'FID'
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0924 sec/batch)INFO: FID Score: 14.004033859557808 [Time Taken: 343.6434 secs]
#INFO: FID (step 100000) [seed 0]: 14.004033859557808
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0933 sec/batch)INFO: FID Score: 13.899393654029325 [Time Taken: 335.3315 secs]
#INFO: FID (step 100000) [seed 1]: 13.899393654029325
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0889 sec/batch)INFO: FID Score: 13.862713973199732 [Time Taken: 334.8318 secs]
#INFO: FID (step 100000) [seed 2]: 13.862713973199732
#INFO: FID (step 100000): 13.922047162262288 (± 0.059876058913451)
#INFO: FID Evaluation completed!
#anirudh@dc1-wks-07:~/nips2020/mimicry/examples$ borgy logs 7ed5f518-1bf1-4899-a4be-c54671d62b5d | grep 'Inception'
#Downloading Inception model
#INFO: Computing Inception Score in memory...
#INFO: Inception Score (step 100000) [seed 0]: 8.420686721801758
#INFO: Computing Inception Score in memory...
#INFO: Inception Score (step 100000) [seed 1]: 8.410181045532227
#INFO: Computing Inception Score in memory...
#INFO: Inception Score (step 100000) [seed 2]: 8.45498275756836
#INFO: Inception Score (step 100000): 8.428616841634115 (± 0.01913048963929562)
#INFO: Inception Score Evaluation completed!

###############################################################3

#Normal, NFL, 2, 3, 32, 32

#Normal NFL
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0859 sec/batch)INFO: FID Score: 13.320379647181596 [Time Taken: 329.4758 secs]
#INFO: FID (step 100000) [seed 0]: 13.320379647181596
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0866 sec/batch)INFO: FID Score: 13.238847502112776 [Time Taken: 317.4488 secs]
#INFO: FID (step 100000) [seed 1]: 13.238847502112776
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0858 sec/batch)INFO: FID Score: 13.527627704035012 [Time Taken: 317.0002 secs]
#INFO: FID (step 100000) [seed 2]: 13.527627704035012
#INFO: FID (step 100000): 13.362284951109794 (± 0.12156079995499623)
#INFO: FID Evaluation completed!
#==========================


#Extra capacity baseline
#anirudh@dc1-wks-07:~$ borgy logs 887468b7-b7be-499a-99e6-245b0f3bb40f | grep 'FID'
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0926 sec/batch)INFO: FID Score: 14.68983806759178 [Time Taken: 352.7942 secs]
#INFO: FID (step 100000) [seed 0]: 14.68983806759178
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.0917 sec/batch)INFO: FID Score: 14.676007390649374 [Time Taken: 342.0534 secs]
#INFO: FID (step 100000) [seed 1]: 14.676007390649374
#INFO: Computing FID in memory...
#INFO: Propagated batch 1000/1000 (0.1157 sec/batch)INFO: FID Score: 14.751740487988002 [Time Taken: 339.2834 secs]
#INFO: FID (step 100000) [seed 2]: 14.751740487988002
#INFO: FID (step 100000): 14.705861982076385 (± 0.03292870970934004)
#INFO: FID Evaluation completed!

###############################################################

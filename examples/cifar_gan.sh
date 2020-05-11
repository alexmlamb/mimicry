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

#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate global_workspace

ad_heads=$1
topk=$2
ad_key=$3
ad_value=$4
add_name='CIFAR_NFL_'$ad_heads'_topk_'$topk'_key_'$ad_key'_val_'$ad_value

python /home/anirudh/nips2020/mimicry/examples/infomax_example.py --n_heads $ad_heads --topk $topk --key_size $ad_key --val_size $ad_value --name $add_name


#dim1=$1
#em=$1
#block1=$2
#topk1=$3
#templates=0
#drop=0.2
#log=100
#train_len=50
#test_len=100
#memory_slots=$4
#num_memory_heads=$5
#memory_head_size=$6
#memory_mlp=$7
#lr=$8
#600 6 4 8 1 16 4 0.001
#name="/home/anirudh/nips2020/soft_mechanisms/copying_logs/Blocks_memory_"$dim1"_"$em"_"$block1"_"$topk1"_"$templates"_FALSE_"$drop"_"$lr"_"$log"_"$train_len"_"$test_len"_mslots_"$memory_slots"_mheads_"$num_memory_heads"_mheadsize_"$memory_head_size"_memory_mlp_"$memory_mlp
#name="${name//./}"
#echo Running version $name
#python /home/anirudh/nips2020/soft_mechanisms/train_copying.py --cuda  --cudnn --n_templates $templates  --algo blocks --do_rel --memory_mlp $memory_mlp  --memory_slot $memory_slots --memory_head_size $memory_head_size --num_memory_heads $num_memory_heads  --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --nlayers 1 --emsize $em --log-interval $log --train_len $train_len --test_len $test_len

# borgy submit --gpu 1 --cpu 2 --mem 48 --gpu-mem 16 --preemptable -i images.borgy.elementai.net/anirudh/anirudh_global_workcase:v1 -v /home/anirudh/:/home/anirudh/ -- /bin/bash /home/anirudh/nips2020/soft_mechanisms/experiment_copying.sh 300 6 4 50 200 4 4 16 4 0.0007

#!/bin/bash
INDEX_NAME=GLOVE200_R24_L64_B1_M32_float
DATA_TYPE=float
DIST_FUN=mips
QUERY_PATH=/home/cqy/cqy/DiskANN/dataset/query.fbin
GT_PATH=/home/cqy/cqy/DiskANN/dataset/1M-topk100-gt
INDEX_PATH="${INDEX_NAME}/${INDEX_NAME}"
LOG_PATH="${INDEX_NAME}/knnsearch.log"
TOPK_LIST=(10)
RESULT_PATH=result/
CACHE_NUM_LIST=(0)
THREAD_NUM=8
L_LIST=(100 200 300)
W_LIST=(1)
for TOPK in ${TOPK_LIST[@]}
do
    for CACHE_NUM in ${CACHE_NUM_LIST[@]}
    do 
        for W in ${W_LIST[@]}
        do
            for L in ${L_LIST[@]}
            do
                echo "beam wide : " $W
                echo "search list :" $L
                sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;
                echo "./tests/search_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FUN --index_path_prefix $INDEX_PATH --num_nodes_to_cache $CACHE_NUM -T $THREAD_NUM -W $W --query_file $QUERY_PATH --gt_file $GT_PATH -K  $TOPK --result_path $RESULT_PATH -L $L "
                ./tests/search_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FUN --index_path_prefix $INDEX_PATH --num_nodes_to_cache $CACHE_NUM -T $THREAD_NUM -W $W --query_file $QUERY_PATH --gt_file $GT_PATH -K  $TOPK --result_path $RESULT_PATH -L $L ;
            done 
        done
    done
done 



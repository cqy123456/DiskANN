
#!/bin/bash
INDEX_PREFIX=SSNPP1M_R24_L64_B0.0610625_M32_temp
INDEX_NAME=SSNPP1M_R24_L64_B0.0610625_M32_temp_part0
DATA_TYPE=float
DIST_FUN=l2
QUERY_PATH=/home/cqy/cqy/ssnpp/FB_ssnpp_public_queries.fbin 
GT_PATH=/home/cqy/cqy/ssnpp/ssnpp-1M-gt
INDEX_PATH="${INDEX_PREFIX}/${INDEX_NAME}"
LOG_PATH="${INDEX_PREFIX}/range_search.log"
RADIUS=96237
RESULT_PATH=result/
CACHE_NUM_LIST=(0)
L_LIST=(60) 
W_LIST=(1)
THREAD_NUM=8
for CACHE_NUM in ${CACHE_NUM_LIST[@]}
do 
    for L in ${L_LIST[@]}
    do
        for W in ${W_LIST[@]}
        do
            echo "beam width : " $W
            echo "search list :" $L
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;
            ./tests/range_search_disk_index --data_type $DATA_TYPE --dist_fn $DIST_FUN --index_path_prefix $INDEX_PATH --num_nodes_to_cache $CACHE_NUM -T $THREAD_NUM -W $W --query_file $QUERY_PATH --gt_file $GT_PATH --range_threshold  $RADIUS result_output_prefix $RESULT_PATH -L $L 
        done 
    done
done
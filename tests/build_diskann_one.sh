#!/bin/bash

DATA_TYPE=float
DIST_FUN=l2
DATA_FILE=/data/cqy/dataset/SSNPP10M_float/FB_ssnpp_database.10M.fbin
#DATA_FILE=/data/datasets/DEEP/base.10M.fbin 
R=24
L=64
B=0.3
M=32
T=8
PQ_DISK_BYTES=0
compressed_level=16
compressed_level_list=(4 8 16)
S_LIST=(1.2 2.4 3.6 4.8)
without_graph=1
for compressed_level in ${compressed_level_list[@]}
do
    for S in ${S_LIST[@]}
    do
        INDEX_NAME=SSNPP10M_R${R}_L${L}_COMPRESS${compressed_level}_M${M}_S${S}_one
        mkdir ${INDEX_NAME}

        INDEX_PATH="${INDEX_NAME}/${INDEX_NAME}"
        LOG_PATH="${INDEX_NAME}/build.log"


        echo "./tests/build_disk_one_segment --data_type $DATA_TYPE --dist_fn $DIST_FUN --data_path $DATA_FILE --index_path_prefix $INDEX_PATH -R $R -L $L -B $B -M $M -T $T --segment_size $S --compressed_level $compressed_level --without_graph $without_graph"
        ./tests/build_disk_index_one_segment  --data_type $DATA_TYPE --dist_fn $DIST_FUN --data_path $DATA_FILE --index_path_prefix $INDEX_PATH -R $R -L $L -B $B -M $M -T $T --segment_size $S --compressed_level $compressed_level --without_graph $without_graph> ${LOG_PATH}
    done
done
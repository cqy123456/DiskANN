#!/bin/bash

DATA_TYPE=float
DIST_FUN=l2
DATA_FILE=/home/cqy/cqy/ssnpp/FB_ssnpp_database.1M.fbin
#DATA_FILE=/data/datasets/DEEP/base.10M.fbin 
R=24
L=64
B=0.0610625
M=32
T=8
PQ_DISK_BYTES=0
S=0.5
#INDEX_NAME=DEEP_R${R}_L${L}_B${B}_M${M}_temp
INDEX_NAME=SSNPP1M_R${R}_L${L}_B${B}_M${M}_temp
INDEX_PATH="${INDEX_NAME}/${INDEX_NAME}"
LOG_PATH="${INDEX_NAME}/build.log"

mkdir ${INDEX_NAME}
#./tests/build_disk_index --data_type $DATA_TYPE $DIST_FUN $DATA_FILE $INDEX_PATH $R $L $B $M $T
echo "./tests/build_disk_index_split  --data_type $DATA_TYPE --dist_fn $DIST_FUN --data_path $DATA_FILE --index_path_prefix $INDEX_PATH -R $R -L $L -B $B -M $M -T $T --segment_size $S"
./tests/build_disk_index_split  --data_type $DATA_TYPE --dist_fn $DIST_FUN --data_path $DATA_FILE --index_path_prefix $INDEX_PATH -R $R -L $L -B $B -M $M -T $T --segment_size $S > ${LOG_PATH}
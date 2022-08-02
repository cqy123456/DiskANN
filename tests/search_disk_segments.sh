#!/bin/bash

DATA_TYPE=float
DIST_FUN=l2
QUERY_PATH=/data/wsl/sift10m/query.fbin
GT_PATH=/data/wsl/sift10m/10M-topk1000-gt
#DATA_FILE=/data/datasets/DEEP/base.10M.fbin 

R=48
L=128
B=0.3
M=32
T=8
PQ_DISK_BYTES=0
S_list=(4.8 2.4 1.6 1.2)
E=16
#INDEX_NAME=DEEP_R${R}_L${L}_B${B}_M${M}_temp
cd /data/wsl/DiskANN/build

TOPK_LIST=(10)
RESULT_PATH=result/
CACHE_NUM_LIST=(0)
THREAD_NUM=8
L_LIST=(100)
W_LIST=(1)
N=4
for S in ${S_list[@]}
do

  INDEX_NAME=SIFT10M_R48_L128_B${B}_M${M}_S${S}
  INDEX_PATH="${INDEX_NAME}/${INDEX_NAME}"
  LOG_PATH="${INDEX_NAME}/search.log"

  for TOPK in ${TOPK_LIST[@]}
  do
      for CACHE_NUM in ${CACHE_NUM_LIST[@]}
      do 
          for W in ${W_LIST[@]}
          do
              for L in ${L_LIST[@]}
              do
                  echo "beam wide : " $W 
                  echo "cache nodes : " $CACHE_NUM 
                  echo "search list :" $L 
                  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches;
                  echo "./tests/search_disk_segments --data_type $DATA_TYPE --dist_fn $DIST_FUN --segments_path_prefix $INDEX_PATH --num_nodes_to_cache $CACHE_NUM -T $THREAD_NUM -W $W --query_file $QUERY_PATH --gt_file $GT_PATH -K  $TOPK --result_path $RESULT_PATH -L $L -N $N"
                  ./tests/search_disk_segments --data_type $DATA_TYPE --dist_fn $DIST_FUN --segments_path_prefix $INDEX_PATH --num_nodes_to_cache $CACHE_NUM -T $THREAD_NUM -W $W --query_file $QUERY_PATH --gt_file $GT_PATH -K  $TOPK --result_path $RESULT_PATH -L $L -N $N >>$LOG_PATH;
              done 
          done
      done
  done 

done

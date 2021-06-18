#!/bin/sh
DATA_DIR=/tmp/jina/mnist
if [ -d ${DATA_DIR} ]; then
  echo ${DATA_DIR}' exists, please remove it before running the script'
  exit 1
fi

mkdir -p ${DATA_DIR}
TRAIN_DATA_DIR=${DATA_DIR}/train
mkdir -p ${TRAIN_DATA_DIR}
cd ${TRAIN_DATA_DIR}
wget -P ${TRAIN_DATA_DIR} https://mnist-jina.obs.cn-north-4.myhuaweicloud.com/train-images-idx3-ubyte
wget -P ${TRAIN_DATA_DIR} https://mnist-jina.obs.cn-north-4.myhuaweicloud.com/train-labels-idx1-ubyte

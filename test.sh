#!/bin/sh
PARTITION=Segmentation

exp_name=$1 # split0 - split3
dataset=$2 # pascal coco 
gpu=$3
if [ $gpu -eq 2 ]; then
  GPU_ID=0,1
elif [ $gpu -eq 1 ]; then
  GPU_ID=0
elif [ $gpu -eq 4 ]; then
  GPU_ID=0,1,2,3
else
  echo "Only 1, 2 and 4 gpu number are supperted"
  exit 1
fi
arch=HDMNet
net=resnet50 # vgg resnet50

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net} 
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_manet.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp test.sh test.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u test.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/test-$now.log
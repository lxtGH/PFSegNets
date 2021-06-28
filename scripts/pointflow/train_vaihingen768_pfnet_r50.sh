#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./vaihingen
mkdir -p ${EXP_DIR}
python -m torch.distributed.launch --nproc_per_node=8 train.py \
  --dataset Vaihingen \
  --cv 0 \
  --arch network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1024 \
  --max_cu_epoch 200 \
  --lr 0.01 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --aux \
  --maxpool_size 14 \
  --avgpool_size 9 \
  --edge_points 128 \
  --match_dim 64 \
  --joint_edge_loss_pfnet \
  --edge_weight 25.0 \
  --ohem \
  --crop_size 768 \
  --max_epoch 200 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --with_aug \
  --wt_bound 1.0 \
  --bs_mult 2 \
  --apex \
  --exp r50 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
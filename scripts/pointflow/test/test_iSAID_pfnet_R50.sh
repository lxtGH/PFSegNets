#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python eval.py \
	--dataset iSAID \
    --arch network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply \
    --inference_mode  whole \
    --single_scale \
    --scales 1.0 \
    --split val \
    --cv_split 0 \
    --maxpool_size 14 \
    --avgpool_size 9 \
    --edge_points 128 \
    --match_dim 64 \
    --resize_scale 896 \
    --mode semantic \
    --no_flip \
    --ckpt_path ${2} \
    --snapshot ${1}
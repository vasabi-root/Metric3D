cd ../../../

python  mono/tools/train.py \
        mono/configs/RAFTDecoder/vit.raft5.small.kubric.py \
        --use-tensorboard \
        --launcher slurm \
        --experiment_name train1 \
        --load-from model_pth \

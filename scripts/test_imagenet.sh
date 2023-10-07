lr=1e-3
shots=16
dataset="imagenet"
backbone="RN50" # RN50 RN101 ViT-B/32 ViT-B/16

gpuid=0

title=FAR_lr${lr}
log_file=${title}.log
checkpoint_dir=./checkpoint/s${shots}_${dataset}_${backbone}/
checkpoint=${checkpoint_dir}${title}_last.pth

CUDA_VISIBLE_DEVICES=${gpuid} python test_imagenet.py \
 --config ./configs/${dataset}.yaml \
 --resume ${checkpoint} \
 --batch_size 64 \
 --shots ${shots} \
 --title ${title} \
 --log_file ${log_file} \
 --desc "test ${daatset} with backbone ${backbone}." \

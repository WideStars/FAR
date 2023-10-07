lr=1e-3
shots=16
seed=1

gpuid=0

title=FAR_domain_shift
log_file=${title}.log
checkpoint_dir=./checkpoint/s${shots}_imagenet_RN50/
checkpoint=${checkpoint_dir}FAR_lr${lr}_last.pth

CUDA_VISIBLE_DEVICES=${gpuid} python test_domain_shift.py \
 --config ./configs/imagenet.yaml \
 --resume ${checkpoint} \
 --batch_size 64 \
 --shots ${shots} \
 --seed ${seed} \
 --title ${title} \
 --log_file ${log_file} \
 --desc "test domain shift" \

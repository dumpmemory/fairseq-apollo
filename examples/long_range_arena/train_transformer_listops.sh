#! /bin/bash

split=0
seeds=(1 11 65537 101 1999 2017)
seed=${seeds[$split]}

DATA=$1
SAVE_ROOT=$2
DEVICES=$3
model=transformer_lra_listop
# exp_name=1_apollo_luna_k16_run${seed}

SAVE=${SAVE_ROOT}
# rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

CUDA_VISIBLE_DEVICES=0,1 python -u train.py ${DATA} \
    --seed $RANDOM --ddp-backend c10d --fp16 \
    -a ${model} --task sentence_prediction \
    --apply-bert-init  --num-classes 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.0001 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --batch-size 8 --sentence-avg --update-freq 2 \
    --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion lra_cross_entropy --max-update 5000 --save-interval-updates 1000 \
    --warmup-updates 1000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --keep-interval-updates 10 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --tensorboard-logdir ${SAVE} | tee ${SAVE}/log.txt


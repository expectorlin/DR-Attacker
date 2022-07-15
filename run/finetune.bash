name=finetune
flag="--attn soft --train auginslistener
 --featdropout 0.3
 --angleFeatSize 128
 --feedback sample
 --mlWeight 0.2
 --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 300000 --maxAction 35
 --finetuneAgent
 --load tasks/R2R/snapshots/snap/adv_train/state_dict/best_val_unseen"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 


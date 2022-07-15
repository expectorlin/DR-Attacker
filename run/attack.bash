name=attack
flag="--attn soft --train listener
 --featdropout 0.3
 --angleFeatSize 128
 --feedback argmax
 --feedbackAttacker sample
 --mlWeight 0.
 --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 10000 --maxAction 35
 --load tasks/R2R/snapshots/snap/pretrain/state_dict/best_val_unseen
 --pretrainAttacker"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name
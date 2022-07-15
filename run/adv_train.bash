name=adv_train
flag="--attn soft --train auginslistener
 --featdropout 0.3
 --angleFeatSize 128
 --feedback sample
 --feedbackAttacker sample
 --mlWeight 0.2
 --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 40000 --maxAction 35
 --advTrain
 --ifSelfSupervised
 --itersAlterNav 3000
 --itersAlterAtt 1000
 --load tasks/R2R/snapshots/snap/pretrain/state_dict/best_val_unseen
 --loadAttacker tasks/R2R/snapshots/snap/attack/state_dict/best_val_unseen_attacker"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name 






name=pretrain
flag="--attn soft --train auginslistener
 --featdropout 0.3
 --angleFeatSize 128
 --feedback sample
 --mlWeight 0.2
 --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 40000 --maxAction 35
 --pretrainAgent"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 python3.6 r2r_src/train.py $flag --name $name

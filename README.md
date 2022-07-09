# Adversarial Reinforced Instruction Attacker for Robust Vision-Language Navigation
PyTorch implementation of the paper ["Adversarial Reinforced Instruction Attacker for Robust Vision-Language Navigation"](https://arxiv.org/abs/2107.11252) (TPAMI 2021).

## Environment Installation
The environment installation follows that in [EnvDrop](https://github.com/airsplay/R2R-EnvDrop).
<br>
Python requirements: Need python3.6 (python 3.5 should be OK)
<br>
```
pip install -r python_requirements.txt
```  
Install Matterport3D simulators:
<br>
```
git submodule update --init --recursive
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make -j8
```

## Data Preparation
Download Room-to-Room navigation data:
<br>
```
bash ./tasks/R2R/data/download.sh
```  
Download image features for environments:
<br>
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

Download [R2R augmentation data](http://people.eecs.berkeley.edu/~ronghang/projects/speaker_follower/data_augmentation/R2R_literal_speaker_data_augmentation_paths.json) from [speaker-follower](https://github.com/ronghanghu/speaker_follower).
<br>

Download R2R navigation data added target words and candidate substitution words.
<br>

Download object word vocabulary.  
<br>

## Code
Coming soon

## Acknowledgement
The implementation relies on resources from [EnvDrop](https://github.com/airsplay/R2R-EnvDrop) and [speaker-follower](https://github.com/ronghanghu/speaker_follower). We thank the original authors for their open-sourcing.

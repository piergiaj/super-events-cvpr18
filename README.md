# Learning Latent Super-Events to Detect Multiple Activities in Videos

This repository contains the code for our [CVPR 2018 paper](https://arxiv.org/abs/1712.01938):

    AJ Piergiovanni and Michael S. Ryoo
    "Learning Latent Super-Events to Detect Multiple Activities in Videos"
    in CVPR 2018

If you find the code useful for your research, please cite our paper:

        @inproceedings{piergiovanni2018super,
              title={Learning Latent Super-Events to Detect Multiple Activities in Videos},
              booktitle={CVPR},
              author={AJ Piergiovanni and Michael S. Ryoo},
              year={2018}
        }


# Temporal Structure Filters
The core of our approach, the temporal structure filters can be found in [temporal_structure_filter.py](temporal_structure_filter.py). This creates the TSF with N cauchy distributions. We create the super-event model in [super_event.py](super_event.py).


# Activity Detection Experiments
To run our pre-trained models:

```python train_model.py -mode joint -dataset multithumos -train False -rgb_model_file models/multithumos/rgb_baseline -flow_model_file models/multithumos/flow_baseline```


We tested our models on the [MultiTHUMOS](http://ai.stanford.edu/~syyeung/everymoment.html), [Charades](http://allenai.org/plato/charades/), and [AVA](https://research.google.com/ava/) datasets, using only the temporal annotations of AVA. We provide our trained models in the model directory as well as the convert json format for the datasets in the data directory.


# Example Learned Super-events
Our trained models on [MultiTHUMOS](http://ai.stanford.edu/~syyeung/everymoment.html) which contains ~2500 videos of 65 different activities in continuous videos and [Charades](http://allenai.org/plato/charades/) which contained ~10,000 continuous videos learned various super-events. Here are some example learned super-events from our models.

For the block action, our model learned to focus on the pass/dribbe before the shot and the shot/dunk action.

![dribble](/examples/dribble.gif?raw=true "Dribble super-event")
![block](/examples/block.gif?raw=true "Block/Dunk up Super-event")


================================================================================


# Requirements

Our code has been tested on Ubuntu 14.04 and 16.04 using python 2.7, [PyTorch](pytorch.org) version 0.3.1 (but will likely work with other versions) with a Titan X GPU.


# Setup

1. Download the code ```git clone https://github.com/piergiaj/super-events-cvpr18.git```

2. Extract features from your dataset. See [Pytorch-I3D](https://github.com/piergiaj/pytorch-i3d) for our code to extract I3D features.

3. [train_model.py](train_model.py) contains the code to train and evaluate models.

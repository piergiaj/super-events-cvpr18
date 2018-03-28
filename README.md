# Learning Latent Super-Events to Detect Multiple Activities in Videos

This repository contains the code for our [CVPR 2018 paper](https://arxiv.org/abs/1712.01938):

    AJ Piergiovanni and Michael S. Ryoo
    "Learning Latent Super-Events to Detect Multiple Activities in Videos"
    in CVPR 2018

If you find the code useful for your research, please cite our paper:

        @inproceedings{piergiovanni2018super,
              title={Learning Latent Super-Events to Detect Multiple Activities in Videos},
              booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
              author={AJ Piergiovanni and Michael S. Ryoo},
              year={2018}
        }


# Temporal Structure Filters
![tsf](/examples/temporal-structure-filter.png?raw=true "tsf")

The core of our approach, the temporal structure filters (TSF) can be found in [temporal_structure_filter.py](temporal_structure_filter.py). This creates the TSF with N cauchy distributions. We create the super-event model in [super_event.py](super_event.py).


# Activity Detection Experiments
![model overview](/examples/model-overview.png?raw=true "model overview")

To run our pre-trained models:

```python train_model.py -mode joint -dataset multithumos -train False -rgb_model_file models/multithumos/rgb_baseline -flow_model_file models/multithumos/flow_baseline```

We tested our models on the [MultiTHUMOS](http://ai.stanford.edu/~syyeung/everymoment.html), [Charades](http://allenai.org/plato/charades/), and [AVA](https://research.google.com/ava/) datasets (only the temporal annotations were used in AVA). We provide our trained models in the model directory as well as the convert json format for the datasets in the data directory.

## Results
On Charades:

|  Method | mAP (%) |
| ------------- | ------------- |
| Two-Stream + LSTM [1] | 9.6  |
| Sigurdsson et al. [1]  | 12.1  |
| I3D [2] baseline      | 17.22 |
| I3D + LSTM          | 18.1  |
| I3D + Super-events | **19.41** |

On MultiTHUMOS

|  Method | mAP (%) |
| ------------- | ------------- |
| Two-Stream [3]  | 27.6  |
| Two-Stream + LSTM [3] | 28.1 | 
| Multi-LSTM [3]  | 29.6  |
| I3D [2] baseline | 29.7 |
| I3D + LSTM | 29.9 |
| I3D + Super-events | **36.4** |


# Example Learned Super-events
Our trained models on [MultiTHUMOS](http://ai.stanford.edu/~syyeung/everymoment.html) which contains ~2500 videos of 65 different activities in continuous videos and [Charades](http://allenai.org/plato/charades/) which contained ~10,000 continuous videos learned various super-events. Here are some example learned super-events from our models.

For the block action, our model learned to focus on the pass/dribbe before the shot and the shot/dunk action.

![basketball](/examples/learned-super-events.png?raw=true "basketball super-event")



Here are examples of the temporal interval focused on by the super-event for the 'block' action detection capturing dribbling:
![dribble](/examples/dribble.gif?raw=true "Dribble super-event") ![block](/examples/dribble3.gif?raw=true "Block/Dunk up Super-event")


Here are examples of the temporal interval focused on by the super-event for the 'block' action detection capturing blocking/dunking:
![dribble](/examples/block.gif?raw=true "Dribble super-event") ![block](/examples/block2.gif?raw=true "Block/Dunk up Super-event")


# Requirements

Our code has been tested on Ubuntu 14.04 and 16.04 using python 2.7, [PyTorch](pytorch.org) version 0.3.1 (but will likely work with other versions) with a Titan X GPU.


# Setup

1. Download the code ```git clone https://github.com/piergiaj/super-events-cvpr18.git```

2. Extract features from your dataset. See [Pytorch-I3D](https://github.com/piergiaj/pytorch-i3d) for our code to extract I3D features.

3. [train_model.py](train_model.py) contains the code to train and evaluate models.


# Refrences
[1] G.  A.  Sigurdsson,  S.  Divvala,  A.  Farhadi,  and  A.  Gupta. Asynchronous temporal fields for action recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017

[2] J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[3] S. Yeung, O. Russakovsky, N. Jin, M. Andriluka, G. Mori, and L. Fei-Fei. Every moment counts: Dense detailed labeling of actions in complex videos. International Journal of Computer Vision (IJCV), pages 1–15, 2015

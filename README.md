# Volumetric landmark detection with a multi-scale translation equivariant neural network

This project contains all the code for the ISBI paper: *Volumetric landmark detection with a multi-scale translation equivariant neural network*. 
(https://arxiv.org/abs/2003.01639)

The architecture is useful for landmark detection, especially with large input image size. 
Every operations including the cropping is differentiable. Thus, the network can be trained end-to-end with the output being the landmark coordinates. 
There is no ground-truth mask needed.  


<img src="https://github.com/tym002/bifurcation_detection/blob/master/overview.png" width="600">

## requirements: 

`tensorflow-gpu 1.15.0`

`python 3.6.13`

## Code:
`model.py` contains the proposed multi-scale Loc-net method for training and testing. 

`LocalizerNet.py` contains the regressing Gaussian heatmap and single-scale Loc-net methods, can be used as baselines

To train or test your own model, run 
`python model.py --mode train/test`

`--multi_loss` whether to use multi-stage-loss schedule. 

`--use_callback` whether to use loss fallback, loss weights of earlier stages will decrease with more epoch during training 

`--random_shift` whether to use random-shift before cropping at each stage

To evaluate the result, run 
`python evaluate.py`

## Citation:

If you find our code useful, please cite our work, thank you!
```
@inproceedings{ma2020volumetric,
  title={Volumetric landmark detection with a multi-scale shift equivariant neural network},
  author={Ma, Tianyu and Gupta, Ajay and Sabuncu, Mert R},
  booktitle={2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)},
  pages={981--985},
  year={2020},
  organization={IEEE}
}
```

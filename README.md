# Volumetric landmark detection with a multi-scale translation equivariant neural network

This project contains all the code for the ISBI paper "Volumetric landmark detection with a multi-scale translation equivariant neural network". 

The architecture is useful for landmark detection, especially with large input image size. 
Every operations including the cropping is differentiable. Thus, the network can be trained end-to-end with the output being the landmark coordinates. 
There is no ground-truth mask needed.  

![alt text](https://github.com/tym002/bifurcation_detection/blob/master/overview.png)

requirements: 
tensorflow-gpu 1.15.0
python 3.6.13

model.py contains the proposed multi-scale Loc-net method. 

LocalizerNet.py contains the regressing Gaussian heatmap and single-scale Loc-net methods 

testing.py contains the test script 

To run the testing with pre-train model, run 
'python testing.py'

Note: change the location of the saved prediction at line 504. replace '...' with your destination

To train your own model, run 
'python model.py'

remember to change the saved file destination 

To evaluate the result, run 
'python evaluate.py'. Remember to download the 'test_ground_truth.npy' file and change the destination 

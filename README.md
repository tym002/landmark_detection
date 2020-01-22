# bifurcation_detection

This project contains all the code for the ISBI paper "Volumetric landmark detection with a multi-scale translation equivariant neural network". 

model.py contains the proposed multi-scale Loc-net method. 

LocalizerNet.py contains the regressing Gaussian heatmap and single-scale Loc-net methods 

testing.py contains the test script 

To run the testing with pre-train model, run 
'python testing.py'
Note: change the location of the saved prediction at line 504. replace '...' with your destination

To train your own model, run 
'python model.py'
remember to change the saved file destination 

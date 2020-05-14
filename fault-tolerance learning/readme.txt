Code for the fault-tolerance learning of HP SNN.  

Instructions 

here are some statements of every program in folder main/:

- main_fashion_gp.py || HP model in the paper. After 100 epoch training, you can get ~ 88.5% best acc.

- main_fashion_hp.py || HP model in the paper. After 100 epoch training, you can get ~ 88.2% best acc.

- test_fashion_crop || testing experiment for cropping inputs on Fashion-MNIST, you can control 'names' and 'model' parameter to choose target model of DP or HP.

- test_fashion_noise || testing experiment for noisy inputs on Fashion-MNIST, you can control 'names' and 'model' parameter to choose target model of DP or HP, and noise type {gaussian, salt, pepper, speckle}can be chosen through setting the parameter 'noise'.

- By running the test_*.py, you can reproduce the results of Robustness Exp. as shown in the Fig.4.c-d.

-Note : to run the results of MNIST£¬you canmodify the data loader and run the same code.
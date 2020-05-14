## Code for the few-shot learning of hybrid plasticity model.

## Instructions 

You MUST download the PYTHON VERSION of Omniglot dataset, unzip the file of 'images_background' and 'images_evaluation',
and load the './python' into this directory.

**Note**: If the environment a configurations are different, the results may fail to work properly.
You may need to modify the package version or adjust code details according to your needs.

## Environment

Linux: Ubuntu 16.04

cuda9.0 & cudnn6.0

Python 3.5.4

torch 1.2.0

torchvision 0.2.2

numpy 1.17.2

scipy 1.2.1

scikit-image 0.15.0


## Instructions

- main_gp.py || Run the GP model in the Omniglot. It can obtain about 98.8% best acc.

- main_hp.py || Run the HP model in the Omniglot.. It can obtain about 28.7% best acc.


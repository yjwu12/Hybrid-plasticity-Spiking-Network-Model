# Hybrid-plasticity-spiking-neural-network
- Code for paper  *Brain-inspired global-local hybrid learning towards human-like intelligence*. 
- Hybrid plasticty (HP) models provides a generic framework for training global-local hybrid SNNs using pytorch.
- DP models is designed to support multiple spike coding methods (rate-based and temporal based), multiple neuron models, and learning rules (Hebbian-based, STDP-based etc.)

## Setup
We use the Pycharm platform. 
- Installation link: https://www.jetbrains.com/pycharm/download/
- Chooose your operating system and Python version 3.5
- Download and install

## Requirements

Linux: Ubuntu 16.04

Cuda 9.0 & cudnn6.0

Python 3.5

Platform：NVIDIA Titan Xp and NVIDIA GTX 1080. 

torch 0.4.0 (gpu)

numpy 1.16.2

scipy 1.0.0

## Instructions for use
- File names starting with ‘main_*’ can be run to reproduce the results in this paper.
- Reproduction instructions 
    - **File 'Classification'** :  It reports the classification performance of the proposed model (Figure 2-3).
    - **File 'Tolerance learning'** : It produces the main results of GP-based and HP-based model on the tolerance learning tasks (Figure 4).
    - **File 'Few-shot learning'**: It produces the main results of GP-based and HP-based model on the few-shot learning tasks (Figure 5).
    - **File 'Multitask learning'**  :  It produces the main results of GP-based and HP-based model on the multitask learning tasks (Figure 5).
  
 

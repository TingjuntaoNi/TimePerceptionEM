# TimePerceptionEM
## Overview
This repository contains the scripts of Artificial neural network part in Human Time Perception project. Its main goal is to model and analyze how humans estimate durations using deep learning techniques.  

## Repository Version
This is the first version of this code repo (2025-6-5).  

## Setups
- **Python version**: Python 3.10.17
  
We recommend using **Conda** to create and manage the environment for this project.  

```bash
# 1. Create a new Conda environment (named "timeperception-ann")
conda create -n timeperception-ann python=3.10.17  

# 2. Activate the environment
conda activate timeperception-ann

# 3. Install dependencies from requirements.txt
pip install -r requirements.txt
```

If you encounter issues with installing torch, please visit the official PyTorch website (https://pytorch.org/get-started/locally/) to find the appropriate installation command for your system (CPU/GPU).  

## Usage
### Quick start

The configuration file `config.yaml` contains all adjustable parameters for model training, such as current fold index (for a 4:1 training-validation split), learning rate, number of epochs, batch size, and data paths. You can modify it to suit your specific requirements.

Once your configuration is ready, start a training session by running the following command in the terminal:

```bash
python train.py
```
A log file will be automatically generated at the start of training, and the best-performing model will be saved based on validation accuracy after each epoch.


### Project Structure

This repository consists of several main components:  

#### 1. Model Architecture





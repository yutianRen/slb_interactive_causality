# A Self-Labeling Method for Adaptive Machine Learning by Interactive Causality


This repo includes the source codes and instructions of how to run the
simulation and run the self-labeling test.

**- 07/31/2023  [Note that]** the self-labeling script hasn't been uploaded yet. We will update the code once paper is
accepted.

## Install required packages


This directory includes a designed simulation to generate simulated datasets. 
The simulation is based on [TDW](https://github.com/threedworld-mit/tdw). In order to 
run the simulation, users need to follow the TDW instruction to install TDW first.

Also, a conda environment is suggested to be created by with dependencies running:
```
conda env create -n slb --file slb_environment.yml
```


## To run the simulation


In this simulation, we use a customized landscape. The landscape obj files are 
in ```/landscape/landV3.1/``` directory. First we need to follow the [method](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/custom_models/custom_models.md) 
of importing
custom object files in TDW to load the landscape files. Here, we can run
```
python3 load_landscape_obj.py
```
to generate the assets files in Unity. Next, users can find the file path to the 
generated assets files and change the url field of each land block in 
```/config/land_params_V3.1.yaml```. This config also includes other simulation parameters
that can be changed. 

To run the simulation, open a terminal and run
```
python3 simulator.py --port 6788 --config_path config/land_params_V3.1.yaml
```

The simulation will then start and the generated files will be in the output_path.

### Dataset resampling
Use ```dset_prep.py``` to preprocess the simulation-generated raw data with resampling function. A ```.pkl``` file
will be generated .



## To run the self-labeling test

The main file of self-labeling is slb.py. In ```/nets/``` directory, the MLP model
and its configurations are defined. Please be sure you have generated a dataset
from the simulation in order to test on the dataset.

Then, to pretrain and test the unperturbed data, we can run the self-labeling by
```
python3 slb.py --pretrain_path <unperturbed dataset path>.pkl 
        --data_path <unperturbed dataset path>.pkl 
        --land_params_yaml_path <path to land_params.yaml>
        --out_dir_path <output path>
```

To pretrain on unperturbed dataset and adapt on perturbed dataset, run
```
python3 slb.py --pretrain_path <unperturbed dataset path>.pkl 
        --data_path <perturbed dataset path>.pkl 
        --land_params_yaml_path <path to land_params.yaml>
        --out_dir_path <output path>
```

Additional training data can be added by the ```--add_data``` argument.


To adjust the penalty for inaccurate interaction time inference, two parameters 
```--x_offset_vel``` and ```--z_offset_vel``` can be set.

Two sample datasets and their corresponding ```land_params.yaml``` can be downloaded from a google 
drive [link](https://drive.google.com/drive/folders/1wPcWQjs88ON8h_E7BlC1A9-qPjeesfM4?usp=sharing). 
A sample cmd of training on perturbed dataset is:
```
python3 slb.py --pretrain_path run_V3.1_b75f25h1015_balanced_6000.pkl 
        --data_path run_V3.1_b75f25h1015w0.5-0.50.5_balanced_6000.pkl 
        --land_params_yaml_path land_params_V3.1.yaml
        --out_dir_path result
```







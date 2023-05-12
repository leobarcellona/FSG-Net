# FSG-Net

The work is developed in collaboration with [bach05](https://github.com/bach05) and [albigotta](https://github.com/albigotta).

## Work in progress

The repository is under construction to provide a user-friendly interface and a more readable code. The draft code we developed is already available. 

## Installation and setup
```commandline
pip install -r requirements.txt
```
### Pretrained Models

Pre-trained models are available, please refer to the following list:

- Few-Shot Module: LINK, destination folder: `./output/models/fewshot_modules`
- Backbone: LINK, destination folder: `./output/models/backbones`
- Quality Head: LINK, destination folder: `./output/models/heads`
- Angle Head: LINK, destination folder: `./output/models/heads`
- Width Head: LINK, destination folder: `./output/models/heads`
 folder. 

## Training

### Configuration file
The training process (e.g. training hyperparameters, paths, ecc...) can be configured through `./config/generic.yaml` files. 

[DETAILED EXPLANTION]

### Run the training

```commandline
python3 train.py
```

## TEST

[ADD THE CODE FOR TESTING]
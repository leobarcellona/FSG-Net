# FSG-Net

The work is developed in collaboration with [bach05](https://github.com/bach05) and [albigotta](https://github.com/albigotta).
Please cite these papers in your publications if FSG-Net helps your research.

    @inproceedings{barcellona2023fsg,
    title={FSG-Net: a Deep Learning model for Semantic Robot Grasping through Few-Shot Learning},
    author={Barcellona, Leonardo and Bacchin, Alberto and Gottardi, Alberto and Menegatti, Emanuele and Ghidoni, Stefano},
    booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
    pages={1793--1799},
    year={2023},
    organization={IEEE}
    }

## Work in progress

The repository is under construction to provide a user-friendly interface and a more readable code. The draft code we developed is already available. 

## Installation and setup
```commandline
pip install -r requirements.txt
```
### Pretrained Models

Pre-trained models are available, please refer to the following list:

- Few-Shot Module: [LINK](https://drive.google.com/file/d/1TWqorlp8rj_6B2rZ1NqXYJyFjBJ38A9V/view?usp=drive_link), destination folder: `./output/models/fewshot_modules`
- Backbone: [LINK](https://drive.google.com/file/d/11dP--yA4kLwp94O85ZBIX9JTUX7SIHfL/view?usp=drive_link), destination folder: `./output/models/backbones`
- Quality Head: [LINK](https://drive.google.com/file/d/1Clid9DRfW43TcUMXJKgaLdjJc6hUCbld/view?usp=drive_link), destination folder: `./output/models/heads`
- Angle Head: [LINK](https://drive.google.com/file/d/1CJNL2jDRcv_m8mh0T73khTTuiQcLRM2F/view?usp=drive_link), destination folder: `./output/models/heads`
- Width Head: [LINK](https://drive.google.com/file/d/1XboYx4olOdyomchxo73cO7Kf8ajs9l_5/view?usp=drive_link), destination folder: `./output/models/heads`
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


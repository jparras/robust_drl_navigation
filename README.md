# Robust Deep Reinforcement Learning for underwater navigation with unknown disturbances

## Introduction

Code used to obtain the results in the paper Parras, J., & Zazo, S. (2021, June). Robust Deep Reinforcement Learning for underwater navigation with unknown disturbances. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3440-3444). IEEE. [DOI](https://doi.org/10.1109/ICASSP39728.2021.9414937).

## Launch

This project has been tested on Python 3.6 on Ubuntu 18. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To show the results obtained in the paper, simply run the main file as:
```
$ python main.py
```

In case that you want to train and/or test, set the train and/or test flag to `True` in the `main_dgm.py` file and then run the same order as before. Note that the results file will be overwritten. 

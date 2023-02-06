# LogitNorm Out-of-Distribution (OOD) Detection Method



## Description
The aim of this toolkit is to make an interface to:
1. apply LogitNorm OOD detection method via using logits normalization on the loss function during the training phase;
2. and evaluate various OOD detection methods on a set of predefined in-distribution (ID) and OOD datasets. 
The toolkit is adapted from an open-source project https://github.com/hongxin001/logitnorm_ood.

The framework is shown in the following image, where red rectangles represent main functions and blue parallelograms represent data flow.

![image](raw/overall.png?raw=true "Interface Framework")


## Installation

To use this toolkit, please clone our repository, change to the workfolder, and install all the dependencies: 
```
git clone https://gricad-gitlab.univ-grenoble-alpes.fr/dingw/logitnorm_tool.git
cd logitnorm_tool
pip install -r requirements.txt
```

## Usage
To work with this toolkit, you can check and run the demo examples [test.ipynb](https://gricad-gitlab.univ-grenoble-alpes.fr/dingw/logitnorm_tool/-/blob/main/test.ipynb) and [train.ipynb](https://gricad-gitlab.univ-grenoble-alpes.fr/dingw/logitnorm_tool/-/blob/main/train.ipynb) on any IDE.

Note that the data folder with training and test images is not in this repo. Please check https://cloud.univ-grenoble-alpes.fr/s/t5ffcbD8f46o7MB to download datasets if needed.



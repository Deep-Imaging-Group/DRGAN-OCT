# PyTorch DRGAN
----A Pytorch implementation of **Noise-Powered Disentangled Representation for Unsupervised Speckle Reduction of Optical Coherence Tomography Images**.

<div align=center><img src="https://github.com/tsmotlp/DRGAN/blob/main/images/Fig1.png" width="700px"/></div>

## 1. File Description
**data**
* **`train_valid`**: the directory to store data for training and validation.
* **`test`**: the directory to store test data.

**dataset**
* **`dataset.py`**: python script to build pytorch `Dataset` and `DataLoader`.

**models**
* **`base_model.py`**: the father class implementing the network building, setup input, forward computation, backpropagation, network saving and loading, learning rate schedulers, and visualization of losses and metrics.
* **`drgan_model.py`**: containing the forward and backward process of our proposed model.
* **`drgan_nets.py`**: the implementation of our proposed network achitectures, such as `Encoder`, `Decoder` etc.

**run**
* **`trainer.py`**: a basic template python file for training from scratch, or resuming training, and validation the `Model`.
* **`tester.py`**: a basic template python file for testing the `Model`.

**utils**
* **`help_functions.py`**: the python file can be used to store and modify the model initilization strategies and optimizer scheduler settings.
* **`metrics.py`**: the python file can be used to store and modify the evaluation metrics.
* **`visualizer.py`**: the python file can be used for visualization of the losses and images.

**main.py**: the script for running the code (train/validation/test).
* if you want to train the model from scratch, run 
```
python main.py --mode train --start_epoch 1
``` 
* if you want to resume the training process, run 
```
python main.py --mode train --start_epoch epoch_you_want_to_resume
``` 
* if you want to test the model, run 
```
python main.py --mode test --load_epoch parameters_epoch_you_want_to_test
``` 

**configs.py**: the python file can be used to store and modify the hyper-parameters for training, validation and testing process.

## 2. Installation
* Clone this repo:
```
git clone https://github.com/tsmotlp/DRGAN-OCT

cd DRGAN
```
* Install PyTorch 1.0+ and other dependencies (e.g., Pillow, torchvision, visdom)

## 3. Train and test DRGAN on your own data
if you want to train and test DRGAN on your own datae, your just need to:
* prepare you own train, validation and test data into directory `data`.
* modify the hyper-parameters in **`configs.py`** to make it suitable for your own data.
* follow the instructions of **`main.py`**.
* To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

## 4. Citation
If you use this code for your research, please cite our paper.

## 5. Contact
if you have any questions, please email to: tsmotlp@163.com.

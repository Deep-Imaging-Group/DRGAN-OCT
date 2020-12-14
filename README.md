# PyTorch DRGAN
----A Pytorch implementation of Noise-Powered Disentangled Representation for Unsupervised Speckle Reduction of Optical Coherence Tomography Images.

![image](https://github.com/tsmotlp/DRGAN/blob/main/images/Fig1.png)

## 1. File Description
**data**
* **`raw_data`**: the directory to store raw data for training, validation, and testing, containing `train.txt`, `valid.txt`, and `test.txt`, respectively.
* **`processed_data`**: the directory to store the processed data of `raw_data` by `preprocessing.py`, containing `train.npy`, `valid.npy` and `test.npy`, respectively.
* **`preprocessing.py`**: python script to split inputs and labels, convert input sentences to indexes, pad index list to the same length, and save the processed data as `.npy` format.
* **`dataset.py`**: python script to build pytorch `Dataset` and `DataLoader`.

**models**
* **`BaseModel.py`**: the father class implementing the network building, setup input, forward computation, backpropagation, network saving and loading, learning rate schedulers, and visualization of losses and metrics.
* **`**Model.py`**: the implementaion that extends `BaseModel` of specific models (methods), such as `LstmModel`, `Seq2SeqModel` etc.
* **`**Net.py`**: the code of network achitectures, such as `LstmNet.py`, `Seq2SeqNet.py` etc.

**run**
* **`trainer.py`**: a basic template python file for training from scratch, or resuming training, and validation the `**Model`.
* **`tester.py`**: a basic template python file for testing the `**Model`.

**utils**
* **`configs.py`**: the python file can be used to store and modify the hyper-parameters for training, validation and testing process.
* **`help_functions.py`**: the python file can be used to store and modify the model initilization strategies and optimizer scheduler settings.
* **`metrics.py`**: the python file can be used to store and modify the evaluation metrics, such as `Precsion`, `Recall`, `F1-score` etc.
* **`visualizer.py`**: the python file can be used for visualization of the losses and images.

**main.py**: the script for running the code.
* if you want to train the model from scratch, run `python main.py --mode train --start_epoch 1` in the command line of your python environment.
* if you want to resume the training process, run `python main.py --mode train --start_epoch epoch_you_want_to_resume` in the command line of your python environment.
* if you want to test the model, run `python main.py --mode test --load_epoch parameters_epoch_you_want_to_test` in the command line of your python environment.

## 2. Train and test your own model
if you want to train and test your own model using this template, your just need to:
* create a `YourOwnNet.py` file in directory `models` and implement every details of your own networks.
* create a  `YourOwnModel.py` that extends the father class `BaseModel` and only implement the `forward()` function and `backward()` function 
* import `YourOwnModel` in `trainer.py` and `tester.py` as well as modify the **MODEL NAME** with `YourOwnModel'.

## 3. Future works
* implement more `**Models` for punctuation prediction.
* try to do punctuation restoration.
* try different kinds of Language materials, such as Chinese etc.

## 4. Contacts
if you have any questions, please email to: [tsmotlp](tsmotlp@163.com) or [sakura](tsmotlp@163.com).

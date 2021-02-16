# mlpractice
machine learning practice
```python
    author  :   Md. Nadim Sarwar
    version :   0.0.2 
```
**Environment**
```python
    OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
    Memory      : 7.7 GiB  
    Processor   : Intel® Core™ i3-5010U CPU @ 2.10GHz × 4     
    Graphics    : Intel® HD Graphics 5500 (BDW GT2)  
    Gnome       : 3.28.2
```
# Requirements
* **Python**: 3.6.9 (default, Oct  8 2020, 12:12:24) 
* **GCC**   : 8.4.0 on linux
* ```pip3 install -r requirements.txt```
# Entry Point
* run **main.py**
```python
    usage: preprocessing script image classification datasets [-h] [--data_path DATA_PATH]
                                            [--save_path SAVE_PATH] [--fmt FMT]
                                            [--data_dim DATA_DIM]
                                            [--image_type IMAGE_TYPE]
                                            [--data_size DATA_SIZE]
                                            [--label_den LABEL_DEN]

    optional arguments:
    -h, --help            show this help message and exit
    --data_path DATA_PATH
                            Path of the data folder that contains Test and Train
    --save_path SAVE_PATH
                            Path to save the tfrecords
    --fmt FMT             format of the image
    --data_dim DATA_DIM   dimension to resize the images
    --image_type IMAGE_TYPE
                            type of image (grayscale,rgb,binary)
    --data_size DATA_SIZE
                            the size of tfrecords
    --label_den LABEL_DEN
                            label denoter (by default : train)

```
# Training
* check **colab.ipynb** for training details
# Folder Structure
**THE FOLDER STRUCTURE OF PASSED DATA PATH MUST BE AS FOLLOWS**
```python
    data
    ├── test
    │   ├── ......
    │   ├── class1
    │   └── class2
    └── train
        ├── ......
        ├── class1
        └── class2

```
# TODO
* Error Handling: add try exception blocks
* **Documentation**
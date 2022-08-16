# Unet_Weizmann-Horse
Unet training with Weizmann Horse database

This work is the assignment of visual cognitive engineering. This work mainly uses UNET to complete the semantic segmentation task, and uses the pytorch framework.

# Installation
This code runs under python3.8 and pytorch1.11.0. The total configuration is as follows:
- python 3.8
- pytorch 1.11.0
- numpy 1.18.5
- cv2

# download model
The download link of the pre_trained optimal model is:

https://pan.baidu.com/s/1nW_c2rcnxOTs3G3pfcXznw?pwd=abcd

Extraction code: abcd

# Quick Start
You need put the pre_trained model in the directory`./pre_model`. At the same time, I also put some pictures to be tested in the directory`./test_image`. Run the `test.py` file.Then in the directory `./test_result`, you can find the image after semantic segmentation of the model.

```python
python test.py
```

# prepare your data
The final path structure used in my code looks like this:

````
$PATH_TO_DATASET/trainimage
├──── images
│    ├──── alpha (hundreds of images)
├──── masks
│    ├──── alpha (Dozens of images)
````

# Train your model
1. Please put your data into the directory `/.trainimage`, where the original image is placed in `./trainimage/images` and the label image is placed in `./trainimage/ masks`.
2. Please ensure that your label image has been normalized.
3. Please run the `train.py` file. 
4. You can find the network parameter model after each epoch and the total Miou, bio and loss indicators under the directory `./output`.

```python
python train.py
```


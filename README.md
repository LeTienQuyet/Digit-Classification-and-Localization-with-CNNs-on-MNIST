# Digit-Classification-and-Localization-with-CNNs-on-MNIST
* In this project, the focus is on using a deep learning network to classify and locate a single object in an image.
## How to use?
* First, you need to navigate to the src directory.
* You can train the model yourself using the command (you can change the values of the hyperparameters):
```
python train.py --epoch 100 --lr 0.0001 --batch_size 256
```
* You can make predictions for new images, provided that the images are located in the image directory:
```
python predict.py {image_name}
```
* In addition, you can make predictions directly on your own drawings:
```
python demo.py
```

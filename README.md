# American Sign Language Detection using Deep Learning



## <u>About the Project</u>

This project aims to achieve American Sign Language Detection using Deep Learning. Also, real time webcam detection is a major aim of this project which will be refined from time to time.
### Some Results
![](https://github.com/sovit-123/American-Sign-Language-Detection-using-Deep-Learning/blob/master/outputs/A_test.jpg)
![](https://github.com/sovit-123/American-Sign-Language-Detection-using-Deep-Learning/blob/master/outputs/Z_test.jpg)
![](https://github.com/sovit-123/American-Sign-Language-Detection-using-Deep-Learning/blob/master/outputs/J_test.jpg)
![](https://github.com/sovit-123/American-Sign-Language-Detection-using-Deep-Learning/blob/master/outputs/S_test.jpg)

### Dataset Used

The dataset used can be found on [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet.).

### Specific Packages

* PyTorch >= 1.4.
* Albumentations >= 0.4.3.
* Scikit-Learn >= 0.22.1.



## <u>Using the Repository</u>

* Download the zip file

  ​			OR

* Clone the repository using: `git clone https://github.com/sovit-123/American-Sign-Language-Detection-using-Deep-Learning.git`.



## <u>Directory Structure and Usage</u>

* ```
  ├───input
  │   ├───asl_alphabet_test
  │   │   └───asl_alphabet_test
  │   ├───asl_alphabet_train
  │   │   └───asl_alphabet_train
  │   │       ├───A
  │   │       ├───B
  │   │       ...
  │   └───preprocessed_image
  │       ├───A
  │       ├───B
  │       ...
  ├───outputs
  └───src
  │   cam_test.py
  │   cnn_models.py
  │   create_csv.py
  │   preprocess_image.py
  │   test.py
  │   train.py
  ```

* Be sure to make a folder named `input` first. This is where all the image data will reside.

* `input` folder contains the the original data from the [Kaggle website](https://www.kaggle.com/grassknoted/asl-alphabet) as well as the preprocessed images that are used for training.
* `input/preprocessed_image` contains the resized images that are used for training. The total images in the original dataset is 87000. The `input/preprocessed_image` may contain 87000 or a subset of images depending upon the number of images preprocessed. These many images will be used for training.
* `outputs` folder contains the trained model (`model.pth`), the loss and accuracy plots, the predicted test images, and the saved webcam feed with the predicted output.
* `src` folder contains the different python files.
  * `preprocess_image.py`: Preprocess the number of images that you want to use for training.
  * `create_csv.py`: Create a CSV file for the preprocessed images mapping the image paths to the labels. All the images are read from disk during training.
  * `cnn_models.py`: Contains the modules of Custom convolutional neural network model to be used during training. Can be expanded with different module. Keeping this file separate provides easier usage of different models during training.
  * `train.py`: Python file to train the CNN model on the dataset.
  * `test.py`: Python file to test on the images provided in `input/asl_alphabet_test/asl_alphabet_test` folder.
  * `cam_test.py`: Python file for real time webcam sign language detection (**The major aim of this project**). 

### Using The Different Python Files (In Order)

* **Execute all the files in the terminal while being within the `src` folder.**

* `preprocess_image.py`: 

  `python preprocess_image.py --num-images 1200`

  `--num_images` is the number of images to preprocess for each category from `A` to `Z`, including `del`, `nothing`, and `space`.

* `create_csv.py`:

  `python create_csv.py`

* `train.py`:

  `python train.py --epochs 10`

* `test.py`:

  `python test.py --img A_test.jpg`

* `cam_test.py`:

  `python cam_test.py `



## <u>References</u>

* Kaggle dataset:
  
  * https://www.kaggle.com/grassknoted/asl-alphabet.
* [Changing the contrast and brightness of an image!](https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html).
* [Real-time American Sign Language Recognition with Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/214_Report.pdf), **Brandon Garcia et al.**

  

  

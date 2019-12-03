
Main purpose of this repository is to gather some image synthesis related GANs and prepare proper database for further projects may be required later. It may be useful for different type of datasets.  

# PGAN  
Synthesis Faces using Progressive Growing GANs

## Related Papers 

* ### "Progressive Growing of GANs for Improved Quality, Stability and Variation" 2018 [(link)](https://arxiv.org/pdf/1710.10196.pdf)
  The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, add new layers that model increasingly fine details as training progresses. Allowing to produce HQ images up to 1024 x 1024 pixels.
  Keras implementation was used to based on Jason Brownlee's [article](https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/).


## Python requirements
This code was tested on 
* Python 3.7
* TensorFlow 1.15
* Anaconda 2019.10

--------
## Preparing Training Set 
* ### CelebA: Large-scale Celebrity Faces Attributes Dataset [(link)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  CelebA is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. 
  Can be directly download from kaggle [(link)](https://www.kaggle.com/jessicali9530/celeba-dataset). After download unzip folder in the same direction with **preprocess.py**. 
----------
  Before training, All faces were detected and cropped using implementation provided by [ipazc/mtcnn](https://github.com/ipazc/mtcnn) project. This implementation is based on: **"Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" (MTCNN) [(link)](https://arxiv.org/abs/1604.02878)**

Can also be installed via pip as follows:  
  
``pip install mtcnn ``  

**preprocess.py:** The task of this python file is to detect face on the image and crop that area and resize it to 128 x 128 pixels (rgb image) and store inside "img_align_celeba_128.npz" file (convert and store images as numpy array format), so it is possible to load .npz file and train model using npz file.

-------

## Training Networks
 Explanation will come later here..
 
 
 
 -------
 ## Analyzing Results 
 According the original article. Authors suggested several ways to analyzed:
 * **Manual inspection:**
 
 * **TensorBoard:** The training script also exports various running statistics in a *.tfevents file that can be visualized in TensorBoard with tensorboard --logdir <result_subdir>. It is not avaliable in this code but will try to implement.
 
 * **Quality metrics:** Sliced Wasserstein distance, Fréchet inception distance, etc.
 
 

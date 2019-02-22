# Machine_Learning
Face Classification Project

The purpose of this project is to perform face image classification using a Bayes classifier and a K-NN classifier.
For this purpose, there is a data set that I divided into training and test samples. 
In this Project, only the data set named “data.mat” was used. It is a 3-dimensional array consisting of 600 face images.
From the start of the data set, every succession of 3 images belongs to a different person. Each of the 3 images are different. 
The first image is a neutral face, the second image is a face with happy facial expression, and the third image has illumination variations. 

Data Pre-processing

The data file contained images of 200 subjects. Each subjects has 3 face images.
The image have size 24 x 21. So the overall size of the data set was 24 x 21 x 600.
We reshaped the data into a single matrix of size 504 x 600 using the reshape function in MATLAB.
Each column of the new matrix contains an image. For example, the first 3 columns of the matrix contain images of person 1. 
The values in the matrix were then scaled to values between zero and 1. We used this new matrix as our data set for all experiments. 
In the first part of the project only the 3rd images of each class are used as test samples. They correspond to columns with indexes that
are multiples of 3. In the second part, only the last 12 columns are used as test samples and the remaining as training data.  


Files: pca_bayes_classifier.m, lda_bayes_classifier.m, pca_knn_classifier.m, lda_knn_classifier.m

To run the above files, copy the "data.m" file to the chosen folder. Then open the corresponding file and click play button in Matlab.
The test images are the 3rd images of each class. These correspond to column 3, 6 , 9, ...600. Enter one of these indexes
at the prompt.

### Instructions

- This notebook has been guided by the following [https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78]

### Description
- pratice-train.ipynb, we will train a face recognition model on a set of images. Step by step, we will:
    - Detect faces in the images with cascade classifier in OpenCV.
    - Align the faces using a landmark detection model in dlib (shape predictor), aligned by center of eyes.
    - Normalize the faces 0-1 from divided by 255.
    - Use PCA to reduce the dimensionality of the faces, and train a Support Vector Machine (SVM) to recognize the faces.
    - Save the trained model to disk.

##### Attention: Before training the model, we can use facenet to extract the features of the faces replacing the PCA. The model will be trained with the features extracted by facenet.

- In face_process.py, we will create a class to process images:
    - Detect faces in the images with cascade classifier in OpenCV.
    - Align the faces using a landmark detection model in dlib (shape predictor), aligned by center of eyes.
    - Drop black padding from the aligned faces.

- In predict-new.ipynb, we will use the trained model to predict faces in a new image.

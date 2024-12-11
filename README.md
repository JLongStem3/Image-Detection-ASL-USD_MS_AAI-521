# Image-Detection-ASL-USD_MS_AAI-521
In this project we use computer vision to perform Image Detection for American Sign Language
The dataset used in this project is available from Kaggle at: https://www.kaggle.com/datasets/datamunge/sign-language-mnist 

## Files:
_Image_Classification_on_American_Sign_Language.ipynb_: Consolidated Project File

*MobileNetV3_FineTuning.ipynb:* Model training and fine-tuning

*Evalutation_Metrics_for_model_01_keras.ipynb:* Model evaluation metrics

*app.py*: App code from HuggingFace for live demo hosting
(Access demo on huggingface at: https://huggingface.co/spaces/kdevoe/ASL_MobileNetV3) 

### Introduction
The objective of our Computer Vision project is to develop an image detection model that can recognize hand gestures from the American Sign Language (ASL) alphabet. We utilize the ASL MNIST dataset, which is an extension of the widely-known MNIST handwritten digit dataset; a popular benchmark for image-based machine learning methods with known challenges for image classification. The ASL MNIST dataset represents 24 letters of the ASL alphabet (A-Y), excluding the letters “J” and “Z”, which involve motion-based gestures. The ASL classification task expands accessibility and has significant real-world applications that aid in communication for the hearing and speech-impaired communities. Including different hand positions, orientations, and lighting conditions, the dataset is complex and is great for testing computer vision models.

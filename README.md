Landslide Prediction and Mapping Using Deep Learning

This project focuses on predicting and mapping landslide-prone areas using a deep learning-based approach, motivated by the need for effective disaster mitigation in regions like Uttarakhand and Mizoram. Leveraging multi-modal satellite and terrain data, I developed a system that integrates Sentinel-2 multispectral imagery, topographic inputs from ALOS PALSAR, and vegetation indices such as NDVI.
Key Features:

Data Preprocessing: The dataset was processed into 128×128 pixel patches (10 m resolution) with pixel-wise segmentation masks and image-level classifications.
Deep Learning Pipeline:
U-Net Architecture: Used for semantic segmentation to identify landslide zones with a segmentation accuracy of 98.8%.
CNN Classifier: Implemented for binary image-level classification, achieving an accuracy of 89.5% with balanced precision, recall, and F1 scores.
Technologies Used:

Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib
Development and training conducted on Google Colab with GPU acceleration
This project not only enhanced my skills in remote sensing data and deep learning pipelines but also demonstrated the impactful role of AI in disaster preparedness and risk reduction.

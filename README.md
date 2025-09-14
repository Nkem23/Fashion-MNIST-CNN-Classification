## Fashion MNIST CNN Classification

##  Assignment Description  
This project implements a **Convolutional Neural Network (CNN)** in **both Python (Keras)** and **R (Keras)** to classify images from the **Fashion MNIST dataset**.  

The assignment requirements were:  
1. **Develop a CNN with six layers** using Keras in Python and R.  
2. **Train the model** on the Fashion MNIST dataset.  
3. **Make predictions** for at least two test images.  
4. **Save the model** and provide outputs showing predictions.  
5. **Submit code, saved models, prediction figures, and this README.**  

---

## 📂 Repository Contents  
- `fashion_cnn_notebook.py` → Python implementation.  
- `fashion_cnn_notebook.r` → R implementation (executed in Google Colab).  
- `fashion_cnn_notebook.keras` → Saved CNN model in Python.
- `fashion_cnn.keras` → Saved CNN model in R. 
- `predictions.png` → Prediction output from Python (with predicted + true labels).  
- `predictions_R.png` → Prediction output from R (with predicted + true labels).  
- `README.md` → Documentation and instructions (this file).  

---

##  Environment Setup  

### 🔹 Python (Jupyter Notebook)  
1. Install TensorFlow and Keras before running:
2. Open fashion_cnn_notebook.py
3. The script will:

       Load the dataset

       Train the CNN (6 layers)

       Save the model (fashion_cnn_notebook.keras)

       Generate predictions for 2+ test images

### 🔹 R (Google Colab) 
1. Install tensorflow and keras in R:

      - install.packages("keras")

      - install.packages("tensorflow")

      - library(keras)

      - library(tensorflow)

      - install_tensorflow()

2. Open fashion_cnn_notebook.R in Colab or RStudio.

3. The script will:

       Train a CNN with 6 layers

       Save the model (fashion_cnn_notebook.keras)

       Generate predictions for 2+ test images

##  Results
- The CNN successfully trained on Fashion MNIST with high test accuracy (>85%).

- Predictions were generated for at least 2 images.

## 📝 Notes
- Prediction Output: Both predicted labels and ground truth labels are shown for verification.

- Environments Used:

      Python: Jupyter Notebook (local, with TensorFlow installed).

      R: Google Colab (with TensorFlow installed before running).

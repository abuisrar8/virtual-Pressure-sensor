# Predictive Model with Ensemble Learning and Moving Average Features

## Problem statement
### Virtual Brake Pressure Sensor

![pressure_sensor](https://github.com/user-attachments/assets/ea5e9267-5704-428b-9594-7357a910f345)

A virtual sensor is a software-based model of a physical sensor that can simulate its behavior and generate sensor readings without the need for actual physical hardware.

Safety functions like braking rely heavily on accurate sensor data. The failure of a sensor failure leads to degradation or non-availability of safety features. In such a scenario, a virtual sensor could provide a degree of redundancy. A very accurate virtual sensor could even save hardware costs.


## Model Description

This repository contains a machine learning pipeline that combines XGBoost and Artificial Neural Networks (ANN) to predict the target variable, PressSent1X1, representing a critical metric in automotive systems. The project is designed to handle large datasets, preprocess features, train models, and evaluate performance metrics. Key steps are as follows:

### Features
Moving Average Features: Applies a moving average to smooth input features, reducing noise and improving model robustness.
Ensemble Learning: Combines XGBoost predictions as an input feature for the ANN to enhance predictive accuracy.
Delay Matrix Features: Introduces delayed versions of input features to account for temporal relationships in the data.
Data Scaling: Ensures all features are standardized for optimal performance during training.
Cross-Validation: Implements fold-based evaluation to validate model performance across different subsets of data.
### Pipeline Overview  
![design](https://github.com/user-attachments/assets/5abc0c76-036e-4e2c-8081-a0cc225f2f0f)

  1. GPU Configuration
The script configures GPUs for training to optimize performance. It uses TensorFlow's set_memory_growth to manage GPU memory allocation dynamically.

  2. Data Preprocessing
Data Loading: Reads input files from the specified directory.
Moving Average Transformation: Smoothens features using a moving average filter.
Feature Filtering: Selects relevant features for training.
Feature Scaling: Standardizes features and target variables using StandardScaler.
  3. XGBoost Training
Model Training: Trains an XGBoost regressor on the selected features.
Feature Augmentation: Adds XGBoost predictions as a new feature for subsequent ANN training.
  4. ANN Training
Input Features: Combines delayed and non-delayed features for training.
Model Definition: A feedforward neural network defined in model.models.Neural_Network_FeedForward_Regularizer.
Training Process: Trains the ANN on the augmented dataset, including XGBoost predictions.
  5. Evaluation
Metrics: Evaluates the models using R², RMSE, MAE, and maximum absolute error.
Threshold R² Score: Introduces a custom metric to prioritize scores above 0.9.
Cross-Validation: Saves metrics and predictions for both training and testing datasets across folds.
  6. Output
Results Storage: Saves results, predictions, and evaluation metrics for each fold in dedicated directories.
Visualization: Generates plots to visualize model performance.

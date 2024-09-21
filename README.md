# **Cancer Mortality Prediction**

### **Student:** Adonai Vera  
### **Course:** CS 5173/6073: Deep Learning  
### **Instructor:** Dr. Jun Bai  

---

## **Project Overview**

This project focuses on predicting cancer mortality rates using a variety of machine learning models, including **Linear Regression**, **Deep Neural Networks (DNN)**, and various state-of-the-art ensemble models. The dataset includes several demographic and healthcare-related features, which are used to predict the `TARGET_deathRate`, representing cancer mortality in different regions.

---

## **Dataset**

The dataset is located at `data/cancer_reg.csv`. It contains various features such as demographic, socioeconomic, and healthcare statistics.

---

## **Project Steps**

1. **Exploratory Data Analysis (EDA)**:
   - Performed initial EDA to understand the dataset, check for missing values, and visualize key features.
   - Code is available in `utils/exploration_data_analysis.py`.

2. **Data Preprocessing**:
   - Handle missing values.
   - Log transformation for skewed features.
   - Normalize features using `MinMaxScaler`.
   - Split the data into training, validation, and test sets.
   - Optional PCA for dimensionality reduction.
   - Preprocessing code can be found in `utils/data_preprocessing.py`.

3. **Model Training**:
   The following models were trained and compared:
   - **Linear Regression Model**: A basic linear regression using Scikit-learn.
   - **Deep Neural Networks (DNNs)**:
     - Various architectures such as DNN-[16], DNN-[30, 8], DNN-[30, 16, 8].
     - **Robust DNN**: A more advanced architecture with 4 layers: [128, 64, 32, 16] and techniques like batch normalization, dropout, and L2 regularization.
   - **Ensemble Models**:
     - **XGBoost**: Extreme Gradient Boosting.
     - **CatBoost**: Categorical Boosting.
     - **Random Forest**: Ensemble of decision trees.
     - **LightGBM**: Gradient Boosting framework using tree-based learning algorithms.
     - **TabNet**: A deep learning-based tabular data model.

4. **Evaluation**:
   - Performance metrics: **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared (R²)**.
   - Plots for training and validation loss (for DNNs) and actual vs. predicted values (for Linear Regression and other models).
   
---

## **Setup Instructions**

### **Prerequisites**

- Python 3.8 or higher.
- Required libraries listed in `requirements.txt`.

### **Installing Dependencies**

Run the following command to install all necessary packages:

```bash
pip install -r requirements.txt
```

### **Running the Project**

To run the project and train the models, execute:

```bash
python main.py
```

This script will:
- Load the dataset.
- Preprocess the data.
- Train the **Linear Regression**, **DNN**, and **Ensemble models**.
- Evaluate all models and print the performance metrics.

---

## **Testing Models**

To test the saved models, run:

```bash
python test_model.py
```

This script will:
- Load the **Robust DNN** and **Linear Regression** models.
- Preprocess the dataset.
- Generate predictions and output performance metrics.

---

## **Project Structure**

```
.
├── README.md                      # This file
├── data                           # Contains the dataset
│   └── cancer_reg.csv
├── models                         # Model scripts
│   ├── weight                     # Folder for saved model weights
│   ├── dnn_robust_model.py        # Robust DNN architecture
│   ├── linear_regression.py       # Linear Regression model
│   ├── xgboost_model.py           # XGBoost model
│   ├── catboost_model.py          # CatBoost model
│   ├── random_forest_regression.py# Random Forest model
│   ├── lightgbm_model.py          # LightGBM model
│   └── tab_net_regressor.py       # TabNet model
├── utils                          # Utility functions
│   ├── data_preprocessing.py      # Preprocessing script
│   └── exploration_data_analysis.py# Script for EDA
├── test_model.py                  # Script to test saved models
├── main.py                        # Main script to run the project
├── requirements.txt               # List of dependencies
├── figures                        # Folder containing generated plots
│   └── *.png
```

---

## **Model Details**

### **Linear Regression**:
- A simple linear model using Scikit-learn.
- **Performance**:
  - MSE: 0.0020
  - MAE: 0.0338
  - R²: 0.7808

### **Deep Neural Networks (DNNs)**:
- Multiple architectures were tested. The **Robust DNN** outperformed the others with 4 hidden layers [128, 64, 32, 16], batch normalization, dropout, and L2 regularization.
- **Performance**:
  - MSE: 0.0013
  - R²: 0.8601

### **XGBoost**:
- Boosting algorithm using gradient boosting on decision trees.
- **Performance**:
  - MSE: 0.0031
  - MAE: 0.0413
  - R²: 0.6632

### **CatBoost**:
- Boosting algorithm designed for categorical features.
- **Performance**:
  - MSE: 0.0024
  - MAE: 0.0366
  - R²: 0.7397

### **Random Forest**:
- Ensemble method combining decision trees for regression.
- **Performance**:
  - MSE: 0.0040
  - MAE: 0.0476
  - R²: 0.5587

### **LightGBM**:
- Gradient boosting framework that uses tree-based learning algorithms.
- **Performance**:
  - MSE: 0.0027
  - MAE: 0.0385
  - R²: 0.7009

### **TabNet**:
- Deep learning-based model for tabular data.
- **Performance**:
  - MSE: 0.6078
  - MAE: 0.5612
  - R²: 0.4108

---

## **Model Comparison**

| Model                  | MSE     | MAE     | R²      |
|------------------------|---------|---------|---------|
| Linear Regression       | 0.0020  | 0.0338  | 0.7808  |
| DNN-30-16-8             | 0.0026  | N/A     | 0.7127  |
| Robust DNN              | 0.0013  | N/A     | 0.8601  |
| XGBoost                 | 0.0031  | 0.0413  | 0.6632  |
| CatBoost                | 0.0024  | 0.0366  | 0.7397  |
| Random Forest           | 0.0040  | 0.0476  | 0.5587  |
| LightGBM                | 0.0027  | 0.0385  | 0.7009  |
| TabNet                  | 0.6078  | 0.5612  | 0.4108  |

---

## **Conclusion**

The **Robust DNN** model significantly outperformed the other models, achieving the lowest MSE (0.0013) and the highest R² (0.8601). The **Linear Regression** model also performed well but did not match the performance of the DNN. Ensemble models like **XGBoost** and **CatBoost** performed reasonably well, while the **TabNet** model did not perform as expected in this case. With further tuning, particularly of the DNN and ensemble models, there is potential for even better performance.

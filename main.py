from utils.data_preprocessing import load_and_preprocess_data
from models.linear_regression import train_linear_regression
from models.dnn_model import train_dnn
from models.catboost_model import train_catboost
from models.lightgbm_model import train_lightgbm
from models.xgboost_model import train_xgboost
from models.random_forest_regression import train_randomforest
from models.tab_net_regressor import train_tabnet
from models.dnn_robust_model import train_robust_dnn
import numpy as np
import datetime


def model_general(X_train, X_val, X_test, y_train, y_val, y_test):
    # Step 2: Train and evaluate the Linear Regression model
    print("\n--- Linear Regression Model ---")
    train_linear_regression(X_train, y_train, X_test, y_test)
    
    # Step 3: Train and evaluate Deep Neural Network models with different architectures, no multiple learning rates, optimizer SGD and loss function MSE.
    print("\n--- DNN Model: DNN-16 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[16],  learning_rates=[], optimizer='adam', loss_function='mse')  # DNN-16
    
    print("\n--- DNN Model: DNN-30-8 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 8], learning_rates=[], optimizer='adam', loss_function='mse')  # DNN-30-8
    
    print("\n--- DNN Model: DNN-30-16-8 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 16, 8], learning_rates=[], optimizer='adam', loss_function='mse')  # DNN-30-16-8
    
    print("\n--- DNN Model: DNN-30-16-8-4 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 16, 8, 4], learning_rates=[], optimizer='adam', loss_function='mse')  # DNN-30-16-8-4


def models_multiple_lr(X_train, X_val, X_test, y_train, y_val, y_test):
    # Step 2: Train and evaluate the Linear Regression model
    print("\n--- Linear Regression Model ---")
    train_linear_regression(X_train, y_train, X_test, y_test)
    
    # Step 3: Train and evaluate Deep Neural Network models with different architectures, no multiple learning rates, optimizer SGD and loss function MSE.
    print("\n--- DNN Model: DNN-16 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[16], optimizer='adam', loss_function='mse')  # DNN-16
    
    print("\n--- DNN Model: DNN-30-8 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 8], optimizer='adam', loss_function='mse')  # DNN-30-8
    
    print("\n--- DNN Model: DNN-30-16-8 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 16, 8], optimizer='adam', loss_function='mse')  # DNN-30-16-8
    
    print("\n--- DNN Model: DNN-30-16-8-4 ---")
    train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 16, 8, 4], optimizer='adam', loss_function='mse')  # DNN-30-16-8-4


def main():
    """
    Main function to train and evaluate both linear regression and DNN models.
    """
    
    # Step 1: Load and preprocess the dataset
    print("Loading and preprocessing the data...")
    print("Time Right now {}".format(datetime.datetime.now()))
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(apply_pca=False, n_components=0.7)

    # Base Models
    model_general(X_train, X_val, X_test, y_train, y_val, y_test)

    # Models Dynamics Learning Rate
    models_multiple_lr(X_train, X_val, X_test, y_train, y_val, y_test)

    # Catboost Model
    train_catboost(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # XGBoost Model
    train_xgboost(X_train, y_train, X_test, y_test)

    # Lightgbm Model
    train_lightgbm(X_train, y_train, X_test, y_test)

    # Random Forest Model
    train_randomforest(X_train, y_train, X_test, y_test)

    # Tab Net Model
    train_tabnet(X_train, X_val, X_test, y_train, y_val, y_test)

    # Robust Neural Network
    train_robust_dnn(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()

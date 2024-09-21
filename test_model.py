from utils.data_preprocessing import load_and_preprocess_data
from models.dnn_robust_model import load_and_predict_robust_dnn
from models.linear_regression import load_and_predict_linear_regression

def main():
    """
    Main function to load data, preprocess it, and test both the Robust DNN and Linear Regression models.
    """
    print("Starting the model evaluation process...\n")

    # Step 1: Load and preprocess the data
    print("Loading and preprocessing the data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(apply_pca=False, n_components=0.7)
    print("Data successfully loaded and preprocessed.\n")

    # Step 2: Test the Robust DNN model
    print("Testing the Robust DNN Model:")
    print("Loading the Robust DNN model and making predictions...")
    load_and_predict_robust_dnn(X_test, y_test, model_path='models/weight/dnn_robust_model.keras')
    print("Robust DNN Model testing completed.\n")

    # Step 3: Test the Linear Regression model
    print("Testing the Linear Regression Model:")
    print("Loading the Linear Regression model and making predictions...")
    load_and_predict_linear_regression(X_test, y_test, model_path='models/weight/linear_regression_model.pkl')
    print("Linear Regression Model testing completed.\n")

    print("Model evaluation process completed.")

if __name__ == "__main__":
    main()

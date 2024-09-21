import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time

def save_linear_regression_plot(y_test, y_pred, folder='figures'):
    """
    Saves the linear regression model's performance plot (True vs Predicted values).
    """
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Create the plot
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')  # Scatter plot for predicted vs actual values
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r')  # Diagonal line (ideal prediction)
    plt.title(f'Linear Regression Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    # Save the plot in the specified folder
    filepath = os.path.join(folder, 'linear_regression_predictions.png')
    plt.savefig(filepath)
    plt.close()  # Close the plot to avoid display in the notebook
    
    print(f"Linear regression plot saved as {filepath}")


def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Trains a linear regression model and evaluates it on the test set.
    """
    
    # Get the current time and print training start information
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training started at {start_time}")
    
    # Initialize the linear regression model
    model = LinearRegression()
    
    # Print a message indicating the training step
    print("Training the Linear Regression model...")

    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Save the trained model to a file
    joblib.dump(model, 'models/weight/linear_regression_model.pkl')
    
    # Print a message indicating the training is complete
    print("Training complete. Model saved as 'linear_regression_model.pkl'")
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Output the performance metrics
    print(f"Linear Regression Model Mean Squared Error: {mse:.4f}")
    print(f"Linear Regression Model Mean Absolyte Error: {mae:.4f}")
    print(f"Linear Regression Model R-squared: {r2:.4f}")

    # Save the prediction plot
    save_linear_regression_plot(y_test, y_pred)

    return model

# New function to load and predict using the saved linear regression model
def load_and_predict_linear_regression(X_test, y_test, model_path='models/weight/linear_regression_model.pkl'):
    """
    Loads the pre-trained linear regression model and makes predictions on the test set.
    
    Parameters:
    X_test: np.array, test features
    y_test: np.array, test labels
    model_path: str, path to the saved model's weights
    
    Returns:
    y_pred: np.array, predicted values for the test set
    """
    
    # Load the saved model
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Loaded Linear Regression Model Mean Squared Error: {mse:.4f}")
    print(f"Loaded Linear Regression Model Mean Absolyte Error: {mae:.4f}")
    print(f"Loaded Linear Regression Model R-squared: {r2:.4f}")
    
    return y_pred

import catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

def save_loss_plot_catboost(model, folder='figures'):
    """
    Saves the CatBoost training and validation loss over iterations as a plot in the specified folder.
    
    Parameters:
    model: CatBoost model object after training
    folder: str, directory where the plot should be saved
    """
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Extract error history from CatBoost model
    iterations = list(range(1, len(model.get_evals_result()['learn']['RMSE']) + 1))
    train_loss = model.get_evals_result()['learn']['RMSE']
    val_loss = model.get_evals_result()['validation']['RMSE']
    
    # Create the plot
    plt.figure()
    plt.plot(iterations, train_loss, label='Training RMSE')
    plt.plot(iterations, val_loss, label='Validation RMSE')
    plt.title('CatBoost Training vs Validation Loss')
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')
    plt.legend(loc='upper right')
    
    # Save the plot
    filepath = os.path.join(folder, 'catboost_loss_plot.png')
    plt.savefig(filepath)
    plt.close()
    
    print(f"Loss plot saved as {filepath}")

def train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, params=None):
    """
    Trains and evaluates a CatBoostRegressor model on the given data and plots the loss.
    
    Parameters:
    X_train, X_val, X_test: Training, validation, and test feature sets
    y_train, y_val, y_test: Training, validation, and test target sets
    params: dict, hyperparameters for the CatBoostRegressor model (optional)
    
    Returns:
    model: The trained CatBoost model
    """
    
    # Step 1: Initialize CatBoostRegressor with default or provided params
    if params is None:
        params = {
            'random_strength': 1, 
            'n_estimators': 1000,
            'max_depth': 7, 
            'loss_function': 'RMSE', 
            'learning_rate': 0.1,  
            'colsample_bylevel': 0.8,
            'bootstrap_type': 'MVS', 
            'bagging_temperature': 1.0
        }
    
    model = CatBoostRegressor(**params)

    # Step 2: Train the model
    print("Training CatBoost model...")
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=1, plot=False)
    #model.fit(X_train, y_train, verbose=1, plot=False)
    
    # Step 3: Plot loss during training
    save_loss_plot_catboost(model)
    
    # Step 4: Evaluate the model on the test set
    print("Evaluating the CatBoost model...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    return model

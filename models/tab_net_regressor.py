import numpy as np
import time
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import matplotlib.pyplot as plt

def save_tabnet_plot(y_test, y_pred, folder='figures'):
    """
    Saves the TabNet model's performance plot (True vs Predicted values).
    
    Parameters:
    y_test: np.array, true labels
    y_pred: np.array, predicted labels
    folder: str, directory where the plot should be saved
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r')
    plt.title('TabNet Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    filepath = os.path.join(folder, 'tabnet_predictions.png')
    plt.savefig(filepath)
    plt.close()
    
    print(f"TabNet plot saved as {filepath}")


def train_tabnet(X_train, X_val, X_test, y_train, y_val, y_test, params=None, seed=42):
    """
    Trains a TabNet model and evaluates it on the test set.
    
    Parameters:
    X_train, y_train: np.array, training data
    X_val, y_val: np.array, validation data
    X_test, y_test: np.array, test data
    params: dict, hyperparameters for TabNet
    seed: int, seed for reproducibility
    
    Returns:
    model: The trained TabNet model
    """
    
    np.random.seed(seed)
    
    if params is None:
        params = {
            'n_d': 8, 'n_a': 8, 'n_steps': 3,
            'gamma': 1.5, 'lambda_sparse': 1e-3,
            'optimizer_params': dict(lr=2e-2),
            'mask_type': 'entmax',  # "sparsemax"
            'scheduler_params': {"step_size": 50, "gamma": 0.9},
            'scheduler_fn': None,
            'seed': seed
        }
    
    # Initialize TabNet model
    model = TabNetRegressor(**params)
    
    # Train the model
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training TabNet started at {start_time}")
    
    # Convert pandas DataFrames to numpy arrays for TabNet
    X_train_np = X_train.values
    X_val_np = X_val.values
    X_test_np = X_test.values

    y_train_np = y_train.values.reshape(-1, 1)
    y_val_np = y_val.values.reshape(-1, 1)
    y_test_np = y_test.values.reshape(-1, 1)

    # Now fit the model using the NumPy arrays
    model.fit(
        X_train=X_train_np, y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_metric=['rmse'],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128
    )
    
    # Save the trained model
    model.save_model('models/weight/tabnet_model')
    
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training complete. Model saved at 'models/weight/tabnet_model'")
    
    # Predict on the test set
    y_pred = model.predict(X_test_np)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_np, y_pred)
    mae = mean_absolute_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)
    
    # Output the performance metrics
    print(f"TabNet Model Mean Squared Error: {mse:.4f}")
    print(f"TabNet Model Mean Absolute Error: {mae:.4f}")
    print(f"TabNet Model R-squared: {r2:.4f}")
    
    # Save the prediction plot
    save_tabnet_plot(y_test, y_pred)
    
    return model

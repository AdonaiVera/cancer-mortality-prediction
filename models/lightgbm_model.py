import lightgbm as lgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import joblib

def save_lightgbm_plot(y_test, y_pred, folder='figures'):
    """
    Saves the LightGBM model's performance plot (True vs Predicted values).
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r')
    plt.title('LightGBM Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    filepath = os.path.join(folder, 'lightgbm_predictions.png')
    plt.savefig(filepath)
    plt.close()
    print(f"LightGBM plot saved as {filepath}")

def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Trains a LightGBM model and evaluates it on the test set.
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training started at {start_time}")
    
    model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.01, n_estimators=1000)
    
    print("Training the LightGBM model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    joblib.dump(model, 'models/weight/lightgbm_model.pkl')
    print("Training complete. Model saved as 'lightgbm_model.pkl'")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"LightGBM Model Mean Squared Error: {mse:.4f}")
    print(f"LightGBM Model Mean Absolute Error: {mae:.4f}")
    print(f"LightGBM Model R-squared: {r2:.4f}")
    
    save_lightgbm_plot(y_test, y_pred)
    
    return model

import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import joblib

def save_xgboost_plot(y_test, y_pred, folder='figures'):
    """
    Saves the XGBoost model's performance plot (True vs Predicted values).
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r')
    plt.title('XGBoost Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    filepath = os.path.join(folder, 'xgboost_predictions.png')
    plt.savefig(filepath)
    plt.close()
    print(f"XGBoost plot saved as {filepath}")

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model and evaluates it on the test set.
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training started at {start_time}")
    
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, subsample=0.8, colsample_bytree=0.8)
    
    print("Training the XGBoost model...")
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/weight/xgboost_model.pkl')
    print("Training complete. Model saved as 'xgboost_model.pkl'")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost Model Mean Squared Error: {mse:.4f}")
    print(f"XGBoost Model Mean Absolute Error: {mae:.4f}")
    print(f"XGBoost Model R-squared: {r2:.4f}")
    
    save_xgboost_plot(y_test, y_pred)
    
    return model

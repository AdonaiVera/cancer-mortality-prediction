import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to save the loss plot
def save_loss_plot(history, lr, layer_structure, folder='figures'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.figure()
    plt.plot(history.history['loss'], label='training set')
    plt.plot(history.history['val_loss'], label='validation set')
    plt.title(f'Model Loss (LR={lr}, Structure={layer_structure})')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    filename = f'dnn_model_{layer_structure}_lr{lr}.png'
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)
    plt.close()

# Function to build the robust DNN model
def build_robust_dnn_model(input_shape, layer_structure=[128, 64, 32, 16], 
                           dropout_rate=0.3, regularizer=0.001):
    model = tf.keras.Sequential()
    
    # Input Layer
    model.add(layers.InputLayer(shape=input_shape))
    
    # Hidden Layers with Regularization, Batch Normalization, and Dropout
    for nodes in layer_structure:
        model.add(layers.Dense(nodes, activation='leaky_relu', 
                               kernel_regularizer=regularizers.l2(regularizer)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Output Layer
    model.add(layers.Dense(1))  # For regression
    
    # AdamW optimizer with weight decay and gradient clipping
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    
    return model

# Function to train the robust DNN
def train_robust_dnn(X_train, X_val, X_test, y_train, y_val, y_test,
                     layer_structure=[128, 64, 32, 16], dropout_rate=0.3, 
                     regularizer=0.001, epochs=200, patience=10):
    input_shape = (X_train.shape[1],)
    
    # Build the model
    model = build_robust_dnn_model(input_shape, layer_structure, dropout_rate, regularizer)
    
    # Early Stopping and Learning Rate Scheduler
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, 
                        callbacks=[early_stopping, lr_scheduler], verbose=1)
    
    # Save loss plot
    save_loss_plot(history, lr=0.001, layer_structure=layer_structure)
    
    # Predict on test set
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model.save(f'models/weight/dnn_robust_model.keras')
    
    print(f"Robust DNN Model Mean Squared Error: {mse:.4f}")
    print(f"Robust DNN Model R-squared: {r2:.4f}")
    
    return model, history

# New function to load and predict using the saved robust DNN model
def load_and_predict_robust_dnn(X_test, y_test, model_path='models/weight/dnn_robust_model.keras'):
    """
    Loads the pre-trained robust DNN model and makes predictions on the test set.
    
    Parameters:
    X_test: np.array, test features
    y_test: np.array, test labels
    model_path: str, path to the pre-trained model's weights
    
    Returns:
    y_pred: np.array, predicted values for the test set
    """
    
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate Mean Squared Error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Loaded Robust DNN Model Mean Squared Error: {mse:.4f}")
    print(f"Loaded Robust DNN Model R-squared: {r2:.4f}")
    
    return y_pred

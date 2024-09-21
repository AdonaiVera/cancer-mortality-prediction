import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os


def save_loss_plot(history, lr, layer_structure, folder='figures'):
    """
    Saves the training and validation loss over epochs as a plot in the specified folder.
    
    Parameters:
    history: Keras History object, containing loss and validation loss
    lr: float, learning rate used for training
    layer_structure: list, structure of the neural network layers
    folder: str, directory where the plot should be saved
    """
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Create the plot
    plt.figure()
    plt.plot(history.history['loss'], label='training set')
    plt.plot(history.history['val_loss'], label='validation set')
    plt.title(f'Model Loss (LR={lr}, Structure={layer_structure})')
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    # Create a filename based on layer structure and learning rate
    filename = f'dnn_model_{layer_structure}_lr{lr}.png'
    
    # Save the plot in the specified folder
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)
    plt.close()  # Close the plot to avoid display in the notebook
    
    print(f"Loss plot saved as {filepath}")

def build_simple_dnn_model(layer_structure, input_shape, optimizer, loss_function):
    """
    Builds a deep neural network model dynamically based on the given layer structure.
    
    Parameters:
    layer_structure: list of int, where each element represents the number of nodes in a hidden layer.
    input_shape: tuple, shape of the input data
    optimizer: str, the optimizer to use for training the model
    loss_function: str, the loss function to use for the model
    
    Returns:
    model: A compiled TensorFlow Keras model.
    """
    
    model = tf.keras.Sequential()

    print(input_shape)
    # Input layer with the correct input shape
    model.add(tf.keras.layers.InputLayer(shape=input_shape))  
    
    # Add hidden layers based on the specified layer structure
    for nodes in layer_structure:
        model.add(tf.keras.layers.Dense(nodes))
    
    # Output layer (single output for regression)
    model.add(tf.keras.layers.Dense(1))  

    print(f"DNN Model {layer_structure} with learning rate fix")
    print(model.summary())
    
    # Compile the model with the dynamic optimizer and loss function
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae', 'mse'])
    
    return model

def train_dnn(X_train, y_train, X_val, y_val, X_test, y_test, layer_structure=[30, 16, 8], 
              learning_rates=[0.1, 0.01, 0.001, 0.0001], optimizer='adam', loss_function='mse', seed=42):
    """
    Trains a deep neural network model and evaluates it on the test set.
    
    Parameters:
    X_train: np.array, training features
    y_train: np.array, training labels (TARGET_deathRate)
    X_val: np.array, validation features
    y_val: np.array, validation labels (TARGET_deathRate)
    X_test: np.array, test features
    y_test: np.array, test labels (TARGET_deathRate)
    layer_structure: list of int, defining the number of nodes in each hidden layer
    learning_rates: list of float, learning rates to iterate over
    optimizer: str, the optimizer to use for training the model (default is 'sgd'), but we know that adam performs better.
    loss_function: str, the loss function to use (default is 'mse')
    seed: int, the seed for random number generation
    
    Returns:
    model: The trained Keras model.
    """
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Iterate over the different learning rates

    if learning_rates:
        for lr in learning_rates:
            print(f"\nTraining with learning rate: {lr}")
            
            # Create a dynamic optimizer with the specified learning rate
            if optimizer == 'adam':
                optimizer_fn = tf.keras.optimizers.Adam(learning_rate=lr)
            elif optimizer == 'sgd':
                optimizer_fn = tf.keras.optimizers.SGD(learning_rate=lr)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
            
            # Build the DNN model dynamically based on the layer structure
            input_shape = (X_train.shape[1],)  # Correctly specify the input shape
            
            model = build_simple_dnn_model(layer_structure, input_shape, optimizer_fn, loss_function)
            
            # Define early stopping based on validation loss
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Log the current timestamp
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{start_time}] Training the Deep Neural Network with structure: {layer_structure} and learning rate: {lr}")
            
            # Train the model with validation data and early stopping
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
                                verbose=1, callbacks=[early_stopping])
            
            # Graph the results
            save_loss_plot(history, lr, layer_structure)
            
            # Save the trained model to a file
            model.save(f'models/weight/dnn_model_{layer_structure}_lr{lr}.keras')
            
            # Log the training completion time
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{end_time}] Training complete for learning rate: {lr}. Model saved as 'dnn_model_{layer_structure}_lr{lr}.keras'")
            
            # Predict on the test set
            y_pred = model.predict(X_test).flatten()
            
            # Calculate Mean Squared Error and R-squared
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Output the performance metrics
            print(f"DNN Model {layer_structure} with learning rate {lr} - Mean Squared Error: {mse:.4f}")
            print(f"DNN Model {layer_structure} with learning rate {lr} - R-squared: {r2:.4f}")
    else:
        print(f"\nTraining without fix learning rate")
            
        # Create a dynamic optimizer with the specified learning rate
        if optimizer == 'adam':
            optimizer_fn = tf.keras.optimizers.Adam()
        elif optimizer == 'sgd':
            optimizer_fn = tf.keras.optimizers.SGD()
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Build the DNN model dynamically based on the layer structure
        input_shape = (X_train.shape[1],)  # Correctly specify the input shape
        model = build_simple_dnn_model(layer_structure, input_shape, optimizer_fn, loss_function)
        
        # Define early stopping based on validation loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        # Log the current timestamp
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{start_time}] Training the Deep Neural Network with structure: {layer_structure}")
        
        # Train the model with validation data and early stopping
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
                            verbose=1, callbacks=[early_stopping])
        
        # Graph the results 
        save_loss_plot(history, 0, layer_structure)

        # Save the trained model to a file
        model.save(f'models/weight/dnn_model_{layer_structure}.keras')
        
        # Log the training completion time
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{end_time}] Training complete for fix learning rate. Model saved as 'dnn_model_{layer_structure}.keras'")
        
        # Predict on the test set
        y_pred = model.predict(X_test).flatten()
        
        # Calculate Mean Squared Error and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Output the performance metrics
        print(f"DNN Model {layer_structure} with learning rate fix - Mean Squared Error: {mse:.4f}")
        print(f"DNN Model {layer_structure} with learning rate fix - R-squared: {r2:.4f}")
    
    return model

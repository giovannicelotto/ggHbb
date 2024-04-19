import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Define custom loss function
def custom_loss(y_true, y_pred):
    # Compute custom loss batch per batch
    s = K.sum(y_pred * y_true)
    
    # Compute b = sum_batch(y_pred * (1 - y_true))
    b = K.sum(y_pred * (1 - y_true))
    
    # Compute the loss using the provided formula
    loss = -K.square(s) / (s + b + K.epsilon())  # Adding epsilon to avoid division by zero
    
    return loss
    #return batch_loss

# Generate some random data for demonstration
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.rand(100, 1)   # Regression targets

# Define the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Compile the model with custom loss
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

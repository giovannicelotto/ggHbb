# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# %%
# 1. Distance Correlation Function
def distance_corr(var_1, var_2, normedweight, power=1):
    """
    Calculate distance correlation between two variables.
    var_1, var_2: 1D tf tensors with the same number of entries
    normedweight: Per-example weight (sum of weights should add up to N)
    power: Exponent used in calculating the distance correlation
    """
    xx = tf.reshape(var_1, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_1)])
    xx = tf.reshape(xx, [tf.size(var_1), tf.size(var_1)])
    
    yy = tf.transpose(xx)
    amat = tf.math.abs(xx - yy)
    
    xx = tf.reshape(var_2, [-1, 1])
    xx = tf.tile(xx, [1, tf.size(var_2)])
    xx = tf.reshape(xx, [tf.size(var_2), tf.size(var_2)])
    
    yy = tf.transpose(xx)
    bmat = tf.math.abs(xx - yy)
    amat = tf.cast(amat, tf.float32)
    bmat = tf.cast(bmat, tf.float32)
    amatavg = tf.reduce_mean(amat * normedweight, axis=1)
    bmatavg = tf.reduce_mean(bmat * normedweight, axis=1)
    
    minuend_1 = tf.tile(amatavg, [tf.size(var_1)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_1), tf.size(var_1)])
    minuend_2 = tf.transpose(minuend_1)
    Amat = amat - minuend_1 - minuend_2 + tf.reduce_mean(amatavg * normedweight)
    
    minuend_1 = tf.tile(bmatavg, [tf.size(var_2)])
    minuend_1 = tf.reshape(minuend_1, [tf.size(var_2), tf.size(var_2)])
    minuend_2 = tf.transpose(minuend_1)
    Bmat = bmat - minuend_1 - minuend_2 + tf.reduce_mean(bmatavg * normedweight)
    
    ABavg = tf.reduce_mean(Amat * Bmat * normedweight, axis=1)
    AAavg = tf.reduce_mean(Amat * Amat * normedweight, axis=1)
    BBavg = tf.reduce_mean(Bmat * Bmat * normedweight, axis=1)
    
    if power == 1:
        dCorr = tf.reduce_mean(ABavg * normedweight) / tf.math.sqrt(tf.reduce_mean(AAavg * normedweight) * tf.reduce_mean(BBavg * normedweight))
    elif power == 2:
        dCorr = (tf.reduce_mean(ABavg * normedweight))**2 / (tf.reduce_mean(AAavg * normedweight) * tf.reduce_mean(BBavg * normedweight))
    else:
        dCorr = (tf.reduce_mean(ABavg * normedweight) / tf.math.sqrt(tf.reduce_mean(AAavg * normedweight) * tf.reduce_mean(BBavg * normedweight)))**power
    
    return dCorr

# %%
# 2. Custom Loss Function
def custom_loss(y_true, y_pred, dijet_mass, normedweight, lambda_corr=1.0):
    # Compute binary cross-entropy loss
    classifier_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    
    # Compute distance correlation loss
    dCorr_loss = distance_corr(y_pred, dijet_mass, normedweight)
    
    # Combine the losses with the lambda_corr regularization term
    total_loss = classifier_loss + lambda_corr * dCorr_loss
    return total_loss

# %%
# 3. Generate Fake Data
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate some fake data (random values)
X_train = np.random.rand(n_samples, 10)  # 10 features
X_test = np.random.rand(300, 10)         # 10 features
y_train = np.random.randint(0, 2, size=(n_samples, 1))  # Binary target
y_test = np.random.randint(0, 2, size=(300, 1))        # Binary target

# A feature to decorrelate with the output, e.g., dijet_mass
dijet_mass_train = np.random.rand(n_samples, 1)  # Some arbitrary feature to correlate with the model predictions
dijet_mass_test = np.random.rand(300, 1)

# Normalized weights (for illustration, we just use uniform weights here)
normed_weight_train = np.ones((n_samples, 1))
normed_weight_test = np.ones((300, 1))
# %%
# 4. Define a Simple Model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(10,)),  # 10 input features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for binary classification
])

# 5. Compile the Model with Custom Loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, dijet_mass=dijet_mass_train, normedweight=normed_weight_train, lambda_corr=0.1),
              metrics=['accuracy'])

# 6. Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 7. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 8. Make Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 9. Plot Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# %%

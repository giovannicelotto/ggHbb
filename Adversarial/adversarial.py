# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import sys
from functions import getCommonFilters
sys.path.append("/t3home/gcelotto/ggHbb/PNN/helpers")
from getFeatures import getFeatures
from tensorflow.keras.optimizers import Adam
# %%
#  IDEA : adversarial su 0 (Medium WP) e 1 (Tight WP)

# Generate features and a target correlated with an auxiliary variable
import glob
fileNamesSignal =  glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/GluGluHToBB/training/*.parquet")[:5]

featuresForTraining, columnsToRead = getFeatures()
featuresForTraining = featuresForTraining+['dijet_cs', 'jet1_btagDeepFlavB']
nEvts = 5e5
signal = pd.read_parquet(fileNamesSignal, columns=featuresForTraining, filters=getCommonFilters(), engine='pyarrow').iloc[:int(nEvts)]
signalLabels = np.ones(len(signal))
fileNamesBkg =  glob.glob("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/flatForGluGluHToBB/Data1A/training/Data_1.parquet")
bkg = pd.read_parquet(fileNamesBkg, columns=featuresForTraining, filters=getCommonFilters(), engine='pyarrow').iloc[:int(nEvts)]
bkgLabels = np.zeros(len(bkg))

features = pd.concat([signal, bkg])
labels = np.concatenate([signalLabels, bkgLabels])


from sklearn.utils import shuffle
features, labels = shuffle(features, labels)
# %%
# Train/test split
advFeature = 'jet1_btagDeepFlavB'
print("Adv feature is ", advFeature)
X_train, X_test, y_train, y_test, advFeature_train, advFeature_test = train_test_split(
    features, labels, features[advFeature], test_size=0.2, random_state=42
)
# %%
X_train = X_train.drop([advFeature], axis=1)
X_test = X_test.drop([advFeature], axis=1)
# %%
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_adv = StandardScaler()
advFeature_train = scaler_adv.fit_transform(pd.DataFrame(advFeature_train))
advFeature_test = scaler_adv.transform(pd.DataFrame(advFeature_test))

# %%
# Step 2: Define the gradient reversal layer
class GradientReversalLayer(Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -dy  # Reverse the gradient sign
            return x, grad
        return reverse_gradient(inputs)

# Step 3: Define the models
# Classifier model
def build_classifier(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(16, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(2, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Classifier")

# Adversary model
def build_adversary():
    inputs = Input(shape=(1,))  # Takes classifier scores as input
    x = GradientReversalLayer()(inputs)  # Reverse the gradients
    x = Dense(8, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(2, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)  # Predict `advFeature`
    return Model(inputs, outputs, name="Adversary")
# %%
# Step 4: Build the combined model
input_shape = (X_train.shape[1],)
classifier = build_classifier(input_shape)
adversary = build_adversary()
# %%
#Pretrain the classifier
optimizer = Adam(learning_rate = 5e-4, name="Adam") 
classifier.compile(optimizer=optimizer, loss='binary_crossentropy')
classifier.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))
# %%
#Fix the layers of classifiers
for layer in classifier.layers:
    layer.trainable = False
classifier_scores = classifier.predict(X_train).reshape(-1, 1)
classifier_scores = classifier.predict(X_train)
adversary_optimizer = Adam(learning_rate=1e-5)
adversary.compile(optimizer=adversary_optimizer, loss='mean_absolute_error')
adversary.fit(classifier_scores, advFeature_train, batch_size=32, epochs=100, validation_data=(classifier.predict(X_test), advFeature_test))
# %%
# Unfreeze the classifier weights
for layer in classifier.layers:
    layer.trainable = True
# %%
inputs = Input(shape=input_shape)
classifier_scores = classifier(inputs)
adversary_output = adversary(classifier_scores)

combined_model = Model(inputs=inputs, outputs=[classifier_scores, adversary_output])
print(combined_model.summary())

# Step 5: Compile the combined model
classification_loss = tf.keras.losses.BinaryCrossentropy()
adversarial_loss = tf.keras.losses.MeanAbsoluteError()
lambda_penalty = .25  # Adjust this hyperparameter
#optimizer = Adam(learning_rate = 5e-5, name="Adam") 
combined_model.compile(
    optimizer='adam',
    loss=[classification_loss, adversarial_loss],
    loss_weights=[1.0, lambda_penalty]
)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the total validation loss
    patience=50,         # Stop after 10 epochs with no improvement
    restore_best_weights=True  # Restore the weights from the best epoch
)

# Step 6: Train the model
history = combined_model.fit(
    X_train, [y_train, advFeature_train],
    validation_data=(X_test, [y_test, advFeature_test]),
    batch_size=32,
    epochs=2000,
    verbose=1,
    callbacks=[early_stopping] 

)
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(history.history['loss'], linestyle='--', label='Total Loss (Train)')
ax.plot(history.history['val_loss'], linestyle='solid', label='Total Loss (Val)')

# Plot classification loss
ax.plot(history.history['Classifier_loss'], linestyle='--', label='Classifier Loss (Train)', color='cyan')
ax.plot(history.history['val_Classifier_loss'], linestyle='solid', label='Classifier Loss (Validation)', color='orange')

# Plot adversarial loss
ax.plot(history.history['Adversary_loss'], linestyle='--', label='Adversary_loss (Train)', color='blue')
ax.plot(history.history['val_Adversary_loss'], linestyle='solid', label='Adversarial Loss (Validation)', color='red')

# Add labels and legend
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss') 
ax.set_yscale('log')
#ax.set_ylim(ax.get_ylim()[0], 1e-10)
ax.legend()
ax.grid()
fig.savefig("/t3home/gcelotto/ggHbb/Adversarial/%s_%d.png"%(advFeature, lambda_penalty))


# %%
# Step 7: Evaluate and plot results
# Predictions
y_pred = classifier.predict(X_test).flatten()
advFeature_pred = adversary.predict(y_pred).flatten()

# %%
# Metrics
fig,ax = plt.subplots(1, 1)
ax.hist(y_pred[y_test==0], bins=100, histtype='step')
ax.hist(y_pred[y_test==1], bins=100, histtype='step')
# %%
auc = roc_auc_score(y_test, y_pred)
import dcor
from sklearn.metrics import roc_auc_score, roc_curve
distance_corr = dcor.distance_correlation(y_pred, advFeature_test)
print(distance_corr)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()
plt.savefig("/t3home/gcelotto/ggHbb/Adversarial/loss.png")


# %%

# %%

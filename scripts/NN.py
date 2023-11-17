from plotScripts.utilsForPlot import loadData
import numpy as np

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Loading files
signalPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Hbb_QCDBackground2023Nov01/GluGluHToBB_M125_13TeV_powheg_pythia8/crab_GluGluHToBB/231101_175738/flatData/withoutArea"
realDataPath = "/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/bb_ntuples/nanoaod_ggH/Data20181A2023Nov08/ParkingBPH1/crab_data_Run2018A_part1/231108_145003/flatData"

signal, background = loadData(signalPath=signalPath, realDataPath=realDataPath, nSignalFiles=100, nRealDataFiles=100)

# prendi bkg fuori dalla massa 
background = background[(background[:,17]<100) | (background[:,17]>150)]
# Droppa la massa
signal = signal[:, np.arange(signal.shape[1]) != 17]
background = background[:, np.arange(background.shape[1]) != 17]

data = np.concatenate((signal, background), axis=0)
labels = np.concatenate((np.ones(len(signal)), np.zeros(len(background))))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer, binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=900, batch_size=4096, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test[y_test==0], y_test[y_test==0])
print(f'Background Test Loss: {loss}, Test Accuracy: {accuracy}')

loss, accuracy = model.evaluate(X_test[y_test==1], y_test[y_test==1])
print(f'Signal Test Loss: {loss}, Test Accuracy: {accuracy}')

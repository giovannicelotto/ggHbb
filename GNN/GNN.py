# %%
from functions import loadMultiParquet_Data_new, loadMultiParquet_v2
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch_geometric.data import DataLoader
import torch.optim as optim
from gnnClass import GNN
import torch.nn as nn
from createGraph import create_graph
# %%
columns = ['jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass',
'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass',
'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass',
'nJets', 'nSV']
dfsMC   = pd.concat(loadMultiParquet_v2(paths=[38,39,40,41,43], columns=columns, nMCs=15))
dfsData = pd.concat(loadMultiParquet_Data_new(dataTaking=[0], columns=columns, nReals=3, training=True)[0])


# Assuming dfsMC and dfsData are already loaded
# Add a 'label' column to signal (dfsMC) and background (dfsData) datasets
dfsMC['label'] = 1  # Label for signal
dfsData['label'] = 0  # Label for background

df = pd.concat([dfsMC, dfsData], axis=0)
# %%
# Define the columns for jet features (e.g., pt, eta, phi, mass for each jet)
jet_columns = [
    'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_mass', 
    'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_mass', 
    'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_mass'
]

X_jet = df[jet_columns].values
#scaler = StandardScaler()
#X_jet_scaled = scaler.fit_transform(X_jet)

global_features = df[['nJets', 'nSV']].values

X_combined = np.hstack([X_jet, global_features])
# %%
#X_combined = scaler.fit_transform(X_combined)
# %%
X_tensor = torch.tensor(X_combined)

y = df['label'].values
y_tensor = torch.tensor(y, dtype=torch.float)

# Split into training and validation sets
df_train, df_val, X_train, X_val, y_train, y_val = train_test_split(df, X_tensor, y_tensor, test_size=0.2, random_state=42)

# %%
# Now, we create graphs for both the training and validation sets
train_graphs = []
val_graphs = []

# Create graphs for the training set
for _, event_data in df_train.iterrows():
    label = float(event_data['label'])  # Signal (1) or Background (0)
    graph = create_graph(event_data, label)
    train_graphs.append(graph)
# %%
# Create graphs for the validation set
for _, event_data in df_val.iterrows():
    label = float(event_data['label'])  # Signal (1) or Background (0)
    graph = create_graph(event_data, label)
    val_graphs.append(graph)

# %%
# Create DataLoaders
train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True, drop_last=True)
val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False, drop_last=True)

# Initialize the model, optimizer, and loss function
model = GNN(input_dim=4, hidden_dim=12)  # input_dim = 4 (jet features)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()
# %%
# Training loop with validation after each epoch
epochs = 30
for epoch in range(epochs):
    model.train()  # Set model to training mode
    train_loss = 0
    val_loss = 0
    
    # Training phase
    for data_train in train_loader:
        optimizer.zero_grad()
        #print("train")
        #print(data_train)
        out_train = model(data_train)
        data_train.y = data_train.y.view(-1, 1).float()
        loss_train = criterion(out_train, data_train.y)
        train_loss += loss_train.item()

        # Backward pass
        loss_train.backward()
        optimizer.step()
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Disable gradient calculation during validation
        for data_val in val_loader:
            #print("val")
            #print(data_val)
            out_val = model(data_val)
            data_val.y = data_val.y.view(-1, 1).float()
            loss_val = criterion(out_val, data_val.y)
            val_loss += loss_val.item()


    # Print the results for this epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# %%



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare the Data
# We already have the X_tensor and y_tensor from the previous steps.

# We will split the data into training and testing datasets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42)

# Convert the data into DMatrix format (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define the model parameters for XGBoost
params = {
    'objective': 'binary:logistic',  # Binary classification task (0 or 1)
    'eval_metric': 'logloss',  # Logarithmic loss for binary classification
    'max_depth': 6,  # Maximum depth of trees
    'eta': 0.1,  # Learning rate
    'subsample': 0.8,  # Subsample ratio for trees
    'colsample_bytree': 0.8,  # Column subsampling ratio
    'nthread': 4  # Number of threads to use
}

# Specify the number of boosting rounds
num_round = 100

# Train the XGBoost model
model = xgb.train(params, dtrain, num_round)

# Predict the labels for the test set
y_pred = model.predict(dtest)
# %%
print(-(y_test*np.log(y_pred) + (1-y_test)*np.log(1-y_pred)).mean())

# Since the prediction returns probabilities, we need to convert them into binary labels
#y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred_binary)
#print(f"Accuracy: {accuracy}")

# Display a classification report
#print("Classification Report:")
#print(classification_report(y_test, y_pred_binary))


# %%

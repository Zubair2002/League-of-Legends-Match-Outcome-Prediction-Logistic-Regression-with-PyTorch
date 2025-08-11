# Import required libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
data = pd.read_csv('league_of_legends_data_large.csv')  # Make sure the file is in the same directory or provide full path

# 2. Split data into features (X) and target (y)
X = data.drop('win', axis=1)
y = data['win']

# 3. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  # (n, 1) shape for binary classification
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Optional: Print shapes to verify
print("X_train shape:", X_train_tensor.shape)
print("y_train shape:", y_train_tensor.shape)
print("X_test shape:", X_test_tensor.shape)
print("y_test shape:", y_test_tensor.shape)


#task 2
#Task 2: Implement a logistic regression model using PyTorch.
#Defining the logistic regression model involves specifying the input dimensions, the forward pass using the sigmoid activation function, and initializing the model, loss function, and optimizer. 


import torch.nn as nn
import torch.optim as optim

# 1. Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Output is 1 for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Apply sigmoid to get probability

# 2. Initialize the Model, Loss Function, and Optimizer

# Get the number of input features
input_dim = X_train_tensor.shape[1]

# Initialize the model
model = LogisticRegressionModel(input_dim)

# Binary Cross Entropy Loss for binary classification
criterion = nn.BCELoss()

# Stochastic Gradient Descent optimizer with learning rate 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)


#task 3  ## Model train

# Number of epochs
num_epochs = 1000

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)

    # Compute loss
    loss = criterion(outputs, y_train_tensor)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ========================
# Model Evaluation
# ========================
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient computation
    # Get predictions for training and test sets
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    # Apply threshold of 0.5
    train_preds_class = (train_preds >= 0.5).float()
    test_preds_class = (test_preds >= 0.5).float()

    # Calculate accuracy
    train_accuracy = (train_preds_class.eq(y_train_tensor).sum() / float(y_train_tensor.shape[0])).item()
    test_accuracy = (test_preds_class.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()

    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")


    #task 4 model optimization

# Re-initialize the model (important for retraining)
model_l2 = LogisticRegressionModel(input_dim)

# Use SGD with L2 regularization (weight_decay)
optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.01)

# Define the same loss function
criterion = nn.BCELoss()

# Train the model with L2 regularization
num_epochs = 1000

for epoch in range(num_epochs):
    model_l2.train()
    optimizer_l2.zero_grad()
    
    outputs = model_l2(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer_l2.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"[L2] Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ========================
# Evaluation
# ========================
model_l2.eval()

with torch.no_grad():
    train_preds = model_l2(X_train_tensor)
    test_preds = model_l2(X_test_tensor)

    train_preds_class = (train_preds >= 0.5).float()
    test_preds_class = (test_preds >= 0.5).float()

    train_accuracy = (train_preds_class.eq(y_train_tensor).sum() / float(y_train_tensor.shape[0])).item()
    test_accuracy = (test_preds_class.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()

    print(f"\n[L2] Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"[L2] Testing Accuracy: {test_accuracy * 100:.2f}%")



#task 5   visulization

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Evaluate Model with Predictions
model_l2.eval()
with torch.no_grad():
    # Get predicted probabilities and binary predictions
    y_test_probs = model_l2(X_test_tensor).numpy().flatten()  # probabilities
    y_test_pred = (y_test_probs >= 0.5).astype(int)            # thresholded
    y_test_true = y_test_tensor.numpy().flatten()              # actual values

# 2. Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_pred)
print("Confusion Matrix:\n", cm)

# Optional: Plot confusion matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Class 0", "Class 1"])
plt.yticks(tick_marks, ["Class 0", "Class 1"])

# Label each cell
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# 3. ROC Curve & AUC
fpr, tpr, _ = roc_curve(y_test_true, y_test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Classification Report
report = classification_report(y_test_true, y_test_pred, target_names=["Class 0", "Class 1"])
print("Classification Report:\n", report)



#task 6 model saving

# ========== Saving the Model ==========
# Save only the model's state dictionary (recommended approach)
torch.save(model_l2.state_dict(), 'logistic_model_l2.pth')
print("Model saved successfully!")

# ========== Loading the Model ==========
# Create a new instance of the same model architecture
loaded_model = LogisticRegressionModel(input_dim)

# Load the saved state_dict into the new model
loaded_model.load_state_dict(torch.load('logistic_model_l2.pth'))
loaded_model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

# ========== Evaluate the Loaded Model ==========
with torch.no_grad():
    test_probs_loaded = loaded_model(X_test_tensor).numpy().flatten()
    test_preds_loaded = (test_probs_loaded >= 0.5).astype(int)

    # Compare to ground truth
    test_accuracy_loaded = (test_preds_loaded == y_test_tensor.numpy().flatten()).mean()

    print(f"\n[Loaded Model] Testing Accuracy: {test_accuracy_loaded * 100:.2f}%")

# task7 hyperparameter tuning

# Hyperparameter: Learning rates to test
learning_rates = [0.01, 0.05, 0.1]
num_epochs = 100

# Store results
lr_results = {}
# Training and evaluation loop
for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")

    # Reinitialize model
    model = LogisticRegressionModel(input_dim)
    # Initialize optimizer with L2 regularization and current learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_tensor).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        y_true = y_test_tensor.numpy().flatten()
        test_accuracy = (test_preds == y_true).mean()
    lr_results[lr] = test_accuracy
    print(f"Test Accuracy for learning rate {lr}: {test_accuracy * 100:.2f}%")
# Find the best learning rate
best_lr = max(lr_results, key=lr_results.get)
print("\n=== Hyperparameter Tuning Summary ===")
for lr, acc in lr_results.items():
    print(f"Learning Rate {lr}: Test Accuracy = {acc * 100:.2f}%")
print(f"\n✅ Best Learning Rate: {best_lr} with Accuracy = {lr_results[best_lr] * 100:.2f}%")

#taskk 8 feature importance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Extract Weights from the Trained Model
# Assuming `model` is your final trained model (with best learning rate)
weights = model.linear.weight.data.numpy().flatten()  # shape: (n_features,)
# 2. Create a DataFrame for Feature Importances
feature_names = X.columns  # original feature names from the DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': weights,
    'Abs_Weight': np.abs(weights)
})
# 3. Sort by absolute weight values (importance)
importance_df_sorted = importance_df.sort_values(by='Abs_Weight', ascending=True)
# 4. Plotting the Feature Importances
plt.figure(figsize=(10, 8))
plt.barh(importance_df_sorted['Feature'], importance_df_sorted['Weight'], color='skyblue')
plt.xlabel("Weight (Feature Importance)")
plt.title("Feature Importance in Logistic Regression")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# Optional: Display the DataFrame (top 10 features)
print("\nTop 10 Most Influential Features:")
print(importance_df_sorted.sort_values(by='Abs_Weight', ascending=False).head(10))
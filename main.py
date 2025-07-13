import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from digit_loader import DigitClassification  # Ensure this is correctly implemented


# ======================
# 1. Load & Preprocess Data
# ======================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0
y = train_df['label'].values.astype(np.int64)
X_test = test_df.values.astype(np.float32) / 255.0

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val)
X_test_tensor = torch.tensor(X_test)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)

# ======================
# 2. Model, Loss, Optimizer
# ======================
model = DigitClassification()  # Must inherit from nn.Module and define forward()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)


# ======================
# 3. Training Loop
# ======================
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


# ======================
# 4. Evaluation on Validation Set
# ======================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nValidation Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("Classification Report:\n", classification_report(all_labels, all_preds))


# ======================
# 5. Prediction on Test Set
# ======================
test_preds = []
with torch.no_grad():
    for i in range(0, X_test_tensor.shape[0], 64):
        batch = X_test_tensor[i:i+64]
        outputs = model(batch)
        preds = outputs.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(test_preds) + 1),
    "Label": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved to submission.csv")

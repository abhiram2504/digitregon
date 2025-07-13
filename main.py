import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from digit_loader import DigitCNN 
from torchvision import transforms

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0
y = train_df['label'].values.astype(np.int64)
X_test = test_df.values.astype(np.float32) / 255.0

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

#tensors
X_train_tensor = torch.tensor(X_train.reshape(-1, 1, 28, 28))
y_train_tensor = torch.tensor(y_train)
X_val_tensor = torch.tensor(X_val.reshape(-1, 1, 28, 28))
y_val_tensor = torch.tensor(y_val)
X_test_tensor = torch.tensor(X_test.reshape(-1, 1, 28, 28))


#DataLoaders
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)

#model 
model = DigitCNN() 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


#Training 
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

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
        scheduler.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


#Eval
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


#prediction
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

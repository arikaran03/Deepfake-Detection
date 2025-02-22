from train import train_model
from test_model import test_model

# Define paths
train_dir = "D:/deepfake_detection/train-20250112T065955Z-001"
test_dir = "D:/deepfake_detection/test-20250112T065939Z-001"

# Train the model
print("Training the model...")
train_model(train_dir, test_dir, epochs=10)

# Evaluate the model
print("Evaluating the model...")
test_model(train_dir, test_dir)  # Pass train_dir and test_dir
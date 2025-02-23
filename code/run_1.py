import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from train_1 import train_model
from evaluating_1 import evaluate_model


train_dir = "data/train"
test_dir = "data/test"

# Train model
train_model(train_dir, test_dir, epochs=10)

# Evaluate model
evaluate_model(test_dir)

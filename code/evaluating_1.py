from tensorflow.keras.models import load_model
from load_data_1 import load_data

def evaluate_model(test_dir, model_path='models/model_v1.keras', batch_size=32):
    # Load data
    _, _, test_generator = load_data(None, test_dir, batch_size=batch_size)

    # Load model
    model = load_model(model_path)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}")
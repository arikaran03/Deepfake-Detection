from tensorflow.keras.models import load_model
from data_loading import load_data

def test_model(train_dir, test_dir, model_path='deepfake_detection_model.h5', batch_size=32):
    # Load data
    _, _, test_generator = load_data(train_dir, test_dir, batch_size=batch_size)

    # Load model
    model = load_model(model_path)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc}")
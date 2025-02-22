from tensorflow.keras.models import load_model

# Load the model
model = load_model('deepfake_detection_model.h5')

# Display the model summary
model.summary()
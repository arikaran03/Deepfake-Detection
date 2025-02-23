from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def build_model(input_shape=(380, 380, 3)):
    # Load EfficientNet-B4 as the base model
    base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Build the final model
    model = Model(inputs=base_model.input, outputs=outputs)
    return model
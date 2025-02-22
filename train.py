from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
from data_loading import load_data

def train_model(train_dir, test_dir, epochs=10, batch_size=32):
    # Load data
    train_generator, val_generator, _ = load_data(train_dir, test_dir, batch_size=batch_size)

    # Build model
    model = build_model()

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    # Save model
    model.save('deepfake_detection_model.h5')

    return history
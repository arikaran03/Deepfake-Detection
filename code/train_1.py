from build_model_1 import build_model
from load_data_1 import load_data
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(train_dir, test_dir, epochs=10, batch_size=32):
    # Load data
    train_generator, val_generator, test_generator = load_data(train_dir, test_dir, batch_size=batch_size)

    # Build model
    model = build_model()

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Add model checkpoint
    checkpoint = ModelCheckpoint(
        'models/model_checkpoint.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint]
    )

    # Save final model
    model.save('models/model_v1.keras')

    return history
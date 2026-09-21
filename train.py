import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

MODEL_NAME = "gender_model.keras"


# ============================================================
# 2. CHECK DATASET FOLDERS
# ============================================================

if not os.path.exists(TRAIN_DIR):
    print("ERROR: Training folder not found!")
    print("Expected:", TRAIN_DIR)
    exit()

if not os.path.exists(TEST_DIR):
    print("ERROR: Test folder not found!")
    print("Expected:", TEST_DIR)
    exit()

print("Training folder found:", TRAIN_DIR)
print("Test folder found:", TEST_DIR)


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    validation_split=0.2,

    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# 4. LOAD TRAINING IMAGES
# ============================================================

print("\nLoading training images...")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True,

    seed=42
)


# ============================================================
# 5. LOAD VALIDATION IMAGES
# ============================================================

print("\nLoading validation images...")

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False,

    seed=42
)


# ============================================================
# 6. LOAD TEST IMAGES
# ============================================================

print("\nLoading test images...")

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


# ============================================================
# 7. DISPLAY DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Class labels:")
print(train_generator.class_indices)

print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Test images:", test_generator.samples)

print("========================================")


# ============================================================
# 8. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


# ============================================================
# 9. CREATE CLASSIFICATION MODEL
# ============================================================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 10. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# 11. DISPLAY MODEL
# ============================================================

print("\nModel created successfully!")

model.summary()


# ============================================================
# 12. CREATE CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    MODEL_NAME,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ============================================================
# 13. START TRAINING
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 14. TEST MODEL
# ============================================================

print("\n========================================")
print("TESTING MODEL")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_generator
)


print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# 15. SAVE MODEL
# ============================================================

model.save(MODEL_NAME)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(f"Model saved as: {MODEL_NAME}")

print("\nCLASS MAPPING:")
print(train_generator.class_indices)

print("\nYou can now run the webcam program.")
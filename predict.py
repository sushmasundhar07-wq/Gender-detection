import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "gender_model.keras"
IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# START CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera started!")
print("Press Q to quit.")


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break


    # --------------------------------------------------------
    # Prepare camera image for MobileNetV2
    # --------------------------------------------------------

    img = cv2.resize(frame, IMAGE_SIZE)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(
        img,
        axis=0
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        img,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )


    # --------------------------------------------------------
    # Convert prediction to class
    # --------------------------------------------------------

    if probability >= 0.5:

        gender = "Male"

        confidence = probability * 100

    else:

        gender = "Female"

        confidence = (1 - probability) * 100


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    text = f"{gender} - {confidence:.1f}%"

    cv2.putText(
        frame,
        text,
        (30, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.2,

        (0, 255, 0),

        3
    )


    cv2.imshow(
        "Gender Detection - Press Q to Quit",
        frame
    )


    # --------------------------------------------------------
    # Quit with Q
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLOSE CAMERA
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("Camera closed.")
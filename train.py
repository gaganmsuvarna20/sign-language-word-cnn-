import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATA_DIR = "data"
GESTURES = ["hello","namaskar","love_you","home","hi","promise","help","name"]

# 1. Load data
X = []
y = []

EXPECTED_LEN = 126   # 2 hands * 21 points * 3 coords

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        path = os.path.join(DATA_DIR, file)
        landmarks = np.load(path)

        # Skip files with wrong shape (e.g., old 63-length data)
        if landmarks.shape[0] != EXPECTED_LEN:
            print(f"Skipping {file}: shape {landmarks.shape}")
            continue

        X.append(landmarks)
        label_name = file.split("_")[0]
        y.append(label_name)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if X.size == 0:
    raise ValueError("No valid samples of length 126 found in 'data' folder.")


# 2. Encode labels (to integers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)           # 0..num_classes-1
num_classes = len(le.classes_)

# 3. Train / test split (use encoded labels for stratify)
X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. One-hot encode AFTER splitting
y_train = to_categorical(y_train_enc, num_classes)
y_test = to_categorical(y_test_enc, num_classes)

# 5. Build CNN model (for 2 hands: 42 points × 3 coords)
model = Sequential([
    InputLayer(input_shape=(126,)),        # 126 = 2 * 21 landmarks * 3 (x,y,z)
    Reshape((42, 3, 1)),                  # 42 points (21 left + 21 right), 3 coords

    Conv2D(32, (3, 2), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPooling2D((2, 1)),

    Conv2D(64, (3, 2), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPooling2D((2, 1)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 6. Callbacks
checkpoint = ModelCheckpoint(
    "sign_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

# 7. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# 8. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# 9. Save label mapping for use in prediction script
np.save("label_classes.npy", le.classes_)
print("Saved model as sign_model.h5 and labels as label_classes.npy")

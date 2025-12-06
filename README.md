# sign-language-word-cnn-
# Sign Language Word Recognition (CNN + MediaPipe)

This project uses **MediaPipe hand landmark tracking** and a **Convolutional Neural Network (CNN)** to recognize Indian / English sign language gestures in real-time.

### ✨ Supported Gestures:
- Hello
- Namaskar
- Love You
- Home
- Hi
- Promise
- Help
- Name

---

## 🚀 Features

✔ Real-time webcam gesture recognition  
✔ Uses Mediapipe Hand tracking  
✔ Works with custom gesture dataset  
✔ Supports multi-hand tracking (Left + Right)  
✔ CNN classification  
✔ Live interface with confidence bars  
✔ Smooth prediction using moving average  

---

## 🛠 Tech Stack

| Component | Used |
|----------|------|
| Language | Python |
| ML Model | TensorFlow / Keras |
| Hand Tracking | MediaPipe |
| UI | OpenCV |
| Dataset | Custom `.npy` |

---

## 📁 Project Structure
├── collect_data.py
├── train.py
├── predict.py
├── data/ # collected gestures
├── sign_model.h5 # saved model
└── label_classes.npy # saved label order



---

## 📦 Install requirements
```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn


📷 Collect gesture dataset
python collect_data.py

🎓 Train the model
python train.py

🤖 Run prediction (webcam)
python predict.py
Press Q to quit.

📊 Model Architecture

Input: 126-dim hand landmarks (42 × 3)
2D CNN
Softmax classification
Adam optimizer

🧠 How it works

MediaPipe extracts hand landmarks (Left + Right)
We convert 3D joint points → a feature vector
CNN learns patterns per gesture
Predicts in real time with smoothing


Accuracy (example)
Model	Accuracy
CNN (custom dataset)	~85–92% depending on data

(results depend on dataset size)

![Gesture Demo](images/hello.png)


🧑‍💻 Author

GAGAN

GitHub profile:
https://github.com/gaganmsuvarna20

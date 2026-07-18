# ✋ Real-Time Hand Gesture Recognition using PyTorch & OpenCV

A real-time hand gesture recognition project built using **PyTorch** and **OpenCV**. The goal of this project was to understand the complete machine learning workflow by building everything myself, from collecting data to training a CNN and finally using it for live webcam prediction.

This project recognizes **7 different hand gestures** in real time and uses a simple majority voting technique over multiple frames to make the predictions more stable.

---

# 🚀 Features

* Custom CNN built from scratch using PyTorch
* Real-time webcam gesture recognition
* Temporal majority voting over 70 frames
* Confidence score based on prediction votes
* Data augmentation during training
* Custom dataset collection script
* Lightweight model that runs on CPU

---

# 📂 Project Structure

```text
Gesture-Recognition/
│
├── training.ipynb
├── vision_predict.py
├── Model1best_83_95.pth
├── requirements.txt
└── README.md
```

---

# 🧠 Model Architecture

The model is a custom Convolutional Neural Network consisting of three convolutional blocks.

```
Input (128 × 128 RGB)

↓

Conv2D
BatchNorm
ReLU

Conv2D
BatchNorm
ReLU

MaxPool

↓

Conv2D
BatchNorm
ReLU

Conv2D
BatchNorm
ReLU

MaxPool

↓

Conv2D
BatchNorm
ReLU

Conv2D
BatchNorm
ReLU

MaxPool

↓

Flatten

↓

Fully Connected Layer

↓

7 Output Classes
```

The network uses:

* Batch Normalization
* ReLU Activation
* Max Pooling
* CrossEntropyLoss
* Adam Optimizer

---

# 📸 Dataset

The model was trained on **7 gesture classes** with around **1,000 images per class** (roughly **7,000 images** in total).

Most of the training images came from the **HaGRID (HAnd Gesture Recognition Image Dataset)**, which contains hand gestures captured in real-world environments with a variety of backgrounds. I also created a small dataset collection tool (`dataset_maker.py`) using OpenCV to capture additional gesture images whenever I needed more samples. Datasetmaker repo is in my repository if anyone wishes to use it.

The supported gesture classes are:

* no_gesture
* palm
* one
* peace
* thumb_index
* ok
* three

Every image is resized to **128 × 128** before being passed to the model.

---

# 🔄 Data Augmentation

To improve the model's ability to generalize, the following augmentations were used during training:

* Resize (128 × 128)
* TrivialAugmentWide
* Random Affine Scaling
* Random Horizontal Flip
* Tensor Conversion

The test dataset only uses resizing and tensor conversion.

---

# 🏋️ Training

The model was trained using:

* PyTorch Dataset and DataLoader
* 80/20 Train-Test Split
* CrossEntropyLoss
* Adam Optimizer
* GPU when available (otherwise CPU)

Training and testing accuracy were calculated after every epoch.

---

# 🎥 Real-Time Prediction

The prediction pipeline works as follows:

1. Capture frames from the webcam.
2. Crop a fixed Region of Interest (ROI).
3. Resize the image to **128 × 128**.
4. Run the image through the trained CNN.
5. Store predictions for **70 consecutive frames**.
6. Apply majority voting to determine the final gesture.
7. Display the prediction only if the confidence is above **70%**.

Using majority voting helped reduce flickering predictions during live inference and produced more stable results.

---

# ⌨️ Controls

| Key   | Function                            |
| ----- | ----------------------------------- |
| **Z** | Start a 3-second prediction session |
| **Q** | Quit the application                |

---

# 🛠 Technologies Used

* Python
* PyTorch
* TorchVision
* OpenCV
* NumPy
* Pillow
* scikit-learn

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Gesture-Recognition.git
cd Gesture-Recognition
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python predict.py
```

---

# ⚠️ Limitations

* The model was primarily trained using the HaGRID dataset, which contains images captured in real-world environments.
* Because of this, the model generally performs best on backgrounds similar to those seen during training.
* Performance may decrease in environments that differ significantly from the training data, such as completely plain or highly controlled backgrounds.I plan to solve this in 
  this in the near future.

---

# 💡 Future Improvements

* Integrate MediaPipe for automatic hand detection instead of using a fixed ROI.
* Train on additional gestures and a larger dataset.
* Experiment with transfer learning using architectures such as ResNet or EfficientNet.
* Display class probabilities during live prediction.
* Export the model using ONNX for faster inference and deployment.

---

# 📄 License

This project was created for learning and educational purposes.

---

# 👤 Author

**Dino Ranjit**

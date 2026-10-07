# ✋ Real-Time Hand Gesture Recognition using PyTorch & OpenCV

A real-time hand gesture recognition system built using **PyTorch** and **OpenCV**.

This project started as an attempt to understand the complete machine learning workflow by building a gesture recognition system from the ground up — from working with image datasets and training CNNs to deploying the trained model for real-time webcam inference.

After multiple iterations, the model was improved by combining the **HaGRID dataset with 500 additional images per gesture collected from my own hand**, allowing the model to better handle variation between different hands and environments.

The current model recognizes **7 different hand gestures** and uses temporal majority voting to produce more stable predictions during live webcam inference.

---

# 🚀 Features

* Custom CNN built using PyTorch
* **512 × 512 RGB input resolution**
* Trained on a combined **HaGRID + custom hand dataset**
* **500 additional images per class** collected from my own hand
* Data augmentation during training
* Real-time webcam gesture recognition
* Temporal majority voting over multiple frames
* Confidence estimation based on prediction votes
* Runs on CPU as well as GPU
* Custom dataset collection tool built using OpenCV

---

# 📈 Model Development

This project went through several iterations rather than being trained once and finalized.

### Initial Model

The first version achieved approximately:

**86% training accuracy / 86% test accuracy**

Although the model worked reasonably well, there was still significant room for improvement.

### Adding Custom Data

To improve generalization to my own hand, I collected approximately **500 additional images for each gesture class**.

I initially experimented with **transfer learning**, but the resulting model did not generalize as well as expected and tended to become overly specialized toward my own hand.

Instead, I integrated the additional images directly into the original HaGRID class folders, creating a combined dataset containing both the original HaGRID images and my own samples.

The model was then retrained on the combined dataset.

### Current Model

The latest training run achieved approximately:

* **Training Accuracy: ~93%**
* **Test Accuracy: ~97%**

This iteration provided a substantial improvement over the original ~86% model.

---

# 🧠 Model Architecture

The current model is a custom convolutional neural network built using PyTorch.

The network progressively extracts visual features through multiple convolutional layers before passing the resulting representation to fully connected layers for classification.

The model uses:

* Convolutional layers
* Batch Normalization
* ReLU activation
* Max Pooling
* Fully connected layers
* CrossEntropyLoss
* Adam optimizer

### Input

```text
512 × 512 × 3
```

### Output

```text
7 gesture classes
```

The model was designed and trained manually rather than using a pretrained classification architecture.

---

# 📸 Dataset

The project uses a combination of the **HaGRID (HAnd Gesture Recognition Image Dataset)** and additional images collected specifically for this project.

The final dataset contains the following gesture classes:

* `no_gesture`
* `palm`
* `one`
* `peace`
* `thumb_index`
* `ok`
* `three`

### Custom Data

To make the model more robust to differences between hands, I collected approximately:

```text
500 images × 7 classes
= ~3,500 additional images
```

These images were integrated into the corresponding HaGRID class folders before training.

This resulted in a training dataset containing both publicly available HaGRID samples and images of my own hand.

---

# 🔄 Data Augmentation

Training images were augmented to introduce additional variation and reduce over-reliance on specific image conditions.

The training pipeline includes transformations such as:

* Resizing to **512 × 512**
* Random horizontal flipping
* Brightness adjustment
* Contrast adjustment
* Rotation
* Random affine transformations

The test data is evaluated without the random training augmentations.

---

# 🏋️ Training

The model was trained using:

* **PyTorch**
* `Dataset` and `DataLoader`
* CrossEntropyLoss
* Adam optimizer
* GPU acceleration when available
* Train/test evaluation after training

The training process was performed using a combined dataset rather than treating the custom hand images as a separate transfer-learning dataset.

The latest model reached approximately **97% test accuracy**, compared with approximately **86%** in the original version.

---

# 🎥 Real-Time Prediction

The trained model can be used for real-time gesture recognition through a webcam.

The inference pipeline works as follows:

```text
Webcam
   ↓
Frame Capture
   ↓
Region of Interest (ROI)
   ↓
Resize to 512 × 512
   ↓
CNN Prediction
   ↓
Collect Predictions
   ↓
Majority Voting
   ↓
Final Gesture
```

Predictions from multiple consecutive frames are collected and a majority vote is used to determine the final gesture.

This prevents individual incorrect frames from immediately changing the displayed prediction and makes the system more stable during real-time use.

---

# ⌨️ Controls

| Key   | Function                   |
| ----- | -------------------------- |
| **X** | Start the program          |
| **Z** | Start a prediction session |
| **Q** | Quit the application       |

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

# 📂 Project Structure

```text
Gesture-Recognition/
│
├── training.ipynb
├── vision_predict.py
├── model_synthesized_94_97.pth
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Gesture-Recognition.git
cd Gesture-Recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the real-time recognition system:

```bash
python vision_predict.py
```

---

# ⚠️ Limitations

Although the model achieved approximately **97% test accuracy**, test accuracy alone does not guarantee the same performance on completely unseen real-world conditions.

The model is trained primarily on HaGRID images together with images collected from my own hand. Performance can therefore vary depending on:

* Lighting conditions
* Camera quality
* Backgrounds
* Hand orientation
* Distance from the camera
* Gestures that differ significantly from the training examples

The current system also uses a fixed region of interest rather than automatically detecting the hand.

---

# 💡 Future Improvements

Possible future improvements include:

* Automatic hand detection instead of a fixed ROI
* Expanding the number of gesture classes
* Collecting data from additional people
* Further improving robustness to different backgrounds and lighting conditions
* Experimenting with other CNN architectures
* Exporting the model for optimized deployment

---

# 📄 License

This project was created for learning and educational purposes.

---

# 👤 Author

**Dino Ranjit**

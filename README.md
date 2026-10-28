# VisionLens — Deep Learning Image Classifier

**See the image. Discover what it contains.**

VisionLens is a web-based image classification application built with Python, PyTorch, and Streamlit. It uses a pretrained ResNet-101 deep learning model to analyze uploaded images and display the five most likely ImageNet categories with their confidence scores.

## ✨ Features

* **Image Upload:** Upload images in JPG, JPEG, PNG, and other supported formats.
* **Deep Learning Classification:** Uses a pretrained ResNet-101 convolutional neural network.
* **Top-5 Predictions:** Displays the five most likely categories for an uploaded image.
* **Confidence Scores:** Shows the model's prediction probabilities.
* **Interactive Interface:** Provides a modern web interface built with Streamlit.
* **Efficient Inference:** Caches the model and uses inference mode to avoid unnecessary gradient calculations.
* **Local Execution:** Run the application on your own computer.

## 🧰 Technology Stack

| Technology  | Purpose                                               |
| ----------- | ----------------------------------------------------- |
| Python      | Application programming                               |
| PyTorch     | Deep learning and model inference                     |
| Torchvision | Pretrained ResNet-101 model and image transformations |
| Streamlit   | Interactive web interface                             |
| Pillow      | Image handling                                        |
| ImageNet    | 1,000-category classification label set               |

## 🧠 Model

VisionLens uses **ResNet-101**, a convolutional neural network pretrained on the ImageNet dataset.

The model produces scores for 1,000 ImageNet categories. VisionLens selects the five highest-scoring categories and displays their normalized probabilities.

**Important:** These scores represent the model's confidence for its predicted categories. They do not guarantee that a prediction is correct.

## 📁 Project Structure

```text
VisionLens/
├── streamlit_ui.py        # Streamlit application and inference logic
├── imagenet_classes.txt   # ImageNet category labels
├── requirements.txt       # Python dependencies
├── LICENSE                # Original MIT license
├── README.md              # Project documentation
└── .gitignore             # Files excluded from Git
```

Virtual environments and Python cache files are excluded from version control.

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.11
* Git
* Internet access for installing packages and downloading pretrained model weights

### 1. Clone the repository

Replace the placeholder below with the URL of your VisionLens GitHub repository after creating it.

```powershell
git clone YOUR_VISIONLENS_REPOSITORY_URL
cd VisionLens
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails because of platform or package availability, install a compatible PyTorch and Torchvision combination using the official PyTorch installation instructions.

### 5. Run VisionLens

```powershell
python -m streamlit run streamlit_ui.py
```

Streamlit will display a local address in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your browser.

## 🚀 How to Use

1. Open VisionLens in your browser.
2. Upload an image using the file uploader.
3. Start the classification process using the interface.
4. Review the predicted categories and their confidence scores.
5. Try another image to explore the model's predictions.

The first run may take longer because pretrained model weights need to be downloaded.

## 🔍 Limitations

* Predictions depend on the categories represented in ImageNet.
* The model may misclassify unusual images, low-quality photographs, or objects outside its training categories.
* Confidence scores should not be treated as a guarantee of correctness.
* The application is intended for educational and demonstration purposes, not safety-critical decisions.

## 📚 Original Project and Attribution

VisionLens is adapted from the following open-source project:

**Image Classification Web App using PyTorch and Streamlit**
Original repository: https://github.com/denistanjingyu/Image-Classification-Web-App-using-PyTorch-and-Streamlit

The original project is distributed under the MIT License. Its original license and copyright notice are retained in this repository.

VisionLens includes modifications to the application interface and code for compatibility with newer versions of Streamlit, PyTorch, and Torchvision.

Please refer to the included `LICENSE` file for the applicable license terms.

## 👩‍💻 Project Purpose

VisionLens demonstrates how a pretrained deep learning model can be integrated into an interactive Python web application. It provides a practical introduction to image classification, model inference, image preprocessing, and presenting machine learning predictions through a user interface.

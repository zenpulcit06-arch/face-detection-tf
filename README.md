# 🧠 Face Detection with TensorFlow — A Learning Journey

> **Goal:** Learn TensorFlow from scratch, using face detection as the north star project.  
> **Approach:** Start fast with a pre-trained model → then build our own CNN from the ground up.  
> **Background:** Know the maths of AI, new to TensorFlow.

---

## 🗺️ Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 01 | TensorFlow Basics — Tensors, Ops, Gradients | 🔄 In Progress |
| 02 | First Neural Net — MNIST Digit Classifier | ⏳ Upcoming |
| 03 | Convolutional Neural Networks (CNN) | ⏳ Upcoming |
| 04 | Face Detection — Pre-trained model (fast win) | ⏳ Upcoming |
| 05 | Face Detection — Train our own CNN | ⏳ Upcoming |

---

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/face-detection-tf.git
cd face-detection-tf

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Requirements

See `requirements.txt`. Core libraries:
- `tensorflow` — our main framework
- `opencv-python` — image loading and webcam access
- `numpy` — numerical ops
- `matplotlib` — visualizing results
- `jupyter` — running notebooks

---

## 📁 Project Structure

```
face-detection-tf/
├── 01_basics/           # Tensor ops, shapes, gradients, tf.function
├── 02_mnist/            # First neural net — handwritten digit classifier
├── 03_cnn/              # CNN from scratch — understanding convolutions
├── 04_face_detection/   # Face detection (pre-trained + custom CNN)
├── notes/               # Math notes, key concepts, learning log
├── requirements.txt
└── README.md
```

---

## 📝 Learning Log

### What I've learned so far
- [ ] What a tensor is and how shapes work
- [ ] Basic TensorFlow operations
- [ ] How automatic differentiation (autograd) works
- [ ] Building a model with `tf.keras`
- [ ] What convolutions actually do to an image
- [ ] Training a face detector end-to-end

---

## 🔑 Key Concepts (My Notes)

> This section grows as we learn. Written in plain language, not textbook speak.

**Tensor:** An n-dimensional array. A single pixel value = scalar (0D). A row of pixels = vector (1D). A grayscale image = matrix (2D). An RGB image = 3D tensor `(height, width, 3)`. A batch of images = 4D tensor `(batch, height, width, 3)`.

---

*Project started: 2025 | Built to learn, not just to ship.*

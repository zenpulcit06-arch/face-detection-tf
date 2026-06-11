# 🧠 Face Detection with TensorFlow — A Learning Journey

> **Goal:** Learn TensorFlow from scratch, using face detection as the north star project.  
> **Approach:** Start fast with a pre-trained model → then build our own CNN from the ground up.  
> **Background:** Know the maths of AI, new to TensorFlow.

---

## 🗺️ Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 01 | TensorFlow Basics — Tensors, Ops, Gradients | ✅ Done |
| 02 | First Neural Net — MNIST Digit Classifier | ✅ Done |
| 03 | Convolutional Neural Networks (CNN) | 🔄 In Progress |
| 04 | Face Detection — Pre-trained model (fast win) | ⏳ Upcoming |
| 05 | Face Detection — Train our own CNN | ⏳ Upcoming |

---

## 🛠️ Setup (WSL / Linux / Mac)

```bash
# Clone the repo
git clone https://github.com/zenpulcit06-arch/face-detection-tf.git
cd face-detection-tf

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open a notebook
jupyter notebook --no-browser
# Copy the localhost link into your browser
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
- [x] What a tensor is and how shapes work
- [x] Basic TensorFlow operations
- [x] How automatic differentiation (autograd) works
- [x] `tf.Variable` vs `tf.constant` — and why it matters for gradients
- [x] Why we need activation functions (stacked linear layers collapse into one)
- [x] Building a model with `tf.keras` — Sequential, Dense, compile, fit
- [x] What softmax does and why it's used for classification output
- [ ] What convolutions actually do to an image
- [ ] Training a face detector end-to-end

---

## 🔑 Key Concepts (My Notes)

> This section grows as we learn. Written in plain language, not textbook speak.

**Tensor:** An n-dimensional array. A single pixel value = scalar (0D). A row of pixels = vector (1D). A grayscale image = matrix (2D). An RGB image = 3D tensor `(height, width, 3)`. A batch of images = 4D tensor `(batch, height, width, 3)`.

**Dense layer:** Computes `y = f(Wx + b)`. W and b are learned by gradient descent. f is an activation function.

**Why activations:** Without them, stacking layers is pointless — `W2(W1x + b1) + b2` is just another linear function. Activations like ReLU break linearity so deep networks can learn complex patterns.

**Softmax:** Converts raw scores (logits) into probabilities that sum to 1. Used at the output of classifiers. `softmax(z_i) = e^z_i / Σ e^z_j`.

**Loss for classification:** `sparse_categorical_crossentropy` — use when labels are integers (0, 1, 2...). `categorical_crossentropy` — use when labels are one-hot vectors.

**Overfitting signal:** Training accuracy much higher than validation accuracy. The model memorized the training data instead of learning general patterns.

**Convolution:** Slide a small filter (e.g. 3×3) across an image. At each position, compute a dot product between the filter and the image patch. Output is a feature map — highlights where the filter pattern was found.

**Feature map:** The output of one filter applied to the whole image. 32 filters → 32 feature maps.

**MaxPooling:** Take max value in each 2×2 patch. Halves spatial dimensions. Reduces computation and adds slight position invariance.

**Dropout:** Randomly zero out neurons during training. Prevents overfitting by forcing the network to not rely on any single neuron.

**Parameter sharing:** One filter scans the entire image — only 9 weights for a 3×3 filter, regardless of image size. Dense layer would need one weight per pixel pair.

---

*Project started: 2025 | Built to learn, not just to ship.*

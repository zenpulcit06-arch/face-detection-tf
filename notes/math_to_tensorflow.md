# 📐 Math ↔ TensorFlow Cheat Sheet

> For someone who knows the maths — here's how TensorFlow names things.

---

## Linear Algebra

| Math notation | TensorFlow |
|---------------|-----------|
| Vector **v** | `tf.constant([1., 2., 3.])` |
| Matrix **A** | `tf.constant([[1,2],[3,4]])` |
| **A** · **B** (matrix multiply) | `A @ B` or `tf.matmul(A, B)` |
| **A**ᵀ (transpose) | `tf.transpose(A)` |
| ‖**v**‖ (L2 norm) | `tf.norm(v)` |
| Element-wise multiply | `A * B` |

---

## Calculus / Autograd

| Math | TensorFlow |
|------|-----------|
| Variable to differentiate wrt | `tf.Variable(x)` |
| Forward pass + record ops | `with tf.GradientTape() as tape:` |
| ∂L/∂w | `tape.gradient(L, w)` |
| w ← w − α·∂L/∂w | `w.assign_sub(lr * grad)` |

---

## Loss Functions

| Loss | Math | Keras name |
|------|------|-----------|
| MSE | (1/n)·Σ(ŷ−y)² | `MeanSquaredError` |
| Binary cross-entropy | −[y·log(ŷ) + (1−y)·log(1−ŷ)] | `BinaryCrossentropy` |
| Categorical cross-entropy | −Σ yᵢ·log(ŷᵢ) | `CategoricalCrossentropy` |

---

## Activation Functions

| Name | Math | When to use |
|------|------|------------|
| ReLU | max(0, x) | Hidden layers in CNNs |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output (face / no face) |
| Softmax | eˣⁱ/Σeˣʲ | Multi-class output |

---

## Shape Convention

```
(batch_size, height, width, channels)
     32        224    224      3
```

Always think in this order. When confused, print `.shape`.

---

## Neural Network Layers

| Concept | Math | Keras |
|---------|------|-------|
| Dense layer | y = f(Wx + b) | `Dense(n, activation='relu')` |
| ReLU | max(0, x) | `activation='relu'` |
| Sigmoid | 1 / (1 + e⁻ˣ) | `activation='sigmoid'` |
| Softmax | eᶻⁱ / Σeᶻʲ | `activation='softmax'` |

## Why Stacking Linear Layers Fails

Two linear layers:
```
y1 = W1·x + b1
y2 = W2·y1 + b2 = (W2·W1)·x + (W2·b1 + b2)
```
This is just one linear layer. No depth benefit without activations.

## Keras Training API

```python
model = tf.keras.Sequential([...])      # define architecture
model.compile(optimizer, loss, metrics) # define training config
model.fit(x, y, epochs, batch_size)     # run training loop
model.evaluate(x_test, y_test)          # measure on unseen data
model.predict(x)                        # get raw output
```

## Loss Functions

| Loss | Labels format | Use case |
|------|--------------|----------|
| `sparse_categorical_crossentropy` | integers `[0, 3, 7]` | multi-class |
| `categorical_crossentropy` | one-hot `[[0,0,1,...]]` | multi-class |
| `binary_crossentropy` | `[0, 1]` | face / no face ← we'll use this |
| `mean_squared_error` | continuous values | regression |

## Reading model.summary()

Layer params = (inputs × outputs) + outputs (bias)  
Example: `Dense(128)` after input of 784:  
`784 × 128 + 128 = 100,480 parameters`

---

## Learning Log

### Session 1
- Learned what tensors are
- Ran gradient descent manually on a linear function
- Saw how `GradientTape` implements the chain rule

### Session 2
- Built first neural network on MNIST
- Understood why activations are necessary
- Learned softmax, cross-entropy loss, model.summary()
- Saw training curves — train vs validation accuracy

## Convolutional Layers

| Concept | Keras |
|---------|-------|
| Convolution | `Conv2D(filters, kernel_size)` |
| Feature map | output of one Conv2D filter |
| MaxPool | `MaxPooling2D(pool_size=2)` |
| Dropout | `Dropout(rate=0.5)` |

## Conv2D Parameter Count

```
params = (kernel_h x kernel_w x in_channels + 1) x out_filters
Conv2D(32, 3) on grayscale  = (3x3x1 + 1) x 32  = 320
Conv2D(64, 3) after 32 maps = (3x3x32 + 1) x 64 = 18,496
```

## Spatial Dimensions Through a CNN

```
Input:              (28, 28,  1)
Conv2D(32, 3) same: (28, 28, 32)
MaxPool(2):         (14, 14, 32)
Conv2D(64, 3) same: (14, 14, 64)
MaxPool(2):         (7,   7, 64)
Flatten:            (3136,)       <- 7 x 7 x 64
```

## Dense vs CNN Parameter Comparison

| Task | Dense params | CNN params |
|------|-------------|------------|
| 32 filters on 28x28 grayscale | 784 x 32 = 25,088 | (3x3+1) x 32 = 320 |

One filter is shared across the whole image — that is parameter sharing.

## Session 3 Learning Log
- Ran hand-crafted vertical and horizontal edge filters manually
- Saw what feature maps look like on a real image
- Built CNN — higher accuracy than Dense with fewer conv parameters
- Visualized 32 learned filters after training
- Key insight: early layers learn edges, deep layers learn complex patterns

## Face Detection Concepts

| Term | Meaning |
|------|---------|
| Bounding box | (x, y, w, h) or (x1, y1, x2, y2) — where the face is |
| Confidence score | Sigmoid output = probability region is a face |
| IOU | Intersection over Union — measures how well two boxes overlap |
| Threshold | If confidence > 0.5 → face. Tune this to trade precision vs recall |

## Bounding Box Formats

```
Format 1: (x, y, w, h)       <- OpenCV Haar style
  x, y = top-left corner
  w, h = width and height

Format 2: (x1, y1, x2, y2)   <- Deep learning style
  x1, y1 = top-left corner
  x2, y2 = bottom-right corner

Convert: x2 = x1 + w,  y2 = y1 + h
```

## Preprocessing for Deep Learning Detectors

```python
# 1. Resize to fixed size the model expects
img = cv2.resize(img, (300, 300))

# 2. Subtract training mean (centers data around 0)
img = img - [104.0, 177.0, 123.0]   # mean BGR of training set

# 3. Add batch dimension
blob = img[np.newaxis, ...]          # (1, 300, 300, 3)
```

## Why Fixed Input Size?
Dense layers have fixed weight matrices — they cannot accept variable input sizes.
Every image must be resized to the same shape the model was trained on.
CNNs with only conv+pool layers CAN accept variable sizes — Dense layers cannot.

## Session 4 Learning Log
- Ran Haar Cascade detector — understood scaleFactor and minNeighbors
- Ran deep learning detector — saw confidence scores on bounding boxes
- Understood BGR vs RGB — common OpenCV bug to watch for
- Understood why mean subtraction helps training
- Key difference: Haar = hand-crafted rules, CNN = learned from data

## tf.data Pipeline

```python
dataset = tfds.load("name", split='train')          # load
dataset = dataset.map(preprocess_fn)                 # transform every sample
dataset = dataset.shuffle(buffer_size=N)             # shuffle (N = full dataset size)
train, test = tf.keras.utils.split_dataset(d, 0.8)  # split
dataset = dataset.batch(64)                          # group into batches
dataset = dataset.prefetch(tf.data.AUTOTUNE)         # load next batch in background
```

## Class Imbalance

| Problem | Symptom | Fix |
|---------|---------|-----|
| More no-face than face | Model always predicts no-face | Undersample majority with .take(N) |
| Imbalanced test set | 100% accuracy (fake) | Shuffle with large buffer, check counts |

## Functional API vs Sequential

```python
# Sequential — one path only
model = tf.keras.Sequential([LayerA(), LayerB()])

# Functional — explicit flow, allows branching
inputs = tf.keras.Input(shape=(...))
x = LayerA()(inputs)
x = LayerB()(x)
out1 = LayerC()(x)   # branch 1
out2 = LayerD()(x)   # branch 2
model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])
```

## Bugs Log — Things That Went Wrong and Why

| Bug | Root cause | Fix |
|-----|-----------|-----|
| 100% val accuracy | All one class in test set | buffer_size = full dataset size |
| Unstable loss | 13k faces vs 50k no-faces | .take(13000) on CIFAR |
| validation_split error | Only works with numpy | validation_data=test_data |
| Pixel values 0.000015 | map applied twice | Reload dataset, map once |
| NameError tfds | Ran cell before imports | Always run cells in order |

## Session 5 Learning Log
- Built complete tf.data pipeline from scratch
- Hit and fixed class imbalance bug
- Hit and fixed shuffle buffer bug  
- Learned Functional API for multi-output models
- Trained face detector: 99.9% validation accuracy
- Tested on real unseen photo: 100% confidence, correct prediction
- Key insight: debugging data is as important as debugging the model

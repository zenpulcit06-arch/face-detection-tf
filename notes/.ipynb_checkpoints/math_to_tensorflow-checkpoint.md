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

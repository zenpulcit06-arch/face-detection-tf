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

## Learning Log

### 2025 — Session 1
- Learned what tensors are
- Ran gradient descent manually on a linear function
- Saw how `GradientTape` implements the chain rule

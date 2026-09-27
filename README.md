# ML/DL From Scratch in C / C++

This project is a personal exploration into **machine learning and deep learning** implemented purely in **C / C++**, with **no external ML/math libraries** (even `log`, `exp`, `pow` are implemented from scratch via series expansion).  
The goal is to deeply understand ML/DL algorithms — and eventually the systems underneath them — by building everything from first principles.

---

## Project Goals

- Reinforce understanding of core ML/DL concepts, from math to implementation
- Practice low-level programming and memory management
- Build intuition by avoiding black-box libraries
- Understand not just *how to compute* a gradient, but *how automatic differentiation itself works*

---

## Part 1: Models with hand-derived gradients (C)

Each model here uses an **analytically hand-derived** gradient, computed and coded manually — no autodiff.

- **LINREG** — Linear regression, gradient descent, MSE
- **LOGREG** — Logistic regression, sigmoid + BCE, gradient via **numerical differentiation (finite differences)** instead of an analytical derivative
- **LINREG2** — Multiple linear regression (matrix-shaped weights, multi-input/multi-output)

All three also implement their own `log`, `exp`, and `min-max scaling` from scratch (no `<math.h>` functions used).

## Part 2: Autograd Engine (C++)

A minimal **reverse-mode automatic differentiation engine** built from scratch — the same core idea behind PyTorch's `autograd`.

- Dynamic computation graph built as operations run (`mul`, `sum`, `sub`, `div`, `power`)
- DFS-based topological sort to determine correct backward execution order
- Reverse-mode gradient accumulation via chain rule
- Demonstrated end-to-end with a working linear regression training loop (gradient descent, `zero_grad`, parameter updates) that converges and generalizes on a held-out test set

Unlike Part 1, gradients here aren't derived or coded by hand per-model — the engine computes them automatically for *any* composition of the supported operators.

---

Feel free to give any feedback or suggestions for improvement.

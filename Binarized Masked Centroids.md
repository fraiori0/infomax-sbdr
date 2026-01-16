# Rescaled Centroid Neural Gas: Mathematical Formulation

A local online learning rule for centroid-based representations with adaptive axis rescaling, driving outputs toward sparse binary codes with controllable sparsity.

---

## 1. Model Architecture

### 1.1 Parameters

For a module with $d_c$ output units and input dimension $d_{in}$:

- **Centroids**: $c_i \in \mathbb{R}^{d_{in}}$ for $i \in \{1, \ldots, d_c\}$
- **Rescaling/masking weights**: $w_i \in \mathbb{R}_+^{d_{in}}$ (strictly positive)
- **Running average of activations**: $\bar{z}_i \in [0, 1]$ for each unit

### 1.2 Forward Pass

Given input $x \in \mathbb{R}^{d_{in}}$:

**Step 1: Compute normalized weighted squared distance**

$$D_i = \frac{1}{d_{in}} \sum_{j=1}^{d_{in}} w_{i,j}^2 \left( c_{i,j} - x_j \right)^2$$

> **Critical**: The $1/d_{in}$ normalization keeps $D_i$ in a reasonable range (typically $O(1)$) regardless of input dimension, preventing exponentially small activations.

**Step 2: Compute unit activation (RBF-like)**

$$z_i = \exp(-D_i) \in (0, 1]$$

The output is $z \in (0, 1]^{d_c}$.

---

## 2. Loss Function: Quadratic Binarizing Loss

### 2.1 Definition

The loss to minimize is:

$$\mathcal{L} = -\sum_{i=1}^{d_c} \left[\alpha_i (z_i - 1)(z_i - 1.5) + (1-\alpha_i) z_i(z_i + 0.5)\right]$$

where $\alpha_i$ is the switching coefficient:

$$\alpha_i = \mathbb{1}[\bar{z}_i > p_*] = 
\begin{cases} 
1 & \text{if } \bar{z}_i > p_* \quad \text{(unit too active on average)} \\
0 & \text{if } \bar{z}_i \leq p_* \quad \text{(unit too inactive on average)}
\end{cases}$$

### 2.2 Loss Shape Analysis

**Case $\alpha_i = 1$ (unit too active):**
$$\mathcal{L}_i = -(z_i - 1)(z_i - 1.5) = -z_i^2 + 2.5z_i - 1.5$$

| $z_i$ | $\mathcal{L}_i$ |
|-------|-----------------|
| 0 | $-1.5$ (minimum) |
| 0.5 | $-0.5$ |
| 1 | $0$ (maximum) |

→ Pushes activation toward 0 ✓

**Case $\alpha_i = 0$ (unit too inactive):**
$$\mathcal{L}_i = -z_i(z_i + 0.5) = -z_i^2 - 0.5z_i$$

| $z_i$ | $\mathcal{L}_i$ |
|-------|-----------------|
| 0 | $0$ (maximum) |
| 0.5 | $-0.5$ |
| 1 | $-1.5$ (minimum) |

→ Pushes activation toward 1 ✓

### 2.3 Key Property: Non-Vanishing Boundary Gradients

The gradient with respect to $z_i$ is:

$$\frac{\partial \mathcal{L}}{\partial z_i} = -(2z_i + 0.5 - 3\alpha_i)$$

**Boundary values:**
- At $z_i = 0$, $\alpha_i = 0$: $\frac{\partial \mathcal{L}}{\partial z_i} = -0.5 \neq 0$
- At $z_i = 1$, $\alpha_i = 1$: $\frac{\partial \mathcal{L}}{\partial z_i} = +0.5 \neq 0$

Unlike cross-entropy losses, the gradient **does not vanish** at the boundaries.

---

## 3. Straight-Through Estimator

### 3.1 The Problem

When chaining through $z_i = e^{-D_i}$:

$$\frac{\partial \mathcal{L}}{\partial D_i} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot \frac{\partial z_i}{\partial D_i} = \frac{\partial \mathcal{L}}{\partial z_i} \cdot (-z_i)$$

The factor $z_i$ causes vanishing gradients when $z_i \to 0$.

### 3.2 The Solution

Use a **straight-through estimator**: in the backward pass, treat $\frac{\partial z_i}{\partial D_i} \approx -1$ instead of $-z_i$.

This gives:
$$\frac{\partial \mathcal{L}}{\partial D_i} \approx \frac{\partial \mathcal{L}}{\partial z_i} \cdot (-1) = 2z_i + 0.5 - 3\alpha_i$$

### 3.3 Mathematical Justification

The straight-through estimator can be viewed as optimizing in log-space:
$$\tilde{\mathcal{L}} = f(-\log z_i) = f(D_i)$$

where the gradient $\frac{\partial f}{\partial D_i}$ is computed directly without the $z_i$ factor.

This is well-established in neural network quantization literature for pushing activations toward discrete values.

---

## 4. Error Signal

Define the **error signal** (negative loss gradient w.r.t. $z$, with straight-through):

$$\boxed{\delta_i = 2z_i + 0.5 - 3\alpha_i}$$

### 4.1 Properties

| Condition | $\alpha_i$ | $\delta_i$ range | Direction |
|-----------|------------|------------------|-----------|
| Too inactive ($\bar{z}_i \leq p_*$) | 0 | $[0.5, 2.5]$ | Positive → increase $z_i$ |
| Too active ($\bar{z}_i > p_*$) | 1 | $[-2.5, -0.5]$ | Negative → decrease $z_i$ |

### 4.2 Boundary Behavior (No Dead Units!)

| Condition | $z_i$ | $\delta_i$ |
|-----------|-------|------------|
| Dead unit that should activate | 0 | $+0.5$ ✓ |
| Saturated unit that should deactivate | 1 | $-0.5$ ✓ |

The error signal is **always non-zero** at the boundaries.

---

## 5. Gradient Derivation

### 5.1 Gradients of Distance

With $D_i = \frac{1}{d_{in}}\sum_j w_{i,j}^2 (c_{i,j} - x_j)^2$:

$$\frac{\partial D_i}{\partial c_{i,j}} = \frac{2 w_{i,j}^2}{d_{in}} (c_{i,j} - x_j)$$

$$\frac{\partial D_i}{\partial w_{i,j}} = \frac{2 w_{i,j}}{d_{in}} (c_{i,j} - x_j)^2$$

### 5.2 Loss Gradients (with Straight-Through)

Using $\frac{\partial \mathcal{L}}{\partial D_i} = -\delta_i$:

$$\frac{\partial \mathcal{L}}{\partial c_{i,j}} = -\delta_i \cdot \frac{2 w_{i,j}^2}{d_{in}} (c_{i,j} - x_j)$$

$$\frac{\partial \mathcal{L}}{\partial w_{i,j}} = -\delta_i \cdot \frac{2 w_{i,j}}{d_{in}} (c_{i,j} - x_j)^2$$

---

## 6. Standard Gradient Update Rules

Performing gradient descent: $\Delta \theta = -\eta \frac{\partial \mathcal{L}}{\partial \theta}$

### 6.1 Centroid Update

$$\boxed{\Delta c_{i,j} = \frac{2 \eta_c}{d_{in}} \, \delta_i \, w_{i,j}^2 \left( x_j - c_{i,j} \right)}$$

**Interpretation:**
- $\delta_i > 0$ (unit should be more active): centroid moves **toward** $x$
- $\delta_i < 0$ (unit should be less active): centroid moves **away from** $x$
- Update magnitude scales with $w_{i,j}^2$ (stronger in attended dimensions)

### 6.2 Weight Update

$$\boxed{\Delta w_{i,j} = -\frac{2 \eta_w}{d_{in}} \, \delta_i \, w_{i,j} \left( c_{i,j} - x_j \right)^2}$$

**Interpretation:**
- $\delta_i > 0$: weights **decrease** → broader receptive field → more activation
- $\delta_i < 0$: weights **increase** → sharper selectivity → less activation

### 6.3 Positivity Constraint

After each update:
$$w_{i,j} \leftarrow \max(w_{i,j}, \epsilon_w)$$

---

## 7. Natural Gradient Update Rules

### 7.1 Fisher Information (Gaussian Interpretation)

Each unit can be viewed as a Gaussian with precision $\Lambda_i = \text{diag}(2w_{i,j}^2/d_{in})$.

**For centroid parameters:**
$$F_{c_{i,j}} = \frac{2 w_{i,j}^2}{d_{in}}$$

**For weight parameters:**
$$F_{w_{i,j}} = \frac{2}{w_{i,j}^2 \cdot d_{in}}$$

### 7.2 Natural Gradient Updates

**Centroid (natural gradient):**

$$\Delta^{\text{nat}} c_{i,j} = \frac{\Delta c_{i,j}}{F_{c_{i,j}}} = \frac{\frac{2\eta_c}{d_{in}} \delta_i w_{i,j}^2 (x_j - c_{i,j})}{\frac{2 w_{i,j}^2}{d_{in}}}$$

$$\boxed{\Delta^{\text{nat}} c_{i,j} = \eta_c \, \delta_i \left( x_j - c_{i,j} \right)}$$

**Key property:** Weight dependence cancels completely—uniform updates across all dimensions.

**Weight (natural gradient):**

$$\boxed{\Delta^{\text{nat}} w_{i,j} = -\eta_w \, \delta_i \, w_{i,j}^3 \left( c_{i,j} - x_j \right)^2}$$

---

## 8. Running Statistics Update

Maintain exponential moving average:

$$\boxed{\bar{z}_i \leftarrow (1 - \beta) \bar{z}_i + \beta \, z_i}$$

After updating, recompute:
$$\alpha_i = \mathbb{1}[\bar{z}_i > p_*]$$

---

## 9. Complete Algorithm

### 9.1 Initialization

```
c_i ← sample from data distribution
w_i ← constant positive value (e.g., 1.0)
z̄_i ← p* (target sparsity)
```

### 9.2 Per-Sample Update

```
Input: x, learning rates η_c, η_w, EMA coefficient β, target sparsity p*

1. FORWARD PASS
   For each unit i:
       D_i = mean_j [w²_{i,j} (c_{i,j} - x_j)²]
       z_i = exp(-D_i)

2. COMPUTE SWITCHING COEFFICIENTS
   For each unit i:
       α_i = 1 if z̄_i > p*, else 0

3. COMPUTE ERROR SIGNALS (straight-through)
   For each unit i:
       δ_i = 2z_i + 0.5 - 3α_i

4. UPDATE PARAMETERS
   Standard gradient:
       c_{i,j} ← c_{i,j} + (2η_c/d_in) · δ_i · w²_{i,j} · (x_j - c_{i,j})
       w_{i,j} ← w_{i,j} - (2η_w/d_in) · δ_i · w_{i,j} · (c_{i,j} - x_j)²
   
   Natural gradient:
       c_{i,j} ← c_{i,j} + η_c · δ_i · (x_j - c_{i,j})
       w_{i,j} ← w_{i,j} - η_w · δ_i · w³_{i,j} · (c_{i,j} - x_j)²
   
   Enforce positivity:
       w_{i,j} ← max(w_{i,j}, ε_w)

5. UPDATE RUNNING STATISTICS
   For each unit i:
       z̄_i ← (1-β)z̄_i + β·z_i
```

---

## 10. Summary Tables

### Error Signal

$$\delta_i = 2z_i + 0.5 - 3\alpha_i, \quad \alpha_i = \mathbb{1}[\bar{z}_i > p_*]$$

### Standard Gradient Updates

| Parameter | Update Rule |
|-----------|-------------|
| Centroid | $\Delta c_{i,j} = \frac{2\eta_c}{d_{in}} \delta_i w_{i,j}^2 (x_j - c_{i,j})$ |
| Weight | $\Delta w_{i,j} = -\frac{2\eta_w}{d_{in}} \delta_i w_{i,j} (c_{i,j} - x_j)^2$ |

### Natural Gradient Updates

| Parameter | Update Rule |
|-----------|-------------|
| Centroid | $\Delta^{\text{nat}} c_{i,j} = \eta_c \delta_i (x_j - c_{i,j})$ |
| Weight | $\Delta^{\text{nat}} w_{i,j} = -\eta_w \delta_i w_{i,j}^3 (c_{i,j} - x_j)^2$ |

---

## 11. Why This Formulation Works

### 11.1 Dimension Normalization
- Prevents $D_i$ from scaling with $d_{in}$
- Keeps activations in a useful range

### 11.2 Quadratic Loss
- Creates energy wells at $z=0$ and $z=1$
- Drives activations toward binary values
- Non-vanishing gradient at boundaries

### 11.3 Straight-Through Estimator
- Removes the $z_i$ factor from gradients
- Dead units ($z_i \approx 0$) can still receive updates
- No "dying unit" problem

### 11.4 Switching Mechanism
- Units above target activity are pushed toward 0
- Units below target activity are pushed toward 1
- At equilibrium: $\bar{z}_i \approx p_*$ for all units

---

## 12. Hyperparameters

| Symbol | Description | Typical Range |
|--------|-------------|---------------|
| $d_c$ | Number of units | 64 - 1024 |
| $p_*$ | Target sparsity | 0.01 - 0.2 |
| $\eta_c$ | Centroid learning rate | $10^{-2}$ - $10^{-1}$ |
| $\eta_w$ | Weight learning rate | $10^{-3}$ - $10^{-2}$ |
| $\beta$ | EMA coefficient | $10^{-3}$ - $10^{-1}$ |
| $\epsilon_w$ | Minimum weight value | $10^{-6}$ |

---

## 13. Expected Behavior

At convergence:
1. **Controlled sparsity**: Mean activation $\bar{z}_i \approx p_*$ for all units
2. **Binarization**: Most activations near 0 or 1
3. **Specialization**: Each unit responds strongly to a subset of inputs
4. **Learned attention**: Weights $w_{i,j}$ highlight relevant input dimensions
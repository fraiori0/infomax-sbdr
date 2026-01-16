# Band Activation Unit with Anti-Hebbian Learning

## Unit Definition

A **band activation unit** creates a hyperplanar activation band in the input space. For input $x \in \mathbb{R}^d$, the unit's activation is:

$$
    f(x) = \mathbb{I}\left(-(w^T x + b_1)^2 + b_2^2 > 0\right)
$$

where:

- $w \in \mathbb{R}^d$ is a **_unit-norm_ weight vector** ($\|w\| = 1$) controlling the orientation of the band
- $b_1 \in \mathbb{R}$ is a **position bias** controlling the shift of the band perpendicular to $w$
- $b_2 \in \mathbb{R}_+$ is a **width parameter** controlling the half-width of the activation band
- $\mathbb{I}(\cdot)$ is the Heaviside step function

This is equivalent to using:

### Geometric Interpretation

The unit activates when:

$$
    |w^T x + b_1| < b_2
$$

This defines a hyperplanar band (or "slab") in the input space:

- **Centerline hyperplane**: $\{x : w^T x + b_1 = 0\}$
- **Center position** (in projected space): $-b_1$
- **Band width**: $2b_2$
- **Orientation**: perpendicular to direction $w$

The signed distance from point $x$ to the centerline is:
$$d(x) = w^T x + b_1$$

## Learning Objective

For a layer of $n$ band activation units, the learning objective is:

1. **Homeostasis**: Each unit $i$ maintains a target average activation rate $\rho$:
   $$\mathbb{E}[f_i(x)] = \rho$$

2. **Decorrelation**: Pairwise co-activation matches a target level $\rho_{\text{pair}}$:
   $$\mathbb{E}[f_i(x) f_j(x)] = \rho_{\text{pair}}$$

   For independence, set $\rho_{\text{pair}} = \rho^2$.

## Update Rules

All updates are gradient-free and local. For unit $i$, given input $x$:

### 1. Width Parameter ($b_2^i$): Homeostatic Control

$$\boxed{\Delta b_2^i = \eta_2 \left(\rho - f_i(x)\right)}$$

**Mechanism**: Directly controls the firing rate by adjusting band width.

- If unit fires too often: narrow the band ($b_2 \downarrow$)
- If unit fires too rarely: widen the band ($b_2 \uparrow$)

**Learning rate**: $\eta_2 \approx 0.1$ (fast adaptation)

### 2. Position Bias ($b_1^i$): Centerline Translation

For each pair of units $(i, j)$:

$$\boxed{\Delta b_1^i = \eta_b \sum_{j \neq i} f_i(x) f_j(x) \cdot e_{ij} \cdot (w_i^T x + b_1^i)}$$

where:

- $e_{ij} = c_{ij} - \rho_{\text{pair}}$ is the co-activity error
- $c_{ij}$ is a running estimate: $c_{ij} \leftarrow (1-\beta) c_{ij} + \beta \cdot f_i(x) f_j(x)$
- Typical $\beta \approx 0.05$ for temporal smoothing

**Mechanism**: Translates the centerline perpendicular to itself.

- If $e_{ij} > 0$ (excessive co-activation): move centerline away from co-activating samples
- If $e_{ij} < 0$ (insufficient co-activation): move centerline toward co-activating samples

**Learning rate**: $\eta_b \approx 0.01$ (medium adaptation)

### 3. Weight Vector ($w^i$): Centerline Rotation

For each pair of units $(i, j)$:

$$\boxed{\Delta w^i = \eta_w \sum_{j \neq i} f_i(x) f_j(x) \cdot e_{ij} \cdot (w_i^T x + b_1^i) \cdot \text{proj}_{\perp w_i}(x)}$$

where:

- $\text{proj}_{\perp w_i}(x) = x - (w_i^T x) w_i$ is the projection of $x$ onto the tangent space at $w_i$
- After update: **renormalize** $w^i \leftarrow w^i / \|w^i\|$

**Mechanism**: Rotates the centerline around a pivot.

- If $e_{ij} > 0$: rotate centerline to increase distance from co-activating samples
- If $e_{ij} < 0$: rotate centerline to decrease distance from co-activating samples
- The factor $(w_i^T x + b_1^i)$ provides signed distance weighting

**Learning rate**: $\eta_w \approx 0.001$ (slow adaptation for stability)

## Combined Effect

The $b_1$ and $w$ updates work synergistically:

- **$b_1$ provides parallel translation** of the centerline
- **$w$ provides rotation** of the centerline
- Together they move the centerline away from (or toward) regions of co-activation
- This reduces (or increases) the geometric overlap between bands

## Equilibrium and Convergence

### Fixed Point Conditions

At equilibrium, the system satisfies:

1. **Homeostasis**: $\mathbb{E}[f_i(x)] = \rho$ for all units $i$

2. **Decorrelation**: $c_{ij} = \rho_{\text{pair}}$ for all pairs $(i,j)$, or equivalently:
   $$\mathbb{E}[f_i(x) f_j(x) \cdot (w_i^T x + b_1^i)] = 0$$

The second condition means: when units co-activate, the sample is equally likely to be on either side of the centerline (balanced push-pull).

### Equilibrium Properties

At equilibrium with $\rho_{\text{pair}} = \rho^2$ (independence target):

1. **Spatial Coverage**: Bands collectively cover approximately $n \rho$ of the data space volume

2. **Diversity**: The system exhibits:

   - **Diverse orientations**: Weight vectors $w_i$ point in different directions
   - **Distributed positions**: Bias values $b_1^i$ spread to minimize overlap
   - **Efficient tiling**: Bands tessellate high-density regions of the data space

3. **Stability**: The system has negative feedback for co-activity:
   - Excessive co-activation → centerlines pushed apart → reduced future co-activation
   - This provides local stability near equilibrium

### Convergence Dynamics

The learning system operates on three time scales:

- **Fast** ($\eta_2$): Band widths adjust quickly to maintain firing rates
- **Medium** ($\eta_b$): Band positions shift to decorrelate activity
- **Slow** ($\eta_w$): Band orientations rotate to maximize coverage and diversity

This hierarchical structure promotes stable convergence:

1. Widths stabilize first, ensuring all units remain active
2. Positions adjust to reduce overlap
3. Orientations slowly optimize for diverse representation

### Potential Degeneracies

Two potential suboptimal equilibria exist:

1. **Parallel collapse**: Multiple units with similar orientations but different positions

   - Creates parallel "stripes" through data space
   - Stable but inefficient for high-dimensional data
   - Can be mitigated with proper initialization

2. **Dead units**: Units that drift outside data support
   - Prevented by fast $b_2$ homeostasis
   - Proper initialization minimizes occurrence

## Initialization Strategy

**Recommended approach** (data-driven):

1. **Orientations** ($w$): Initialize using principal components of the data

   - Run PCA on data sample
   - Assign first $n$ principal directions to units
   - Add small random noise to break symmetry
   - If $n >$ number of PCs, add random orthonormal directions

2. **Positions** ($b_1$): Center bands on data mean

   - $b_1^i = -w_i^T \bar{x}$ where $\bar{x}$ is the data mean

3. **Widths** ($b_2$): Set to capture target fraction of data

   - Project data onto each $w_i$
   - Set $b_2^i$ to the $\rho$-percentile of $|w_i^T x + b_1^i|$

4. **Correlations** ($c_{ij}$): Initialize to independence assumption
   - $c_{ij} = \rho^2$ for all $i \neq j$

This initialization ensures:

- All units are active from the start
- Initial coverage of data space
- Diverse initial orientations
- Fast convergence to equilibrium

## Summary

The band activation unit with anti-Hebbian learning provides a biologically-inspired mechanism for discovering sparse, decorrelated representations of data. The learning rules are:

- **Local**: Only require information about the current sample and unit activations
- **Gradient-free**: No backpropagation needed
- **Interpretable**: Clear geometric meaning for each parameter
- **Stable**: Hierarchical time scales promote convergence

The equilibrium configuration tessellates the data space with oriented hyperplanar bands that capture diverse aspects of the data distribution while maintaining decorrelated activity patterns.

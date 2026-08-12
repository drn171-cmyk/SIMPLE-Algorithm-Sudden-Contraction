# 🌊 Laminar Flow Through a Sudden Contraction
### Solved with the SIMPLE Algorithm

**Python 3.8+** · **NumPy** · **Matplotlib** · 🧪 *Research / Educational* · 📄 *MIT-style license*

*A from-scratch, pure-Python/NumPy finite-difference CFD solver — no external CFD libraries.*

---

## 📖 Overview

This project simulates **laminar, incompressible flow** through an axisymmetric pipe with a **sudden contraction** (large diameter `D` → small diameter `d`), using the classic **SIMPLE** (Semi-Implicit Method for Pressure-Linked Equations) algorithm to couple pressure and velocity.

```
   ┌──────────────────────────┐
   │                          │
D  │   ───────────▶           │───────┐
   │                          │   d   │  ───────▶
   │                          │───────┘
   └──────────────────────────┘
        large pipe               small pipe
```

---

## ⚙️ Physical Model

| Quantity | Symbol | Value |
|---|:---:|:---:|
| Large pipe diameter | `D` | `0.01 m` |
| Small pipe diameter | `d` | `0.005 m` |
| Large pipe length | `L` | `10·D` |
| Small pipe length | `l` | `10·d` |
| Kinematic viscosity | `nu` | `1.5e-5 m²/s` |
| Inlet velocity | `U` | `2 m/s` (configurable) |
| Reynolds number | `Re = U·D/nu` | printed at runtime |

> 🧊 Only half the geometry is modeled, using a **symmetry boundary condition** along the centerline.

---

## 🧮 Numerical Method

- 🟦 **Grid**: `u`, `v`, `p` on a single collocated grid.
- 🔁 **Momentum equations**: solved via point-by-point Gauss-Seidel-style iteration.
- ⚡ **Stability fix — hybrid/upwind scheme**: the `a1..a4` convection-diffusion coefficients are clamped to stay non-negative. This stops the central-difference scheme from diverging to `NaN` once the grid Péclet number (`U·h/nu`) exceeds the stability limit of `2`.
- 🌀 **Pressure correction**: `p'` Poisson equation, solved with 50 Jacobi/Gauss-Seidel sub-iterations per outer step.
- 🎚️ **Under-relaxation**: `omega1 = 0.5` (velocity), `omega2 = 0.3` (pressure).
- ✅ **Convergence**: tracked via the RMS change in the velocity field; the loop exits once the error drops below `tol`, after at least `min_iter` iterations.

---

## 🚀 Getting Started

### Install dependencies
```bash
pip install numpy matplotlib
```

### What you'll see
1. 🖨️ Reynolds number + convergence error printed every 50 iterations
2. 📉 **Convergence history** plot (log-scale error vs. iteration)
3. 🌡️ **Velocity magnitude contour** plot
4. 🌬️ **Streamlines** plot

---

## 🎛️ Configurable Parameters

| Variable | Description | Default |
|---|---|:---:|
| `U` | Inlet velocity (m/s) | `2` |
| `N` | Grid points in the y-direction | `50` |
| `tol` | Convergence tolerance | `1e-4` |
| `min_iter` | Min. iterations before checking convergence | `200` |
| `max_iter` | Max. number of iterations | `3000` |
| `omega1` / `omega2` | Velocity / pressure under-relaxation | `0.5` / `0.3` |

---

## ⚠️ Known Limitations

- ⏱️ **Performance**: pure Python `for` loops mean one iteration at this grid size (N=50, ~1500 columns) takes **~6 seconds**, mostly from the nested 50-sub-iteration pressure Poisson solve. Real convergence may need thousands of iterations, so a full run can take **hours**. Shrink `N`, `M1`, `M2` for quick tests.
- 📐 The pressure-correction `alpha` neglects convection (a standard SIMPLE simplification) and this isn't a staggered-grid Rhie–Chow formulation, so mild checkerboard pressure oscillations are possible at higher `Re`.
- 🌡️ Valid for the **laminar regime only**.

---

Made for educational / research purposes · contributions & forks welcome 🙌

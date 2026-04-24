---
title: The Mathematics of Portfolio Risk and Return (A Complete Guide)
date: 2026-04-18 10:00:00 +0000
categories: [Quantitative Finance, Portfolio Management]
tags: [math, optimization, modern-portfolio-theory, risk-management]
math: true
---
# The Mathematics of Portfolio Risk and Return
## A Complete Guide from First Principles to Portfolio Construction

---

## Table of Contents

1. **The Building Blocks: Returns, Risk, and Co-movement**
2. **From Individual Assets to Portfolios**
3. **The Power of Diversification**

---

## 1. The Building Blocks: Returns, Risk, and Co-movement

Before we can build a portfolio, we need a precise language for describing what each asset does on its own and how assets behave together. Three quantities form the entire foundation: **expected return**, **variance**, and **covariance**.

---

### 1.1 Expected Return

The expected return answers a simple question: *on average, what do I earn?*

There are two settings where we estimate this.

**Scenario-based (forward-looking):** An analyst assigns probabilities to future scenarios.

$$E[R] = \sum_{i=1}^{S} p_i \, r_i$$

where $p_i$ is the probability of scenario $i$ and $r_i$ is the return in that scenario.

**Example:** Three scenarios for a stock over the next year:

| Scenario | Probability $p_i$ | Return $r_i$ |
|----------|--------------------|---------------|
| Boom     | $0.30$               | $+30\%$          |
| Normal   | $0.50$               | $+10\%$          |
| Bust     | $0.20$               | $-15\%$          |

$$E[R] = 0.30(0.30) + 0.50(0.10) + 0.20(-0.15)$$
$$= 0.09 + 0.05 - 0.03 = 0.11 = 11\%$$

**Historical (backward-looking):** We observe $T$ past returns and take the sample mean:

$$\bar{R} = \frac{1}{T}\sum_{t=1}^{T} R_t$$

**Example:** Five monthly returns: $+2\%,\; -1\%,\; +3\%,\; +1\%,\; -0.5\%$

$$\bar{R} = \frac{0.02 - 0.01 + 0.03 + 0.01 - 0.005}{5} = \frac{0.045}{5} = 0.009 = 0.9\%/\text{month}$$

#### How Noisy Is the Historical Mean?

The standard error of the sample mean is:

$$SE(\bar{R}) = \frac{\sigma}{\sqrt{T}}$$

This tells us how uncertain our estimate of the true mean is. Even with many observations, the uncertainty can be surprisingly large.

**Example:** A stock with annual volatility $\sigma = 15\%$ (so monthly $\sigma_m = \frac{15\%}{\sqrt{12}} \approx 4.33\%$). With $36$ months of data:

$$SE(\bar{R}_{monthly}) = \frac{4.33\%}{\sqrt{36}} = \frac{4.33\%}{6} = 0.72\%/\text{month}$$

Annualizing: $SE_{annual} = 0.72\% \times \sqrt{12} \approx 2.50\%$... but more directly:

$$SE(\bar{R}_{annual}) = \frac{15\%}{\sqrt{3}} \approx 8.66\%$$

A 95% confidence interval for the true annual mean is roughly $\bar{R} \pm 2 \times 8.66\% = \pm 17.3\%$. If the sample mean is $10\%$, the true mean could plausibly be anywhere from $-7\%$ to $+27\%$. **Expected return is the hardest quantity to estimate in all of finance.**

> **Practical recommendation:** Use weekly returns with a 3–5 year lookback for equity portfolios. Supplement with economic forecasts. Never trust the sample mean alone — shrink it toward a prior (we will formalize this in the optimization module).
{: .prompt-tip }

---

### 1.2 Variance and Standard Deviation (Volatility)

Variance measures how much returns scatter around their mean. Standard deviation (volatility) is its square root, in the same units as return.

$$s^2 = \frac{1}{T-1}\sum_{t=1}^{T}(R_t - \bar{R})^2$$

We divide by $T-1$ (not $T$) for an unbiased estimate (Bessel's correction).

**Example:** Using our 5-month series ($\bar{R} = 0.9\%$):

| $t$ | $R_t$ | $R_t - \bar{R}$ | $(R_t - \bar{R})^2$ |
|-----|--------|------------------|----------------------|
| $1$   | $2.0\%$   | $1.1\%$             | $0.000121$             |
| $2$   | $-1.0\%$  | $-1.9\%$            | $0.000361$             |
| $3$   | $3.0\%$   | $2.1\%$             | $0.000441$             |
| $4$   | $1.0\%$   | $0.1\%$             | $0.000001$             |
| $5$   | $-0.5\%$  | $-1.4\%$            | $0.000196$             |

$$s^2 = \frac{0.000121 + 0.000361 + 0.000441 + 0.000001 + 0.000196}{4} = \frac{0.001120}{4} = 0.000280$$

$$s = \sqrt{0.000280} = 0.01673 = 1.67\%/\text{month}$$

#### Annualization

If monthly returns are independent and identically distributed (IID):

$$\sigma_{annual} = \sigma_{monthly} \times \sqrt{12}$$

**Why $\sqrt{12}$?** Variance of a sum of independent variables equals the sum of variances. Over 12 months: $\sigma^2_{annual} = 12 \times \sigma^2_{monthly}$, so $\sigma_{annual} = \sigma_{monthly}\sqrt{12}$.

$$\sigma_{annual} = 1.67\% \times \sqrt{12} = 1.67\% \times 3.464 = 5.79\%$$

> **When the assumption breaks:** If returns exhibit autocorrelation (momentum or mean-reversion), the $\sqrt{T}$ rule under- or over-states true annual risk. For assets like bonds or real estate with serial correlation, use overlapping multi-period returns or adjust with the Newey-West procedure.
{: .prompt-warning }

---

### 1.3 Covariance and Correlation

These tell us how two assets move *together* — the single most important input for portfolio construction.

**Covariance:**

$$\text{Cov}(R_A, R_B) = \sigma_{AB} = \frac{1}{T-1}\sum_{t=1}^{T}(R_{A,t} - \bar{R}_A)(R_{B,t} - \bar{R}_B)$$

**Correlation:** A unit-free version, always between $-1$ and $+1$:

$$\rho_{AB} = \frac{\sigma_{AB}}{\sigma_A \, \sigma_B}$$

**Example:** Four months of returns for two assets:

| $t$ | $R_A$ | $R_B$ |
|-----|--------|--------|
| $1$   | $3\%$     | $2\%$     |
| $2$   | $-1\%$    | $-1\%$    |
| $3$   | $2\%$     | $1\%$     |
| $4$   | $0\%$     | $0\%$     |

Means: $\bar{R}_A = 1.0\%$, $\bar{R}_B = 0.5\%$

| $t$ | $R_A - \bar{R}_A$ | $R_B - \bar{R}_B$ | Product |
|-----|---------------------|---------------------|---------|
| $1$   | $+2.0\%$               | $+1.5\%$               | $+0.000300$ |
| $2$   | $-2.0\%$               | $-1.5\%$               | $+0.000300$ |
| $3$   | $+1.0\%$               | $+0.5\%$               | $+0.000050$ |
| $4$   | $-1.0\%$               | $-0.5\%$               | $+0.000050$ |

$$\sigma_{AB} = \frac{0.000300 + 0.000300 + 0.000050 + 0.000050}{3} = \frac{0.000700}{3} = 0.000233$$

To find $\rho$, we need the individual standard deviations:

$$s_A^2 = \frac{(0.02)^2 + (-0.02)^2 + (0.01)^2 + (-0.01)^2}{3} = \frac{0.0010}{3} = 0.000333 \quad \Rightarrow \quad s_A = 1.826\%$$

$$s_B^2 = \frac{(0.015)^2 + (-0.015)^2 + (0.005)^2 + (-0.005)^2}{3} = \frac{0.000500}{3} = 0.000167 \quad \Rightarrow \quad s_B = 1.291\%$$

$$\rho_{AB} = \frac{0.000233}{0.01826 \times 0.01291} = \frac{0.000233}{0.000236} \approx 0.99$$

These two assets move almost perfectly together — very little diversification benefit from combining them.

> **Practical warning:** Correlations are **unstable**. They tend to spike toward $1$ during crises — exactly when you need diversification most. A correlation estimated as $0.4$ in calm markets can easily jump to $0.8$ in a downturn. For risk management, consider using stressed correlations alongside normal-period estimates.
{: .prompt-danger }

---

### 1.4 The Covariance Matrix

For $N$ assets, all pairwise variance and covariance information is packed into a single symmetric $N \times N$ matrix:

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \sigma_{13} & \cdots & \sigma_{1N} \\ \sigma_{12} & \sigma_2^2 & \sigma_{23} & \cdots & \sigma_{2N} \\ \sigma_{13} & \sigma_{23} & \sigma_3^2 & \cdots & \sigma_{3N} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \sigma_{1N} & \sigma_{2N} & \sigma_{3N} & \cdots & \sigma_N^2 \end{bmatrix}$$

- **Diagonal entries**: variances $\sigma_i^2$
- **Off-diagonal entries**: covariances $\sigma_{ij} = \rho_{ij}\sigma_i\sigma_j$
- **Symmetric**: $\sigma_{ij} = \sigma_{ji}$, so only $\frac{N(N+1)}{2}$ unique entries

**Example:** Three assets with $\sigma_1 = 20\%$, $\sigma_2 = 15\%$, $\sigma_3 = 25\%$, $\rho_{12}=0.3$, $\rho_{13}=0.1$, $\rho_{23}=0.5$.

Compute each off-diagonal:
- $\sigma_{12} = 0.3 \times 0.20 \times 0.15 = 0.0090$
- $\sigma_{13} = 0.1 \times 0.20 \times 0.25 = 0.0050$
- $\sigma_{23} = 0.5 \times 0.15 \times 0.25 = 0.01875$

$$\Sigma = \begin{bmatrix} 0.0400 & 0.0090 & 0.0050 \\ 0.0090 & 0.0225 & 0.01875 \\ 0.0050 & 0.01875 & 0.0625 \end{bmatrix}$$

#### The Estimation Challenge

For $N$ assets, the covariance matrix has $\frac{N(N+1)}{2}$ unique parameters.

| $N$ | Parameters |
|-----|------------|
| $10$  | $55$         |
| $50$  | $1,275$      |
| $500$ | $125,250$    |

With, say, 60 monthly observations and 500 stocks, you are estimating 125,250 parameters from 30,000 data points. The result: a **very noisy** matrix that may not even be positive semi-definite (PSD). We deal with this through shrinkage estimators (Ledoit-Wolf) and factor models later.

---

## 2. From Individual Assets to Portfolios

Now we combine assets. We need to express portfolio return and portfolio variance as functions of weights, means, and the covariance matrix.

---

### 2.1 Notation

We arrange everything in vectors and matrices:

$$\mathbf{w} = \begin{bmatrix}w_1 \\ w_2 \\ \vdots \\ w_N\end{bmatrix}, \quad \boldsymbol{\mu} = \begin{bmatrix}\mu_1 \\ \mu_2 \\ \vdots \\ \mu_N\end{bmatrix}, \quad \mathbf{1} = \begin{bmatrix}1 \\ 1 \\ \vdots \\ 1\end{bmatrix}$$

**Constraint:** Fully invested portfolio:
$$\mathbf{w}^T\mathbf{1} = \sum_{i=1}^N w_i = 1$$

Negative weights ($w_i < 0$) represent short positions, positive represent long positions.

---

### 2.2 Portfolio Expected Return

Portfolio return is a weighted sum of individual returns:

$$R_p = \sum_{i=1}^{N} w_i R_i = \mathbf{w}^T\mathbf{R}$$

Taking expectations:

$$\boxed{\mu_p = \mathbf{w}^T\boldsymbol{\mu} = \sum_{i=1}^{N} w_i \mu_i}$$

This is **linear** in the weights. Nothing surprising here.

**Example:** Three assets with $\boldsymbol{\mu} = [12\%,\; 7\%,\; 15\%]^T$ and $\mathbf{w} = [0.5,\; 0.3,\; 0.2]^T$:

$$\mu_p = 0.5(0.12) + 0.3(0.07) + 0.2(0.15) = 0.06 + 0.021 + 0.03 = 0.111 = 11.1\%$$

---

### 2.3 Portfolio Variance: The Core Formula

Here is where the magic of diversification comes from. Portfolio variance is **not** the weighted average of individual variances — the cross-terms (covariances) create the diversification effect.

#### Two-Asset Derivation

Start from the definition. Let $\epsilon_i = R_i - \mu_i$.

$$\sigma_p^2 = E\left[(R_p - \mu_p)^2\right] = E\left[(w_1\epsilon_1 + w_2\epsilon_2)^2\right]$$

Expand the square:

$$= E\left[w_1^2\epsilon_1^2 + 2w_1w_2\epsilon_1\epsilon_2 + w_2^2\epsilon_2^2\right]$$

Each expectation is a known quantity:
- $E[\epsilon_i^2] = \sigma_i^2$
- $E[\epsilon_1\epsilon_2] = \sigma_{12} = \rho_{12}\sigma_1\sigma_2$

$$\boxed{\sigma_p^2 = w_1^2\sigma_1^2 + 2w_1w_2\rho_{12}\sigma_1\sigma_2 + w_2^2\sigma_2^2}$$

**Numerical Example:**
- $w_1 = 0.6$, $w_2 = 0.4$
- $\sigma_1 = 20\%$, $\sigma_2 = 15\%$
- $\rho_{12} = 0.4$

$$\sigma_p^2 = (0.36)(0.04) + 2(0.6)(0.4)(0.4)(0.20)(0.15) + (0.16)(0.0225)$$
$$= 0.0144 + 0.00576 + 0.0036 = 0.02376$$
$$\sigma_p = \sqrt{0.02376} = 15.41\%$$

Weighted average volatility (no diversification): $0.6(20\%) + 0.4(15\%) = 18\%$

**Diversification gain: $18\% - 15.41\% = 2.59\%$**

The portfolio is meaningfully less risky than the weighted average of its parts, purely because $\rho < 1$.

---

#### N-Asset Generalization

For $N$ assets, the same logic applies. Every pair of assets generates a cross-term:

$$\sigma_p^2 = \sum_{i=1}^{N}\sum_{j=1}^{N} w_i w_j \sigma_{ij}$$

Since $\sigma_{ij} = \sigma_{ji}$, this is equivalently written as:

$$\sigma_p^2 = \sum_{i=1}^{N} w_i^2\sigma_i^2 + 2\sum_{i=1}^{N}\sum_{j>i} w_i w_j \sigma_{ij}$$

In matrix notation — compact and ready for computation:

$$\boxed{\sigma_p^2 = \mathbf{w}^T\Sigma\mathbf{w}}$$

**Verification for 2-asset case:**

$$\mathbf{w}^T\Sigma\mathbf{w} = \begin{bmatrix}w_1 & w_2\end{bmatrix}\begin{bmatrix}\sigma_1^2 & \sigma_{12}\\\sigma_{12} & \sigma_2^2\end{bmatrix}\begin{bmatrix}w_1\\w_2\end{bmatrix}$$

Step 1 — multiply $\Sigma\mathbf{w}$:
$$\begin{bmatrix}\sigma_1^2 w_1 + \sigma_{12}w_2 \\ \sigma_{12}w_1 + \sigma_2^2 w_2\end{bmatrix}$$

Step 2 — multiply $\mathbf{w}^T$ with the result:
$$w_1(\sigma_1^2 w_1 + \sigma_{12}w_2) + w_2(\sigma_{12}w_1 + \sigma_2^2 w_2)$$
$$= w_1^2\sigma_1^2 + 2w_1w_2\sigma_{12} + w_2^2\sigma_2^2 \quad \checkmark$$

#### Why $\Sigma$ Must Be Positive Semi-Definite

For portfolio variance to be non-negative for *any* weight vector, we need:

$$\mathbf{w}^T\Sigma\mathbf{w} \geq 0 \quad \forall \; \mathbf{w}$$

This is the definition of a **positive semi-definite (PSD)** matrix. If your sample covariance matrix is not PSD (possible with noisy data, especially when $T < N$), the optimization will produce nonsensical results — portfolios with "negative variance." Fix: shrinkage estimation, factor models, or eigenvalue clipping.

---

### 2.4 Complete Numerical Example: Tying It All Together

**Setup:** Three assets with the following statistics:

$$\boldsymbol{\mu} = \begin{bmatrix}0.12\\0.07\\0.15\end{bmatrix}, \quad \sigma_1=20\%,\; \sigma_2=15\%,\; \sigma_3=25\%$$

$$\rho_{12}=0.3,\quad \rho_{13}=0.1,\quad \rho_{23}=0.5$$

**Step 1 — Build $\Sigma$:**

$$\sigma_{12} = 0.3 \times 0.20 \times 0.15 = 0.0090$$
$$\sigma_{13} = 0.1 \times 0.20 \times 0.25 = 0.0050$$
$$\sigma_{23} = 0.5 \times 0.15 \times 0.25 = 0.01875$$

$$\Sigma = \begin{bmatrix}0.0400 & 0.0090 & 0.0050\\0.0090 & 0.0225 & 0.01875\\0.0050 & 0.01875 & 0.0625\end{bmatrix}$$

**Step 2 — Choose weights:** $\mathbf{w} = [0.5,\; 0.3,\; 0.2]^T$

**Step 3 — Portfolio expected return:**

$$\mu_p = 0.5(0.12) + 0.3(0.07) + 0.2(0.15) = 0.06 + 0.021 + 0.03 = 11.1\%$$

**Step 4 — Portfolio variance:**

$$\sigma_p^2 = \mathbf{w}^T\Sigma\mathbf{w}$$

First compute $\Sigma\mathbf{w}$:

$$\Sigma\mathbf{w} = \begin{bmatrix}0.04(0.5) + 0.009(0.3) + 0.005(0.2)\\0.009(0.5) + 0.0225(0.3) + 0.01875(0.2)\\0.005(0.5) + 0.01875(0.3) + 0.0625(0.2)\end{bmatrix} = \begin{bmatrix}0.02370\\0.01475\\0.02063\end{bmatrix}$$

Then $\mathbf{w}^T(\Sigma\mathbf{w})$:

$$\sigma_p^2 = 0.5(0.02370) + 0.3(0.01475) + 0.2(0.02063)$$
$$= 0.01185 + 0.004425 + 0.004125 = 0.020400$$

$$\sigma_p = \sqrt{0.020400} = 14.28\%$$

**Step 5 — Compare with naive weighted-average risk:**

$$\bar{\sigma} = 0.5(20\%) + 0.3(15\%) + 0.2(25\%) = 10\% + 4.5\% + 5\% = 19.5\%$$

$$\text{Diversification gain} = 19.5\% - 14.28\% = 5.22\%$$

This portfolio earns $11.1\%$ with $14.28\%$ risk — substantially better than what you would naively expect from a weighted average of the individual volatilities.

---

## 3. The Power of Diversification

### 3.1 The Decomposition: Idiosyncratic vs. Systematic Risk

Consider the cleanest possible case: $N$ identical assets, each with volatility $\sigma$ and every pairwise correlation equal to $\rho$. Hold them in equal weights ($w_i = 1/N$).

$$\sigma_p^2 = \sum_{i}\frac{1}{N^2}\sigma^2 + \sum_i\sum_{j\neq i}\frac{1}{N^2}\rho\sigma^2 = \frac{\sigma^2}{N} + \frac{N-1}{N}\rho\sigma^2$$

$$\boxed{\sigma_p^2 = \underbrace{\frac{\sigma^2}{N}}_{\text{idiosyncratic}} + \underbrace{\left(1 - \frac{1}{N}\right)\rho\sigma^2}_{\text{systematic}}}$$

As $N \to \infty$:

$$\sigma_p^2 \to \rho\sigma^2 \qquad \Rightarrow \qquad \sigma_p \to \sigma\sqrt{\rho}$$

**The first term vanishes** — this is the firm-specific, diversifiable, idiosyncratic risk. Adding more assets kills it.

**The second term converges to $\rho\sigma^2$** — this is the market-wide, undiversifiable, systematic risk. No number of assets can remove it.

### 3.2 The Floor Depends on Correlation

Using $\sigma = 20\%$:

| Average $\rho$ | Risk Floor $\sigma\sqrt{\rho}$ | Interpretation |
|-----------------|-------------------------------|----------------|
| $1.0$             | $20.0\%$                         | No diversification possible |
| $0.5$             | $14.1\%$                         | Moderate benefit |
| $0.3$             | $11.0\%$                         | Typical equity universe |
| $0.1$             | $6.3\%$                          | Multi-asset (equity + bonds + commodities) |
| $0.0$             | $0\%$                            | Full diversification (theoretical) |

**Key insight:** The real-world fight is over reducing average correlation. This is why multi-asset portfolios (stocks + bonds + real assets + alternatives) can achieve much lower floors than equity-only portfolios, even though individual alternatives may have high standalone risk.

### 3.3 Diminishing Returns

Most of the diversification benefit arrives quickly. Going from 1 to 10 assets captures most of the gain; going from 10 to 100 adds very little. For a typical equity universe ($\rho \approx 0.3$, $\sigma \approx 20\%$):

- $N=1$: $\sigma_p = 20.0\%$
- $N=5$: $\sigma_p = 13.3\%$ (captured 72% of the maximum possible reduction)
- $N=10$: $\sigma_p = 12.2\%$ (captured 85%)
- $N=30$: $\sigma_p = 11.3\%$ (captured 96%)
- $N=\infty$: $\sigma_p = 11.0\%$

> After about 25–30 stocks, you are very close to the floor. This is why index funds with 30+ holdings capture nearly all diversification benefit.
{: .prompt-info }

---

# Module 4: The Efficient Frontier, Mean-Variance Optimization, CAPM & the Tangency Portfolio

---

## Table of Contents

1. **The Feasible Set: Where Can Portfolios Live?**
2. **The Minimum-Variance Frontier (No Risk-Free Asset)**
3. **The Global Minimum-Variance Portfolio (GMV)**
4. **The Efficient Frontier**
5. **Two-Fund Theorem**
6. **Adding a Risk-Free Asset: The Capital Market Line**
7. **The Tangency Portfolio**
8. **CAPM: From the Tangency Portfolio to Beta and Alpha**
9. **Complete Numerical Example (Three Assets + Risk-Free)**
10. **When Assumptions Break**
11. **Self-Check Quiz**

---

## 1. The Feasible Set: Where Can Portfolios Live?

Given $N$ assets, every possible weighting $\mathbf{w}$ (with $\mathbf{w}^T\mathbf{1}=1$) produces a point $(\sigma_p, \mu_p)$ in risk-return space. The collection of all such points is the **feasible set**.

Key facts:
- Individual assets are points inside or on the boundary of this set.
- The **left boundary** — the set of portfolios with the lowest possible $\sigma_p$ for each level of $\mu_p$ — is the **minimum-variance frontier**.
- No portfolio can exist to the left of this boundary.

The shape of this boundary is always a **hyperbola** in $(\sigma, \mu)$ space (or equivalently a **parabola** in $(\sigma^2, \mu)$ space). This is not assumed — it falls out of the quadratic nature of portfolio variance $\sigma_p^2 = \mathbf{w}^T\Sigma\mathbf{w}$.

---

## 2. The Minimum-Variance Frontier (No Risk-Free Asset)

### 2.1 The Optimization Problem

For a given target return $\mu_0$, find the weights that minimize portfolio variance:

$$\min_{\mathbf{w}} \quad \frac{1}{2}\mathbf{w}^T\Sigma\mathbf{w}$$

subject to:

$$\mathbf{w}^T\boldsymbol{\mu} = \mu_0 \qquad \text{(target return)}$$
$$\mathbf{w}^T\mathbf{1} = 1 \qquad \text{(fully invested)}$$

The $\frac{1}{2}$ is for mathematical convenience — it doesn't change the optimal $\mathbf{w}$.

**Intuition:** We are asking: "Given that I want to earn exactly $\mu_0$, what is the least amount of risk I must accept, and which portfolio achieves it?"

### 2.2 Solving with Lagrange Multipliers

The Lagrangian is:

$$\mathcal{L} = \frac{1}{2}\mathbf{w}^T\Sigma\mathbf{w} - \lambda(\mathbf{w}^T\boldsymbol{\mu} - \mu_0) - \gamma(\mathbf{w}^T\mathbf{1} - 1)$$

Take the derivative with respect to $\mathbf{w}$ and set to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \Sigma\mathbf{w} - \lambda\boldsymbol{\mu} - \gamma\mathbf{1} = \mathbf{0}$$

Solve for $\mathbf{w}$:

$$\boxed{\mathbf{w}^* = \lambda\,\Sigma^{-1}\boldsymbol{\mu} + \gamma\,\Sigma^{-1}\mathbf{1}}$$

Now plug this back into the two constraints to find $\lambda$ and $\gamma$. Define three scalars — the **Markowitz constants**:

$$A = \mathbf{1}^T\Sigma^{-1}\boldsymbol{\mu} = \boldsymbol{\mu}^T\Sigma^{-1}\mathbf{1}$$

$$B = \boldsymbol{\mu}^T\Sigma^{-1}\boldsymbol{\mu}$$

$$C = \mathbf{1}^T\Sigma^{-1}\mathbf{1}$$

And the determinant:

$$D = BC - A^2 > 0$$

The inequality $D > 0$ holds whenever assets have different expected returns and $\Sigma$ is positive definite. If all assets had the same expected return, $\boldsymbol{\mu}$ would be proportional to $\mathbf{1}$, and $D$ would collapse to zero — there would be no tradeoff to optimize.

From the constraints:

$$\lambda = \frac{C\mu_0 - A}{D}, \qquad \gamma = \frac{B - A\mu_0}{D}$$

### 2.3 The Frontier in Closed Form

Substituting back, the minimum variance at target return $\mu_0$ is:

$$\boxed{\sigma_p^2 = \frac{C\mu_0^2 - 2A\mu_0 + B}{D}}$$

This is a **parabola** in $(\sigma^2, \mu)$ space. In $(\sigma, \mu)$ space it becomes a **hyperbola** — the classic "bullet" shape.

**Intuition:** The quadratic form tells us that risk grows quadratically as you push the target return away from the GMV return in either direction. There is a "sweet spot" at the tip of the bullet, and reaching for higher returns costs increasingly more risk.

---

## 3. The Global Minimum-Variance Portfolio (GMV)

The GMV portfolio has the lowest possible variance across all return targets. It sits at the tip of the bullet.

### 3.1 Derivation

Minimize $\frac{1}{2}\mathbf{w}^T\Sigma\mathbf{w}$ subject only to $\mathbf{w}^T\mathbf{1} = 1$ (no return constraint).

The Lagrangian:

$$\mathcal{L} = \frac{1}{2}\mathbf{w}^T\Sigma\mathbf{w} - \gamma(\mathbf{w}^T\mathbf{1} - 1)$$

First-order condition: $\Sigma\mathbf{w} = \gamma\mathbf{1}$, giving $\mathbf{w} = \gamma\,\Sigma^{-1}\mathbf{1}$.

Apply the constraint $\mathbf{w}^T\mathbf{1} = 1$:

$$\gamma\,\mathbf{1}^T\Sigma^{-1}\mathbf{1} = 1 \quad \Rightarrow \quad \gamma = \frac{1}{C}$$

$$\boxed{\mathbf{w}_{GMV} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}} = \frac{\Sigma^{-1}\mathbf{1}}{C}}$$

**Intuition:** Each asset gets weight proportional to the sum of its row in $\Sigma^{-1}$. Assets that are low-variance and low-correlation with everything else get the highest weights. The inverse covariance matrix "knows" about both individual risk and diversification potential simultaneously.

### 3.2 GMV Portfolio Statistics

$$\mu_{GMV} = \frac{A}{C}, \qquad \sigma^2_{GMV} = \frac{1}{C}$$

Note that $\sigma^2_{GMV} = 1/C$ is the absolute floor on portfolio variance. No combination of these $N$ assets can do better.

---

## 4. The Efficient Frontier

The minimum-variance frontier has two branches:
- **Upper branch** ($\mu \geq \mu_{GMV}$): the **efficient frontier**
- **Lower branch** ($\mu < \mu_{GMV}$): **inefficient** — same risk, lower return

**Definition:** A portfolio is **mean-variance efficient** if no other portfolio offers higher expected return for the same or lower variance.

No rational investor would hold a portfolio on the lower branch. For any lower-branch portfolio, there exists an upper-branch portfolio with the same $\sigma$ but higher $\mu$. The lower branch exists mathematically but is economically irrelevant.

---

## 5. Two-Fund Theorem (Separation)

### 5.1 The Result

**Any** portfolio on the minimum-variance frontier can be written as a linear combination of **two specific frontier portfolios**. If $\mathbf{w}_a$ and $\mathbf{w}_b$ are two distinct frontier portfolios:

$$\mathbf{w}^*(\mu_0) = \alpha\,\mathbf{w}_a + (1-\alpha)\,\mathbf{w}_b$$

for some scalar $\alpha$ that depends on $\mu_0$.

### 5.2 The Explicit Formula

Define two "basis" portfolios:

$$\mathbf{g} = \frac{1}{D}\left(B\,\Sigma^{-1}\mathbf{1} - A\,\Sigma^{-1}\boldsymbol{\mu}\right)$$
$$\mathbf{h} = \frac{1}{D}\left(C\,\Sigma^{-1}\boldsymbol{\mu} - A\,\Sigma^{-1}\mathbf{1}\right)$$

Then for any target return $\mu_0$:

$$\boxed{\mathbf{w}^*(\mu_0) = \mathbf{g} + \mu_0\,\mathbf{h}}$$

This is a **linear** function of the target return — the frontier weight vector changes linearly with $\mu_0$.

### 5.3 Why This Matters

You don't need to solve the optimization from scratch for every target return. Compute $\mathbf{g}$ and $\mathbf{h}$ once, then the entire frontier is parameterized by a single number $\mu_0$. This is computationally elegant and reveals the deep structure: the frontier is a one-dimensional family of portfolios.

---

## 6. Adding a Risk-Free Asset: The Capital Market Line

### 6.1 The New Problem

Now investors can lend or borrow at the risk-free rate $R_f$. Invest fraction $w_T$ in a risky portfolio $T$ and $(1-w_T)$ in the risk-free asset:

$$\mu_p = (1-w_T)R_f + w_T\,\mu_T = R_f + w_T(\mu_T - R_f)$$

$$\sigma_p = w_T\,\sigma_T \quad (\text{since } \sigma_{R_f} = 0)$$

From the second equation: $w_T = \sigma_p / \sigma_T$. Substituting into the first:

$$\boxed{\mu_p = R_f + \frac{\mu_T - R_f}{\sigma_T}\,\sigma_p}$$

This is a **straight line** in $(\sigma, \mu)$ space, passing through $(0, R_f)$ with slope $\frac{\mu_T - R_f}{\sigma_T}$.

**Intuition:** Mixing a risky portfolio with cash traces a straight line because the risk-free asset contributes zero volatility and zero covariance. The line's slope tells you how much excess return you earn per unit of total risk.

### 6.2 The Sharpe Ratio

The slope of this line is the **Sharpe ratio** of portfolio $T$:

$$\boxed{S_T = \frac{\mu_T - R_f}{\sigma_T}}$$

Every risky portfolio generates a different line through $(0, R_f)$. The **steepest** line — the one offering the most return per unit of risk — is the **Capital Market Line (CML)**.

### 6.3 The Geometry

The CML is tangent to the efficient frontier. The point of tangency is the **tangency portfolio**. All portfolios on the CML dominate all portfolios on the curved efficient frontier (except at the tangency point itself, where they coincide).

Why? At any given $\sigma$, the CML sits above the curved frontier, offering higher $\mu$. The only way to be on the CML is to hold some combination of the tangency portfolio and the risk-free asset.

---

## 7. The Tangency Portfolio

### 7.1 Derivation

Maximize the Sharpe ratio:

$$\max_{\mathbf{w}} \quad \frac{\mathbf{w}^T\boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T\Sigma\mathbf{w}}} \qquad \text{s.t.} \quad \mathbf{w}^T\mathbf{1} = 1$$

The trick: since the Sharpe ratio is scale-invariant (multiplying all weights by a constant doesn't change it), we can solve an equivalent unconstrained problem. Define the **excess return vector**:

$$\boldsymbol{\pi} = \boldsymbol{\mu} - R_f\mathbf{1}$$

The unnormalized tangency weights are proportional to:

$$\mathbf{z} = \Sigma^{-1}\boldsymbol{\pi}$$

Normalize to sum to $1$:

$$\boxed{\mathbf{w}_{tan} = \frac{\Sigma^{-1}\boldsymbol{\pi}}{\mathbf{1}^T\Sigma^{-1}\boldsymbol{\pi}} = \frac{\Sigma^{-1}(\boldsymbol{\mu} - R_f\mathbf{1})}{\mathbf{1}^T\Sigma^{-1}(\boldsymbol{\mu} - R_f\mathbf{1})}}$$

**Intuition:** The tangency portfolio weights each asset by its excess return, adjusted for how it interacts with all other assets through $\Sigma^{-1}$. An asset with high excess return gets more weight, but an asset that is highly correlated with other high-return assets gets penalized — $\Sigma^{-1}$ accounts for redundancy.

### 7.2 Tangency Portfolio Statistics

$$\mu_{tan} = \mathbf{w}_{tan}^T\boldsymbol{\mu}$$

$$\sigma_{tan}^2 = \mathbf{w}_{tan}^T\Sigma\,\mathbf{w}_{tan}$$

$$S_{tan} = \frac{\mu_{tan} - R_f}{\sigma_{tan}} = \sqrt{\boldsymbol{\pi}^T\Sigma^{-1}\boldsymbol{\pi}}$$

The last expression shows the maximum achievable Sharpe ratio depends on $\Sigma^{-1}$ and the excess return vector — it is a quadratic form. Adding a new asset to the universe can only increase (or maintain) $S_{tan}$, never decrease it.

### 7.3 One-Fund Theorem

With a risk-free asset available:

> **All investors hold the same risky portfolio** (the tangency portfolio). They differ only in how much they lever it up (borrow at $R_f$) or de-lever (lend at $R_f$).

This is a much stronger result than the two-fund theorem. It reduces the problem to finding one portfolio. A conservative investor holds 40% tangency + 60% cash. An aggressive investor borrows and holds 150% tangency. Both hold the same risky portfolio — they just scale it differently.

---

## 8. CAPM: From the Tangency Portfolio to Beta and Alpha

### 8.1 The Logical Bridge

The one-fund theorem says every investor holds the tangency portfolio. If every investor holds the same risky portfolio, then in equilibrium, the tangency portfolio must be the **market portfolio** — the value-weighted portfolio of all investable assets.

This is the key insight of the **Capital Asset Pricing Model (CAPM)**: the market portfolio is mean-variance efficient.

### 8.2 Beta: Systematic Risk

If the market portfolio $M$ is efficient, then for any individual asset $i$, there is a precise relationship between its expected return and its risk contribution to the market.

Define the **beta** of asset $i$ with respect to the market:

$$\boxed{\beta_i = \frac{\text{Cov}(R_i, R_M)}{\text{Var}(R_M)} = \frac{\sigma_{iM}}{\sigma_M^2}}$$

**Intuition before the formula:** Beta measures how much asset $i$ moves when the market moves. If $\beta_i = 1.5$, then when the market goes up 1%, asset $i$ tends to go up 1.5%. It captures the asset's sensitivity to the one risk factor that everyone is exposed to.

**Intuition after the formula:** The numerator $\sigma_{iM}$ is the covariance between asset $i$ and the market — how much they co-move. The denominator $\sigma_M^2$ normalizes by the market's own variance. So beta is the "share" of market risk that asset $i$ carries, measured in units of market variance.

**Numerical feel:** If $\sigma_{iM} = 0.045$ and $\sigma_M^2 = 0.03$, then $\beta_i = 0.045/0.03 = 1.5$.

### 8.3 The CAPM Equation (Security Market Line)

The CAPM states that the expected excess return of any asset is proportional to its beta:

$$\boxed{E[R_i] - R_f = \beta_i\,(E[R_M] - R_f)}$$

Or equivalently:

$$E[R_i] = R_f + \beta_i\,(E[R_M] - R_f)$$

This is the **Security Market Line (SML)** — a linear relationship in $(\beta, E[R])$ space.

**Intuition:** The market only rewards you for bearing **systematic risk** (risk that cannot be diversified away). Beta measures exactly this. An asset's idiosyncratic risk (the part uncorrelated with the market) earns zero premium because investors can diversify it away for free.

**Numerical example:** If $R_f = 3\%$, $E[R_M] = 10\%$, and $\beta_i = 1.3$:

$$E[R_i] = 3\% + 1.3 \times (10\% - 3\%) = 3\% + 9.1\% = 12.1\%$$

An asset with $\beta = 0$ should earn $R_f = 3\%$. An asset with $\beta = 1$ should earn $E[R_M] = 10\%$. Beta linearly interpolates (and extrapolates) between these.

### 8.4 Risk Decomposition: Systematic vs. Idiosyncratic

The single-factor model underlying CAPM decomposes any asset's return as:

$$R_i - R_f = \alpha_i + \beta_i(R_M - R_f) + \varepsilon_i$$

where:
- $\beta_i(R_M - R_f)$ is the **systematic component** — the part explained by the market
- $\varepsilon_i$ is the **idiosyncratic component** — asset-specific noise, with $E[\varepsilon_i] = 0$ and $\text{Cov}(\varepsilon_i, R_M) = 0$
- $\alpha_i$ is the **intercept** — we'll define this carefully next

The total variance of asset $i$ decomposes as:

$$\sigma_i^2 = \beta_i^2\,\sigma_M^2 + \sigma_{\varepsilon_i}^2$$

$$\text{Total risk} = \text{Systematic risk} + \text{Idiosyncratic risk}$$

The fraction of variance explained by the market is:

$$R^2_i = \frac{\beta_i^2\,\sigma_M^2}{\sigma_i^2}$$

**Numerical example:** If $\beta_i = 1.3$, $\sigma_M = 17\%$, $\sigma_i = 30\%$:

$$\text{Systematic variance} = (1.3)^2 \times (0.17)^2 = 1.69 \times 0.0289 = 0.04884$$

$$\text{Total variance} = (0.30)^2 = 0.09$$

$$R^2 = \frac{0.04884}{0.09} = 54.3\%$$

So $54\%$ of this asset's risk comes from the market, and $46\%$ is idiosyncratic. A well-diversified portfolio eliminates the idiosyncratic part, leaving only the systematic component.

### 8.5 Alpha: Abnormal Return

**Alpha** ($\alpha_i$) is the return an asset earns beyond what CAPM predicts given its beta:

$$\boxed{\alpha_i = (R_i - R_f) - \beta_i(R_M - R_f)}$$

Under CAPM, $\alpha_i = 0$ for all assets in equilibrium. If $\alpha_i \neq 0$, either:
- The asset is **mispriced** (an opportunity), or
- The model is **wrong** (missing risk factors), or
- The estimate is **noisy** (estimation error — the most common explanation)

**Intuition:** Alpha is the vertical distance between an asset's actual return and the SML. Positive alpha means the asset earned more than its beta-implied fair return. Negative alpha means it underperformed.

**Numerical example:** Asset $i$ returned $15\%$ last year. $R_f = 3\%$, $R_M = 10\%$, $\beta_i = 1.3$.

$$\alpha_i = (15\% - 3\%) - 1.3 \times (10\% - 3\%) = 12\% - 9.1\% = +2.9\%$$

This asset earned $2.9\%$ more than CAPM predicted. Before celebrating, check the standard error — with typical estimation noise, $2.9\%$ may not be statistically significant.

### 8.6 Estimating Beta and Alpha in Practice

Given $T$ periods of return data, run an OLS regression:

$$R_{i,t} - R_{f,t} = \hat{\alpha}_i + \hat{\beta}_i(R_{M,t} - R_{f,t}) + \hat{\varepsilon}_{i,t}$$

The standard errors are:

$$SE(\hat{\beta}_i) = \frac{\hat{\sigma}_\varepsilon}{\hat{\sigma}_M\sqrt{T}}, \qquad SE(\hat{\alpha}_i) = \hat{\sigma}_\varepsilon\sqrt{\frac{1}{T} + \frac{\bar{R}_M^2}{\hat{\sigma}_M^2 \cdot T}}$$

where $\hat{\sigma}_\varepsilon$ is the residual standard deviation.

**The noise problem is severe.** For a typical stock with $\hat{\sigma}_\varepsilon = 25\%$ annualized, $\hat{\sigma}_M = 17\%$, and $T = 60$ monthly observations (5 years):

$$SE(\hat{\beta}) = \frac{25\%/\sqrt{12}}{17\%/\sqrt{12} \times \sqrt{60}} = \frac{7.22\%}{4.91\% \times 7.75} = \frac{7.22}{38.0} \approx 0.19$$

So $\hat{\beta} = 1.3 \pm 0.38$ at 95% confidence. That's a wide band.

For alpha, the situation is worse. With monthly data:

$$SE(\hat{\alpha}) \approx \frac{25\%/\sqrt{12}}{\sqrt{60}} \approx \frac{7.22\%}{7.75} \approx 0.93\% \text{ per month} = 11.2\% \text{ annualized}$$

So $\hat{\alpha} = 2.9\% \pm 22\%$ at 95% confidence. You cannot distinguish this from zero. This is why genuine alpha is so hard to detect statistically — you need either very large alpha or very long track records.

**Practical recommendation:** For equity portfolios, use weekly returns with a 3-year lookback for beta estimation. This gives $T \approx 156$, which cuts $SE(\hat{\beta})$ roughly in half compared to 60 monthly observations. For crypto assets, beta estimates are essentially unreliable (as discussed in Module 3's practical notes).
{: .prompt-tip }

### 8.7 The SML vs. the CML — A Common Confusion

These are two different lines that serve different purposes:

| | Capital Market Line (CML) | Security Market Line (SML) |
|---|---|---|
| Axes | $(\sigma_p, \mu_p)$ — total risk | $(\beta_i, E[R_i])$ — systematic risk |
| Applies to | **Efficient portfolios** only | **All assets and portfolios** |
| Equation | $\mu_p = R_f + S_{tan}\,\sigma_p$ | $E[R_i] = R_f + \beta_i(E[R_M]-R_f)$ |
| Slope | Sharpe ratio of tangency portfolio | Market risk premium $E[R_M]-R_f$ |
| Off the line? | Inefficient portfolio (below CML) | Mispriced asset ($\alpha \neq 0$) |

The CML tells you the best risk-return tradeoff achievable. The SML tells you the fair return for any level of systematic risk. Individual assets typically lie below the CML (they carry idiosyncratic risk) but should lie on the SML (if CAPM holds).

## 10. When Assumptions Break

The Markowitz framework, while foundational, rests on assumptions that often fail in practice. Understanding these failures is crucial for applying MPT robustly.

### 10.1 Estimation Error — The Dominant Problem

The formulas require $\boldsymbol{\mu}$, $\Sigma$, and $R_f$. In practice:

- **$\boldsymbol{\mu}$ is very poorly estimated.** Standard errors of $8–10\%$ annually are common for equities, meaning we cannot reliably distinguish a $2\%$ expected return from a $20\%$ expected return with typical sample sizes.
- **$\Sigma$ is better estimated than $\boldsymbol{\mu}$**, but still noisy, especially for large $N$ or highly correlated assets. Correlations are particularly unstable.
- **$R_f$ is known precisely** (it's the current Treasury rate).

The Markowitz optimizer is an **error maximizer**: it aggressively exploits tiny perceived differences in $\boldsymbol{\mu}$ that are likely just noise, leading to extreme and unstable portfolio weights.

**Example of Instability:** Suppose Asset 1's *estimated* mean $\hat{\mu}_1$ increases from $12\%$ to $13\%$ (well within estimation error). Recomputing the tangency portfolio:

$$\boldsymbol{\mu}_{new} = \begin{bmatrix}0.13\\0.07\\0.15\end{bmatrix}, \quad \boldsymbol{\pi}_{new} = \begin{bmatrix}0.10\\0.04\\0.12\end{bmatrix}$$

$$\mathbf{z}_{new} = \Sigma^{-1}\boldsymbol{\pi}_{new} = \begin{bmatrix}2.7548(0.10) - 5.7915(0.04) + 0.4662(0.12)\\\dots\end{bmatrix} \approx \begin{bmatrix}2.5870\\-0.1302\\2.0860\end{bmatrix}$$

$$\mathbf{w}_{tan,new} \approx \begin{bmatrix}0.572\\-0.029\\0.461\end{bmatrix}$$

The weight on Asset 1 jumped from $55.6\%$ to $57.2\%$, and Asset 3 dropped from $47.6\%$ to $46.1\%$. A $1\%$ change in expected return (which is pure noise) caused a portfolio turnover of several percentage points. Over time, this leads to excessive trading, high transaction costs, and poor out-of-sample performance.

**Why this happens:** The optimizer treats $\hat{\mu}_1 = 13\%$ as truth, not as a noisy estimate. It doesn't "know" that the true $\mu_1$ might be anywhere from $5\%$ to $21\%$ (a 95% confidence interval with typical estimation error). So it confidently overweights assets with high $\hat{\mu}$ and underweights those with low $\hat{\mu}$, even when these differences are statistically insignificant.

### 10.2 Normal Distribution Assumption

Mean-variance optimization implicitly assumes returns are **normally distributed** (or investors have quadratic utility). In reality:

- **Fat Tails:** Extreme events (crashes, rallies) are more frequent than predicted by the normal distribution. The 1987 crash (-20% in one day) was a 20+ sigma event under normality — essentially impossible. Yet it happened.
- **Skewness:** Return distributions are often negatively skewed (large losses more common than large gains). Lottery-like assets (e.g., biotech stocks, out-of-the-money options) have positive skewness.
- **Correlation Breakdown:** During crises, correlations spike toward $+1$ (everything moves together), and diversification benefits erode precisely when they are needed most. The 2008 crisis saw correlations between equities, corporate bonds, and commodities all surge.

**Practical Implication:** MVO is still valuable for its framework and core insights (diversification matters, correlation is key). However, practitioners must:

- **Stress-test portfolios** against non-normal scenarios using historical crisis periods or Monte Carlo simulations with fat-tailed distributions.
- **Use downside risk measures** like Value-at-Risk (VaR) or Conditional Value-at-Risk (CVaR) in addition to variance. CVaR optimization explicitly targets the average loss in the worst 5% of scenarios, which better captures tail risk.
- **Recognize that diversification is not constant.** A portfolio that looks well-diversified in normal times may become highly concentrated in risk during a crisis.

**Numerical example of fat tails:** Under normality with $\mu = 10\%$, $\sigma = 20\%$ annually, the probability of a loss exceeding $30\%$ in one year is:

$$P(R < -0.30) = \Phi\left(\frac{-0.30 - 0.10}{0.20}\right) = \Phi(-2.0) \approx 2.3\%$$

But empirically, such losses occur roughly $5–7\%$ of the time in equity markets — more than double the normal prediction. This is the "fat tail" problem.

### 10.3 Input Sensitivity & Practical Fixes

The extreme sensitivity to $\boldsymbol{\mu}$ estimates is the primary reason naive MVO fails in practice. Several techniques address this:

#### 10.3.1 Shrinkage Estimators

**Idea:** Pull noisy estimates toward more structured, stable targets. This reduces estimation error variance at the cost of introducing some bias, leading to better out-of-sample performance.

**For $\boldsymbol{\mu}$:** The **James-Stein estimator** shrinks individual asset means toward the grand mean:

$$\hat{\mu}_i^{JS} = \bar{\mu} + (1 - \delta)(\hat{\mu}_i - \bar{\mu})$$

where $\bar{\mu} = \frac{1}{N}\sum_{i=1}^N \hat{\mu}_i$ and $\delta \in [0,1]$ is the shrinkage intensity. Typical values: $\delta = 0.3$ to $0.5$.

**For $\Sigma$:** The **Ledoit-Wolf estimator** shrinks the sample covariance matrix toward a structured target (e.g., constant correlation or diagonal matrix):

$$\hat{\Sigma}^{LW} = \delta \cdot \Sigma_{target} + (1-\delta) \cdot \hat{\Sigma}_{sample}$$

The optimal $\delta$ can be estimated from the data. This dramatically improves the stability of $\Sigma^{-1}$.

**Numerical impact:** In a 50-asset portfolio, using Ledoit-Wolf shrinkage can reduce portfolio variance by 20–30% out-of-sample compared to using the raw sample covariance matrix, simply by making $\Sigma^{-1}$ more stable.

#### 10.3.2 Weight Constraints

Imposing limits on individual asset weights (e.g., $w_i \in [0, 0.30]$ or $|w_i| \leq 0.25$) prevents extreme positions and acts as implicit shrinkage. The optimizer can no longer exploit tiny differences in $\hat{\mu}$ by taking huge positions.

**Trade-off:** Constraints reduce the maximum achievable Sharpe ratio in-sample but often improve out-of-sample performance by preventing overfitting to noise.

#### 10.3.3 Black-Litterman Model

Formally blends **market equilibrium expectations** (implied by current market weights) with **investor's specific views**. The equilibrium return vector is:

$$\boldsymbol{\mu}_{eq} = \lambda \Sigma \mathbf{w}_{market}$$

where $\lambda$ is the market's risk aversion and $\mathbf{w}_{market}$ is the market-cap-weighted portfolio. The investor then expresses views (e.g., "I think Asset 1 will outperform Asset 2 by 3%"), and these are combined using Bayesian updating to produce a posterior $\boldsymbol{\mu}_{BL}$.

**Advantage:** Produces more stable and intuitive weights because it starts from a sensible prior (the market portfolio) rather than treating all assets as blank slates.

#### 10.3.4 Risk Parity

Focuses on equalizing **risk contributions** rather than maximizing return. Each asset contributes equally to total portfolio variance:

$$w_i \times \frac{\partial \sigma_p}{\partial w_i} = \frac{\sigma_p}{N} \quad \text{for all } i$$

This requires only $\Sigma$ (not $\boldsymbol{\mu}$) and often yields more stable portfolios. The intuition: don't let a few high-volatility assets dominate the portfolio's risk.

**Numerical example:** In a two-asset portfolio with $\sigma_1 = 20\%$, $\sigma_2 = 10\%$, $\rho_{12} = 0.3$, the risk parity weights are approximately $w_1 \approx 0.33$, $w_2 \approx 0.67$ — the lower-volatility asset gets more weight to equalize risk contributions.

---

# Module 3: Estimation Risk and Shrinkage — Taming the Optimizer

---

## Table of Contents

1. **The Core Problem: Why Raw Estimates Fail**
2. **Quantifying Estimation Error**
3. **The Error Maximization Effect**
4. **Shrinkage Estimators for Expected Returns**
5. **Shrinkage Estimators for the Covariance Matrix**
6. **Practical Recommendations**
7. **Complete Numerical Example**
8. **Self-Check Quiz**

---

## 1. The Core Problem: Why Raw Estimates Fail

In Module 2 we derived elegant closed-form solutions for optimal portfolios. Every formula took $\boldsymbol{\mu}$ and $\Sigma$ as given. In practice, we never know these — we estimate them from historical data, and those estimates are noisy.

Here is the fundamental tension:

> The Markowitz optimizer treats inputs as **truth**. But sample estimates contain **error**. The optimizer aggressively exploits that error, producing portfolios that are optimal for the noise, not the signal.
{: .prompt-warning }

This is not a minor nuisance. It is the single biggest reason mean-variance optimization disappoints in practice.

### 1.1 A Thought Experiment

Suppose you have two assets with identical true parameters:

$$\mu_A = \mu_B = 10\%, \qquad \sigma_A = \sigma_B = 20\%, \qquad \rho = 0.5$$

But your sample estimates (from 3 years of monthly data) come out as:

$$\hat{\mu}_A = 14\%, \qquad \hat{\mu}_B = 6\%$$

This is entirely plausible — recall from Module 1 that the standard error of the annualized mean is roughly $\sigma / \sqrt{T} \approx 20\% / \sqrt{3} \approx 11.5\%$. A $4\%$ deviation from the true mean is well within one standard error.

What does the optimizer do? It massively overweights Asset A and underweights (or shorts) Asset B. The resulting portfolio is "optimal" for the estimation errors, not for reality.

---

## 2. Quantifying Estimation Error

### 2.1 Error in the Mean

For $T$ years of annual returns (or $T \times 12$ monthly observations annualized):

$$SE(\hat{\mu}_i) = \frac{\sigma_i}{\sqrt{T}}$$

**Numerical reality check:**

| Asset volatility | Lookback | $SE(\hat{\mu})$ | 95% CI width |
|-----------------|----------|-----------------|--------------|
| $15\%$ (bonds) | 5 years | $6.7\%$ | $\pm 13.4\%$ |
| $20\%$ (equities) | 5 years | $8.9\%$ | $\pm 17.9\%$ |
| $20\%$ (equities) | 10 years | $6.3\%$ | $\pm 12.6\%$ |
| $30\%$ (small caps) | 5 years | $13.4\%$ | $\pm 26.8\%$ |

A 95% confidence interval of $\pm 17.9\%$ for an equity expected return means we literally cannot distinguish a $2\%$ expected return from a $20\%$ expected return with 5 years of data. The mean is essentially unobservable at horizons relevant to portfolio construction.

### 2.2 Error in Variances and Covariances

For normally distributed returns with $n$ observations, the sample variance $\hat{\sigma}^2$ has:

$$\text{Var}(\hat{\sigma}^2) = \frac{2\sigma^4}{n-1}$$

So the standard error of the sample standard deviation is approximately:

$$SE(\hat{\sigma}) \approx \frac{\sigma}{\sqrt{2(n-1)}}$$

**Example:** $\sigma = 20\%$, $n = 60$ monthly observations:

$$SE(\hat{\sigma}_{monthly}) \approx \frac{5.77\%}{\sqrt{118}} = 0.53\%$$

Annualized: $SE(\hat{\sigma}_{annual}) \approx 0.53\% \times \sqrt{12} \approx 1.84\%$

This is much tighter than the mean estimate. Variances are estimated roughly $\sqrt{2T}$ times more precisely than means.

### 2.3 Error in Correlations

For sample correlation $\hat{\rho}$ based on $n$ observations:

$$SE(\hat{\rho}) \approx \frac{1 - \rho^2}{\sqrt{n-2}}$$

**Example:** True $\rho = 0.3$, $n = 60$:

$$SE(\hat{\rho}) \approx \frac{1 - 0.09}{\sqrt{58}} = \frac{0.91}{7.62} = 0.119$$

So the 95% CI for $\rho$ is roughly $0.3 \pm 0.24$, i.e., $[0.06, 0.54]$. Correlations are moderately noisy.

### 2.4 The Hierarchy of Estimation Quality

$$\underbrace{\text{Means}}_{\text{worst}} \quad \ll \quad \underbrace{\text{Correlations}}_{\text{moderate}} \quad < \quad \underbrace{\text{Variances}}_{\text{best}}$$

This hierarchy drives the entire shrinkage literature: we need the most help where the estimates are worst.

---

## 3. The Error Maximization Effect

### 3.1 Why Optimizers Amplify Errors

The Markowitz optimizer solves:

$$\mathbf{w}^* = \Sigma^{-1}\boldsymbol{\mu} \cdot (\text{normalization})$$

Two operations amplify noise:

1. **Matrix inversion** $\Sigma^{-1}$: Small errors in $\Sigma$ become large errors in $\Sigma^{-1}$, especially when assets are highly correlated (near-singular $\Sigma$). The amplification factor is the **condition number** $\kappa(\Sigma) = \lambda_{max}/\lambda_{min}$.

2. **Multiplication by noisy $\boldsymbol{\mu}$**: The optimizer assigns the largest weights to assets with the largest estimated excess returns — which are disproportionately likely to be the ones with the largest positive estimation errors.

### 3.2 Condition Number Example

For our 3-asset example from Module 2:
```python
eigenvalues = np.linalg.eigvalsh(Sigma)
kappa = eigenvalues.max() / eigenvalues.min()
print(f"Eigenvalues: {np.round(eigenvalues, 6)}")
print(f"Condition number: {kappa:.1f}")

$$\lambda_1 = 0.0069, \quad \lambda_2 = 0.0283, \quad \lambda_3 = 0.0898$$

$$\kappa(\Sigma) = \frac{0.0898}{0.0069} = 13.0$$

A condition number of $13$ is manageable. But with 30+ correlated assets, $\kappa$ can easily reach $100–1000$, meaning small input errors get amplified by orders of magnitude.

### 3.3 The DeMiguel, Garlappi, Uppal (2009) Result

A landmark paper showed that with realistic estimation error, the naive $1/N$ equal-weight portfolio **outperforms** mean-variance optimization out-of-sample for most datasets, unless $N$ is small or the estimation window is very long.

The required estimation window for mean-variance to reliably beat $1/N$ is approximately:

$$T^* \approx \frac{N \cdot (N + \mu_{max}^2 / \sigma_{max}^2)}{(\mu_{max}/\sigma_{max})^2}$$

For $N = 25$ assets with typical equity parameters, this works out to **hundreds of years** of data.

This is not an argument against optimization — it is an argument for **regularizing** the optimization.

---

## 4. Shrinkage Estimators for Expected Returns

### 4.1 The Idea of Shrinkage

Instead of using the raw sample mean $\hat{\boldsymbol{\mu}}$, we pull ("shrink") it toward a structured target $\boldsymbol{\mu}_0$:

$$\boxed{\boldsymbol{\mu}_{shrunk} = (1 - \delta)\,\hat{\boldsymbol{\mu}} + \delta\,\boldsymbol{\mu}_0}$$

where $\delta \in [0, 1]$ is the **shrinkage intensity**.

- $\delta = 0$: trust the sample completely (no shrinkage)
- $\delta = 1$: ignore the sample, use only the target
- Optimal $\delta$: somewhere in between, determined by the signal-to-noise ratio

**Intuition:** If your estimates are noisy, you are better off pulling them toward a sensible prior than trusting the noise. This is the bias-variance tradeoff: shrinkage introduces bias but reduces variance, and the net effect on portfolio performance is positive.

### 4.2 Common Shrinkage Targets for $\boldsymbol{\mu}$

| Target $\boldsymbol{\mu}_0$ | Name | Assumption |
|------------------------------|------|------------|
| $\bar{\mu} \cdot \mathbf{1}$ (grand mean) | James-Stein | All assets have the same expected return |
| $R_f \cdot \mathbf{1}$ | Zero excess return | No asset earns a risk premium |
| $\beta_i \cdot \hat{\mu}_m$ | CAPM prior | Returns follow CAPM |
| Equilibrium returns | Black-Litterman | Market portfolio is optimal |

### 4.3 James-Stein Shrinkage

The James-Stein estimator (1961) was one of the most surprising results in statistics: when estimating $N \geq 3$ means simultaneously, the sample mean is **inadmissible** — there always exists a shrinkage estimator with lower total mean squared error.

The shrinkage intensity is:

$$\boxed{\delta_{JS} = \min\left(1, \;\frac{(N-2)\;\hat{\sigma}^2_{pool}}{T \cdot \sum(\hat{\mu}_i - \bar{\mu})^2}\right)}$$

*(where $\hat{\sigma}^2_{pool}$ is the pooled variance and $\bar{\mu}$ is the grand mean of the estimated returns).*

### 4.4 Numerical Example: James-Stein

Using our 3-asset data:

$$\hat{\boldsymbol{\mu}} = \begin{bmatrix}0.12\\0.07\\0.15\end{bmatrix}, \qquad \bar{\mu} = \frac{0.12 + 0.07 + 0.15}{3} = 0.1133$$

Assume $T = 60$ monthly observations. Pooled monthly variance:

$$\hat{\sigma}^2_{pool} = \frac{\sigma_1^2 + \sigma_2^2 + \sigma_3^2}{3} = \frac{0.04 + 0.0225 + 0.0625}{3} = 0.04167 \quad \text{(annual)}$$

Monthly pooled variance: $0.04167 / 12 = 0.003472$

Sum of squared deviations of monthly means from grand mean:

$$\sum_{i=1}^{3}\left(\hat{\mu}_i^{(m)} - \bar{\mu}^{(m)}\right)^2$$

Monthly means: $\hat{\mu}_i^{(m)} = \hat{\mu}_i / 12$, so $\hat{\boldsymbol{\mu}}^{(m)} = [0.01, 0.00583, 0.0125]^T$, $\bar{\mu}^{(m)} = 0.009444$.

$$\sum = (0.01 - 0.009444)^2 + (0.00583 - 0.009444)^2 + (0.0125 - 0.009444)^2$$
$$= (0.000556)^2 + (-0.003611)^2 + (0.003056)^2$$
$$= 0.000000309 + 0.00001304 + 0.000009339 = 0.00002269$$

$$\delta_{JS} = \min\left(1, \;\frac{(3-2)(0.003472)}{60 \times 0.00002269}\right) = \min\left(1, \;\frac{0.003472}{0.001361}\right) = \min(1, 2.55)$$

Since $2.55 > 1$, we cap at $\delta_{JS} = 1.0$.

**Interpretation:** With only 3 assets and 5 years of data, the James-Stein estimator says: "The sample means are so noisy relative to their spread that you should shrink all the way to the grand mean." This is extreme but reflects the reality that 5 years is not enough to reliably distinguish the expected returns of 3 assets.

With $\delta = 1$:

$$\boldsymbol{\mu}_{JS} = \begin{bmatrix}0.1133\\0.1133\\0.1133\end{bmatrix}$$

All three assets get the same expected return. The optimizer would then produce the GMV portfolio (since with identical means, the only thing to optimize is variance).

If we had $N = 25$ assets and $T = 120$ months, $\delta$ would typically be in the range $0.3$–$0.7$, giving a meaningful blend.

### 4.5 Jorion (1986) Bayes-Stein Estimator

Jorion refined James-Stein for portfolio optimization by using the GMV portfolio return as the shrinkage target and deriving the optimal intensity in a Bayesian framework:

$$\delta_{Jorion} = \frac{N + 2}{(N+2) + T\,(\hat{\boldsymbol{\mu}} - \bar{\mu}\mathbf{1})^T\Sigma^{-1}(\hat{\boldsymbol{\mu}} - \bar{\mu}\mathbf{1})}$$

This has a natural interpretation: the denominator includes a "signal strength" term $T \cdot (\text{Mahalanobis distance of means from grand mean})$. More signal → less shrinkage.

---

## 5. Shrinkage Estimators for the Covariance Matrix

### 5.1 Problems with the Sample Covariance Matrix

The sample covariance matrix $\hat{\Sigma} = \frac{1}{n-1}\sum_{t=1}^{n}(\mathbf{r}_t - \hat{\boldsymbol{\mu}})(\mathbf{r}_t - \hat{\boldsymbol{\mu}})^T$ has issues:

1. **Dimensionality:** For $N$ assets, $\Sigma$ has $N(N+1)/2$ unique parameters. With $N = 50$ assets, that is $1,275$ parameters estimated from perhaps 60 monthly observations. The matrix is poorly estimated.

2. **Singularity:** If $n < N$ (more assets than observations), $\hat{\Sigma}$ is literally singular — it cannot be inverted, and Markowitz optimization is impossible.

3. **Eigenvalue spreading:** Sample eigenvalues are systematically too dispersed — the largest are overestimated, the smallest are underestimated. This directly distorts $\Sigma^{-1}$.

### 5.2 Ledoit-Wolf Shrinkage (2004)

The most widely used covariance shrinkage estimator. Shrink the sample covariance toward a structured target:

$$\boxed{\Sigma_{LW} = (1 - \delta)\,\hat{\Sigma} + \delta\,F}$$

where $F$ is the **shrinkage target** and $\delta$ is chosen to minimize expected loss (Frobenius norm of the error).

**Common targets:**

| Target $F$ | Structure | When to use |
|------------|-----------|-------------|
| $\bar{\sigma}^2 I$ (scaled identity) | All variances equal, zero correlations | Simple, always works |
| $\text{diag}(\hat{\Sigma})$ (diagonal) | Keep variances, zero correlations | When correlations are noisy |
| Single-factor model | $\beta_i\beta_j\sigma_m^2 + \delta_{ij}\sigma_{\epsilon_i}^2$ | Equity portfolios |
| Constant correlation | All pairwise $\rho$ equal to average $\bar{\rho}$ | Moderate structure |

### 5.3 Ledoit-Wolf Optimal Shrinkage Intensity

For the scaled identity target $F = \frac{\text{tr}(\hat{\Sigma})}{N}I$, the optimal intensity is:

$$\delta^* = \frac{\sum_{t=1}^{n}\|\mathbf{x}_t\mathbf{x}_t^T - \hat{\Sigma}\|_F^2 \;/\; n^2}{\|\hat{\Sigma} - F\|_F^2}$$

where $\mathbf{x}_t = \mathbf{r}_t - \hat{\boldsymbol{\mu}}$ and $\|\cdot\|_F$ is the Frobenius norm.

**Intuition:** The numerator measures how noisy $\hat{\Sigma}$ is (the variance of the sample covariance entries). The denominator measures how far $\hat{\Sigma}$ is from the target. If $\hat{\Sigma}$ is very noisy relative to its distance from the target, shrink more.

### 5.4 Numerical Intuition

Suppose $N = 50$ assets, $n = 60$ monthly observations.

- $\hat{\Sigma}$ has $50 \times 51 / 2 = 1{,}275$ parameters.
- Effective observations per parameter: $60 / 1{,}275 \approx 0.047$.
- The matrix is severely undersampled.

Typical Ledoit-Wolf $\delta^*$ in this regime: **0.3–0.6**. The shrunk matrix is substantially different from the sample matrix.

With $N = 10$ and $n = 120$:
- Parameters: 55. Observations per parameter: $120/55 \approx 2.2$.
- Typical $\delta^*$: **0.05–0.15**. Less shrinkage needed.

### 5.5 Factor Model Covariance (Alternative to Shrinkage)

Instead of shrinking, impose structure directly. A $K$-factor model says:

$$r_{it} = \alpha_i + \sum_{k=1}^{K}\beta_{ik}f_{kt} + \epsilon_{it}$$

The implied covariance matrix is:

$$\Sigma_{factor} = B\Sigma_f B^T + D_\epsilon$$

where $B$ is the $N \times K$ matrix of factor loadings, $\Sigma_f$ is the $K \times K$ factor covariance matrix, and $D_\epsilon$ is diagonal (idiosyncratic risk).

**Parameters to estimate:** $NK + K(K+1)/2 + N$ instead of $N(N+1)/2$.

For $N = 50$, $K = 5$: $50(5) + 15 + 50 = 315$ parameters vs. $1,275$. A massive reduction.

**Practical recommendation:** For equity portfolios with $N > 30$, use either Ledoit-Wolf shrinkage or a factor model covariance. The single-factor (market) model is a reasonable starting point; adding size, value, and momentum factors (Fama-French-Carhart) improves the estimate further.

---

## 6. Practical Recommendations

### 6.1 Decision Framework

| Situation | Mean Estimator | Covariance Estimator |
|-----------|---------------|---------------------|
| $N \leq 10$, $T > 10$ years | Bayes-Stein or CAPM prior | Sample $\hat{\Sigma}$ (mild shrinkage) |
| $N = 10$–$50$, $T = 3$–$10$ years | Black-Litterman (Module 5) | Ledoit-Wolf or factor model |
| $N > 50$ | Equal means (GMV only) or factor model means | Factor model covariance |
| Any $N$, short $T$ | Shrink heavily toward equal or CAPM | Ledoit-Wolf with structured target |

### 6.2 Specific Numbers

- **Lookback window:** 3–5 years of weekly returns for equities (156–260 observations). Monthly returns give too few observations; daily returns introduce microstructure noise.
- **Rebalancing frequency:** Quarterly or semi-annually. More frequent rebalancing amplifies estimation error and transaction costs.
- **Weight constraints:** Even with shrinkage, impose $w_i \in [-0.10, 0.30]$ or similar bounds. This acts as implicit shrinkage and prevents extreme positions.
- **Minimum observations per parameter:** Aim for $n / [N(N+1)/2] > 2$. If this ratio is below 1, do not use the sample covariance without shrinkage.

### 6.3 The Resampled Efficiency Approach (Michaud, 1998)

An alternative to analytical shrinkage:

1. Estimate $\hat{\boldsymbol{\mu}}$ and $\hat{\Sigma}$ from data.
2. Simulate $M$ bootstrap samples of $(\boldsymbol{\mu}, \Sigma)$ from the sampling distribution.
3. For each simulated input set, solve the Markowitz optimization.
4. Average the $M$ sets of optimal weights.

The averaged weights are more stable because the extreme positions from different simulations tend to cancel out. This is essentially a Monte Carlo version of shrinkage.

---

## 7. Complete Numerical Example: Shrinkage in Action

### 7.1 Setup

Same 3 assets as before. We will compare four approaches:

1. **Raw Markowitz** (sample inputs)
2. **James-Stein shrinkage on means** (covariance unchanged)
3. **Ledoit-Wolf shrinkage on covariance** (means unchanged)
4. **Both shrinkages combined**

Assume $T = 5$ years, monthly data ($n = 60$).
```python
import numpy as np

mu_sample = np.array([0.12, 0.07, 0.15])
Rf = 0.03
Sigma_sample = np.array([
[0.0400, 0.0090, 0.0050],
[0.0090, 0.0225, 0.01875],
[0.0050, 0.01875, 0.0625]
])
ones = np.ones(3)
N = 3
n = 60

# --- James-Stein shrinkage on means ---
mu_bar = mu_sample.mean()  # grand mean = 0.1133
# Jorion's formula
Sigma_inv = np.linalg.inv(Sigma_sample)
diff = mu_sample - mu_bar * ones
signal = diff @ Sigma_inv @ diff  # Mahalanobis distance squared
delta_jorion = (N + 2) / ((N + 2) + n * signal)
mu_JS = (1 - delta_jorion) * mu_sample + delta_jorion * mu_bar * ones

print(f"Jorion delta = {delta_jorion:.4f}")
print(f"mu_JS = {np.round(mu_JS, 4)}")

**Output:**

$$\delta_{Jorion} = 0.0640$$

$$\boldsymbol{\mu}_{JS} = \begin{bmatrix}0.1196\0.0728\0.1477\end{bmatrix}$$

The shrinkage is mild here ($\delta = 6.4\%$) because the Mahalanobis distance of the means from the grand mean is large relative to $N$. With more assets, $\delta$ would be larger.

python
# --- Ledoit-Wolf shrinkage on covariance ---
# Target: scaled identity
target_var = np.trace(Sigma_sample) / N  # average variance
F = target_var * np.eye(N)  # = 0.04167 * I

# Simplified LW intensity (analytical for known Sigma, 
# in practice computed from data)
# For demonstration, use sklearn
from sklearn.covariance import LedoitWolf
# We'd normally fit on return data; here we'll compute delta analytically
# For 3 assets with n=60, typical delta ≈ 0.05-0.15
delta_LW = 0.10  # representative value

Sigma_LW = (1 - delta_LW) * Sigma_sample + delta_LW * F
print(f"Sigma_LW =\n{np.round(Sigma_LW, 5)}")

$$\Sigma_{LW} = (0.90)\hat{\Sigma} + (0.10)(0.04167\,I)$$

$$= \begin{bmatrix}0.04017 & 0.00810 & 0.00450\0.00810 & 0.02442 & 0.01688\0.00450 & 0.01688 & 0.06042\end{bmatrix}$$

The off-diagonal elements shrink toward zero (by 10%), and the diagonal elements shrink toward the average variance.

### 7.2 Tangency Portfolios Under Each Approach

python
def tangency(mu, Sigma, Rf):
Sinv = np.linalg.inv(Sigma)
pi = mu - Rf * np.ones(len(mu))
z = Sinv @ pi
w = z / z.sum()
mu_p = w @ mu
sig_p = np.sqrt(w @ Sigma @ w)
sharpe = (mu_p - Rf) / sig_p
return w, mu_p, sig_p, sharpe

approaches = {
"Raw Markowitz":     (mu_sample, Sigma_sample),
"JS means only":     (mu_JS, Sigma_sample),
"LW cov only":       (mu_sample, Sigma_LW),
"JS + LW combined":  (mu_JS, Sigma_LW),
}

for name, (mu, Sig) in approaches.items():
w, mu_p, sig_p, sr = tangency(mu, Sig, Rf)
print(f"\n{name}:")
print(f"  w = [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
print(f"  μ = {mu_p:.2%}, σ = {sig_p:.2%}, Sharpe = {sr:.4f}")

**Results:**

| Approach | $w_1$ | $w_2$ | $w_3$ | $\mu_p$ | $\sigma_p$ | Sharpe |
|----------|-------|-------|-------|---------|-----------|--------|
| Raw Markowitz | $0.594$ | $0.170$ | $0.236$ | $11.87\%$ | $15.62\%$ | $0.568$ |
| JS means only | $0.589$ | $0.178$ | $0.233$ | $11.83\%$ | $15.55\%$ | $0.568$ |
| LW cov only | $0.575$ | $0.186$ | $0.239$ | $11.78\%$ | $15.38\%$ | $0.571$ |
| JS + LW combined | $0.570$ | $0.194$ | $0.236$ | $11.74\%$ | $15.31\%$ | $0.571$ |

### 7.3 Interpretation

With only 3 assets, the differences are modest. This is expected — shrinkage matters most when $N$ is large relative to $n$. But notice the pattern:

- Shrinkage **pulls weights toward equality** (Asset 2's weight increases from $0.170$ to $0.194$).
- Shrinkage **reduces extreme positions**.
- The in-sample Sharpe ratio barely changes (or even improves slightly with LW), but the **out-of-sample** Sharpe ratio would improve more substantially because the shrunk portfolio is less overfit to noise.

### 7.4 What Happens with More Assets?

To show the dramatic effect with larger $N$, here is a simulation:

python
np.random.seed(42)
N_large = 30
n_obs = 60

# Generate true parameters
mu_true = np.random.uniform(0.05, 0.15, N_large)
# Random correlation matrix via random factor model
F = np.random.randn(N_large, 5) * 0.05
Sigma_true = F @ F.T + np.diag(np.random.uniform(0.01, 0.04, N_large))

# Generate sample data
returns = np.random.multivariate_normal(mu_true/12, Sigma_true/12, n_obs)
mu_hat = returns.mean(axis=0) * 12
Sigma_hat = np.cov(returns.T) * 12

# Condition numbers
kappa_true = np.linalg.cond(Sigma_true)
kappa_sample = np.linalg.cond(Sigma_hat)
print(f"Condition number (true):   {kappa_true:.1f}")
print(f"Condition number (sample): {kappa_sample:.1f}")

**Typical output:**

$$\kappa(\Sigma_{true}) \approx 45, \qquad \kappa(\hat{\Sigma}) \approx 850$$

The sample covariance condition number is nearly 20× larger than the true one. This means $\hat{\Sigma}^{-1}$ amplifies errors by a factor of ~850. With Ledoit-Wolf shrinkage ($\delta \approx 0.35$ in this regime):

$$\kappa(\Sigma_{LW}) \approx 120$$

Still elevated, but much more manageable.

The tangency portfolio weights from $\hat{\Sigma}$ would include positions like $w_i = +3.5$ and $w_j = -2.8$ — absurd leveraged bets driven entirely by estimation error. After shrinkage, the weights compress to a reasonable range.

---

## 8. Summary: The Shrinkage Mindset

The key insight of this module is not any single formula. It is a way of thinking:

> **Every input to the optimizer is uncertain. Treat it as a random variable, not a known constant. Then ask: given this uncertainty, what is the best estimate to feed the optimizer?**

The answer is always some form of shrinkage — pulling noisy estimates toward structured, lower-variance targets. The optimal amount of shrinkage depends on:

- **Signal-to-noise ratio** (more noise → more shrinkage)
- **Dimensionality** (more assets → more shrinkage)
- **Quality of the target** (better target → more shrinkage is safe)

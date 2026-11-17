---
type: quiz-prep
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[1.pdf]]'
- '[[2.pdf]]'
- '[[Quiz 3 - Question.pdf]]'
- '[[4.pdf]]'
- '[[Sample Qz 4.pdf]]'
tags:
- atml
- course/cs6304
- quiz-prep
title: ATML — Quiz Prep, Lectures 1–6
aliases:
- ATML Quiz Prep
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_6 - Open-Set Recognition]]
%% nav:end %%

# ATML — Quiz Prep: Lectures 1–6

*Every hand-computable quantity from [[ATML_Lecture_1 - Representation and Generalization Theory]] through [[ATML_Lecture_6 - Open-Set Recognition]], one section per lecture and one subsection per topic, each linked to its theory section. Each question states the formulas it needs, including the intermediate ones such as the mean, standard deviation, sigmoid derivative or $2\times2$ inverse, before the numbers are substituted. Questions are tagged by origin: **[Notes]** is a problem from the lecture notes, **[Past Quiz]** is from last year's quizzes, and **[New]** is written for this sheet. Lecture 3 Section 2 (discriminative models) is not drilled; its CNN and ViT size formulas appear only where Past Quiz 1 needs them (§4.3).*

**Past-quiz map.** Last year's quizzes that fall inside Lectures 1–6:

| Past quiz | Topic | Where here |
|---|---|---|
| Quiz 1 | CNN vs ViT: spread of a local perturbation, uniform attention | §4.3, Q4.8 |
| Quiz 2 | GAN non-saturating loss: gradients at two operating points | §4.1, Q4.4 |
| Quiz 3 | CLIP prompt learning: logit-sum vs probability-average gradients | §5.5, Q5.9 |
| Quiz 4 | IRMv1 scale collapse | §6.2, Q6.5 |
| Sample Qz 4, P1 | InfoNCE gradients and hard-negative concentration | §6.1, Q6.3 |
| Sample Qz 4, P2 | ERM vs IRM with causal and spurious features | §6.2, Q6.6 |

Quizzes 7 and 8 (RLHF/PPO/DPO and mechanistic interpretability) cover later lectures and are excluded.

---

## 0. Calculation Toolbox

These are the formulas that questions assume you already have. Look here first when a step says "compute the standard deviation" or "invert $\Sigma$".

**Logs and exponentials.**

| $x$ | $e^{x}$ | | $x$ | $\ln x$ | $\log_2 x$ |
|---|---|---|---|---|---|
| $-3$ | $0.0498$ | | $0.1$ | $-2.3026$ | $-3.3219$ |
| $-2$ | $0.1353$ | | $0.2$ | $-1.6094$ | $-2.3219$ |
| $-1$ | $0.3679$ | | $0.25$ | $-1.3863$ | $-2$ |
| $-0.5$ | $0.6065$ | | $0.3$ | $-1.2040$ | $-1.7370$ |
| $0.5$ | $1.6487$ | | $0.4$ | $-0.9163$ | $-1.3219$ |
| $1$ | $2.7183$ | | $0.5$ | $-0.6931$ | $-1$ |
| $2$ | $7.3891$ | | $0.75$ | $-0.2877$ | $-0.4150$ |
| $3$ | $20.086$ | | $2$ | $0.6931$ | $1$ |
| $4$ | $54.598$ | | $3$ | $1.0986$ | $1.5850$ |
| $5$ | $148.41$ | | $10$ | $2.3026$ | $3.3219$ |

Base change: $\log_2 x = \ln x / \ln 2 = 1.4427\ln x$. The unit is nats with $\ln$ and bits with $\log_2$. Log rules: $\ln(ab) = \ln a + \ln b$, $\ln(a/b) = \ln a - \ln b$, and $\ln(e^{a}+e^{b}) = a + \ln(1 + e^{b-a})$, which is the trick for large logits.

**Softmax, log-sum-exp, sigmoid.**

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad \operatorname{LSE}(z) = \ln\sum_j e^{z_j}, \qquad \max_k z_k \le \operatorname{LSE}(z) \le \max_k z_k + \ln K
$$

Softmax is shift-invariant: $\operatorname{softmax}(z + c) = \operatorname{softmax}(z)$, so only logit differences matter. For two classes, $\frac{e^{z_1}}{e^{z_1}+e^{z_2}} = \sigma(z_1 - z_2)$.

$$
\sigma(a) = \frac{1}{1+e^{-a}}, \qquad \sigma(-a) = 1-\sigma(a), \qquad \sigma'(a) = \sigma(a)\big(1-\sigma(a)\big), \qquad \sigma^{-1}(p) = \ln\frac{p}{1-p}
$$

$$
\frac{d}{da}\big[-\ln\sigma(a)\big] = \sigma(a) - 1, \qquad \frac{d}{da}\ln\big(1-\sigma(a)\big) = -\sigma(a)
$$

**Cross-entropy and its gradient.** With probabilities $p = \operatorname{softmax}(z)$ and target $y$ (one-hot or soft),

$$
\mathcal{L}_{\text{CE}} = -\sum_k y_k \ln p_k \;\overset{\text{one-hot}}{=}\; -z_j + \operatorname{LSE}(z), \qquad \frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = p_k - y_k .
$$

**Statistics.** For samples $x_1,\ldots,x_N$,

$$
\bar{x} = \frac{1}{N}\sum_i x_i, \qquad \sigma^2 = \frac{1}{N}\sum_i (x_i-\bar{x})^2 = \overline{x^2} - \bar{x}^2 \;\;\text{(population)}, \qquad s^2 = \frac{1}{N-1}\sum_i (x_i - \bar{x})^2 \;\;\text{(sample)}, \qquad \text{std} = \sqrt{\text{variance}}
$$

Use $1/N$ unless the question says "unbiased" or "sample"; the course's covariance uses $1/N$ ([[ATML_Lecture_1 - Representation and Generalization Theory#8. Mathematical Background|L1 §8]]). Weighted mean: $\sum_i \pi_i x_i$ with $\sum_i \pi_i = 1$. Covariance matrix: $R_x = \frac{1}{N}\sum_i (x_i-\mu)(x_i-\mu)^\top$. Correlation: $\rho = \sigma_{12}/(\sigma_1\sigma_2)$. $\text{Beta}(a,b)$ has mean $a/(a+b)$ and variance $ab/\big((a+b)^2(a+b+1)\big)$.

**Vectors and norms.**

$$
\lVert x\rVert_2 = \sqrt{\textstyle\sum_i x_i^2}, \qquad \hat{x} = \frac{x}{\lVert x\rVert_2}, \qquad \cos(x,y) = \frac{x^\top y}{\lVert x\rVert\,\lVert y\rVert}, \qquad \lVert W\rVert_F = \sqrt{\textstyle\sum_{ij} W_{ij}^2}
$$

Spectral norm: $\lVert W\rVert_2 = \sigma_{\max}(W) = \sqrt{\lambda_{\max}(W^\top W)}$. It equals $\max_i \lvert d_i\rvert$ for a diagonal $W$ and $\max_i \lvert\lambda_i\rvert$ for a symmetric $W$. It always satisfies $\lVert W\rVert_2 \le \lVert W\rVert_F$, with equality iff $W$ has rank 1.

**$2\times 2$ matrices.** For $A = \begin{bmatrix} a & b\\ c & d\end{bmatrix}$,

$$
\lvert A\rvert = ad - bc, \qquad A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b\\ -c & a\end{bmatrix}, \qquad \lambda_{1,2} = \frac{\operatorname{tr}A \pm \sqrt{(\operatorname{tr}A)^2 - 4\lvert A\rvert}}{2}
$$

The eigenvector for $\lambda$ solves $(A - \lambda I)u = 0$; for symmetric $\begin{bmatrix} a & b\\ b & a\end{bmatrix}$ the eigenvectors are $(1,1)/\sqrt2$ with $\lambda = a+b$ and $(1,-1)/\sqrt2$ with $\lambda = a-b$. The quadratic form of a symmetric matrix is $x^\top A x = a x_1^2 + 2b\,x_1x_2 + d\,x_2^2$.

**Calculus.** Chain rule: $\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial a}\frac{\partial a}{\partial\theta}$. Gradient step: $\theta' = \theta - \eta\,\nabla_\theta L$. First-order change: $a' \approx a + \frac{\partial a}{\partial\theta}(\theta'-\theta)$. For a weighted sum of squares,

$$
\arg\min_u \big[\alpha(u-b)^2 + \beta(u-d)^2\big] = \frac{\alpha b + \beta d}{\alpha+\beta}, \qquad \frac{d}{dw}(w\theta - c)^2 = 2\theta(w\theta - c).
$$

**Gaussian.** In one dimension, $\mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$, so $\ln\mathcal{N} = -\frac{(x-\mu)^2}{2\sigma^2} - \ln\sigma - \frac12\ln 2\pi$, with $\frac12\ln 2\pi = 0.9189$.

**Counting.** Pairwise AUROC counts a tie as $\tfrac12$. For a floor, $\lfloor 223/2\rfloor = 111$.

---

## 1. Lecture 1: Generalization, Cross-Entropy, Moments

### 1.1 Capacity and the Generalization Bound

Theory: [[ATML_Lecture_1 - Representation and Generalization Theory#5.2 Margin, Capacity, and the Refined Bound|L1 §5.2]], [[ATML_Lecture_1 - Representation and Generalization Theory#5.4 Controlling the Complexity/Margin Ratio|L1 §5.4]].

$$
\beta = \prod_{\ell=1}^{L}\lVert W_\ell\rVert_2, \qquad S = \sum_{\ell=1}^{L}\frac{\lVert W_\ell\rVert_F^2}{\lVert W_\ell\rVert_2^2}, \qquad \text{Loss}_{\text{Test}} \le \text{Loss}_{\text{Train}} + \frac{1}{\sqrt N}\cdot\frac{\beta\sqrt S}{\gamma}
$$

**Q1.1 — Capacity from norms.** *[Notes L1-P1]* Spectral norms are $(2, 1.5, 1)$ and Frobenius norms are $(4, 3, 2)$. Find $\beta$, $S$ and the capacity.

*Solution.* $\beta = 2 \times 1.5 \times 1 = 3$. Each ratio $\lVert W\rVert_F^2/\lVert W\rVert_2^2$ is $16/4 = 4$, $9/2.25 = 4$ and $4/1 = 4$, so $S = 12$. The capacity is $\beta\sqrt S = 3\sqrt{12}$.

$$\boxed{\beta = 3,\; S = 12,\; \beta\sqrt S = 6\sqrt3 \approx 10.39}$$

**Q1.2 — Capacity from the matrices themselves.** *[New]* Take $W_1 = \begin{bmatrix}3 & 0\\0 & 1\end{bmatrix}$ and $W_2 = \begin{bmatrix}1 & 1\\1 & 1\end{bmatrix}$. Compute the capacity.

*Formulas needed first.* The spectral norm is the largest singular value, which is $\max\lvert d_i\rvert$ for a diagonal matrix and $\max\lvert\lambda_i\rvert$ for a symmetric one (§0). The Frobenius norm is the root of the sum of squared entries.

*Solution.* $W_1$ is diagonal, so $\lVert W_1\rVert_2 = 3$ and $\lVert W_1\rVert_F^2 = 9 + 1 = 10$. $W_2$ is symmetric with $\operatorname{tr} = 2$ and $\lvert W_2\rvert = 0$, so $\lambda = 2, 0$, giving $\lVert W_2\rVert_2 = 2$ and $\lVert W_2\rVert_F^2 = 4$. Then $\beta = 3 \times 2 = 6$ and $S = 10/9 + 4/4 = 2.111$, so $\beta\sqrt S = 6 \times 1.453$.

$$\boxed{\beta = 6,\; S \approx 2.111,\; \beta\sqrt S \approx 8.72}$$

$W_2$ has rank 1, so its ratio is exactly 1, the minimum. $S = L$ only when every layer is rank-one.

**Q1.3 — Using the bound.** *[Notes L1-P2]* With $N = 10{,}000$, training loss $0.15$, $\beta\sqrt S = 20$ and $\gamma = 0.5$, bound the test loss. Which margin halves the generalization term?

*Solution.* The generalization term is $\frac{1}{100}\cdot\frac{20}{0.5} = 0.4$, so the test loss is at most $0.15 + 0.4$. The term is $\propto 1/\gamma$, so doubling the margin to $\gamma = 1$ halves it to $0.2$.

$$\boxed{\text{Loss}_{\text{Test}} \le 0.55;\quad \gamma = 1.0 \text{ gives a bound of } 0.35}$$

**Q1.4 — Required sample size.** *[New]* A model has training loss $0.05$, capacity $30$ and margin $1.5$. How many samples guarantee a test-loss bound of at most $0.15$?

*Solution.* We need $\frac{1}{\sqrt N}\cdot\frac{30}{1.5} \le 0.10$, i.e. $\frac{20}{\sqrt N} \le 0.1$, so $\sqrt N \ge 200$.

$$\boxed{N \ge 40{,}000}$$

Halving the gap needs four times as much data, because of the $1/\sqrt N$ rate.

### 1.2 Softmax Head and Cross-Entropy

Theory: [[ATML_Lecture_1 - Representation and Generalization Theory#7.2 Matrix Form of the Classifier Head|L1 §7.2]], [[ATML_Lecture_1 - Representation and Generalization Theory#7.3 Cross-Entropy Loss Derivation|L1 §7.3]], [[ATML_Lecture_1 - Representation and Generalization Theory#7.4 The Similarity-Based Form of Cross-Entropy|L1 §7.4]].

$$
s = \Psi z,\;\; \hat y = \operatorname{softmax}(s), \qquad \mathcal{L}_{\text{CE}} = -s_j + \ln\sum_k e^{s_k} = \ln\Big(1 + \sum_{k\ne j} e^{s_k - s_j}\Big)
$$

**Q1.5 — CE from logits, two ways.** *[Notes L1-P3, P4]* The logits are $s = (2, 1, 0.5, -1)$ and the true class is $j = 2$.

*Solution.* Log-sum-exp form: $e^{2} + e^{1} + e^{0.5} + e^{-1} = 7.389 + 2.718 + 1.649 + 0.368 = 12.124$, and $\ln 12.124 = 2.495$, so $\mathcal{L} = -1 + 2.495$. Ratio form: $e^{1} + e^{-0.5} + e^{-2} = 2.718 + 0.607 + 0.135 = 3.460$, and $\ln 4.460 = 1.495$.

$$\boxed{\mathcal{L}_{\text{CE}} \approx 1.495 \text{ nats};\quad \hat y_2 = e^{-1.495} \approx 0.224}$$

**Q1.6 — Full head forward pass.** *[New]* The latent is $z = (1, 2)^\top$ and the prototypes are the rows of $\Psi = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$. The true class is 1. Compute the logits, the probabilities, the loss and the predicted class.

*Solution.* $s = \Psi z = (1, 2, 3)$. The exponentials are $2.718 + 7.389 + 20.086 = 30.19$, so $\hat y = (0.090, 0.245, 0.665)$ and $\mathcal{L} = -\ln 0.090$.

$$\boxed{s = (1,2,3),\; \hat y \approx (0.090, 0.245, 0.665),\; \mathcal{L} \approx 2.41 \text{ nats},\; \text{predicted class } 3}$$

The prediction is the prototype with the largest inner product $\psi_k^\top z$ ([[ATML_Lecture_1 - Representation and Generalization Theory#7.5 Geometric Interpretation: Class Prototypes|L1 §7.5]]).

**Q1.7 — CE with a soft target.** *[New]* The target is $y = (0.7, 0.2, 0.1)$ and the prediction is $\hat y = (0.5, 0.3, 0.2)$.

*Solution.* $\mathcal{L} = -[0.7\ln 0.5 + 0.2\ln 0.3 + 0.1 \ln 0.2] = 0.7(0.6931) + 0.2(1.2040) + 0.1(1.6094)$.

$$\boxed{\mathcal{L}_{\text{CE}} \approx 0.887 \text{ nats}}$$

It cannot fall below $H(y) = 0.802$ nats, which is reached only when $\hat y = y$, because $\mathcal{L}_{\text{CE}} = H(y) + D_{\mathrm{KL}}(y\Vert\hat y)$.

### 1.3 Mean and Covariance

Theory: [[ATML_Lecture_1 - Representation and Generalization Theory#8. Mathematical Background|L1 §8]].

**Q1.8 — Sample covariance.** *[Notes L1-P5]* The data are $(1,2)$, $(3,4)$ and $(5,0)$.

*Solution.* $\mu = (3, 2)$, and the centred points are $(-2,0)$, $(0,2)$ and $(2,-2)$. The outer products sum to $\begin{bmatrix}8&-4\\-4&8\end{bmatrix}$; divide by $N = 3$.

$$\boxed{\mu = (3,2)^\top,\quad R_x = \begin{bmatrix}8/3 & -4/3\\-4/3 & 8/3\end{bmatrix}}$$

The negative off-diagonal means the dimensions are correlated, which a good latent should avoid ([[ATML_Lecture_2 - Information Theory and Divergences#2.2 Correlated Features and the Case for Independent Latents|L2 §2.2]]).

**Short answers (L1).**
- *Why is $P(X)$ not the histogram of the dataset?* The histogram is a finite sample and is zero wherever no data point landed, whereas $P(X)$ is a density over the whole input space.
- *Which way does each quantity move the bound?* More data ($1/\sqrt N$) and more margin ($\gamma$) tighten it; more capacity ($\beta\sqrt S$) loosens it.
- *Why can low training loss coexist with OOD failure?* The model can exploit spurious, domain-dependent features (the watermark example). The fix is domain-invariant features, not more data from the same domain.

---

## 2. Lecture 2: Information Theory, Divergences, Optimal Transport

### 2.1 Monte Carlo and Mutual Information

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#1.2 The Monte Carlo Approximation|L2 §1.2]], [[ATML_Lecture_2 - Information Theory and Divergences#3.5 Mutual Information|L2 §3.5]].

$$
\mathbb{E}_{P}[g(x)] \approx \frac1N\sum_{i=1}^N g(x_i), \qquad I(X;Y) = \sum_{x,y}P(x,y)\log\frac{P(x,y)}{P(x)P(y)} = H(X) - H(X\mid Y)
$$

**Q2.1 — Monte Carlo estimate.** *[New]* Samples are $\{1, 2, 3, 4\}$. Estimate $\mathbb{E}[x^2]$ and the variance.

*Solution.* $\widehat{\mathbb{E}}[x^2] = (1+4+9+16)/4 = 7.5$ and $\hat\mu = 2.5$, so $\widehat{\text{Var}} = \overline{x^2} - \bar x^2 = 7.5 - 6.25$.

$$\boxed{\widehat{\mathbb{E}}[x^2] = 7.5,\quad \widehat{\text{Var}} = 1.25,\quad \text{std} \approx 1.118}$$

**Q2.2 — Mutual information from a joint table.** *[New]* The joint distribution is $P(X,Y) = \begin{bmatrix}0.4 & 0.1\\0.1 & 0.4\end{bmatrix}$. Compute $I(X;Y)$ in bits.

*Solution.* Both marginals are $(0.5, 0.5)$, so $P(x)P(y) = 0.25$ for every cell. Then $I = 2(0.4)\log_2(0.4/0.25) + 2(0.1)\log_2(0.1/0.25) = 0.8(0.6781) + 0.2(-1.3219)$.

$$\boxed{I(X;Y) \approx 0.278 \text{ bits}}$$

Check: $H(X) = 1$ bit and $H(X\mid Y) = -\sum P(x,y)\log_2 P(x\mid y) = 0.722$ bits, and $1 - 0.722 = 0.278$. MI is zero iff $X$ and $Y$ are independent.

### 2.2 Entropy and Information Content

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#3.2 Information Content|L2 §3.2]], [[ATML_Lecture_2 - Information Theory and Divergences#3.3 Entropy as Average Information Content|L2 §3.3]].

$$
I(x_k) = -\log_2 P(x_k), \qquad H(X) = -\sum_x P(x)\log_2 P(x) \le \log_2 K
$$

**Q2.3 — Entropy.** *[Notes L2-P1]* Compute $H$ for $P = (0.1, 0.2, 0.3, 0.4)$.

*Solution.* $H = 0.1(3.3219) + 0.2(2.3219) + 0.3(1.7370) + 0.4(1.3219) = 0.3322 + 0.4644 + 0.5211 + 0.5288$.

$$\boxed{H \approx 1.846 \text{ bits}} \;(< \log_2 4 = 2)$$

**Q2.4 — Surprise of a rare event.** *[Notes L2-P2]* Compare $P(\text{heads}) = 0.05$ with a fair coin.

*Solution.* $-\log_2 0.05 = \log_2 20 = 4 + \log_2 1.25$, against $-\log_2 0.5 = 1$ for the fair coin.

$$\boxed{4.32 \text{ bits vs } 1 \text{ bit}}$$

### 2.3 Multivariate Gaussian

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#2.3 The Multivariate Gaussian — Shape and Properties|L2 §2.3]].

$$
d_M(x) = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}, \qquad \text{semi-axes } \sqrt{\lambda_i} \text{ along } u_i, \qquad H = \tfrac12\ln\big((2\pi e)^d\lvert\Sigma\rvert\big)
$$

$$
x_2\mid x_1 \sim \mathcal{N}\Big(\mu_2 + \tfrac{\sigma_{12}}{\sigma_{11}}(x_1-\mu_1),\; \sigma_{22} - \tfrac{\sigma_{12}^2}{\sigma_{11}}\Big), \qquad D_{\mathrm{KL}}(\mathcal{N}_0\Vert\mathcal{N}_1) = \tfrac12\Big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + \Delta\mu^\top\Sigma_1^{-1}\Delta\mu - d + \ln\tfrac{\lvert\Sigma_1\rvert}{\lvert\Sigma_0\rvert}\Big]
$$

**Q2.5 — Shape, Mahalanobis distance, entropy, conditional.** *[Notes L2-P8]* $\Sigma = \begin{bmatrix}3&1\\1&3\end{bmatrix}$ and $\mu = 0$.

*Solution.* The eigenvalues are $3 \pm 1 = 4, 2$, with $u_1 = (1,1)/\sqrt2$ and $u_2 = (1,-1)/\sqrt2$, and $\rho = 1/3$. With $\lvert\Sigma\rvert = 8$, $\Sigma^{-1} = \frac18\begin{bmatrix}3&-1\\-1&3\end{bmatrix}$. For $a = (2,2)$: $\frac18(12 - 8 + 12) = 2$. For $b = (2,-2)$: $\frac18(12+8+12) = 4$. The entropy is $H = \ln(2\pi e) + \frac12\ln 8 = 2.838 + 1.040$. The conditional mean is $\frac13 \cdot 3 = 1$ and the conditional variance is $3 - \frac13 = \frac83$.

$$\boxed{\text{semi-axes } 2, \sqrt2;\; d_M(a) = \sqrt2,\; d_M(b) = 2;\; H \approx 3.878 \text{ nats};\; x_2\mid x_1{=}3 \sim \mathcal{N}(1, 8/3)}$$

**Q2.6 — KL between 1-D Gaussians.** *[New]* Compute $D_{\mathrm{KL}}\big(\mathcal{N}(1, 1)\,\Vert\,\mathcal{N}(0, 4)\big)$.

*Solution.* With $d = 1$, $\Sigma_0 = 1$, $\Sigma_1 = 4$ and $\Delta\mu = 1$: $\frac12\big[\frac14 + \frac14 - 1 + \ln 4\big] = \frac12(-0.5 + 1.3863)$.

$$\boxed{D_{\mathrm{KL}} \approx 0.443 \text{ nats}}$$

### 2.4 f-Divergences: KL and JS

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#5.2 Forward KL Divergence|L2 §5.2]], [[ATML_Lecture_2 - Information Theory and Divergences#5.4 Reverse KL Divergence Is Mode-Seeking|L2 §5.4]], [[ATML_Lecture_2 - Information Theory and Divergences#5.5 Jensen–Shannon Divergence|L2 §5.5]].

$$
D_{\mathrm{KL}}(P\Vert Q) = \sum_i P_i\ln\frac{P_i}{Q_i}, \qquad \mathrm{JS}(P\Vert Q) = \tfrac12 D_{\mathrm{KL}}(P\Vert M) + \tfrac12 D_{\mathrm{KL}}(Q\Vert M),\;\; M = \tfrac{P+Q}{2}
$$

**Q2.7 — Forward KL, reverse KL, JS.** *[Notes L2-P3, P4, P5]* $P = (0.6, 0.3, 0.1)$ and $Q = (0.4, 0.4, 0.2)$.

*Solution.* Forward KL: $0.6\ln 1.5 + 0.3\ln 0.75 + 0.1\ln 0.5 = 0.2433 - 0.0863 - 0.0693$. Reverse KL: $0.4\ln\frac23 + 0.4\ln\frac43 + 0.2\ln 2 = -0.1622 + 0.1151 + 0.1386$. For JS, $M = (0.5, 0.35, 0.15)$, with $D_{\mathrm{KL}}(P\Vert M) = 0.0226$ and $D_{\mathrm{KL}}(Q\Vert M) = 0.0217$.

$$\boxed{D_{\mathrm{KL}}(P\Vert Q) \approx 0.0877,\;\; D_{\mathrm{KL}}(Q\Vert P) \approx 0.0915,\;\; \mathrm{JS} \approx 0.0222 \text{ nats}}$$

**Q2.8 — Disjoint supports.** *[New]* On positions $1..4$, $P = (0.5, 0.5, 0, 0)$ and $Q = (0, 0, 0.5, 0.5)$. Compute the KL, JS, TVD and $W_1$ (cost $\lvert i - j\rvert$).

*Solution.* KL divides by $Q_1 = 0$, so it is infinite. $M = (0.25, 0.25, 0.25, 0.25)$, which gives $D_{\mathrm{KL}}(P\Vert M) = 2(0.5)\ln 2 = \ln 2$, and likewise for $Q$, so $\mathrm{JS} = \ln 2$. $\text{TVD} = \frac12(4 \times 0.5) = 1$. For $W_1$, move each half-unit two steps ($1\to3$, $2\to4$): $0.5(2) + 0.5(2) = 2$.

$$\boxed{D_{\mathrm{KL}} = \infty,\;\; \mathrm{JS} = \ln 2 \approx 0.693 \text{ (its maximum)},\;\; \text{TVD} = 1 \text{ (its maximum)},\;\; W_1 = 2}$$

JS and TVD have saturated at their maxima, so they would not change if $Q$ moved further away; $W_1$ would grow. This is the argument for OT ([[ATML_Lecture_2 - Information Theory and Divergences#7.1 Why OT? The Failure of KL on Disjoint Supports|L2 §7.1]]).

### 2.5 IPMs: MMD and TVD

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#6.2 Maximum Mean Discrepancy|L2 §6.2]], [[ATML_Lecture_2 - Information Theory and Divergences#6.3 Total Variation Distance|L2 §6.3]].

$$
\text{MMD}^2 = (P-Q)^\top K (P-Q) \;\;\text{(discrete)}, \qquad \text{MMD}^2 = \overline{k(x,x')} + \overline{k(y,y')} - 2\,\overline{k(x,y)} \;\;\text{(samples)}, \qquad \text{TVD} = \tfrac12\sum_i \lvert P_i - Q_i\rvert
$$

**Q2.9 — Discrete MMD.** *[New]* $P = (0.6, 0.4)$, $Q = (0.4, 0.6)$ and $K = \begin{bmatrix}1&0.5\\0.5&1\end{bmatrix}$.

*Solution.* $d = P - Q = (0.2, -0.2)$, so $d^\top K d = 0.04 + 0.04 + 2(0.5)(0.2)(-0.2) = 0.08 - 0.04$.

$$\boxed{\text{MMD}^2 = 0.04,\;\; \text{MMD} = 0.2}$$

**Q2.10 — Sample MMD with an RBF kernel.** *[New]* The samples are $X = \{0, 1\}$ and $Y = \{1, 2\}$, with $k(a,b) = e^{-(a-b)^2/2}$. Average over all pairs, including $i = j$.

*Solution.* The kernel values are $k = 1$ at distance 0, $e^{-0.5} = 0.6065$ at distance 1, and $e^{-2} = 0.1353$ at distance 2. Within $X$: $(1 + 0.6065 + 0.6065 + 1)/4 = 0.8033$, and the same within $Y$. Across: pairs $(0,1), (0,2), (1,1), (1,2)$ give $(0.6065 + 0.1353 + 1 + 0.6065)/4 = 0.5871$. So $\text{MMD}^2 = 0.8033 + 0.8033 - 2(0.5871)$.

$$\boxed{\text{MMD}^2 \approx 0.432}$$

**Q2.11 — TVD vs Wasserstein.** *[Notes L2-P6]* $P = (0.6, 0.3, 0.1)$ and $Q = (0.4, 0.4, 0.2)$ at positions $1,2,3$, with $C_{ij} = \lvert i-j\rvert$.

*Solution.* $\text{TVD} = \frac12(0.2+0.1+0.1) = 0.2$. $P$ has a surplus of $0.2$ at bin 1, while $Q$ needs $0.1$ extra at bin 2 and $0.1$ at bin 3. The mass flows $1 \to 2$ ($0.1$, cost 1) and $1 \to 3$ ($0.1$, cost 2). *(The notes phrase this as moving 2 → 1 and 3 → 1, which has the same cost read in the other direction.)* The CDF shortcut agrees: $\sum\lvert F_P - F_Q\rvert = 0.2 + 0.1 = 0.3$.

$$\boxed{\text{TVD} = 0.2,\;\; \langle T^*, C\rangle = 0.3 \;(\text{WD} = \sqrt{0.3} \approx 0.548 \text{ under the notes' square-root convention})}$$

### 2.6 Optimal Transport and Sinkhorn

Theory: [[ATML_Lecture_2 - Information Theory and Divergences#7.3 The Optimal Transport Linear Program and the Wasserstein Distance|L2 §7.3]], [[ATML_Lecture_2 - Information Theory and Divergences#7.5 Entropic Regularization, Sinkhorn Distance, and the Sinkhorn Algorithm|L2 §7.5]].

$$
\min_T \langle T, C\rangle = \sum_{ij}T_{ij}C_{ij}\;\; \text{s.t. } T\mathbf 1 = P,\; T^\top\mathbf 1 = q,\; T \ge 0; \qquad \text{WD} = \sqrt{\langle T^*, C\rangle}
$$

$$
K = e^{-C/\varepsilon},\quad u^{t+1} = \frac{P}{K v^{t}},\quad v^{t+1} = \frac{q}{K^\top u^{t+1}},\quad T = \operatorname{diag}(u)\,K\,\operatorname{diag}(v), \qquad v^0 = \mathbf 1
$$

**Q2.12 — Is the lecture's plan optimal?** *[New]* Take the lecture's example: $P = (0.5, 0.3, 0.2)$ and $q = (0.4, 0.4, 0.2)$ at positions $0, 1, 2$, with $C_{ij} = (i-j)^2$, and the plan $T = \begin{bmatrix}0.38&0.12&0\\0.02&0.25&0.03\\0&0.03&0.17\end{bmatrix}$. Check its marginals and cost, then find the optimal plan.

*Solution.* The rows sum to $(0.5, 0.3, 0.2) = P$ and the columns to $(0.4, 0.4, 0.2) = q$, so the plan is admissible. The only non-zero off-diagonal costs are 1, so $\langle T, C\rangle = (0.12 + 0.02 + 0.03 + 0.03)(1) = 0.20$. But $P$ and $q$ differ only in that bin 0 has $0.1$ too much and bin 1 has $0.1$ too little, so $T^* = \begin{bmatrix}0.4&0.1&0\\0&0.3&0\\0&0&0.2\end{bmatrix}$ costs $0.1 \times 1$.

$$\boxed{\langle T, C\rangle = 0.20 \text{ (admissible, not optimal)};\;\; \langle T^*, C\rangle = 0.10,\;\; \text{WD} = \sqrt{0.1} \approx 0.316}$$

The lecture's first plan shows admissibility only; the second, sparse plan in L2 §7.4 is the optimal one.

**Q2.13 — One Sinkhorn iteration (symmetric).** *[Notes L2-P7]* $P = q = (0.5, 0.5)$, $C = \begin{bmatrix}0&1\\1&0\end{bmatrix}$ and $\varepsilon = 1$.

*Solution.* $K = \begin{bmatrix}1&0.3679\\0.3679&1\end{bmatrix}$ and $K\mathbf 1 = (1.3679, 1.3679)$, so $u^1 = 0.5/1.3679$.

$$\boxed{u^1 \approx (0.3656, 0.3656)}$$

**Q2.14 — A full Sinkhorn iteration (asymmetric).** *[New]* The same $C$ and $\varepsilon$, but $P = (0.6, 0.4)$ and $q = (0.5, 0.5)$. Compute $u^1$, $v^1$ and $T$, and check its marginals.

*Solution.* $u^1 = (0.6, 0.4)/1.3679 = (0.4386, 0.2924)$. Then $K^\top u^1 = (0.4386 + 0.3679 \times 0.2924,\; 0.3679 \times 0.4386 + 0.2924) = (0.5462, 0.4538)$, so $v^1 = (0.5/0.5462, 0.5/0.4538) = (0.9154, 1.1018)$. $T_{ij} = u_i K_{ij} v_j$.

$$\boxed{T \approx \begin{bmatrix}0.4015 & 0.1778\\0.0985 & 0.3222\end{bmatrix},\;\; \text{columns } (0.5, 0.5) = q,\;\; \text{rows } (0.579, 0.421) \ne P,\;\; \langle T,C\rangle \approx 0.276}$$

The $v$-update fixes the columns exactly and breaks the rows; further iterations alternate until both hold. The cost is above the exact OT cost ($0.1$, moving $0.1$ from bin 1 to bin 2) because the entropy term spreads mass.

**Short answers (L2).**
- *Forward vs reverse KL?* Forward $D_{\mathrm{KL}}(P\Vert Q)$ is infinite where $Q = 0 < P$, so $Q$ covers every mode (mode-covering). Reverse $D_{\mathrm{KL}}(Q\Vert P)$ is infinite where $P \approx 0 < Q$, so $Q$ sits on one mode (mode-seeking).
- *Why OT over KL, JS or TVD?* Only OT measures how far mass moves, so it stays finite and keeps growing, and therefore keeps a gradient, when the supports are disjoint (Q2.8).
- *Why Sinkhorn?* The exact LP is expensive, and its optimum is sparse and non-smooth. The entropic term makes the problem strictly convex and turns the solution into alternating matrix-vector scalings, which are differentiable. As $\varepsilon \to 0$ the solution approaches the exact OT plan; larger $\varepsilon$ is faster and more diffuse.
- *Uncorrelated ⇒ independent?* Only for jointly Gaussian vectors.

---

## 3. Lecture 3: Contrastive and Generative Models

### 3.1 InfoNCE and the MI Bound

Theory: [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.2 The Contrastive Pipeline and the InfoNCE Loss|L3 §3.2]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.3 InfoNCE as a Lower Bound on Mutual Information|L3 §3.3]].

$$
\mathcal{L}_{\text{InfoNCE}} = -\ln\frac{e^{s_+/\tau}}{e^{s_+/\tau} + \sum_k e^{s_k/\tau}}, \qquad s = \cos(z_1, z_2), \qquad I(z_1; z_1^+) \ge \ln N - \mathcal{L}_{\text{InfoNCE}}
$$

Here $N$ is the number of candidates in the softmax (1 positive plus $N-1$ negatives). Dividing by the temperature $\tau$ is the same as multiplying by an inverse temperature $\alpha = 1/\tau$.

**Q3.1 — Large batch.** *[Notes L3-P4]* $N = 256$, $s_+ = 0.9$, $s_k = 0.1$ for all 255 negatives, and $\alpha = 10$.

*Solution.* The logits are 9 and 1: $\mathcal{L} = -\ln\frac{8103.1}{8103.1 + 255(2.718)} = -\ln 0.9212$, and $\ln 256 = 5.545$.

$$\boxed{\mathcal{L} \approx 0.082 \text{ nats},\;\; I \ge 5.46 \text{ nats}}$$

**Q3.2 — Unnormalized vectors and a temperature.** *[New]* The anchor has cosine $0.8$ with its positive and $0.5$, $0.2$, $-0.1$ with three negatives, and $\tau = 0.5$.

*Solution.* The logits $s/\tau$ are $(1.6, 1.0, 0.4, -0.2)$. The exponentials are $4.953 + 2.718 + 1.492 + 0.819 = 9.982$, so $p_+ = 0.496$, and $\ln 4 = 1.386$.

$$\boxed{\mathcal{L} \approx 0.701 \text{ nats},\;\; I \ge 1.386 - 0.701 \approx 0.686 \text{ nats}}$$

The bound can never exceed $\ln N$, which is why larger batches (more negatives) matter.

### 3.2 CLIP Loss and Zero-Shot Inference

Theory: [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.5 CLIP Training|L3 §3.5]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.6 Zero-Shot Inference with CLIP|L3 §3.6]].

$$
L_{ij} = \text{scale}\cdot v_i^\top t_j, \qquad \mathcal{L}_{\text{I2T}} = \frac1N\sum_i -\ln\frac{e^{L_{ii}}}{\sum_j e^{L_{ij}}}\;(\text{rows}), \qquad \mathcal{L}_{\text{T2I}} = \frac1N\sum_j -\ln\frac{e^{L_{jj}}}{\sum_i e^{L_{ij}}}\;(\text{columns}), \qquad \mathcal{L} = \tfrac12(\mathcal{L}_{\text{I2T}} + \mathcal{L}_{\text{T2I}})
$$

**Q3.3 — CLIP loss from a logit matrix.** *[Notes L3-P5]* $L = \begin{bmatrix}3&1\\0&2\end{bmatrix}$.

*Solution.*

For $\mathcal{L}_{\text{I2T}}$, apply the row softmax. Row 1 has $e^3 \approx 20.086$ and $e^1 \approx 2.718$, summing to $22.804$, so the diagonal term contributes $-\ln\frac{20.09}{22.80} = 0.127$. Row 2 has $e^0 = 1$ and $e^2 \approx 7.389$, summing to $8.389$, contributing $-\ln\frac{7.389}{8.389} = 0.127$. Averaging, $\mathcal{L}_{\text{I2T}} \approx 0.1269$.

For $\mathcal{L}_{\text{T2I}}$, apply the column softmax. Column 1 has $e^3 \approx 20.086$ and $e^0 = 1$, summing to $21.086$, contributing $-\ln\frac{20.09}{21.09} = 0.049$. Column 2 has $e^1 \approx 2.718$ and $e^2 \approx 7.389$, summing to $10.107$, contributing $-\ln\frac{7.389}{10.107} = 0.313$. Averaging, $\mathcal{L}_{\text{T2I}} \approx 0.1810$. 

Combining the two halves,
$$\boxed{\mathcal{L}_{\text{I2T}} \approx 0.127,\;\; \mathcal{L}_{\text{T2I}} \approx 0.181,\;\; \mathcal{L} \approx 0.154}$$

**Q3.4 — CLIP loss from embeddings.** *[New]* The images are $v_1 = (1, 0)$ and $v_2 = (0.6, 0.8)$, the captions are $t_1 = (0.8, 0.6)$ and $t_2 = (0, 1)$ (all unit norm), and scale $= 5$.

*Solution.* $v_1^\top t_1 = 0.8$, $v_1^\top t_2 = 0$, $v_2^\top t_1 = 0.48 + 0.48 = 0.96$ and $v_2^\top t_2 = 0.8$, so $L = \begin{bmatrix}4 & 0\\4.8 & 4\end{bmatrix}$. Row 1: $\ln(1 + e^{-4}) = 0.018$. Row 2: $\ln(1 + e^{0.8}) = 1.171$. The columns give the same two numbers, $1.171$ and $0.018$.

$$\boxed{\mathcal{L}_{\text{I2T}} = \mathcal{L}_{\text{T2I}} \approx 0.595}$$

Image 2 is closer to caption 1 than to its own caption ($0.96 > 0.8$), and that single mismatch produces most of the loss.

**Q3.5 — Zero-shot classification.** *[New]* The test image embedding is $(0.6, 0.8)$, the prompt embeddings are cat $(1, 0)$, dog $(0, 1)$ and car $(0.8, 0.6)$, and scale $= 10$.

*Solution.* The similarities are $0.6, 0.8, 0.96$, so the logits (scaled sim) are $(6, 8, 9.6)$. 

Compute $e^{s_k}$ for each class:

| Class   | Logit | $e^{s_k}$           |
| ------- | ----- | ------------------- |
| cat     | 6.0   | $e^6 = 403.4$       |
| dog     | 8.0   | $e^8 = 2981.0$      |
| car     | 9.6   | $e^{9.6} = 14765.0$ |
| **Sum** |       | **18149.4**         |
$p_{\text{cat}} = \frac{403.4}{18149.4} \approx 0.022$, $p_{\text{dog}} = \frac{2981.0}{18149.4} \approx 0.164$, $p_{\text{car}} = \frac{14765.0}{18149.4} \approx 0.814$, softmax is $(0.022, 0.164, 0.814)$.
$$\boxed{\text{prediction: car } (p \approx 0.81)}$$

### 3.3 VAE: KL Term, Reparameterization, ELBO

Theory: [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.3 The ELBO|L3 §5.3]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.5 Gradients of the ELBO and the Reparameterization Trick|L3 §5.5]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.6 Failure Modes: Posterior Collapse and Blurry Samples|L3 §5.6]].

$$
D_{\mathrm{KL}}\big(\mathcal N(\mu,\sigma^2)\Vert\mathcal N(0,I)\big) = \tfrac12\sum_k\big(\mu_k^2 + \sigma_k^2 - 1 - \ln\sigma_k^2\big), \qquad z = \mu + \sigma\odot\epsilon, \qquad \text{ELBO} = \mathbb{E}_q[\ln p(x\mid z)] - D_{\mathrm{KL}}
$$

For a Gaussian decoder with unit variance, $\ln p(x\mid z) = -\frac12\lVert x - \mu_d\rVert^2 - \frac d2\ln 2\pi$.

**Q3.6 — KL term.** *[Notes L3-P6]* $\mu = (0.5, -1)$ and $\sigma = (1, 0.5)$.

*Solution.* Dimension 1: $0.25 + 1 - 1 - 0 = 0.25$. Dimension 2: $1 + 0.25 - 1 + 1.3863 = 1.6363$.

$$\boxed{D_{\mathrm{KL}} \approx 0.943 \text{ nats}}$$

**Q3.7 — Reparameterized sample and a one-sample ELBO.** *[New]* The encoder gives $\mu = (1, -1)$ and $\sigma = (0.5, 2)$, and we draw $\epsilon = (0.2, -1)$. The decoder output for that $z$ is $\mu_d = (0.8, 2.4)$ for the input $x = (1, 2)$, with unit variance. Compute $z$, the KL, the reconstruction term and the ELBO.

*Solution.* $z = \mu + \sigma\odot\epsilon = (1 + 0.1,\; -1 - 2) = (1.1, -3)$. KL, dimension 1: $1 + 0.25 - 1 - \ln 0.25 = 1.636$. Dimension 2: $1 + 4 - 1 - \ln 4 = 2.614$. Halving the sum gives $2.125$. Reconstruction: $-\frac12(0.04 + 0.16) - \ln 2\pi = -0.1 - 1.838$.

$$\boxed{z = (1.1, -3),\;\; D_{\mathrm{KL}} \approx 2.125,\;\; \ln p(x\mid z) \approx -1.938,\;\; \text{ELBO} \approx -4.063}$$

> [!warning] Convention
> The notes write the trick as $z = \mu_e + \sigma_e^2\odot\epsilon$ but say to implement it with $\sigma_e$ ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.5 Gradients of the ELBO and the Reparameterization Trick|L3 §5.5]]). Multiply by the standard deviation. With the variance you would get $z_2 = -1 + 4(-1) = -5$, which is wrong.

**Q3.8 — Autoregressive likelihood.** *[New]* $P(x_1) = 0.5$, $P(x_2\mid x_1) = 0.4$ and $P(x_3\mid x_1,x_2) = 0.9$.

*Solution.* $P(x) = \prod_i P(x_i\mid x_{<i}) = 0.5 \times 0.4 \times 0.9$.

$$\boxed{P(x) = 0.18,\;\; \ln P(x) \approx -1.715}$$

**Short answers (L3).**
- *Why does CLIP average both directions?* The row and column softmaxes penalize different confusions (Q3.3 gives $0.127$ vs $0.181$). Using both makes images retrieve captions and captions retrieve images.
- *Why is the reparameterization trick needed?* You cannot backpropagate through the act of sampling. Writing $z = \mu + \sigma\epsilon$ moves the randomness into $\epsilon$, which does not depend on the parameters, so $\nabla_\psi$ passes through $\mu$ and $\sigma$.
- *What is posterior collapse?* The KL term drives $q(z\mid x) \to \mathcal N(0, I)$, so $z$ carries no information about $x$; the KL of that dimension is exactly 0.
- *Why is the ELBO a lower bound?* $\ln p(x) = \text{ELBO} + D_{\mathrm{KL}}(q(z\mid x)\Vert p(z\mid x))$, and the KL is $\ge 0$.

---

## 4. Lecture 4: GANs and Inductive Biases

### 4.1 GAN Value Function and Generator Gradients

Theory: [[ATML_Lecture_4 - GANs, and Inductive Biases#1.3 The Min-Max Game|L4 §1.3]], [[ATML_Lecture_4 - GANs, and Inductive Biases#1.6 Training GANs in Two Phases|L4 §1.6]], [[ATML_Lecture_4 - GANs, and Inductive Biases#1.7 Saturating vs. Non-Saturating GANs|L4 §1.7]].

$$
V(G,D) = \mathbb{E}[\ln D(x)] + \mathbb{E}[\ln(1 - D(G(z)))] \le 0, \qquad \mathcal{L}_D = -\tfrac1m\sum_i\big[\ln D(x^i) + \ln(1 - D(\hat x^i))\big], \qquad \mathcal{L}_G^{\text{NS}} = -\tfrac1m\sum_i \ln D(\hat x^i)
$$

With $D = \sigma(a)$, where $a$ is the discriminator logit: $\frac{d}{da}\big[-\ln D\big] = D - 1$ (non-saturating) and $\frac{d}{da}\ln(1-D) = -D$ (saturating).

**Q4.1 — Value function at three operating points.** *[Notes L4-P1]* (a) $D(x) = 1$, $D(G(z)) = 0$. (b) Both are $0.5$. (c) $0.8$ and $0.3$.

*Solution.* 
(a) $V_{(a)} = \log(1) + \log(1-0) = 0 + 0 = 0$. (Discriminator wins - the maximum)
(b) $V_{(b)} = \log(0.5) + \log(0.5) = -0.6931 - 0.6931=-1.387$. (Equilibrium, generator wins)
(c) $V_{(c)} = \log(0.8) + \log(0.7) = -0.2231 - 0.3567=-0.590$.

$$\boxed{V_a = 0\;(\max, \text{ discriminator wins}),\;\; V_b = -2\ln 2 \approx -1.386\;(\text{equilibrium}),\;\; V_c \approx -0.580}$$

**Q4.2 — Saturating vs non-saturating, in terms of $D$.** *[Notes L4-P2]* Early in training the discriminator is confident, $D(\hat x) = 0.05$. Differentiate with respect to $u = D$, and compare the magnitude of $\left|\frac{\partial}{\partial D(\hat{x})}\right|$ for the saturating objective $\log(1-D(\hat{x}))$ and the non-saturating objective $-\log D(\hat{x})$.

*Solution.* Differentiating each with respect to $u = D(\hat x)$: for the saturating form, $\frac{d}{du}\log(1-u) = \frac{-1}{1-u}$, whose magnitude at $u = 0.05$ is $\frac{1}{0.95} \approx 1.053$. For the non-saturating form, $\frac{d}{du}\big(-\log u\big) = \frac{-1}{u}$, whose magnitude at $u = 0.05$ is $\frac{1}{0.05} = 20$.

$$
\boxed{\left|\text{saturating}\right| \approx 1.05, \qquad \left|\text{non-saturating}\right| = 20, \qquad \text{ratio} \approx 19\times}
$$

**Q4.3 — Discriminator and generator losses on a batch.** *[New]* Take $m = 2$, with $D(x) = (0.9, 0.7)$ on real samples and $D(\hat x) = (0.2, 0.4)$ on fakes. Compute $\mathcal{L}_D$, $\mathcal{L}_G^{\text{NS}}$ and the saturating objective $\frac1m\sum\ln(1 - D(\hat x))$.

*Solution.* $\mathcal{L}_D = -\frac12\big[(\ln 0.9 + \ln 0.8) + (\ln 0.7 + \ln 0.6)\big] = -\frac12(-0.1054 - 0.2231 - 0.3567 - 0.5108)$. $\mathcal{L}_G^{\text{NS}} = -\frac12(\ln 0.2 + \ln 0.4) = \frac12(1.6094 + 0.9163)$. Saturating: $\frac12(\ln 0.8 + \ln 0.6) = \frac12(-0.2231 - 0.5108)$.

$$\boxed{\mathcal{L}_D \approx 0.598,\;\; \mathcal{L}_G^{\text{NS}} \approx 1.263,\;\; \text{saturating objective} \approx -0.367}$$

**Q4.4 — Past Quiz 2: non-saturating gradients at two operating points.** *[Past Quiz 2]* On a fake sample, the discriminator logit is $a$ and $D = \sigma(a)$. The generator has one scalar parameter $\theta$ with $\partial a/\partial\theta = 0.5$ and loss $\mathcal{L}_G = \ln(1 + e^{-a}) = -\ln D$. The operating points are OP-1 with $a = -2.1972$ and OP-2 with $a = +2.1972$. (a) In which case is $D$ strong? (b) Show $\partial\mathcal{L}_G/\partial a = D - 1$ and compare. (c) Take one GD step with $\eta = 0.1$ and update $a$ to first order. (d) Should $G$ push $a$ up or down?

*Formulas needed.* $\sigma(a) = \frac{1}{1+e^{-a}}$. Chain rule: $\frac{\partial\mathcal{L}_G}{\partial\theta} = \frac{\partial\mathcal{L}_G}{\partial a}\frac{\partial a}{\partial\theta}$. GD: $\theta' = \theta - \eta\frac{\partial\mathcal{L}_G}{\partial\theta}$. First order: $a' \approx a + \frac{\partial a}{\partial\theta}(\theta' - \theta)$. Note that $2.1972 = \ln 9$.

*Solution.* (a) OP-1: $e^{-a} = e^{\ln 9} = 9$, so $D = 1/10 = 0.1$. OP-2: $e^{-a} = 1/9$, so $D = 0.9$. **OP-1 is the strong discriminator**, since it gives the fake a probability of only $0.1$ of being real. At OP-2 it is fooled.

(b) $\frac{d}{da}\ln(1 + e^{-a}) = \frac{-e^{-a}}{1+e^{-a}}$, and $\frac{e^{-a}}{1+e^{-a}} = 1 - \sigma(a)$, so $\frac{\partial\mathcal{L}_G}{\partial a} = -(1 - D) = D - 1$. OP-1 gives $-0.9$ and OP-2 gives $-0.1$. **The gradient is 9× larger at OP-1**: the non-saturating loss gives the generator its strongest signal exactly when it is losing. The saturating form would give $-D = -0.1$ there instead, which vanishes.

(c) OP-1: $\partial\mathcal{L}_G/\partial\theta = -0.9 \times 0.5 = -0.45$, so $\theta' - \theta = +0.045$ and $a' = -2.1972 + 0.5(0.045) = -2.1747$. OP-2: $-0.05$, so $\Delta\theta = +0.005$ and $a' = 2.1972 + 0.0025 = 2.1997$.

$$\boxed{\text{OP-1: } a' \approx -2.1747\;(\Delta a = +0.0225); \qquad \text{OP-2: } a' \approx 2.1997\;(\Delta a = +0.0025)}$$

(d) **Up in both cases.** $\partial\mathcal{L}_G/\partial a = D - 1 < 0$ for every $a$, so raising $a$, which raises $D(G(z))$ and makes the fake look real, always lowers $\mathcal{L}_G$. The push is 9× stronger at OP-1.

### 4.2 Shape and Texture Bias

Theory: [[ATML_Lecture_4 - GANs, and Inductive Biases#4.1 Shape Bias and Texture Bias|L4 §4.1]], [[ATML_Lecture_4 - GANs, and Inductive Biases#4.3 The Cue-Conflict Problem|L4 §4.3]].

$$
\text{SB} = \frac{\#\text{shape-correct}}{\#\text{correct (shape or texture)}}, \qquad \text{TB} = \frac{\#\text{texture-correct}}{\#\text{correct}}, \qquad \text{SB} + \text{TB} = 1
$$

**Q4.5 — Cue-conflict counts.** *[Notes L4-P5]* Of 500 images, 400 are recognized as one of the two cue classes: 120 by shape and 280 by texture.

*Solution.* The denominator is the 400 correct recognitions, not the 500 images: SB $= 120/400$ and TB $= 280/400$.

$$\boxed{\text{SB} = 0.30,\;\; \text{TB} = 0.70 \;\;(\text{CNN-like; humans are near SB} \approx 1)}$$

### 4.3 Receptive Field, CNN vs ViT, and Past Quiz 1

Theory: [[ATML_Lecture_4 - GANs, and Inductive Biases#5.1 Locality Bias and the Receptive Field|L4 §5.1]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.2 Receptive Field and Depth|L4 §5.2]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.3 Pooling and the Receptive Field|L4 §5.3]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.5 Locality Bias vs. Global Context — CNN vs. ViT|L4 §5.5]]. The size formulas come from [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.5 CNNs — The Convolutional Layer|L3 §2.5]] and [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding|L3 §2.6]].

$$
H_2 = \Big\lfloor\frac{H_1 + 2P - F}{S}\Big\rfloor + 1, \qquad N_P = \frac{H}{P}\cdot\frac{W}{P}, \qquad \text{patch } (r,c) \text{ covers rows } P(r-1)+1 \ldots Pr
$$

RF rules (lecture convention): each $F\times F$ conv adds $F-1$ to the linear RF, scaled by the product of earlier strides. A $2\times2$, stride-2 pool doubles the linear RF, which multiplies the area by 4 ($9 \to 36$). Self-attention: $\operatorname{SA}(X) = V\operatorname{Softmax}(K^\top Q)$, so each column of $A$ holds the weights of one query and output token $i$ is $\sum_j a_{ji} v_j$.

**Q4.6 — CNN output shapes.** *[Notes L3-P1]* The input is $224\times224\times3$. Layer 1: 64 kernels, $F = 7$, $P = 3$, $S = 2$. Layer 2: 128 kernels, $F = 3$, $P = 1$, $S = 2$.

*Solution.* $\lfloor(224 + 6 - 7)/2\rfloor + 1 = 111 + 1 = 112$, then $\lfloor(112 + 2 - 3)/2\rfloor + 1 = 55 + 1 = 56$.
$$\boxed{112\times112\times64 \;\to\; 56\times56\times128}$$

**Q4.7 — Receptive field through depth and pooling.** *[Notes L4-P3]* Three $3\times3$ convs, then a $2\times2$/2 pool, then two more $3\times3$ convs.

*Solution.* The convs give $3 \to 5 \to 7$. The pool doubles this to $14$. After the pool each conv adds $2 \times 2 = 4$, giving $18 \to 22$.
$$\boxed{3, 5, 7 \xrightarrow{\text{pool}} 14, 18, 22 \;\;(\text{area } 22^2 = 484)}$$

**Q4.8 — Past Quiz 1: how far a local perturbation spreads.** *[Past Quiz 1]* The input is $x \in \mathbb{R}^{64\times64\times3}$. A perturbation $\delta$ is non-zero only on pixels in rows 1–2 and columns 1–2. The CNN is one $3\times3$ conv with stride 1, padding 1 and $C_{\text{out}} = 1$. The ViT has $P = 8$ (an $8\times8$ grid, $N = 64$), $D = 256$, single-head SA, and no positional encodings. Answer:

(a) CNN: What is the output size? How many output pixels does $\delta$ affect?
(b) ViT: When is the attention uniform, $a_{ij} = 1/N$? Which token(s) does $\delta$ hit at embedding? With uniform attention, how many tokens does $\delta$ influence after one block?

*Solution.* (a) $\lfloor(64 + 2 - 3)/1\rfloor + 1 = 64$, so the output is **$64\times64\times1$**. Output pixel $(r, c)$ reads input rows $r-1..r+1$ and columns $c-1..c+1$. It sees $\delta$ iff $r - 1 \le 2$ and $c - 1 \le 2$, i.e. $r, c \in \{1,2,3\}$ (there are no rows below 1).
Every output pixel $(r, c)$ is a weighted sum of a **3×3 window** of input pixels centered at $(r, c)$:
$$\text{output}(r,c) = \sum_{i=r-1}^{r+1}\sum_{j=c-1}^{c+1} w_{ij} \cdot x_{ij}$$

An output pixel is **affected by $\delta$** if its 3×3 window **overlaps** the perturbed region.
$$\boxed{64\times64\times1;\;\; 3\times3 = 9 \text{ affected pixels (out of 4096)}}$$

**Why not 4×4 = 16?**
The general formula for a $k \times k$ perturbation + $F \times F$ filter is:
$$(k + F - 1)^2 = (2 + 3 - 1)^2 = 4^2 = 16$$
But this assumes the patch is **interior** to the image. Here the patch is in the **corner** (rows/cols start at 1, nothing above or to the left), so the 4×4 footprint gets **clipped** to 3×3.

(b) *Uniform weights.* Softmax is shift-invariant, so $\operatorname{softmax}_j(s_{ij}) = 1/N$ iff $s_{ij} = q_i^\top k_j$ is the same for every key $j$. This happens when:
- $q_i = 0$, for example $W_Q = 0$;
- all keys are identical, $k_j = k$, for example $W_K = 0$ or identical tokens;
- more generally, $q_i \perp (k_j - k_l)$ for all $j, l$;
- or, in the limit, the scores are scaled toward 0 (an infinite temperature).

*Embedding.* Patch $(r, c)$ covers rows $8(r-1)+1 \ldots 8r$. Rows and columns 1–2 lie in patch $(1,1)$ only.
$$\boxed{\text{only token } (1,1)}$$

*After one uniform-attention block,* every output is $\operatorname{SA}(x)_i = \sum_j \frac1N v_j = \bar v$, the mean of all values. Since $v_{(1,1)} = W_V e_{(1,1)}$ changed, **every one of the $N = 64$ output tokens changes**, each by $\frac{1}{64}W_V\Delta e_{(1,1)}$.

$$\boxed{\text{all } 64 \text{ tokens}}$$

This is the contrast behind the lecture's bias claims: convolution confines $\delta$ to 9 of 4096 locations (strong locality), while one attention layer spreads it globally.

**Q4.9 — Variant: a perturbation straddling a patch boundary.** *[New]* The same set-up, but $\delta$ sits on rows 8–9 and columns 8–9. Which tokens does it hit, and how many CNN outputs does it affect after one and after two $3\times3$ convs (stride 1, pad 1)?

*Solution.* Row 8 lies in patch row 1 and row 9 in patch row 2, and the same holds for columns, so it hits tokens $(1,1), (1,2), (2,1), (2,2)$. After one conv the affected rows are $7..10$, giving $4\times4 = 16$. After a second conv they are $6..11$, giving $6\times6 = 36$.

$$\boxed{4 \text{ tokens};\;\; 16 \text{ then } 36 \text{ CNN outputs}}$$

### 4.4 Convolution vs Attention on Toy Inputs

Theory: [[ATML_Lecture_4 - GANs, and Inductive Biases#5.6 A Concrete Contrast on a Toy Matrix|L4 §5.6]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.7 Computational Scaling|L4 §5.7]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.9 Combining Both, Method 2 — Post- and Pre-Normalization Combination|L4 §5.9]], [[ATML_Lecture_4 - GANs, and Inductive Biases#5.10 Combining Both, Method 3 — Convolutional Modulation of Values|L4 §5.10]].

$$
(A * W)_{ij} = \sum_{m,n} A_{i+m,\,j+n}W_{mn}\;(\text{valid: } (n-f+1)\times(n-f+1)), \qquad D = \operatorname{Softmax}(QK^\top)V \;(\text{row-wise softmax}),
$$

$$
y_i^{\text{Post}} = \sum_j (A_{ij} + w_{i-j})x_j, \qquad y_i^{\text{Pre}} = \sum_j \operatorname{softmax}_j(x_i^\top x_j + w_{i-j})\,x_j
$$

**Q4.10 — Convolution enhances local blocks only.** *[New]* Convolve (valid, no padding) with $W = \begin{bmatrix}1&1\\1&1\end{bmatrix}$ each of $A = \begin{bmatrix}1&1&0&0\\1&1&0&0\\0&0&1&1\\0&0&1&1\end{bmatrix}$ (two local blocks) and $B = \begin{bmatrix}1&0&0&1\\0&1&1&0\\0&1&1&0\\1&0&0&1\end{bmatrix}$ (an X pattern).

*Solution.* The output is $3\times3$, and each entry is the sum of a $2\times2$ window.

$$\boxed{A * W = \begin{bmatrix}4&2&0\\2&2&2\\0&2&4\end{bmatrix}, \qquad B * W = \begin{bmatrix}2&2&2\\2&4&2\\2&2&2\end{bmatrix}}$$

For $A$ the two blocks light up with 4s. For $B$ only the centre crossing does; the four corners of the X, which belong to one global pattern, are never related to each other by a $2\times2$ window.

**Q4.11 — Self-attention relates distant tokens.** *[New]* Take three tokens with $Q = K = V = X = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$ (rows are tokens, no scaling). Compute $\operatorname{Softmax}(XX^\top)X$.

*Solution.* $XX^\top = \begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}$. Row softmax: row 1 has $(e^{1}, e^{0}, e^{1})$, giving $(0.422, 0.155, 0.422)$, and row 2 is its mirror; row 3 has $(e^1, e^1, e^2)$, giving $(0.212, 0.212, 0.576)$.

$$\boxed{A = \begin{bmatrix}0.422&0.155&0.422\\0.155&0.422&0.422\\0.212&0.212&0.576\end{bmatrix},\qquad D = AX = \begin{bmatrix}0.845&0.578\\0.578&0.845\\0.788&0.788\end{bmatrix}}$$

Every output mixes all tokens with content-dependent weights: $\text{RF} = G$.

**Q4.12 — Post- vs pre-normalization.** *[New]* The scores are $x_i^\top x_j = (2, 1, 0)$ over three positions, and the relative bias is $w_{i-j} = (0.5, 0, 0)$. Give the weights under each scheme.

*Solution.* $\operatorname{softmax}(2,1,0) = (0.665, 0.245, 0.090)$. Post adds the bias after the softmax: $(1.165, 0.245, 0.090)$, which sums to $1.5$. Pre adds it before: $\operatorname{softmax}(2.5, 1, 0) = (0.766, 0.171, 0.063)$.

$$\boxed{\text{Post: } (1.165, 0.245, 0.090),\; \Sigma = 1.5; \qquad \text{Pre: } (0.766, 0.171, 0.063),\; \Sigma = 1}$$

Only pre-normalization keeps a convex combination.

**Q4.13 — Cost and shapes.** *[Notes L4-P4 + New]* (a) Compare the costs of conv and attention on a $224\times224$ map, and with $16\times16$ patches. (b) For $N = 196$ tokens and $C = 64$ channels, give the shape of the attention matrix and of the convolutional-modulation $A$.

*Solution.* (a) Conv: $224^2 = 50{,}176$. Per-pixel attention: $50{,}176^2 \approx 2.5\times10^9$. Patch attention: $196^2 = 38{,}416$. (b) Attention: $A$ is $N\times N = 196\times196 = 38{,}416$ entries. CM: $A = \text{DWConv}(W_1X)$ is $N\times C = 196\times64 = 12{,}544$ entries, combined with $V$ by $\odot$.

$$\boxed{2.5\times10^9 \text{ vs } 5.0\times10^4 \text{ vs } 3.8\times10^4;\quad A_{\text{SA}}: 196\times196,\;\; A_{\text{CM}}: 196\times64}$$

### 4.5 Permutation Invariance and Equivariance

Theory: [[ATML_Lecture_4 - GANs, and Inductive Biases#6.3 Permutation Invariance|L4 §6.3]], [[ATML_Lecture_4 - GANs, and Inductive Biases#6.4 Permutation Equivariance|L4 §6.4]].

$$
T_\pi(X) = XP_\pi,\;\; P_\pi = [e_{\pi(1)}\cdots e_{\pi(n)}],\;\; P_\pi P_\pi^\top = I; \qquad \text{P.I.: } A(T_\pi x) = A(x); \qquad \text{P.E.: } T_\pi(A(x)) = A(T_\pi x)
$$

**Q4.14 — Permutation matrix action.** *[Notes L4-P6]* $P_\pi = \begin{bmatrix}0&1&0\\0&0&1\\1&0&0\end{bmatrix}$ and $X = [c_1\; c_2\; c_3]$.

*Solution.* Column $j$ of $XP_\pi$ is $X$ times column $j$ of $P_\pi$: $(0,0,1)^\top \to c_3$, $(1,0,0)^\top \to c_1$ and $(0,1,0)^\top \to c_2$.

$$\boxed{XP_\pi = [c_3\; c_1\; c_2],\;\; P_\pi P_\pi^\top = I}$$

**Short answers (L4).**
- *Why is the saturating GAN loss bad early in training?* When $D(\hat x) \approx 0$, $\ln(1 - D)$ is flat, so the generator gets almost no gradient (Q4.2 and Q4.4).
- *What is mode collapse?* The generator maps many $z$ to a few outputs that currently fool $D$, covering one mode of $P(x)$.
- *CNN or ViT with little data?* CNN. Its strong locality bias is a correct prior for images, so it needs less data. The ViT's weak bias must be learned from lots of data.
- *What is the ERF?* The Gaussian-weighted central part of the theoretical RF that actually influences a unit, so the usable context is smaller than the RF.
- *Which operators are P.I. or P.E.?* Convolution is neither. Self-attention is P.E. With a learnable query ($Q$ not computed from $X$) it is P.I. A ViT with positional embeddings is neither, which is why those embeddings are needed.
- *Invariance vs equivariance?* Invariance discards the transformation ($f(x+\delta) = f(x)$, as for classification). Equivariance carries it through ($f(x+\delta) = f(x) + \delta$, as for segmentation). Conv without pooling is translation-equivariant; pooling adds partial invariance.

---

## 5. Lecture 5.1: Domain Adaptation

### 5.1 ERM Error Decomposition and Margins

Theory: [[ATML_Lecture_5_1 - Domain Adaptation#3.2 Approximation Error|L5.1 §3.2]], [[ATML_Lecture_5_1 - Domain Adaptation#3.4 Estimation Error and Excess Risk|L5.1 §3.4]], [[ATML_Lecture_5_1 - Domain Adaptation#3.5 When Does the Estimation Error Vanish?|L5.1 §3.5]].

$$
AE = R^*_{\min} - R_{\min}, \qquad EE = R_{\text{emp}}(\hat f) - R_{\text{True}}(f^*), \qquad R_{\text{excess}} = AE + EE, \qquad \mathcal{L}_{\text{CE}_i} = \ln\Big(1 + \sum_{k\ne j}e^{-m_{jk}}\Big),\; m_{jk} = \psi_j^\top z - \psi_k^\top z
$$

**Q5.1 — Excess risk.** *[Notes L5.1-P1]* $R_{\min} = 0.05$, $R^*_{\min} = 0.12$ and $R_{\text{emp}}(\hat f) = 0.18$.

*Solution.* $AE = 0.07$ and $EE = 0.06$.

$$\boxed{R_{\text{excess}} = 0.13;\;\; \text{a bigger architecture reduces only } AE}$$

**Q5.2 — Margins and CE.** *[Notes L5.1-P2]* The logits are $(1, 4, 0.5, 2)$ with true class 2. Then raise the true logit to 8.

*Solution.* $m = (3, 3.5, 2)$, so $\ln(1 + e^{-3} + e^{-3.5} + e^{-2}) = \ln 1.2153$. With the true logit at 8, $m = (7, 7.5, 6)$ and $\ln 1.00394$.

$$\boxed{0.195 \text{ nats} \to 0.0039 \text{ nats}}$$

A modest increase in one logit makes the loss nearly zero. This is why low capacity is enough to fit a spurious correlation.

### 5.2 The DA Bound

Theory: [[ATML_Lecture_5_1 - Domain Adaptation#5.3 The DA Upper-Bound Theorem|L5.1 §5.3]].

$$
L_T \le L_S + \text{Div}(P_S(x), P_T(x)) + \min(a, b), \qquad a = \mathbb{E}_{P_S}\lvert P_S(Y\mid x) - P_T(Y\mid x)\rvert,\; b = \mathbb{E}_{P_T}[\cdot]
$$

**Q5.3 — CSDA vs PSDA.** *[Notes L5.1-P3]* $L_S = 0.10$ and Div $= 0.25$. Take $\min(a,b) = 0$ for CSDA, and $a = 0.30$, $b = 0.18$ for PSDA.

*Solution.* CSDA: $0.10 + 0.25 + 0$. PSDA: $+ \min(0.30, 0.18) = 0.18$.

$$\boxed{L_T \le 0.35 \text{ (CSDA)},\;\; 0.53 \text{ (PSDA)}}$$

### 5.3 Measure- and Adversarial-Based DA

Theory: [[ATML_Lecture_5_1 - Domain Adaptation#6.1 Measure-Based — Deep Adaptation Network (DAN)|L5.1 §6.1]], [[ATML_Lecture_5_1 - Domain Adaptation#6.2 Adversarial-Based — Domain Discriminator and the Gradient Reversal Layer (DANN)|L5.1 §6.2]]. MMD is from [[ATML_Lecture_2 - Information Theory and Divergences#6.2 Maximum Mean Discrepancy|L2 §6.2]].

$$
\mathcal{L}_{\text{DAN}} = \mathcal{L}_{\text{CE}} + \lambda\sum_{\ell=\ell_1}^{\ell_2} d^2(\mathcal{D}_S^\ell, \mathcal{D}_T^\ell); \qquad \text{linear kernel: } d^2 = \lVert\bar z_S - \bar z_T\rVert^2; \qquad \text{GRL backward: } \times(-\lambda)
$$

**Q5.4 — DAN objective.** *[Notes L5.1-P4]* $\lambda = 0.5$, the distances are $0.4, 0.3, 0.2$ and $\mathcal{L}_{\text{CE}} = 0.35$.

*Solution.* Square the distances: $0.16 + 0.09 + 0.04 = 0.29$. Then $0.35 + 0.5(0.29)$.

$$\boxed{\mathcal{L}_{\text{Total}} = 0.495 \;(\text{not } 0.80\text{: square the distances})}$$

**Q5.5 — MMD with a linear kernel, into DAN.** *[New]* The source features are $(1,0)$ and $(3,2)$, and the target features are $(0,1)$ and $(2,1)$, at one layer. With $\lambda = 1$ and $\mathcal{L}_{\text{CE}} = 0.3$, compute the DAN loss.

*Solution.* $\bar z_S = (2, 1)$ and $\bar z_T = (1, 1)$, so $d^2 = \lVert(1, 0)\rVert^2 = 1$. The total is $0.3 + 1 \times 1$.

$$\boxed{d^2 = 1,\;\; \mathcal{L}_{\text{DAN}} = 1.3}$$

A linear kernel matches only the means. An RBF kernel matches all moments.

**Q5.6 — Gradient reversal.** *[Notes L5.1-P5]* $\partial\mathcal{L}_{\text{dom}}/\partial u = 0.8$, $\partial u/\partial z = 0.5$, $\partial z/\partial\theta_f = 2$ and $\lambda = 0.1$.

*Solution.* Without the GRL the product is $0.8 \times 0.5 \times 2 = 0.8$. With it, multiply by $-\lambda$.

$$\boxed{+0.8 \text{ (makes features domain-separable)} \;\to\; -0.08 \text{ (makes them domain-invariant)}}$$

### 5.4 Contrastive DA: Prototypes, Pseudo-Labels, CDC Loss

Theory: [[ATML_Lecture_5_1 - Domain Adaptation#6.3 Contrastive-Based — Prototypes, Pseudo-Labels, and Cross-Domain Contrastive Loss|L5.1 §6.3]], [[ATML_Lecture_5_1 - Domain Adaptation#6.4 Pseudo-Label Generation|L5.1 §6.4]].

$$
\mu_c = \frac{1}{N_c}\sum_{i: y_i = c}\frac{f(x_i)}{\lVert f(x_i)\rVert}, \qquad \mathcal{L}_{S2T}(a) = -\ln\frac{\sum_p e^{\text{sim}(z_a, z_p)}}{\sum_k e^{\text{sim}(z_a, z_k)}}
$$

In $\mathcal{L}_{S2T}$ the denominator runs over the positives and the other-class samples.

**Q5.7 — Prototype and gated target loss.** *[Notes L5.1-P6]* The features are $(3,4)$, $(0,5)$ and $(6,8)$. The target pseudo-label probabilities are $(0.95, 0.60, 0.88, 0.40)$ with $\tau = 0.7$.

*Solution.* Normalize first: $(0.6, 0.8)$, $(0, 1)$ and $(0.6, 0.8)$, then average. The gate keeps only $0.95$ and $0.88$: $\frac12(0.0513 + 0.1278)$.

$$\boxed{\mu_c = (0.4, 0.867),\;\; \mathcal{L}_T \approx 0.090 \;(2 \text{ of } 4 \text{ kept})}$$

**Q5.8 — Pseudo-label by nearest prototype, and one S2T term.** *[New]* The prototypes are $\mu_1 = (1, 0)$ and $\mu_2 = (0.6, 0.8)$, and the target feature is $f = (3, 1)$. (a) Assign its pseudo-label by cosine. (b) A source anchor has similarity $0.9$ and $0.7$ with two same-class target samples and $0.2$, $0.1$ with two other-class samples. Compute $\mathcal{L}_{S2T}$.

*Solution.* (a) $\lVert f\rVert = \sqrt{10}$, so $\hat f = (0.949, 0.316)$. Then $\cos(\hat f, \mu_1) = 0.949$ and $\cos(\hat f, \mu_2) = 0.569 + 0.253 = 0.822$, so the label is **class 1**. (b) Numerator: $e^{0.9} + e^{0.7} = 2.460 + 2.014 = 4.474$. Denominator: $4.474 + e^{0.2} + e^{0.1} = 4.474 + 1.221 + 1.105 = 6.800$. $-\ln(4.474/6.800)$.

$$\boxed{\text{pseudo-label } 1;\;\; \mathcal{L}_{S2T} \approx 0.419}$$

### 5.5 Prompt Learning (CLIP-Style DA)

Theory: [[ATML_Lecture_5_1 - Domain Adaptation#6.5 Multimodality-Based — Prompt Learning in CLIP Style|L5.1 §6.5]], [[ATML_Lecture_5_1 - Domain Adaptation#6.6 Insights on Prompt Learning|L5.1 §6.6]].

$$
P(\hat y = K\mid x^s) = \frac{e^{\text{sim}(g(t_K^s), f(x^s))}}{\sum_{d\in\{S,T\}}\sum_j e^{\text{sim}(g(t_j^d), f(x^s))}}, \qquad t^{SH} \leftarrow t^{SH} - \eta\Big(\nabla\mathcal{L}_T + \lambda\sum_i\nabla\mathcal{L}_{S_i}\Big)
$$

**Q5.9 — Past Quiz 3: logit-sum vs probability-average prompt ensembles.** *[Past Quiz 3]* Frozen CLIP classifies dog vs cat. The dog image embedding is $v = (1, 0)$, and the unit prompts are $t_1 = t_{\text{photo-dog}} = (0.8, 0.6)$, $t_2 = t_{\text{sketch-dog}} = (0.6, 0.8)$ and $t_3 = t_{\text{photo-cat}} = (0.58, 0.81)$. The logits are $z_k = v^\top t_k/\gamma$ with $\gamma = 0.1$, giving $z = (8.0, 6.0, 5.8)$. The loss is BCE with ground truth dog, i.e. $\mathcal{L} = -\ln p_{\text{dog}}$.
- S1 (logit sum): $z_{\text{dog}} = z_1 + z_2$, $z_{\text{cat}} = z_3$, $p_{\text{dog}} = \frac{e^{z_{\text{dog}}}}{e^{z_{\text{dog}}} + e^{z_{\text{cat}}}}$.
- S2 (probability average): $p_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_3}}$, $p_2 = \frac{e^{z_2}}{e^{z_2}+e^{z_3}}$, $p_{\text{dog}} = \frac12(p_1 + p_2)$.

Find $\partial\mathcal{L}/\partial t_k$ and $\lVert\partial\mathcal{L}/\partial t_k\rVert_2$ under each scheme, then compare them.

*Formulas needed.*
- A two-way softmax is a sigmoid of the difference: $\frac{e^{a}}{e^{a}+e^{b}} = \sigma(a - b)$.
- $\sigma' = \sigma(1-\sigma)$.
- $\frac{d}{dp}(-\ln p) = -\frac1p$.
- Since $z_k = v^\top t_k/\gamma$, $\frac{\partial z_k}{\partial t_k} = \frac{v}{\gamma} = (10, 0)$, with $\lVert v/\gamma\rVert = 10$.
- Chain rule: $\frac{\partial\mathcal{L}}{\partial t_k} = \frac{\partial\mathcal{L}}{\partial z_k}\frac{v}{\gamma}$.

*S1.* $p_{\text{dog}} = \sigma(z_1 + z_2 - z_3) = \sigma(8.2) = 0.999725$. Then $\frac{\partial\mathcal{L}}{\partial z_{\text{dog}}} = p_{\text{dog}} - 1$ and $\frac{\partial\mathcal{L}}{\partial z_{\text{cat}}} = 1 - p_{\text{dog}}$. Since $z_{\text{dog}}$ contains both $z_1$ and $z_2$,

$$
\frac{\partial\mathcal{L}}{\partial t_1} = \frac{\partial\mathcal{L}}{\partial t_2} = -(1 - p_{\text{dog}})\frac{v}{\gamma}, \qquad \frac{\partial\mathcal{L}}{\partial t_3} = (1 - p_{\text{dog}})\frac{v}{\gamma}.
$$

$$\boxed{\text{S1: } \lVert\cdot\rVert_2 = 10(2.75\times10^{-4}) \approx 2.75\times10^{-3} \text{ for all three prompts};\;\; \mathcal{L} \approx 2.7\times10^{-4}}$$

*S2.* $p_1 = \sigma(z_1 - z_3) = \sigma(2.2) = 0.9002$ and $p_2 = \sigma(z_2 - z_3) = \sigma(0.2) = 0.5498$, so $p_{\text{dog}} = 0.7250$ and $\mathcal{L} = 0.3215$. With $\frac{\partial p_1}{\partial z_1} = p_1(1-p_1)$ and $\frac{\partial p_1}{\partial z_3} = -p_1(1-p_1)$ (and similarly for $p_2$),

$$
\frac{\partial\mathcal{L}}{\partial z_1} = -\frac{p_1(1-p_1)}{2p_{\text{dog}}}, \qquad \frac{\partial\mathcal{L}}{\partial z_2} = -\frac{p_2(1-p_2)}{2p_{\text{dog}}}, \qquad \frac{\partial\mathcal{L}}{\partial z_3} = \frac{p_1(1-p_1) + p_2(1-p_2)}{2p_{\text{dog}}}.
$$

Numerically, $p_1(1-p_1) = 0.0898$ and $p_2(1-p_2) = 0.2475$, with $2p_{\text{dog}} = 1.4501$. That gives $\partial\mathcal{L}/\partial z = (-0.0619, -0.1707, +0.2326)$; multiply by $v/\gamma$.

$$\boxed{\text{S2: } \lVert\partial\mathcal{L}/\partial t_1\rVert \approx 0.619,\;\; \lVert\partial\mathcal{L}/\partial t_2\rVert \approx 1.707,\;\; \lVert\partial\mathcal{L}/\partial t_3\rVert \approx 2.326}$$

*Comparison.* Under S1 the strong photo prompt inflates $z_{\text{dog}}$ until the loss saturates. Every prompt then gets the same near-zero gradient, so the weak sketch prompt never adapts. Under S2 each prompt competes with the cat prompt separately. The weak sketch prompt ($p_2 = 0.55$) gets about 2.8× the gradient of the strong one, so **S2 adapts better under domain shift**. The cost is a less confident prediction ($p_{\text{dog}} = 0.725$ vs $0.9997$, i.e. a higher loss), and the cat prompt $t_3$ gets the largest push, which can disturb the cat class.

**Q5.10 — Posterior over $2K$ prompts.** *[New]* There are $K = 2$ classes and two domains. A source dog image has similarities $0.9$ (dog, S), $0.3$ (cat, S), $0.6$ (dog, T) and $0.2$ (cat, T), with scale 10. Compute $P(\hat y = \text{dog}\mid x, t^S_{\text{dog}})$ and $\mathcal{L}_S$ for this sample.

*Solution.* The logits are $(9, 3, 6, 2)$ and the softmax over all four is $(0.9495, 0.0024, 0.0473, 0.0009)$. The loss is $-\ln 0.9495$.

$$\boxed{P \approx 0.950,\;\; \mathcal{L}_S \approx 0.052}$$

Using only the source prompts would give $0.9975$. The target dog prompt is the main competitor, which is how the shared denominator ties the domains together.

**Q5.11 — Shared-token update and gradient conflict.** *[New]* $t^{SH} = (1, 1)$, $\eta = 0.1$, $\lambda = 0.5$, $\nabla\mathcal{L}_T = (0.2, -0.4)$, $\nabla\mathcal{L}_{S_1} = (0.4, 0.2)$ and $\nabla\mathcal{L}_{S_2} = (-0.2, 0.6)$. Update the token and check for conflict.

*Solution.* $\sum_i\nabla\mathcal{L}_{S_i} = (0.2, 0.8)$. The combined gradient is $(0.2, -0.4) + 0.5(0.2, 0.8) = (0.3, 0)$, so $t' = (1, 1) - 0.1(0.3, 0)$. For conflict: $\cos\big(\nabla\mathcal{L}_T, \sum\nabla\mathcal{L}_S\big) = \frac{0.04 - 0.32}{0.447 \times 0.825} = -0.76$.

$$\boxed{t^{SH\prime} = (0.97, 1.00);\;\; \cos \approx -0.76 \Rightarrow \text{conflict}}$$

The second coordinate's gradients cancel exactly, so it does not move. The notes' rule is to resolve conflict before adding.

**Short answers (L5.1).**
- *Covariate vs semantic shift?* Covariate shift: $P(x)$ changes while $P(y\mid x)$ stays the same. Semantic (concept) shift: $P(y\mid x)$ changes or new classes appear. DA in this course handles covariate shift (CSDA).
- *Why doesn't marginal alignment (DAN/DANN) fix semantic shift?* It aligns $P_S(x)$ with $P_T(x)$ regardless of class, so it can align a source cow with a target camel. The $\min(a,b)$ term remains.
- *Why learn the source prompts too?* If they were frozen, the shared token would absorb source-specific fitting, and the frozen prompts (which are the class prototypes) could be miscalibrated for this encoder and data.
- *What does the confidence threshold $\tau$ do?* It drops low-confidence pseudo-labels, which are likely wrong, so noise is not reinforced.

---

## 6. Lecture 5.2: Domain Generalization and SSL

### 6.1 DG Bound and Contrastive Gradients

Theory: [[ATML_Lecture_5_2 - Domain Generalization#1.2 The DG Error Bound|L5.2 §1.2]], [[ATML_Lecture_5_2 - Domain Generalization#2.5 The Hard-Positives Problem|L5.2 §2.5]], [[ATML_Lecture_5_2 - Domain Generalization#2.6 Hard Positives Also Waste the Negative Gradients|L5.2 §2.6]], [[ATML_Lecture_5_2 - Domain Generalization#2.8 Proxy-Based Contrastive Loss|L5.2 §2.8]].

$$
L_T \le \sum_i \pi_i L_i + \gamma + \rho + \text{const}, \qquad \frac{\partial L_{\text{CL}}}{\partial s_{\text{pos}}} = P_{\text{pos}} - 1,\;\; \frac{\partial L_{\text{CL}}}{\partial s_k} = P_k,\;\; \sum_k P_k = 1 - P_{\text{pos}}
$$

With an inverse temperature $\alpha$ (logits $\alpha s$), every gradient picks up a factor $\alpha$: $\partial L/\partial s_+ = \alpha(p_+ - 1)$ and $\partial L/\partial s_j = \alpha p_j$.

**Q6.1 — DG bound.** *[Notes L5.2-P1]* $L = (0.10, 0.14, 0.18)$ with $\pi = 1/3$ each, $\gamma = 0.22$, $\rho = 0.09$ and const $= 0.02$.

*Solution.* $0.14 + 0.22 + 0.09 + 0.02$. Halving each lever saves $0.07$ (source risk), $0.11$ ($\gamma$) or $0.045$ ($\rho$).

$$\boxed{L_T \le 0.47;\;\; \gamma \text{ is the largest lever but can't be minimized directly (no target data)}}$$

**Q6.2 — Pull/push identity.** *[Notes L5.2-P2]* $s_{\text{pos}} = 3$ and the negatives are $s = (1, 0.5, 0, -1)$.

*Solution.* The exponentials sum to $20.086 + 2.718 + 1.649 + 1 + 0.368 = 25.821$, so $P_{\text{pos}} = 0.778$ and the negatives are $(0.105, 0.064, 0.039, 0.014)$.

$$\boxed{\lvert\partial L/\partial s_{\text{pos}}\rvert = 0.222 = \textstyle\sum_k P_k}$$

With a hard positive at $s_{\text{pos}} = -1$, $P_{\text{pos}} \approx 0.06$, so the pull, and the total push, is about $0.94$.

**Q6.3 — Sample Quiz 4, P1: inverse temperature and hard-negative concentration.** *[Past Quiz: Sample Qz 4 P1]* All vectors are unit norm. $s_+ = 0.50$ and the negatives are $s_1 = 0.48$ (a near miss), $s_2 = 0.10$ and $s_3 = -0.20$. The logits are $\alpha s$ with $\alpha \in \{5, 10\}$, $Z = e^{\alpha s_+} + \sum_j e^{\alpha s_j}$ and $L = -\alpha s_+ + \ln Z$. (1) Show $\partial L/\partial s_+ = \alpha(p_+ - 1)$ and $\partial L/\partial s_j = \alpha p_j$. (2) Compute the probabilities and gradients for both $\alpha$. (3) Show that as $\alpha\to\infty$ the gradient concentrates on the hardest pair. (4) What share of the negative push lands on $s_1$?

*Solution.* (1) $\frac{\partial\ln Z}{\partial s_+} = \frac{\alpha e^{\alpha s_+}}{Z} = \alpha p_+$, so $\frac{\partial L}{\partial s_+} = -\alpha + \alpha p_+ = \alpha(p_+ - 1)$. Likewise $\frac{\partial L}{\partial s_j} = \frac{\alpha e^{\alpha s_j}}{Z} = \alpha p_j$.

(2) $\alpha = 5$: the logits are $(2.5, 2.4, 0.5, -1)$ and the exponentials $(12.182, 11.023, 1.649, 0.368)$, with $Z = 25.222$. $\alpha = 10$: the logits are $(5, 4.8, 1, -2)$ and the exponentials $(148.41, 121.51, 2.718, 0.135)$, with $Z = 272.78$.

$$\boxed{\begin{aligned}\alpha=5:&\;\; p = (0.483, 0.437, 0.065, 0.015),\;\; \tfrac{\partial L}{\partial s} = (-2.585,\; 2.185,\; 0.327,\; 0.073)\\ \alpha=10:&\;\; p = (0.544, 0.446, 0.010, 0.0005),\;\; \tfrac{\partial L}{\partial s} = (-4.559,\; 4.455,\; 0.100,\; 0.005)\end{aligned}}$$

(3) With $s_{\max} = \max_\ell s_\ell$, write $p_k = \frac{e^{\alpha(s_k - s_{\max})}}{\sum_\ell e^{\alpha(s_\ell - s_{\max})}}$. Every exponent is $\le 0$. Terms with $s_\ell = s_{\max}$ stay at $e^0 = 1$, and the rest go to $0$ as $\alpha\to\infty$, so $p_k \to 1/\lvert\arg\max\rvert$ on the maximizers and $0$ elsewhere. Among the negatives, $p_j/p_1 = e^{\alpha(s_j - s_1)} \to 0$ for $j \ne 1$, so the whole push $\alpha p_j$ lands on the hardest negative $s_1$.

(4) The share is $\frac{p_1}{p_1+p_2+p_3}$; the factor $\alpha$ cancels. For $\alpha = 5$: $\frac{0.437}{0.517}$. For $\alpha = 10$: $\frac{0.4455}{0.4560}$.

$$\boxed{\text{share}(s_1) \approx 84.5\% \;(\alpha = 5),\;\; 97.7\% \;(\alpha = 10)}$$

High $\alpha$ is hard-negative mining: nearly all the repulsion goes to the near miss. In DG the near miss is often a cross-domain sample with similar semantics, so over-repelling it distorts the geometry, which hurts DG (compare hard positives, [[ATML_Lecture_5_2 - Domain Generalization#2.6 Hard Positives Also Waste the Negative Gradients|L5.2 §2.6]]). Low $\alpha$ spreads the push more smoothly but discriminates less sharply.

**Q6.4 — Proxy contrastive term.** *[New]* An anchor has similarity $0.8$ with its own proxy, $0.1$ with the other class's proxy and $0.5$ with one other sample. Compute $L_{\text{PCL}}$ for this anchor, adding the $-\ln$ the notes omit.

*Solution.* $-\ln\frac{e^{0.8}}{e^{0.8} + e^{0.1} + e^{0.5}} = -\ln\frac{2.226}{2.226 + 1.105 + 1.649}$.

$$\boxed{L_{\text{PCL}} \approx 0.805}$$

The positive is a stable proxy $v_c$, not a single hard positive sample, so it avoids the huge pull.

### 6.2 ERM vs IRM, IRMv1 and Scale Collapse

Theory: [[ATML_Lecture_5_2 - Domain Generalization#3.1 Invariant Risk Minimization — Idea and Intuition|L5.2 §3.1]], [[ATML_Lecture_5_2 - Domain Generalization#3.3 IRMv1 — A Practical Surrogate|L5.2 §3.3]], [[ATML_Lecture_5_2 - Domain Generalization#3.5 Issues with the First-Order Surrogate, and Alternatives|L5.2 §3.5]].

$$
\min_\theta\sum_e R_e + \lambda\sum_e\big\lVert\nabla_w R_e\vert_{w=1}\big\rVert^2, \qquad \arg\min_u\big[\alpha(u-1)^2 + \beta(u+1)^2\big] = \frac{\alpha - \beta}{\alpha + \beta}, \qquad \frac{\partial}{\partial w}(w\theta - c)^2 = 2\theta(w\theta - c)
$$

**Q6.5 — Past Quiz 4: IRMv1 scale collapse.** *[Past Quiz 4]* The prediction is $\hat y = w\theta$ with scalars $w, \theta$. The environment risks are $R_1 = (w\theta - 1)^2 + 0.01$ and $R_2 = (w\theta + 1)^2 + 0.01$, and the pooled risk is $R_{\text{pool}} = 0.9R_1 + 0.1R_2$. (1) With $u = w\theta$, find the ERM optimum $u^*$ with $w = 1$, and compare $R_{\text{pool}}$ there against the collapsed $\theta = 0$. (2) Compute the IRMv1 penalty $P(\theta) = \big(\partial_w R_1\vert_{w=1}\big)^2 + \big(\partial_w R_2\vert_{w=1}\big)^2$ and find $\lambda^*$ where $J^{nc}_\lambda = J^{col}_\lambda$. (3) Why is this scale collapse and not invariance?

*Solution.* (1) $\frac{d}{du}R_{\text{pool}} = 1.8(u-1) + 0.2(u+1) = 2u - 1.6 = 0$, so $u^* = 0.8$ and $(w, \theta^*) = (1, 0.8)$. $R_{\text{pool}}(0.8) = 0.9(0.04) + 0.1(3.24) + 0.01 = 0.37$ and $R_{\text{pool}}(0) = 0.9 + 0.1 + 0.01 = 1.01$.

$$\boxed{u^* = 0.8,\;\; \theta^*_{\text{erm}} = 0.8;\;\; R_{\text{pool}} = 0.37 \text{ vs } 1.01 \Rightarrow \text{ERM prefers non-collapsed}}$$

(2) $\partial_w R_1 = 2\theta(w\theta - 1)$ and $\partial_w R_2 = 2\theta(w\theta + 1)$, so at $w = 1$, $P(\theta) = 4\theta^2\big[(\theta-1)^2 + (\theta+1)^2\big] = 8\theta^2(\theta^2 + 1)$. $P(0.8) = 8(0.64)(1.64) = 8.397$ and $P(0) = 0$. So $J^{nc} = 0.37 + 8.397\lambda$ and $J^{col} = 1.01$. Setting them equal gives $\lambda^* = 0.64/8.397$.

$$\boxed{P(\theta) = 8\theta^2(\theta^2+1),\;\; \lambda^* \approx 0.0762;\;\; \lambda > \lambda^* \Rightarrow \text{collapse } (\theta = 0),\;\; \lambda < \lambda^* \Rightarrow \text{non-collapsed}}$$

(3) The penalty is proportional to $\theta^2$, because $\partial_w R_e$ carries a factor $\theta$. It can therefore be zeroed by shrinking the representation's scale rather than by making it invariant. $\theta = 0$ makes the gradients vanish trivially, since a constant predictor is stationary in every environment. This is exactly the spurious stationary solution of a first-order surrogate ([[ATML_Lecture_5_2 - Domain Generalization#3.5 Issues with the First-Order Surrogate, and Alternatives|L5.2 §3.5]]). It is not genuine invariance, because $\theta = 0$ carries no information about $y$. The environments' optima ($u = 1$ vs $u = -1$) conflict, and IRMv1 resolves the conflict by collapsing.

**Q6.6 — Sample Quiz 4, P2: causal vs spurious representation.** *[Past Quiz: Sample Qz 4 P2]* For the causal feature, $R_1^{(c)} = R_2^{(c)} = (w-1)^2 + 0.12$. For the spurious feature, $R_1^{(s)} = (w-1)^2$ and $R_2^{(s)} = (w+1)^2 + 1.5$. The pooled risk is $0.99R_1 + 0.01R_2$, and the IRMv1 penalty is $\sum_e\lVert\nabla_w R_e\rVert^2$ at $w = 1$. (1) Give the ERM solutions and which feature ERM picks. (2) Give the IRM penalties and which feature IRM picks. (3) Compare target risks for the mixture $0.99/0.01$, held-out $R_2$, and a balanced mix. (4) Find the mixture weight $p^*$ at which the two fixed models tie.

*Solution.* (1) Causal: $R_{\text{pool}} = (w-1)^2 + 0.12$, so $w_{\text{ERM}}^{(c)} = 1$ with risk $0.12$. Spurious: $1.98(w-1) + 0.02(w+1) = 0$ gives $w_{\text{ERM}}^{(s)} = 0.98$, with risk $0.99(0.0004) + 0.01(1.98^2 + 1.5) = 0.0004 + 0.0542$.

$$\boxed{w^{(c)} = 1,\; R = 0.12;\;\; w^{(s)} = 0.98,\; R \approx 0.0546 \Rightarrow \text{ERM picks spurious (lower pooled risk)}}$$

(2) Causal: $\nabla_w R_e = 2(w-1) = 0$ in both environments, so the penalty is $0$. Spurious: $\nabla R_1 = 0$ and $\nabla R_2 = 2(w+1) = 4$, so the penalty is $16$.

$$\boxed{\text{IRMv1}(z_c) = 0,\;\; \text{IRMv1}(z_s) = 16 \Rightarrow \text{IRM picks causal}}$$

(3) Using $w^{(c)} = 1$ and $w^{(s)} = 0.98$: $R_1^{(s)} = 0.0004$ and $R_2^{(s)} = 1.98^2 + 1.5 = 5.4204$, while the causal risk is always $0.12$.

$$\boxed{\text{(a) mixture: } 0.12 \text{ vs } 0.0546 \to \text{spurious};\;\; \text{(b) } R_2: 0.12 \text{ vs } 5.42 \to \text{causal};\;\; \text{(c) balanced: } 0.12 \text{ vs } 2.71 \to \text{causal}}$$

(4) $p(0.0004) + (1-p)(5.4204) = 0.12$, so $p^* = \frac{5.4204 - 0.12}{5.4204 - 0.0004} = \frac{5.3004}{5.42}$.

$$\boxed{p^* \approx 0.978}$$

For $p > p^*$ (the target looks like the training mixture) the spurious model wins. For $p < p^*$, which is almost any shift, the causal model wins. ERM is optimal only when the target matches the training mix.

### 6.3 SAM, SWA, and Soft-Label Sharpness

Theory: [[ATML_Lecture_5_2 - Domain Generalization#3.7 Approximating the Worst-Case Perturbation by Linearization|L5.2 §3.7]], [[ATML_Lecture_5_2 - Domain Generalization#3.8 Including Curvature Information|L5.2 §3.8]], [[ATML_Lecture_5_2 - Domain Generalization#3.10 Stochastic Weight Averaging and SWAD|L5.2 §3.10]], [[ATML_Lecture_5_2 - Domain Generalization#3.12 Refining Each Domain's Loss Landscape with Soft Labels|L5.2 §3.12]], [[ATML_Lecture_5_2 - Domain Generalization#3.13 Group Distributionally Robust Optimization|L5.2 §3.13]].

$$
\bar\epsilon = \rho\frac{g}{\lVert g\rVert_2}, \qquad \theta \leftarrow \theta - \eta\,\nabla L(\theta + \bar\epsilon), \qquad \text{max sharpness at a minimum} = \frac{\rho^2}{2}\lambda_{\max}(H), \qquad \bar\theta = \frac1K\sum_i\theta_i
$$

$$
S_i = L_{\text{sup}}(\theta + \epsilon_i) - L_{\text{sup}}(\theta), \qquad \text{gap} = \lvert S_i - S_j\rvert, \qquad \text{G-DRO} = \max_g \mathbb{E}_{P_g}[L]
$$

**Q6.7 — SAM perturbation and sharpness.** *[Notes L5.2-P3]* $g = (3, 4)$ and $\rho = 0.1$. Separately, a minimum has $\lambda_{\max} = 50$.

*Solution.* $\lVert g\rVert = 5$, so $\bar\epsilon = 0.1(0.6, 0.8)$. The sharpness is $\frac{0.01}{2}(50)$.

$$\boxed{\bar\epsilon = (0.06, 0.08),\;\; \text{sharpness} = 0.25}$$

**Q6.8 — One full SAM step.** *[New]* $L(\theta) = \theta_1^2 + 3\theta_2^2$ at $\theta = (1, 2)$, with $\rho = 0.1$ and $\eta = 0.01$. Take one SAM step. What is the maximum sharpness at the minimum?

*Solution.* $\nabla L = (2\theta_1, 6\theta_2) = (2, 12)$ and $\lVert g\rVert = \sqrt{148} = 12.166$, so $\bar\epsilon = 0.1(2, 12)/12.166 = (0.0164, 0.0986)$. The gradient at $\theta + \bar\epsilon = (1.0164, 2.0986)$ is $(2.0329, 12.592)$, so $\theta' = (1, 2) - 0.01(2.0329, 12.592)$. The Hessian is $\operatorname{diag}(2, 6)$, so $\lambda_{\max} = 6$ and the sharpness is $\frac{0.01}{2}(6)$.

$$\boxed{\bar\epsilon \approx (0.016, 0.099),\;\; \theta' \approx (0.980, 1.874),\;\; \text{max sharpness} = 0.03}$$

Note that $\lVert\bar\epsilon\rVert = \rho$ exactly, and that the step uses the gradient at the perturbed point, which costs two forward-backward passes.

**Q6.9 — SWA.** *[New]* The tail iterates are $(1.0, 2.0)$, $(1.2, 1.8)$, $(0.8, 2.2)$ and $(1.4, 2.0)$. Compute $\bar\theta$. When is averaging valid?

*Solution.* $\bar\theta = \frac14(4.4, 8.0)$.

$$\boxed{\bar\theta = (1.1, 2.0)}$$

Averaging is valid only if all the iterates lie in the same basin, which is why it is done at the tail of training. Averaging across two basins would land on the ridge between them.

**Q6.10 — Sharpness gap and the refinement loss.** *[New]* Domain 1: $L_{\text{sup}}$ goes from $0.30$ to $0.42$ under its perturbation. Domain 2: from $0.25$ to $0.55$. The soft label is $\tilde y = (0.9, 0.05, 0.05)$ for $y = (1, 0, 0)$, and $\lambda_1 = 1$. Compute $S_1$, $S_2$, the gap and $L_{\text{Refine}}$.

*Solution.* $S_1 = 0.12$ and $S_2 = 0.30$, so the gap is $0.18$. For the KL term, see the warning below: $D_{\mathrm{KL}}(y\Vert\tilde y) = 1 \cdot \ln(1/0.9) = 0.105$.

$$\boxed{S_1 = 0.12,\; S_2 = 0.30,\; \text{gap} = 0.18,\;\; L_{\text{Refine}} \approx 0.105 + 0.18 = 0.285}$$

> [!warning] Direction of the KL
> The notes write $\text{KL}(\tilde y\,\Vert\,y)$, but with a one-hot $y$ that divides by $y_k = 0$ and is infinite. Only $\text{KL}(y\Vert\tilde y) = -\ln\tilde y_{\text{true}}$ is finite, so compute that, and state which direction you used if this is asked.

**Q6.11 — ERM vs G-DRO.** *[New]* Three groups have losses $(0.1, 0.5, 0.3)$ and sizes $(70, 10, 20)$. Give the ERM objective and the G-DRO objective.

*Solution.* ERM: $0.7(0.1) + 0.1(0.5) + 0.2(0.3) = 0.07 + 0.05 + 0.06$. G-DRO: $\max = 0.5$.

$$\boxed{\text{ERM} = 0.18,\;\; \text{G-DRO} = 0.5 \text{ (group 2)}}$$

ERM barely notices the minority group (10%); G-DRO optimizes it directly. The catches are that over-parameterized models drive every group to 0, and that G-DRO needs group labels.

### 6.4 β-VAE and Disentanglement Metrics

Theory: [[ATML_Lecture_5_2 - Domain Generalization#4.10 β-VAE|L5.2 §4.10]], [[ATML_Lecture_5_2 - Domain Generalization#4.13 Verifying Disentanglement — the Z-Diff (β-VAE) Metric|L5.2 §4.13]], [[ATML_Lecture_5_2 - Domain Generalization#4.14 Verifying Disentanglement — Mutual Information Gap|L5.2 §4.14]], [[ATML_Lecture_5_2 - Domain Generalization#4.16 Capacity Control|L5.2 §4.16]].

$$
\text{penalty}_\beta = \beta D_{\mathrm{KL}}, \qquad \text{penalty}_C = \beta\lvert D_{\mathrm{KL}} - C\rvert, \qquad z_{\text{diff}} = \frac1L\sum_\ell\lvert z^{(1,\ell)} - z^{(2,\ell)}\rvert, \qquad \text{MIG} = \frac1K\sum_i\frac{I(v_i; z_*) - I(v_i; z_0)}{H(v_i)}
$$

**Q6.12 — β-VAE vs capacity control.** *[Notes L5.2-P4]* $\mu = (0, 0.5, 1)$, $\sigma = (1, 1, 0.5)$, $\beta = 4$ and $C = 0.5$.

*Solution.* Per-dimension KL: $0$, $0.125$ and $\frac12(1 + 0.25 - 1 + 1.386) = 0.818$, summing to $0.943$. Then $4(0.943)$ and $4\lvert0.943 - 0.5\rvert$.

$$\boxed{D_{\mathrm{KL}} \approx 0.943;\;\; \beta\text{-VAE} = 3.772,\;\; \text{capacity control} = 1.772}$$

Dimension 1 has collapsed (KL $= 0$).

**Q6.13 — MIG.** *[Notes L5.2-P5]* $v_1$: $H = 2$, $I_* = 1.6$, $I_0 = 0.3$. $v_2$: $H = 1.5$, $I_* = 1.2$, $I_0 = 0.9$.

*Solution.* $\text{MIG}_1 = 1.3/2 = 0.65$ and $\text{MIG}_2 = 0.3/1.5 = 0.20$.

$$\boxed{\text{MIG} = 0.425;\;\; v_1 \text{ is better disentangled}}$$

**Q6.14 — Z-diff.** *[New]* Factor $v_i$ is fixed, and two simulated pairs give $z^{(1)} = (0.2, 1.5, -0.3)$ with $z^{(2)} = (0.9, 1.4, 0.8)$, and $z^{(1)} = (1.0, -0.5, 0.4)$ with $z^{(2)} = (0.5, -0.55, -0.5)$. Which latent dimension encodes $v_i$?

*Solution.* The differences are $(0.7, 0.1, 1.1)$ and $(0.5, 0.05, 0.9)$, whose average is $(0.6, 0.075, 1.0)$.

$$\boxed{z_{\text{diff}} = (0.6, 0.075, 1.0) \Rightarrow \text{dimension 2 encodes } v_i}$$

The metric is the accuracy of a *linear* classifier that predicts $i$ from $z_{\text{diff}}$.

### 6.5 Self-Supervised Learning

Theory: [[ATML_Lecture_5_2 - Domain Generalization#5.9 Inferring Structure — Jigsaw Puzzles|L5.2 §5.9]], [[ATML_Lecture_5_2 - Domain Generalization#5.11 Transformation Prediction — Rotation|L5.2 §5.11]], [[ATML_Lecture_5_2 - Domain Generalization#5.15 Instance Classification|L5.2 §5.15]], [[ATML_Lecture_5_2 - Domain Generalization#5.17 Clustering|L5.2 §5.17]].

$$
\text{jigsaw: } 9! = 362{,}880 \to L \text{ classes}, \qquad \text{rotation: } \Omega = \{0°, 90°, 180°, 270°\}, \qquad z_i = \arg\min_j\lVert c_j - r_i\rVert
$$

**Q6.15 — Pretext scaling.** *[Notes L5.2-P6]* (a) Give the jigsaw reduction for a $3\times3$ grid restricted to $L = 100$ permutations. (b) Compare instance-classification head sizes for $10^5$ images with a 2048-d representation.

*Solution.* (a) $362{,}880/100$. (b) A CE head needs $2048 \times 10^5 = 2.05\times10^8$ weights; a contrastive projection needs about $2048\times128 \approx 2.6\times10^5$.

$$\boxed{\approx 3{,}629\times \text{ fewer classes};\;\; \approx 790\times \text{ fewer weights}}$$

**Q6.16 — Rotation loss and a clustering pseudo-label.** *[New]* (a) The rotation head outputs $(0.1, 0.7, 0.1, 0.1)$ for an image rotated by $90°$. (b) The centroids are $c_1 = (0,0)$, $c_2 = (4, 0)$ and $c_3 = (0, 3)$, and the representation is $r = (1, 2)$.

*Solution.* (a) $-\ln 0.7$. (b) The distances are $\sqrt5 = 2.236$, $\sqrt{13} = 3.606$ and $\sqrt2 = 1.414$.

$$\boxed{\mathcal{L}_{\text{rot}} \approx 0.357;\;\; z = 3}$$

**Short answers (L5.2).**
- *What are hard positives?* Same-class, cross-domain pairs that are far apart because of style. The loss gives them a huge pull, which collapses the domain sub-manifolds, and by the pull/push identity an equally huge push onto negatives that were already well placed. The fix is proxies.
- *Why is vanilla IRM intractable?* It is a bi-level MPEC. The per-domain argmin sets move with $\theta$, so their intersection is non-convex and may be empty.
- *Why can IRMv1 fail?* It uses only a first-order condition, which is satisfied by spurious stationary points such as scale collapse (Q6.5).
- *Why use SAM for DG?* A domain shift moves the loss surface, and a flat minimum stays low when the surface moves. SAM costs two passes per step.
- *Why can shared flatness fail?* Domains have inconsistent landscapes, so their gradients conflict. The remedies are SWA/SWAD, gradient alignment or masking, and soft labels.
- *What trade-off does β control in a β-VAE?* $\beta > 1$ gives more disentanglement but worse reconstruction and more collapsed dimensions. Capacity control targets a KL of $C$ instead of pushing it to zero.
- *Why is MIG better than raw MI?* A factor can be well encoded but spread across two dimensions. Only the gap to the runner-up reveals this (Q6.13, $v_2$).
- *What is an SSL shortcut?* A pretext task solvable from low-level cues (chromatic aberration, patch edges). Prevent it with gaps and jitter between patches, colour dropping, and similar tricks.

---

## 7. Lecture 6: Open-Set Recognition

### 7.1 Post-hoc Unknownness Scores

Theory: [[ATML_Lecture_6 - Open-Set Recognition#1.4 Unknownness Score and Rejection Threshold|L6 §1.4]], [[ATML_Lecture_6 - Open-Set Recognition#5.2 Maximum Softmax Probability (MSP)|L6 §5.2]], [[ATML_Lecture_6 - Open-Set Recognition#5.3 Maximum Logit Score (MLS)|L6 §5.3]], [[ATML_Lecture_6 - Open-Set Recognition#5.4 Energy Score|L6 §5.4]], [[ATML_Lecture_6 - Open-Set Recognition#5.5 Mahalanobis Distance|L6 §5.5]], [[ATML_Lecture_6 - Open-Set Recognition#5.6 How the Four Scores Relate|L6 §5.6]].

$$
u_{\text{MSP}} = 1 - \max_k p_k, \qquad u_{\text{MLS}} = -\max_k z_k, \qquad u_{\text{E}} = -\ln\sum_k e^{z_k}, \qquad u_{\text{MAH}} = \min_c (h - \mu_c)^\top\Sigma^{-1}(h - \mu_c)
$$

$$
u \le \tau \Rightarrow \text{accept}, \qquad u_{\text{MSP}} = 1 - e^{u_{\text{E}} - u_{\text{MLS}}}, \qquad u_{\text{MLS}} - \ln K \le u_{\text{E}} \le u_{\text{MLS}}
$$

**Q7.1 — All three logit scores on one input.** *[New]* $z = (2, 1, 0)$. Compute MSP, MLS and energy, and verify the identity.

*Solution.* The softmax is $(0.665, 0.245, 0.090)$, so $u_{\text{MSP}} = 0.335$. $u_{\text{MLS}} = -2$. $\ln(7.389 + 2.718 + 1) = \ln 11.107 = 2.408$, so $u_{\text{E}} = -2.408$. Check: $1 - e^{-2.408 + 2} = 1 - e^{-0.408} = 1 - 0.665$. Bound: $-2 - 1.099 \le -2.408 \le -2$.

$$\boxed{u_{\text{MSP}} \approx 0.335,\;\; u_{\text{MLS}} = -2,\;\; u_{\text{E}} \approx -2.408}$$

**Q7.2 — MSP vs MLS.** *[Notes L6-P1]* $z_A = (6, 5.9, 0)$ and $z_B = (6, 0, 0)$.

*Solution.* MLS is $-6$ for both. MSP: $403.43/769.47 = 0.524$ for A and $403.43/405.43 = 0.995$ for B.

$$\boxed{u_{\text{MSP}}(A) \approx 0.476,\;\; u_{\text{MSP}}(B) \approx 0.005;\;\; \text{only MSP separates them}}$$

**Q7.3 — Energy vs MSP.** *[Notes L6-P2]* $z_A = (6, 0, 0)$ and $z_B = (5.5, 5.5, 5.5)$.

*Solution.* $u_{\text{E}}(A) = -\ln 405.43 = -6.005$ and $u_{\text{E}}(B) = -(5.5 + \ln 3) = -6.599$. $u_{\text{MSP}}(A) = 0.005$ and $u_{\text{MSP}}(B) = 1 - 1/3$.

$$\boxed{u_{\text{E}}: A = -6.005,\; B = -6.599 \;(\text{B more known});\;\; u_{\text{MSP}}: A = 0.005,\; B = 0.667 \;(\text{B less known})}$$

**Q7.4 — Mahalanobis, diagonal covariance.** *[Notes L6-P3]* $\mu_{\text{dog}} = (0,0)$, $\mu_{\text{cat}} = (4, 0)$, $\Sigma = \operatorname{diag}(1, 4)$ and $\tau = 4$. Score $h_1 = (1, 3)$ and $h_2 = (0, 5)$.

*Solution.* The distance is $\Delta_1^2 + \Delta_2^2/4$. $h_1$: $3.25$ to dog and $11.25$ to cat. $h_2$: $6.25$ to dog and $22.25$ to cat.

$$\boxed{h_1: 3.25 \to \text{accept, dog};\;\; h_2: 6.25 \to \text{reject}}$$

**Q7.5 — Mahalanobis, full covariance.** *[New]* $\mu_1 = (0, 0)$, $\mu_2 = (3, 3)$, $\Sigma = \begin{bmatrix}2&1\\1&2\end{bmatrix}$, $h = (2, 0)$ and $\tau = 3$.

*Formulas needed.* $\Sigma^{-1} = \frac{1}{ad - bc}\begin{bmatrix}d & -b\\-c & a\end{bmatrix}$ and $x^\top A x = a x_1^2 + 2b x_1x_2 + d x_2^2$.

*Solution.* $\lvert\Sigma\rvert = 3$, so $\Sigma^{-1} = \frac13\begin{bmatrix}2&-1\\-1&2\end{bmatrix}$. For class 1, $\Delta = (2, 0)$: $\frac13(2 \cdot 4) = 2.667$. For class 2, $\Delta = (-1, -3)$: $\frac13(2 + 18 - 2(3)) = 14/3 = 4.667$.

$$\boxed{u_{\text{MAH}} = 2.667 \le 3 \Rightarrow \text{accept as class 1}}$$

### 7.2 PROSER and Reciprocal Points

Theory: [[ATML_Lecture_6 - Open-Set Recognition#5.8 Placeholder Learning — PROSER|L6 §5.8]], [[ATML_Lecture_6 - Open-Set Recognition#5.9 Geometry Reshaping — Reciprocal Points|L6 §5.9]].

$$
h_u = \lambda h_i + (1-\lambda)h_j,\;\lambda\sim\text{Beta}(2,2); \qquad L_{\text{cls}} = \ell_{\text{ce}}([z, z_u], y_j) + \beta\,\ell_{\text{ce}}([z_{\setminus y_j}, z_u], K{+}1); \qquad L_{\text{RPL}} = \ell_{\text{ce}} + \lambda(d - R)^2,\; d = \lVert h - P_j\rVert^2
$$

**Q7.6 — PROSER mixup and loss.** *[Notes L6-P6]* $h_{\text{cat}} = (2, 0, 1)$, $h_{\text{truck}} = (0, 4, 1)$ and $\lambda = 0.45$. For the loss, $z = (3, 1, 0)$ with true class 1, $z_u = 2$ and $\beta = 1$.

*Solution.* $h_u = (0.9, 2.2, 1.0)$. Beta(2,2) has mean $0.5$ and variance $\frac{4}{16\cdot5} = 0.05$. Term 1: $-3 + \ln(e^3 + e^1 + e^0 + e^2) = -3 + \ln 31.19 = 0.440$. Term 2 (true class masked): $-2 + \ln(e^1 + e^0 + e^2) = -2 + \ln 11.11 = 0.408$.

$$\boxed{h_u = (0.9, 2.2, 1.0);\;\; L_{\text{cls}} \approx 0.440 + 0.408 = 0.848}$$

**Q7.7 — Data-placeholder loss.** *[New]* A mixup placeholder has known logits $(1.5, 1.4, 0.2)$ and dummy logit $z_u = 1.0$. Compute its CE toward class $K+1$ (the "fuller picture" data-placeholder term).

*Solution.* $-1.0 + \ln(e^{1.5} + e^{1.4} + e^{0.2} + e^{1.0}) = -1 + \ln(4.482 + 4.055 + 1.221 + 2.718) = -1 + \ln 12.476$.

$$\boxed{\approx 1.524 \text{ nats}}$$

The loss is large because the between-class point still scores high on cat and truck; training raises $z_u$ there.

**Q7.8 — RPL loss.** *[Notes L6-P7]* $h = (3, 4)$, $P_j = (0, 0)$, $R = 20$, $\lambda = 0.1$ and CE $= 0.2$.

*Solution.* $d = 25$, so $0.1(25 - 20)^2 = 2.5$.

$$\boxed{L_{\text{RPL}} = 2.7;\;\; d > R, \text{ so } h \text{ is pulled back toward the shell}}$$

**Q7.9 — RPL classification and rejection.** *[New]* The reciprocal points are $P_1 = (0,0)$ and $P_2 = (4, 0)$. Classify $h = (1, 3)$, and give $u = -\max_k d_k$.

*Solution.* $d_1 = 1 + 9 = 10$ and $d_2 = 9 + 9 = 18$. The distances act as logits, so the larger evidence wins: class 2, with $\operatorname{softmax}(10, 18) \approx (0.0003, 0.9997)$.

$$\boxed{\text{class 2};\;\; u = -18}$$

An unknown near the centre, where all reciprocal points cluster, has small $d$ for every class and therefore a high $u$, so it is rejected.

### 7.3 Open-Space Risk and the LRT

Theory: [[ATML_Lecture_6 - Open-Set Recognition#4.2 The Open-Space Risk Objective|L6 §4.2]], [[ATML_Lecture_6 - Open-Set Recognition#4.3 The Optimal Detector — the Likelihood-Ratio Test|L6 §4.3]].

$$
R_O = \frac{\int_O f\,dx}{\int_{S_O} f\,dx}, \qquad u^*(x) = \ln\frac{P_U(x)}{P_K(x)}, \qquad \ln\mathcal N(x;\mu,\sigma^2) = -\frac{(x-\mu)^2}{2\sigma^2} - \ln\sigma - \tfrac12\ln 2\pi
$$

**Q7.10 — Open-space risk in 1-D.** *[Notes L6-P8]* The supported region is $[-2, 6]$. The classifier accepts $[-4, 8]$, then a tighter one accepts $[-2.5, 6.5]$.

*Solution.* Open part over accepted part: $4/12$, then $1/9$.

$$\boxed{R_O = 0.333 \to 0.111}$$

**Q7.11 — LRT with equal variances.** *[Notes L6-P9]* $P_K = \mathcal N(0,1)$, $P_U = \mathcal N(3,1)$ and $\tau = 0$.

*Solution.* $u^* = \frac12\big(x^2 - (x-3)^2\big) = 3x - 4.5$.

$$\boxed{\text{reject iff } x > 1.5;\;\; x = 1 \text{ accepted},\; x = 2 \text{ rejected}}$$

**Q7.12 — LRT with unequal variances.** *[New]* $P_K = \mathcal N(0, 1)$, $P_U = \mathcal N(0, 4)$ and $\tau = 0$.

*Solution.* $u^* = \big[-\frac{x^2}{8} - \ln 2\big] - \big[-\frac{x^2}{2}\big] = \frac38x^2 - \ln 2$. Reject when $x^2 > \frac83\ln 2 = 1.848$.

$$\boxed{\text{reject iff } \lvert x\rvert > 1.36}$$

The unknowns are more spread out, so the rejection region is both tails: two thresholds, not one.

### 7.4 OSR Metrics

Theory: [[ATML_Lecture_6 - Open-Set Recognition#7.3 Closed-Set Accuracy (CSA)|L6 §7.3]], [[ATML_Lecture_6 - Open-Set Recognition#7.4 AUROC|L6 §7.4]], [[ATML_Lecture_6 - Open-Set Recognition#7.5 TPR, FPR, FRR and FPR@95TPR|L6 §7.5]], [[ATML_Lecture_6 - Open-Set Recognition#7.6 AUROC versus FPR@95TPR, and OSCR|L6 §7.6]].

$$
\text{CSA} = \frac{\#\text{correct knowns}}{\#\text{knowns}}, \qquad \text{AUROC} = \frac{\#\{u(\text{unk}) > u(\text{known})\} + \tfrac12\#\text{ties}}{N_K N_U}, \qquad \text{TPR} = \frac{\#\text{knowns with } u \le \tau}{N_K}, \qquad \text{FPR} = \frac{\#\text{unknowns with } u\le\tau}{N_U}, \qquad \text{FRR} = 1 - \text{TPR}
$$

FPR@95TPR: take the smallest $\tau$ that accepts at least 95% of knowns and report the FPR there. OSCR: sweep $\tau$ upward; for each unknown admitted (an FPR step of $1/N_U$), add $\frac{1}{N_U}\times$ the CCR reached just before it, where CCR is the number of knowns accepted *and* correct divided by $N_K$.

**Q7.13 — The lecture's running example.** *[Notes L6 §7 + P5]* Knowns score $(0.1\,✓, 0.2\,✓, 0.25\,✗, 0.35\,✓, 0.5\,✓)$ and unknowns $(0.3, 0.45, 0.6, 0.8, 0.9)$.

*Solution.* CSA $= 4/5$. AUROC: $(5 + 5 + 5 + 4 + 3)/25$. At $\tau = 0.5$: all knowns are accepted, and unknowns $0.3$ and $0.45$ are too. OSCR: the CCR before each unknown is $0.4, 0.6, 0.8, 0.8, 0.8$, so $0.2 \times 3.4$.

$$\boxed{\text{CSA} = 0.8,\;\; \text{AUROC} = 0.88,\;\; \tau{=}0.5\!: \text{TPR} = 1,\; \text{FPR} = 0.4,\; \text{FRR} = 0;\;\; \text{FPR@95TPR} = 0.4,\;\; \text{OSCR} = 0.68}$$

**Q7.14 — Small AUROC and thresholds.** *[Notes L6-P4]* Knowns score $(0.2, 0.4, 0.5, 0.7)$ and unknowns $(0.3, 0.6, 0.8)$.

*Solution.* The favourable pairs are $3 + 2 + 2 + 1 = 8$ out of $12$. At $\tau = 0.5$: 3 of 4 knowns and 1 of 3 unknowns are accepted. At $\tau = 0.7$: 4 of 4 knowns and 2 of 3 unknowns.

$$\boxed{\text{AUROC} = 0.667;\;\; \tau{=}0.5\!: (0.75, 0.33, 0.25);\;\; \tau{=}0.7\!: (1, 0.67, 0);\;\; \text{FPR@95TPR} = 0.67}$$

**Q7.15 — Full metric set with a tie.** *[New]* Knowns score $(0.1\,✓, 0.2\,✓, 0.3\,✗, 0.4\,✓, 0.6\,✓, 0.7\,✗)$ and unknowns $(0.3, 0.5, 0.8, 0.9)$. Compute CSA, AUROC, TPR/FPR/FRR at $\tau = 0.4$, FPR@95TPR and OSCR.

*Solution.* CSA $= 4/6$. AUROC: for each known, count the unknowns above it, with ties as $\tfrac12$. That gives $0.1 \to 4$, $0.2 \to 4$, $0.3 \to 3 + \tfrac12$, $0.4 \to 3$, $0.6 \to 2$ and $0.7 \to 2$, a total of $18.5$ out of $6 \times 4 = 24$. At $\tau = 0.4$: knowns $\le 0.4$ number 4 of 6, and unknowns $\le 0.4$ number 1 ($0.3$) of 4. For 95% TPR with 6 knowns, all must be accepted, so $\tau = 0.7$, where unknowns $0.3$ and $0.5$ pass. OSCR: the correct knowns below each unknown in turn are $\{0.1, 0.2\}$ below $0.3$, adding $0.4$ below $0.5$, and adding $0.6$ below $0.8$ and below $0.9$. That gives CCRs of $2/6, 3/6, 4/6, 4/6$, and the OSCR is $\frac14 \cdot \frac{2+3+4+4}{6}$.

$$\boxed{\text{CSA} = 0.667,\;\; \text{AUROC} \approx 0.771,\;\; \tau{=}0.4\!: \text{TPR} = 0.667,\; \text{FPR} = 0.25,\; \text{FRR} = 0.333,\;\; \text{FPR@95TPR} = 0.5,\;\; \text{OSCR} \approx 0.542}$$

OSCR sits below CSA because it also pays for the unknowns it accepts.

**Short answers (L6).**
- *Why can't a closed-set softmax say "none of these"?* The probabilities are forced to sum to 1 over the known classes, so every input is assigned somewhere, often with high confidence.
- *When does MSP fail, and when does MLS fail?* MSP ignores the magnitude of the logits, so a uniformly large vector looks uncertain even when it is strongly activated. MLS reads only the top logit, so it cannot see a near-tie (Q7.2). Energy blends the two, $u_{\text{E}} \in [u_{\text{MLS}} - \ln K,\, u_{\text{MLS}}]$.
- *What are the three closed-set failure mechanisms?* CE says nothing about unknowns; decision regions (cones) are unbounded, so far-away inputs get extreme confidence; and the features are generic and discriminative, so unknowns reuse known features. Add shortcuts and spurious features.
- *What are the two design principles?* Bound the known regions (low open-space risk), and approximate the LRT, which is optimal but needs $P_U$.
- *AUROC vs FPR@95TPR?* AUROC measures the overall ranking. FPR@95TPR measures one deployment operating point. A few confident unknowns can leave AUROC high but FPR@95TPR poor.
- *Why does PROSER mask the true logit?* So the dummy class learns to be the runner-up everywhere. It becomes an input-adaptive threshold that the true class beats for knowns and that nothing beats for unknowns.
- *What does GCSC claim?* A better closed-set classifier (higher CSA) gives better OSR with plain MLS; the two are strongly correlated.

---

## Exam-Day Formula Index

| Need | Formula | Where |
|---|---|---|
| std / variance | $\sigma^2 = \overline{x^2} - \bar x^2$ ($1/N$), $s^2$ uses $1/(N-1)$ | §0 |
| sigmoid derivative | $\sigma' = \sigma(1-\sigma)$; $\frac{d}{da}[-\ln\sigma(a)] = \sigma(a)-1$ | §0 |
| softmax-CE gradient | $\partial\mathcal{L}/\partial z_k = p_k - y_k$ | §0 |
| $2\times2$ inverse / eigen | $A^{-1} = \frac{1}{ad-bc}[\,d\;{-b};\,{-c}\;a\,]$; $\lambda = \frac{\text{tr}\pm\sqrt{\text{tr}^2 - 4\det}}{2}$ | §0 |
| spectral norm | $\sigma_{\max}$; diag $\to\max\lvert d_i\rvert$; symmetric $\to\max\lvert\lambda\rvert$ | §0, Q1.2 |
| weighted-quadratic minimizer | $\frac{\alpha b + \beta d}{\alpha + \beta}$ | §0, Q6.5 |
| two-way softmax | $\frac{e^a}{e^a + e^b} = \sigma(a-b)$ | §0, Q5.9 |
| CNN size | $\lfloor(H + 2P - F)/S\rfloor + 1$ | Q4.6 |
| Beta moments | mean $\frac{a}{a+b}$, var $\frac{ab}{(a+b)^2(a+b+1)}$ | §0, Q7.6 |
| Gaussian log-density | $-\frac{(x-\mu)^2}{2\sigma^2} - \ln\sigma - 0.919$ | §0, Q7.12 |
| AUROC ties | count $\tfrac12$ | Q7.15 |

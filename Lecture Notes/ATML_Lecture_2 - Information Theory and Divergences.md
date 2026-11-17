---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module I - Notes.pdf]]'
- '[[ATML Lec 2.pdf]]'
tags:
- course/cs6304
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_1 - Representation and Generalization Theory]] · [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] →
Concepts: [[Optimal Transport]], [[f-Divergence]]
%% nav:end %%

# ATML — Lecture 2: Information Theory, Divergences, and Optimal Transport

## Where This Lecture Sits

*Continues directly from [[ATML_Lecture_1 - Representation and Generalization Theory]], which closed by naming three measurement families — information-theoretic, geometric, group-theoretic — without supplying any of them. This lecture builds the first two.*

The arc is one question: given two distributions, how far apart are they? Sections 5–7 give three answers — compare densities by their ratio, through a test function, or by the work of moving one into the other — each failing where the next succeeds. Later lectures use these rather than re-derive them: forward KL (Kullback–Leibler) becomes maximum likelihood and the VAE (variational autoencoder) objective in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], Jensen–Shannon the GAN (generative adversarial network) objective in [[ATML_Lecture_4 - GANs, and Inductive Biases]], and MMD (maximum mean discrepancy) and Wasserstein the alignment losses of [[ATML_Lecture_5_1 - Domain Adaptation]] and [[ATML_Lecture_5_2 - Domain Generalization]].

## Topic Inventory

- 1. Random Variables, Expectation, and Monte Carlo Estimation
  - 1.1 Random Variables and Expectation
  - 1.2 The Monte Carlo Approximation
- 2. Covariance Structure and Independent Latent Representations
  - 2.1 The Identity Covariance and Independent Dimensions
  - 2.2 Correlated Features and the Case for Independent Latents
  - 2.3 The Multivariate Gaussian — Shape and Properties
- 3. Information Theory
  - 3.1 From the Real World to Data
  - 3.2 Information Content
  - 3.3 Entropy as Average Information Content
  - 3.4 Entropy at the Extremes, and the Maximum-Entropy Gaussian
  - 3.5 Mutual Information
  - 3.6 Entropy Reduction as the Basis of Loss Functions
- 4. Loss Functions in Deep Representation Learning
  - 4.1 The Two Families of Losses
  - 4.2 Comparing a Predicted Distribution to a Reference Distribution
- 5. f-Divergences
  - 5.1 The f-Divergence Framework
  - 5.2 Forward KL Divergence
  - 5.3 Forward KL Is Mode-Covering (Support-Seeking)
  - 5.4 Reverse KL Divergence Is Mode-Seeking
  - 5.5 Jensen–Shannon Divergence
- 6. Integral Probability Metrics
  - 6.1 The IPM Framework
  - 6.2 Maximum Mean Discrepancy
  - 6.3 Total Variation Distance
- 7. Optimal Transport
  - 7.1 Why OT? The Failure of KL on Disjoint Supports
  - 7.2 Formal Setup: Cost Matrix and Transport Plan
  - 7.3 The Optimal Transport Linear Program and the Wasserstein Distance
  - 7.4 Worked Example: Transporting $P$ into $Q$
  - 7.5 Entropic Regularization, Sinkhorn Distance, and the Sinkhorn Algorithm
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Random Variables, Expectation, and Monte Carlo Estimation

Lecture 1 Section 8 defined the mean as an integral against a density; Lecture 1 Section 4 established that the density is never available. This section resolves the conflict by replacing the integral with an average over the samples at hand — the reason a loss computed on a mini-batch is a legitimate stand-in for a quantity defined over an entire distribution.

### 1.1 Random Variables and Expectation

A quantity $X$ is a **random variable** when it takes values according to some underlying probability distribution, written $X \sim P(X)$. A recurring modeling choice in this course is to assume a $D$-dimensional random variable follows a multivariate standard normal,

$$
P(X) \sim \mathcal{N}(\mathbf{0}_D, \, I_{D \times D}),
$$

where $\mathbf{0}_D \in \mathbb{R}^{D}$ is the zero mean vector and $I_{D\times D} \in \mathbb{R}^{D \times D}$ is the identity covariance matrix — the meaning of this specific choice of covariance is developed in Section 2.

The **expectation** of a random variable is defined, in the continuous case, as an integral weighted by the density, and in the discrete case as a sum weighted by the probability mass function:

$$
\mathbb{E}[X] = \int x\, P(x)\, dx \qquad \text{(continuous)}, \qquad \mathbb{E}[X] = \sum_{x_i} x_i\, P(x_i) \qquad \text{(discrete)}.
$$

This definition generalizes immediately to the expectation of *any* function of $X$, not just $X$ itself: for an arbitrary function "anything" (or, more formally, $g(x)$),

$$
\mathbb{E}[\text{anything}] = \int \text{anything} \cdot P(x)\, dx, \qquad \mathbb{E}_{P(x)}[g(x)] = \int g(x) P(x)\, dx.
$$

### 1.2 The Monte Carlo Approximation

In practice, this expectation is almost never computable in closed form, because the true data distribution $P(X)$ is itself **unknown** — for example, there is no accessible analytic expression for "the distribution of cat images." The standard workaround is to **take samples** from the distribution (e.g., a dataset of cat images) and approximate the true, unreachable **statistical average** with an empirical **sample average**:

$$
\mathbb{E}_{P(x)}[g(x)] = \int g(x) P(x)\, dx \;\;\approx\;\; \frac{1}{N} \sum_{i=1}^{N} g(x_i),
$$

where the left-hand integral is the *statistical average* (over the true, unknown population distribution) and the right-hand sum is the *sample average* (computed from $N$ observed data points). This substitution — replacing an intractable population expectation with a tractable empirical mean over a finite sample — is the basic **Monte Carlo estimation** principle that underlies virtually every loss function and gradient estimate used in deep learning, since every training loss computed over a mini-batch is exactly this kind of sample-average stand-in for a true, unobservable expectation.

*Intuition.* The estimator works by the law of large numbers, converging at rate $1/\sqrt{N}$ — the same $1/\sqrt{N}$ appearing in the generalization bound of Lecture 1 Section 5.1, and for the same reason: both quantify how fast a finite sample stops misrepresenting the population it came from. Two consequences recur later. The estimate is unbiased but noisy, and that noise is not purely a nuisance — it is what makes SGD (stochastic gradient descent) explore rather than descend directly into the nearest minimum, the mechanism the flatness-based methods of [[ATML_Lecture_5_2 - Domain Generalization]] exploit. More importantly, the substitution is legitimate only when the samples come from the distribution the expectation is written over; violating this is precisely what domain shift does, and it is the subject of Module II.

---

## 2. Covariance Structure and Independent Latent Representations

Section 1 introduced $\mathcal{N}(\mathbf{0}, I)$ without justifying the $I$; that justification is the content of this short section. The identity covariance is not a convenience but the matrix statement of the **disentanglement** property from Lecture 1 Section 3, and the prior every VAE in this course is trained against.

### 2.1 The Identity Covariance and Independent Dimensions

Consider a $3$-dimensional identity covariance matrix,

$$
I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}.
$$

The diagonal entries being $1$ means each dimension has unit variance, and — critically — the off-diagonal entries being $0$ means every pair of dimensions has **zero covariance**. The underlying intuition is that each **dimension contains information that does not come from (i.e. cannot be predicted by) any other dimension** — the dimensions are **independent random variables**. For jointly Gaussian variables in particular, zero covariance is equivalent to full statistical independence, which is why the identity covariance matrix is the natural way to express "these latent dimensions carry independent information."

### 2.2 Correlated Features and the Case for Independent Latents

Contrast this with a covariance matrix that has non-zero off-diagonal terms:

$$
\begin{bmatrix} 1 & 0.5 & 0.1 \\ 0.5 & 1 & 0.3 \\ 0.1 & 0.3 & 1 \end{bmatrix}
$$

(the notes reuse the symbol $I$ here loosely to mean "the covariance matrix," even though it is no longer literally the identity). Here, every feature is still a dimension of the latent space, but the non-zero off-diagonal entries mean these dimensions are **correlated**: different latent dimensions end up **learning similar stuff**, i.e. redundant information. This redundancy is undesirable, because it wastes representational capacity — two correlated latent dimensions carry less combined information than two independent ones of the same individual variance. It is also precisely the opposite of the **disentanglement** property listed in the representation theory of Lecture 1.

This is why, when designing an encoder that maps data into a $d$-dimensional latent code $z = (z_1, z_2, \ldots, z_d)$, the desired target distribution for the latent space is

$$
\boxed{\begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_d \end{bmatrix} \sim \mathcal{N}(\mathbf{0}, I)}
$$

i.e. a standard multivariate Gaussian with **identity covariance** — encouraging each latent dimension to be independent of the others and hence non-redundant. This same prior reappears as the VAE prior $P_\theta(z) \sim \mathcal{N}(0, I)$ in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]].

*Intuition.* With a budget of $d = 64$ latent dimensions, two perfectly correlated dimensions leave the model working with 63 while paying for 64; partial correlation is a partial version of the same waste. What is wasted is representational volume, since a correlated Gaussian concentrates along a thin ellipsoid instead of filling a sphere. A second reason matters specifically for generation: if the encoder maps data onto a diagonal ridge but sampling draws $z \sim \mathcal{N}(0,I)$, most sampled codes land in regions the decoder never saw and decode to nonsense. Forcing the aggregate posterior toward $\mathcal{N}(0,I)$ is what makes "sample from the prior and decode" valid, which is why this target reappears as a *constraint* in the VAE objective.

Two cautions. Zero covariance implies independence only for jointly Gaussian variables; in general it excludes linear but not nonlinear dependence, making identity covariance necessary rather than sufficient for disentanglement — which is why [[ATML_Lecture_5_2 - Domain Generalization]] requires dedicated metrics rather than an inspection of the covariance matrix. And independence is not interpretability: a rotation of an isotropic Gaussian remains isotropic, so identity covariance alone cannot identify which axes are meaningful. Breaking that degeneracy is the difficulty $\beta$-VAE addresses.

### 2.3 The Multivariate Gaussian — Shape and Properties

*(Assigned in lecture for self-study, with emphasis on the shape of the distribution and its properties. The professor did not develop it on the board; the material below fills it in.)*

Sections 2.1 and 2.2 compared two covariance matrices by inspecting their entries. That comparison becomes precise once one sees what a covariance matrix does to the *shape* of the distribution it parameterizes, which is the purpose of this subsection. The multivariate Gaussian recurs throughout the course — as the VAE prior and posterior in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], as the noise source of the reparameterization trick, and as the maximum-entropy distribution of Section 3.4 — so its geometry is worth having exactly.

**Density.** A random vector $x \in \mathbb{R}^d$ with mean $\mu \in \mathbb{R}^d$ and symmetric positive-definite covariance $\Sigma \in \mathbb{R}^{d\times d}$ is **multivariate Gaussian**, $x \sim \mathcal{N}(\mu, \Sigma)$, when

$$
\boxed{p(x) = \frac{1}{(2\pi)^{d/2}\,\lvert\Sigma\rvert^{1/2}}\exp\!\left(-\tfrac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right)}
$$

where $\lvert\Sigma\rvert$ is the determinant. Every dependence on $x$ passes through the quadratic form in the exponent, whose square root is the **Mahalanobis distance**

$$
\boxed{d_M(x, \mu) = \sqrt{(x-\mu)^{\top}\Sigma^{-1}(x-\mu)}}
$$

The density is therefore a decreasing function of Mahalanobis distance alone, and its level sets $\{x : d_M(x,\mu) = c\}$ are the surfaces of constant probability. For $\Sigma = I$ the Mahalanobis distance reduces to Euclidean distance; in general it measures distance in units of the distribution's own spread along each direction, so a displacement along a high-variance direction counts for less than the same Euclidean displacement along a low-variance one.

**Shape from the eigendecomposition.** Since $\Sigma$ is symmetric positive-definite, it factors as

$$
\Sigma = U\Lambda U^{\top}, \qquad U = [\,u_1 \;\cdots\; u_d\,] \text{ orthonormal}, \qquad \Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d),\; \lambda_i > 0.
$$

Substituting $\Sigma^{-1} = U\Lambda^{-1}U^{\top}$ and writing $y = U^{\top}(x - \mu)$ for the coordinates of $x - \mu$ in the eigenbasis, the quadratic form separates into a sum of squares,

$$
(x-\mu)^{\top}\Sigma^{-1}(x-\mu) = \sum_{i=1}^{d}\frac{y_i^2}{\lambda_i},
$$

which is the equation of an ellipsoid. Hence the shape is read off directly: the level sets are **ellipsoids centred at $\mu$**, their **principal axes point along the eigenvectors $u_i$**, and the **semi-axis length along $u_i$ is $c\sqrt{\lambda_i}$** at Mahalanobis radius $c$. The eigenvectors fix the orientation, the eigenvalues fix the elongation, and the mean fixes only the location. Three cases cover the matrices of Sections 2.1–2.2:

| Covariance | Level sets | Interpretation |
|---|---|---|
| $\sigma^2 I$ (isotropic) | spheres of radius $c\sigma$ | all directions equivalent; rotation-invariant |
| $\operatorname{diag}(\sigma_1^2, \ldots, \sigma_d^2)$ | ellipsoids aligned with the coordinate axes | dimensions independent but unequally spread |
| full $\Sigma$, non-zero off-diagonals | ellipsoids rotated off the axes | dimensions correlated; the tilt *is* the correlation |

In two dimensions with variances $\sigma_1^2, \sigma_2^2$ and correlation coefficient $\rho = \sigma_{12}/(\sigma_1\sigma_2)$, the ellipse tilts toward the diagonal $x_1 = x_2$ when $\rho > 0$ and toward $x_1 = -x_2$ when $\rho < 0$, and it flattens toward a line segment as $\lvert\rho\rvert \to 1$, at which point $\Sigma$ becomes singular and the distribution collapses onto a lower-dimensional subspace. This is the geometric content of Section 2.2's warning: a correlated latent is a Gaussian flattened toward fewer effective dimensions than it nominally has.

The eigenvalues also fix the distribution's **volume**. Since $\lvert\Sigma\rvert = \prod_i \lambda_i$, the volume of any fixed-$c$ ellipsoid is proportional to $\sqrt{\lvert\Sigma\rvert}$, so the determinant measures total spread, and a near-zero eigenvalue — a nearly flat direction — drives it toward zero however large the other eigenvalues are.

**Properties.** Six properties are used repeatedly in this course.

1. **Closure under affine maps.** If $x \sim \mathcal{N}(\mu, \Sigma)$ and $y = Ax + b$, then $y \sim \mathcal{N}(A\mu + b,\; A\Sigma A^{\top})$. This follows from linearity of expectation for the mean and from $\mathbb{E}[(y - \mathbb{E}y)(y-\mathbb{E}y)^{\top}] = A\,\mathbb{E}[(x-\mu)(x-\mu)^{\top}]A^{\top}$ for the covariance. Two consequences follow immediately. Taking $x = \epsilon \sim \mathcal{N}(0, I)$ and $A = L$ with $LL^{\top} = \Sigma$ (for example the Cholesky factor, or $U\Lambda^{1/2}$) gives $\mu + L\epsilon \sim \mathcal{N}(\mu, \Sigma)$, which is exactly how any Gaussian is sampled and is the multivariate form of the **reparameterization trick** of Lecture 3 Section 5.5. Conversely, $\Sigma^{-1/2}(x - \mu) \sim \mathcal{N}(0, I)$, the **whitening** transform, which maps any Gaussian ellipsoid back to the unit sphere. Geometrically, every Gaussian is a standard Gaussian that has been stretched by $\Lambda^{1/2}$, rotated by $U$, and translated by $\mu$.

2. **Marginals are Gaussian.** Partitioning $x = (x_a, x_b)$ with $\mu = (\mu_a, \mu_b)$ and $\Sigma = \begin{bmatrix}\Sigma_{aa} & \Sigma_{ab}\\ \Sigma_{ba} & \Sigma_{bb}\end{bmatrix}$, the marginal is $x_a \sim \mathcal{N}(\mu_a, \Sigma_{aa})$ — obtained simply by deleting the rows and columns of the other block, which is property 1 with $A$ a selection matrix.

3. **Conditionals are Gaussian.**$$
x_a \mid x_b \sim \mathcal{N}\!\left(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b),\;\; \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\right).
$$The conditional mean shifts linearly with the observed $x_b$, and the conditional covariance (the Schur complement) is never larger than the marginal $\Sigma_{aa}$ and does not depend on the observed value. Observing a correlated variable always reduces uncertainty, and reduces it by the same amount whatever is observed — which is the Gaussian case of the entropy reduction of Section 3.6.

4. **Uncorrelated implies independent.** If $\Sigma$ is diagonal, the quadratic form is a sum of per-coordinate terms, the exponential factorizes, and $p(x) = \prod_i p(x_i)$. This is the property Section 2.1 relied on, and it holds for jointly Gaussian vectors only; outside the Gaussian family zero covariance does not imply independence.

5. **Entropy is fixed by the determinant.**
$$
\boxed{H\big(\mathcal{N}(\mu,\Sigma)\big) = \tfrac{1}{2}\ln\!\big((2\pi e)^d\,\lvert\Sigma\rvert\big) \text{ nats}}
$$
independent of $\mu$. Entropy grows with the log-volume of the ellipsoid, so among Gaussians with fixed per-dimension variances the diagonal (independent) one has the largest determinant — by Hadamard's inequality $\lvert\Sigma\rvert \le \prod_i \Sigma_{ii}$ — and therefore the largest entropy. Correlation is lost entropy: a quantitative restatement of Section 2.2's claim that correlated dimensions waste capacity. Together with the maximum-entropy characterization of Section 3.4, $\mathcal{N}(0, I)$ is the highest-entropy distribution with unit variance per dimension.

6. **KL divergence has a closed form.** Between two Gaussians in $d$ dimensions,
$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu_0, \Sigma_0)\,\Vert\,\mathcal{N}(\mu_1, \Sigma_1)\big) = \tfrac{1}{2}\!\left[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^{\top}\Sigma_1^{-1}(\mu_1-\mu_0) - d + \ln\frac{\lvert\Sigma_1\rvert}{\lvert\Sigma_0\rvert}\right],
$$
which for a diagonal $\Sigma_0 = \operatorname{diag}(\sigma_j^2)$ against the standard prior $\mathcal{N}(0, I)$ reduces to $\tfrac{1}{2}\sum_j\big(\mu_j^2 + \sigma_j^2 - 1 - \ln\sigma_j^2\big)$. This is the prior-matching term of the VAE objective in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] and of $\beta$-VAE in [[ATML_Lecture_5_2 - Domain Generalization]], and its closed form is the reason Gaussian posteriors are used there: the term is computed exactly rather than estimated by sampling.

**Shape in high dimensions.** One property of the shape runs against low-dimensional intuition and matters whenever one samples from $\mathcal{N}(0, I_d)$ with large $d$, as in generating from a VAE latent. The density is maximal at the origin, yet almost no samples lie near it. Since $\lVert z\rVert^2 = \sum_{i=1}^{d} z_i^2$ is a sum of $d$ independent squared standard normals, it follows a $\chi^2_d$ distribution with mean $d$ and variance $2d$, so $\lVert z\rVert$ concentrates tightly around $\sqrt{d}$ with a standard deviation approaching $1/\sqrt{2}$ independently of $d$. The mass of a high-dimensional standard Gaussian therefore lies in a **thin shell of radius $\approx\sqrt{d}$**, not in a ball around the mode: the density falls as one moves outward, but the volume available at radius $r$ grows as $r^{d-1}$, and the two balance at the shell. For $d = 512$ nearly every sample has norm within a few units of $\sqrt{512} \approx 22.6$. Two practical consequences follow. The origin, though the single most probable point, is an atypical sample that a decoder may never have seen during training; and linearly interpolating between two latent samples cuts through the interior of the shell, producing codes with atypically small norm, which is why spherical interpolation is often preferred for traversing a Gaussian latent space.

*Intuition.* The whole subsection reduces to one picture: a Gaussian is a unit sphere that has been stretched along some axes, rotated, and moved. The covariance matrix records the stretching and rotation, its eigendecomposition undoes them, and every property above follows from the fact that stretching, rotating, and moving a Gaussian produces another Gaussian. Section 2.2's preference for $\mathcal{N}(0, I)$ is then a preference for the unstretched, unrotated sphere — the shape with no redundant directions, no preferred directions, and maximal entropy per unit variance.

---

## 3. Information Theory

Sections 1 and 2 dealt with moments, which are a coarse summary — two very different distributions can share a mean and covariance exactly. Information theory supplies the finer vocabulary, the one Lecture 1 Section 3 promised when it named sufficiency and minimality as information-theoretic properties. The three quantities built here state precisely what a representation *knows*, and Section 3.6 then claims every loss function is at bottom an entropy-reduction device — a claim cashed out in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], where cross-entropy and InfoNCE (the contrastive loss built on noise-contrastive estimation) are both derived from mutual information.

### 3.1 From the Real World to Data

The chain that connects reality to a trainable model is short and worth stating explicitly: the **real world** is passed through **data acquisition** to produce **data** $\{X, Y\}$. Everything downstream — every distribution, loss, and bound — is defined relative to that acquired dataset, not to the world itself, which is why the histogram-is-not-the-density warning from Lecture 1 matters so much.

### 3.2 Information Content

Let $x$ be a discrete random variable with probability mass function $P(x)$, and suppose we observe the specific outcome $x = x_k$. The **information content** of that observation is defined as

$$
\boxed{I(x = x_k) \triangleq -\log_2 P(x = x_k) \quad \text{[bits]}}
$$

This is a **measure of "surprise."** The formalization captures the intuition that **highly unlikely events convey more information**: an outcome far from the mean, or more generally an outcome with low probability, tells you more precisely because it is surprising. As $P(x=x_k) \to 0$ the information content grows without bound, and as $P(x=x_k) \to 1$ (a near-certain event) it approaches $0$.

*Intuition for the logarithm.* Two requirements determine the functional form almost completely: the measure must decrease in probability, and it must be additive over independent events, so that two unrelated coin flips deliver twice the information of one. Since independence multiplies probabilities, additivity requires a function turning products into sums, and the logarithm is the only continuous such function; the minus sign follows because $\log P \le 0$.

The base fixes only the units. Base $2$ gives bits, with the operational reading that $-\log_2 P(x_k)$ is the length of the codeword an optimal compressor assigns to outcome $x_k$ — probability $1/8$ costs three bits, probability $1/2$ costs one. This coding-length reading is also the cleanest route to cross-entropy: $-\sum P \log Q$ is the average codeword length paid when data drawn from $P$ is compressed with a code designed for $Q$, minimized exactly at $Q = P$, and the excess over that minimum is the KL divergence of Section 5.2.

### 3.3 Entropy as Average Information Content

**Entropy** is the **average information content** over the whole distribution — the expectation of the information content, weighted by the probability of each outcome:

$$
\boxed{H(x) = -\sum_{x} P(x) \log_2 P(x) \quad \text{[bits]}}
$$

Entropy measures the **irreducible uncertainty** in the variable: the part of the randomness that no amount of cleverness can remove, because it is a property of the distribution itself.

### 3.4 Entropy at the Extremes, and the Maximum-Entropy Gaussian

Two limiting cases illustrate what entropy measures. First, a **uniform distribution** over an interval $[a,b]$, with constant density $P(x) = \tfrac{1}{b-a}$, has the **highest** entropy — every outcome is equally likely, so there is no way to predict the next draw better than chance, which corresponds to maximum randomness.

At the opposite extreme, a distribution concentrated entirely on a single point $b$ — a spike of height $1$ at $x=b$ and $0$ everywhere else — has the **lowest** (zero) entropy. Because the outcome is certain in advance, there is no randomness at all, and correspondingly no information is gained by observing it.

A third, more subtle result is flagged as a note in the professor's material:

> **If we fix the mean and the variance, then among all distributions with that mean and variance, the Gaussian has maximum entropy.**

This is the **maximum-entropy characterization of the Gaussian**, and it is the deeper reason why Gaussian priors and Gaussian noise models appear everywhere in this course: assuming a Gaussian is the *least committal* assumption one can make once the first two moments are fixed.

### 3.5 Mutual Information

Let $X \sim P(X)$, $Y \sim P(Y)$, and jointly $(X,Y) \sim P(X,Y)$. The **mutual information** between them is

$$
\boxed{I(X;Y) = \mathbb{E}_{P(X,Y)}\!\left[\log \frac{P(X,Y)}{P(X)P(Y)}\right]}
$$

Interpretation: **how much does knowing one variable reduce the uncertainty about the other?** The ratio inside the logarithm compares the true joint against the joint that *would* hold if $X$ and $Y$ were independent; when they are independent the ratio is $1$ everywhere and $I(X;Y) = 0$. Mutual information is the primary "information-theoretic term" referred to in the representation theory of Lecture 1, and it reappears as the objective behind both the cross-entropy loss ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], discriminative models) and the InfoNCE loss ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], contrastive models).

*Intuition.* The equivalent form $I(X;Y) = H(Y) - H(Y\mid X)$ is the one to retain: uncertainty about $Y$ before observing $X$, minus uncertainty after — literally the reduction in surprise that observing $X$ buys. The expression is unchanged under swapping its arguments, so $X$ says exactly as much about $Y$ as $Y$ does about $X$, however asymmetric their causal relation.

Two properties make it the natural yardstick here. It vanishes exactly under independence and is strictly positive otherwise, detecting *any* statistical dependence rather than only the linear dependence covariance sees — the gap flagged at the end of Section 2.2. And it is invariant under invertible reparameterization, so it measures information content rather than the coordinate system that content is written in, which is what makes it meaningful to say $z$ retains the information in $x$ about $y$ without first fixing what the axes of $z$ mean.

Lecture 1's first two properties now have exact statements: **sufficiency** is $I(z;Y)$ as large as possible, ideally equal to $I(X;Y)$; **minimality** is $I(z;X)$ as small as possible subject to that. The pair constitutes the **information bottleneck**, and since both are maximized by $z = x$, the constraint does all the work. Each of the three DRL families in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] is a tractable surrogate for one of these terms — necessarily a surrogate, because mutual information between high-dimensional continuous variables cannot itself be computed.

### 3.6 Entropy Reduction as the Basis of Loss Functions

The general claim connecting information theory to model training is that **loss functions are fundamentally based on reducing entropy**: the input to a model starts out with relatively high entropy (high uncertainty about the correct output), and this entropy **decreases as the data moves through the model's layers**, ideally collapsing to near-zero entropy (high certainty) on the correct label by the time it reaches the output. In particular, the cross-entropy loss $\mathcal{L}_{\text{CE}}$ introduced in Lecture 1 arises from **bounds on mutual information** between the model's representation and the true label — minimizing cross-entropy tightens a bound that connects the informativeness of the learned representation to the achievable predictive accuracy. This connection is made explicit in Lecture 3.

For the Gaussian case this reduction is exact and easy to see: by property 3 of Section 2.3, conditioning on a correlated variable shrinks the covariance to its Schur complement, and by property 5 the entropy falls with the log-determinant of that shrunken covariance.

---

## 4. Loss Functions in Deep Representation Learning

Section 3 argued that training is entropy reduction; the practical follow-up is what a loss function requires in order to perform it. The answer splits in two, and the split is the hinge of the lecture: some losses need a **reference distribution**, and are therefore unusable until a way of measuring distance between distributions exists — which is what Sections 5–7 supply — while others constrain the output or the latent geometry directly and need no reference at all.

### 4.1 The Two Families of Losses

Loss functions in DRL shape a model's **behavior, geometry, and representation** in one of two ways, and the professor's notes present them as a strict two-way split:

**(1) Distribution-referenced losses.** These require an explicit **reference distribution** $P(x)$ against which the **model's assumed distribution** $Q(x)$ is compared. The goal is simply to **match $Q$ with $P$**, sketched as two overlapping density curves being pushed together. There are **many ways to compare $P(x)$ and $Q(x)$** — the whole of Sections 5–7 is an enumeration of them.

**(2) Behavior / geometry / decision losses.** These **directly constrain the model output without needing any reference distribution.** Two sub-families are given:

- **(a) Reconstruction losses** — compare a reconstructed output to the original input directly:
  - Mean squared error, $\text{MSE}(x, \hat{x})$.
  - **Perceptual loss**, $\lVert \phi(x) - \phi(\hat{x}) \rVert_2^2$, where $\phi(\cdot)$ is a fixed feature extractor: the comparison is made in a learned feature space rather than in raw pixel space, so that perceptually similar images are counted as similar even when their pixels differ.

- **(b) Geometry-in-latent losses** — shape the arrangement of points in latent space directly:
  - **Contrastive** losses.
  - **InfoNCE**.
  - **Triplet** loss, and others.

The two families map onto the model families of Lecture 3: generative models lean on family (1) and on reconstruction; contrastive models are exactly family (2b).

![[03 - Attachments/ATML_L2_loss_families.png|600]]

*The professor's split: distribution-referenced losses match a model $Q(x)$ to a reference $P(x)$; behaviour/geometry losses constrain the output directly (reconstruction, or latent geometry via contrastive/InfoNCE/triplet).*

### 4.2 Comparing a Predicted Distribution to a Reference Distribution

For a $K$-class classification problem, the **softmax** function (introduced in Lecture 1) produces a categorical distribution over the $K$ classes, conditioned on the input: $P(Y \mid X)$. For brevity, this predicted distribution is renamed $Q$:

$$
P(Y \mid X) \;\coloneqq\; Q.
$$

Alongside this predicted distribution, we also have a **one-hot encoded ground-truth distribution** for the true class $j$ — a distribution placing all of its probability mass on class $j$ (a single spike at position $j$, zero everywhere else). This reference distribution is called $P$. The notes emphasize that this reference "$P$" is a stand-in: **the reference distribution can in general be anything**, not only a one-hot vector — a one-hot ground truth is simply the most common special case in classification. In every case, however, the **goal is to make $Q$ and $P$ as similar as possible**.

To make the subsequent divergence measures concrete, the lecture works with a running toy example over $3$ discrete states, $i \in \{1,2,3\}$:

$$
P(x): \quad P_1 = 0.5,\; P_2 = 0.3,\; P_3 = 0.2, \qquad\qquad Q(x): \quad Q_1 = 0.4,\; Q_2 = 0.4,\; Q_3 = 0.2.
$$

To compare $P$ and $Q$ meaningfully, we first need a way to **quantify how different two distributions are**, and then **minimize** that quantity during training.

---

## 5. f-Divergences

Section 4.2 ended by asking how to quantify the difference between two distributions. This section gives the first and most familiar of three answers, whose defining commitment is **pointwise comparison of densities**: at each point form the ratio $P(x)/Q(x)$, score its departure from $1$, and average. Different choices of the scoring function $f$ yield forward KL, reverse KL, Jensen–Shannon, and the rest — and all inherit the commitment together with its blind spot.

That blind spot should be named in advance, since Sections 6 and 7 exist to repair it: a pointwise ratio knows nothing of the geometry of the space its points inhabit. It registers that $P$ and $Q$ disagree at $x$ but not whether the nearest agreement is adjacent or distant, because locations are mere indices to it. Section 7.1 exhibits the failure in its starkest form.

### 5.1 The f-Divergence Framework

The framing question for the whole f-divergence family is: *if the true world is $P$ but we use $Q$, how wrong are the likelihood ratios $P/Q$ on average?* This gives the general definition

$$
\boxed{D_f(P \,\|\, Q) = \mathbb{E}\!\left[f\!\left(\frac{P}{Q}\right)\right]}
$$

for some function $f$. Every specific divergence below is this same expression with a particular choice of $f$; all of them are built on the **ratio of the two distributions**, $\tfrac{P}{Q}$, evaluated at every state.

### 5.2 Forward KL Divergence

The (forward) **Kullback–Leibler divergence** takes $f(t) = t\log t$, giving

$$
\boxed{D_{\mathrm{KL}}(P \,\|\, Q) = \mathbb{E}_{P(x)}\!\left[\log \frac{P(x)}{Q(x)}\right] = \sum_i P_i \log\!\left(\frac{P_i}{Q_i}\right)}
$$

Its two limiting behaviors are stated directly:

- If $Q \approx P$, then $D_{\mathrm{KL}} \to 0$.
- If $Q \ll P$ (i.e. $Q$ assigns far less mass than $P$ somewhere), then $D_{\mathrm{KL}} \to$ very large.

Applying this to the toy example, the elementwise difference and ratio are

$$
P - Q = (0.1,\, -0.1,\, 0), \qquad \frac{P}{Q} = (1.25,\, 0.75,\, 1),
$$

and substituting into the formula (using the natural logarithm, i.e. working in **nats**),

$$
D_{\mathrm{KL}}(P\|Q) = 0.5\ln(1.25) + 0.3\ln(0.75) + 0.2\ln(1) \approx 0.0253 \text{ nats.}
$$

> **[Unclear in source]** — the lecture notes record this value as $0.02567$; carrying out the computation with the stated $P$ and $Q$ gives $\approx 0.0253$ nats. The discrepancy is small and consistent with a minor arithmetic slip; both the method and the stated toy distributions are reproduced faithfully.

An important qualitative remark from the lecture: the **exact numerical value of a divergence is not as important as its behavior** — what matters for training dynamics is *how* a divergence responds to changes in $Q$ (its shape, its gradients, which errors it penalizes most), not the precise number it evaluates to for any one snapshot of $P$ and $Q$.

### 5.3 Forward KL Is Mode-Covering (Support-Seeking)

Rewriting the forward KL makes its behavior transparent:

$$
\mathrm{KL}(P\|Q) = \mathbb{E}_{P(x)}\!\left[\log P(x) - \log Q(x)\right] = \underbrace{\text{const.}}_{-H(P)} - \;\mathbb{E}_P\!\left[\log Q(x)\right].
$$

Since the first term does not depend on $Q$, minimizing forward KL is exactly **maximizing $\mathbb{E}_P[\log Q(x)]$** — which is maximum likelihood. Now consider a region $R$ where $P > 0$ but $Q \approx 0$. On $R$, $\log Q \to -\infty$, and the averaging is done with respect to $P$, which is *non-zero* there — so $\mathbb{E}_P[\log Q]$ blows up and the loss goes to infinity.

Therefore **forward KL heavily punishes holes in $Q$ on $P$'s support**. Minimizing it forces $Q$ to spread out until it **covers the support of $P$**. This behavior is called **mode-covering**, or equivalently **support-of-$P$ seeking** (also **zero-avoiding**, since $Q$ avoids assigning zero probability anywhere $P$ is positive).

> **[Possible source inconsistency]** — the in-class board notes labelled this behavior "support or mode-seeking." The professor's typed module notes and the standard literature both classify forward KL as **mode-covering / support-covering**, and reserve **mode-seeking** for reverse KL (Section 5.4). The module-notes terminology is used throughout this document.

![[03 - Attachments/ATML_L2_forward_kl_sketch.png|404]]

*Forward KL: $Q$ must spread over the whole region $R$ where $P > 0$, or $\mathbb{E}_P[\log Q]$ blows up.*

### 5.4 Reverse KL Divergence Is Mode-Seeking

Swapping the roles of $P$ and $Q$ gives the **reverse KL divergence**:

$$
\boxed{D_{\mathrm{KL}}(Q \,\|\, P) = \mathbb{E}_{Q(x)}\!\left[\log \frac{Q(x)}{P(x)}\right] = \mathbb{E}_Q\!\left[\log Q - \log P\right] = \sum_i Q_i \log\!\left(\frac{Q_i}{P_i}\right)}
$$

Evaluating this on the toy example,

$$
D_{\mathrm{KL}}(Q\|P) = 0.4\ln\!\left(\tfrac{0.4}{0.5}\right) + 0.4\ln\!\left(\tfrac{0.4}{0.3}\right) + 0.2\ln\!\left(\tfrac{0.2}{0.2}\right) \approx 0.0258 \text{ nats,}
$$

which matches the value given in the notes. Crucially,

$$
D_{\mathrm{KL}}(Q\|P) \neq D_{\mathrm{KL}}(P\|Q),
$$

confirming explicitly that **KL divergence is not symmetric** — swapping the two arguments generally changes the value and, more importantly, the *behavior* of the resulting loss.

The behavioral analysis mirrors Section 5.3 with the roles reversed. If $Q > 0$ but $P \approx 0$ somewhere, then $\log P \to -\infty$ and we are averaging with respect to $Q > 0$ — so the loss blows up. Therefore **reverse KL punishes extra mass in $Q$ where $P$ is tiny**. As a result, minimizing reverse KL makes $Q$ conservative about spreading its mass: $Q$ concentrates tightly around a single **dominant mode** of $P$ rather than attempting to cover all of them. This is **mode-seeking** (equivalently **zero-forcing**) behavior, and it is why reverse KL is associated with sharp but potentially mode-dropping models (a connection made concrete in the GAN discussion in [[ATML_Lecture_4 - GANs, and Inductive Biases]]).

*Intuition: one rule generates both behaviours.* The asymmetry is not two facts but one — **the distribution in the first argument is the one averaged over, and a penalty is incurred only where that distribution puts mass.**

In forward KL the average is over $P$, so $Q$ is graded only where $P$ is supported and its failing is *absence*: being small where $P$ is large drives $\log(P/Q)$ toward $+\infty$ at a point of real weight, while excess mass where $P$ vanishes costs nothing, those points carrying zero weight. The minimizer therefore hedges, stretching to cover every mode; a single Gaussian fitted to a bimodal $P$ straddles both modes and deposits substantial mass in the empty valley between them. In reverse KL the average is over $Q$, which thereby selects the points at which it is graded, and its failing is *presence*: mass where $P$ is near zero is penalized, while ignoring a mode of $P$ entirely is free. The minimizer retreats onto one mode and commits.

*Why this matters beyond this lecture.* The asymmetry is the root of a dichotomy recurring throughout the course.

| | Forward KL $D(P \,\Vert\, Q)$ | Reverse KL $D(Q \,\Vert\, P)$ |
|---|---|---|
| Expectation taken over | $P$ (the data) | $Q$ (the model) |
| Punishes | $Q$ too small where $P>0$ | $Q$ too large where $P\approx 0$ |
| Behaviour | mode-covering, zero-avoiding | mode-seeking, zero-forcing |
| Equivalent to | maximum likelihood (Section 5.3) | variational inference |
| Typical artifact | blurry, averaged samples | sharp samples, dropped modes |
| Shows up as | VAE / MLE training ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]]) | GAN mode collapse ([[ATML_Lecture_4 - GANs, and Inductive Biases]]) |

The last row is the payoff. The characteristic blurriness of VAE samples ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]]) and the characteristic sharpness-with-mode-collapse of GANs ([[ATML_Lecture_4 - GANs, and Inductive Biases]]) are not unrelated empirical quirks of two architectures but the two halves of this table observed in practice: a VAE maximizes likelihood, hence forward KL, hence averages over modes it cannot separately represent, and averaging images yields blur; a GAN generator is pushed toward a reverse-KL-like objective rewarding commitment to whatever region convinces the discriminator, which yields sharpness together with the temptation to abandon modes that are hard to fake.

A practical corollary: always establish which argument is the data and which the model before quoting a KL value. The two numbers above, $0.0253$ and $0.0258$ nats, agree closely only because $P$ and $Q$ are close here; for well-separated distributions they can differ by orders of magnitude, and one may be infinite while the other is finite.

![[03 - Attachments/ATML_L2_reverse_kl_sketch.png|360]]

*Reverse KL: $Q$ is punished wherever it puts mass on $R$ where $P \approx 0$, so it collapses onto a mode of $P$.*

### 5.5 Jensen–Shannon Divergence

A symmetrized alternative that avoids choosing a direction is the **Jensen–Shannon divergence**, defined by first forming the mixture $M$ of the two distributions and then averaging the KL divergence of each against it:

$$
\boxed{\mathrm{JS}(P\|Q) = \tfrac{1}{2}\mathrm{KL}(P\|M) + \tfrac{1}{2}\mathrm{KL}(Q\|M), \qquad M = \frac{P+Q}{2}}
$$

Because $M$ has support wherever *either* $P$ or $Q$ does, neither KL term can blow up to infinity, so JS divergence stays finite even when $P$ and $Q$ have partially disjoint supports. It is also symmetric by construction. JS divergence is exactly what the original GAN objective implicitly minimizes (Lecture 4).

---

## 6. Integral Probability Metrics

Every $f$-divergence requires the ratio $P(x)/Q(x)$, which is a serious practical obstacle: evaluating a ratio requires evaluating both densities, yet in every setting of interest we possess samples and not densities — no function returns "the probability of this cat image" — and the ratio is in any case undefined wherever the denominator vanishes.

Integral probability metrics evade both problems by changing the question. Rather than asking how the densities differ at each point, they ask whether any **test function** can distinguish samples of $P$ from samples of $Q$. A test function is evaluated on samples and averaged, so no density is written down and nothing is divided. The price is that an IPM is only as discriminating as the permitted family $\mathcal{F}$: too small a family declares distinct distributions identical, too large a family renders the maximization intractable. Choosing $\mathcal{F}$ is therefore the whole design problem, and the two choices recurring in this course are a kernel ball (MMD, below) and the $1$-Lipschitz functions (the Wasserstein distance of Section 7).

### 6.1 The IPM Framework

An **Integral Probability Metric (IPM)** takes a completely different view from the f-divergences: rather than comparing densities pointwise via their ratio, it says that **$P$ and $Q$ are close if no test function $f$ in a chosen family $\mathcal{F}$ can tell them apart.** The divergence is then the maximum discrepancy achievable by any admissible test function. This "adversarial witness" formulation is what makes IPMs well-behaved even when densities cannot be evaluated — only samples are needed.

### 6.2 Maximum Mean Discrepancy

The **Maximum Mean Discrepancy (MMD)** is the IPM obtained by measuring the **maximum difference in the mean of $P$ and $Q$ (in feature space)**:

$$
\boxed{\text{MMD}(P,Q) = \max_{f} \left| \mathbb{E}_{P(x)}[f(x)] - \mathbb{E}_{Q(x)}[f(x)] \right|}
$$

When $\mathcal{F}$ is the unit ball of a reproducing kernel Hilbert space, this maximum has a closed form in terms of the kernel, and for the discrete case treated in the lecture it reduces to the quadratic form
$$
\text{MMD}^2 = (P - Q)^{\top} K (P - Q),
$$

where $K$ is a **kernel matrix** — for example a Radial Basis Function (RBF) kernel of the kind used in Support Vector Machines — whose entries measure similarity between states in a (possibly infinite-dimensional) feature space, rather than comparing $P$ and $Q$ directly in their original space.

*Intuition.* Take the crudest admissible test function, $f(x) = x$. The induced IPM is $|\mathbb{E}_P[x] - \mathbb{E}_Q[x]|$, the gap between means — a genuine measure of difference, but useless alone, since distributions sharing a mean score zero however differently shaped. Admitting $f(x) = x^2$ additionally detects second moments, $x^3$ detects skew, and so on: enlarging $\mathcal{F}$ monotonically sharpens the test.

The kernel takes this to its limit without enumerating functions. A kernel corresponds implicitly to a feature map $\phi$ with $k(x,x') = \langle\phi(x),\phi(x')\rangle$, infinite-dimensional in the RBF case and thereby encoding moments of every order. With $\mathcal{F}$ the unit ball of the associated Hilbert space the maximization admits a closed form, the optimal witness being proportional to $\mathbb{E}_P[\phi(x)] - \mathbb{E}_Q[\phi(x)]$, so that

$$
\text{MMD}^2(P,Q) = \left\lVert \mathbb{E}_{P}[\phi(x)] - \mathbb{E}_{Q}[\phi(x)] \right\rVert_{\mathcal{H}}^2 .
$$

MMD is thus the distance between the means of $P$ and $Q$ after both are lifted into a space rich enough that matching means there implies matching everything. A kernel with this property is called *characteristic*; the RBF kernel is one, which is why $\text{MMD}(P,Q) = 0$ if and only if $P = Q$ despite only a single vector per distribution ever being compared. Expanding the norm gives the form used in practice,

$$
\text{MMD}^2 = \mathbb{E}_{x,x'\sim P}[k(x,x')] + \mathbb{E}_{y,y'\sim Q}[k(y,y')] - 2\,\mathbb{E}_{x\sim P,\,y\sim Q}[k(x,y)],
$$

that is, average self-similarity within $P$, plus average self-similarity within $Q$, less twice the average cross-similarity — small exactly when points of the two samples are on average as similar to each other as to their own kind. Nothing is evaluated but kernel values on samples, which is the escape from the $f$-divergence requirement motivating this section, and the reason MMD serves as the alignment loss of [[ATML_Lecture_5_1 - Domain Adaptation]]. One caution: the RBF bandwidth sets the length scale at which differences register, so a poorly chosen bandwidth yields a near-constant loss with vanishing gradients, and implementations typically sum several kernels across a range of bandwidths.

### 6.3 Total Variation Distance

The **Total Variation Distance (TVD)** is the **absolute difference of probability values**, halved:

$$
\boxed{\text{TVD}(P,Q) = \frac{1}{2}\int \left| P(x) - Q(x) \right| dx \;\;=\;\; \frac{1}{2}\sum_i |P_i - Q_i| \text{ (discrete)}}
$$

Unlike KL divergence, TVD is symmetric in $P$ and $Q$ and remains finite even when the two distributions have mismatched supports. Its weakness is the flip side of its simplicity: it registers only *how much* mass differs, not *where* — a shortcoming that optimal transport addresses directly.

---

## 7. Optimal Transport

Sections 5 and 6 compare distributions vertically: they stand at a location and compare the two heights there, whether directly through a ratio or after a feature lift. Neither asks how far apart two locations are, which Section 6.3 already registered as TVD's weakness — it sees how much mass differs, never where. Optimal transport is the third answer, and the one taking the geometry of the space seriously, measuring the distance between $P$ and $Q$ as the minimum work required to carry one mass distribution into the shape of the other.

The reframing carries a cost and a payoff. The cost is computational: where an $f$-divergence is a sum over $n$ bins and MMD a sum over $n^2$ kernel entries, optimal transport requires solving an optimization problem over $n^2$ plan entries under $2n$ constraints at every evaluation, which is what Section 7.5's entropic regularization exists to make tractable. The payoff is that OT alone varies continuously with the separation of two non-overlapping distributions, and so remains usable as a gradient signal in exactly the regime where the others break.

### 7.1 Why OT? The Failure of KL on Disjoint Supports

The motivating question for **optimal-transport-based metrics** is: *how much effort is needed to move probability mass from $P$ to match $Q$, per unit distance?*

Two properties are desirable in any distance metric:

- **Symmetry** — $d(P,Q) = d(Q,P)$.
- **Triangle inequality** — $d(P,R) \le d(P,Q) + d(Q,R)$.

**Many of the divergences above do not satisfy these.** In particular, $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$, and $D_{\mathrm{KL}} = \infty$ whenever the support overlap is not equal — because the ratio $P/Q$ involves a **division by zero**.

The pathological consequence is illustrated with two box distributions: place $P$ and $Q$ as adjacent non-overlapping boxes, then move them slightly further apart. **Intuitively $P$ and $Q$ in the first configuration seem closer than in the second, but $D_{\mathrm{KL}} = \infty$ in both** — KL cannot distinguish "just barely disjoint" from "very far apart." Optimal transport gives an alternative notion of distance that **does not have these issues**, because it measures *ground distance travelled*, not density ratios.

Two important measures fall under OT:

1. **Wasserstein Distance (WD)**
2. **Sinkhorn Distance (SD)**

The lecture describes OT itself in enthusiastic terms — *"the best thing humans have discovered,"* *"very amazing."*

*Intuition.* Optimal transport is classically the **earth mover's distance**, and the name carries the idea: $P$ is a pile of sand, $Q$ the profile that sand must be made to form, a transport plan is a work order specifying how much to carry between each pair of locations, and the cost matrix prices each route by distance. Two piles are close when little sand must travel far.

This resolves the two-box pathology directly. Separating the boxes by $2\varepsilon$ instead of $\varepsilon$ doubles the distance every grain is carried and so doubles the cost — a finite, monotone response to a smooth change in configuration — whereas the $f$-divergences report $\infty$ in both cases, never having asked how far the mass must go.

The consequence concerns gradients rather than aesthetics. A loss equal to $\infty$ throughout a region has zero gradient there, so a model whose output does not overlap the data receives no signal indicating which way to move. This is not a corner case: at initialization a generator's output and the data distribution are essentially always disjoint in high dimensions, both concentrating on thin manifolds (Lecture 1 Section 6.2) that generically fail to intersect. Supplying a usable gradient in that regime is the entire argument for the Wasserstein GAN, and in Module II for a Wasserstein-style discrepancy between source and target domains whose features have not yet begun to overlap.

The connection back to Section 6 makes OT less of an outlier than it appears. By Kantorovich–Rubinstein duality, the $1$-Wasserstein distance has the equivalent dual form

$$
W_1(P,Q) = \sup_{\lVert f\rVert_{L}\le 1} \left(\mathbb{E}_P[f(x)] - \mathbb{E}_Q[f(x)]\right),
$$

which is the IPM template of Section 6.1 with $\mathcal{F}$ the $1$-Lipschitz functions. OT is therefore simultaneously a mass-moving problem and an integral probability metric, the dual's Lipschitz constraint being the counterpart of the primal's travel cost: limiting how fast the witness may change is what forces it to respect distance. The weight-clipping and gradient-penalty devices of Wasserstein GANs are attempts to enforce $\lVert f\rVert_L \le 1$ on a network.

![[03 - Attachments/ATML_L2_kl_disjoint_support.png|600]]

*Disjoint supports: $P$ and $Q$ look close, yet $D_{\text{KL}} = \infty$ because of division by zero.*

### 7.2 Formal Setup: Cost Matrix and Transport Plan

Write the two discrete distributions as weighted sums of Dirac deltas supported on the same $n$ points $x_1, \ldots, x_n$:

$$
P(x) = \sum_{i=1}^{n} P_i\, \delta(x - x_i), \qquad\qquad Q(x) = \sum_{i=1}^{n} Q_i\, \delta(x - x_i).
$$

Define $C$ as the **cost matrix**, where

$$
C_{ij} = \text{cost of moving one unit of mass from bin } i \text{ to bin } j.
$$

A natural choice is the squared Euclidean distance between the bin locations,

$$
C_{ij} = \lVert x_i - x_j \rVert_2^2,
$$

which is **symmetric**. Other choices are equally valid — for instance $C_{ij} = |i-j|$, the absolute distance. In either case the intuition is the same: **the larger the distance between bins $i$ and $j$, the higher the cost** of moving probability mass between them.

In this framing, $P(x)$ is **true and fixed**, while $Q(x)$ is the distribution obtained when **mass is redistributed according to a transport plan $T$**. The entry $T_{ij}$ is the **mass moved from $x_i$ to $x_j$**, and $C_{ij}$ is the **cost of that movement**, so the **total cost** is the sum of costs weighted by mass moved along each route:

$$
\text{Total Cost} = \langle T, C\rangle = \sum_{i=1}^{n}\sum_{j=1}^{n} T_{ij}\, C_{ij},
$$

the elementwise (Frobenius) inner product of the two matrices.

A valid plan must satisfy the **marginal constraints** — its row sums must recover $P$ and its column sums must recover $Q$:

$$
T\mathbf{1} = \sum_j T_{ij} = P_i \;\; \forall i, \qquad\qquad T^{\top}\mathbf{1} = \sum_i T_{ij} = q_j \;\; \forall j,
$$

with $\underline{P} = [P_1, \ldots, P_n]^\top$ and $\underline{q} = [q_1, \ldots, q_n]^\top$.

![[03 - Attachments/ATML_L2_transport_plan_grid.png|300]]

*The transport plan as an $n \times n$ grid: rows index where mass leaves $P$, columns where it arrives in $Q$.*

### 7.3 The Optimal Transport Linear Program and the Wasserstein Distance

We need to find an **optimal plan $T^{*}$ minimizing the total cost**, subject to the marginal constraints and non-negativity:

$$
\boxed{
\begin{aligned}
&\min_{T} \;\; \langle T, C\rangle \\
&\text{s.t.} \quad T\mathbf{1} = \underline{P}, \quad T^{\top}\mathbf{1} = \underline{q}, \quad T_{ij} \ge 0
\end{aligned}
}
$$

This is a **linear program**, whose solution is denoted $T_w^{*}$. The **Wasserstein distance** is then the square root of the optimal cost:

$$
\boxed{\text{WD}(P,Q) = \sqrt{\langle T_w^{*}, C\rangle}}
$$

The linear program is described as **very hard** to solve exactly in general (its size grows with the size of the support), and in practice it is **solved with specialized solvers** rather than in closed form. When the ground cost is based on an $\ell_1$ or $\ell_2$ norm, the resulting distances are denoted $WD_1$ and $WD_2$ respectively.

### 7.4 Worked Example: Transporting $P$ into $Q$

Take the running example with support at positions $\{0, 1, 2\}$:

$$
\underline{P} = \begin{bmatrix}0.5\\0.3\\0.2\end{bmatrix}, \qquad \underline{q} = \begin{bmatrix}0.4\\0.4\\0.2\end{bmatrix}, \qquad C_{ij} = (\text{Pos}_i - \text{Pos}_j)^2 \;\Rightarrow\; C = \begin{bmatrix}0&1&4\\1&0&1\\4&1&0\end{bmatrix}.
$$

Conceptually, the procedure is: take the mass at bin $i=1$ and **split it into pieces**, distributing those pieces across the bins according to some plan; repeat for $i=2$ and $i=3$. One valid transport plan is

$$
T^{*} = \begin{bmatrix} 0.38 & 0.12 & 0 \\ 0.02 & 0.25 & 0.03 \\ 0 & 0.03 & 0.17 \end{bmatrix}.
$$

Verifying the marginals confirms this is admissible:

$$
T^{*}\mathbf{1} = \begin{bmatrix}0.5\\0.3\\0.2\end{bmatrix} = \underline{P} \quad \text{(row sums)}, \qquad\qquad T^{*\top}\mathbf{1} = \begin{bmatrix}0.4\\0.4\\0.2\end{bmatrix} = \underline{q} \quad \text{(column sums)}.
$$

**Interpretation of the diagonal entries:** mass on the diagonal **stays in the same bin** (cost $C_{ii} = 0$). Reading the plan row by row shows the splitting explicitly. Distributing the mass of bin $0$ (namely, $0.5$) according to the first row of $T^{*}$ gives $[0.38,\;0.12,\;0]$, which sums to $0.5$; the second row gives $[0.02,\;0.25,\;0.03]$, summing to $0.3$; the third gives $[0,\;0.03,\;0.17]$, summing to $0.2$. **Summing these three distributions column-wise recovers $Q(x) = [0.4,\;0.4,\;0.2]$ exactly.** Running the same construction in the reverse direction (reading the columns of $T^*$ instead) recovers $P(x) = [0.5,\;0.4,\;0.2]$ — the professor's notes work through both directions to make the symmetry of the plan concrete.

> **[Possible source inconsistency]** — in the reverse-direction (column) decomposition, the reconstructed distribution is labelled $P(x) = [0.5,\;0.4,\;0.2]$, whose entries sum to $1.1$ rather than $1$; the stated $\underline{P}$ on the same page is $[0.5,\;0.3,\;0.2]$. The middle entry appears to be a transcription slip.

The plan is also **not unique** — there can be **multiple valid plans** achieving the same marginals. A second, sparser example (with unit-distance ground cost $C_{ij} = |i-j|$) is

$$
T^{*} = \begin{bmatrix} 0.4 & 0.1 & 0 \\ 0 & 0.3 & 0 \\ 0 & 0 & 0.2 \end{bmatrix},
$$

where the **diagonal entries** ($0.4$, $0.3$, $0.2$) again represent mass staying put, and the single **off-diagonal entry** $T_{12} = 0.1$ represents mass moving from position $1$ to position $2$. For this plan,

$$
WD_1 = 0.1 \times |1-2| = 0.1, \qquad\qquad WD_2 = 0.1 \times (1-2)^2 = 0.1.
$$

![[03 - Attachments/ATML_L2_ot_mass_movement.png|622]]

*Reading $T^*$: each bin of $P$ is split along its row (diagonal entries stay put), and summing the pieces column-wise rebuilds $Q = [0.4, 0.4, 0.2]$; running it backwards recovers $P$.*

### 7.5 Entropic Regularization, Sinkhorn Distance, and the Sinkhorn Algorithm

Because the exact LP is expensive, the standard practical fix is to add an **entropic regularization** term to the objective, which makes the problem strictly convex and solvable by fast iterative scaling. The **entropic OT problem** is

$$
\boxed{
\begin{aligned}
&\min_{T \ge 0} \;\; \langle T, C\rangle - \varepsilon H(T), \qquad H(T) = -\sum_{ij} T_{ij}\log T_{ij}\\
&\text{s.t.} \quad T\mathbf{1} = \underline{P}, \quad T^{\top}\mathbf{1} = \underline{q}
\end{aligned}
}
$$

whose solution is denoted $T_s^{*}$, giving the **Sinkhorn distance**

$$
\boxed{\text{SD}(P,Q) = \sqrt{\langle T_s^{*}, C\rangle}}
$$

**Solving it.** Introducing Lagrange multipliers $\alpha_i$ for the row constraints and $\beta_j$ for the column constraints gives the Lagrangian

$$
\mathcal{L}(\alpha, \beta, T) = \sum_{ij} T_{ij}C_{ij} \;-\; \varepsilon \sum_{ij} T_{ij}\log T_{ij} \;+\; \sum_i \alpha_i\!\left(P_i - \sum_j T_{ij}\right) \;+\; \sum_j \beta_j\!\left(q_j - \sum_i T_{ij}\right).
$$

Setting $\partial\mathcal{L}/\partial T_{ij} = 0$ shows the optimal plan factorizes as a diagonal rescaling of the **Gibbs kernel** $K = e^{-C/\varepsilon}$, which yields the **Sinkhorn algorithm**:

1. Form the kernel $K = e^{-C/\varepsilon}$ (elementwise exponential).
2. Iterate the two scaling updates until convergence, starting from $\underline{u}^0 =\underline{v}^0 = \mathbf{1}$:
   $$
   \underline{u}^{t+1} = \frac{\underline{P}}{K\,\underline{v}^{t}}, \qquad\qquad \underline{v}^{t+1} = \frac{\underline{q}}{K^{\top}\underline{u}^{t+1}}
   $$
   (both divisions elementwise).
3. Form the plan $T^{*} = \operatorname{diag}(\underline{u})\, K \,\operatorname{diag}(\underline{v})$.

Each iteration alternately fixes the row marginals and then the column marginals; the entropic term guarantees the iteration converges to the unique optimum. The regularization strength $\varepsilon$ trades accuracy against speed: as $\varepsilon \to 0$ the Sinkhorn solution approaches the exact LP solution $T_w^*$ but converges more slowly, while larger $\varepsilon$ gives a smoother, more diffuse plan much faster.

*Intuition.* The exact problem is a linear program, and linear programs place their optima at vertices of the feasible polytope — here **sparse, all-or-nothing plans** in which each source bin ships along one or two routes and nothing along the rest. That sparsity is why the problem is combinatorially awkward, why the optimum can be non-unique as Section 7.4 observed, and why the cost fails to be a smooth function of $C$, which is fatal for backpropagation.

Adding $-\varepsilon H(T)$ rewards plans that spread mass out, since entropy is maximized by the uniform plan. The optimum moves into the interior of the polytope, becomes unique by strict convexity, and becomes smooth in the inputs. In transport terms the regularized plan hedges, sending most mass along cheap routes and a little along expensive ones, with $\varepsilon$ setting the degree: as $\varepsilon \to \infty$ cost is ignored and the plan degenerates to the independent coupling $P_i q_j$, while as $\varepsilon \to 0$ the sharp LP solution returns.

The form of the algorithm follows from one line of algebra. Setting the Lagrangian derivative to zero gives $T_{ij} = u_i K_{ij} v_j$ with $K = e^{-C/\varepsilon}$: **the optimal plan is always the fixed matrix $K$ with its rows and columns rescaled**, leaving two scaling vectors as the only unknowns rather than all $n^2$ entries of $T$. Since $K_{ij}$ is large for cheap routes and exponentially small for expensive ones, $K$ encodes a prior preference over routes and $u, v$ merely bend it to satisfy the marginals. The iteration is then immediate — rescale rows to sum to $P$, which breaks the columns; rescale columns to sum to $q$, which breaks the rows; repeat — and each half-step is a single matrix–vector product, differentiable end to end. That differentiability is why Sinkhorn rather than exact OT is the version appearing inside deep learning losses.

The lecture closes at this point, with the next session set to begin with **Vision Transformers** and the three families of DRL models.

---

## Looking Ahead

Later lectures reach for these instruments without re-deriving them, so it is worth recording which answers which question.

| Question | Instrument | Where it returns |
|---|---|---|
| How do I estimate an expectation I cannot compute? | Monte Carlo sample average (§1.2) | every loss and gradient in the course |
| What should a latent covariance look like? | $\mathcal{N}(\mathbf{0}, I)$ (§2.2) | the VAE prior, [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]]; $\beta$-VAE, [[ATML_Lecture_5_2 - Domain Generalization]] |
| How much does $z$ know about $Y$? | mutual information (§3.5) | the discriminative and contrastive families, [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] |
| Match a model density to a data density | forward KL / maximum likelihood (§5.3) | VAE and ELBO (evidence lower bound), [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] |
| Commit to one mode, sharply | reverse KL (§5.4) | GAN mode collapse, [[ATML_Lecture_4 - GANs, and Inductive Biases]] |
| Compare two distributions symmetrically and finitely | Jensen–Shannon (§5.5) | the GAN value function, [[ATML_Lecture_4 - GANs, and Inductive Biases]] |
| Compare two sample sets without densities | MMD (§6.2) | Deep Adaptation Network, [[ATML_Lecture_5_1 - Domain Adaptation]] |
| Compare distributions that do not yet overlap | Wasserstein / Sinkhorn (§7.3, §7.5) | domain alignment, [[ATML_Lecture_5_1 - Domain Adaptation]] |

[[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] turns from comparing distributions to training an encoder, organizing every method into three families by the information-theoretic quantity it optimizes. The useful question to carry forward is which instrument from the table each family is implicitly using.

---

## Numerical Problems

**Problem 1 — Entropy of a discrete distribution.** A discrete random variable $X$ takes $4$ values with probabilities $P(x) = (0.1, 0.2, 0.3, 0.4)$. Compute the entropy $H(X)$ in bits.

*Solution.* Applying the entropy definition from Section 3.3, $H(X) = -\sum_x P(x)\log_2 P(x)$, each term is evaluated separately using $\log_2(0.1) \approx -3.3219$, $\log_2(0.2) \approx -2.3219$, $\log_2(0.3)\approx -1.7370$, and $\log_2(0.4)\approx -1.3219$. Multiplying each probability by the negative of its own log and summing,

$$
H(X) \approx 0.1(3.3219) + 0.2(2.3219) + 0.3(1.7370) + 0.4(1.3219) = 0.3322 + 0.4644 + 0.5211 + 0.5288.
$$

$$
\boxed{H(X) \approx 1.846 \text{ bits}}
$$

*Reasonableness check:* this is comfortably below the maximum possible entropy for $4$ outcomes, $\log_2 4 = 2$ bits (attained only by the uniform distribution), which is expected since this distribution is somewhat — though not extremely — non-uniform.

**Problem 2 — Information content of a rare event.** A biased coin lands heads with probability $P(\text{heads}) = 0.05$. Compute the information content (in bits) of observing a heads outcome, and compare it to the information content of a heads outcome from a *fair* coin.

*Solution.* Using the definition from Section 3.2, $I = -\log_2 P(x=x_k)$; substituting $P(\text{heads})=0.05$ gives $I = -\log_2(0.05) = \log_2(20)$. Since $\log_2(20) = \log_2(16) + \log_2(1.25) \approx 4 + 0.3219$,

$$
\boxed{I \approx 4.32 \text{ bits}}
$$

For a fair coin, $P(\text{heads}) = 0.5$, giving $I_{\text{fair}} = -\log_2(0.5) = 1$ bit.

*Reasonableness check:* $4.32 > 1$, confirming the core intuition of Section 3.2 — a rarer outcome is far more surprising, and hence carries far more information, than the same outcome on a fair coin.

**Problem 3 — Forward KL divergence.** Let $P = (0.6, 0.3, 0.1)$ and $Q = (0.4, 0.4, 0.2)$ over states $\{1,2,3\}$. Compute the forward KL divergence $D_{\mathrm{KL}}(P\|Q)$ in nats.

*Solution.* Using the definition from Section 5.2, $D_{\mathrm{KL}}(P\|Q) = \sum_i P_i \ln(P_i/Q_i)$, the three ratios are $P_1/Q_1 = 1.5$, $P_2/Q_2 = 0.75$, and $P_3/Q_3 = 0.5$, with natural logarithms $\ln(1.5)\approx 0.4055$, $\ln(0.75) \approx -0.2877$, and $\ln(0.5) \approx -0.6931$. Weighting each by its corresponding $P_i$ and summing,

$$
D_{\mathrm{KL}}(P\|Q) \approx 0.6(0.4055) + 0.3(-0.2877) + 0.1(-0.6931) = 0.2433 - 0.0863 - 0.0693.
$$

$$
\boxed{D_{\mathrm{KL}}(P\|Q) \approx 0.0877 \text{ nats}}
$$

*Reasonableness check:* the value is small and positive, as required for KL divergence between two similar (but not identical) categorical distributions; it is zero only when $P=Q$ exactly.

**Problem 4 — Reverse KL divergence and asymmetry.** For the same $P$ and $Q$ as in Problem 3, compute the reverse KL divergence $D_{\mathrm{KL}}(Q\|P)$, and confirm that KL divergence is not symmetric.

*Solution.* Using the definition from Section 5.4, $D_{\mathrm{KL}}(Q\|P) = \sum_i Q_i \ln(Q_i/P_i)$, the ratios are $Q_1/P_1 = 0.4/0.6 \approx 0.6667$, $Q_2/P_2 = 0.4/0.3 \approx 1.3333$, and $Q_3/P_3 = 0.2/0.1 = 2$, with logarithms $\ln(0.6667)\approx -0.4055$, $\ln(1.3333)\approx 0.2877$, and $\ln(2) \approx 0.6931$. Weighting by $Q_i$ and summing,

$$
D_{\mathrm{KL}}(Q\|P) \approx 0.4(-0.4055) + 0.4(0.2877) + 0.2(0.6931) = -0.1622 + 0.1151 + 0.1386.
$$

$$
\boxed{D_{\mathrm{KL}}(Q\|P) \approx 0.0915 \text{ nats}}
$$

*Reasonableness check:* comparing to Problem 3, $0.0915 \neq 0.0877$ — the two directions give different values for the same pair of distributions, directly confirming the asymmetry of Section 5.4.

**Problem 5 — Jensen–Shannon divergence.** For the same $P = (0.6,0.3,0.1)$ and $Q = (0.4,0.4,0.2)$, compute $\mathrm{JS}(P\|Q)$ in nats and verify that it is finite and symmetric.

*Solution.* First form the mixture $M = (P+Q)/2 = (0.5,\, 0.35,\, 0.15)$. For $\mathrm{KL}(P\|M)$, the ratios are $0.6/0.5 = 1.2$, $0.3/0.35 \approx 0.8571$, and $0.1/0.15 \approx 0.6667$, with logs $0.1823$, $-0.1542$, and $-0.4055$; weighting by $P_i$ gives $0.1094 - 0.0463 - 0.0405 \approx 0.0226$. For $\mathrm{KL}(Q\|M)$, the ratios are $0.4/0.5 = 0.8$, $0.4/0.35 \approx 1.1429$, and $0.2/0.15 \approx 1.3333$, with logs $-0.2231$, $0.1335$, and $0.2877$; weighting by $Q_i$ gives $-0.0892 + 0.0534 + 0.0575 \approx 0.0217$. Averaging the two,

$$
\mathrm{JS}(P\|Q) = \tfrac{1}{2}(0.0226) + \tfrac{1}{2}(0.0217).
$$

$$
\boxed{\mathrm{JS}(P\|Q) \approx 0.0222 \text{ nats}}
$$

*Reasonableness check:* the value lies between the forward KL ($0.0877$) and zero, as expected since each KL is taken against the *mixture* rather than against the other distribution — which is exactly the mechanism that keeps JS finite. Swapping $P$ and $Q$ leaves $M$ unchanged and merely swaps the two averaged terms, so the result is identical: JS is symmetric.

**Problem 6 — TVD and the Wasserstein-1 distance via optimal transport.** For $P = (0.6, 0.3, 0.1)$ and $Q = (0.4, 0.4, 0.2)$ at positions $\{1,2,3\}$, compute (a) the Total Variation Distance, and (b) the Wasserstein-1 cost using the $\ell_1$ ground cost $C_{ij} = |i-j|$, by finding the minimum-cost transport plan.

*Solution.* (a) Using the definition from Section 6.3, the absolute differences are $|0.6-0.4|=0.2$, $|0.3-0.4|=0.1$, and $|0.1-0.2|=0.1$, giving $\text{TVD} = \tfrac{1}{2}(0.2+0.1+0.1)$.

$$
\boxed{\text{TVD} = 0.2}
$$

(b) Comparing $P$ and $Q$ bin by bin, $P$ has a surplus of $0.2$ at bin $1$, while $Q$ has a surplus of $0.1$ at bin $2$ and $0.1$ at bin $3$. The cheapest reconciliation moves $0.1$ units from bin $2$ to bin $1$ (unit cost $C_{21}=1$) and $0.1$ units from bin $3$ to bin $1$ (unit cost $C_{31}=2$), leaving $0.4$, $0.3$, and $0.1$ in place respectively — this plan satisfies both marginal constraints and routes mass only along the shortest available paths. The total transport cost is $(0.1)(1) + (0.1)(2)$.

$$
\boxed{\langle T^*, C\rangle = 0.3 \;\;\Rightarrow\;\; WD_1 = 0.3 \text{ (or } \sqrt{0.3}\approx 0.548 \text{ under the square-root convention of §7.3)}}
$$

*Reasonableness check:* the transport cost can be cross-checked with the closed-form shortcut for the $1$-Wasserstein cost on an ordered line — the sum of absolute differences between the two CDFs (cumulative distribution functions): $\text{CDF}_P = (0.6, 0.9, 1.0)$ and $\text{CDF}_Q = (0.4, 0.8, 1.0)$, giving $|0.6-0.4| + |0.9-0.8| = 0.3$, matching exactly. Note also that the raw transport cost $(0.3)$ exceeds the TVD $(0.2)$: unlike TVD, the Wasserstein distance accounts for *how far* mass must travel, not just *how much* differs.

**Problem 7 — One Sinkhorn iteration.** Take $\underline{P} = (0.5, 0.5)^\top$, $\underline{q} = (0.5, 0.5)^\top$, cost matrix $C = \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}$, and regularization $\varepsilon = 1$. Form the Gibbs kernel $K$ and perform the first Sinkhorn update for $\underline{u}$.

*Solution.* Applying step 1 of the algorithm in Section 7.5, $K = e^{-C/\varepsilon}$ elementwise with $\varepsilon = 1$ gives $K = \begin{bmatrix}e^{0} & e^{-1}\\ e^{-1} & e^{0}\end{bmatrix} = \begin{bmatrix}1 & 0.3679\\ 0.3679 & 1\end{bmatrix}$. Initializing $\underline{v}^0 = (1,1)^\top$, the matrix–vector product is $K\underline{v}^0 = (1.3679,\, 1.3679)^\top$, and the elementwise division $\underline{u}^1 = \underline{P}/(K\underline{v}^0)$ gives

$$
\boxed{\underline{u}^{1} = \left(\tfrac{0.5}{1.3679},\; \tfrac{0.5}{1.3679}\right)^{\top} \approx (0.3656,\; 0.3656)^{\top}}
$$

*Reasonableness check:* both components are equal, as they must be by the symmetry of $P$, $q$, and $C$ in this problem. Continuing the iteration would give $\underline{v}^1 = \underline{u}^1$ and converge to a plan $T^* = \operatorname{diag}(u)K\operatorname{diag}(v)$ whose diagonal dominates — mass mostly stays put, which is correct since $P = q$ here and the exact LP optimum is the identity-scaled plan.

**Problem 8 — Shape and properties of a 2-D Gaussian.** Let $x \sim \mathcal{N}(0, \Sigma)$ with $\Sigma = \begin{bmatrix} 3 & 1 \\ 1 & 3\end{bmatrix}$. (a) Find the principal axes and semi-axis lengths of the Mahalanobis-radius-1 ellipse, and the correlation coefficient. (b) Compare the Mahalanobis distances of $a = (2, 2)^{\top}$ and $b = (2, -2)^{\top}$ from the mean. (c) Compute the entropy in nats. (d) Find the conditional distribution of $x_2$ given $x_1 = 3$.

*Solution.* (a) The characteristic equation $(3-\lambda)^2 - 1 = 0$ gives $\lambda_1 = 4$ and $\lambda_2 = 2$, with unit eigenvectors $u_1 = \tfrac{1}{\sqrt2}(1, 1)^{\top}$ and $u_2 = \tfrac{1}{\sqrt2}(1,-1)^{\top}$. By Section 2.3 the semi-axes are $\sqrt{\lambda_i}$ along $u_i$, and the correlation is $\rho = \sigma_{12}/(\sigma_1\sigma_2) = 1/3$.

$$
\boxed{\text{major semi-axis } 2 \text{ along } (1,1)/\sqrt2, \quad \text{minor semi-axis } \sqrt2 \approx 1.414 \text{ along } (1,-1)/\sqrt2, \quad \rho = 1/3}
$$

(b) Since $\lvert\Sigma\rvert = 9 - 1 = 8$, the inverse is $\Sigma^{-1} = \tfrac18\begin{bmatrix}3 & -1\\ -1 & 3\end{bmatrix}$. Evaluating the quadratic form, $a^{\top}\Sigma^{-1}a = \tfrac18(12 - 8 + 12) = 2$ and $b^{\top}\Sigma^{-1}b = \tfrac18(12 + 8 + 12) = 4$.

$$
\boxed{d_M(a) = \sqrt2 \approx 1.414, \qquad d_M(b) = 2}
$$

(c) By property 5, $H = \tfrac12\ln\big((2\pi e)^2 \cdot 8\big) = \ln(2\pi e) + \tfrac12\ln 8 \approx 2.838 + 1.040$.

$$
\boxed{H \approx 3.878 \text{ nats}}
$$

(d) By property 3 with scalar blocks, the conditional mean is $0 + \tfrac{1}{3}(3 - 0) = 1$ and the conditional variance is $3 - \tfrac{1\cdot 1}{3} = \tfrac83$.

$$
\boxed{x_2 \mid x_1 = 3 \;\sim\; \mathcal{N}\!\left(1,\; \tfrac{8}{3}\right)}
$$

*Reasonableness check:* $a$ and $b$ are equidistant from the origin in Euclidean terms ($2\sqrt2$ each), yet $a$ lies along the major axis, where the distribution is spread wide, and so is Mahalanobis-closer — exactly the effect of measuring distance in units of the distribution's own spread. The uncorrelated Gaussian with the same variances, $\operatorname{diag}(3,3)$, has determinant $9 > 8$ and so entropy $\tfrac12\ln 9 - \tfrac12\ln 8 \approx 0.059$ nats higher, consistent with Hadamard's inequality; and the conditional variance $8/3 < 3$ confirms that observing a correlated coordinate reduces uncertainty, by an amount independent of the observed value.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Expectation (continuous / discrete) | $\mathbb{E}[X] = \int x P(x)\,dx = \sum_{x_i} x_i P(x_i)$ |
| Expectation of a function | $\mathbb{E}_{P(x)}[g(x)] = \int g(x)P(x)\,dx$ |
| Monte Carlo approximation | $\mathbb{E}_{P(x)}[g(x)] \approx \frac{1}{N}\sum_{i=1}^{N} g(x_i)$ |
| Independent latent prior | $[z_1,\ldots,z_d]^\top \sim \mathcal{N}(0, I)$ |
| Multivariate Gaussian density | $p(x) = (2\pi)^{-d/2}\lvert\Sigma\rvert^{-1/2}\exp\!\big(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$ |
| Mahalanobis distance | $d_M(x,\mu) = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$ |
| Gaussian shape | $\Sigma = U\Lambda U^\top$: ellipsoid axes along $u_i$, semi-axes $c\sqrt{\lambda_i}$; volume $\propto\sqrt{\lvert\Sigma\rvert}$ |
| Affine closure / sampling | $Ax+b \sim \mathcal{N}(A\mu+b,\, A\Sigma A^\top)$; $\mu + L\epsilon \sim \mathcal{N}(\mu,\Sigma)$ with $LL^\top=\Sigma$ |
| Gaussian conditional | $x_a\mid x_b \sim \mathcal{N}\big(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b),\; \Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\big)$ |
| Gaussian entropy | $H = \tfrac12\ln\big((2\pi e)^d\lvert\Sigma\rvert\big)$ nats |
| KL between Gaussians | $\tfrac12\big[\operatorname{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0) - d + \ln\tfrac{\lvert\Sigma_1\rvert}{\lvert\Sigma_0\rvert}\big]$ |
| KL to standard prior (diagonal) | $\tfrac12\sum_j(\mu_j^2+\sigma_j^2-1-\ln\sigma_j^2)$ |
| High-dimensional shell | $z\sim\mathcal{N}(0,I_d)$: $\lVert z\rVert^2\sim\chi^2_d$, $\lVert z\rVert \approx \sqrt{d}$ |
| Information content | $I(x=x_k) = -\log_2 P(x=x_k)$ [bits] |
| Entropy | $H(x) = -\sum_x P(x)\log_2 P(x)$ |
| Max-entropy result | Among all distributions with fixed mean and variance, the Gaussian has maximum entropy |
| Mutual information | $I(X;Y) = \mathbb{E}_{P(X,Y)}\!\left[\log\frac{P(X,Y)}{P(X)P(Y)}\right]$ |
| Perceptual loss | $\lVert \phi(x)-\phi(\hat{x})\rVert_2^2$ |
| General f-divergence | $D_f(P\Vert Q) = \mathbb{E}\!\left[f\!\left(\frac{P}{Q}\right)\right]$ |
| Forward KL divergence | $D_{\mathrm{KL}}(P\Vert Q) = \mathbb{E}_{P}\!\left[\log\frac{P(x)}{Q(x)}\right]$ — mode covering |
| Reverse KL divergence | $D_{\mathrm{KL}}(Q\Vert P) = \mathbb{E}_{Q}\!\left[\log\frac{Q(x)}{P(x)}\right]$ — mode seeking |
| Jensen–Shannon divergence | $\mathrm{JS}(P\Vert Q) = \tfrac12\mathrm{KL}(P\Vert M) + \tfrac12\mathrm{KL}(Q\Vert M)$, $M=\frac{P+Q}{2}$ |
| Maximum Mean Discrepancy (IPM form) | $\text{MMD}(P,Q) = \max_f \lvert \mathbb{E}_{P}[f(x)] - \mathbb{E}_{Q}[f(x)]\rvert$ |
| MMD (discrete kernel form) | $\text{MMD}^2 = (P-Q)^\top K (P-Q)$ |
| Total Variation Distance | $\text{TVD}(P,Q) = \tfrac12\int\lvert P(x)-Q(x)\rvert dx$ |
| OT cost matrix | $C_{ij} = \lVert x_i - x_j\rVert_2^2$ (or $\lvert i-j\rvert$) |
| OT total cost | $\langle T, C\rangle = \sum_i\sum_j T_{ij}C_{ij}$ |
| OT marginal constraints | $T\mathbf{1} = \underline{P}$, $T^\top\mathbf{1} = \underline{q}$ |
| OT linear program | $\min_T \langle T,C\rangle$ s.t. $T\mathbf{1}=\underline{P}$, $T^\top\mathbf{1}=\underline{q}$, $T_{ij}\ge 0$ |
| Wasserstein distance | $\text{WD}(P,Q) = \sqrt{\langle T_w^*, C\rangle}$ |
| Entropic OT objective | $\min_{T\ge0}\langle T,C\rangle - \varepsilon H(T)$, $H(T) = -\sum_{ij}T_{ij}\log T_{ij}$ |
| Sinkhorn distance | $\text{SD}(P,Q) = \sqrt{\langle T_s^*, C\rangle}$ |
| Sinkhorn algorithm | $K = e^{-C/\varepsilon}$; $u^{t+1} = \frac{P}{Kv^t}$, $v^{t+1} = \frac{q}{K^\top u^{t+1}}$; $T^* = \operatorname{diag}(u)K\operatorname{diag}(v)$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the divergence zoo** — *"**K**ind **R**abbits **J**ump **M**ostly **T**oward **W**ater"*: **K**L forward, **R**everse KL, **J**S, **M**MD, **T**VD, **W**asserstein — roughly in order of increasing robustness to disjoint support.

**Quick-reference intuition per topic**

- **Monte Carlo:** can't compute a true expectation because $P(X)$ is unknown → replace the integral with an average over samples you actually have.
- **Independent latents:** identity covariance = "each latent dimension is its own independent source of information"; off-diagonal correlation = "dimensions are wasting capacity learning the same thing."
- **Gaussian shape:** a Gaussian is a unit sphere stretched by $\Lambda^{1/2}$, rotated by $U$, moved by $\mu$. Eigenvectors = axis directions, $\sqrt{\text{eigenvalues}}$ = axis lengths, off-diagonals = tilt. Density depends on Mahalanobis, not Euclidean, distance.
- **Gaussian gotchas** *(from the self-study section)*: uncorrelated ⇒ independent holds *only* for jointly Gaussian vectors; correlation lowers entropy ($\lvert\Sigma\rvert \le \prod_i\Sigma_{ii}$); in high $d$ the mass sits on a shell of radius $\sqrt d$, not at the mode.
- **Information/entropy:** rare events are surprising (high info); entropy is the average surprise; uniform = max entropy, deterministic = zero entropy; **fix mean and variance → Gaussian is the max-entropy choice.**
- **Mutual information:** how much knowing one variable reduces uncertainty about the other; zero exactly when the two are independent.
- **Two loss families:** distribution-referenced (need a reference $P$; match $Q$ to it) vs. behavior/geometry/decision (no reference needed — reconstruction losses like MSE and perceptual, or latent-geometry losses like contrastive/InfoNCE/triplet).
- **Divergences, one line each:**
  - Forward KL $D(P\|Q)$: blows up if $Q = 0$ where $P > 0$ → **mode-covering / covers support**.
  - Reverse KL $D(Q\|P)$: blows up if $Q > 0$ where $P \approx 0$ → **mode-seeking / covers the dominant mode**.
  - JS: symmetrized via the mixture $M$; stays finite on disjoint supports; the implicit GAN objective.
  - MMD: an IPM — "no test function can tell $P$ and $Q$ apart"; needs only samples, not densities.
  - TVD: symmetric, simple, half the $\ell_1$ gap — cares how *much* differs, not *where*.
  - Wasserstein / Sinkhorn: the only ones that account for *distance moved*.
- **KL asymmetry:** $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$ — always check which argument is "true" and which is "model" before quoting a KL value.
- **Why OT exists:** KL is $\infty$ for any two distributions with unequal support overlap, so it cannot distinguish "barely disjoint" from "very far apart." OT can, because it measures ground distance travelled.
- **Reading a transport plan:** diagonal entries = mass staying put (zero cost); off-diagonal entry $T_{ij}$ = mass moving from bin $i$ to bin $j$ at cost $C_{ij}$.
- **Row vs. column:** rows of $T$ sum to $P$, columns sum to $Q$. If a candidate plan fails either, it isn't admissible.
- **Sinkhorn vs. exact OT:** exact OT is a hard LP; Sinkhorn adds $-\varepsilon H(T)$ to make it strictly convex and solvable by alternating $u$/$v$ rescaling of $K = e^{-C/\varepsilon}$. Smaller $\varepsilon$ = closer to exact but slower.

**Recall checklist**

- [ ] Can you state why we replace a statistical average with a sample average?
- [ ] Can you explain, via the covariance matrix, why we want latent dimensions independent?
- [ ] Given a $2\times2$ covariance, can you find the ellipse's axes and lengths, a Mahalanobis distance, the entropy, and a conditional?
- [ ] Can you state the six Gaussian properties (affine closure, marginals, conditionals, uncorrelated⇒independent, entropy, closed-form KL) and where each is used later?
- [ ] Can you explain why most samples of $\mathcal{N}(0, I_d)$ lie near radius $\sqrt d$?
- [ ] Can you compute entropy and information content for a small discrete distribution in bits?
- [ ] Can you state the max-entropy property of the Gaussian and its precise condition?
- [ ] Can you write the mutual information definition and say when it equals zero?
- [ ] Can you name the two families of DRL losses and give two examples of each?
- [ ] Can you write $D_f(P\|Q) = \mathbb{E}[f(P/Q)]$ and recover forward KL from it?
- [ ] Can you say which KL direction blows up under which support mismatch, and name the resulting behavior?
- [ ] Can you write JS divergence and explain why it stays finite?
- [ ] Can you state MMD in IPM form and in kernel form?
- [ ] Can you list the two desirable metric properties KL fails, and give the two-box example?
- [ ] Can you write the OT linear program with all three constraints, and the WD and SD formulas?
- [ ] Can you verify a transport plan's marginals and compute its cost from a cost matrix?
- [ ] Can you write out the three steps of the Sinkhorn algorithm from memory?

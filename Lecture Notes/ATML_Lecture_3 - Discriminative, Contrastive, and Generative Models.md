---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module I - Notes.pdf]]'
- '[[ATML Lec 3.pdf]]'
tags:
- course/cs6304
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_2 - Information Theory and Divergences]] · [[ATML_Lecture_4 - GANs, and Inductive Biases]] →
Concepts: [[Contrastive Learning]]
%% nav:end %%

# ATML — Lecture 3: The Three Families of DRL — Discriminative, Contrastive, and Generative Models (through VAEs)

## Where This Lecture Sits

*Continues from [[ATML_Lecture_2 - Information Theory and Divergences]], which built the instruments for comparing distributions but said nothing about how an encoder is trained. This lecture answers that, and the instruments are used throughout rather than re-derived.*

Every way of training $g_\theta$ falls into one of three families, distinguished by which Lecture 2 quantity the loss optimizes: discriminative models minimize $H(Y\mid z)$ via cross-entropy, contrastive models maximize $I(x, x^+)$ via the InfoNCE (the contrastive loss built on noise-contrastive estimation) bound, and generative models maximize likelihood (forward KL (Kullback–Leibler)), relaxed to the ELBO (evidence lower bound) for VAEs (variational autoencoders). Each is a surrogate for an intractable information-theoretic target, and which surrogate is used explains each family's failure mode. Sections 2.5–2.10 (CNN (convolutional neural network) and Transformer mechanics) are the prerequisite for the architectural biases of [[ATML_Lecture_4 - GANs, and Inductive Biases]].

## Topic Inventory

- 1. The Three Families of Deep Representation Learning
  - 1.1 The Common Pipeline
  - 1.2 Family Comparison
- 2. Family 1 — Discriminative Models
  - 2.1 Setup and Objective
  - 2.2 The Information-Theoretic View: Reducing $H(Y \mid z)$
  - 2.3 From $H(Y \mid z)$ to Cross-Entropy and Negative Log-Likelihood
  - 2.4 The Decomposition $\mathcal{L}_{\text{CE}} = I(Y;z) + D_{\mathrm{KL}}$
  - 2.5 CNNs — The Convolutional Layer
  - 2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding
  - 2.7 The Transformer Layer
  - 2.8 Multi-Head Self-Attention
  - 2.9 Attention as Query–Key–Value Retrieval
  - 2.10 The Block-Matrix / Kronecker View of Self-Attention
- 3. Family 2 — Contrastive Models
  - 3.1 Positive and Negative Augmentations
  - 3.2 The Contrastive Pipeline and the InfoNCE Loss
  - 3.3 InfoNCE as a Lower Bound on Mutual Information
  - 3.4 CLIP — Contrastive Language–Image Pretraining
  - 3.5 CLIP Training
  - 3.6 Zero-Shot Inference with CLIP
- 4. Family 3 — Generative Models
  - 4.1 Setup and the Modelling Goal
  - 4.2 From KL Minimization to Maximum Likelihood
  - 4.3 What Maximizing the Log-Likelihood Actually Buys
  - 4.4 The Concept of the Latent $z$
  - 4.5 Three Sub-Families of Generative Models
- 5. Variational Autoencoders
  - 5.1 From AE to DAE to VAE
  - 5.2 Prior, Posterior, and the VAE Architecture
  - 5.3 The ELBO
  - 5.4 Proof That the ELBO Lower-Bounds the Log-Likelihood
  - 5.5 Gradients of the ELBO and the Reparameterization Trick
  - 5.6 Failure Modes: Posterior Collapse and Blurry Samples
  - 5.7 Generation
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. The Three Families of Deep Representation Learning

Lecture 1 fixed the pipeline and Lecture 2 supplied the measurement tools; what remains unspecified is the loss. Since the loss is the only thing that differs between the methods in this course, classifying losses classifies methods — and the taxonomy below is the one the remainder of the course uses.

### 1.1 The Common Pipeline

All three families share the same skeleton established in Lecture 1:

$$
x \;\xrightarrow{\;g_\theta(\cdot)\;}\; z \;\xrightarrow{\;h_\psi(\cdot)\;}\; \hat{Y} \;\longrightarrow\; \text{Loss} \;\longleftarrow\; Y
$$

with two responsibilities cleanly separated: $g_\theta(\cdot)$ **maps $x$ to the latent $z$**, and $h_\psi(\cdot)$ **turns $z$ into whatever the loss needs**. The three families differ entirely in *what the loss needs* — and therefore in what **geometry** they impose on $z$.

![[03 - Attachments/ATML_L3_three_families.png|620]]

*The common pipeline $x \to g_\theta \to z \to h_\psi \to \hat{Y}$ and how each family shapes the geometry of $z$: separated class clusters (discriminative), generative manifolds (generative), cross-view/modality alignment (contrastive).*

### 1.2 Family Comparison

| | **Discriminative** | **Generative** | **Contrastive** |
|---|---|---|---|
| **Supervision** | Supervised | Unsupervised or conditional | Supervised / self-supervised / multimodal |
| **Goal** | Make $z$ a simple predictor for $Y$ | Make $z$ explain $X$ (or $X\mid Y$) | Make $z$ *relative* — pull positives together, push negatives apart |
| **Examples** | CNN, ViT (vision transformer), Transformer, MLP (multilayer perceptron) | VAE, GAN (generative adversarial network), Diffusion Models | CLIP, VLMs (vision-language models) |
| **Geometry of $z$** | Class clusters well-separated through margins | $z$ lies on generative manifolds | $z$ aligned across modalities / augmentations |
| **Mechanism** | Directly predict the class maximizing $P(Y\mid X)$: $P(\hat{y}=k\mid x) = \text{Softmax}(\psi_k^\top z)$ | Model $P(X)$ either explicitly or implicitly | Maximize MI between augmented views/modalities: $I(x, x^+) = \max$, $I(x, x^-) = \min$ |

The rest of the lecture takes each family in turn. The row worth dwelling on is the last: each family's mechanism is an attempt at an intractable information-theoretic quantity, and the distinctive geometry each imposes on $z$ is a consequence of which approximation it settles for.

---

## 2. Family 1 — Discriminative Models

The first family is the one Lecture 1 already treated implicitly: the softmax head and cross-entropy loss of Lecture 1 Section 7 are exactly a discriminative model. What this section adds is the justification — why cross-entropy is the right objective rather than merely a convenient one — which turns out to follow from the mutual-information framing of Lecture 2 Section 3.5.

### 2.1 Setup and Objective

The dataset is the standard supervised one,
$$
\mathcal{D} = \{\vec{x}_i,\, y_i\}_{i=1}^{N}, \qquad x_i \in \mathbb{R}^{d}, \qquad y_i \in \{1, \ldots, K\},
$$
and the pipeline is
$$
x \;\xrightarrow{g_\theta(\cdot)}\; z \in \mathbb{R}^{d\times 1}\;(d \ll D) \;\xrightarrow{h_\psi(\cdot)}\; P(Y = i \mid \underbrace{x}_{\text{e.g. class}=\text{cat}}).
$$
Of the various distributions available — $P(X,Y)$, $P(Y)$, $P(z)$, $P(X)$ — **discriminative models try to learn $P(Y \mid X)$, or equivalently $P(Y \mid z)$**. The relationship is written
$$
Y \mid X \;\equiv\; Y \mid z
$$
and the notes are careful about what "$\equiv$" means here: the two are **equivalent, but not necessarily the same**. They are equivalent in the sense that $z$ is supposed to be a *sufficient statistic* of $x$ for predicting $Y$ (property 1 of representation theory, Lecture 1) — so conditioning on $z$ should be as informative as conditioning on $x$ — but they are not literally identical distributions, since $z$ is a lossy compression of $x$.

Expanded, the forward pass is
$$
x \;\xrightarrow{g_\theta(\cdot)}\; z \;\xrightarrow{h_\psi(\cdot)}\; s \;(\text{logits}) \;\xrightarrow{\;\text{softmax}\;}\; \underbrace{P_\psi(Y \mid z)}_{\text{output distribution}}.
$$

### 2.2 The Information-Theoretic View: Reducing $H(Y \mid z)$

Before seeing any data, the label $Y$ carries some **uncertainty**, quantified by its entropy $H(Y)$. **We have to decrease this randomness.** After conditioning on an observed input — e.g. $Y \mid x = \text{cat}$ — the remaining uncertainty is the **conditional entropy** $H(Y \mid X)$, or equivalently $H(Y \mid z)$. **The model's job is to reduce $H(Y \mid z)$.**

This is exactly the definition of **mutual information**:
$$
\boxed{I(Y; z) = \underbrace{H(Y)}_{\text{fixed}} - \underbrace{H(Y \mid z)}_{\text{make } z \text{ predictive of } Y}}
$$

The first term, the **entropy of the original class distribution, is fixed** — it is a property of the dataset, not of the model. So the only way to increase mutual information is to **drive $H(Y \mid z)$ down**, which is precisely what it means to **force the latent to carry information about the label** and to **make $z$ predictive of $Y$**. Reducing conditional entropy and maximizing mutual information are two descriptions of the same optimization.

### 2.3 From $H(Y \mid z)$ to Cross-Entropy and Negative Log-Likelihood

The conditional entropy we would like to minimize is

$$
H(Y \mid z) = -\int P(Y \mid z) \log\big(\underbrace{P(Y \mid z)}_{\text{true distribution of } Y\mid z}\big)\, dz,
$$

and here is the obstacle: **we never have the true $P(Y \mid z)$**. What we have instead is the model's assumed distribution, fitted to data:

$$
\underbrace{P_\psi(Y \mid z)}_{\text{assumed, from data}} \;\neq\; \underbrace{P(Y \mid z)}_{\text{unknown, true}}.
$$

We therefore **approximate $P(Y\mid z)$ using the cross-entropy loss**. For a single example whose true label is $j$,

$$
\boxed{\mathcal{L}_{\text{CE}} = -\log P_\psi(Y = j \mid z)}
$$

This quantity is called the **Negative Log-Likelihood (NLL)** — the two names refer to the same expression. Here $j$ is the true label of example $i$. Averaged over a batch or dataset of $N$ examples,

$$
\boxed{\mathcal{L}_{\text{CE}} = \frac{1}{N}\sum_{i=1}^{N} -\log P_\psi(Y = j \mid z_i)}
$$

which is the Monte Carlo estimator (Lecture 2, Section 1.2) of the population expectation.

### 2.4 The Decomposition $\mathcal{L}_{\text{CE}} = I(Y;z) + D_{\mathrm{KL}}$

The gap between the assumed and true conditionals is itself a KL divergence, and it is non-negative:

$$
D_{\mathrm{KL}}\big(P_\psi(Y\mid z) \,\|\, P(Y\mid z)\big) \;\ge\; 0.
$$

This lets the cross-entropy be decomposed into an information term plus an approximation-error term:

$$
\boxed{\mathcal{L}_{\text{CE}} = I(Y; z) + \underbrace{D_{\mathrm{KL}}\big(P_\psi(Y\mid z)\,\|\,P(Y\mid z)\big)}_{\text{we don't have this}}}
$$

The practical reading is: minimizing cross-entropy simultaneously pushes on the *information* content of the representation and on the *fidelity* of the model's assumed conditional to the true one — but only the first is under our control in any explicit way, because the second involves the inaccessible $P(Y\mid z)$. This is the precise sense in which, as stated in Lecture 2, "$\mathcal{L}_{\text{CE}}$ comes from bounds on mutual information."

*Intuition.* Since the KL term is non-negative, cross-entropy upper-bounds the information term, and minimizing an upper bound on a quantity we cannot evaluate is the only available move — a pattern that recurs in each of the three families, with InfoNCE (Section 3.3) and the ELBO (Section 5.3) as the other two instances. The structure of the decomposition also explains a familiar practical fact: a model can reduce its cross-entropy by genuinely making $z$ more informative about $Y$, or merely by calibrating $P_\psi$ closer to the truth, and the loss cannot distinguish the two. A confidently miscalibrated network and an uninformative one are both punished, but only the first is repairable by temperature scaling rather than by better representations.

> **[Possible source inconsistency]** — the notes write the first term as $I(Y \mid z)$, using a conditional bar. Mutual information is written with a semicolon, $I(Y; z)$, and the surrounding derivation (Section 2.2) is unambiguously about mutual information, so $I(Y; z)$ is used here. Note also that under the standard identity $\mathcal{L}_{\text{CE}} = H(Y\mid z) + D_{\mathrm{KL}}$, the first term would be the *conditional entropy* rather than the mutual information; since $H(Y)$ is a fixed constant, minimizing either is equivalent up to that constant, which is presumably why the professor writes it this way.

**Models under this discriminative family:** CNNs, ViT, Transformer, MLP. The next sections work through the layer structure of the first two.

### 2.5 CNNs — The Convolutional Layer

For a CNN, **the layer is a convolutional layer**. An input tensor $x_{\text{in}}$ of shape $H_1 \times W_1 \times C_1$ passes through a **3D convolution layer** and produces an output $x_{\text{out}}$ of shape $H_2 \times W_2 \times C_2$. The layer is specified by four hyperparameters:

- **Number of kernels** $= C_2$ (this is what sets the output channel count).
- **Kernel size** $F \times F$.
- **Padding** $P$.
- **Stride** $S$.

The output spatial dimensions follow the standard formula:

$$
\boxed{H_2 = \left\lfloor \frac{H_1 + 2P - F}{S} \right\rfloor + 1, \qquad\qquad W_2 = \left\lfloor \frac{W_1 + 2P - F}{S} \right\rfloor + 1}
$$

The convolution is called "3D" because each kernel spans all $C_1$ input channels ($F \times F \times C_1$) even though it slides in only two spatial dimensions.

![[03 - Attachments/ATML_L3_cnn_layer.png|681]]

*A conv layer maps $H_1 \times W_1 \times C_1$ to $H_2 \times W_2 \times C_2$ with $C_2$ kernels of size $F\times F$, padding $P$, stride $S$.*

### 2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding

For a Vision Transformer, **the layer is an attention layer**, and the image must first be turned into a sequence of tokens. This happens in three highlighted stages.

**Stage 1 — Patchify.** The input $x_{\text{in}}$ of shape $H_1 \times W_1 \times C_1$ is cut into non-overlapping square tiles of **patch size** $P \times P$. The number of patches is

$$
N_P = \frac{H_1}{P} \times \frac{W_1}{P},
$$

and each patch has shape $P \times P \times C_1$.

**Stage 2 — Patch vectors.** Each patch tile is **flattened** from $P \times P \times C_1$ into a column vector of length $P^2 C_1$. Stacking all patches row-wise gives

$$
x_{\text{flat}} \in \mathbb{R}^{N_P \times P^2 C_1}.
$$

**Stage 3 — Patch embedding.** A single learned linear map projects each patch vector into the $D$-dimensional embedding space, where **$D$ is the embedding dimension**:

$$
\boxed{x_{\text{embed}} = \underbrace{x_{\text{flat}}}_{N_P \times P^2C_1} \times \underbrace{W_E}_{P^2C_1 \times D} + \underbrace{\mathbf{1}}_{N_P \times 1}\underbrace{b_E^{\top}}_{1 \times D}, \qquad x_{\text{embed}} \in \mathbb{R}^{N_P \times D}}
$$

The outer-product term $\mathbf{1}\,b_E^\top$ is just the bias $b_E \in \mathbb{R}^{D\times 1}$ broadcast identically across all $N_P$ patch rows.

**Stage 4 — CLS (a special classification token prepended to the sequence) token and positional embedding.** Because attention is permutation-equivariant (see Lecture 4), position information must be injected explicitly. Optionally a learned classification token is prepended:

$$
x_0 =
\begin{cases}
\underbrace{[\,x_{\text{CLS}} \,;\, x_{\text{embed}}\,]}_{(N_P + 1) \times D} + \underbrace{E_{\text{pos}}}_{(N_P+1)\times D} & \text{if [CLS] = YES},\\[2ex]
\underbrace{x_{\text{embed}}}_{N_P \times D} + \underbrace{E_{\text{pos}}}_{N_P \times D} & \text{otherwise.}
\end{cases}
$$

Finally the sequence is transposed so that the **embedding dimension runs down the rows and the token/patch dimension runs across the columns**:

$$
X = x_0^{\top} \in \mathbb{R}^{D \times N},
$$

i.e. $X$ is a matrix of $N$ column vectors, each of size $D \times 1$, one per token. This $D \times N$ convention is used consistently for the rest of the transformer material.

![[03 - Attachments/ATML_L3_vit_patchify.png|620]]

*ViT input: patchify into $N_p$ patches of $P\times P$, flatten, embed with $W_E$, prepend CLS and add positional embeddings to get $X_0 \in \mathbb{R}^{D \times N}$.*

### 2.7 The Transformer Layer

A transformer layer applies attention and an MLP, each wrapped in a residual connection and followed by layer normalization. In the order written:

$$
X = X + \text{MHSA}(X), \qquad X \in \mathbb{R}^{D\times N}
$$
$$
X = \text{LayerNorm}(X), \qquad X \in \mathbb{R}^{D\times N}
$$
$$
\underline{x}_n = \underline{x}_n + \text{MLP}(\underline{x}_n), \qquad \forall\, n = 1, 2, \ldots, N
$$
$$
X_1 = \text{LayerNorm}(X), \qquad X_1 \in \mathbb{R}^{D\times N}
$$

**LayerNorm, in detail.** For each embedding (i.e. each column of $X$, hence **column-wise normalization**), compute the mean and variance across the $D$ feature dimensions and standardize:

$$
\mu = \frac{1}{D}\sum_i x_i, \qquad \sigma^2 = \frac{1}{D}\sum_i (x_i - \mu)^2 \;\Longrightarrow\; \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},
$$

$$
\boxed{x_i = \gamma_i\, \hat{x}_i + \beta_i}
$$

where $\gamma_i$ and $\beta_i$ are learned per-feature scale and shift parameters and $\varepsilon$ guards against division by zero. Note that both the normalization and the MLP act **column-wise** — i.e. independently per token — which is why they are drawn in the block diagram as $N$ parallel identical blocks. Only the attention block mixes information *across* tokens.

The full block diagram is: $X$ ($D\times N$) $\to$ MHSA $\to$ residual add $\to$ LayerNorm (column-wise) $\to$ column-wise MLP blocks $\to$ residual add $\to$ LayerNorm $\to$ $X_1$ ($D\times N$).

![[03 - Attachments/ATML_L3_transformer_layer.png|620]]

*One Transformer layer: MHSA with a residual, column-wise LayerNorm, column-wise MLP blocks with a second residual, then LayerNorm.*

### 2.8 Multi-Head Self-Attention

Assuming for simplicity **only one head**, $h = 1$:

$$
\underbrace{\text{MHSA}(X)}_{D\times N} = \underbrace{W_c}_{D \times D}\,\big[\underbrace{\text{SA}_1(X)}_{D\times N}\big]
$$

where $W_c$ is the output projection that mixes heads (trivial when $h=1$). The single-head self-attention itself is

$$
\boxed{\underbrace{\text{SA}_1(X)}_{D\times N} = \underbrace{V}_{D\times N}\,\text{Softmax}\!\left(\frac{K^{\top}Q}{\sqrt{D}}\right)}, \qquad \frac{K^\top Q}{\sqrt{D}} \in \mathbb{R}^{N\times N}
$$

with the three projections of the input each formed as an affine map:

$$
\underbrace{V}_{D\times N} = W_V X + \underbrace{\beta_V}_{D\times1}\underbrace{\mathbf{1}^{\top}}_{1\times N}, \qquad
Q = W_Q X + \beta_Q \mathbf{1}^{\top}, \qquad
K = W_K X + \beta_K \mathbf{1}^{\top}.
$$

Writing $A$ for the softmax output gives the compact form

$$
\boxed{\underbrace{\text{SA}_1(X)}_{D\times N} = \underbrace{V}_{D\times N}\,\underbrace{A}_{N\times N}}
$$

with the naming: $A$ = **attention matrix**, $V$ = **values**, $Q$ = **queries**, $K$ = **keys**. The $1/\sqrt{D}$ factor keeps the logits' scale independent of the embedding dimension, so the softmax does not saturate as $D$ grows.

![[03 - Attachments/ATML_L3_attention_matrix.png|600]]

*Single-head self-attention: $K^\top Q$ gives the $N\times N$ score matrix, column-wise softmax gives $A$, and $VA$ gives $SA_1(X)$.*

### 2.9 Attention as Query–Key–Value Retrieval

The mechanical formula above has an intuitive reading, developed step by step.

The **query** $\underline{q}$ answers: *what is the most important information in $X$, given the current prediction context?* Each **key** $\underline{k}_\ell$, $\ell = 1, \ldots, N$, is scored against the query to produce a scalar,

$$
\text{score}(\underline{k}_\ell, \underline{q}) = e_\ell \in \mathbb{R}^{1\times1},
$$

which is the **weight for the $\ell$-th value** $\underline{v}_\ell$. Collecting all $N$ scores gives the **attention score** vector $\underline{e} = [e_1, e_2, \ldots, e_N]^\top$. This vector is then **redistributed into $[0,1]$** by the **align function** (the softmax):

$$
\underline{a} = \text{align}(\underline{e}).
$$

The output is the weighted sum of values — the **context vector**:

$$
\boxed{\underline{c} = \sum_{\ell=1}^{N} a_\ell\, \underline{v}_\ell = a_1\underline{v}_1 + a_2\underline{v}_2 + \cdots + a_N\underline{v}_N}
$$

which **contains a summary of the information that is important to $\underline{q}$**. Taking the score function to be the inner product,

$$
\text{score}(\cdot) = \underbrace{\underline{q}^{\top}}_{\text{query}}\,\underbrace{\underline{k}}_{\text{key}},
$$

recovers exactly the $K^\top Q$ score matrix of Section 2.8, and $A = \text{Softmax}(K^\top Q)$ is just the align function applied column-wise. Repeating this for all $N$ queries stacks the context vectors side by side:

$$
\text{SA}_1(X) = [\,\underline{c}_1 \;\; \underline{c}_2 \;\; \cdots \;\; \underline{c}_N\,] \in \mathbb{R}^{D\times N}.
$$

Geometrically, the notes draw three fans of vectors — the queries $\{q_1, \ldots, q_N\}$, the keys $\{K_1, \ldots, K_N\}$, and the values $\{V_1, \ldots, V_N\}$ — and pick out one query $q_1$. Its inner products against every key, $\langle q_1, K_1\rangle = a_1, \ldots, \langle q_1, K_N\rangle = a_N$, become the mixing coefficients of the values: $\underline{c}_1 = a_1 V_1 + \cdots + a_N V_N$.

*Intuition.* The three projections implement a soft dictionary lookup. A hard lookup matches a query against keys, finds the one that matches, and returns its value; attention instead scores the query against every key, normalizes the scores into weights, and returns the weighted average of all values. The separation of $K$ from $V$ is what makes this more than a similarity computation: a token advertises one thing (its key, *what I can answer*) and contributes another (its value, *what I supply if asked*), so a token may be highly retrievable without its retrieved content resembling the query. Three consequences follow, and each is taken up later. The weights are **input-dependent**, unlike a convolution's fixed kernel — the contrast [[ATML_Lecture_4 - GANs, and Inductive Biases]] develops as the locality-versus-global-context tradeoff. Every token attends to every other regardless of position, so the receptive field is global at the first layer but the operation is permutation-equivariant and thus blind to order, which is why positional embeddings are not optional (Lecture 4 Section 6.4). And the $N \times N$ score matrix is what makes the cost quadratic in sequence length.

![[03 - Attachments/ATML_L3_qkv_retrieval.png|600]]

*A query scores every key, the scores are normalized into weights $\underline{a}$, and the output $\underline{c}$ is the weighted sum of values.*

![[03 - Attachments/ATML_L3_qkv_vectors.png|600]]

*The same retrieval geometrically: inner products of $q$ with each $k_i$ give $a_i$, which weight the value vectors.*

### 2.10 The Block-Matrix / Kronecker View of Self-Attention

Stacking the context vectors into one long column of length $DN$ makes the linear-algebraic structure of attention explicit. The stacked output is a block matrix–vector product:

$$
\underbrace{\begin{bmatrix} \underline{c}_1 \\ \underline{c}_2 \\ \vdots \\ \underline{c}_N \end{bmatrix}}_{DN \times 1}
=
\underbrace{\begin{bmatrix}
a_{11}I_D & a_{21}I_D & \cdots & a_{N1}I_D \\
a_{12}I_D & a_{22}I_D & \cdots & a_{N2}I_D \\
\vdots & \vdots & \ddots & \vdots \\
a_{1N}I_D & a_{2N}I_D & \cdots & a_{NN}I_D
\end{bmatrix}}_{\text{one } D\times D \text{ diagonal block per entry of } A}
\underbrace{\begin{bmatrix} V_1 \\ V_2 \\ \vdots \\ V_N \end{bmatrix}}_{DN\times 1}
$$

where each block is a scalar attention weight times the $D\times D$ identity — i.e. **$N$ diagonal matrices in each row and column**. This entire block matrix is a **Kronecker product**:

$$
\boxed{\underbrace{M}_{DN \times DN} = \underbrace{A^{\top}}_{N\times N} \otimes \underbrace{I_D}_{D\times D}}
$$

The interpretation is worth holding onto: **attention mixes across tokens but acts identically across embedding dimensions.** The $I_D$ factor in the Kronecker product says every feature dimension of a value vector receives the same attention weight — attention re-weights *which tokens* contribute, never *which features within a token*.

![[03 - Attachments/ATML_L3_block_matrix.png|600]]

*Stacking all outputs: $SA_1(X)$ as a block matrix of $N\times N$ diagonal blocks, i.e. $M = A^\top \otimes I_D$.*

---

## 3. Family 2 — Contrastive Models

The discriminative family requires a label for every example, which limits it to the fraction of data that has been annotated. Contrastive models remove that requirement by manufacturing the supervision instead: if two views of the same image must be labelled "same" and views of different images "different," then the comparison structure of the batch replaces the annotation. The resulting objective still targets mutual information, as Section 2.2 did, but between two views rather than between a representation and a label.

### 3.1 Positive and Negative Augmentations

Contrastive models **learn by comparing among input samples**, and require only unlabeled data $\{X_i\}_{i=1}^N$. The construction is:

- The original image $x$ is the **anchor**.
- Applying a label-preserving transformation — e.g. **rotating** the cat image — produces a **positive** $x^+$, forming a **+ve pair** with the anchor. This is **positive augmentation**, and there are **many types** of it (crops, colour jitter, blur, and so on).
- Applying a transformation that destroys the semantic content — "destroy cat" — or simply taking a different image, produces a **negative** $x^-$, forming a **−ve pair**. This is **negative augmentation**.

In practice one takes a **batch of images containing one cat**: that cat's augmented view is $x^+$, and *every other image in the batch* serves as a negative $x^-$. This is why contrastive learning benefits so much from large batches — the batch supplies the negatives for free.

The requirement is stated compactly: positive pairs are of **"similar"** inputs, negative pairs of **"dissimilar"** inputs. Positives may be **two views of the same input** (augmentations) **or two modalities of the same input** (image and its caption) — the latter is what makes CLIP a contrastive model.

### 3.2 The Contrastive Pipeline and the InfoNCE Loss

**Goal:** *bring $x$ and $x^+$ together in latent space, and push $x$ and $x^-$ apart.*

All three inputs pass through the **same** encoder — this weight sharing is essential, since otherwise the embeddings would not be comparable:

$$
x,\; x^{+},\; x^{-} \;\xrightarrow{\;\text{Vision Encoder } g_\theta(\cdot)\;}\; z,\; z^{+},\; z^{-} \;\longrightarrow\; \text{Contrastive Loss}.
$$

The loss is **InfoNCE**:

$$
\boxed{\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\big(\text{sim}(z_1, z_1^{+})\big)}{\exp\big(\text{sim}(z_1, z_1^{+})\big) + \sum_{k=2}^{N} \exp\big(\text{sim}(z_1, z_k^{-})\big)}}
$$

with the terms:

- $\text{sim}(z_1, z_1^+)$ = similarity between the two embeddings, e.g. **cosine similarity** $\dfrac{\langle z_1, z_1^{+}\rangle}{\lVert z_1\rVert\,\lVert z_1^{+}\rVert}$.
- $z_1, z_1^{+}$ = positive-pair embeddings.
- $z_2^{-}, \ldots, z_K^{-}$ = negative-pair embeddings, for $k = 2, 3, \ldots, N$ (i.e. $k \neq 1$).
- $x_1$ = the **anchor**.

Structurally this is a softmax cross-entropy over $N$ "classes," where the correct class is the positive and the incorrect classes are the negatives — which is why the same log-sum-exp geometry from Lecture 1 applies: the numerator pulls $z_1$ toward $z_1^+$, and the denominator pushes it away from every negative.

![[03 - Attachments/ATML_L3_contrastive_pipeline.png|682]]

*Two views $x_1, x_1^+$ and negatives $x_k$ pass through the same encoder $g_\theta$; the contrastive loss compares $z_1$ with $z_1^+$ against the $z_k^-$.*

### 3.3 InfoNCE as a Lower Bound on Mutual Information

The reason InfoNCE is the *right* loss rather than merely a plausible one is that it provides a **tractable lower bound on the mutual information** between the two views:

$$
\boxed{I(z_1, z_1^{+}) \;\ge\; \log(N) - \mathcal{L}_{\text{InfoNCE}}}
$$

Since $\log N$ is fixed for a given batch size, **minimizing $\mathcal{L}_{\text{InfoNCE}}$ maximizes the lower bound on $I(z_1, z_1^+)$**. Two levers tighten the bound:

- **Increasing $N$** (larger batches → more negatives → higher $\log N$ ceiling).
- **Decreasing $\mathcal{L}_{\text{InfoNCE}}$** (better representations).

The bound is drawn in the notes as the true $I(z_1, z_1^+)$ curve sitting above the $\log N - \mathcal{L}_{\text{InfoNCE}}$ curve, with the gap closing as either lever is pulled. This is the concrete realization of the family goal from Section 1.2: $I(x, x^+) = \max$ and $I(x, x^-) = \min$.

*Intuition.* The bound is best read as the statement that InfoNCE poses mutual information estimation as a classification problem. If the encoder can reliably identify which of $N$ candidates is the true partner of the anchor, the two views must share substantial information; if it cannot do better than chance, they share none. Chance performance costs $\log N$, which is why that term appears as the ceiling — and why the bound saturates there, making a large batch a prerequisite for estimating large mutual information at all. This ceiling is the structural reason contrastive methods are batch-hungry in a way the other two families are not.

The choice of positives carries the entire inductive bias, and it is an active one rather than a technicality. Every transformation declared label-preserving is an assertion that the information it destroys is task-irrelevant, so colour jitter as a positive augmentation instructs the encoder to discard colour. This is precisely the invariance-driven route to a tighter bound named in Lecture 1 Section 5.4, and it is also the failure mode: an augmentation destroying task-relevant information produces a representation that is invariant to something the downstream task needed. The same tension reappears in [[ATML_Lecture_5_2 - Domain Generalization]] under the name of shortcut learning.

![[03 - Attachments/ATML_L3_infonce_bound.png|697]]

*$I(z_1, z_1^+)$ sits above $\log N - \mathcal{L}_{\text{InfoNCE}}$; the gap closes as $N$ grows or the loss falls.*

### 3.4 CLIP — Contrastive Language–Image Pretraining

**CLIP** learns a mapping **from images and texts into a shared embedding space**. Two separate encoders run in parallel:

$$
\text{cat image} \;\longrightarrow\; \boxed{\text{Vision Encoder (VE)}} \;\longrightarrow\; Z_I
$$
$$
\text{“This is a cat”} \;\longrightarrow\; \boxed{\text{Text Encoder (TE)}} \;\longrightarrow\; Z_T
$$

and the contrastive loss operates *between* the two, aligning $Z_I$ and $Z_T$.

For a batch of $N$ image–text pairs, stacking the $N$ image embeddings against the $N$ text embeddings produces an $N \times N$ **similarity matrix** in the shared embedding space, with entries $\langle I_i, T_j\rangle$. The **diagonal entries are the true (positive) pairs**; every off-diagonal entry is a negative.

This matrix supports **two classification tasks simultaneously**, one per axis:

- Reading along a row: *to which text description does this image belong?*
- Reading down a column: *to which image does this text description belong?*

Hence the two loss directions $\mathcal{L}_{\text{I2T}}$ and $\mathcal{L}_{\text{T2I}}$, combined symmetrically:

$$
\mathcal{L}_{\text{InfoNCE}} = \tfrac{1}{2}\big(\mathcal{L}_{\text{T2I}} + \mathcal{L}_{\text{I2T}}\big).
$$

### 3.5 CLIP Training

**Dataset:** $\{I_i, T_i\}_{i=1}^{N}$ — image–text pairs.

**Encoders:** a vision encoder $f_\theta(\cdot) \to \mathbb{R}^{d\times1}$ and a text encoder $g_\psi(\cdot) \to \mathbb{R}^{d\times1}$, both mapping into the same $d$-dimensional space.

**Forward pass.** Encode, then $\ell_2$-normalize so that inner products become cosine similarities:

$$
v_i = f_\theta(I_i), \quad t_i = g_\psi(T_i) \;\;\forall i, \qquad\qquad v_i \leftarrow \frac{v_i}{\lVert v_i\rVert}, \quad t_i \leftarrow \frac{t_i}{\lVert t_i \rVert}.
$$

**Embedding / logit matrix.** Stack the normalized embeddings into matrices $V$ and $T$ and form the scaled similarity matrix:

$$
L = \text{scale} \times V T^{\top} \in \mathbb{R}^{N\times N}, \qquad L_{ij} = \text{scale} \times v_i^{\top} t_j.
$$

The `scale` factor is the learnable inverse temperature; without it the cosine similarities are confined to $[-1,1]$ and the softmax is too flat to train through.

**Losses.** Two symmetric cross-entropy terms, one per softmax direction:

$$
\boxed{\mathcal{L}_{\text{I2T}} = \frac{1}{N}\sum_{i=1}^{N} -\log \frac{e^{L_{ii}}}{\sum_{j=1}^{N} e^{L_{ij}}}} \qquad \text{(row softmax: for each image } I_i \text{, pick the correct caption)}
$$

$$
\boxed{\mathcal{L}_{\text{T2I}} = \frac{1}{N}\sum_{j=1}^{N} -\log \frac{e^{L_{jj}}}{\sum_{i=1}^{N} e^{L_{ij}}}} \qquad \text{(column softmax: for each caption } j \text{, pick the correct image)}
$$

In both, **the diagonal entries are the true pairs**. The final objective is their average:

$$
\boxed{\mathcal{L}_{\text{Total}} = \tfrac{1}{2}\left(\mathcal{L}_{\text{I2T}} + \mathcal{L}_{\text{T2I}}\right)}
$$

**Backpropagation:** $\nabla \mathcal{L}_{\text{Total}}$ updates both $\theta$ and $\psi$ jointly.

![[03 - Attachments/ATML_L3_clip_training.png|620]]

*CLIP: text and image encoders fill an $N\times N$ similarity matrix whose diagonal holds the true pairs; rows and columns are each a classification task.*

### 3.6 Zero-Shot Inference with CLIP

Because CLIP has aligned images and text in one space, classification requires no classifier head at all — only a set of candidate label strings.

1. Pass the test image through the image encoder to obtain $I_{\text{Test}}$.
2. Take the label set (Cat, Bird, …, Dog) and wrap each label in a template — **"A photo of a \<label\>"** — a step explicitly labelled **prompt engineering** in the notes. Pass each through the text encoder to obtain $T_1, \ldots, T_N$.
3. Compute the similarities $\langle I_{\text{Test}}, T_1\rangle, \ldots, \langle I_{\text{Test}}, T_N\rangle$ and take the largest.

The result is drawn as a bar chart over the candidate labels, with the tallest bar giving the prediction. The classifier is defined entirely by the choice of label strings, which is what makes it **zero-shot**: the model was never trained on this label set.

![[03 - Attachments/ATML_L3_clip_zero_shot.png|600]]

*Zero-shot: embed "A photo of a ⟨label⟩" for every label, score the test image against each, and take the highest bar.*

---

## 4. Family 3 — Generative Models

The first two families learn $z$ in service of a prediction or a comparison, and in both cases $z$ need only retain what that task requires. The generative family sets the most demanding target of the three: $z$ must suffice to *reconstruct* $x$, which by Lecture 1's terms is sufficiency taken to its extreme. The objective is correspondingly the forward KL of Lecture 2 Section 5.3 between the data distribution and the model — and, as Section 4.2 shows, that reduces to maximum likelihood without any further assumption.

### 4.1 Setup and the Modelling Goal

Data is drawn from an underlying process, $P_{\text{data}} \sim \mathcal{D}: \{\vec{x}_i\}_{i=1}^N$, and can be of many kinds: **images, videos, text, actions (robotics), tabular data**. The dataset may also come with paired side information, $\{x_i, y_i\}_{i=1}^N$, where crucially $y_i$ is **not a label** but rather **extra information / a condition about what to generate** — this is what distinguishes conditional generation from classification.

$P_{\text{data}}$ is **complex**, drawn in the notes as an amorphous blob. A deep neural network with parameters $\phi$ is trained so that its induced distribution approximates it:

$$
x_i \;\longrightarrow\; \boxed{\text{DNN model, params } \phi} \;\longrightarrow\; P_\phi \approx P_{\text{data}},
$$

after which **sampling** $x_i \sim P_\phi$ should produce outputs that **resemble the data**. The scale of the problem is worth stating: a single image has **thousands of dimensions**, so **the model must capture the dependencies between all of those dimensions** — a pixel is only meaningful relative to its neighbours.

The training objective is to minimize a distance between the two distributions,

$$
\min_{\phi} \; \text{Dist}(P_{\text{data}}, P_\phi), \qquad\text{concretely}\qquad \min_{\phi}\; D_{\mathrm{KL}}(P_{\text{data}} \,\|\, P_\phi),
$$

drawing on the whole toolkit of Lecture 2.

### 4.2 From KL Minimization to Maximum Likelihood

Expanding the forward KL divergence separates it into a term that does not depend on $\phi$ and one that does:

$$
\min_\phi D_{\mathrm{KL}} = \mathbb{E}_{P_{\text{data}}}\big[\log P_{\text{data}}\big] - \mathbb{E}_{P_{\text{data}}}\big[\log P_\phi\big]
$$

$$
= \underbrace{H(P_{\text{data}})}_{\text{constant w.r.t. } \phi} - \underbrace{\big[\text{average calculated from data}\big]}_{\text{depends on } \phi}
$$

Because the first term is the (negative) entropy of the data distribution and carries no $\phi$ dependence, minimizing the KL is **equivalent to maximizing the second term**:

$$
\equiv \max_\phi \; \mathbb{E}_{P_{\text{data}}}\big[\log P_\phi(x)\big] \;=\; \max_\phi \; \frac{1}{N}\sum_{i=1}^{N} \log P_\phi(x_i).
$$

$$
\boxed{\min_\phi D_{\mathrm{KL}}(P_{\text{data}}\|P_\phi) \;\equiv\; \max_\phi \frac{1}{N}\sum_{i=1}^{N}\log P_\phi(x_i)}
$$

The final step is again the Monte Carlo substitution from Lecture 2: the population expectation under $P_{\text{data}}$ becomes a sample average over the dataset. This is exactly **maximum likelihood estimation** — it was not assumed, it fell out of minimizing forward KL.

*Intuition.* The identity is worth noting in both directions, since it explains an empirical fact by a theoretical one. Maximum likelihood *is* forward KL, and forward KL is mode-covering by Lecture 2 Section 5.3, so a likelihood-trained generative model is structurally obliged to place mass over every region the data occupies — including regions it lacks the capacity to represent sharply. When capacity is insufficient, covering is achieved by averaging, and averaged images are blurry. The characteristic blurriness of VAE samples is therefore not an implementation defect but a property of the objective, which is precisely why [[ATML_Lecture_4 - GANs, and Inductive Biases]] obtains sharpness by abandoning likelihood altogether.

The same statement in the professor's notation, with $P(x)$ the true underlying distribution and $P_\theta(x)$ the model assumption, is: we want $P_\theta(x) \approx P(x)$, which we obtain by **assuming a structure for $P_\theta(x)$** — e.g. $\mathcal{N}(\mu, \sigma^2)$ — **and then estimating** $\theta = [\mu, \sigma^2]$.

### 4.3 What Maximizing the Log-Likelihood Actually Buys

$$
\boxed{\max_\theta \; \log P_\theta(x)}
$$

Three consequences are spelled out, and together they define what "a good generative model" means:

- Values of $x$ **which look like images should get high probabilities**.
- Values of $x$ **which look like noise should get low probabilities**.
- The model should **produce more $x$ which are most likely to be those already in the data**.

The third point contains the central tension of generative modelling: an objective that rewards assigning high probability to the training data alone would be satisfied by memorization, so generalization to *new but plausible* samples has to come from the model's structure and inductive bias rather than from the likelihood objective itself.

### 4.4 The Concept of the Latent $z$

**All generative models make use of a latent somehow.** The latent $z$ is a vector that **contains information about the generative process** — for MNIST (a dataset of handwritten digits) digits, that means factors such as:

- Digit identity ($0, 1, 2, \ldots, 9$).
- Angle between lines.
- Stroke width.
- Style properties, and so on.

The notes illustrate with a compositional constraint: the left half of a "5" cannot be paired with the right half of a "0" — the generative factors are not independent of one another, and a good latent has to encode which combinations are admissible.

**How do we get this information?** Two answers are given.

**(1) No simple interpretation.** Draw $z \sim \mathcal{N}(0, I)$ — pure randomness — and pass it through a neural network acting as a **functional map**:

$$
\underbrace{z}_{\text{random}} \;\longrightarrow\; \boxed{\text{Neural Network}} \;\longrightarrow\; \underbrace{x = \text{image}}_{\text{structured}}.
$$

The insight is that **randomness contains every possible structure inside it, if we are able to learn $g(z)$** — the network's job is to carve structure out of noise. A toy example makes this concrete: taking a uniform cloud of points $z$ and applying

$$
g(z) = \frac{z}{10} + \frac{z}{\lVert z\rVert}
$$

maps the cloud onto a **ring**. A simple deterministic function has turned an unstructured distribution into a structured one; a deep network does the same thing with far more expressive maps.

**(2) Use only the latents the data actually produced.** Rather than using all $z \sim \mathcal{N}(0,I)$, restrict attention to those **most likely to have produced the observed data**. This is the **posterior of $z$**:

$$
P(z \mid x) = \text{posterior of } z — \text{take } x \text{ and it gives us the distribution of } z.
$$

Route (1) leads to GANs (Lecture 4); route (2) leads to VAEs.

![[03 - Attachments/ATML_L3_latent_z.png|621]]

*The latent carries the generative factors (digit identity, stroke width, style); sampling all of $\mathcal{N}(0,I)$ through a learned map $g(z)$ versus using only the posterior $P(z\mid x)$.*

### 4.5 Three Sub-Families of Generative Models

| | **(i) Latent-Variable Models** | **(ii) Auto-Regressive Models** | **(iii) Implicit Models** |
|---|---|---|---|
| **Idea** | Introduce $z$ explicitly and marginalize it out ($x \to z \to x$) | Directly factorize the joint | Do not model $P_\theta(x)$ explicitly; learn only the sampling process — sample $P(x)$ via the latent |
| **Formula** | $P_\theta(x) = \int P_\theta(x\mid z)P(z)\,dz$ | $P_\theta(x) = \prod_{i=1}^{n} P(x_i \mid x_{<i})$ | $z \to x_T \to x_{T-1} \to \cdots \to x_0$ |
| **Examples** | VAE, Normalizing Flows | GPT (Generative Pre-trained Transformer, a decoder-only language model), PixelCNN (an autoregressive CNN that generates an image pixel by pixel) | GAN, Diffusion Models |

The trade-off across the three: latent-variable models give an interpretable latent but only a bound on the likelihood; auto-regressive models give an exact likelihood but slow sequential sampling; implicit models give fast, sharp sampling but no likelihood at all.

---

## 5. Variational Autoencoders

Section 4.5 classified the VAE as a latent-variable model whose likelihood $P_\theta(x) = \int P_\theta(x\mid z)P(z)\,dz$ is available only as an integral over every possible $z$ — and that integral is intractable in any realistic dimension. The whole of this section is the consequence: since the objective cannot be evaluated, it is replaced by a bound (Section 5.3), the bound requires sampling (Section 5.5), and sampling requires a device for differentiating through the sample (the reparameterization trick). Each complication traces back to that one intractable integral.

VAEs are used **primarily for images** in this course.

### 5.1 From AE to DAE to VAE

**Autoencoder (AE).** The plain autoencoder is

$$
X \;\longrightarrow\; \boxed{E} \;\xrightarrow{\;z\;}\; \boxed{D} \;\longrightarrow\; \hat{x} \;\longrightarrow\; \text{MSE}(x, \hat{x}),
$$

trained purely on reconstruction error. The problem appears when you look at the latent it produces: plotting $z_1$ against $z_2$ for MNIST gives a **highly unstructured latent** — clumped, with large empty regions and no consistent geometry. Sampling a random point from such a space produces garbage, because most of the space corresponds to no training example at all. Hence: **we need to regularize this space.**

**Denoising Autoencoder (DAE).** One partial fix is to **add noise to $x$**, producing a corrupted input $\tilde{x}$, and require the network to reconstruct the clean $x$ from it. This forces the encoder to be robust to perturbations, but it still does not guarantee a *globally* smooth latent.

**Why smoothness matters.** For **generation**, we need a **smooth latent**: if we want to sample new digits by drawing points from the latent space, then every point we might draw must decode to something plausible, and nearby points must decode to similar things. That is exactly the **smoothness** property from the representation theory of Lecture 1.

### 5.2 Prior, Posterior, and the VAE Architecture

The VAE achieves smoothness by *imposing* a distribution on the latent rather than hoping one emerges.

**Prior.** Assume $z$ has the prior distribution

$$
P_\theta(z) \sim \mathcal{N}(0, I) \qquad \text{(isotropic Gaussian)}.
$$

Note the key property: **$P(z)$ contains no information about $X$.** It is a fixed, structureless target — which is precisely what makes it safe to sample from at generation time.

**Posterior.** The encoder does not output a point; it outputs the *parameters of a distribution*. We say $z$ has another distribution, the **posterior**:

$$
z \sim q_\psi(z \mid x) \sim \mathcal{N}(\mu_e, \sigma_e^2 I),
$$

and the training pressure is to **minimize the distance between posterior and prior**:

$$
D_{\mathrm{KL}}\big(q_\psi(z\mid x) \,\|\, P(z)\big).
$$

**Architecture.** Putting it together, with the encoder *learning a distribution* rather than a point:

$$
X \;\longrightarrow\; \boxed{\mathcal{E}} \;\longrightarrow\; \underbrace{\mu_e,\; \sigma_e^2}_{\text{computed parameters}} \;\xrightarrow{\;q_\psi\;}\; z \;\longrightarrow\; \boxed{D_\theta} \;\longrightarrow\; \hat{x} \;\longrightarrow\; \text{MSE}_{\psi,\theta}(x, \hat{x})
$$

with the decoder defining $P_\theta(x \mid z)$. The parameter groups are:

$$
\psi: \{\psi_e,\, \mu_e,\, \Sigma_e\} \quad \text{(encoder)}, \qquad\qquad \theta: \{\theta_d,\, \mu_d,\, \Sigma_d\} \quad \text{(decoder)}.
$$

**The three modelling assumptions**, stated explicitly:

1. $P_\theta(z) \sim \mathcal{N}(0, I)$ — the **prior on $z$**.
2. $P_\theta(z \mid x) \approx q_\psi(z\mid x) \sim \mathcal{N}(\mu_e, \Sigma_e)$ — the **posterior on $z$**, approximated by the encoder because the true posterior is intractable.
3. $P_\theta(x \mid z) \sim \mathcal{N}(\mu_d, \Sigma_d)$ — the decoder's output distribution.

![[03 - Attachments/ATML_L3_vae_architecture.png|620]]

*VAE: encoder $q_\psi(z\mid x)$ maps x-space to z-space under the prior $P_\theta(z)$; decoder $P_\theta(x\mid z)$ maps back.*

### 5.3 The ELBO

The central claim, which Section 5.4 proves, is that maximizing the intractable log-likelihood is equivalent to maximizing a tractable surrogate called the **Evidence Lower BOund (ELBO)**:

$$
\boxed{\max_\theta \log P_\theta(x) \iff \max_{\theta,\psi} \; \mathcal{L}_{\theta,\psi}(x) = \mathbb{E}_{q_\psi}\!\left[\log \frac{P_\theta(x,z)}{q_\psi(z\mid x)}\right]}
$$

Expanding $P_\theta(x,z) = P_\theta(x\mid z)P_\theta(z)$ and splitting the logarithm gives the form actually used in practice:

$$
\boxed{\mathcal{L}_{\theta,\psi}(x) = \underbrace{\mathbb{E}_{q_\psi}\big[\log P_\theta(x\mid z)\big]}_{\text{how good the decoder is}} \;-\; \underbrace{D_{\mathrm{KL}}\big(q_\psi(z\mid x)\,\|\,P_\theta(z)\big)}_{\text{how good the encoder is}}}
$$

The two terms have clean names and clean meanings:

- The first is the **reconstruction** term. A **good decoder produces a good image of $x$ if we feed it $z$.** Under the Gaussian decoder assumption, maximizing this log-likelihood reduces to minimizing a squared error, which is why it appears in the in-class version of the loss as an MSE (mean squared error) term.
- The second is the **prior-matching** term. A **good encoder turns $x \to z$ such that $z \sim \mathcal{N}(0,I)$** — i.e. the aggregate of the encoder's outputs looks like the prior we intend to sample from later.

The ELBO is therefore **maximized when $\mathbb{E}_{q_\psi}[P_\theta(x\mid z)]$ is maximal and $D_{\mathrm{KL}}(\cdot) = 0$.** Written as a loss to be *minimized* (flipping the sign), this is the form used on the board:

$$
\mathcal{L}_{\text{ELBO}}(\theta, \psi) = D_{\mathrm{KL}}\big(q_\psi(z\mid x),\, P(z)\big) \;+\; \text{MSE}_{\psi,\theta}(X, \hat{x}).
$$

*Intuition.* The two terms are in direct opposition, and reading them as a tug-of-war explains most VAE behaviour. Reconstruction wants the code to be as informative about this particular $x$ as possible, which favours posteriors that are narrow and widely separated — in the limit, a lookup table. Prior-matching wants every posterior to resemble $\mathcal{N}(0,I)$, which favours posteriors that are broad and overlapping and therefore carry little information about which $x$ produced them. This is the information bottleneck of Lecture 1 Section 3 realized as two concrete loss terms, and the overlap the KL term enforces is exactly what makes the latent smooth enough to sample from: neighbouring codes must decode to plausible images because neighbouring codes are routinely produced by different inputs during training.

The two extremes of the tug-of-war are the two standard failure modes. Let the KL term dominate and every posterior collapses onto the prior, the code becomes independent of the input, and the decoder ignores it — posterior collapse, Section 5.6. Let reconstruction dominate and the VAE degenerates into the plain autoencoder of Section 5.1, with a latent full of holes that cannot be sampled. The explicit weight $\beta$ that [[ATML_Lecture_5_2 - Domain Generalization]] places on the KL term is simply this tradeoff made adjustable.

### 5.4 Proof That the ELBO Lower-Bounds the Log-Likelihood

Start from the observation that $\log P_\theta(x)$ does not depend on $z$, so taking an expectation over any distribution of $z$ leaves it unchanged:

$$
\mathbb{E}_{z\sim q_\psi(z\mid x)}\big[\log P_\theta(x)\big] = \log P_\theta(x).
$$

Applying the definition of conditional probability, $P_\theta(z\mid x) = P_\theta(x,z)/P_\theta(x)$, lets us rewrite the marginal as a ratio inside the expectation:

$$
\log P_\theta(x) = \mathbb{E}_{q_\psi}\!\left[\log \frac{P_\theta(x,z)}{P_\theta(z\mid x)}\right].
$$

Multiplying and dividing by the variational posterior $q_\psi(z\mid x)$ — the one step that introduces the encoder into an otherwise decoder-only identity — and then splitting the logarithm of the product into a sum of logarithms:

$$
= \mathbb{E}_{q_\psi}\!\left[\log\!\left(\frac{P_\theta(x,z)}{q_\psi(z\mid x)} \cdot \frac{q_\psi(z\mid x)}{P_\theta(z\mid x)}\right)\right]
$$

$$
= \underbrace{\mathbb{E}_{q_\psi}\!\left[\log \frac{P_\theta(x,z)}{q_\psi(z\mid x)}\right]}_{= \;\text{ELBO}\; = \;\mathcal{L}_{\theta,\psi}} \;+\; \underbrace{\mathbb{E}_{q_\psi}\!\left[\log \frac{q_\psi(z\mid x)}{P_\theta(z\mid x)}\right]}_{=\; D_{\mathrm{KL}}(q_\psi \,\|\, P_\theta) \;\ge\; 0}
$$

The second term is, by definition, a KL divergence, and KL divergences are non-negative. Therefore

$$
\boxed{\mathcal{L}_{\theta,\psi} \;\le\; \log P_\theta(x), \qquad \text{with gap exactly } D_{\mathrm{KL}}\big(q_\psi(z\mid x)\,\|\,P_\theta(z\mid x)\big)}
$$

The notes draw this as two curves — the true $\log P_\theta(\cdot)$ above, the ELBO below, and a vertical arrow between them labelled $D_{\mathrm{KL}}(q_\psi\|P_\theta)$. Pushing the ELBO up either improves the true log-likelihood or closes the approximation gap; either way progress is made. Hence

$$
\max \text{ELBO} \iff \max \log P_\theta(x),
$$

and training solves

$$
\{\theta, \psi\} = \operatorname*{arg\,max}_{\theta,\psi} \sum_{x_i \in \text{Data}} \text{ELBO}(x_i) = \mathcal{L}_{\theta,\psi}(x_i).
$$

The question this leaves open — and which Section 5.5 answers — is: what are $\nabla_\theta \mathcal{L}_{\theta,\psi}(x_i)$ and $\nabla_\psi \mathcal{L}_{\theta,\psi}(x_i)$?

![[03 - Attachments/ATML_L3_elbo_bound.png|500]]

*The ELBO lies below $\log P_\theta(x)$ everywhere; the gap is $D_{\text{KL}}(q_\psi \| P_\theta)$.*

### 5.5 Gradients of the ELBO and the Reparameterization Trick

**The decoder gradient is easy.** Writing the ELBO as $\mathbb{E}_{q_\psi}[\log P_\theta(x,z) - \log q_\psi(z\mid x)]$ and differentiating with respect to $\theta$, the expectation's distribution $q_\psi$ does not depend on $\theta$, so the gradient passes inside; and the second term does not depend on $\theta$ at all, so it vanishes:

$$
\nabla_\theta \mathcal{L}_{\theta,\psi} = \mathbb{E}_{q_\psi}\!\left[\nabla_\theta \log P_\theta(x,z) - \underbrace{\nabla_\theta \log q_\psi(z\mid x)}_{= \, 0}\right] = \mathbb{E}_{q_\psi}\big[\nabla_\theta \log P_\theta(x,z)\big],
$$

which is estimated with a single Monte Carlo sample:

$$
\nabla_\theta \mathcal{L}_{\theta,\psi} \approx \nabla_\theta \log P_\theta(x, z^{i}), \qquad z_i \sim q_\psi(z\mid x).
$$

**The encoder gradient is the problem.** Writing $f_1(z) = \log P_\theta(x,z) - \log q_\psi(z\mid x)$, we need

$$
\nabla_\psi \mathcal{L}_{\theta,\psi} = \nabla_\psi \,\mathbb{E}_{q_\psi}\big[f_1(z)\big],
$$

which is **not possible to evaluate directly, because the expectation is taken with respect to $q_\psi(\cdot)$, which is itself a function of $\psi$.** The gradient cannot simply move inside the expectation.

**The reparameterization trick.** The fix is to **isolate the randomness out of $z$ and make $z$ a deterministic function of $\psi$**:

$$
\boxed{z = \mu_e + \sigma_e^2 \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)}
$$

where $\odot$ is the elementwise product. **The randomness is now isolated in $\epsilon$**, whose distribution carries no $\psi$ dependence at all. This lets the gradient and the expectation commute:

$$
\boxed{\nabla_\psi \mathcal{L}_{\theta,\psi} = \nabla_\psi\,\mathbb{E}_{q_\psi}\big[f_1(z)\big] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\big[\nabla_\psi f_1(z)\big]}
$$

> **[Possible source inconsistency]** — the notes write $z = \mu_e + \sigma_e^2 \odot \epsilon$, using the *variance* $\sigma_e^2$ as the scale factor. The standard reparameterization uses the *standard deviation*, $z = \mu_e + \sigma_e \odot \epsilon$, which is what makes $z \sim \mathcal{N}(\mu_e, \sigma_e^2)$ come out correctly. The professor's form is reproduced as given; in implementation, use $\sigma_e$ (typically computed as $\exp(\tfrac12 \log\sigma_e^2)$ for numerical stability).

*Intuition.* The difficulty is that a sampling operation has no derivative: the encoder's parameters determine *which distribution is sampled from*, and there is no path along which a small change in $\psi$ propagates to the drawn value. The trick relocates the randomness rather than removing it. Instead of asking the encoder to emit a random $z$, it draws a fixed, parameter-free $\epsilon \sim \mathcal{N}(0,I)$ and has the encoder emit a deterministic affine map of it — so $z$ becomes an ordinary differentiable function of $\mu_e$ and $\sigma_e$ with the noise entering as an input rather than as an operation. The distribution of $z$ is unchanged; only the computational graph is, and now a gradient path exists from $z$ back to $\psi$.

Two points generalize beyond VAEs. The trick applies to any distribution admitting a *location–scale* form — a sample expressible as a deterministic function of the parameters and parameter-free noise — which covers the Gaussian and a few others, but not discrete latents, and this is precisely why discrete latent-variable models require different estimators. And the mechanism is exactly the one justified by the law of the unconscious statistician from Lecture 1 Section 8: an expectation under $q_\psi(z)$ is rewritten as an expectation under a fixed $P(\epsilon)$ of a function that carries the $\psi$ dependence, which is what permits the gradient and the expectation to commute.

**The full forward pass with reparameterization:**

$$
x \to \boxed{\text{Encoder}} \to \mu_e, \sigma_e^2 \;\Rightarrow\; q_\psi(z\mid x)\sim\mathcal{N}(\mu_e, \sigma_e^2); \quad \epsilon \sim \mathcal{N}(0,I) \;\Rightarrow\; z = \mu_e + \sigma_e^2\odot\epsilon
$$
$$
z \to \boxed{\text{Decoder}} \to \mu_d, \sigma_d^2 \;\Rightarrow\; P_\theta(x\mid z) = \mathcal{N}(\mu_d, \sigma_d^2); \quad \hat{x} \sim \mu_d + \sigma_d^2\odot\epsilon
$$

![[03 - Attachments/ATML_L3_reparameterization.png|627]]

*Reparameterization: the encoder outputs $\mu_e, \sigma_e^2$, noise $\epsilon \sim \mathcal{N}(0,I)$ enters from outside, and $z = \mu_e + \sigma_e \odot \epsilon$ feeds the decoder.*

### 5.6 Failure Modes: Posterior Collapse and Blurry Samples

**Posterior collapse.** If $D_{\mathrm{KL}} = 0$, the posterior has become *identical* to the prior — meaning the encoder outputs $\mathcal{N}(0,I)$ regardless of the input, so $z$ carries no information about $x$ whatsoever. The cause given is that the **decoder is too powerful**: it learns to ignore $z$ and **starts generating directly from the data** (i.e. from what it has memorized about the marginal), because it can achieve reasonable reconstruction without using the latent at all. The KL term, which was supposed to regularize, has been trivially satisfied.

**Blurry images.** Two causes are given, and they compound:

1. **MSE is an averaging operation.** When several plausible reconstructions exist, the squared-error-optimal output is their *mean*, which is a blur rather than any one of them.
2. **The very smooth prior.** Forcing the latent to match an isotropic Gaussian makes nearby latents decode to similar images, which is exactly what we wanted for generation — but the same smoothness limits how sharply the decoder can commit to fine detail.

This blurriness is the standard contrast drawn against GANs, which produce sharp samples for exactly the opposite reasons (Lecture 4).

### 5.7 Generation

Once training is done, **throw away the encoder.** Generation needs only the prior and the decoder:

$$
z \sim \mathcal{N}(0, I) \;\longrightarrow\; \boxed{D} \;\longrightarrow\; \boxed{3}
$$

Draw a latent from the prior, push it through the decoder, and read off a novel sample — here, a generated digit "3". This is the payoff of the prior-matching term: because training forced the encoder's outputs to look like $\mathcal{N}(0,I)$, a fresh draw from $\mathcal{N}(0,I)$ lands in a region of latent space the decoder knows how to handle.

---

## Looking Ahead

The three families share one structure: each targets an intractable information-theoretic quantity and optimizes a bound on it instead, and each family's characteristic weakness is a property of the bound rather than of the implementation.

| Family | Target | Tractable surrogate | Characteristic weakness |
|---|---|---|---|
| Discriminative | $\max I(Y;z)$ | cross-entropy (§2.3) | needs labels; learns only what the label requires |
| Contrastive | $\max I(x, x^+)$ | InfoNCE, bound $\log N - \mathcal{L}$ (§3.3) | bound ceilinged by batch size; augmentations dictate the invariances |
| Generative | $\max \log P_\theta(x)$ | ELBO (§5.3) | forward KL is mode-covering, hence blurry samples |

[[ATML_Lecture_4 - GANs, and Inductive Biases]] continues the generative family with GANs, which abandon likelihood altogether and thereby trade the blurriness of the last row for sharpness plus mode collapse — the reverse-KL half of the dichotomy tabulated in Lecture 2 Section 5.4. It then turns to inductive bias, for which the CNN and Transformer mechanics of Sections 2.5–2.10 are the prerequisite.

---

## Numerical Problems

**Problem 1 — CNN output dimensions.** An input of shape $224 \times 224 \times 3$ passes through a convolutional layer with $C_2 = 64$ kernels of size $F = 7$, padding $P = 3$, and stride $S = 2$. Give the output shape. Then give the shape after a second layer with $F = 3$, $P = 1$, $S = 2$, and $C_2 = 128$.

*Solution.* Applying the formula of Section 2.5, $H_2 = \lfloor (H_1 + 2P - F)/S\rfloor + 1$, the first layer gives $\lfloor (224 + 6 - 7)/2 \rfloor + 1 = \lfloor 223/2\rfloor + 1 = 111 + 1 = 112$, and identically for the width, with the channel count set by the kernel count.

$$
\boxed{\text{After layer 1: } 112 \times 112 \times 64}
$$

Feeding that in, the second layer gives $\lfloor (112 + 2 - 3)/2\rfloor + 1 = \lfloor 111/2\rfloor+ 1 = 55 + 1 = 56$.

$$
\boxed{\text{After layer 2: } 56 \times 56 \times 128}
$$

*Reasonableness check:* each stride-2 layer roughly halves the spatial dimensions while doubling the channels — the standard resolution-for-depth trade in CNN design, and it keeps the per-layer activation volume roughly constant.

**Problem 2 — ViT tokenization dimensions.** A ViT processes $224 \times 224 \times 3$ images with patch size $P = 16$ and embedding dimension $D = 768$, using a CLS token. Compute $N_P$, the flattened patch dimension, the shape of $W_E$, the shape of $x_0$, and the shape of $X$.

*Solution.* From Section 2.6, the patch count is $N_P = (224/16)\times(224/16) = 14 \times 14 = 196$. Each patch flattens to $P^2 C_1 = 16^2 \times 3 = 768$ entries, so $x_{\text{flat}} \in \mathbb{R}^{196 \times 768}$ and the embedding matrix has shape $W_E \in \mathbb{R}^{P^2C_1 \times D} = \mathbb{R}^{768 \times 768}$. Prepending the CLS token adds one row, giving $x_0 \in \mathbb{R}^{197\times768}$, and transposing gives $X = x_0^\top$.

$$
\boxed{N_P = 196, \quad P^2C_1 = 768, \quad W_E: 768\times768, \quad x_0: 197\times768, \quad X: 768 \times 197}
$$

*Reasonableness check:* the coincidence $P^2C_1 = D = 768$ is specific to ViT-Base with $16\times16$ patches and is why its patch-embedding matrix is square; with $P = 32$ the flattened dimension would be $3072$ and $W_E$ would be $3072 \times 768$, decidedly non-square.

**Problem 3 — Attention matrix and context vector.** A single-head attention layer has $N = 3$ tokens and $D = 4$. For query $q_1$, the raw scores against the three keys are $e = (2.0,\, 1.0,\, 0.0)$, and the value vectors are $v_1 = (1,0,0,0)$, $v_2 = (0,1,0,0)$, $v_3 = (0,0,1,0)$. Compute the attention weights (ignoring the $\sqrt{D}$ scaling) and the context vector $c_1$.

*Solution.* Applying the align function of Section 2.9, $a = \text{softmax}(e)$: the exponentials are $e^{2.0} \approx 7.389$, $e^{1.0}\approx 2.718$, $e^{0.0} = 1$, summing to $11.107$. Dividing each by the sum gives $a \approx (0.665,\, 0.245,\, 0.090)$. The context vector is the weighted sum $c_1 = \sum_\ell a_\ell v_\ell$, and since the values here are the first three standard basis vectors, the weights land directly in the corresponding coordinates:

$$
\boxed{a \approx (0.665,\, 0.245,\, 0.090), \qquad c_1 \approx (0.665,\, 0.245,\, 0.090,\, 0)^{\top}}
$$

*Reasonableness check:* the weights sum to $1$ as a softmax must, and the fourth coordinate of $c_1$ is exactly zero because no value vector has any mass there — attention can only produce outputs in the span of the value vectors, never outside it.

**Problem 4 — InfoNCE and its mutual-information bound.** In a batch of $N = 256$, an anchor has cosine similarity $0.9$ with its positive and cosine similarity $0.1$ with each of the $255$ negatives, all scaled by an inverse temperature of $10$. Compute $\mathcal{L}_{\text{InfoNCE}}$ and the resulting lower bound on $I(z_1, z_1^+)$ in nats.

*Solution.* After scaling, the positive logit is $10 \times 0.9 = 9$ and each negative logit is $10 \times 0.1 = 1$. Applying the definition in Section 3.2, the numerator is $e^{9} \approx 8103.1$ and the denominator is $e^9 + 255e^1 \approx 8103.1 + 255(2.718) = 8103.1 + 693.1 = 8796.2$. Therefore $\mathcal{L}_{\text{InfoNCE}} = -\log(8103.1/8796.2) = -\log(0.9212) \approx 0.0821$.

$$
\boxed{\mathcal{L}_{\text{InfoNCE}} \approx 0.082 \text{ nats}}
$$

Using the bound of Section 3.3 with $\log(256) \approx 5.545$,

$$
\boxed{I(z_1, z_1^{+}) \;\ge\; 5.545 - 0.082 \approx 5.46 \text{ nats}}
$$

*Reasonableness check:* the bound is close to its ceiling of $\log N = 5.545$ nats, which is correct — the positive is separated from the negatives by 8 logits, so the model is nearly certain, and a nearly-zero loss saturates the bound. Note that no matter how good the representation gets, this bound can never exceed $\log N$, which is exactly why larger batches matter.

**Problem 5 — CLIP loss on a $2\times2$ batch.** A CLIP batch of $N=2$ produces the scaled logit matrix $L = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$, where $L_{ij}$ is the similarity between image $i$ and caption $j$. Compute $\mathcal{L}_{\text{I2T}}$, $\mathcal{L}_{\text{T2I}}$, and $\mathcal{L}_{\text{Total}}$ in nats.

*Solution.* For $\mathcal{L}_{\text{I2T}}$, apply the row softmax of Section 3.5. Row 1 has $e^3 \approx 20.086$ and $e^1 \approx 2.718$, summing to $22.804$, so the diagonal term contributes $-\log(20.086/22.804) = -\log(0.8808) \approx 0.1269$. Row 2 has $e^0 = 1$ and $e^2 \approx 7.389$, summing to $8.389$, contributing $-\log(7.389/8.389) = -\log(0.8808) \approx 0.1269$. Averaging, $\mathcal{L}_{\text{I2T}} \approx 0.1269$.

For $\mathcal{L}_{\text{T2I}}$, apply the column softmax. Column 1 has $e^3 \approx 20.086$ and $e^0 = 1$, summing to $21.086$, contributing $-\log(20.086/21.086) \approx 0.0486$. Column 2 has $e^1 \approx 2.718$ and $e^2 \approx 7.389$, summing to $10.107$, contributing $-\log(7.389/10.107) = -\log(0.7311) \approx 0.3133$. Averaging, $\mathcal{L}_{\text{T2I}} \approx 0.1810$. Combining the two halves,

$$
\boxed{\mathcal{L}_{\text{I2T}} \approx 0.127, \quad \mathcal{L}_{\text{T2I}} \approx 0.181, \quad \mathcal{L}_{\text{Total}} \approx 0.154 \text{ nats}}
$$

*Reasonableness check:* the two directions differ, which is the whole reason CLIP averages both — this logit matrix is "better" for images picking captions than for captions picking images, and a one-directional loss would leave that asymmetry uncorrected. Both are well below $\log 2 \approx 0.693$, the loss of a model guessing at chance.

**Problem 6 — KL term of the VAE ELBO.** For a 2-dimensional latent with encoder outputs $\mu_e = (0.5,\, -1.0)$ and $\sigma_e = (1.0,\, 0.5)$, compute $D_{\mathrm{KL}}\big(\mathcal{N}(\mu_e, \sigma_e^2 I)\,\|\,\mathcal{N}(0,I)\big)$ using the standard closed form $D_{\mathrm{KL}} = \tfrac12\sum_{k}\left(\mu_k^2 + \sigma_k^2 - 1 - \log\sigma_k^2\right)$.

*Solution.* This closed form is the diagonal-Gaussian instance of the prior-matching term in Section 5.3. For dimension 1, $\mu_1^2 = 0.25$, $\sigma_1^2 = 1$, and $\log\sigma_1^2 = 0$, giving $0.25 + 1 - 1 - 0 = 0.25$. For dimension 2, $\mu_2^2 = 1$, $\sigma_2^2 = 0.25$, and $\log(0.25) \approx -1.3863$, giving $1 + 0.25 - 1 + 1.3863 = 1.6363$. Summing and halving,

$$
D_{\mathrm{KL}} = \tfrac12(0.25 + 1.6363).
$$

$$
\boxed{D_{\mathrm{KL}} \approx 0.943 \text{ nats}}
$$

*Reasonableness check:* the value is positive, as any KL must be, and would be exactly zero only for $\mu_e = 0$ and $\sigma_e = 1$ — i.e. posterior collapse (Section 5.6). Dimension 2 contributes over six times as much as dimension 1, which is right: it is both offset from the prior mean and over-concentrated relative to unit variance.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Mutual information objective (discriminative) | $I(Y; z) = H(Y) - H(Y\mid z)$ |
| Conditional entropy | $H(Y\mid z) = -\int P(Y\mid z)\log P(Y\mid z)\,dz$ |
| Cross-entropy / NLL (single example) | $\mathcal{L}_{\text{CE}} = -\log P_\psi(Y = j\mid z)$ |
| Cross-entropy over a dataset | $\mathcal{L}_{\text{CE}} = \frac{1}{N}\sum_{i=1}^{N}-\log P_\psi(Y=j\mid z_i)$ |
| CE decomposition | $\mathcal{L}_{\text{CE}} = I(Y;z) + D_{\mathrm{KL}}(P_\psi(Y\mid z)\,\Vert \,P(Y\mid z))$ |
| CNN output height / width | $H_2 = \lfloor\frac{H_1+2P-F}{S}\rfloor + 1$, $W_2 = \lfloor\frac{W_1+2P-F}{S}\rfloor+1$ |
| ViT patch count | $N_P = \frac{H_1}{P}\times\frac{W_1}{P}$, each patch $P\times P\times C_1$ |
| Flattened patch dimension | $P^2C_1$; $x_{\text{flat}} \in \mathbb{R}^{N_P \times P^2C_1}$ |
| Patch embedding | $x_{\text{embed}} = x_{\text{flat}}W_E + \mathbf{1}b_E^\top \in \mathbb{R}^{N_P\times D}$ |
| CLS + positional embedding | $x_0 = [x_{\text{CLS}};x_{\text{embed}}] + E_{\text{pos}}$ (else $x_{\text{embed}}+E_{\text{pos}}$); $X = x_0^\top \in \mathbb{R}^{D\times N}$ |
| Transformer layer | $X \mathrel{+}= \text{MHSA}(X)$; $X = \text{LN}(X)$; $x_n \mathrel{+}= \text{MLP}(x_n)$; $X_1 = \text{LN}(X)$ |
| LayerNorm | $\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}$, $x_i = \gamma_i\hat{x}_i + \beta_i$ |
| MHSA (one head) | $\text{MHSA}(X) = W_c[\text{SA}_1(X)]$ |
| Self-attention | $\text{SA}_1(X) = V\,\text{Softmax}\!\left(\frac{K^\top Q}{\sqrt{D}}\right) = VA$ |
| Q, K, V projections | $V = W_VX + \beta_V\mathbf{1}^\top$; $Q = W_QX+\beta_Q\mathbf{1}^\top$; $K = W_KX+\beta_K\mathbf{1}^\top$ |
| Context vector | $\underline{c} = \sum_{\ell=1}^{N}a_\ell \underline{v}_\ell$, $\underline{a} = \text{align}(\underline{e})$ |
| Kronecker form of SA | $M = A^\top \otimes I_D \in \mathbb{R}^{DN\times DN}$ |
| InfoNCE | $\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp(\text{sim}(z_1,z_1^+))}{\exp(\text{sim}(z_1,z_1^+)) + \sum_{k=2}^{N}\exp(\text{sim}(z_1,z_k^-))}$ |
| Cosine similarity | $\text{sim}(z_1,z_1^+) = \frac{\langle z_1, z_1^+\rangle}{\lVert z_1\rVert\lVert z_1^+\rVert}$ |
| InfoNCE MI bound | $I(z_1,z_1^+) \ge \log N - \mathcal{L}_{\text{InfoNCE}}$ |
| CLIP logit matrix | $L = \text{scale}\times VT^\top$, $L_{ij} = \text{scale}\times v_i^\top t_j$ |
| CLIP image-to-text loss | $\mathcal{L}_{\text{I2T}} = \frac1N\sum_i -\log\frac{e^{L_{ii}}}{\sum_j e^{L_{ij}}}$ (row softmax) |
| CLIP text-to-image loss | $\mathcal{L}_{\text{T2I}} = \frac1N\sum_j -\log\frac{e^{L_{jj}}}{\sum_i e^{L_{ij}}}$ (column softmax) |
| CLIP total loss | $\mathcal{L}_{\text{Total}} = \frac12(\mathcal{L}_{\text{I2T}}+\mathcal{L}_{\text{T2I}})$ |
| Generative objective | $\min_\phi D_{\mathrm{KL}}(P_{\text{data}}\Vert P_\phi) \equiv \max_\phi\frac1N\sum_i \log P_\phi(x_i)$ |
| Latent-variable model | $P_\theta(x) = \int P_\theta(x\mid z)P(z)\,dz$ |
| Auto-regressive model | $P_\theta(x) = \prod_{i=1}^{n}P(x_i\mid x_{<i})$ |
| VAE prior / posterior / decoder | $P_\theta(z)\sim\mathcal{N}(0,I)$; $q_\psi(z\mid x)\sim\mathcal{N}(\mu_e,\Sigma_e)$; $P_\theta(x\mid z)\sim\mathcal{N}(\mu_d,\Sigma_d)$ |
| ELBO | $\mathcal{L}_{\theta,\psi}(x) = \mathbb{E}_{q_\psi}\!\left[\log\frac{P_\theta(x,z)}{q_\psi(z\mid x)}\right]$ |
| ELBO (two-term form) | $\mathcal{L}_{\theta,\psi} = \mathbb{E}_{q_\psi}[\log P_\theta(x\mid z)] - D_{\mathrm{KL}}(q_\psi(z\mid x)\,\Vert \,P_\theta(z))$ |
| ELBO bound | $\mathcal{L}_{\theta,\psi} \le \log P_\theta(x)$, gap $= D_{\mathrm{KL}}(q_\psi\Vert P_\theta)$ |
| Decoder gradient | $\nabla_\theta\mathcal{L} = \mathbb{E}_{q_\psi}[\nabla_\theta\log P_\theta(x,z)] \approx \nabla_\theta\log P_\theta(x,z^i)$ |
| Reparameterization trick | $z = \mu_e + \sigma_e^2\odot\epsilon$, $\epsilon\sim\mathcal{N}(0,I)$ (implement with $\sigma_e$) |
| Encoder gradient | $\nabla_\psi\mathcal{L} = \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)}[\nabla_\psi f_1(z)]$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the three families** — **"D-G-C: Discriminate, Generate, Compare."** **D**iscriminative makes $z$ *predictive*; **G**enerative makes $z$ *explanatory*; **C**ontrastive makes $z$ *relative*.

**Quick-reference intuition per topic**

- **Discriminative in one line:** learn $P(Y\mid z)$; the training signal is "reduce the leftover uncertainty about $Y$ once you've seen $z$."
- **Why CE is the loss:** $H(Y)$ is fixed, so minimizing $H(Y\mid z)$ = maximizing $I(Y;z)$; we can't evaluate the true $P(Y\mid z)$, so we substitute $P_\psi$ and get cross-entropy = NLL.
- **CE = NLL:** same expression, two names. If someone says "negative log-likelihood," they mean $-\log P_\psi(Y=j\mid z)$.
- **CNN dimension formula:** $\lfloor (\text{in} + 2P - F)/S\rfloor + 1$. Memorize it as "input plus padding, minus filter, over stride, plus one."
- **ViT tokenization order:** patchify → flatten → linear-embed → prepend CLS → add positional. Getting the order wrong (e.g. embedding before flattening) is a common exam slip.
- **Kronecker insight:** $M = A^\top \otimes I_D$ means attention mixes *tokens*, never *features within a token*. Every feature dimension of a value gets the same weight.
- **Why $\sqrt{D}$:** keeps logit scale independent of embedding dimension so the softmax doesn't saturate.
- **LayerNorm vs. BatchNorm:** LayerNorm here is **column-wise** — per token, across features. It does not look at other examples in the batch.
- **Contrastive in one line:** pull the anchor toward its positive, push it away from all negatives in the batch; the batch supplies the negatives for free.
- **InfoNCE ceiling:** the MI bound can never exceed $\log N$. If you need a tighter bound, you need a bigger batch — no amount of training helps past that ceiling.
- **CLIP's two softmaxes:** row softmax = "which caption for this image" (I2T); column softmax = "which image for this caption" (T2I). Diagonal = true pairs. Average both.
- **Zero-shot classification:** the classifier is just a set of prompt strings; there is no trained head. Prompt wording ("a photo of a X") is part of the model.
- **Generative in one line:** minimizing forward KL to the data distribution *is* maximum likelihood — the entropy term drops out because it has no parameter dependence.
- **Three generative sub-families:** latent-variable (integrate $z$ out; bound on likelihood), auto-regressive (chain rule; exact likelihood, slow sampling), implicit (no density; fast, sharp sampling).
- **Why VAE not AE:** an AE's latent is unstructured and full of holes, so you cannot sample from it. The VAE's KL term regularizes the latent toward $\mathcal{N}(0,I)$ so that sampling works.
- **ELBO's two terms:** reconstruction = "how good is the decoder"; prior-matching (KL) = "how good is the encoder." Maximize the first, drive the second toward zero — *but not all the way*.
- **Posterior collapse:** $D_{\mathrm{KL}} = 0$ is *not* success — it means the encoder ignores the input and the decoder ignores $z$. A KL of exactly zero is a red flag, not a win.
- **Why VAE samples blur:** MSE averages over plausible reconstructions, and the smooth prior limits sharp commitment. Two separate causes with the same symptom.
- **Reparameterization, one line:** you cannot backprop through a sample, so move the randomness into an input $\epsilon$ that carries no parameters, and the sample becomes a deterministic function of $\mu_e, \sigma_e$.
- **Generation:** throw away the encoder; sample $z \sim \mathcal{N}(0,I)$ and decode.

**Recall checklist**

- [ ] Can you fill in the three-family comparison table (supervision, goal, examples, geometry of $z$) from memory?
- [ ] Can you derive why minimizing $H(Y\mid z)$ is the same as maximizing $I(Y;z)$?
- [ ] Can you state why we substitute $P_\psi(Y\mid z)$ for $P(Y\mid z)$, and what the resulting loss is called?
- [ ] Can you compute CNN output dimensions and ViT patch/embedding dimensions for given hyperparameters?
- [ ] Can you write the four lines of a transformer layer, and the LayerNorm formula?
- [ ] Can you write $\text{SA}_1(X) = V\,\text{Softmax}(K^\top Q/\sqrt{D})$ and explain the roles of $Q$, $K$, $V$, and $A$?
- [ ] Can you state what $M = A^\top\otimes I_D$ tells you about what attention can and cannot mix?
- [ ] Can you write the InfoNCE loss and its MI lower bound, and name the two ways to tighten it?
- [ ] Can you write both CLIP losses, say which axis each softmax runs along, and describe zero-shot inference including the prompt template?
- [ ] Can you show that minimizing $D_{\mathrm{KL}}(P_{\text{data}}\|P_\phi)$ is maximum likelihood?
- [ ] Can you name the three generative sub-families with one formula and one example each?
- [ ] Can you write the ELBO in both forms and name what each term measures?
- [ ] Can you reproduce the ELBO proof — the multiply-and-divide step and the non-negative KL?
- [ ] Can you explain why $\nabla_\psi$ needs the reparameterization trick but $\nabla_\theta$ does not?
- [ ] Can you explain posterior collapse and the two independent causes of blurry VAE samples?

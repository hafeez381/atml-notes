---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module I - Notes.pdf]]'
- '[[ATML Lec 1.pdf]]'
tags:
- course/cs6304
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · [[ATML_Lecture_2 - Information Theory and Divergences]] →
Concepts: [[Generalization Theory]], [[Manifold Hypothesis]]
%% nav:end %%

# ATML — Lecture 1: Course Introduction, Representation Theory, and Generalization Theory

## Where This Lecture Sits

Every model in the course is an encoder producing $z$ and a head consuming it; this lecture asks what makes one $z$ better than another and answers with the generalization bound of Section 5.2, whose three movable quantities — sample size, capacity, margin — are the spine of the course. [[ATML_Lecture_2 - Information Theory and Divergences]] builds the instruments for measuring distributional distance, [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] compares the three families of representation learners, [[ATML_Lecture_4 - GANs, and Inductive Biases]] attacks capacity through inductive bias, and [[ATML_Lecture_5_1 - Domain Adaptation]]–[[ATML_Lecture_5_2 - Domain Generalization]] drop the assumption that train and test share a distribution.

## Topic Inventory

- 1. Course Roadmap and Structure
  - 1.1 Lecture Sequence
  - 1.2 Module Structure
  - 1.3 Grading Breakdown
- 2. The Deep Model Pipeline
  - 2.1 Pipeline Components and Naming
  - 2.2 The Model as a Composite Function
  - 2.3 Characterizing Every Component by Its Distribution
  - 2.4 Modalities, Tasks, and Vectorization
  - 2.5 Manifold-Structured Data
- 3. Representation Theory — What Makes a Representation Good?
- 4. Probability Distributions in Machine Learning
  - 4.1 Supervised vs. Unsupervised Data
  - 4.2 Marginal Distribution $P(X)$
  - 4.3 $P(X)$ Is Not a Histogram — the CIFAR-10 Example
  - 4.4 Marginal Distribution $P(Y)$
  - 4.5 Joint Distribution $P(X,Y)$ and Its Deep Approximation
- 5. Generalization Theory
  - 5.1 The IID Setting and the Generalization Bound
  - 5.2 Margin, Capacity, and the Refined Bound
  - 5.3 The OOD Setting and Domain Shift
  - 5.4 Controlling the Complexity/Margin Ratio
  - 5.5 Spurious Correlations and Domain-Invariant Features
- 6. Dimensionality Reduction and the Manifold Hypothesis
  - 6.1 Latent-Space Mapping
  - 6.2 The Manifold Hypothesis
- 7. The Classification Task Head and Cross-Entropy Loss
  - 7.1 The Softmax Classifier Head
  - 7.2 Matrix Form of the Classifier Head
  - 7.3 Cross-Entropy Loss Derivation
  - 7.4 The Similarity-Based Form of Cross-Entropy
  - 7.5 Geometric Interpretation: Class Prototypes
- 8. Mathematical Background
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Course Roadmap and Structure

### 1.1 Lecture Sequence

The technical content is organized into two large blocks, sketched on the board as a timeline of topics connected by arrows.

The first block, spanning **3–4 weeks**, begins with the basics of **Deep Representation Learning (DRL)** together with accompanying tutorials, then proceeds to **Transformers** (architecture and working principles), then to **Reasoning**, and finally to **Test-Time Adaptation**. The arrow drawn back from "Test-time Adap." to "DRL Basics + Tutorials" indicates that these four topics build cumulatively on one another rather than being covered as isolated units — reasoning and test-time adaptation both presuppose the representation-learning and transformer material covered earlier.

The second block, spanning **12 weeks**, covers **Inference**, then **Safety Alignment**, and finally **Mechanistic Interpretability**. As with the first block, the underlying arrow signals that these topics form a connected progression: understanding how a trained model performs inference is a prerequisite for reasoning about how to align it safely, which in turn motivates mechanistic interpretability as a tool for verifying what an aligned model has actually learned internally.

### 1.2 Module Structure

The course as a whole is organized under a single unifying theme — **good representation learning for better generalization** — which branches into six modules:

| Module   | Contents                                                                                      |
| -------- | --------------------------------------------------------------------------------------------- |
| Module 1 | DRL types; inductive biases; OOD (out-of-distribution) performance                                                  |
| Module 2 | Domain Generalization (DG), branching into Unsupervised Domain Adaptation (UDA) and DG proper |
| Module 3 | Understanding the *why* of DRL; interpretability                                              |
| Module 4 | Model compression                                                                             |
| Module 5 | DG in distributed systems                                                                     |
| Module 6 | Reasoning as a way to achieve better DG                                                       |

The single sentence at the root of this tree is worth internalizing, because it is the thesis of the whole course: every module is an attempt to answer the question of how to learn representations that generalize.

### 1.3 Grading Breakdown

| Component | Weight | Notes |
|---|---|---|
| Attendance | 5% | — |
| Quizzes | 10% | 3 to 5 quizzes |
| Programming Assignments (PAs) | 35% | 5 to 6 assignments |
| Final Exam | 15% | — |
| Final Project | 35% | Proposal 0%; Mid-Report 5%; Presentation & VIVA 25%; Final Report 5% |

Note that the project proposal carries no marks of its own but is still a required deliverable, and that the bulk of the project's weight (25 of its 35 points) sits on the presentation and viva rather than on the written report.

---

## 2. The Deep Model Pipeline

Since the modules of Section 1.2 are organized around representation learning, the first technical task is to locate the representation within a model. This section does so by reducing every architecture in the course — CNN (convolutional neural network), ViT (vision transformer), VAE (variational autoencoder), GAN (generative adversarial network), CLIP (Contrastive Language–Image Pre-training) — to one skeleton of two parameterized blocks, which is what makes those architectures comparable at all: across the three families of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], only the training of the *same* encoder block changes.

### 2.1 Pipeline Components and Naming

Nearly every deep model considered in this course can be described by the same generic pipeline. Data is assumed to be drawn from an underlying distribution, $P(X) \sim X$, where $X$ denotes the input modality and each sample is paired with label information:

$$
\mathcal{D} = \{x_i,\, y_i\}_{i=1}^{N}, \qquad x_i = \text{input}, \quad y_i = \text{label information.}
$$

The input is passed through an encoder parameterized by $\theta$, producing a **latent representation** (also called a **feature map**) $z = g_\theta(x)$. This latent representation is then passed to a **downstream task head** parameterized by $\psi$, producing the final prediction $\hat{y} = h_\psi(z)$:

$$
x \;\xrightarrow{\;\theta\;}\; z = g_\theta(x) \;\xrightarrow{\;\psi\;}\; \hat{y} = h_\psi(z), \qquad \text{model parameters } \{\theta, \psi\}.
$$

The first block goes by several interchangeable names, all of which appear in the literature and all of which mean the same thing: the **Deep Representation Learning** block, the **feature extractor**, or the **backbone network**. It is the component **responsible for Deep Representation Learning** — its job is to learn a compact, informative representation $z$ of the raw input, independent of the specific downstream task. The task head $h_\psi(\cdot)$ then specializes that representation to a particular output.

The training objective closes the loop: the **loss** $\mathcal{L}(\theta, \psi)$ is defined using the discrepancy between the true labels $y_i$ and the predictions $\hat{y}_i$, and is then averaged over a batch of examples.

### 2.2 The Model as a Composite Function

Formally, the whole model is a single function of the input,

$$
\boxed{Y = f(x) \quad \text{where} \quad f = g_\theta \circ h_\psi, \qquad \text{with parameters } W = \{\theta, \psi\}}
$$

so that the encoder and the head together form one **composite function**.

> **[Possible source inconsistency]** — the notes write the composition as $f = g_\theta \circ h_\psi$. Under the standard convention $(g \circ h)(x) = g(h(x))$, the diagram's actual data flow ($x \to g_\theta \to z \to h_\psi \to \hat{y}$) corresponds to $f = h_\psi \circ g_\theta$. The professor's ordering is reproduced as given; read it as "first $g_\theta$, then $h_\psi$," which is unambiguous from the accompanying diagram.

Crucially, $g_\theta$ and $h_\psi$ are **themselves composite functions**: each is a stack of layers, so the pipeline expands to

$$
x \to \underbrace{\boxed{\;} \to \boxed{\;} \to \cdots \to \boxed{\;}}_{g_\theta \text{ (layers)}} \to z \to \underbrace{\boxed{\;} \to \boxed{\;} \to \cdots \to \boxed{\;}}_{h_\psi \text{ (layers)}} \to \hat{Y}.
$$

This nested-composition view is what makes deep networks "deep," and it is also what makes them capable of approximating distributions that no single transformation could capture.

### 2.3 Characterizing Every Component by Its Distribution

A methodological commitment stated at the outset of the course: **every component of the pipeline will be characterized by its underlying probability distribution**. These fall into three groups:

| Group | Distributions |
|---|---|
| **Data distributions** | $P(X)$, $P(Y)$, $P(X,Y)$, $P(X \mid Y)$, $P(Y \mid X)$, etc. |
| **Latent distributions** | $P(z)$, $P(z \mid x)$, etc. |
| **Parameter distributions** | $P(\theta)$, $P(\psi)$, etc. |

Treating even the *parameters* as random variables with their own distributions is what makes concepts such as Bayesian deep learning, weight priors, and uncertainty quantification expressible in this framework.

### 2.4 Modalities, Tasks, and Vectorization

The pipeline is deliberately modality-agnostic. The modalities considered are **images/patches** (the primary focus of this course), **language/tokens**, **time series**, and **augmented data**. Before entering the network, an image is passed through a **vectorization** step that flattens it into a column vector:

$$
x_i = \text{Image} \;\longrightarrow\; \text{Vectorization} \;\longrightarrow\; x_i \in \mathbb{R}^{D \times 1}.
$$

On the output side, the same pipeline supports a wide range of tasks depending on what $h_\psi(\cdot)$ is trained to do:

- **Vision** — classification, object detection, and so on.
- **Language** — sentiment analysis, text generation (via either diffusion-based or autoregressive decoding).
- **Multimodal** — Visual Question Answering (VQA), Text-to-Image (T2I), Image-to-Text (I2T).

Image and video generation belong to the same list. The key conceptual point is that the encoder-then-head decomposition is shared across all of these; what differs from task to task is primarily the design of the task head and the training objective, not the overall architecture of the pipeline.

### 2.5 Manifold-Structured Data

All of the above data is described as **manifold-structured data**: data that, although it *appears* to reside in very high dimensions, is actually **concentrated near a much lower-dimensional smooth surface with its own geometry**. Diagrammatically, an amorphous blob in $\mathbb{R}^D$ collapses onto a curved sheet in $\mathbb{R}^d$ with
![[03 - Attachments/Screenshot 2026-09-21 at 12.30.15 PM.png|421]]
$$
d \ll D, \qquad d = \text{intrinsic dimension}, \qquad D = \text{ambient dimension}.
$$

This is the same manifold hypothesis developed in more detail in Section 6, and it is the structural fact that makes dimensionality reduction possible at all.

---

## 3. Representation Theory — What Makes a Representation Good?

Section 2 established that a representation $z$ sits between encoder and head, but not what a good one looks like. The six properties below are not arbitrary preferences: each is a hypothesis about what makes the downstream head's job easy, and each is later turned into an explicit training objective.

**Representation theory**, in the sense used in this course, asks a single question: *what makes the learned representation $z = g_\theta(x)$ good?* Six desirable properties are listed:

1. **Sufficiency for the task** — $z$ must retain all the information needed to solve the downstream task; nothing task-relevant may be discarded by the encoder.
2. **Minimality of $z$** — subject to sufficiency, $z$ should contain as little else as possible. Together, sufficiency and minimality express the information-bottleneck ideal: keep everything that matters, discard everything that does not.
3. **Group/geometry is present**, formalized as $$g(x + \delta x) \approx g(x),$$ i.e. the representation should be stable under small perturbations $\delta x$ of the input. This is the robustness/invariance property, and it is what connects representation quality to the notion of a symmetry group acting on the input space.
4. **Disentanglement** — distinct latent dimensions should correspond to distinct, independent factors of variation in the data, rather than each dimension mixing several factors together.
5. **Smoothness** — nearby points in input space should map to nearby points in latent space, so that the latent space can be interpolated and traversed meaningfully.
6. **Dimensionality reduction** — $z$ should live in a much lower-dimensional space than $x$, consistent with the manifold hypothesis.

These six properties are not measured with a single yardstick. The notes group the measurement tools into three families:

1. **Information-theoretic terms** — entropy, mutual information, cross-entropy (used to assess sufficiency and minimality).
2. **Geometric terms** — margins, distances, curvature, manifold structure (used to assess smoothness and separation).
3. **Group-theoretic terms** — invariance and equivariance under transformation groups (used to assess the symmetry property in item 3).

Most of the rest of the course is an elaboration of these three families. The information-theoretic family is developed in [[ATML_Lecture_2 - Information Theory and Divergences]] and becomes the organizing principle of the three DRL families in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]]; the geometric family begins in Sections 5 and 7 below, since margin is a geometric quantity, and returns as optimal transport in Lecture 2 and loss-landscape flatness in [[ATML_Lecture_5_2 - Domain Generalization]]; the group-theoretic family is deferred to [[ATML_Lecture_4 - GANs, and Inductive Biases]], where invariance and equivariance formalize architectural inductive bias.

Properties 1 and 2 are in genuine tension rather than merely coexisting. Sufficiency is trivially achieved by $z = x$ and minimality by $z = 0$; the object of interest is the frontier between them, which the **information bottleneck** names precisely — maximize $I(Y;z)$ subject to a constraint on $I(X;z)$. Several failure modes catalogued later (posterior collapse, shortcut learning, spurious correlations) are cases of satisfying a proxy for one while violating the other.

---

## 4. Probability Distributions in Machine Learning

Section 2.3 committed to characterizing every pipeline component by its distribution; this section says what those distributions look like. The central point is a negative one — they are objects we can sample from but never write down — and much of the course is downstream of it. Generalization theory (Section 5) exists because an unseen $P(X,Y)$ must be reasoned about from a finite sample; the divergences of [[ATML_Lecture_2 - Information Theory and Divergences]] exist because two such objects must be compared from samples alone; and generative modelling in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] is the attempt to approximate $P(X)$ by something that *can* be written down.

### 4.1 Supervised vs. Unsupervised Data

Machine learning datasets are characterized by which probability distribution they implicitly provide samples from. In the **supervised** setting, the dataset consists of paired inputs and labels,

$$
\mathcal{D} = \{ \vec{x}_i, y_i \}_{i=1}^{N},
$$

which can be understood as $N$ i.i.d. samples from the joint distribution $P(X,Y)$ (or, viewed separately, from $P(X)$ and $P(Y)$). In the **unsupervised** setting, only inputs are available,

$$
\mathcal{D} = \{ \vec{x}_i \}_{i=1}^{N},
$$

corresponding to samples from the marginal $P(X)$ alone, with no accompanying label information.

### 4.2 Marginal Distribution $P(X)$

Consider $X$ ranging over a space of images, e.g. $X \in \{\text{cats, dogs}, \ldots\}$. If each image has spatial resolution $256 \times 256$, then a naive view of $P(X)$ treats every pixel as having its own distribution, giving $P(X)$ an apparent dimensionality of $256^2$ — one dimension per pixel.

Because a distribution over $256^2$ dimensions cannot be visualized or reasoned about directly, it is useful to collapse this high-dimensional distribution down onto a single conceptual axis $x$, plotting a rough "probability" curve against it. 
![[03 - Attachments/Pasted image 20260921124025.png|437]]
On this collapsed axis, the density is not uniform: it exhibits high-probability regions clustered around what the notes label "Dog" images, and separate humps around "avg cat" and "cat painting" images. The point of this sketch is that $P(X)$ over natural images is highly non-uniform and multimodal — most of the $256^2$-dimensional pixel space corresponds to unstructured noise with near-zero probability density, while recognizable images (dogs, average-looking cats, stylized cat paintings) occupy separated, high-density modes.

### 4.3 $P(X)$ Is Not a Histogram — the CIFAR-10 Example

A point emphasized in a box in the professor's notes, and easy to get wrong:

$$
\boxed{P(x) \neq \text{histogram of all } \{x_i\}_{i=1}^{N}}
$$

The underlying data distributions $P(X)$, $P(Y)$, $P(X,Y)$, and so on are **very complex and cannot be known or modeled perfectly**. A finite dataset gives you an empirical histogram, which is a *sample-based approximation*, not the true density — the true density has support and structure everywhere the finite sample happens not to land.

The concrete illustration is **CIFAR-10** (from the Canadian Institute for Advanced Research), a dataset of 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Sketching $P(X)$ over a collapsed axis shows two labeled peaks, one for **dog** and one for **cat**, positioned around the "average dog" and "average cat" respectively. The region *between* the two peaks is annotated as the **dimensions common to dog and cat** — the shared visual factors (four legs, fur, similar poses, similar backgrounds) that both classes possess. This overlapping middle region is exactly where classification is hard, and it is the geometric picture behind the margin discussion in Section 5.2.

### 4.4 Marginal Distribution $P(Y)$

The label distribution $P(Y)$, by contrast, is typically a **discrete categorical distribution** representing every class in the task. For a $K$-class classification problem (the notes use $K=10$, as in CIFAR-10 or MNIST (a dataset of handwritten digits)), $P(Y)$ is visualized as $K$ separate bars, one per class, each with its own probability mass.

### 4.5 Joint Distribution $P(X,Y)$ and Its Deep Approximation

The **joint distribution** $P(X,Y)$ is, in principle, extremely informative: knowing it fully would mean knowing exactly how inputs and labels co-occur. However, as the notes state directly, this joint distribution **"exists but can't be found"** — that is, while $P(X,Y)$ is a well-defined mathematical object underlying any real-world data-generating process, it is not something we have direct analytic access to; it can only be approximated from finite samples.

Deep networks approximate this intractable object through a chain of learned representations. Starting from the encoder $g_\theta(\cdot)$ applied to $P(X)$, the network passes the input through a sequence of $L$ layers, each performing a non-linear operation, producing a chain of intermediate representations:

$$
P(X) \;\longrightarrow\; g_\theta(\cdot):\quad z^{(1)} \to z^{(2)} \to \cdots \to z^{(L)},
$$

where each intermediate variable $z^{(i)}$ has its own associated distribution $P(z^{(i)})$, and the final representation satisfies $P(z^{(L)}) = z$. In other words, the encoder does not merely compute a single deterministic transformation; conceptually, each layer can be understood as inducing its own distribution over its output activations, and the composition of these $L$ non-linear layers is what allows a deep network to approximate a target distribution — such as the otherwise inaccessible $P(X,Y)$ — that a single shallow, linear transformation could not capture.

---

## 5. Generalization Theory

Section 4 established that $P(X,Y)$ is inaccessible and that training touches only a finite sample of it. If performance can be measured only on the sample but matters only on the distribution, the discipline reduces to bounding the gap between the two; that gap is what "generalization" names. The boxed inequality of Section 5.2 is the formula the rest of the course returns to most often.

**Generalization theory** answers the question: *when does the learned function $f(\cdot)$ work on new data?* Everything depends on what kind of "new data" is meant, which splits into two cases.

### 5.1 The IID Setting and the Generalization Bound

The standard assumption underlying most machine learning theory is the **IID (independent and identically distributed)** assumption: the training and test data are drawn from the *same* underlying joint distribution,

$$
P_{\text{Test}}(X,Y) = P_{\text{Train}}(X,Y).
$$

Under this assumption, a classical generalization bound relates test performance to training performance, sample size, and model complexity:

$$
\text{Loss}_{\text{Test}} \;\le\; \underbrace{\text{Loss}_{\text{Train}}}_{\text{how good training is}} \;+\; \underbrace{\frac{1}{\sqrt{N}} \cdot \frac{\text{Complexity}}{\text{Margin}}}_{\text{generalization term}}
$$

The decomposition matters as much as the formula: the first term measures how well the model fits what it has already seen, and the second term — the **generalization term** — measures how much that fit is expected to degrade on unseen data. **For a fixed number of training examples $N$, there are exactly two levers available**: reduce the complexity, and/or increase the margin.

*Intuition.* Each term admits a one-line reading. The training loss is the floor: a model that cannot fit what it has seen has no claim on anything else. The factor $1/\sqrt{N}$ is the rate at which a sample average converges to the expectation it estimates — the standard error of a mean — so data helps with sharply diminishing returns, four times as many examples halving the penalty. The complexity numerator charges for flexibility: a model able to fit *any* labelling of the training set conveys nothing by fitting this one. The margin denominator is confidence expressed as distance, since correct-by-a-wide-margin predictions survive the perturbation that a test point represents relative to a training point. Taken together: *a model generalizes when it fits the data, using few effective degrees of freedom, with room to spare.* The two levers follow, complexity and margin being the only such quantities training controls.

One caveat should be carried forward. Bounds of this form are frequently **vacuous** for over-parameterized networks — the right-hand side can exceed the maximum possible loss while the network generalizes well in practice. The bound is used here as a design compass, identifying which quantities matter and in which direction, not as a numerical prediction. Section 5.3 relies on exactly this reading when carrying it into the OOD regime, where no rigorous analogue exists.

This bound is also connected to a **robustness/perturbation** perspective: a well-generalizing function should satisfy $f(x) \approx f(x + \delta x)$ for a small perturbation $\delta x$ — i.e. its output should not change much under small input perturbations — which links directly to property 3 of representation theory (Section 3) and to why controlling weight norms helps generalization.

### 5.2 Margin, Capacity, and the Refined Bound

**Margin is separation.** Sketched as two class clusters with a gap between them, the margin $\gamma$ is the distance separating the decision boundary from the nearest examples of each class. The notes flag an open question here — **what is the right definition of margin for the multi-class case?** — which does not have as clean an answer as the binary case.

**Capacity is sensitivity.** For a network of $L$ layers with weight matrices $W_1, \ldots, W_L$, two quantities are defined:

$$
\beta = \prod_{\ell=1}^{L} \lVert W_\ell \rVert_2, \qquad\qquad
S = \sum_{\ell=1}^{L} \frac{\lVert W_\ell \rVert_F^2}{\lVert W_\ell \rVert_2^2},
$$

where $\lVert \cdot \rVert_2$ is the spectral norm (largest singular value) and $\lVert \cdot \rVert_F$ is the Frobenius norm. The product $\beta$ measures the worst-case amplification the network can apply to an input perturbation as it propagates through all $L$ layers — hence "capacity is sensitivity." The sum $S$ measures how spread out each layer's singular values are: a layer whose Frobenius norm greatly exceeds its spectral norm has many comparably-sized directions of action, contributing more to $S$. Together they give

$$
\text{Capacity} = \beta \sqrt{S}.
$$

*Intuition for why capacity requires both factors.* The spectral norm $\lVert W_\ell\rVert_2$ is the worst-case magnification a layer applies to an input perturbation along its strongest axis; stacking $L$ layers multiplies these magnifications, which is $\beta$. A large $\beta$ means an infinitesimal input change can produce a visible output change, and a function that volatile can carve the input space finely enough to memorize it — hence "capacity is sensitivity."

$\beta$ is, however, blind to *how many* directions a layer magnifies: a rank-one layer stretching one direction has the same spectral norm as a layer stretching a hundred directions equally, though the latter is plainly more expressive. This is what $S$ detects. Since $\lVert W\rVert_F^2$ sums the squares of all singular values while $\lVert W\rVert_2^2$ retains only the largest, the ratio $\lVert W_\ell\rVert_F^2/\lVert W_\ell\rVert_2^2$ counts the **effective number of active directions** in layer $\ell$, equalling $1$ for a rank-one layer and the full rank when all singular values coincide; hence $S \ge L$ always. So $\beta$ measures how hard the network stretches and $S$ in how many directions, and $\beta\sqrt{S}$ combines them.

This also separates the two standard interventions of Module 4: weight decay and spectral normalization cap $\beta$; pruning and low-rank compression reduce $S$. They target different factors of the same product.

Substituting this capacity measure and the margin $\gamma$ into the general bound produces the refined, course-specific form:

$$
\boxed{\text{Loss}_{\text{Test}} \;\le\; \text{Loss}_{\text{Train}} \;+\; \frac{1}{\sqrt{N}} \cdot \frac{\beta \sqrt{S}}{\gamma}}
$$

The notes label this boxed result as the **capacity/complexity control** equation and mark it as "(1)" for later reference — it is the single most-cited formula of the module. A simpler form of the complexity term, used in the earlier lecture sketch, is

$$
\text{Complexity} \;\propto\; \sum_{i=1}^{L} \lVert W_i \rVert_2^2,
$$

which conveys the same intuition — larger-magnitude weights mean a more complicated, more overfitting-prone function — but the $\beta\sqrt{S}$ version is the precise one.

### 5.3 The OOD Setting and Domain Shift

The **OOD (out-of-distribution)** setting instead assumes that the training and test distributions differ:

$$
P_{\text{Test}}(X,Y) \neq P_{\text{Train}}(X,Y).
$$

This regime is called **OOD generalization** or, equivalently, **domain generalization (DG)** — the term used throughout the rest of the course. The canonical example is training on **real cats** but testing on **painting/cartoon cats**: the label space (still "cat") is unchanged, but the input distribution has shifted substantially in style and appearance. This mismatch is called **domain shift**.

The theoretical situation here is explicitly acknowledged as weak: **the theory for OOD is very nascent**. There is no clean, established OOD analogue of the IID bound. The working position taken in the course is that the boxed equation (1) from Section 5.2 is **more-or-less applicable** even in the OOD case, and is used as the guiding principle for designing OOD-robust methods.

### 5.4 Controlling the Complexity/Margin Ratio

Given that equation (1) is the governing bound in both regimes, the central practical question becomes: **how do we control the complexity/margin ratio?** Three answers are given, and they are grouped under a single bracket labelled **good representation learning**:

1. **Model compression and flatness** — reduce $\beta\sqrt{S}$ directly by pruning, quantizing, or distilling the network, and by seeking flat minima in the loss landscape where the loss is insensitive to parameter perturbations. (This is the content of Module 4.)
2. **The model's inductive bias** — bake the right structural assumptions into the architecture so that it does not need excess capacity to fit the data. (This is a major part of Module 1.)
3. **Invariance-driven generalization** — force the representation to be invariant to nuisance transformations, which effectively increases the margin by collapsing irrelevant variation.

The three bullets are the reason the whole course is framed as *good representation learning for better generalization*: each one is a route to tightening the same bound.

### 5.5 Spurious Correlations and Domain-Invariant Features

A concrete illustration of *why* OOD generalization is hard: if a watermark is added to every cat image in the training set, the model may learn to associate that watermark with the "cat" label. This is a **spurious correlation** — the model exploits an "easy" feature that happens to correlate with the label in the training distribution but carries no genuine causal relationship to the concept being learned. Such a model performs well in-domain (where the watermark reliably appears) but fails catastrophically the moment the spurious cue is absent or altered, which is exactly what a domain shift at test time would do.

The remedy is to force the model to learn the **underlying common features that make a cat a cat**, rather than incidental artifacts of the training data. This is illustrated with three differently-drawn cat images (Cat 1, Cat 2, Cat 3): despite their visual differences, the model should extract the features they share in common. These shared, distribution-independent features are called **domain-invariant features**, and learning them is the key strategy for improving OOD robustness — the third bullet of Section 5.4.

---

## 6. Dimensionality Reduction and the Manifold Hypothesis

Section 5 left a mandate — shrink complexity, grow margin — without a reason to believe it achievable: why should a high-dimensional input admit a low-complexity description without losing what the task needs? The manifold hypothesis is that reason. Were natural data to occupy all of $\mathbb{R}^D$, compressing to $\mathbb{R}^d$ with $d \ll D$ would necessarily destroy task-relevant content, and Section 3's properties 1 and 6 would be incompatible. The hypothesis asserts instead that the ambient dimension is an artifact of the coordinate system.

### 6.1 Latent-Space Mapping

Revisiting the general pipeline with explicit dimensionalities: the input $X$ lives in a high-dimensional space $\mathbb{R}^{D \times 1}$, the encoder $g_\theta(\cdot)$ maps it to a latent code $z \in \mathbb{R}^{d \times 1}$, and the task head $h_\psi(\cdot)$ maps $z$ to the prediction $y$:

$$
P(X) \sim X \in \mathbb{R}^{D\times 1} \;\xrightarrow{g_\theta(\cdot)}\; z \in \mathbb{R}^{d \times 1} \;\xrightarrow{h_\psi(\cdot)}\; y,
\qquad d \ll D.
$$

The condition $d \ll D$ is the defining property of **dimensionality reduction**: the encoder compresses a high-dimensional input into a much lower-dimensional latent code, discarding redundant or task-irrelevant information while (ideally) preserving what is needed for prediction. The target space of this mapping is a **manifold**.

### 6.2 The Manifold Hypothesis

The **manifold hypothesis** states that although real-world high-dimensional data (such as natural images embedded in $\mathbb{R}^D$) may appear to require many dimensions to describe, the data actually lies on or near a much lower-dimensional manifold embedded within that ambient space. The notes give a physical analogy: a manifold can have a higher *apparent* dimension while its true underlying dimension is lower — just as **folding a piece of paper** increases the number of ambient coordinates seemingly needed to describe points on it, even though the paper's intrinsic (underlying) dimensionality remains $2$, provided its thickness is taken to be zero. This is exactly the intrinsic-vs-ambient distinction of Section 2.5: the *ambient* dimensionality $D$ of the raw data can be far larger than the *intrinsic* dimensionality $d$ of the structure that actually generates it, and dimensionality reduction is precisely the task of recovering that lower intrinsic dimension.

*Intuition.* A $256\times256$ grayscale image carries $65{,}536$ pixel coordinates, but the scene it depicts is determined by comparatively few degrees of freedom — object identity and pose, lighting direction, camera angle. Every remaining direction in pixel space corresponds to a perturbation no physical process produces: changing one pixel independently of its neighbours yields sensor noise, not a photograph. The images a camera *can* produce therefore form a $d$-dimensional surface occupying essentially none of the ambient volume, which is the same statement as Section 4.2's observation that $P(X)$ is near-zero almost everywhere.

The folded-paper analogy supplies the second half of the point: the surface is *curved*, not merely small. An unfolded sheet is a linear subspace that PCA (principal component analysis) would recover exactly; once folded, no linear projection recovers the intrinsic coordinates, since the fold brings ambient-nearby points that are far apart along the paper. This is why $g_\theta$ must be a stack of non-linear layers rather than a single matrix — each layer performs a partial unfolding — and it is the concrete sense in which the depth of Section 2.2 earns its keep.

The stated mechanism for achieving this is to **flatten the manifold into a (Euclidean) vector space via sparse autoencoders** — using an autoencoder with a sparsity constraint on its latent activations as the encoder $g_\theta(\cdot)$ that performs this unfolding, producing a latent representation $z$ in which the data's true low-dimensional structure is made explicit and linearly accessible.

---

## 7. The Classification Task Head and Cross-Entropy Loss

Sections 2 through 6 concerned the encoder half of the pipeline; this section opens the other box, the task head $h_\psi$, for the case of $K$-class classification. It belongs in the foundation lecture because cross-entropy proves to be a geometric object in disguise: rewritten in Section 7.4, it says *pull each latent code toward its own class direction and away from the others*, which is a statement about margin. The classifier head is therefore where the abstract $\gamma$ of Section 5.2 becomes a quantity the optimizer manipulates directly.

### 7.1 The Softmax Classifier Head

For **$K$-class classification**, the task head $h_\psi(\cdot)$ takes the concrete form of a **classifier head** built from a single linear layer followed by a **softmax** nonlinearity. The latent vector $z = (z_1, z_2, \ldots, z_d) \in \mathbb{R}^{d\times 1}$ is passed through a single linear layer with **no activation function**, whose number of output neurons equals the number of classes $K$. This produces a vector of **logits** $s \in \mathbb{R}^{K \times 1}$, which is then passed through softmax to produce the final predicted class probabilities $\hat{y} \in \mathbb{R}^{K \times 1}$:

$$
z_1, \ldots, z_d \;\longrightarrow\; \text{[Linear layer, no activation]} \;\longrightarrow\; s_{K\times 1} \;\longrightarrow\; \text{Softmax} \;\longrightarrow\; \hat{y}_{K \times 1}.
$$

### 7.2 Matrix Form of the Classifier Head

Writing the linear layer's rows explicitly makes the geometry visible. Collect the $K$ **class-specific weight vectors** $\psi_k \in \mathbb{R}^{d\times 1}$ as the rows of a matrix $\Psi \in \mathbb{R}^{K \times d}$:

$$
\begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_K \end{bmatrix}
= \text{Softmax}\!\left(
\underbrace{\begin{bmatrix} \text{---}\;\psi_1^{\top}\;\text{---} \\ \text{---}\;\psi_2^{\top}\;\text{---} \\ \vdots \\ \text{---}\;\psi_K^{\top}\;\text{---} \end{bmatrix}}_{\Psi:\; K \times d}
\underbrace{\begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_d \end{bmatrix}}_{d \times 1}
\right)
$$

so that componentwise,

$$
\boxed{\hat{y}_k = \text{softmax}\!\left(\psi_k^{\top} z\right), \qquad k = 1, 2, \ldots, K}
$$

and this quantity **is** the class posterior, $\hat{y}_k = P(\hat{Y} = k \mid x)$.

Two pieces of terminology attach here. First, $\psi_k$ is the class-specific weight vector, also called the **class prototype** for class $k$. Second, the inner product $\psi_k^\top z$ is interpreted as a **distance/similarity measure** between the latent code and the prototype, written $\delta(\psi_k, z)$. The sign convention adopted is

$$
\delta(\psi_k, z) = -\,\psi_k^{\top} z,
$$

i.e. $\delta$ is the **negative of the dot product**, so that **smaller $\delta$ is better** (higher similarity). This convention is what makes the reformulation in Section 7.4 possible.

### 7.3 Cross-Entropy Loss Derivation

Training this classifier head uses the **cross-entropy loss**. For a one-hot encoded true-label vector $y$ (with $y_k = 1$ for the true class and $0$ otherwise), the general cross-entropy loss over all $K$ classes is

$$
\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{y}_k.
$$

Because $y$ is one-hot with the true class indexed by $j$, every term in this sum vanishes except the one corresponding to $k=j$, collapsing the expression to

$$
\mathcal{L}_{\text{CE}} = -\log \hat{y}_j = -\log\!\left(\text{softmax}(s_j)\right), \qquad s_j = \psi_j^{\top} z.
$$

Substituting the explicit softmax expression and applying the logarithm of a quotient (splitting into the log of the numerator minus the log of the denominator) gives the standard **log-sum-exp** form:

$$
\mathcal{L}_{\text{CE}} = -\log\!\left(\frac{e^{s_j}}{\sum_{k=1}^{K} e^{s_k}}\right)
= -\left(\log e^{s_j}\right) + \log\!\left(\sum_{k=1}^{K} e^{s_k}\right),
$$

$$
\boxed{\mathcal{L}_{\text{CE}} = -s_j + \log\!\left(\sum_{k=1}^{K} e^{s_k}\right)}
$$

### 7.4 The Similarity-Based Form of Cross-Entropy

Rewriting the same loss in terms of the similarity measure $\delta$ exposes what the objective is really doing geometrically. Using $s_j = \psi_j^{\top} z = -\delta(\psi_j, z)$ and $s_k = -\delta(\psi_k, z)$, the boxed result above becomes

$$
\mathcal{L}_{\text{CE}} = +\,\delta(\psi_j, z) + \log\!\left(\sum_{k=1}^{K} e^{-\delta(\psi_k, z)}\right).
$$

Pulling the $e^{-\delta(\psi_j, z)}$ factor out of the sum — equivalently, dividing numerator and denominator of the original softmax by $e^{s_j}$ — separates the true class from the rest and yields the ratio form:

$$
\mathcal{L}_{\text{CE}} = \log\!\left(1 + \sum_{k \neq j} \frac{e^{-\delta(z, \psi_k)}}{e^{-\delta(z, \psi_j)}}\right).
$$

Since this holds for any $i$-th example $(x_i, y_i)$ with $z_i = g_\theta(x_i)$, the per-example loss is

$$
\boxed{\mathcal{L}_{\text{CE}_i} = \log\!\left(1 + \sum_{k \neq j} \frac{e^{-\delta(z_i, \psi_k)}}{e^{-\delta(z_i, \psi_j)}}\right)}
$$

*Intuition.* The log-sum-exp form is convenient computationally but obscures what the loss rewards, since both its terms grow with the logits and only their difference matters. The ratio form makes this explicit: every surviving quantity is a gap $s_k - s_j$ between a wrong class's logit and the true one's, absolute scale having cancelled. Each summand $e^{s_k-s_j}$ is one competitor's complaint, small exactly when class $k$ sits well below the true class — so the gap $s_j - s_k$ *is* the per-competitor margin, exponentially penalized once it goes negative. The leading $1$ is the true class against itself and cannot be removed, which is why $\mathcal{L}_{\text{CE}} \ge \log 1 = 0$, with equality approached only as every gap diverges. Since $\log(1+u)\approx u$ for small $u$, the gradient is dominated by whichever wrong class is closest to winning: cross-entropy is a soft, differentiable max-margin objective, and this is the mechanism by which minimizing it grows the $\gamma$ of Section 5.2.

The same reading anticipates a point central to [[ATML_Lecture_5_1 - Domain Adaptation]]. Because the loss keeps decreasing as margins grow and never attains zero, an over-parameterized network trained past perfect classification continues inflating margins along whatever direction separates the classes — and the objective never specifies *which* direction. That freedom is what a spurious feature exploits.

Under the assumption $\delta(z_i, \psi_k) = -\psi_k^{\top} z_i$, this form makes the training signal explicit:

$$
\boxed{\text{We maximize } \psi_j^{\top} z_i \text{ and minimize } \psi_k^{\top} z_i \;(k \neq j) \text{ at the same time.}}
$$

> **[Unclear in source]** — the subscripts on this boxed statement are ambiguous in the handwriting (they read approximately "maximize $\psi_k^\top z_j$ & minimize $\psi_k^\top z_i$"). The version given above is the one consistent with the derivation immediately preceding it: the numerator term involves the *true* class prototype $\psi_j$ and the summed denominator terms involve the *other* prototypes $\psi_k$, $k\neq j$.

### 7.5 Geometric Interpretation: Class Prototypes

This form makes the optimization behavior of the classifier transparent. Since $s_j = \psi_j^\top z$ appears with a negative sign in the log-sum-exp form, **increasing $s_j$ directly decreases** $\mathcal{L}_{\text{CE}}$, while the log-sum-exp term penalizes large logits for the *other*, incorrect classes. Consequently, minimizing cross-entropy drives the network to do two things simultaneously: **align** the latent code $z \in \mathbb{R}^d$ as closely as possible with the prototype vector $\psi_j$ of its true class (maximizing their inner product $s_j$), while pushing $z$ to be as **orthogonal as possible to every other class prototype** $\psi_k$, $k \neq j$ (minimizing their inner products and hence the log-sum-exp penalty).

Geometrically, this is sketched as a set of prototype vectors $\psi_1, \psi_2, \ldots, \psi_j, \ldots, \psi_K$ radiating outward from a common origin, playing the role of **class prototypes**: training pulls each class's latent representations toward its own prototype direction and away from all other classes' prototype directions, effectively carving the latent space into angularly-separated regions, one per class. This is the same "margin" that appears in the generalization bound of Section 5.2 — the classifier head and the generalization theory are talking about the same geometric quantity.

---

## 8. Mathematical Background

Section 7 closed the lecture's conceptual arc; this final section is a toolbox, collecting three identities the course uses constantly and usually without comment. Each carries a specific load later: the mean-as-integral is what lets every loss be written as an expectation under $P(X)$, and — since that expectation is uncomputable for the reasons of Section 4 — what forces the Monte Carlo approximation opening [[ATML_Lecture_2 - Information Theory and Divergences]]; the covariance matrix is the object whose off-diagonal entries measure the entanglement disentangled representation learning removes; and the law of the unconscious statistician is what makes the reparameterization trick of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] legal.

A short recap of the probability facts used throughout the course. Let $x \in \mathbb{R}^{d \times 1}$ be a random variable distributed as $x \sim P(x)$ — for example, $x \sim \mathcal{N}(\mu_x, \Sigma_x)$. Then:

**(1) Mean.** The expectation of $x$ is defined as an integral in the continuous case and a sum in the discrete case:

$$
\mu_x = \mathbb{E}[x] =
\begin{cases}
\displaystyle\int x\, P(x)\, dx & \text{if } x \text{ is continuous},\\[2ex]
\displaystyle\sum_i x_i\, P(x_i) & \text{if } x \text{ is discrete}.
\end{cases}
$$

**(2) Variance / covariance.** The covariance matrix is the expectation of the outer product of the centered variable with itself:

$$
R_x = \mathbb{E}\!\left[(x - \mu_x)(x - \mu_x)^{\top}\right], \qquad R_x \in \mathbb{R}^{d \times d}.
$$

Because it is an outer product, $R_x$ is a $d \times d$ matrix rather than a scalar: its diagonal holds the per-dimension variances and its off-diagonal entries hold the pairwise covariances.

**(3) Expectation of a function.** Let $a = b(x)$ be some function of $x$. Then its mean is obtained by integrating the function against the density of $x$, not the density of $a$:

$$
\mu_a = \mathbb{E}[a] = \int b(x)\, P(x)\, dx,
$$

and so on for higher moments. This is the *law of the unconscious statistician*, and it is the identity that lets every loss function in the course be written as an expectation under the data distribution and then estimated by a sample average.

---

## Looking Ahead

This lecture settles the pipeline, the six properties, the three measurement families, and the bound $\text{Loss}_{\text{Test}} \le \text{Loss}_{\text{Train}} + \frac{1}{\sqrt{N}}\frac{\beta\sqrt{S}}{\gamma}$ with the two levers it exposes. Four questions are left open, and the remaining lectures take them in order.

- *How is the distance between two distributions measured from samples alone?* Section 3 named the measurement families but gave no instrument. [[ATML_Lecture_2 - Information Theory and Divergences]] supplies them: entropy, mutual information, the $f$-divergences, the integral probability metrics, and optimal transport.
- *In what ways can the encoder $g_\theta$ be trained?* [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] answers with three families — discriminative, contrastive, generative — each optimizing a different information-theoretic quantity over the same pipeline.
- *How is capacity reduced without reducing fit?* [[ATML_Lecture_4 - GANs, and Inductive Biases]] answers with inductive bias: structure built into the architecture so that less capacity is required in the first place.
- *What happens when the IID assumption of Section 5.1 fails?* Section 5.3 flagged the theory as nascent; [[ATML_Lecture_5_1 - Domain Adaptation]] and [[ATML_Lecture_5_2 - Domain Generalization]] make it the subject, replacing the single-distribution bound with one carrying an explicit domain-discrepancy term.

---

## Numerical Problems

These exercise the lecture's three quantitative results: the capacity/margin bound of Section 5.2, the cross-entropy forms of Sections 7.3–7.4, and the moment definitions of Section 8. Problems 1 and 2 are paired, the first computing the capacity the second substitutes into the bound; Problems 3 and 4 compute the same number two ways, which checks whether the algebra of Section 7.4 is understood rather than memorized.

**Problem 1 — Capacity of a small network.** A 3-layer network has weight matrices with spectral norms $\lVert W_1\rVert_2 = 2$, $\lVert W_2\rVert_2 = 1.5$, $\lVert W_3\rVert_2 = 1$, and Frobenius norms $\lVert W_1 \rVert_F = 4$, $\lVert W_2 \rVert_F = 3$, $\lVert W_3 \rVert_F = 2$. Compute $\beta$, $S$, and the capacity $\beta\sqrt{S}$.

*Solution.* From the definition $\beta = \prod_\ell \lVert W_\ell\rVert_2$, the spectral norms multiply to $\beta = 2 \times 1.5 \times 1 = 3$. For $S = \sum_\ell \lVert W_\ell\rVert_F^2 / \lVert W_\ell \rVert_2^2$, each layer contributes the squared ratio of its two norms: $16/4 = 4$, $9/2.25 = 4$, and $4/1 = 4$, so $S = 12$. Combining these, $\beta\sqrt{S} = 3\sqrt{12} = 3(2\sqrt{3}) = 6\sqrt{3}$.

$$
\boxed{\beta = 3, \quad S = 12, \quad \text{Capacity} = 6\sqrt{3} \approx 10.39}
$$

*Reasonableness check:* $S = 12$ sits between its theoretical bounds — $S \geq L = 3$ always (since $\lVert W\rVert_F \geq \lVert W\rVert_2$), with equality only for rank-one layers — so a value of 12 for three layers indicates each layer has several comparably-sized singular directions, which is typical of a genuinely wide network.

**Problem 2 — Using the generalization bound.** A model trained on $N = 10{,}000$ examples achieves $\text{Loss}_{\text{Train}} = 0.15$, has capacity $\beta\sqrt{S} = 20$, and achieves a margin $\gamma = 0.5$. Bound its test loss. Then determine what margin would be needed to halve the generalization term.

*Solution.* Substituting into the boxed bound of Section 5.2, the generalization term is $\frac{1}{\sqrt{N}}\cdot\frac{\beta\sqrt{S}}{\gamma} = \frac{1}{100}\cdot\frac{20}{0.5} = \frac{40}{100} = 0.4$, so

$$
\text{Loss}_{\text{Test}} \le 0.15 + 0.4.
$$

$$
\boxed{\text{Loss}_{\text{Test}} \le 0.55}
$$

Because the generalization term is inversely proportional to $\gamma$, halving it requires doubling the margin, $\gamma = 1.0$, giving a generalization term of $0.2$ and a bound of $0.35$.

*Reasonableness check:* the bound $0.55$ exceeds the training loss, as it must — the bound is only useful when the generalization term is small relative to the training loss, which here it is not, signalling that this model is under-regularized for its sample size.

**Problem 3 — Cross-entropy from logits.** A 4-class classifier produces logits $s = (2.0,\, 1.0,\, 0.5,\, -1.0)$ for an example whose true class is $j = 2$ (i.e. the logit $s_2 = 1.0$). Compute $\mathcal{L}_{\text{CE}}$ in nats.

*Solution.* Using the log-sum-exp form $\mathcal{L}_{\text{CE}} = -s_j + \log\sum_k e^{s_k}$, the exponentials are $e^{2.0} \approx 7.389$, $e^{1.0} \approx 2.718$, $e^{0.5} \approx 1.649$, and $e^{-1.0} \approx 0.368$, summing to $12.124$. Taking the natural logarithm, $\log(12.124) \approx 2.495$, and subtracting the true-class logit $s_2 = 1.0$,

$$
\mathcal{L}_{\text{CE}} = -1.0 + 2.495.
$$

$$
\boxed{\mathcal{L}_{\text{CE}} \approx 1.495 \text{ nats}}
$$

*Reasonableness check:* the implied predicted probability of the true class is $e^{-1.495} \approx 0.224$, which matches $e^{1.0}/12.124 = 2.718/12.124 \approx 0.224$ computed directly from softmax — and it is below $1/K = 0.25$, correctly reflecting that the model currently favours class 1 over the true class 2.

**Problem 4 — The similarity-based form.** For the same example as Problem 3, verify the ratio form $\mathcal{L}_{\text{CE}} = \log\left(1 + \sum_{k\neq j} e^{s_k - s_j}\right)$.

*Solution.* Using $\delta(z,\psi_k) = -s_k$, the ratio $e^{-\delta(z,\psi_k)}/e^{-\delta(z,\psi_j)} = e^{s_k - s_j}$. With $s_j = 1.0$, the three non-true-class terms are $e^{2.0-1.0} = e^{1.0} \approx 2.718$, $e^{0.5-1.0} = e^{-0.5} \approx 0.607$, and $e^{-1.0-1.0} = e^{-2.0} \approx 0.135$, summing to $3.460$. Adding the leading $1$ and taking the logarithm, $\log(4.460) \approx 1.495$.

$$
\boxed{\mathcal{L}_{\text{CE}} \approx 1.495 \text{ nats — identical to Problem 3}}
$$

*Reasonableness check:* the two forms agree exactly, as they must, since the ratio form is obtained from the log-sum-exp form by dividing numerator and denominator by $e^{s_j}$. Note also that the loss is bounded below by $\log 1 = 0$, attained only when every competing logit is infinitely far below $s_j$.

**Problem 5 — Covariance matrix from a small sample.** Three 2-dimensional samples are observed: $x^{(1)} = (1,2)^\top$, $x^{(2)} = (3,4)^\top$, $x^{(3)} = (5,0)^\top$. Compute the sample mean $\mu_x$ and the covariance matrix $R_x$ (using the $1/N$ normalization consistent with the definition in Section 8).

*Solution.* The sample mean is the componentwise average, $\mu_x = \frac{1}{3}\left((1,2) + (3,4)(5,0)\right)^\top = (3, 2)^\top$. Centering each sample gives $(-2, 0)^\top$, $(0, 2)^\top$, and $(2, -2)^\top$. Applying $R_x = \frac{1}{N}\sum_i (x^{(i)}-\mu_x)(x^{(i)}-\mu_x)^\top$, the three outer products are

$$
\begin{bmatrix}4&0\\0&0\end{bmatrix},\qquad
\begin{bmatrix}0&0\\0&4\end{bmatrix},\qquad
\begin{bmatrix}4&-4\\-4&4\end{bmatrix};
$$

summing these and dividing by $3$,

$$
\boxed{\mu_x = \begin{bmatrix}3\\2\end{bmatrix}, \qquad R_x = \frac{1}{3}\begin{bmatrix}8 & -4 \\ -4 & 8\end{bmatrix} = \begin{bmatrix}8/3 & -4/3 \\ -4/3 & 8/3\end{bmatrix}}
$$

*Reasonableness check:* $R_x$ is symmetric with positive diagonal entries, as any covariance matrix must be. The negative off-diagonal entry indicates the two dimensions are negatively correlated in this sample — visible directly in the data, where the largest first coordinate ($5$) pairs with the smallest second coordinate ($0$). Per Section 2 of Lecture 2, a covariance matrix like this one (non-zero off-diagonals) is exactly what we do *not* want in a learned latent space.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| General deep model pipeline | $x \xrightarrow{g_\theta} z \xrightarrow{h_\psi} \hat{y}$ |
| Model as composite function | $Y = f(x)$, $f = g_\theta \circ h_\psi$, $W = \{\theta,\psi\}$ |
| Supervised dataset | $\mathcal{D} = \{\vec{x}_i, y_i\}_{i=1}^{N}$ |
| Unsupervised dataset | $\mathcal{D} = \{\vec{x}_i\}_{i=1}^{N}$ |
| Robustness / smoothness criterion | $g(x + \delta x) \approx g(x)$ |
| Deep approximation of $P(X)$ via $L$ layers | $z^{(1)} \to z^{(2)} \to \cdots \to z^{(L)}$, $P(z^{(L)}) = z$ |
| $P(X)$ vs. empirical histogram | $P(x) \neq$ histogram of $\{x_i\}_{i=1}^N$ |
| IID assumption | $P_{\text{Test}}(X,Y) = P_{\text{Train}}(X,Y)$ |
| OOD setting | $P_{\text{Test}}(X,Y) \neq P_{\text{Train}}(X,Y)$ |
| General generalization bound | $\text{Loss}_{\text{Test}} \le \text{Loss}_{\text{Train}} + \frac{1}{\sqrt{N}}\cdot\frac{\text{Complexity}}{\text{Margin}}$ |
| Amplification factor | $\beta = \prod_{\ell=1}^{L}\lVert W_\ell\rVert_2$ |
| Spread factor | $S = \sum_{\ell=1}^{L}\frac{\lVert W_\ell\rVert_F^2}{\lVert W_\ell\rVert_2^2}$ |
| Capacity | $\text{Capacity} = \beta\sqrt{S}$ |
| Refined generalization bound (eq. 1) | $\text{Loss}_{\text{Test}} \le \text{Loss}_{\text{Train}} + \frac{1}{\sqrt{N}}\cdot\frac{\beta\sqrt{S}}{\gamma}$ |
| Simplified complexity term | $\text{Complexity}\propto\sum_{i=1}^{L}\lVert W_i\rVert_2^2$ |
| Dimensionality reduction condition | $x\in\mathbb{R}^{D\times1}$, $z\in\mathbb{R}^{d\times1}$, $d \ll D$ |
| Classifier head (matrix form) | $\hat{y} = \text{Softmax}(\Psi z)$, $\Psi \in \mathbb{R}^{K\times d}$ |
| Softmax classifier output | $\hat{y}_k = \text{softmax}(\psi_k^\top z) = P(\hat{Y}=k\mid x)$ |
| Similarity convention | $\delta(\psi_k, z) = -\psi_k^\top z$ |
| Cross-entropy loss (general) | $\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k\log\hat{y}_k$ |
| Cross-entropy loss (one-hot) | $\mathcal{L}_{\text{CE}} = -\log\hat{y}_j$ |
| Cross-entropy (log-sum-exp form) | $\mathcal{L}_{\text{CE}} = -s_j + \log\left(\sum_{k=1}^{K}e^{s_k}\right)$ |
| Cross-entropy (similarity form) | $\mathcal{L}_{\text{CE}} = \delta(\psi_j,z) + \log\left(\sum_{k=1}^{K}e^{-\delta(\psi_k,z)}\right)$ |
| Cross-entropy (ratio form, per example) | $\mathcal{L}_{\text{CE}_i} = \log\left(1+\sum_{k\neq j}\frac{e^{-\delta(z_i,\psi_k)}}{e^{-\delta(z_i,\psi_j)}}\right)$ |
| Mean | $\mu_x = \mathbb{E}[x] = \int x P(x)\,dx = \sum_i x_i P(x_i)$ |
| Covariance matrix | $R_x = \mathbb{E}[(x-\mu_x)(x-\mu_x)^\top]$, $d\times d$ |
| Expectation of a function | $\mu_a = \mathbb{E}[b(x)] = \int b(x)P(x)\,dx$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the six representation-quality properties** — *"**S**ome **M**odels **G**enerate **D**eeply **S**mooth **D**escriptions"*: **S**ufficiency, **M**inimality, **G**roup/geometry, **D**isentanglement, **S**moothness, **D**imensionality reduction. The three measurement lenses are **I-G-G**: **I**nformation-theoretic, **G**eometric, **G**roup-theoretic.

**Quick-reference intuition per topic**

- **Pipeline:** every deep model = encoder (learns representation $z$) + task head (maps $z \to \hat{y}$). The encoder is where representation learning happens; the head is task-specific.
- **Everything has a distribution:** data ($P(X), P(Y), P(X,Y), P(X|Y), P(Y|X)$), latents ($P(z), P(z|x)$), and even parameters ($P(\theta), P(\psi)$).
- **$P(X)$ vs $P(Y)$ vs $P(X,Y)$:** $P(X)$ is a messy, multimodal, high-dimensional density over raw inputs; $P(Y)$ is a simple discrete categorical; $P(X,Y)$ is the informative but intractable joint — approximable only via deep, layered, non-linear models.
- **Histogram trap:** the empirical histogram of your dataset is *not* $P(X)$. It is a finite sample from it, with zero mass everywhere your data happened not to land.
- **The bound's three moving parts:** more data ($1/\sqrt{N}$) helps, more capacity ($\beta\sqrt{S}$) hurts, more margin ($\gamma$) helps. Complexity and margin sit on opposite sides of a fraction — don't flip which one you want large.
- **$\beta$ vs $S$:** $\beta$ is the *product* of spectral norms (worst-case amplification through depth); $S$ is the *sum* of Frobenius-to-spectral ratios (how many directions each layer acts on meaningfully). Capacity multiplies them as $\beta\sqrt{S}$.
- **Three routes to a tighter bound:** compression & flatness, inductive bias, invariance-driven generalization — all three are what "good representation learning" means in this course.
- **Spurious correlation trap:** low training loss can coexist with total OOD failure (the watermark example). The fix is domain-invariant features, not more in-domain data.
- **Manifold hypothesis in one sentence:** high ambient dimension $D$ does not imply high intrinsic dimension $d$ — folded paper still has intrinsic dimension 2 however many ambient coordinates it seems to occupy.
- **Softmax head:** the linear layer before softmax has **no activation function** — the non-linearity is entirely in the softmax step.
- **Cross-entropy, one line:** minimizing $\mathcal{L}_{\text{CE}}$ = maximizing the true-class logit $\psi_j^\top z$ while suppressing every other class's logit via the log-sum-exp term.
- **Class prototypes:** the rows $\psi_k$ of $\Psi$ are fixed "directions" per class; training rotates $z$ toward its own class direction and orthogonal to all others. The angular separation this creates *is* the margin $\gamma$ from the generalization bound.
- **Sign convention:** $\delta = -\psi^\top z$, so **smaller $\delta$ means more similar**. Getting this sign backwards inverts the whole similarity-form derivation.

**Recall checklist**

- [ ] Can you name the six modules and the one-sentence theme that unifies them?
- [ ] Can you write the generic pipeline from memory and say which component does representation learning?
- [ ] Can you list the six properties of a good representation and the three measurement lenses?
- [ ] Can you write $\beta$, $S$, and $\beta\sqrt{S}$ from memory, and state the boxed bound (1)?
- [ ] Can you name the two levers available at fixed $N$, and the three ways to control the complexity/margin ratio?
- [ ] Can you explain why $P(x)$ is not the histogram of the dataset?
- [ ] Can you give the watermark example and define a domain-invariant feature?
- [ ] Can you explain the manifold hypothesis using the folded-paper analogy?
- [ ] Can you derive $\mathcal{L}_{\text{CE}} = -s_j + \log\sum_k e^{s_k}$ and then the ratio form $\log(1 + \sum_{k\neq j} e^{s_k - s_j})$?
- [ ] Can you state, geometrically, what minimizing cross-entropy does to $z$ relative to the class prototypes?
- [ ] Can you write the definitions of $\mu_x$, $R_x$, and $\mathbb{E}[b(x)]$ from memory?

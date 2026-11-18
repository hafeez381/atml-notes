---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module II -DG.pdf]]'
tags:
- atml
- course/cs6304
- disentanglement
- domain-generalization
- self-supervised-learning
title: ATML — Lecture 5.2 — Domain Generalization
module: Module II
lecture: 5.2
aliases:
- DG
- Module II DG
- ATML Lecture 5.2
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_5_1 - Domain Adaptation]] · [[ATML_Lecture_6 - Open-Set Recognition]] →
Concepts: [[Domain Generalization]]
%% nav:end %%

# ATML — Lecture 5.2: Domain Generalization (Module II)

## Where This Lecture Sits

*Continues from [[ATML_Lecture_5_1 - Domain Adaptation]] by deleting one assumption. There, unlabelled target data was available and the divergence term of the DA (domain adaptation) bound could be measured and minimized directly; here the target domain is absent at training time, so that term cannot be evaluated at all.*

The DG bound of Section 1 exposes three levers, and each method family pulls one: **domain alignment** (Section 2) shrinks the source hull radius $\rho$, **learning strategies** (Section 3) seek solutions stable under the perturbation an unseen domain represents, and **feature disentanglement** (Section 4) separates causal from domain-specific factors — the constructive form of Lecture 5.1 Section 4.4's feature taxonomy. **Self-supervised learning** (Section 5) is cross-cutting: it supplies the representation all three need when labels are scarce, reusing the contrastive material of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] Section 3.

## Topic Inventory

- 1. Domain Generalization — Setup and Error Bound
  - 1.1 Definition and Goal
  - 1.2 The DG Error Bound
  - 1.3 What We Can Do About the Bound
- 2. Family 1 — Domain Alignment
  - 2.1 Measure-Based Alignment
  - 2.2 Adversarial-Based Alignment
  - 2.3 A Combined Alignment Architecture
  - 2.4 Contrastive-Based Alignment
  - 2.5 The Hard-Positives Problem
  - 2.6 Hard Positives Also Waste the Negative Gradients
  - 2.7 The Fix — Proxies for Positive Samples
  - 2.8 Proxy-Based Contrastive Loss
- 3. Family 2 — Learning Strategies
  - 3.1 Invariant Risk Minimization — Idea and Intuition
  - 3.2 Vanilla IRM and Why It Is Intractable
  - 3.3 IRMv1 — A Practical Surrogate
  - 3.4 The Scale-Probe Idea
  - 3.5 Issues with the First-Order Surrogate, and Alternatives
  - 3.6 Sharpness-Aware Minimization — Idea and Formulation
  - 3.7 Approximating the Worst-Case Perturbation by Linearization
  - 3.8 Including Curvature Information
  - 3.9 Challenges of SAM for DG
  - 3.10 Stochastic Weight Averaging and SWAD
  - 3.11 Gradient Alignment or Masking Across Domains
  - 3.12 Refining Each Domain's Loss Landscape with Soft Labels
  - 3.13 Group Distributionally Robust Optimization
  - 3.14 DRO over Perturbation, and CCFP
- 4. Family 3 — Feature Disentanglement and Disentangled Representation Learning
  - 4.1 Motivation — Humans versus Deep Models
  - 4.2 Three Benefits of DRL
  - 4.3 Definition — Hidden Generative Factors
  - 4.4 DRL Structure: Dimension-Wise
  - 4.5 DRL Structure: Vector-Wise
  - 4.6 Design of the Loss Function
  - 4.7 Formal Setup — Independent and Dependent Factors
  - 4.8 The VAE as the Vehicle for DRL
  - 4.9 The Isotropic Gaussian Prior and the VAE Objective
  - 4.10 β-VAE
  - 4.11 Training β-VAE
  - 4.12 Verifying Disentanglement — Visual Inspection
  - 4.13 Verifying Disentanglement — the Z-Diff (β-VAE) Metric
  - 4.14 Verifying Disentanglement — Mutual Information Gap
  - 4.15 Cons of β-VAE
  - 4.16 Capacity Control
  - 4.17 Other Generative Models for DRL
  - 4.18 Incorporating Label Information — Semi-Supervised β-VAE
  - 4.19 Vector-Wise DRL Architecture
- 5. Self-Supervised Learning
  - 5.1 Motivation
  - 5.2 Supervised and Unsupervised Models Side by Side
  - 5.3 The SSL Recipe — Pretext Task and Pseudo-Labels
  - 5.4 Downstream Adaptation — Fine-Tuning versus Linear Readout
  - 5.5 Summary Pipeline, and Transfer Learning versus SSL
  - 5.6 Designing Pretext Tasks
  - 5.7 Inferring Structure — Context Prediction
  - 5.8 Shortcut Learning and How to Avoid It
  - 5.9 Inferring Structure — Jigsaw Puzzles
  - 5.10 Inferring Structure — Counting Features
  - 5.11 Transformation Prediction — Rotation
  - 5.12 Reconstruction
  - 5.13 Colorization and Split-Brain Decoders
  - 5.14 SSL versus Unsupervised Learning
  - 5.15 Instance Classification
  - 5.16 Contrastive Instance Discrimination
  - 5.17 Clustering
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Domain Generalization — Setup and Error Bound

### 1.1 Definition and Goal

> **Domain Generalization (DG):** improve model generalization on a target domain $\mathcal{D}_T$, with the model having access only to $M$ source domains $\mathcal{D}_i$, $i = 1, \ldots, M$, as labelled datasets.

This is the harder sibling of Domain Adaptation from Lecture 5.1. In DA the unlabelled target data was available at training time and could be aligned against; in DG it is simply absent. All $M$ source domains, each with labelled data, are pooled and fed into a single training run:

$$\underbrace{\{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_M\}}_{\text{source domains with labelled data}} \;\longrightarrow\; \boxed{\text{Train } f_\theta(\cdot)} \;\longrightarrow\; \text{deploy on } \mathcal{D}_T$$

**Goal:** good prediction performance on the $\mathcal{D}_T$ datasets, assuming

$$P_T(x, Y) \neq P_i(x, Y), \qquad i = 1, 2, \ldots, M.$$

Two distinct inequalities are in play and they should not be conflated. The target differs from every source, which is the whole difficulty; but additionally the sources differ from *each other*,

$$P_i(x, Y) \neq P_j(x, Y), \qquad i \neq j,$$

and that second fact is what makes DG possible at all. If all sources were identical there would be nothing to learn about *what varies across domains*, and hence no basis for guessing what will vary again at test time. Source diversity is the training signal.

### 1.2 The DG Error Bound

**Assumption.** The target marginal lies inside the convex hull of the source marginals:

$$P_T(x) \in \Lambda = \text{convex hull of the source marginals } \{P_i(x)\}_{i=1}^{M}.$$

In words: the target domain is assumed to be expressible as some mixture of the source domains, not as something wholly outside their span. This is the formal statement of "we cannot generalize to a domain unlike anything we have seen."

Under that assumption, the target error of a model $f_\theta(\cdot)$ is bounded by

$$\boxed{L_T(f_\theta) \;\le\; \sum_{i=1}^{M} \pi_i\, L_i(f_\theta) \;+\; \gamma\!\left(P_T(x),\, \underbrace{\sum_{i=1}^{M} \pi_i\, P_i(x)}_{=\,\Lambda}\right) \;+\; \rho \;+\; \text{const.}}$$

with the two new quantities read as follows. The term $\gamma$ is the **minimum discrepancy between $P_T$ and $\Lambda$** — how far the target sits from the best achievable mixture of sources. The term $\rho$ **captures how far apart the source domains are**; concretely, it is the **radius of the convex hull**. The mixture weights $\pi_i$ are the convex coefficients defining which point in $\Lambda$ we compare against.

The structure mirrors the DA bound of Lecture 5.1: a weighted source-risk term, a distributional-discrepancy term, and a residual that no amount of training removes. The difference is that $\gamma$ is not directly minimizable here — the target is unavailable — so the design effort shifts onto the terms that *are* controllable.

*Intuition.* The convex-hull assumption is doing more work than its one line suggests, and it is the honest statement of what DG can and cannot promise. If the target lies inside the hull, it is some mixture of conditions already witnessed — sketches and cartoons when the sources include photographs, paintings, and line drawings — and a model covering the hull covers it. If the target lies outside, $\gamma$ is large by construction and the bound is vacuous: no method in this lecture can generalize to a domain unlike anything seen, and a method appearing to do so has benefited from a hull wider than it looked.

### 1.3 What We Can Do About the Bound

Three levers follow directly from reading the three terms:

1. **Reduce all individual source risks $L_i(f_\theta)$.** This is ordinary supervised training, done well, on every source.
2. **Shrink the source–source distance $\rho$.** Pulling the source representations together shrinks the convex hull's radius, which shrinks the residual term.
3. **Reduce $\gamma(\cdot,\cdot)$** — that is, **learn a representation where a (convex) mixture of source marginals covers future targets.** Since we cannot move $P_T$, the only move available is to reshape the representation space so that the hull spanned by the sources is as *inclusive* as possible.

**There are many families of DG methods and techniques, based on different ideas.** The three developed in this lecture are domain alignment, learning strategies, and feature disentanglement — with self-supervised learning treated as a fourth, cross-cutting theme.

---

## 2. Family 1 — Domain Alignment

> **Domain alignment:** align all $\mathcal{D}_i$, $i = 1, 2, \ldots, M$, to capture feature representations that are **domain-invariant**.

This attacks lever 2 of Section 1.3 directly — squeezing the sources together in feature space so that whatever remains is the part they all share.

### 2.1 Measure-Based Alignment

The measure-based route is to **reduce a probability distance measure between all pairs of source domains**. The canonical choice is Maximum Mean Discrepancy, applied pairwise:

$$\text{MMD}(\mathcal{D}_i, \mathcal{D}_j) \quad \text{for every pair } (i, j).$$

Note the difference from the DA case: in DA there was one source and one target and therefore one alignment term, while here there are $\binom{M}{2}$ pairs among the sources and no target term at all.

### 2.2 Adversarial-Based Alignment

The adversarial route is to **learn features that hide domain identity while keeping labels predictable**. This is the DANN (domain-adversarial neural network) idea from Lecture 5.1 generalized from a binary source-versus-target discriminator to an $M$-way domain discriminator: the discriminator tries to name which source domain a feature came from, and the feature extractor tries to make that impossible while the label head keeps working.

### 2.3 A Combined Alignment Architecture

The professor's notes draw a single architecture that stacks several of these alignment pressures at once. The $M$ source domains each pass through a shared feature extractor $\theta_f$ producing latents $z_1, z_2, \ldots, z_M$. These feed an encoder $Q$, producing intermediate codes $H_1, H_2, \ldots, H_M$, which a decoder $P$ reconstructs into $\hat{z}_1, \hat{z}_2, \ldots, \hat{z}_n$, scored by an autoencoder loss $L_{\text{AE}}$:

![[03 - Attachments/ATML_L5_2_alignment_architecture.png|620]]

*The professor's sketch: $M$ source domains share the extractor $\theta_f$; the codes $H_i$ feed the decoder ($L_{\text{AE}}$), the pairwise-MMD and supervised branches, and a discriminator that compares them with the prior $P(H)$.*

$$\{\mathcal{D}_1, \ldots, \mathcal{D}_M\} \to \boxed{\text{Feature Extractor } \theta_f} \to \{z_1, \ldots, z_M\} \to \boxed{\text{Encoder } Q} \to \{H_1, \ldots, H_M\} \to \boxed{\text{Decoder } P} \to \{\hat{z}_1, \ldots, \hat{z}_n\} \to L_{\text{AE}}$$

Branching off the $H_i$ codes are three further losses. A **pairwise MMD loss** $L_{\text{MMD}}$ aligns the codes across domains in the measure-based sense of Section 2.1. A **supervised loss** $L_{\text{sup}}$ keeps the representation label-predictive. And a **discriminator** with parameters $\theta_d$, fed both the $H_i$ codes and a **prior $P(H)$ on $H_i$**, produces the adversarial loss $L_D$ — pushing the aggregate code distribution toward the chosen prior rather than merely toward each other.

The total objective is the sum of all four:

$$\boxed{L_{\text{Total}} = L_{\text{sup}} + L_{\text{AE}} + L_{\text{MMD}} + L_D}$$

Each term is doing distinct work: $L_{\text{sup}}$ preserves task information, $L_{\text{AE}}$ preserves input information (guarding against the alignment terms collapsing the representation), $L_{\text{MMD}}$ aligns domains explicitly, and $L_D$ aligns them implicitly against a prior.

### 2.4 Contrastive-Based Alignment

The contrastive route pulls same-class samples together *across* domains and pushes different-class samples apart. Given data drawn from $\mathcal{D}_i$, $i = 1, 2, \ldots, M$, as pairs $\{x_j, y_j\}$, each input passes through $f_\theta(\cdot)$ to a latent $z$, and the loss is

$$\boxed{L_{\text{CL}} = -\log\left(\frac{\exp\big(\text{sim}(z_a, z_+)\big)}{\sum_k \exp\big(\text{sim}(z_a, z_k)\big) + \exp\big(\text{sim}(z_a, z_+)\big)}\right)}$$

with the pairing rule made explicit: **if $z_a \in \mathcal{D}_i$, then $z_+ \in \mathcal{D}_j$ with $j \neq i$**, and the $z_k$ are **negatives drawn from both domains**, subject to $\text{class}_k \neq \text{class}_a$. The cross-domain requirement on the positive is the essential part — a positive drawn from the *same* domain would teach class discrimination without teaching domain invariance.

The one-line summary given in the notes: **pull positives across domains together and push negatives far apart.** Computationally, $L_{\text{CL}}$ is evaluated with the anchor $z_a$, its positive $z_+$, and the full set $\{z_1, z_2, \ldots, z_N\}$ of all negatives.

### 2.5 The Hard-Positives Problem

> [!warning] The central failure mode of cross-domain contrastive learning **However, we face the issue of hard positives.**

![[03 - Attachments/ATML_L5_2_hard_positives.png|600]]

*The bicycle example: a line-art anchor and a textured photo positive are far apart because of style, and pulling them together collapses the two domains' sub-manifolds of class $j$.*

**Two images of the same class but from different domains can look very different.** The worked example is the **bicycle class**: a **thin-line art** rendering gives anchor $z_a$, while a **textured kid's bicycle** photograph gives positive $z_+$. **The visual gap is huge.** Critically, they are **far apart because of domain style, and not because of class semantics.**

$$\boxed{\textbf{Hard positives: they are already far apart in the latent space.}}$$

**$L_{\text{CL}}$ gives a very large pull to such hard positives, in order to bring them closer.** That pull is exactly what the loss is designed to do — but **this pull becomes incompatible with preserving other positives (from a different domain) with their own local structure.**

The geometric picture: class $j$ occupies **sub-manifolds across two domains** — one blob per domain, each with its own internal arrangement. A hard positive pair sits at opposite ends of a **large separation**, and the loss **wants to collapse them into a single manifold**. Doing so **distorts the local geometries** of both sub-manifolds. The loss achieves domain invariance at the price of destroying the within-domain structure that the representation also needs.

### 2.6 Hard Positives Also Waste the Negative Gradients

> [!warning] Second consequence **Another issue with hard positives: they create huge positive gradients, making negative samples useless.**

The argument is a short gradient calculation. Let

$$s_k = \text{sim}(z_a, z_k), \qquad s_{\text{pos}} = \text{sim}(z_a, z_+), \qquad P_k = \frac{\exp(s_k)}{\sum_j \exp(s_j)}, \qquad P_{\text{pos}} = \frac{\exp(s_{\text{pos}})}{\sum_j \exp(s_j)}.$$

Writing the contrastive loss for the anchor $a$ in log-sum-exp form (the same manipulation used for cross-entropy in Lecture 1),

$$L_{\text{CL}}(a) = -\log P_{\text{pos}} = -s_{\text{pos}} + \log\left(\sum_j e^{s_j}\right).$$

Differentiating with respect to any individual similarity score gives the standard softmax gradient, which splits into two cases:

$$\frac{\partial L_{\text{CL}}}{\partial s_i} = \begin{cases} P_{\text{pos}} - 1 & \text{if } i = \text{pos},\\[1ex] P_i & \text{if } i \text{ is any negative,}\end{cases}$$

and because the softmax probabilities are normalized,

$$\boxed{P_{\text{pos}} + \sum_k P_k = 1} \qquad \text{— due to the InfoNCE nature of the loss.}$$

Now summing the gradients over all negatives and substituting the normalization identity,

$$\sum_{i,\, i \neq \text{pos}} \frac{\partial L}{\partial s_i} = \sum_{i,\, i \neq \text{pos}} P_i = 1 - P_{\text{pos}} = \big|P_{\text{pos}} - 1\big| = \left|\frac{\partial L_{\text{CL}}}{\partial s_{\text{pos}}}\right|,$$

which gives the boxed conclusion:

$$\boxed{\textbf{The total push on all negative samples equals the magnitude of the positive's pull.}}$$

This is **true for contrastive learning in general**, not just in the DG setting — it is a structural property of the softmax, not an artifact of this application.

The consequence is severe. **This automatic coupling makes a single hard positive pair dictate big global moves in the space**, because the positive's enormous pull is mirrored by an equally enormous total push distributed across the negatives — negatives that may already be perfectly well placed. The result is **wasted gradients and distortion of already-aligned geometry.**

### 2.7 The Fix — Proxies for Positive Samples

**How to solve it? Use proxies for positive samples.**

![[03 - Attachments/ATML_L5_2_proxy_vs_contrastive.png|600]]

*Contrastive loss (anchor vs. one positive and all negatives) beside cross-entropy (anchor vs. class prototypes); the proxy $w_c$ passes through a small head $h_c$ to give $v_c$.*

The insight comes from placing the contrastive loss and the cross-entropy loss side by side and noticing they have the same shape. In the **contrastive loss**, the anchor $z_a$ is compared against one positive $z_+$ and the set of all negatives $\{z_1, z_2, \ldots, z_N\}$. In the **cross-entropy loss**, the anchor $z_a$ is compared against its own class vector $\psi_a$ and the remaining **class prototypes** $\{\psi_1, \psi_2, \ldots, \psi_{c-1}\}$:

$$L_{\text{CE}} = -\psi_a^{\top} z_a + \log\left(\sum_{k=1}^{c-1} e^{\psi_k^{\top} z_a}\right)$$

and the observation that makes the whole method work is that the **class prototypes are equivalent to the negatives in the contrastive loss**. A prototype is a single stable vector standing in for an entire class, so it cannot be a wild outlier the way an individual hard positive can.

A **proxy for class $c$**, written $w_c$, is therefore **a learnable vector which is trained by the contrastive loss**. It is passed through **a small projection head** $h_c(\cdot)$ to give the usable proxy embedding:

$$v_c = h_c(w_c), \qquad c = 1, 2, \ldots, K.$$

> [!note] Notation The projection head is written $h_c(\cdot)$ when the proxy is introduced and $g_c(\cdot)$ in the loss that follows. Both refer to the same small projection head; $g_c(\cdot)$ is used from here on, matching the update rule the professor lists at the end of Section 2.8.

### 2.8 Proxy-Based Contrastive Loss

Let $x_i$ be an **anchor with class $c$**, with **proxy $w_c$** and projected proxy $v_c = g_c(w_c)$. The **proxy-based contrastive loss (PCL)** is

$$\boxed{L_{\text{PCL}} = -\frac{1}{N}\sum_{i=1}^{N} \frac{\exp\big(\text{sim}(v_c,\, z_i)\big)}{E}}$$

where the denominator collects the anchor's own proxy, the other class proxies, and the other sample embeddings:

$$E = \underbrace{\exp\big(\text{sim}(v_c, z_i)\big)}_{\text{own proxy}} + \underbrace{\sum_{K=1}^{c-1}\exp\big(\text{sim}(v_K, z_i)\big)}_{(c-1)\text{ other proxies}} + \underbrace{\sum_{\substack{j=1 \\ j \neq i}}^{N}\exp\big(\text{sim}(z_i, z_j)\big)}_{(N-1)\text{ negative samples}}$$

The composition is drawn explicitly: the anchor $z_a$ is contrasted against $w_a$ on the positive side, and against the union $\{w_1, w_2, \ldots, w_{c-1}\} \cup \{z_1, z_2, \ldots, z_{N-1}\}$ — that is, **$(c-1)$ proxies and $(N-1)$ negative samples** — on the negative side. As with the embeddings elsewhere in this course, **$z_i$ and $z_j$ can also be passed through a small projection head** before the similarity is computed.

> [!warning] Possible source inconsistency $L_{\text{PCL}}$ is written as a bare ratio $-\frac{1}{N}\sum_i \frac{\exp(\cdot)}{E}$ without a logarithm, whereas every other contrastive loss in this course (InfoNCE in Lecture 3, $L_{\text{CL}}$ in §2.4, $L_{\text{const.}}$ in §5.16) takes $-\log$ of the ratio. The missing $\log$ is almost certainly a transcription slip; the professor's form is reproduced above as written, but read it as $-\frac{1}{N}\sum_i \log\frac{\exp(\text{sim}(v_c, z_i))}{E}$.

The overall objective combines the ordinary classification loss with the proxy contrastive term,

$$\boxed{L_{\text{Total}} = L_{\text{CE}} + L_{\text{PCL}}}$$

and the update touches three parameter groups: **$\theta$** (the feature extractor), **$w_c$** (the proxies themselves), and **$g_c(\cdot)$** (the projection head). Batches are constructed to span the domains, drawn as a single batch partitioned into contiguous blocks for $\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_M$ — so that every gradient step sees cross-domain pairs rather than accidentally training on one domain at a time.

---

## 3. Family 2 — Learning Strategies

> **In addition to designing specific models for domain alignment, DG can be obtained through designing new learning strategies.**

Where Family 1 changes *what the representation must look like*, Family 2 changes *what the optimization is asked to achieve*. The two are complementary and are often combined.

### 3.1 Invariant Risk Minimization — Idea and Intuition

> **Invariant Risk Minimization (IRM):** learn features that support **a single classifier head to be optimal in every domain.**

The architecture is the familiar one, $x \to g_\theta(\cdot) \to z \to h_w(\cdot) \to \hat{Y}$, but the requirement placed on it is new: the *same* last layer $h_w(\cdot)$ must be simultaneously optimal for all $M$ domains.

**Intuition:** if $g_\theta(\cdot)$ learns **only causal / invariant features** across all source domains, then the **optimal last layer does not need to change across domains.** Read the other way around — which is how it is used as a training signal — if you *force* one head to be optimal everywhere, then the features feeding it must have been invariant, since a head cannot compensate for domain-specific feature drift without changing. The diagram states it compactly: if $z$ is **causal / invariant across $\mathcal{D}_i$** for all $i$, then **the same $h_w(\cdot)$ across all $\mathcal{D}_i$ is good enough.**

This is the practical realization of the "causal and domain-invariant" quadrant of the feature taxonomy from Lecture 5.1.

### 3.2 Vanilla IRM and Why It Is Intractable

Written as a constrained program, vanilla IRM is

$$\boxed{\min_{\theta,\, w} \;\sum_{i=1}^{M} R_i\big(h_w \circ g_\theta\big) \qquad \text{s.t.} \qquad w^{*} = \operatorname*{arg\,min}_{w} R_i\big(h_w \circ g_\theta\big) \;\; \forall i} \tag{1}$$

with $R_i(\cdot)$ denoting the **risk (loss) for domain $\mathcal{D}_i$**, so the objective is the **average risk over all domains**, and the constraint says $w^{*}$ **minimizes each domain's risk when applied on the same features $g_\theta$.**

> [!warning] Why this cannot be optimized directly **(1) is a nasty bi-level non-convex problem** — an **MPEC program** (Mathematical Program with Equilibrium Constraints) — so **we can't solve it directly.**

The reason is worth spelling out, because it explains why every practical IRM variant is a surrogate. Define the per-domain optimal-head set

$$w_i^{*}(\theta) = \operatorname*{arg\,min}_{u} R_i\big(h_u \circ g_\theta\big).$$

The task is then to **find $\theta$ so that the intersection $\bigcap_{i=1}^{M} w_i^{*}(\theta)$ is non-empty, and pick $w$ in that intersection.** But **as $\theta$ moves, each set $w_i^{*}(\theta)$ moves, and the intersection may appear or disappear.** Consequently the **feasible region becomes nonconvex, discontinuous, and there might be no solution under some scenarios.**

**What to do? Find some practical surrogate for the IRM objective.**

### 3.3 IRMv1 — A Practical Surrogate

The surrogate replaces the exact "same argmin" constraint with its **standard first-order optimality condition**. If the same $w^{*}$ minimizes each domain's risk $R_i(h_w \circ g_\theta)$, then

$$\nabla_w R_i\big(h_w \circ g_\theta\big)\Big|_{w = w^{*}} = 0 \quad \forall i.$$

The picture is several per-domain risk curves $R_1, R_2, \ldots, R_M$ plotted against $w$, all of which must be flat at the same point $w^{*}$ — hence "fix $w$" and check that every curve is stationary there.

![[03 - Attachments/ATML_L5_2_irmv1_landscape.png|260]]

The trick that makes this cheap is to **pick a fixed probe value, e.g. $w = 1$**, and form a penalty measuring how far from stationary each domain is at that fixed probe:

$$\sum_{i=1}^{M}\left\lVert \nabla_w R_i\big(h_w \circ g_\theta\big)\Big|_{w=1}\right\rVert^{2}$$

Combining with the ordinary average risk gives **IRMv1** (the practical first-order version of invariant risk minimization):

$$\boxed{\min_{\theta} \;\underbrace{\sum_{i=1}^{M} R_i\big(h_w \circ g_\theta\big)}_{\text{ERM term}} \;+\; \lambda \underbrace{\sum_{i=1}^{M}\left\lVert \nabla_w R_i \Big|_{w=1}\right\rVert^{2}}_{\text{IRMv1 penalty}}}$$

**If we manage to bring this penalty small during training, it pushes the system towards a situation where the same classifier is first-order optimal in every domain.** Note that $w$ is no longer a free variable being optimized — it has been frozen at the probe value and the entire burden has been shifted onto $\theta$, which is precisely what turns a bi-level problem into a single-level one.

### 3.4 The Scale-Probe Idea

Freezing $w$ at a vector value is still awkward; the scale-probe reformulation reduces the probe to a single scalar. Consider the network as an encoder $\theta$ producing $z \in \mathbb{R}^{d \times 1}$, followed by a **linear head** $w \in \mathbb{R}^{d \times c}$ giving $\hat{y}$. Equivalently, absorb the head into the parameters, $\theta : \{\theta, w\}$, and write the head's output as **logits** $\tilde{z} \in \mathbb{R}^{c \times 1}$ feeding a softmax.

Now **modify** the logits by a single **scalar scale factor** $s \in \mathbb{R}$:

$$\hat{z} = s\,\tilde{z}.$$

The per-domain risk becomes a function of $\theta$ and that scalar:

$$\boxed{R_i(\theta, s) = \mathbb{E}_{P_i(x,Y)}\Big[L_{\text{CE}}\big(s\hat{z},\, Y\big)\Big] = \mathbb{E}_{P_i(x,Y)}\Big[L_{\text{CE}}\big(s\, f_\theta(x_i),\, Y\big)\Big]}$$

**Now probing the optimality of a single shared classifier across domains reduces to asking:**

$$\boxed{\textit{"Is the loss stationary if we scale all logits by the same scalar } s\textit{?"}}$$

which yields the final scale-probe objective:

$$\boxed{\min_{\theta}\;\sum_{i=1}^{M} R_i(\theta,\, s=1) \;+\; \lambda\sum_{i=1}^{M}\left\lVert \nabla_s R_i(s, \theta)\Big|_{s=1}\right\rVert_2^{2}}$$

The elegance is that the penalty is now the squared norm of a *scalar* derivative per domain, which costs almost nothing to compute relative to a full per-domain head optimization.

### 3.5 Issues with the First-Order Surrogate, and Alternatives

> [!warning] Spurious stationary solutions **Since we use the $\sum_i \lVert\nabla_s R_i|_{s=1}\rVert^2 \to 0$ condition, which is only a first-order optimality condition, the gradients can be driven to zero without actually finding an aligned optimal $w$ across domains** — producing **spurious stationary solutions.**

A first-order condition certifies stationarity, not optimality: a saddle point, a plateau, or a degenerate representation all satisfy it. The optimizer can therefore satisfy the penalty by making the problem *flat* rather than by making the features *invariant*.

**Solution: search for more stable surrogates.** Three directions are listed:

- **Variance of risk penalties** — REx (risk extrapolation) / VREx (variance risk extrapolation) and similar methods, which penalize the variance of $R_i$ across domains rather than its gradient.
- **Matching per-domain gradient statistics** — Fish and related methods.
- **Entropy measures**, and others.

### 3.6 Sharpness-Aware Minimization — Idea and Formulation

> **Sharpness-Aware Minimization (SAM):** simultaneously minimize the **loss value** and the **loss sharpness**.

**Traditional deep models seek $\theta_{\text{sharp}}$** — whichever parameter vector attains the lowest loss value, regardless of what the loss surface looks like around it. SAM instead **seeks $\theta_{\text{SAM}}$ even if $L(\theta_{\text{SAM}}) \geq L(\theta_{\text{sharp}})$**: a slightly worse loss in a flat basin is preferred to a slightly better loss at the bottom of a narrow crevice.

**Why this gives better generalization:** a **small perturbation in $\theta$ or in the data will be absorbed much more effectively** at $\theta_{\text{SAM}}$ than at $\theta_{\text{sharp}}$. Since a domain shift *is* a perturbation of the data distribution, flatness is a direct proxy for domain robustness — which is why SAM appears in a DG lecture rather than only in a generalization one.

The objective is a min-max problem:

$$\boxed{\theta_{\text{SAM}} = \min_{\theta}\;\underbrace{\max_{\lVert\epsilon\rVert \le \rho} L(\theta + \epsilon)}_{L_{\text{SAM}}}} \tag{1}$$

Reading it: **search for the worst-case perturbation $\epsilon$ within a radius $\rho$ around the current estimate $\theta$**, and then **find $\theta$ that not only minimizes $L(\theta)$ but also remains stable against perturbations in its neighborhood.**

The solution procedure has two nested parts. **For a given $\theta$, solve $\max_{\lVert\epsilon\rVert\le\rho} L(\theta + \epsilon)$**, whose solution is written $\bar{\epsilon}$. **Then for that solution $\bar{\epsilon}$, solve $\min_\theta L_{\text{SAM}} = L(\theta + \bar{\epsilon})$** — the **adversarially perturbed loss around $\theta$.** The effect is **convergence on flatter regions**, characterized by **low worst-case loss in a small neighborhood.**

The flatness comparison is drawn on a loss landscape with two minima $\theta_1$ and $\theta_2$: **$\theta_1$ is flatter than $\theta_2$**, so to produce the same loss increase, $\theta_1$ requires a larger displacement — $\lVert\epsilon_1\rVert > \lVert\epsilon_2\rVert$ for $\Delta L_1 = \Delta L_2$.

*Intuition, and why flatness is a DG objective rather than a generic regularizer.* A domain shift moves the loss landscape: the target domain's loss surface is a displaced version of the source's, since it is computed on different data. A parameter sitting at the bottom of a narrow valley of the source loss can therefore land far up the wall once the surface moves, whereas a parameter in a broad basin remains near the bottom under the same displacement. Perturbing $\theta$ is a proxy for that displacement — it cannot be applied to the data, which is unavailable, so it is applied to the parameters instead, and a solution insensitive to one is empirically insensitive to the other.

This also connects SAM to Lecture 1 Section 5.2 more tightly than it first appears. Sharpness measures how rapidly the loss changes under parameter perturbation, which is the same sensitivity the capacity term $\beta\sqrt{S}$ measures through weight norms; SAM penalizes it directly and locally rather than through a global surrogate. It is, in the vocabulary of Lecture 1 Section 5.4, the "model compression and flatness" lever, arrived at from the DG side.

### 3.7 Approximating the Worst-Case Perturbation by Linearization

> [!warning] Computational obstacle **Directly solving (1) is computationally expensive: both $\theta$ and $\epsilon$ reside in high dimensions, so searching over all possible $\epsilon$-directions at a given $\theta$ is infeasible.**

**Solution: approximate $\bar{\epsilon}$ somehow.**

**Approach 1 — linearization around $\theta$.** In the inner maximization $\max_{\lVert\epsilon\rVert\le\rho} L(\theta+\epsilon)$, with $\theta$ **fixed** and $\epsilon$ the thing to **find**, a **first-order Taylor expansion** for small $\rho$ gives

$$L(\theta + \epsilon) \approx \underbrace{L(\theta)}_{\text{constant w.r.t. } \epsilon} + \underbrace{\nabla_\theta L(\theta)^{\top}}_{=\, g_\theta}\epsilon.$$

Dropping the constant, the inner problem reduces to a linear maximization over a ball,

$$\max_{\lVert\epsilon\rVert\le\rho} L(\theta+\epsilon) \;\approx\; \max_{\lVert\epsilon\rVert\le\rho} g_\theta^{\top}\epsilon,$$

and **$g_\theta^{\top}\epsilon$ is maximal whenever $\epsilon$ is co-linear with $g_\theta$ and has norm $\rho$** — the Cauchy–Schwarz equality condition. Hence

$$\boxed{\bar{\epsilon} = \rho\,\frac{g_\theta}{\lVert g_\theta\rVert_2} = \rho\,\frac{\nabla_\theta L_\theta}{\lVert\nabla_\theta L_\theta\rVert_2}}$$

In practice this means SAM costs two forward-backward passes per step: one to get $g_\theta$ and hence $\bar\epsilon$, and a second at $\theta + \bar\epsilon$ to get the gradient actually applied.

### 3.8 Including Curvature Information

**Approach 2 — include curvature information as well.** The linearization above is blind to how the surface bends; **curvature information is contained in the Hessian matrix**

$$H_\theta = \nabla_\theta^{2} L(\theta),$$

whose **maximum eigenvalue $\lambda_{\max}(H_\theta)$** is the **maximum curvature (a scalar) of the loss surface along any direction around $\theta$.**

To see why $\lambda_{\max}$ is the right target, evaluate the sharpness **at a minimum**, where $\nabla_\theta L = 0$. Substituting $\epsilon \to \rho\epsilon$ with $\lVert\epsilon\rVert \le 1$ and expanding to second order,

$$\max_{\lVert\epsilon\rVert\le\rho} L(\theta+\epsilon) - L(\theta) \;=\; \max_{\lVert\epsilon\rVert\le 1} L(\theta + \rho\epsilon) - L(\theta) \;\approx\; \max_{\lVert\epsilon\rVert\le 1}\left[\rho\,\epsilon^{\top}\nabla_\theta L + \frac{\rho^{2}}{2}\epsilon^{\top}H_\theta\,\epsilon\right]$$

and since the first-order term vanishes at a minimum, the maximization is a Rayleigh quotient whose optimum is the top eigenvalue:

$$\boxed{\text{Maximum value of sharpness (not argmax)} = \frac{\rho^{2}}{2}\,\lambda_{\max}(H_\theta)}$$

with $H_\theta$ **the Hessian evaluated at $\theta$**. Therefore **reducing $\lambda_{\max}$ will shrink the sharpness around $\theta$.**

**The direction of maximum sharpness is given by the eigenvector $V_1$ corresponding to $\lambda_{\max}$.** Adding this direction to the linearized perturbation gives the curvature-aware perturbation:

$$\boxed{\bar{\epsilon} = \rho\,\frac{g_\theta}{\lVert g_\theta\rVert} \;+\; \alpha\,\text{sign}\big(\langle g_\theta, V_1\rangle\big)\,V_1^{\perp}}$$

Two details justify the specific form. Decomposing $V_1 = V_1^{\parallel} + V_1^{\perp}$ into components parallel and perpendicular to $g_\theta$, **only $V_1^{\perp}$ gives new information** — the parallel part is already captured by the first term, so including it would merely rescale what we have. And **$\text{sign}(\langle g_\theta, V_1\rangle)$ makes sure the approximation of $V_1$ — usually obtained through power iteration — always gives a consistent sign for $V_1$**, since an eigenvector is only defined up to sign and power iteration may return either.

### 3.9 Challenges of SAM for DG

> [!warning] Flatness is not automatically shared **No shared flatness across domains**, due to **domain shifts**, which produce **inconsistent loss landscapes** — and therefore **gradients across domains are inconsistent.**

The contrast is drawn as two pictures. On the left, three domains $\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3$ produce wildly different, oscillating loss curves over $\theta$ whose minima do not coincide: **inconsistent loss landscapes**. On the right, three nearly-coincident smooth bowls: **consistent loss landscapes** — the situation we want. In the inconsistent case the per-domain gradients $g_\theta^{1}, g_\theta^{2}, g_\theta^{3}$ point in conflicting directions, so a flat basin found for one domain need not be flat for another.

Three remedies follow, numbered ① to ③ in the notes.

### 3.10 Stochastic Weight Averaging and SWAD

**① Stochastic Weight Averaging (SWA) and SWA Densely (SWAD).**

SWA simply averages the parameter iterates:

$$\boxed{\bar{\theta} = \frac{1}{K}\sum_{i=1}^{K}\theta_i}$$

The averaged point **is the point closest to all visited $\theta_i$**, so the reasoning goes: **if you are in a low-loss region, you will settle at its center.** Averaging points scattered around a basin lands you in the middle of the basin — which is exactly the flat interior SAM was trying to reach, obtained for free without any min-max solve.

The method depends on a condition, drawn as the iterates $\theta_1, \theta_2, \ldots, \theta_K$ all lying within the **same basin**: averaging is only meaningful at the **tail end of training**, i.e. **after the model has mostly fit the data and is not showing jumps between different minima.** Averaging across two different basins would land in the ridge between them, which is worse than either.

**SWAD** extends this as

$$\boxed{\text{SWAD} = \text{SWA} + \text{dense snapshots}}$$

taking many more, more closely spaced snapshots so the average is a better estimate of the basin center.

### 3.11 Gradient Alignment or Masking Across Domains

**② Gradient alignment or masking across domains:** **align gradients across domains, or mask inconsistent directions, to reduce gradient conflicts.**

This addresses the diagnosis of Section 3.9 head-on. Rather than hoping a flat region is shared, it either rotates the per-domain gradients into agreement, or discards the coordinates on which they disagree — so that only the directions all domains endorse contribute to the update.

### 3.12 Refining Each Domain's Loss Landscape with Soft Labels

**③ Refine each domain's individual loss landscapes.**

> **One part of the problem is the one-hot encoding of labels: it overfits and sharpens the landscapes across $\mathcal{D}_i$, $i = 1, \ldots, M$.**

A one-hot target $y = [1, 0, 0, 0]^{\top}$ demands infinite confidence, which drives the network into a narrow, high-curvature minimum; the picture in the notes shows a jagged multi-minimum landscape being transformed into a set of smooth, nearly-coincident bowls.

**Instead of forcing one-hot labels, learn labels: soft labels $\tilde{y}_K \iff y_K$, through a separate head $\tilde{h}_\psi(\cdot)$, but keep $\tilde{y}_K$ close to $y_K$.** The architecture adds a second head alongside the task head:

$$x \to f_\theta(\cdot) \xrightarrow{\;z\;} \begin{cases} \boxed{\text{Task head}} \to \hat{y} \iff Y \text{ (one-hot encoding)}\\[1ex] \boxed{\tilde{h}_\psi} \to \tilde{y} \quad \text{(learn } \psi \text{ separately)}\end{cases}$$

**For domains $\mathcal{D}_i$ and $\mathcal{D}_j$**, using the supervised loss on soft labels $L_{\text{sup}}(x, \tilde{y})$, the SAM perturbation for domain $i$ is

$$\epsilon_i = \rho\,\frac{\nabla_\theta L_{\text{sup}}}{\lVert\nabla_\theta L_{\text{sup}}\rVert}$$

(compare the one-hot version, $\bar\epsilon = \rho\,\nabla_\theta L(x_i, y_i)/\lVert\nabla_\theta L(x_i,y_i)\rVert$, which is what the soft labels are replacing), and the **sharpness of $\mathcal{D}_i$** is the loss increase under that perturbation:

$$\boxed{S_i = L_{\text{sup}}(\theta + \epsilon_i) - L_{\text{sup}}(\theta)}$$

with **$\theta$ trained while $\psi$ is held fixed**. The **sharpness gap**

$$\boxed{\text{Sharpness Gap} = |S_i - S_j|}$$

**measures cross-domain inconsistency** — exactly the quantity Section 3.9 identified as the obstacle.

The refinement loss, optimized **for fixed $\theta$, updating $\psi$**, is

$$\boxed{L_{\text{Refine}}(\psi) = \text{KL}\big(\tilde{y} \,\|\, y\big) + \lambda_1\big|S_i - S_j\big|}$$

where the KL (Kullback–Leibler) term is what **keeps $\tilde{y}$ close to $y$** (preventing the soft labels from drifting into meaninglessness) and the second term explicitly equalizes sharpness across domains.

The mechanism, stated as a chain: **soft labels act as controllable handles for controlling the flatness of each domain's loss landscape, in order to force shared flatness.** Concretely, **$\tilde{y}$ controls $\nabla_\theta L_{\text{sup}}$, which controls $\epsilon_i$, so $S_i$ is shrunk.** The alternation is the key structural point — $\theta$ and $\psi$ are updated in separate phases, each holding the other fixed.

### 3.13 Group Distributionally Robust Optimization

> **Group Distributionally Robust Optimization (G-DRO):** if we can carve the training data (or their representations, equivalently) into groups $g \in \{1, 2, \ldots, m\}$ such that each $g_i$ plausibly captures spurious correlations (e.g. backgrounds), then solve

$$\boxed{\min_{\theta}\;\max_{g \in \{1,2,\ldots,m\}}\;\mathbb{E}_{(x,Y)\sim P_g}\Big[L(\theta; x, Y)\Big]} \tag{1}$$

**Minimize the loss on the worst group** — and it will then **robustify against every perturbation $\epsilon \in g$.** The logic is that if the worst-performing group is acceptable, every group is, so no spurious grouping can be silently exploited.

> [!warning] Two problems with G-DRO **A direct application of (1) fails in deep learning, as models are over-parameterized: if they fit $L_{\text{avg}} \to 0$ during training, they can easily make $L_{\text{worst}} \to 0$ too.** The worst-group loss becomes uninformative once the model can memorize everything. **Empirical fix: prevent zero training loss during training.** **Another problem: G-DRO needs group labels, and the DG setup does not have them.**

**The alternative is DRO over perturbation.**

### 3.14 DRO over Perturbation, and CCFP

**Define an uncertainty set $\mathcal{U}$ of plausible domain-shifts** (e.g. style or texture changes), **then solve a min-max problem:**

$$\boxed{\min_{\theta}\;\mathbb{E}_{P_i(x,Y)}\left[\max_{s \in \mathcal{U}} L\big(\theta;\, T_s(x),\, Y\big)\right]} \tag{2}$$

where $T_s(\cdot)$ applies shift $s$, and the inner maximization **synthesizes hard domain-shifts using an adversary**. This sidesteps the group-label requirement: instead of needing someone to annotate which spurious group each sample belongs to, the adversary manufactures the shifts.

**Example: CCFP (Cross-Contrastive Feature Perturbation)**, which **realizes (2) in feature space.**

> **Learn style perturbations that make the two streams' features maximally different while keeping semantics intact.**

The input branches into two streams: a **perturbed stream** and an **original stream**. Equivalently, in the block form drawn in the notes, the perturbed path runs through an **LDP** (learnable domain perturbation) module producing features $f_p^{(i)}$, while the original path produces $f_o^{(i)}$.

Three losses balance the adversary against the semantics. The **discrepancy loss** is maximized by the perturbation parameters:

$$\boxed{L_{\text{discrepancy}} = -\sum_{i=1}^{K}\left\lVert G\big(f_o^{(i)}(x)\big) - G\big(f_p^{(i)}(x)\big)\right\rVert_F}$$

where **$G(\cdot)$ is some measure of discrepancy** and the update to the perturbation parameters $\gamma$ is made **to increase $L_{\text{discrepancy}}$** — note the leading minus sign, which is what makes minimizing this term equivalent to maximizing the feature gap. The **semantic loss** constrains the perturbation from destroying content:

$$\boxed{L_{\text{semantic}} = \big\lVert g_o(x) - g_p(x)\big\rVert_2^{2}}$$

and the total combines both streams' classification losses with these two terms:

$$\boxed{L_{\text{Total}} = L_{\text{semantic}} + L_{\text{CE}}^{o} + L_{\text{CE}}^{p} + L_{\text{discrepancy}}}$$

The parameters of $g_o(\cdot)$ and $g_p(\cdot)$ are **updated on $L_{\text{Total}}$ to reduce it**, while the perturbation adversary moves in the opposite direction on the discrepancy term — the standard adversarial two-player structure, here played out in feature space rather than pixel space.

---

## 4. Family 3 — Feature Disentanglement and Disentangled Representation Learning

Families 1 and 2 both accept that causal and spurious features are mixed together in $z$ and work around it — alignment by removing whatever differs across domains, learning strategies by penalizing reliance on anything unstable. The third family attempts something more direct: rather than suppressing the domain-specific content, separate it into its own coordinates, where it can simply be ignored. If this succeeds it subsumes the other two, since a head restricted to the causal coordinates is invariant by construction rather than by pressure. Whether it succeeds is the question Sections 4.12–4.15 take up, and the honest answer there is qualified.

### 4.1 Motivation — Humans versus Deep Models

Shown an object, a **human** — drawing on **prior knowledge** — immediately decomposes it into **various properties of the object**: **shape, size, colour**, and so on. These are the **hidden attributes** of the object, and crucially the human represents them *separately*.

![[03 - Attachments/ATML_L5_2_drl_motivation.png|620]]

*Human (prior knowledge → shape, size, colour) vs. a deep model (fits the data, takes shortcuts) vs. DRL (one feature slot per factor of variation).*

A **deep learning model**, by contrast, **learns the object's representation by fitting to the data distribution and discrimination criteria** — and in doing so, **takes shortcuts**. This is the same shortcut pathology diagnosed in Lecture 5.1: nothing in the objective requires the model to separate shape from colour, so it does not.

**Disentangled Representation Learning (DRL)**, also called **Feature Disentanglement (FD)**, aims to close that gap. An input passes through a DRL model into a **feature space** in which the **hidden factors are disentangled** — each slot of the representation corresponding to one **factor of variation carrying semantic meaning**: **object size, shape, colour, viewing angle**, and so forth. The stated aspiration is **human-like generalization ability.**

### 4.2 Three Benefits of DRL

**Explainability.** DRL **learns semantically meaningful and separate representations which are aligned with latent generative factors.** Because each dimension means something nameable, the representation can be inspected rather than merely used.

**Generalizability.** An input with **entangled factors** goes into DRL and comes out as **disentangled representations of those factors** — for example, a **causal** slot and a **non-causal** slot. **DRL has better generalizability**, and the reason connects directly to Lecture 5.1's feature taxonomy: once causal and non-causal factors occupy separate coordinates, a downstream head can be restricted to the causal ones, and spurious correlations lose their channel.

**Controllability.** **Learned disentangled representations can be manipulated to achieve better controllable generation of the modality.** If one coordinate is azimuth, generation with a specified azimuth is a matter of setting that coordinate.

### 4.3 Definition — Hidden Generative Factors

The formal picture: **observed data** contains a set of shapes (circles, squares, triangles, stars, diamonds in the professor's sketch), and passing it through DRL produces **latent factors (LF)**, each of which is tied to one underlying shape.

The data is generated by **hidden generative factors (GF)**, which are required to be:

1. **Distinct**
2. **Independent**
3. **Informative**

and the goal, highlighted in the notes, is that

$$\boxed{\textbf{GF are separated into LF in the latent domain or feature domain}}$$

subject to the condition that each latent slot **is sensitive only to changes in its own generative factor** — the first slot responds only to changes in the circle factor, the second only to the square factor, and so on down the list. That one-factor-one-slot sensitivity *is* the definition of disentanglement.

### 4.4 DRL Structure: Dimension-Wise

> **Dimension-wise structure: for multiple fine-grained factors in simple scenarios.**

Each generative factor (shape, size, colour, …) maps to **a single scalar dimension** of the latent:

$$z = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_{K_1}\end{bmatrix} \in \mathbb{R}^{K_1 \times 1}, \qquad z_i = \text{scalar} = 1\times 1,$$

where the **number of generative factors is $K_1$** and these are **fine-grained factors**. Reading the roles: **$z$ is the whole representation**, and **$z_i$ is a single dimension representing one fine-grained latent factor corresponding to a single GF.** Consequently **every $z_i$ has a different semantic meaning.**

**Dimension-wise structure is used on simple datasets, e.g. synthetic datasets**, and the **model architectures** used are **VAE (variational autoencoder) and GAN (generative adversarial network).** The restriction to simple data is not arbitrary — asking a single scalar to carry a real-world factor like "object identity" is asking too much.

### 4.5 DRL Structure: Vector-Wise

> **Vector-wise structure: for several coarse-grained factors in complex scenarios.**

![[03 - Attachments/ATML_L5_2_vector_wise_factors.png|600]]

*Vector-wise structure: size and shape grouped as *content*, colour as *style*, each group encoded by its own vector $\underline{z}_i$.*

Here the factors are grouped into higher-level bundles — the sketch shows **object size** and **object shape** grouped under **content**, and **object colour** grouped under **style** — and each bundle gets a whole **vector** rather than a scalar:

$$Z = \big[\,\underline{z}_1 \;\; \underline{z}_2 \;\; \cdots \;\; \underline{z}_{K_2}\,\big] \in \mathbb{R}^{d \times K_2}, \qquad d \ge 2,$$

giving **$K_2$ disentangled latent vectors**, each with **dimension $\ge 2$**, where **every $\underline{z}_i$ has a different semantic meaning.** The **number of GF is $K_2$**, these being **coarse-grained factors**, and **one single vector $\underline{z}_i$ is used to represent one coarse-grained GF.** As before, $Z$ is the **whole representation**.

**Vector-wise structure is used on real-world, more challenging datasets**, and is **more flexible** in a specific sense: **task-specific encoders can be designed to extract each $\underline{z}_i$.** The **model architectures** named are **DR-GAN, DRNet, DRANet**, and similar, all of which **deploy different encoders for each $\underline{z}_i$.**

| | **Dimension-wise**\vert **Vector-wise** |
|---| ---\vert --- |
| Factor granularity | Fine-grained\vert Coarse-grained |
| Latent unit per factor | Scalar $z_i$ ($1\times1$)\vert Vector $\underline{z}_i$ ($d \ge 2$) |
| Whole representation | $z \in \mathbb{R}^{K_1\times1}$\vert $Z \in \mathbb{R}^{d\times K_2}$ |
| Scenario | Simple / synthetic datasets\vert Real-world, challenging datasets |
| Architectures | VAE, GAN\vert DR-GAN, DRNet, DRANet |
| Encoders | One shared encoder\vert Task-specific encoder per $\underline{z}_i$ |

### 4.6 Design of the Loss Function

**In both cases, an appropriate loss function must be designed to guarantee disentanglement without losing information.** Overall:

$$\boxed{L = \lambda\, L_{\text{recons.}} + \lambda_2\, L_{\text{Disent.}} + \lambda_3\, L_{\text{Task}}}$$

Each term earns its place. **$L_{\text{recons.}}$ ensures the disentangled representation is semantically meaningful** — concretely, that we **can recover the original data** from it — and it **can also facilitate disentanglement** as a side effect. **$L_{\text{Disent.}}$ enforces disentanglement of the representation** directly. **$L_{\text{Task}}$** is the downstream objective, which may be **discriminative** or **generative**.

The reconstruction term is the guard against a degenerate solution: a representation can be perfectly disentangled and perfectly useless (e.g. all zeros), and only the requirement that $x$ be recoverable rules that out.

### 4.7 Formal Setup — Independent and Dependent Factors

The dataset is a triple

$$\mathcal{D} = \{x,\, v,\, w\}$$

with **$x$ = input**, and the **ground-truth data generative factors** split into **$v$ = conditionally independent factors** and **$w$ = conditionally dependent factors**. **$x$ is generated from $v$ and $w$ using a "true-world simulator",** modelled as $P(x \mid v, w)$ with $v$ and $w$ as parents and $x \in \mathbb{R}^{d}$ as the child.

The examples make the distinction concrete:

- **$v$ (independent):** circles, colour, size, pose; skin tone, hair colour, face shape, age; lighting condition, texture.
- **$w$ (dependent):** shadows, occlusion, reflection — all of which **depend on some factors in $v$**; wrinkles, glasses, hair highlights; surface illumination.

The dependency is the point: a shadow is not a free parameter, it is determined by the lighting and pose already present in $v$. So $w$ is redundant given $v$, which is why the goal below is to ignore it.

**Suppose there are $K$ independent GF, $v = \{v_1, v_2, \ldots, v_K\}$. Then the model for DRL should treat them as independent GF**, which means the posterior over factors must factorize:

$$P(v \mid x) = \prod_{i=1}^{K} P(v_i \mid x) \qquad\Longrightarrow\qquad \log P(v\mid x) = \sum_{i=1}^{K}\log P(v_i \mid x)$$

**Example:** $P(\text{object shape} \mid x)$ is **independent of its colour or size**.

**Goal:** learn a model (a probabilistic generative model) $P_\theta(x\mid z)$ that can

1. **Reconstruct $x$**, and
2. **Learn a latent $z \in \mathbb{R}^{M}$ with $M \ge K$ that encodes the GFs $v_i$ in a disentangled manner, while also separating them from the $w$ factors.**

Note $M \ge K$: the latent is allowed more slots than there are true factors, which gives the model room to fail gracefully rather than being forced to merge two factors into one coordinate.

### 4.8 The VAE as the Vehicle for DRL

**How to do it? Use a VAE.**

The encoder $E$ **learns a Gaussian posterior**

$$q_\phi(z \mid x) = \mathcal{N}\big(\mu_\phi,\, \text{diag}(\sigma_\phi^{2})\big),$$

so **each dimension $z_i \sim \mathcal{N}(\mu_\phi^{i},\, \sigma_{\phi,i}^{2})$.** Concretely, **$E$ outputs $\mu_\phi \in \mathbb{R}^{M\times1}$ and $\sigma_\phi \in \mathbb{R}^{M\times1}$**, and $z_i$ is **generated** by the reparameterization trick from Lecture 3:

$$\boxed{z_i = \mu_\phi^{i} + \sigma_{\phi,i}\odot\epsilon_i, \qquad \epsilon_i \sim \mathcal{N}(0,1)}$$

from which **we can show the pdf of $z_i$ is $\mathcal{N}(\mu_\phi^{i},\, \sigma_{\phi,i}^{2})$** — an affine map of a standard normal is normal with the shifted mean and scaled variance.

The diagonal covariance is not an incidental modelling convenience here; it is the entire mechanism. A diagonal posterior is what makes the latent coordinates candidates for independent factors in the first place.

### 4.9 The Isotropic Gaussian Prior and the VAE Objective

**In order to promote disentanglement in the dimensions of $z$, we now bring in an $M$-variate isotropic Gaussian density as the prior on $z$** — **isotropic** meaning **the same in all directions**:

$$\boxed{P(z) = \text{prior on } z = \mathcal{N}(0,\, I_{M\times M})}$$

**All dimensions of $P(z)$ are independent of each other**, therefore **$\sigma^{2} = I$ and all have the same variance, namely 1.** This is the same identity-covariance argument made in Lecture 2: zero off-diagonal covariance among jointly Gaussian variables *is* independence.

**We want $z$ to be as close to $P(z)$ as possible**, which means we **reduce the KL divergence between $q_\phi(z\mid x)$ and $P(z)$:**

$$\boxed{D_{\text{KL}}\big(q_\phi(z\mid x)\,\big\|\,P(z)\big)} \quad \longrightarrow \quad \text{decrease it.}$$

**In order to make sure $z$ is representative of $x$, put a decoder** after it: $x \to E \to z \to D \to \hat{x}$, with $P(z) \sim \mathcal{N}(0,I)$ imposed on the bottleneck. **$D$ learns the likelihood $P_\theta(x\mid z)$ by assuming $P_\theta(x\mid z) \sim \mathcal{N}(\mu_\theta,\, \sigma^{2}I)$ with fixed variance, such that $\hat{x} = \mu_\theta$, by minimizing the reconstruction loss.** Under that fixed-variance Gaussian assumption the log-likelihood reduces to a squared error,

$$\log P_\theta(x\mid z) \;\propto\; \underbrace{\big\lVert x - \mu_\theta(z)\big\rVert_2^{2}}_{\text{MSE between } x \text{ and } \hat{x}},$$

which is why VAE reconstruction is implemented as MSE (mean squared error) in practice.

**If the variance is NOT fixed**, the log-likelihood keeps its full per-dimension form:

$$\log P_\theta(x\mid z) \;\propto\; \frac{1}{2}\sum_{j=1}^{d}\left[\frac{\big(x_j - \mu_\theta^{j}\big)^{2}}{\sigma_{\theta,j}^{2}} + \log\sigma_{\theta,j}^{2}\right]$$

subject to $P_\theta(x\mid z) \sim \mathcal{N}\big(\mu_\theta,\, \text{diag}(\sigma_{\theta,j}^{2})\big)$, equivalently $P_\theta(x\mid z) = \prod_{j=1}^{d}\mathcal{N}\big(\mu_\theta^{j},\, \sigma_{\theta,j}^{2}\big)$. The extra $\log\sigma^2$ term is what prevents the model from driving the variance to zero to inflate the likelihood.

**Overall the VAE learns as:**

$$\boxed{L_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi}\big[\log P_\theta(x\mid z)\big]}_{\text{reconstruction term}} \;-\; \underbrace{D_{\text{KL}}\big(q_\phi(z\mid x)\,\|\,P(z)\big)}_{\text{alignment term}}}$$

### 4.10 β-VAE

**A simple modification makes it β-VAE:** weight the alignment term.

$$\boxed{L_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}\big[\log P_\theta(x\mid z)\big] \;-\; \beta\, D_{\text{KL}}\big(q_\phi(z\mid x)\,\|\,P(z)\big)}$$

Reading the parameter:

- **$\beta = 1 \Rightarrow$ β-VAE = VAE.**
- **$\beta > 1 \Rightarrow$ β-VAE gives more weight to $D_{\text{KL}}$, reducing it more and more. When $\beta > 1$, we are forcing latent dimensions to resemble more and more the $P(z)$ dimensions, i.e. become independent.**

**When all $z_i$ become independent, they are disentangled, and each $z_i$ corresponds to some underlying $v_i$ GF in $x$. At this point, $E$ forgets about all redundant or correlated factors, i.e. $w$ is ignored.** This is the formal payoff of Section 4.7's setup: the dependent factors $w$ are by construction correlated with the independent ones, so a pressure toward independence is automatically a pressure to discard $w$.

> [!warning] The β-VAE trade-off **However, for $\beta > 1$, though we get a disentangled $z$ by putting extra pressure on the posterior $q_\phi(z\mid x)$ to be similar to $P(z)$, this comes at the cost of the first term — we lose reconstruction fidelity in $\mathbb{E}_{q_\phi}[\log P_\theta(x\mid z)]$.** $$\text{Strong bottleneck pressure} \;\longleftrightarrow\; \text{Reconstruction fidelity} \qquad \textbf{(trade-off)}$$

### 4.11 Training β-VAE

The training procedure, step by step with all dimensions tracked.

**①** **$\beta$ is a hyperparameter.**

**② Input.** A **batch of $B$ images** $\{x^{(1)}, x^{(2)}, \ldots, x^{(b)}, \ldots, x^{(B)}\}$ with each **$x^{(b)} \in \mathbb{R}^{d\times1}$**, so one batch is $X \in \mathbb{R}^{B\times d}$ — the **original batch**.

**③ Encoder.** For each image,

$$q_\phi\big(z \mid x^{(b)}\big) \sim \mathcal{N}\Big(\underbrace{\mu_\phi\big(x^{(b)}\big)}_{M\times1},\; \underbrace{\text{diag}\big(\sigma_\phi^{2}(x^{(b)})\big)}_{M\times M \text{ diagonal, } M \text{ non-zero values}}\Big)$$

so across the batch, $X$ produces $\mu_\phi(X) \in \mathbb{R}^{B\times M}$ and $\sigma_\phi^{2}(X) \in \mathbb{R}^{B\times M}$.

**④ Latent sampling.** Apply the reparameterization trick per image,

$$z^{(b)} = \mu_\phi\big(x^{(b)}\big) + \sigma_\phi\big(x^{(b)}\big)\odot\epsilon^{(b)} \in \mathbb{R}^{M\times1} \qquad\Longrightarrow\qquad Z = \big\{z^{(1)}, \ldots, z^{(B)}\big\},$$

giving **$B$ latent vectors**.

**⑤ Decoder.** $P_\theta\big(x \mid z^{(b)}\big) \sim \mathcal{N}\big(\mu_\theta(z^{(b)}),\, \sigma^{2}I\big)$ with $\mu_\theta(z^{(b)}) \in \mathbb{R}^{d\times1}$, $\sigma^{2}I \in \mathbb{R}^{d\times d}$ and the variance **fixed**. Across the batch, $\mu_\theta(Z) \in \mathbb{R}^{B\times d}$ — **the reconstructed batch.**

**⑥ Batch loss.**

$$\boxed{L_{\text{Batch}} = \frac{1}{B}\sum_{b=1}^{B}\left[-\log P_\theta\big(x^{(b)}\mid z^{(b)}\big) + \beta\, D_{\text{KL}}\Big(q_\phi\big(z\mid x^{(b)}\big)\,\Big\|\,P(z)\Big)\right]}$$

**For each image $x^{(b)}$**, the two terms evaluate in closed form. The reconstruction term, under the fixed-variance Gaussian decoder, is a squared error:

$$-\log P_\theta\big(x^{(b)}\mid z^{(b)}\big) = \Big\lVert x^{(b)} - \mu_\theta\big(z^{(b)}\big)\Big\rVert_2^{2}$$

and the KL term against a standard normal prior has the standard diagonal-Gaussian closed form:

$$D_{\text{KL}}\Big(q_\phi\big(z\mid x^{(b)}\big)\,\Big\|\,P(z)\Big) = \frac{1}{2}\sum_{j=1}^{M}\left[1 + \log\sigma_{\phi,j}^{2} - \mu_{\phi,j}^{2} - \sigma_{\phi,j}^{2}\right]$$

> [!warning] Possible source inconsistency The KL expression as written is the **negative** of the standard closed form, which is $\frac{1}{2}\sum_j\big[\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big]$ and is non-negative for all $\mu, \sigma$. The version in the notes is reproduced above as given; note that it evaluates to $0$ at $\mu = 0, \sigma = 1$ and is negative elsewhere, so the sign must be flipped for the $L_{\text{Batch}}$ expression to behave as a penalty. Use the standard form in implementation.

### 4.12 Verifying Disentanglement — Visual Inspection

**① Visual inspection.** The procedure:

1. **Take $x$ from the dataset.**
2. **Get $z = q_\phi(z\mid x)$.**
3. **Vary one dimension, say $z_K$, while keeping the others constant.**
4. **Decode $\hat{x} = P_\theta(x\mid z)$ and inspect.**

The output is a strip of reconstructions $\hat{x}$ as $z_K$ sweeps across its range. **If $z_K$ is hypothesized to be azimuth-rotation, all these $\hat{x}$ should differ only in rotation** — and nothing else. Any co-varying attribute (the object also changing colour as it rotates) is direct visual evidence of entanglement.

### 4.13 Verifying Disentanglement — the Z-Diff (β-VAE) Metric

**② Quantitative — supervised.** The first quantitative measure is the **Z-Diff or β-VAE metric**, which requires access to the ground-truth factors (hence "supervised").

At inference time, with **$K$ total generative factors** $v_1, \ldots, v_K$:

**Choose $v_i$ randomly.** Then **simulate two images with the same $v_i$**, while **the other $v_j$, $j \neq i$, vary in both images**:

$$x^{(1)} \sim \text{Sim}\big(v_i = \alpha\big) \;\to\; z^{(1)}, \qquad x^{(2)} \sim \text{Sim}\big(v_i = \alpha\big) \;\to\; z^{(2)}$$

and take the absolute difference of their latent codes:

$$z_{\text{diff}}^{(1)} = \big|z^{(1)} - z^{(2)}\big|$$

The logic: **the dimension of $z$ associated with $v_i$ should have the same value in both** — since $v_i$ was held fixed — so that coordinate of $z_{\text{diff}}$ should be near zero while all others are large. Which coordinate is near zero therefore *identifies* $v_i$.

**Do this $L$ times with fixed $v_i$ in every $\ell = 1, 2, \ldots, L$** and **average**:

$$\boxed{z_{\text{diff}} = \frac{1}{L}\sum_{\ell=1}^{L} z_{\text{diff}}^{(\ell)}}$$

Averaging suppresses the noise from the randomly varying $v_j$, sharpening the signature. Finally, **predict $v_i$ using $z_{\text{diff}}$ using a simple linear classifier** with outputs $v_1, v_2, \ldots, v_K$:

$$\boxed{\text{Metric} = \text{accuracy of this linear classifier}}$$

A perfectly disentangled representation makes this classification trivial (one coordinate is zero, and which one tells you the answer), so accuracy near 1 indicates disentanglement. The use of a *linear* classifier is deliberate — a nonlinear one could recover the factor even from an entangled code, defeating the purpose of the measurement.

### 4.14 Verifying Disentanglement — Mutual Information Gap

**③ Mutual Information Gap (MIG).** Given the $K$ generative factors $v_1, \ldots, v_K$ and the $M$ latent factors $z_1, \ldots, z_M$, **compute $I(v_i; z_j)$ for all $i, j$.**

**Let $z_{*}$ be the dimension of $z$ with maximum mutual information with $v_i$, and let $z_0$ be the dimension with the second-highest mutual information with $v_i$.** Then the per-factor gap, normalized by the factor's own entropy, is

$$\boxed{\text{MIG}_i = \frac{I(v_i, z_{*}) - I(v_i, z_0)}{H(v_i)}}$$

where **$H(v_i)$ is the entropy of factor $v_i$**. Averaging over factors,

$$\boxed{\text{MIG} = \frac{1}{K}\sum_{i=1}^{K}\frac{1}{H(v_i)}\Big[I(v_i, z_{*}) - I(v_i, z_0)\Big]}$$

The design is clever: a *gap* rather than a raw mutual information is used because a factor could be strongly encoded in several coordinates at once, which is informative but entangled. Requiring the top coordinate to dominate the runner-up is exactly the "one factor, one slot" criterion. The normalization by $H(v_i)$ puts factors of differing intrinsic uncertainty on a common scale, bounding $\text{MIG} \in [0,1]$.

Plotting **MIG and accuracy against $\beta$** gives curves that rise with $\beta$ — larger $\beta$ does buy disentanglement, which sets up the next section's counterpoint.

### 4.15 Cons of β-VAE

> [!warning] The information cost of disentanglement **Although a larger $\beta$ promotes disentanglement in $z$, it decreases the ability of $z$ to preserve the information from the input $x$.**

The picture: $x$ is squeezed through a narrow bottleneck $z$ and reconstructed as $\hat{x}$, and **large $\beta \Rightarrow z_i \perp z_j$ but $\hat{x}$ is different from $x$, i.e. reconstruction is bad. It is due to $z$ capturing very little info in $x$.**

The diagnostic quantity is the KL term itself: **$D_{\text{KL}}(q_\phi \| P(z))$ reflects the amount of info encoded in $z$ from $x$.** Specifically, **higher KL $\Rightarrow$ $z$ has more info of $x$**, and **lower KL $\Rightarrow$ $z$ has less info of $x$.** This makes the trade-off unavoidable rather than incidental: the KL term *is* the information channel capacity, so penalizing it harder necessarily narrows the channel.

The refined diagnosis: **in β-VAE, increasing $\beta$ makes only a few axes of $z$ truly independent and information-carrying; the rest are just independent with no info of $x$ in them.** A collapsed coordinate — one that has reverted to exactly the prior $\mathcal{N}(0,1)$ regardless of input — is trivially independent of everything and carries nothing. It is posterior collapse (Lecture 3, §5.6) occurring dimension by dimension.

### 4.16 Capacity Control

> **Fixed $\beta$ is the problem. Solution: capacity control.**

Rather than penalizing the KL toward zero, penalize its *deviation from a target capacity* $C$:

$$\boxed{L_{\text{C-VAE}} = \underbrace{\mathbb{E}_{q_\phi}\big[P_\theta(x\mid z)\big]}_{A} - \beta\Big|\underbrace{D_{\text{KL}}\big(q_\phi(z\mid x)\,\|\,P(z)\big) - C}_{B}\Big|}$$

with **$C$ explicitly controlling capacity** and being **varying** over training rather than fixed. The absolute value is what changes the behaviour: the model is now penalized for using *less* than $C$ nats as well as more, so a coordinate cannot simply collapse.

Reading the two regimes:

**(a) When $C$ is small:** the term $B$ is **kept small**, $\beta$ is effectively **large**, so the **model is enforcing a strict bottleneck** and **encodes only the most important factors of $x$ in $z$.**

**(b) Increase $C$:** the effective pressure **becomes smaller and smaller**, which **allows the model to gradually encode more info of $x$ into $z$.**

The schedule — start with $C$ small and anneal it upward — is what makes this work: the model first commits its limited capacity to the single most salient factor, then the next, and so on, which naturally produces an ordered, one-factor-at-a-time allocation of coordinates.

### 4.17 Other Generative Models for DRL

**Apart from VAEs, other generative models have also been explored extensively for DRL:**

- **GAN** — **InfoGAN (a GAN that maximizes mutual information between part of its latent code and the output) is the pioneering work.**
- **Diffusion models** — these **outperform VAEs and GANs for DRL**, e.g. the **DisDiff** model.
- **Pre-trained VLMs (vision-language models).**
- And others.

> [!important] The impossibility result **However, almost all of these works are in the unsupervised DRL category, and it has been shown that any unsupervised model can only achieve disentangled representation if it has a strong inductive bias.**

This is the Locatello et al. impossibility result, and it connects straight back to Lecture 4: without an inductive bias, the factorization of a latent space is not identifiable from the data alone, because infinitely many equally-good factorizations exist. The remedy is either a strong architectural prior or, as the next section shows, a few labels.

### 4.18 Incorporating Label Information — Semi-Supervised β-VAE

**We can incorporate label information during training to improve DRL performance.** Suppose **very few labels are available against the GF of $x$**, pairing the ground-truth factor vector with the latent:

$$\underline{v} = \begin{bmatrix} v_1 \\ \vdots \\ v_K\end{bmatrix} \quad\longleftrightarrow\quad \underline{z} = \begin{bmatrix} z_1 \\ \vdots \\ z_K\end{bmatrix} \qquad \text{— very few such labelled pairs.}$$

![[03 - Attachments/ATML_L5_2_semisup_beta_vae.png|560]]

*Semi-supervised β-VAE: the logistic of $\mu_\phi$ is matched to the few labelled latents via $L_{\text{sup}}$, while the reparameterized $z$ feeds the decoder as usual.*

**Then we can modify β-VAE as follows.** The encoder $E$ produces $\mu_\phi$ and $\sigma_\phi$ as usual. A **logistic function $G(\cdot)$** is applied to the mean, giving $G(\mu_\phi(x))$, which is compared against the **labelled latent $\underline{z}$** through a supervised loss $L_{\text{sup}}$. In parallel, $\mu_\phi$ and $\sigma_\phi$ feed the **reparameterization trick** to produce $z$, and a switch selects between the sampled $z$ and the labelled $\underline{z}$ as the decoder $D$'s input, producing $\hat{x}$.

The loss gains a third term:

$$\boxed{L_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}\big(P_\theta(x\mid z)\big) - \beta\, D_{\text{KL}}\big(q_\phi(z\mid x)\,\|\,P(z)\big) + \beta_{\otimes}\, L_{\text{sup}}}$$

Using the mean $\mu_\phi$ rather than a sample for the supervised branch is deliberate — it gives a deterministic, low-variance signal for the handful of labelled pairs, where a sampled $z$ would add noise the supervision cannot afford.

> [!note] Notation The coefficient on the supervised term is written with a subscript that is not clearly legible in the scan; it is reproduced as $\beta_{\otimes}$ and should be read simply as a second, independent weighting hyperparameter distinct from $\beta$.

### 4.19 Vector-Wise DRL Architecture

For the vector-wise structure of Section 4.5, the concrete architecture uses **two separate encoders** on the same input and an independence constraint between their outputs:

$$x \;\longrightarrow\; \begin{cases} \boxed{E_1} \to z_1 \\[1ex] \boxed{E_2} \to z_2\end{cases} \;\longrightarrow\; \boxed{z_1 \perp z_2} \;\longrightarrow\; \boxed{\text{Reconstruction } D} \;\longrightarrow\; x$$

![[03 - Attachments/ATML_L5_2_vector_wise_architecture.png|600]]

*Two encoders $E_1, E_2$ produce $z_1 \perp z_2$, and the decoder $D$ must reconstruct $x$ from both.*

with the three defining relations:

$$z_1 = E_1(x), \qquad z_2 = E_2(x)$$
$$z_1 \perp z_2 \;\Longrightarrow\; \textbf{both are independent}$$
$$x = D(z_1, z_2) \;\Longrightarrow\; \textbf{$z_1$ and $z_2$ are meaningful representations of the input } x$$

The third relation is doing the same guard duty as the reconstruction term in Section 4.6 — independence alone is cheap, and only reconstructability forces the two codes to carry real content.

The loss has three weighted terms:

$$\boxed{L_{\text{oss}} = \lambda_0\, L_{\text{orth.}} + \lambda_1\, L_{\text{recons.}} + \lambda_2\, L_{\text{Task}}}$$

where **$L_{\text{orth.}}$** is, for example, the **cosine similarity distance between $z_1$ and $z_2$** — a computationally trivial surrogate for statistical independence, penalizing alignment between the two subspaces.

---

## 5. Self-Supervised Learning

All three families assume a representation is already being learned and ask what pressure to place on it. That assumption has a cost the preceding sections left implicit: every one of them requires *labelled* data from several domains, and multi-domain labelled data is the scarcest resource in the setting. Self-supervised learning addresses the precondition rather than the objective, manufacturing supervision from unlabelled data so that a representation exists to which the previous three families can then be applied. It belongs here rather than as a fourth family because it is orthogonal to them: an SSL-pretrained encoder can be aligned, flattened, or disentangled exactly as a supervised one can.

### 5.1 Motivation

**Given a task and enough labels, supervised learning works really well. However:**

1. **Collecting a large amount of labelled data is expensive.**
2. **Scaling is difficult.**
3. **A very large amount of unlabelled data is available.**

Two responses exist, and the distinction matters:

| | **Unsupervised Learning**\vert **Self-Supervised Learning (SSL)** |
|---| ---\vert --- |
| Definition | **Relies on reconstruction or density estimation objectives**\vert **A form of unsupervised learning where the data (not a human) provides labels or a supervision signal** |

The SSL framing is the key conceptual move: the supervision is real supervision — there is a target, a loss, and a gradient — it is simply *manufactured from the data itself* rather than annotated by a person.

Notation used throughout this section:

$$\text{Labelled data: } \mathcal{D}_L = \{x_i, y_i\}_{i=1}^{N}, \qquad \text{Unlabelled data: } \mathcal{D}_U = \{x_i\}_{i=1}^{M}, \qquad M \gg N.$$

### 5.2 Supervised and Unsupervised Models Side by Side

**Supervised model.** Data $\mathcal{D}_L = \{x_i, y_i\}$ passes through a **feature extractor** $h_\theta$ and a **downstream task head** $g_\phi$ to produce $\hat{y}_i$:

![[03 - Attachments/ATML_L5_2_sup_vs_unsup.png|600]]

*Supervised model (encoder $h_\theta$ + task head $g_\phi$ on $\mathcal{D}_L$) and unsupervised model (encoder–decoder reconstructing $x_i$ on $\mathcal{D}_U$).*

$$\boxed{\text{Loss:} \quad \operatorname*{arg\,min}_{\theta,\phi}\;\sum_{i \in \mathcal{D}_L} L\Big(g_\phi\big(h_\theta(x_i)\big),\, y_i\Big)}$$

**$\theta, \phi$ are hundreds of millions of parameters; to fit them, $N$ must be large enough** — which is precisely the constraint motivation (1) identified as expensive.

**Unsupervised model.** Data $\mathcal{D}_U = \{x_i\}$ passes through an **encoder** $h_\theta$ and a **decoder** $g_\phi$ to produce $\hat{x}_i$, with the input itself as the target:

$$\boxed{\text{Loss:} \quad \operatorname*{arg\,min}_{\theta,\phi}\;\sum_{i \in \mathcal{D}_U} L\Big(g_\phi\big(h_\theta(x_i)\big),\, x_i\Big)}$$

The only structural difference is the target: $y_i$ versus $x_i$.

### 5.3 The SSL Recipe — Pretext Task and Pseudo-Labels

**SSL aims for the best of both worlds:** a **pretext task on $\mathcal{D}_U$** followed by a **target task on $\mathcal{D}_L$.**

An automated process $P$ converts the unlabelled set into a pseudo-labelled one:

$$\mathcal{D}_U = \{x_i\}_{i=1}^{M} \;\xrightarrow{\;P\;}\; P(\mathcal{D}_U) = \widetilde{\mathcal{D}}_L = \{x_i,\, z_i\}_{i=1}^{M}, \qquad z_i = \text{pseudo-label for } x_i.$$

> **Pretext task: a process $P$ to generate pseudo-labels, and an objective to guide learning.**

The **self-supervised objective is defined using $\widetilde{\mathcal{D}}_L$** with the feature extractor $h_\theta$ followed by a **pretext head** $K_\gamma$ producing $\hat{z}_i$:

$$\boxed{\text{Loss:} \quad \operatorname*{arg\,min}_{\theta,\gamma}\;\sum_{i \in \widetilde{\mathcal{D}}_L} L\Big(K_\gamma\big(h_\theta(x_i)\big),\, z_i\Big)}$$

**Once $h_\theta$ is trained, $K_\gamma$ is discarded** — it was scaffolding, useful only for generating a gradient — **and the model is adapted to the downstream target task using $\mathcal{D}_L = \{x_i, y_i\}_{i=1}^{N}$.**

### 5.4 Downstream Adaptation — Fine-Tuning versus Linear Readout

The adapted model is $x_i \to h_\theta \to g_\phi \to \hat{y}_i$, and there are two ways to train it.

![[03 - Attachments/ATML_L5_2_finetune_vs_linear.png|600]]

*After pretext training, $K_\gamma$ is dropped and $h_\theta$ is adapted: fine-tuning updates $\theta$ and $\phi$ starting from the SSL solution $\theta^*$; linear readout freezes $h_\theta$ and trains only $g_\phi$.*

**Fine-tuning:**

$$\boxed{\operatorname*{arg\,min}_{\theta,\phi}\;\sum_{i \in \mathcal{D}_L} L\Big(g_\phi\big(h_\theta(x_i)\big),\, y_i\Big)}$$

where **$h_\theta(\cdot)$ is initialized from the $h_\theta(\cdot)$ learned during SSL pre-training.** The loss-landscape picture shows $\theta^{*}$ **from SSL** sitting on a slope and migrating to a nearby $\theta$ **after fine-tuning** — a short move, because pre-training already placed the parameters in a good basin.

**Linear readout / probing:**

$$\boxed{\operatorname*{arg\,min}_{\phi}\;\sum_{i \in \mathcal{D}_L} L\Big(g_\phi\big(h_\theta(x_i)\big),\, y_i\Big)}$$

with **$h_\theta$ = fixed** and only **$g_\phi$ = trained.** This is **used when $\mathcal{D}_L$ is very sparse, i.e. $N$ is too small** — fine-tuning hundreds of millions of parameters on a handful of labels would simply overfit. **$g_\phi$ is often a linear function, hence the name linear readout.**

In both cases, **we are doing knowledge transfer from the pretext task to the target task.** Fine-tuning is preferred **in moderately sparse data, i.e. when $N$ is not too small and not too large.**

### 5.5 Summary Pipeline, and Transfer Learning versus SSL

The three stages of SSL in one line:

$$\underbrace{\mathcal{D}_U \xrightarrow{P} \widetilde{\mathcal{D}}_L}_{\text{pseudo-label generation}} \;\longrightarrow\; \underbrace{x_i \to h_\theta \to K_\gamma \to \hat{z}_i}_{\text{SSL pre-training}} \;\longrightarrow\; \underbrace{x \to h_\theta \to g_\phi \to \hat{y}_i}_{\text{downstream task adaptation}}$$

![[03 - Attachments/ATML_L5_2_ssl_summary_tl.png|620]]

*Top: the SSL pipeline (pseudo-label generation → SSL pre-training → downstream adaptation). Bottom: transfer learning pre-trains on a labelled supervised task, SSL on a pretext task over unlabelled data.*

with the setup $\mathcal{D}_U = \{x_i\}_{i=1}^{M}$, $\mathcal{D}_L = \{x_i, y_i\}_{i=1}^{N}$, and **$N \ll M$.**

**Transfer learning versus SSL.** Both consist of a **pre-training** stage followed by an **adaptation** stage, and both perform **knowledge transfer** into the same downstream model. The difference is entirely in what the pre-training stage consumes and optimizes:

| | **Pre-training data**\vert **Pre-training objective** |
|---| ---\vert --- |
| **SSL** | Unlabelled data\vert **Pretext task** |
| **TL** | Labelled data\vert **Supervised task** |

Both then transfer into a model adapted to the **target task** using **labelled data**.

### 5.6 Designing Pretext Tasks

> **SSL uses the intrinsic structure of unlabelled data $\mathcal{D}_U$ through an automated process $P$ to get $\widetilde{\mathcal{D}}_L = P(\mathcal{D}_U)$.**

The design flow: the **intrinsic structure of data**, combined with **human knowledge**, is used to **define a pretext task**, which is then **solved to get $h_\theta$** (and, incidentally, the $\hat{z}_i$ predictions, which are themselves discarded).

**Pretext tasks can be defined in many ways. The main things to keep in consideration are:**

1. **It should be meaningful.**
2. **It should induce some invariance properties of the data.**

The second criterion is what ties SSL back to the DG theme of this lecture: an invariance induced by a pretext task is an invariance the representation carries into the target domain.

### 5.7 Inferring Structure — Context Prediction

**Approach ① — inferring structure:** **define a pretext task (PT) in such a way that solving it is equivalent to inferring some structure of the data.**

![[03 - Attachments/ATML_L5_2_context_prediction.png|620]]

*Context prediction: sample a reference patch at the centre of a 3×3 grid and a target from its 8 neighbours; the model classifies which of 8 relative positions the target came from.*

**(a) Context prediction.** **The network should learn to recognize object parts and their spatial relations.**

Given a training image (the example is a **train on tracks**), **$x$ is divided into 2 pairs of patches** — one showing the train, one showing the tracks. The model must **predict the relative position between the two patches**, chosen from **8 classes**: **top-left, top, top-right, bottom-left, left, right, bottom, bottom-right.**

The concrete protocol on a $3\times3$ grid of **9 patches**:

- **Randomly sample the 1st patch** and call it the **reference patch (RP)**.
- **Consider that this patch is at the center of the grid, and sample the 2nd patch from its 8 neighbours** — the **target patch (TP)**.
- **Train the model to predict: "which one of the 8 neighbouring locations is the second patch selected from?"** $\Rightarrow$ **classification over 8 classes.**

So for reference patch $\text{RP} = 3$ and target patch $\text{TP} = 7$, the label is the relative direction (bottom-left), and the enumerated mapping runs $\text{RP}=1 \to \text{B.L.}$, $\text{RP}=2 \to \text{B.L.}$, $\text{RP}=3 \to \text{B.L.}$, $\text{RP}=4 \to \text{B}$, $\text{RP}=5 \to \text{B.L.}$, $\text{RP}=6 \to \text{B.L.}$, and so on through $\text{RP}=9 \to \text{Left}$.

**Pros:**

1. **First SSL method.**
2. **Intuitive task that should enable learning about object parts.**

**Cons:**

1. **Assumption: images have some canonical orientation.** A rotated or arbitrarily-oriented image makes "top-left" meaningless.
2. **Patches are used for training, with the expectation of learning an image representation** — a representation of parts is not automatically a representation of wholes.
3. **The network learns "shortcuts"** (developed in Section 5.8).
4. **No reason to distinguish cat from dog eyes.** The task can be solved perfectly without ever forming a category-level concept, since spatial arrangement does not require semantic identity.

### 5.8 Shortcut Learning and How to Avoid It

> **Shortcut learning: the model learns to solve a task by exploiting low-level trivial cues rather than capturing the actual pattern.**

Three specific shortcuts are identified for context prediction.

**① Straight-line continuity.** **When 2 patches are sampled from an image, natural straight lines might extend across their boundaries. If the model sees these lines, it will learn relative position based on that.** The illustration is a window frame: the mullion lines continue across the patch boundary, so the patches can be re-assembled by line-matching alone. **The model does not learn what defines a window if it learns to extend straight lines.**

**② Matching local patterns.** **At the boundary, there might be some pattern in (i) texture, (ii) colour, (iii) stripes, etc.** Any continuous low-level statistic across the seam is a giveaway.

**③ Pixel-level alignment.** **The model may rely on exact pixel relative alignment between the edges of the patches to predict relative position** — literally **alignment of pixel intensities** at the boundary, no semantics required at all.

**How to avoid?** Four countermeasures, all of which work by destroying the low-level continuity while preserving the semantic content:

- **Add gaps between patches.**
- **Add jitter to break continuity.**
- **Randomize colour, downsample patches, blur details.**
- **Chromatic aberration** — **shift colour channels to remove smoothness.**

The unifying principle: **add some noise to remove trivial cases as solutions for the model.** If the shortcut is made unreliable, the only remaining path to low loss is the intended one — the same logic as Stylized ImageNet (a 1.2-million-image, 1000-class image benchmark) in Lecture 4.

### 5.9 Inferring Structure — Jigsaw Puzzles

**(b) Jigsaw puzzles.** The motivating question: **why not use all 9 patches?**

![[03 - Attachments/ATML_L5_2_jigsaw.png|600]]

*Jigsaw: shuffle the 9 patches with one of $L$ predefined permutations and classify which permutation was applied.*

**Pretext task = jigsaw puzzle game:** the model must **put shuffled patches back to their original position.**

The combinatorial problem and its fix: **there are $9! = 362{,}880$ possible ways to shuffle 9 patches. Restrict it to, say, 100 pre-defined permutation sets. One of these 100 is chosen randomly for each image.**

The pipeline: the $3\times3$ patch grid goes through a **permutation** block (which **randomly chooses 1 permutation pattern from $L$**), and the shuffled grid is fed to the network, which performs **$L$-class classification** — it **predicts the probability of each predefined permutation pattern.**

Turning an intractable $9!$-way problem into a tractable $L$-way classification is the entire engineering contribution; the model never has to output a permutation, only recognize which of a known catalogue was applied.

### 5.10 Inferring Structure — Counting Features

**(c) Counting features.** **Features learned from different transformations of an image should satisfy a specific additive relationship.**

![[03 - Attachments/ATML_L5_2_counting_features.png|600]]

*Counting: the feature of the downsampled image should equal the sum of the features of its 4 tiles; the second hinge term stops the trivial solution $h_\theta(\cdot) = 0$.*

Two transformations are applied to the same image $x$. **Downsampling** gives $D(x)$, encoded as $h_\theta(D(x))$. **Tiling** splits $x$ into four tiles $T_1(x), T_2(x), T_3(x), T_4(x)$, each encoded as $h_\theta(T_i(x))$. The required relationship is

$$\boxed{h_\theta\big(D(x)\big) = \sum_{i=1}^{4} h_\theta\big(T_i(x)\big)}$$

which **ensures that features of the global image are distributed across its constituents and can be added together to reconstruct the whole.** The name "counting" comes from the intuition: if a feature dimension counts occurrences of some visual element, then the count over the whole image equals the sum of counts over its tiles.

The loss has two parts:

$$\boxed{\text{Loss} = \underbrace{\left\lVert h_\theta\big(D(x)\big) - \sum_{i=1}^{4} h_\theta\big(T_i(x)\big)\right\rVert_2^{2}}_{\text{feature consistency loss}} + \underbrace{\max\left(0,\; m - \left\lVert h_\theta\big(D(x)\big) - \sum_{i=1}^{4} h_\theta\big(T_i(x)\big)\right\rVert_2^{2}\right)}_{\text{feature differentiation loss}}}$$

The second term exists **to ensure $h_\theta(\cdot) = 0$ is not a solution to the task** — without it, the trivial all-zero encoder satisfies the additive constraint perfectly. The margin $m$ forces a minimum discrepancy on mismatched pairs, which is the standard hinge-style anti-collapse device.

> [!warning] Unclear in source The feature-differentiation term is written with the *same* argument as the consistency term, which would make the two terms directly contradictory (one minimizes the discrepancy, the other demands it exceed $m$). In the original Noroozi et al. formulation the second term uses a *different* image $y$, i.e. $\max\big(0,\, m - \lVert h_\theta(D(y)) - \sum_i h_\theta(T_i(x))\rVert_2^2\big)$, so that the model is pushed to give *different* counts for *different* images. The professor's form is reproduced as written; read the second term as applying to a mismatched image pair.

### 5.11 Transformation Prediction — Rotation

**Approach ② — transformation prediction.** A **transformation to change the canonical view of $x$** is applied, producing one of $\{0°, 90°, 180°, 270°\}$, and the **pretext task is: guess how much rotation is applied?**

![[03 - Attachments/ATML_L5_2_rotation.png|600]]

*Rotation prediction: rotate $x$ by $\omega \in \{0°, 90°, 180°, 270°\}$ and train $K_\gamma(h_\theta(\cdot))$ to predict $\omega$.*

The justification for why this teaches anything: **it becomes easier if the content is recognized first.** You cannot tell that a picture of a dog is upside-down without first knowing it is a dog and knowing which way dogs normally face — so solving the rotation task forces the network to acquire object-level knowledge as a by-product.

Formally, with **$\Omega = $ set of rotations $= \{0°, 90°, \ldots, 270°\}$** and **$\omega \sim \Omega$ one member sampled randomly from $\Omega$**, let **$T_\omega(x)$ be the transformation $\omega$ applied to $x$**, and take the **pseudo-label for $x$ to be $z_i = \omega$.** The pipeline is $x \to T_\omega(x) \to h_\theta \to K_\gamma \to$ loss, with the pseudo-label $z = \omega$ supplied from $\Omega$:

$$\boxed{\text{Loss} = \operatorname*{arg\,min}_{\theta,\gamma}\;\sum_{i} L_{\text{CE}}\Big(K_\gamma\big(h_\theta(x_i)\big),\, z_i\Big)}$$

### 5.12 Reconstruction

**Approach ③ — reconstruction.** **The pretext task $P$ distorts $x$, and the goal is to recover $x$.**

![[03 - Attachments/ATML_L5_2_reconstruction.png|560]]

*Reconstruction: mask a region $I$ of $x_i$, then regress the hidden pixels $z_i$ from the visible ones.*

For the masking case: **$I$ = the set of indices inside a square mask**, with **pixels inside $I$ = $z_i$** (the target to be predicted) and **pixels outside $I$ = $x_i$** (the visible context). The distorted image and its masked-out content form the pseudo-labelled pair in $\widetilde{\mathcal{D}}_L$, and the model $x_i \to h_\theta \to K_\gamma \to \hat{z}_i$ is trained with a squared loss:

$$\boxed{\text{Loss} = \operatorname*{arg\,min}_{\theta,\gamma}\;\sum_{i \in \widetilde{\mathcal{D}}_L}\Big(K_\gamma\big(h_\theta(x_i)\big) - z_i\Big)^{2}}$$

**Common distortion methods:** **masking**, **noise**, **blur**, and others. This is the family that later produced masked autoencoders and, in language, masked language modelling.

### 5.13 Colorization and Split-Brain Decoders

**Colorization.** A **coloured** image $x$ is converted to **greyscale** and fed to the model, which must answer: **what is the colour of every pixel?** The task **needs object recognition** — knowing that grass is green and skin is not requires knowing what grass and skin are — **but it is a very hard and ambiguous process**, since many plausible colourings exist for the same greyscale input (a car may be red or blue with equal prior support), and the loss penalizes all but the true one.

![[03 - Attachments/ATML_L5_2_colorization_splitbrain.png|600]]

*Colorization predicts colour from greyscale; split-brain decoders train two networks that predict each channel group from the other and reconstruct $x$ from both.*

**Split-brain decoders.** A symmetric two-branch refinement. The coloured image $x$ is split into its **greyscale** part $x_g$ and its **colour channel** part $x_c$. One network $h_{\theta_1}$ takes $x_g$ and **predicts colour $x_c$**; the other, $h_{\theta_2}$, takes $x_c$ and **predicts greyscale $x_g$**. The two predictions are then combined to **reconstruct $x$.**

Making the task symmetric is what fixes the asymmetry of plain colorization: each half of the representation must be predictable from the other, so neither channel can be ignored.

### 5.14 SSL versus Unsupervised Learning

**SSL uses a pretext task** — context prediction, contrastive learning, reconstruction, and others — **while UL uses reconstruction or clustering.**

Two arguments are given for why the pretext framing is preferable, and both are DG arguments.

**First:** **reconstruction usually overfits the model on the given data distribution. In a pretext task, we introduce artificial shifts to the given data distribution — in a way, we are simulating OOD (out-of-distribution) shifts.** This is the crucial link back to the lecture's theme: rotating, masking, and shuffling are precisely synthetic domain shifts, so a representation that survives them has already been trained for the kind of variation a real target domain will impose.

**Second:** **reconstruction usually leads to memorization of the input pdf, because the reconstruction loss emphasizes pixel-level fidelity rather than higher-level semantics — e.g. it may focus more on reconstructing texture, or paint, or shadow, or lighting conditions.** Those are exactly the domain-dependent, non-causal features of Lecture 5.1's taxonomy. A pixel-fidelity objective actively *rewards* encoding them.

### 5.15 Instance Classification

**Approach ④ — instance classification: treat each $x_i \in \mathcal{D}_U$ as its own class, to train a model to distinguish between different instances.**

**Using cross-entropy.** With $|\mathcal{D}_U| = 10^{5}$, the sample $x_{126}$ is paired with a one-hot target $y_{126} \in \mathbb{R}^{10^{5}}$ **with 1 at the 126th position.** **The network can be trained to distinguish $x_{126}$ from the rest of the $x_i$.** Two objections follow immediately:

- **Not scalable, due to the softmax limitation** — a $10^5$-way (or, at web scale, $10^9$-way) softmax is computationally prohibitive.
- **Only 1 instance in a class** $\longrightarrow$ the fix is **data augmentation**, which manufactures additional members for each singleton class.

**Using contrastive.** **Idea: do not predict the exact class for any given $x_i$; rather, predict whether pairs of inputs belong to the same or different classes:**

$$(x_i, x_j),\; i \neq j \;\longrightarrow\; \boxed{\text{Prediction}} \;\longrightarrow\; \begin{cases}\text{same class} \to \text{label} = 1\\ \text{diff. class} \to \text{label} = 0\end{cases}$$

Reformulating an $N$-way classification as a binary same-or-different judgement is what dissolves the softmax scaling problem.

**Constructing the pairs.** Sample $x_i$ from $\mathcal{D}_U$ and apply **data augmentation $T(x_i)$** to obtain **multiple views of $x_i$**. Then:

- **$x^{a}$: one input is chosen to be the anchor.**
- **$x^{+}$: another view of $x^{a}$ — the positive pair.**
- **$x^{-}$: a view of a different image — the negative pair.** **Get $K$ negative pairs by repeating this process:** $x_k^{-} = T(x_j)$ for $k = 1, 2, \ldots, K$, with $x_j$ **sampled from $\mathcal{D}_U$ such that $j \neq i$.**

### 5.16 Contrastive Instance Discrimination

Assembling the above, the pretext process $P$ maps each $x_i$ to a set of $K$ pairs with their binary pseudo-labels:

$$x_i \;\xrightarrow{\;P\;}\; x_i = \Big\{(x^{a}, x^{+}),\, (x^{a}, x_1^{-}),\, \ldots,\, (x^{a}, x_K^{-})\Big\} \quad (K \text{ pairs}), \qquad z_i = \{1,\, 0,\, 0,\, \ldots,\, 0\} \in \mathbb{R}^{K\times1}$$

![[03 - Attachments/ATML_L5_2_instance_discrimination.png|560]]

*Instance discrimination: $P$ builds one positive pair $(x^a, x^+)$ and $K$ negatives; the encoder features go through $K_\gamma$ into the contrastive loss.*

Each view is encoded by $h_\theta(\cdot)$ into a representation:

$$\Big\{\underbrace{h_\theta(x^{a})}_{r^{a}},\; \underbrace{h_\theta(x^{+})}_{r^{+}},\; \underbrace{h_\theta(x_1^{-})}_{r_1^{-}},\; \ldots,\; \underbrace{h_\theta(x_K^{-})}_{r_K^{-}}\Big\}$$

and passed through the pretext head $K_\gamma(\cdot)$, giving the objective

$$\boxed{\text{Loss:} \quad \operatorname*{arg\,min}_{\theta,\gamma}\;\sum_{i \in \widetilde{\mathcal{D}}_L} L_{\text{const.}}\Big(K_\gamma\big(h_\theta(x_i)\big),\, z_i\Big)}$$

where the contrastive loss itself is the familiar InfoNCE form, **averaged over a batch**:

$$\boxed{L_{\text{const.}} = -\,\mathbb{E}\left[\log \frac{\text{Sim}\big(r^{a},\, r^{+}\big)}{\text{Sim}\big(r^{a}, r^{+}\big) + \sum_{j=1}^{K}\text{Sim}\big(r^{a},\, r_j^{-}\big)}\right]}$$

**Regularization.** The $\sum_{j=1}^{K}$ term over negatives is where the practical difficulty lives: **collecting negative samples is problematic, and advanced methods use some sort of regularization to get rid of the $K$ negative samples.** Three named approaches:

- **MoCo** (Momentum Contrast, contrastive learning with a memory bank of negatives) — a **memory bank of negative samples**, so negatives need not all be recomputed in the current batch.
- **SimCLR** (a simple contrastive-learning framework using large batches of augmented views) — which instead relies on very large batches to supply negatives.
- **Temperature scaling inside the $\text{Sim}(\cdot)$ function** — controlling how sharply the loss distinguishes near from far negatives, which (per Lecture 3) governs how much of the gradient the hardest negatives absorb.

### 5.17 Clustering

**Approach ⑤ — clustering: divide $\mathcal{D}_U$ into a number of groups with**

1. **High intra-group similarity**
2. **Low inter-group similarity**

![[03 - Attachments/ATML_L5_2_clustering.png|480]]

*Clustering: cluster the features $h_\theta(x_i)$ into $K$ groups, use the nearest centre as pseudo-label $z_i$, train with cross-entropy, then re-cluster and repeat.*

The procedure alternates between assigning clusters and training on those assignments. Each $x_i \sim \mathcal{D}_U$, $i = 1, 2, \ldots, M$, is encoded as $r_i = h_\theta(x_i)$. These representations are **clustered into $K$ clusters**, yielding **cluster centres $\{c_1, c_2, \ldots, c_K\}$.** The pseudo-label process $P$ then takes each $x_i$ together with the centres and its representation $r_i$, and assigns the nearest centre:

$$\boxed{z_i = \operatorname*{arg\,min}_{j \in \{1,\ldots,K\}}\;\big\lVert c_j - r_i\big\rVert}$$

**Given the cluster assignments**, train the network on them as if they were true labels:

$$\boxed{\operatorname*{arg\,min}_{\theta,\gamma}\;\sum_{i \in \widetilde{\mathcal{D}}_L} L_{\text{CE}}\Big(K_\gamma\big(h_\theta(x_i)\big),\, z_i\Big)}$$

**Given the current $\theta, \gamma$, do the cluster assignments again** — and **so on**, alternating indefinitely. This is structurally identical to the K-means pseudo-labelling used in contrastive DA (Lecture 5.1, §6.3), with the difference that there are no source-class prototypes to seed the centres, so the clusters must bootstrap themselves from a randomly-initialized representation.

---

## Looking Ahead

Module II closes here, and the two lectures are best held together as one argument with a parameter. Lecture 5.1 has target data and can therefore measure the divergence it needs to minimize; this lecture has none and must bound the target error over a hull instead. Every difference between the two follows from that, including why DG methods are so much more indirect: with nothing to align *to*, one aligns the sources to each other, or abandons alignment for a property — flatness, invariance of the optimal head, worst-case risk — that is defined without reference to any target at all.

The three families and the levers they pull are worth retaining as a unit:

| Family | Bound term attacked\vert Mechanism\vert Principal limitation |
|---| ---\vert ---\vert --- |
| Domain alignment (§2) | hull radius $\rho$\vert make source features mutually indistinguishable\vert marginal alignment need not be class-conditional (§2.5) |
| Learning strategies (§3) | source risks $L_i$, robustly\vert flat minima, invariant heads, worst-case risk\vert surrogates are first-order and can be gamed (§3.5) |
| Disentanglement (§4) | $\gamma$, by isolating causal factors\vert one factor per latent slot\vert unidentifiable without supervision or bias (§4.15) |

A recurring caution unites the three. Each family optimizes a *proxy* for domain invariance — indistinguishable marginals, a flat minimum, an axis-aligned latent — and each proxy admits solutions that satisfy it while failing the underlying goal. Recognizing which proxy a method uses, and how it can be satisfied vacuously, is the most transferable skill this module offers, and it is the form every result in the remaining modules should be read in.

Looking back across the course, the arc from [[ATML_Lecture_1 - Representation and Generalization Theory]] is closed: the bound stated there, $\text{Loss}_{\text{Test}} \le \text{Loss}_{\text{Train}} + \frac{1}{\sqrt{N}}\frac{\beta\sqrt{S}}{\gamma}$, assumed a single distribution, and Module II replaced it with bounds in which the relationship between two distributions is an explicit, and usually dominant, term.

---

## Numerical Problems

> [!example]- Problem 1 — Reading the DG error bound
> Three source domains have risks $L_1 = 0.10$, $L_2 = 0.14$, $L_3 = 0.18$ under uniform mixture weights $\pi_i = 1/3$. The discrepancy term evaluates to $\gamma = 0.22$, the convex-hull radius is $\rho = 0.09$, and the constant is $0.02$. Compute the bound on $L_T(f_\theta)$, and determine which single lever gives the largest reduction if it could be halved.
>
> *Solution.* Applying the bound of Section 1.2, the weighted source risk is $\frac{1}{3}(0.10 + 0.14 + 0.18) = \frac{0.42}{3} = 0.14$. Adding the remaining three terms, $L_T \le 0.14 + 0.22 + 0.09 + 0.02$.
>
> $$\boxed{L_T(f_\theta) \le 0.47}$$
>
> Halving each term in turn gives reductions of $0.07$ (source risk), $0.11$ (discrepancy $\gamma$), and $0.045$ (radius $\rho$), so the discrepancy term dominates.
>
> *Reasonableness check:* that $\gamma$ is the largest single contributor is the expected situation in DG and is precisely the awkward one — $\gamma$ measures the distance from the target to the convex hull, and since the target is unavailable at training time it cannot be minimized directly. This is why lever 3 of Section 1.3 is phrased indirectly, as *learning a representation whose source hull covers future targets*, rather than as minimizing $\gamma$ outright.

> [!example]- Problem 2 — The positive-pull / negative-push identity
> A contrastive batch has one positive and four negatives, with similarity scores $s_{\text{pos}} = 3.0$ and $s_1 = 1.0$, $s_2 = 0.5$, $s_3 = 0.0$, $s_4 = -1.0$. Compute $P_{\text{pos}}$, the gradient magnitude at the positive, and the summed gradient over the negatives. Verify the identity of Section 2.6.
>
> *Solution.* The exponentials are $e^{3.0} \approx 20.086$, $e^{1.0}\approx 2.718$, $e^{0.5}\approx 1.649$, $e^{0.0} = 1$, and $e^{-1.0}\approx 0.368$, summing to $25.821$. Therefore $P_{\text{pos}} = 20.086/25.821 \approx 0.7779$, and the individual negative probabilities are $P_1 \approx 0.1053$, $P_2 \approx 0.0639$, $P_3 \approx 0.0387$, $P_4 \approx 0.0143$. Using $\partial L/\partial s_{\text{pos}} = P_{\text{pos}} - 1$, the magnitude at the positive is $|0.7779 - 1| = 0.2221$; summing the negatives' gradients $\sum_i P_i \approx 0.1053 + 0.0639 + 0.0387 + 0.0143 = 0.2222$.
>
> $$\boxed{P_{\text{pos}} \approx 0.778, \qquad \left|\partial L/\partial s_{\text{pos}}\right| \approx 0.222 = \sum_{i \neq \text{pos}} \partial L/\partial s_i}$$
>
> *Reasonableness check:* the two quantities agree to rounding, confirming the boxed identity. Now consider a hard positive with $s_{\text{pos}} = -1.0$ instead: $P_{\text{pos}}$ would fall to roughly $0.06$, making the pull magnitude about $0.94$ — more than four times larger — and that entire $0.94$ would be redistributed as push onto the four negatives regardless of whether they needed moving. That amplification is exactly the wasted-gradient pathology of Section 2.6.

> [!example]- Problem 3 — SAM perturbation and sharpness
> At the current parameters the gradient is $g_\theta = (3, 4)^{\top}$ and the neighbourhood radius is $\rho = 0.1$. Compute the linearized worst-case perturbation $\bar\epsilon$. Separately, at a minimum where $\nabla_\theta L = 0$ with $\lambda_{\max}(H_\theta) = 50$, compute the maximum sharpness for the same $\rho$.
>
> *Solution.* Applying the boxed result of Section 3.7, $\bar\epsilon = \rho\, g_\theta/\lVert g_\theta\rVert_2$ with $\lVert g_\theta\rVert_2 = \sqrt{9+16} = 5$, giving $\bar\epsilon = 0.1 \times (3/5,\, 4/5)^{\top}$.
>
> $$\boxed{\bar{\epsilon} = (0.06,\; 0.08)^{\top}, \qquad \lVert\bar\epsilon\rVert_2 = 0.1 = \rho}$$
>
> For the sharpness, the second-order result of Section 3.8 gives $\frac{\rho^2}{2}\lambda_{\max}(H_\theta) = \frac{0.01}{2}(50)$.
>
> $$\boxed{\text{Max sharpness} = 0.25}$$
>
> *Reasonableness check:* $\bar\epsilon$ has norm exactly $\rho$, as the Cauchy–Schwarz equality condition requires — the worst-case perturbation always sits on the boundary of the ball, never inside it. And the sharpness scales as $\rho^2$, so halving the neighbourhood radius to $0.05$ would cut the sharpness by a factor of four to $0.0625$; comparisons of sharpness across models are only meaningful at a fixed $\rho$.

> [!example]- Problem 4 — β-VAE KL term and the capacity objective
> A 3-dimensional latent has encoder outputs $\mu_\phi = (0,\, 0.5,\, 1.0)$ and $\sigma_\phi = (1.0,\, 1.0,\, 0.5)$. Using the standard closed form $D_{\text{KL}} = \frac{1}{2}\sum_j\big[\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big]$, compute the KL. Then, with $\beta = 4$ and target capacity $C = 0.5$, compute the penalty contributed by the capacity-control objective of Section 4.16, and compare it against plain β-VAE.
>
> *Solution.* Per dimension: for $j=1$, $0 + 1 - 1 - \log 1 = 0$; for $j=2$, $0.25 + 1 - 1 - 0 = 0.25$; for $j=3$, $1 + 0.25 - 1 - \log(0.25) = 0.25 + 1.3863 = 1.6363$. Summing and halving, $D_{\text{KL}} = \frac{1}{2}(0 + 0.25 + 1.6363)$.
>
> $$\boxed{D_{\text{KL}} \approx 0.943 \text{ nats}}$$
>
> Under plain β-VAE the penalty is $\beta\,D_{\text{KL}} = 4(0.943) = 3.772$. Under capacity control it is $\beta\,|D_{\text{KL}} - C| = 4\,|0.943 - 0.5| = 4(0.443)$.
>
> $$\boxed{\text{β-VAE penalty} = 3.772; \qquad \text{capacity-control penalty} = 1.772}$$
>
> *Reasonableness check:* dimension 1 contributes exactly zero — it has collapsed to the prior, carrying no information about $x$, which is the per-dimension collapse described in Section 4.15. The capacity-control penalty is smaller here, but the more important difference is directional: if the KL fell *below* $C$, plain β-VAE would keep rewarding the drop while capacity control would start penalizing it, which is precisely the mechanism that prevents further collapse.

> [!example]- Problem 5 — Mutual Information Gap
> A generative factor $v_1$ has entropy $H(v_1) = 2.0$ bits. Its mutual information with the best-matching latent dimension is $I(v_1, z_*) = 1.6$ bits and with the runner-up $I(v_1, z_0) = 0.3$ bits. A second factor has $H(v_2) = 1.5$, $I(v_2, z_*) = 1.2$, $I(v_2, z_0) = 0.9$. Compute $\text{MIG}_1$, $\text{MIG}_2$, and the overall MIG, and say which factor is better disentangled.
>
> *Solution.* Applying the definition of Section 4.14, $\text{MIG}_1 = (1.6 - 0.3)/2.0 = 1.3/2.0 = 0.65$ and $\text{MIG}_2 = (1.2 - 0.9)/1.5 = 0.3/1.5 = 0.20$. Averaging over $K = 2$ factors, $\text{MIG} = \frac{1}{2}(0.65 + 0.20)$.
>
> $$\boxed{\text{MIG}_1 = 0.65, \qquad \text{MIG}_2 = 0.20, \qquad \text{MIG} = 0.425}$$
>
> Factor $v_1$ is better disentangled.
>
> *Reasonableness check:* both values lie in $[0,1]$ as the entropy normalization guarantees. Note that $v_2$ is actually *well encoded* — $1.2$ of its $1.5$ bits are captured — but it is smeared across two dimensions almost equally, so its MIG is low. This is exactly the behaviour the gap formulation is designed to expose: high mutual information alone would have rated $v_2$ favourably, and only the margin over the runner-up reveals the entanglement.

> [!example]- Problem 6 — SSL pretext task scaling
> A jigsaw pretext task uses a $3\times3$ grid. Compute the full permutation count, and the reduction factor achieved by restricting to $L = 100$ predefined permutations. Then, for instance classification on $|\mathcal{D}_U| = 10^5$ images, compare the output-layer parameter count of the cross-entropy formulation against the contrastive formulation, assuming a $2048$-dimensional representation.
>
> *Solution.* From Section 5.9, the permutation count is $9! = 362{,}880$, and restricting to $L = 100$ classes gives a reduction factor of $362{,}880/100$.
>
> $$\boxed{9! = 362{,}880 \;\longrightarrow\; 100, \qquad \text{reduction} \approx 3{,}629\times}$$
>
> For instance classification (Section 5.15), the cross-entropy head is a linear map from the representation to one logit per instance, requiring $2048 \times 10^5 = 2.048 \times 10^8$ weights. The contrastive formulation replaces the classification head with a similarity computation between pairs of $2048$-dimensional embeddings, which needs no per-instance parameters at all — only an optional projection head of roughly $2048 \times 128 \approx 2.6\times10^5$ weights.
>
> $$\boxed{\text{CE head: } \approx 2.05\times10^{8} \text{ weights}; \qquad \text{contrastive: } \approx 2.6\times10^{5} \text{ weights (}\sim790\times \text{ fewer)}}$$
>
> *Reasonableness check:* both numbers make the same point from different directions — the naive formulation scales with the *size of the dataset*, which is untenable, while the reformulated one scales only with the *embedding dimension*, which is fixed. This is the concrete content of the "not scalable due to the softmax limitation" objection in Section 5.15, and it is the same kind of combinatorial reduction the jigsaw permutation restriction achieves.

---

## Formula Summary Sheet

| Concept                           | Formula                                                                                                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------\vert ----------------------------------------------------------\vert --- |
| DG assumption                     | $P_T(x) \in \Lambda = $ convex hull of $\{P_i(x)\}_{i=1}^{M}$                                                                                                              |
| DG error bound                    | $L_T(f_\theta) \le \sum_{i=1}^{M}\pi_i L_i(f_\theta) + \gamma\big(P_T(x), \sum_i \pi_i P_i(x)\big) + \rho + \text{const.}$                                                 |
| Bound terms                       | $\gamma$ = min discrepancy between $P_T$ and $\Lambda$; $\rho$ = radius of convex hull                                                                                     |
| Combined alignment objective      | $L_{\text{Total}} = L_{\text{sup}} + L_{\text{AE}} + L_{\text{MMD}} + L_D$                                                                                                 |
| Cross-domain contrastive loss     | $L_{\text{CL}} = -\log\frac{\exp(\text{sim}(z_a,z_+))}{\sum_k \exp(\text{sim}(z_a,z_k)) + \exp(\text{sim}(z_a,z_+))}$                                                      |
| Softmax probabilities             | $P_k = \frac{\exp(s_k)}{\sum_j \exp(s_j)}$, $P_{\text{pos}} + \sum_k P_k = 1$                                                                                              |
| Contrastive gradients             | $\frac{\partial L_{\text{CL}}}{\partial s_i} = P_{\text{pos}}-1$ if $i=\text{pos}$; $P_i$ if $i$ is a negative                                                             |
| Pull/push identity                | $\sum_{i\neq\text{pos}} \frac{\partial L}{\partial s_i} = 1 - P_{\text{pos}} = \big\vert \frac{\partial L_{\text{CL}}}{\partial s_{\text{pos}}}\big\vert $ |
| CE as proxy contrastive           | $L_{\text{CE}} = -\psi_a^\top z_a + \log\big(\sum_{k=1}^{c-1} e^{\psi_k^\top z_a}\big)$                                                                                    |
| Class proxy                       | $v_c = g_c(w_c)$, $c = 1,\ldots,K$; $w_c$ learnable                                                                                                                        |
| Proxy contrastive loss            | $L_{\text{PCL}} = -\frac{1}{N}\sum_{i=1}^{N}\frac{\exp(\text{sim}(v_c, z_i))}{E}$                                                                                          |
| PCL denominator                   | $E = \exp(\text{sim}(v_c,z_i)) + \sum_{K=1}^{c-1}\exp(\text{sim}(v_K,z_i)) + \sum_{j\neq i}^{N}\exp(\text{sim}(z_i,z_j))$                                                  |
| PCL total                         | $L_{\text{Total}} = L_{\text{CE}} + L_{\text{PCL}}$; update $\theta$, $w_c$, $g_c(\cdot)$                                                                                  |
| Vanilla IRM                       | $\min_{\theta,w}\sum_{i=1}^M R_i(h_w\circ g_\theta)$ s.t. $w^* = \arg\min_w R_i(h_w\circ g_\theta)\;\forall i$                                                             |
| IRM first-order condition         | $\nabla_w R_i(h_w \circ g_\theta)\big\vert _{w=w^*} = 0 \;\forall i$ |
| IRMv1                             | $\min_\theta \sum_i R_i(h_w\circ g_\theta) + \lambda\sum_i \lVert\nabla_w R_i\vert _{w=1}\rVert^2$ |
| Scale-probe risk                  | $R_i(\theta, s) = \mathbb{E}_{P_i(x,Y)}\big[L_{\text{CE}}(s f_\theta(x_i), Y)\big]$                                                                                        |
| Scale-probe objective             | $\min_\theta\sum_i R_i(\theta, s{=}1) + \lambda\sum_i\lVert\nabla_s R_i(s,\theta)\vert _{s=1}\rVert_2^2$ |
| SAM objective                     | $\theta_{\text{SAM}} = \min_\theta \max_{\lVert\epsilon\rVert\le\rho} L(\theta+\epsilon)$                                                                                  |
| SAM linearization                 | $L(\theta+\epsilon) \approx L(\theta) + \nabla_\theta L(\theta)^\top \epsilon$                                                                                             |
| SAM perturbation (1st order)      | $\bar\epsilon = \rho\,\frac{g_\theta}{\lVert g_\theta\rVert_2} = \rho\,\frac{\nabla_\theta L_\theta}{\lVert\nabla_\theta L_\theta\rVert_2}$                                |
| Max sharpness at a minimum        | $\frac{\rho^2}{2}\lambda_{\max}(H_\theta)$, $H_\theta = \nabla_\theta^2 L(\theta)$                                                                                         |
| SAM perturbation (with curvature) | $\bar\epsilon = \rho\frac{g_\theta}{\lVert g_\theta\rVert} + \alpha\,\text{sign}(\langle g_\theta, V_1\rangle)V_1^{\perp}$                                                 |
| Stochastic Weight Averaging       | $\bar\theta = \frac{1}{K}\sum_{i=1}^{K}\theta_i$; SWAD = SWA + dense snapshots                                                                                             |
| Per-domain sharpness              | $S_i = L_{\text{sup}}(\theta+\epsilon_i) - L_{\text{sup}}(\theta)$, $\epsilon_i = \rho\frac{\nabla_\theta L_{\text{sup}}}{\lVert\nabla_\theta L_{\text{sup}}\rVert}$       |
| Sharpness gap                     | $\lvert S_i - S_j\rvert$                                                                                                                                                   |
| Soft-label refinement loss        | $L_{\text{Refine}}(\psi) = \text{KL}(\tilde{y}\,\Vert \,y) + \lambda_1\lvert S_i - S_j\rvert$                                                                                  |
| G-DRO                             | $\min_\theta \max_{g\in\{1,\ldots,m\}} \mathbb{E}_{(x,Y)\sim P_g}[L(\theta;x,Y)]$                                                                                          |
| DRO over perturbation             | $\min_\theta \mathbb{E}_{P_i(x,Y)}\big[\max_{s\in\mathcal{U}} L(\theta; T_s(x), Y)\big]$                                                                                   |
| CCFP discrepancy loss             | $L_{\text{discrepancy}} = -\sum_{i=1}^{K}\lVert G(f_o^{(i)}(x)) - G(f_p^{(i)}(x))\rVert_F$                                                                                 |
| CCFP semantic loss                | $L_{\text{semantic}} = \lVert g_o(x) - g_p(x)\rVert_2^2$                                                                                                                   |
| CCFP total                        | $L_{\text{Total}} = L_{\text{semantic}} + L_{\text{CE}}^{o} + L_{\text{CE}}^{p} + L_{\text{discrepancy}}$                                                                  |
| Dimension-wise latent             | $z \in \mathbb{R}^{K_1\times1}$, $z_i$ scalar                                                                                                                              |
| Vector-wise latent                | $Z = [\underline{z}_1 \cdots \underline{z}_{K_2}] \in \mathbb{R}^{d\times K_2}$, $d\ge2$                                                                                   |
| DRL loss                          | $L = \lambda L_{\text{recons.}} + \lambda_2 L_{\text{Disent.}} + \lambda_3 L_{\text{Task}}$                                                                                |
| Factor independence               | $P(v\mid x) = \prod_{i=1}^K P(v_i\mid x) \Rightarrow \log P(v\mid x) = \sum_i \log P(v_i\mid x)$                                                                           |
| VAE posterior                     | $q_\phi(z\mid x) = \mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))$; $z_i = \mu_\phi^i + \sigma_{\phi,i}\odot\epsilon_i$                                                 |
| Isotropic prior                   | $P(z) = \mathcal{N}(0, I_{M\times M})$                                                                                                                                     |
| VAE objective                     | $L_{\text{VAE}} = \mathbb{E}_{q_\phi}[\log P_\theta(x\mid z)] - D_{\text{KL}}(q_\phi(z\mid x)\Vert P(z))$                                                                      |
| β-VAE objective                   | $L_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}[\log P_\theta(x\mid z)] - \beta D_{\text{KL}}(q_\phi(z\mid x)\Vert P(z))$                                                          |
| Fixed-variance decoder            | $\log P_\theta(x\mid z) \propto \lVert x - \mu_\theta(z)\rVert_2^2$                                                                                                        |
| Free-variance decoder             | $\log P_\theta(x\mid z)\propto\frac{1}{2}\sum_{j=1}^d\big[\frac{(x_j-\mu_\theta^j)^2}{\sigma_{\theta,j}^2} + \log\sigma_{\theta,j}^2\big]$                                 |
| β-VAE batch loss                  | $L_{\text{Batch}} = \frac{1}{B}\sum_{b=1}^{B}\big[-\log P_\theta(x^{(b)}\mid z^{(b)}) + \beta D_{\text{KL}}(q_\phi(z\mid x^{(b)})\Vert P(z))\big]$                             |
| Diagonal-Gaussian KL (standard)   | $D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{M}\big[\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - 1 - \log\sigma_{\phi,j}^2\big]$                                                        |
| Z-Diff metric                     | $z_{\text{diff}} = \frac{1}{L}\sum_{\ell=1}^{L}z_{\text{diff}}^{(\ell)}$, $z_{\text{diff}}^{(\ell)} = \lvert z^{(1)} - z^{(2)}\rvert$; metric = linear-classifier accuracy |
| Mutual Information Gap            | $\text{MIG} = \frac{1}{K}\sum_{i=1}^{K}\frac{1}{H(v_i)}\big[I(v_i,z_*) - I(v_i,z_0)\big]$                                                                                  |
| Capacity control                  | $L_{\text{C-VAE}} = \mathbb{E}_{q_\phi}[P_\theta(x\mid z)] - \beta\big\lvert D_{\text{KL}}(q_\phi(z\mid x)\Vert P(z)) - C\big\rvert$                                           |
| Semi-supervised β-VAE             | $L = \mathbb{E}_{q_\phi}(P_\theta(x\mid z)) - \beta D_{\text{KL}}(q_\phi(z\mid x)\Vert P(z)) + \beta_{\otimes}L_{\text{sup}}$                                                  |
| Vector-wise DRL loss              | $L_{\text{oss}} = \lambda_0 L_{\text{orth.}} + \lambda_1 L_{\text{recons.}} + \lambda_2 L_{\text{Task}}$                                                                   |
| Supervised objective              | $\arg\min_{\theta,\phi}\sum_{i\in\mathcal{D}_L} L(g_\phi(h_\theta(x_i)), y_i)$                                                                                             |
| Unsupervised objective            | $\arg\min_{\theta,\phi}\sum_{i\in\mathcal{D}_U} L(g_\phi(h_\theta(x_i)), x_i)$                                                                                             |
| SSL pretext objective             | $\arg\min_{\theta,\gamma}\sum_{i\in\widetilde{\mathcal{D}}_L} L(K_\gamma(h_\theta(x_i)), z_i)$                                                                             |
| Fine-tuning                       | $\arg\min_{\theta,\phi}\sum_{i\in\mathcal{D}_L} L(g_\phi(h_\theta(x_i)), y_i)$, $h_\theta$ initialized from SSL                                                            |
| Linear readout                    | $\arg\min_{\phi}\sum_{i\in\mathcal{D}_L} L(g_\phi(h_\theta(x_i)), y_i)$, $h_\theta$ fixed                                                                                  |
| Counting-features constraint      | $h_\theta(D(x)) = \sum_{i=1}^{4} h_\theta(T_i(x))$                                                                                                                         |
| Counting loss                     | $\lVert h_\theta(D(x)) - \sum_i h_\theta(T_i(x))\rVert_2^2 + \max\big(0, m - \lVert\cdot\rVert_2^2\big)$                                                                   |
| Rotation prediction               | $\Omega = \{0°,90°,180°,270°\}$, $z_i = \omega$; $\arg\min_{\theta,\gamma}\sum_i L_{\text{CE}}(K_\gamma(h_\theta(x_i)), z_i)$                                              |
| Reconstruction pretext            | $\arg\min_{\theta,\gamma}\sum_{i}\big(K_\gamma(h_\theta(x_i)) - z_i\big)^2$                                                                                                |
| Contrastive instance loss         | $L_{\text{const.}} = -\mathbb{E}\Big[\log\frac{\text{Sim}(r^a, r^+)}{\text{Sim}(r^a,r^+) + \sum_{j=1}^{K}\text{Sim}(r^a, r_j^-)}\Big]$                                     |
| Clustering pseudo-label           | $z_i = \arg\min_{j\in\{1,\ldots,K\}}\lVert c_j - r_i\rVert$                                                                                                                |
| Jigsaw combinatorics              | $9! = 362{,}880$ permutations, restricted to $L$ (e.g. 100) predefined sets                                                                                                |

---

## Cheat Sheet & Recall Aids

**Mnemonics**

- **Three DG families — "A-L-D: Align, Learn, Disentangle."** Alignment changes what the representation *looks like*; learning strategies change what the optimizer is *asked to achieve*; disentanglement changes how the representation is *structured*.
- **Three bound levers — "Risk, Radius, Reach":** lower each source risk $L_i$, shrink the hull radius $\rho$, extend the hull's reach so it covers future targets ($\gamma$).
- **Five SSL pretext families — "I-T-R-I-C":** Inferring structure (context, jigsaw, counting), Transformation prediction (rotation), Reconstruction (masking, colorization), Instance classification, Clustering.

**Gotchas**

- **DG vs DA:** DA sees unlabelled target data; DG sees none, so source *diversity* is the training signal. The bound means nothing unless $P_T \in \Lambda$.
- **$\gamma$ vs $\rho$:** $\gamma$ (target-to-hull) cannot be minimized, since the target is absent; $\rho$ (hull radius) can, by pulling sources together.
- **Hard positives hurt twice:** the large pull collapses two domains' sub-manifolds, and by $\sum_{i\ne\text{pos}}\partial L/\partial s_i = |\partial L/\partial s_{\text{pos}}|$ the same magnitude is pushed onto well-placed negatives. Proxies fix it because one stable vector per class cannot be an outlier.
- **IRMv1 certifies stationarity, not optimality:** the penalty can be zeroed by flattening the problem (spurious stationary solutions). Fixes: REx/VREx, Fish, entropy measures.
- **SAM's $\bar\epsilon$ lies along the gradient** (linearize, then Cauchy–Schwarz). Curvature adds only $V_1^{\perp}$, since the parallel part is already in the first term.
- **SWA only at the tail of training:** averaging iterates from two basins lands on the ridge between them.
- **G-DRO fails twice in DG:** over-parameterized nets drive the worst-group loss to 0 as well, and there are no group labels. DRO over a perturbation set (CCFP) replaces the groups with an adversary.
- **Reconstruction terms are mandatory in DRL:** an all-zeros code is perfectly disentangled and useless.
- **β-VAE's trade-off is structural:** the KL term *is* the information channel, so a larger $\beta$ buys independence by collapsing some axes to the prior. Capacity control penalizes $|D_{\text{KL}} - C|$, so under-use is punished too.
- **The Z-Diff classifier must be linear** (a nonlinear one decodes entangled codes). **MIG uses a gap** because a factor spread over two slots has high MI but is entangled.
- **Unsupervised DRL needs a strong inductive bias** (factorizations are not identifiable), hence the semi-supervised β-VAE.
- **Pretext tasks beat reconstruction for DG:** they inject artificial shifts, while reconstruction rewards pixel fidelity (texture, lighting), which is exactly the non-causal content to discard.
- **$K_\gamma$ is scaffolding:** only $h_\theta$ transfers. Fine-tune when $N$ is moderate, and use a linear readout when $\mathcal{D}_L$ is very sparse.

**If you remember nothing else**

- DG bound = source risk + target-to-hull distance $\gamma$ + hull radius $\rho$; methods attack $L_i$ and $\rho$ because $\gamma$ is out of reach.
- Alignment makes domains look alike, IRM/SAM/DRO make the solution stable under shift, and DRL puts causal and spurious factors in separate slots.
- SSL manufactures supervision from unlabelled data; the pretext head is thrown away.

**Recall checklist**

- [ ] Write the DG bound, name every term, and say which lever cannot be pulled directly.
- [ ] Derive the pull/push identity and explain why proxies fix hard positives; write $L_{\text{PCL}}$.
- [ ] Write vanilla IRM, say why it is an MPEC, and write IRMv1 and the scale-probe objective.
- [ ] Derive $\bar\epsilon = \rho g_\theta/\lVert g_\theta\rVert$ and the $\frac{\rho^2}{2}\lambda_{\max}$ sharpness at a minimum.
- [ ] Name SAM's DG problem and its three remedies; explain the soft-label chain $\tilde y \to \nabla_\theta L \to \epsilon_i \to S_i$.
- [ ] State both G-DRO failures and CCFP's three losses.
- [ ] Compare dimension-wise and vector-wise DRL; distinguish $v$ from $w$ factors.
- [ ] Write $L_{\beta\text{-VAE}}$ and the capacity-controlled loss, and walk through β-VAE training with dimensions.
- [ ] Describe visual inspection, Z-Diff and MIG.
- [ ] Sketch context prediction, jigsaw, counting, rotation, colorization, instance discrimination and clustering, with each one's loss.

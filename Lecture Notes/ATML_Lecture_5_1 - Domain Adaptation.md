---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module II -DA.pdf]]'
tags:
- course/cs6304
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_4 - GANs, and Inductive Biases]] · [[ATML_Lecture_5_2 - Domain Generalization]] →
Concepts: [[Domain Adaptation]]
%% nav:end %%

# ATML — Lecture 5.1: Domain Adaptation (Module II)

## Where This Lecture Sits

*Opens Module II. [[ATML_Lecture_4 - GANs, and Inductive Biases]] closed on a model that prefers whichever cue is easiest to extract and looks perfectly healthy on an in-distribution test set; this lecture drops the IID (independent and identically distributed) assumption under which that test set was meaningful.*

Module I rested on Lecture 1 Section 5.1's premise $P_{\text{Test}}(X,Y) = P_{\text{Train}}(X,Y)$. This lecture takes the easier setting without it — **domain adaptation**, where unlabelled target data is available — and leaves the harder case to [[ATML_Lecture_5_2 - Domain Generalization]]. The remedies of Section 6 are organized by which [[ATML_Lecture_2 - Information Theory and Divergences]] instrument measures the domain gap: MMD (maximum mean discrepancy), an adversarial witness, or InfoNCE (the contrastive loss built on noise-contrastive estimation).

## Topic Inventory

- 1. Domains, Tasks, and Types of Learning Problems
  - 1.1 Domain and Task — Formal Definitions
  - 1.2 Three Types of Learning Problems
  - 1.3 Transfer Learning — DA vs. DG
- 2. Domain Shift
  - 2.1 The OOD Scenario, Near OOD and Far OOD
  - 2.2 Covariate Shift
  - 2.3 Semantic Shift / Concept Shift
- 3. Why Models Are Prone to Domain Shift — Empirical Risk Minimization
  - 3.1 Hypothesis Space and Expected (True) Risk
  - 3.2 Approximation Error
  - 3.3 Empirical Risk and the ERM Rule
  - 3.4 Estimation Error and Excess Risk
  - 3.5 When Does the Estimation Error Vanish?
  - 3.6 Why $EE \to 0$ Even at Low Capacity — the Margin View of Cross-Entropy
- 4. Spurious Correlations and the Feature Taxonomy
  - 4.1 Spurious Correlations
  - 4.2 The Cows-and-Camels Example
  - 4.3 The Causal Graph of Latent Factors
  - 4.4 The Feature Taxonomy — Causality × Domain Dependence
- 5. Domain Adaptation
  - 5.1 Definition and Goal
  - 5.2 The Four DA Settings
  - 5.3 The DA Upper-Bound Theorem
- 6. Mainstream CSDA Methods
  - 6.1 Measure-Based — Deep Adaptation Network
  - 6.2 Adversarial-Based — Domain Discriminator and the Gradient Reversal Layer
  - 6.3 Contrastive-Based — Prototypes, Pseudo-Labels, and Cross-Domain Contrastive Loss
  - 6.4 Pseudo-Label Generation
  - 6.5 Multimodality-Based — Prompt Learning in CLIP Style
  - 6.6 Insights on Prompt Learning
- 7. Open Problems and Challenges in DA
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Domains, Tasks, and Types of Learning Problems

Module I could speak of "the" data distribution because there was only ever one. Once training and test data may be drawn from different distributions, that singular usage becomes ambiguous and the vocabulary has to be rebuilt — which is what this section does. The definitions look like bookkeeping, but they earn their keep immediately in Section 2, where the several ways two domains can differ turn out to correspond to different components of the triple defined below, and each admits a different remedy.

### 1.1 Domain and Task — Formal Definitions

The module opens by re-stating the standard pipeline from Module I,

$$
x \;\longrightarrow\; g_\theta(\cdot) \;\xrightarrow{\;z\;}\; \boxed{\text{Task Head}} \;\longrightarrow\; \hat{y},
$$

and then asks a question that Module I never needed: *what exactly is a "domain," and what exactly is a "task"?*

> **Domain:** the **context of the knowledge-field where the problem is defined.**

Mathematically, a domain is a triple:

$$
\boxed{\mathcal{D}_i \triangleq \Big\{\underbrace{P_i(x)}_{\text{input data distribution}},\; \underbrace{P_i(y\mid x)}_{\text{labels distribution}},\; \underbrace{\mathcal{L}}_{\text{set of loss functions defined}}\Big\}}
$$

**Within a given domain $\mathcal{D}_i$, a given problem is called a task $T_j$.** Within one domain a number of tasks can be defined, and what distinguishes them is the loss function — not the data:
$$
\boxed{T_j \triangleq \big\{P_i(x),\; P_i(y\mid x),\; \mathcal{L}_j\big\}, \qquad j = 1, 2, \ldots, T}
$$

where $T$ is the **total number of tasks**. Note carefully what is *shared* and what *varies*: the two distributions $P_i(x)$ and $P_i(y\mid x)$ carry the subscript $i$ of the *domain* and stay fixed across tasks; only $\mathcal{L}_j$ carries the task index. A task is a choice of objective over a fixed data-generating process.

### 1.2 Three Types of Learning Problems

| | Setting | Example |
|---|---|---|
| **(A)** | **Single-domain, single-task** | MNIST (a dataset of handwritten digits) $\to$ image classification |
| **(B)** | **Multiple-domains, single-task** | PACS (a domain-generalization benchmark with Photo, Art painting, Cartoon and Sketch domains) $\to$ image classification |
| **(C)** | **Single-domain, multiple-tasks** | One encoder, many heads — **multi-task learning** |

Case (C) is drawn as one image space feeding a single encoder whose latent branches into several heads:

$$
x \;\longrightarrow\; g_\theta(\cdot) \;\longrightarrow\;
\begin{cases}
T_1: & \text{image classification}\\
T_2: & \text{object detection}\\
\;\;\vdots & \\
T_T: & \text{segmentation}
\end{cases}
$$

This is exactly the structure the formal definition predicts: one $\{P(x), P(y\mid x)\}$ pair, $T$ different loss functions, hence $T$ different tasks sharing one backbone.

Case (B) — several domains, one task — is the setting the rest of Module II is about.

### 1.3 Transfer Learning — DA vs. DG

> **Transfer learning:** use the knowledge gained from $\mathcal{D}_S$ (**source domains**) to perform well on $\mathcal{D}_T$ (**target domain**).

There are **two main paradigms**, and the difference between them is entirely about *what access you have to the target domain at training time*:

**(1) Domain Adaptation (DA).** **Adapt a model which is trained on $\mathcal{D}_S$, plus unlabelled $\mathcal{D}_T$, to do well on $\mathcal{D}_T$.** So $\mathcal{D}_S$ is **labelled** and $\mathcal{D}_T$ is **unlabelled**, and both are available during training. The objective: **train a model to predict labels for $\mathcal{D}_T$.**

**(2) Domain Generalization (DG).** **No access to $\mathcal{D}_T$ at all.** Train a model on $\mathcal{D}_S$ alone and hope it does well on $\mathcal{D}_T$.

This lecture (5.1) covers DA; DG is Module II's second half.

---

## 2. Domain Shift

With a domain defined as the triple $\{P_i(x), P_i(y\mid x), \mathcal{L}\}$, the possible ways two domains may differ are no longer open-ended: either the input distribution moves, or the conditional does, or both. Section 2.2 treats the first case and Section 2.3 the second, and the distinction is not academic — the whole of Section 5 onward is built on the assumption that only $P(x)$ has moved, which is what makes adaptation tractable at all.

### 2.1 The OOD Scenario, Near OOD and Far OOD

Given a set of **source domains** $\{\mathcal{D}_{S_1}, \mathcal{D}_{S_2}, \ldots\}$ and a **target domain** $\mathcal{D}_T$, the phenomenon that separates them is **domain shift**:

$$
\boxed{P_S(X,Y) \;\neq\; P_T(X,Y)}
$$

This is the **Out-of-Distribution (OOD) scenario** — the same OOD regime introduced in Lecture 1, now examined in detail. The picture is two multimodal densities: $P_S(X,Y)$ over the **train samples**, $P_T(X,Y)$ over the **test samples**, with the mismatch between them being the shift.

![[03 - Attachments/Screenshot 2026-09-19 at 11.33.26 PM.png|564]]

The OOD scenario subdivides by *how badly* the two differ:

- **Near OOD:** **only semantic shift.**
- **Far OOD:** **semantic + covariate shift.**

The two constituent shift types are defined next.

### 2.2 Covariate Shift

Recall the factorization
$$
\boxed{P(X,Y) = P(X)\,P(Y\mid X)}
$$

**Covariate shift** is the case where $P_T(X,Y) \neq P_S(X,Y)$ **due entirely to the first factor**:

$$
\boxed{\underbrace{P_T(X) \neq P_S(X)}_{\text{input data distribution is different}} \qquad \text{but} \qquad \underbrace{P_T(Y\mid X) = P_S(Y\mid X)}_{\text{label space is the same}}}
$$

The canonical example: a **real-world CAT** image drawn from $P_S(X)$ versus a **cartoon CAT** image drawn from $P_T(X)$. The pixels change completely; the meaning of the label "cat" does not.

### 2.3 Semantic Shift / Concept Shift

**Semantic shift** (also **concept shift**) is the case where the second factor changes:

$$
\boxed{P_T(Y\mid X) \neq P_S(Y\mid X)}
$$

**The relationship between $x_i$ and $y_i$ changes in a semantic way.** This **can happen due to** either of two causes:

- **Unseen class labels arrive in $Y$** — the label set itself grows or changes.
- **The semantic relationship between visual features and labels changes** — the same visual evidence now implies a different label.

Two examples:

- **Environmental change** — speech classification task $\tau$: $P_S$ = **quiet** recordings, $P_T$ = **noisy** recordings.
- **Conditions change** — vehicle identification task $\tau$: $P_S$ = **daytime images**, $P_T$ = **night-time images**.

---

## 3. Why Models Are Prone to Domain Shift — Empirical Risk Minimization

Section 2 described what domain shift *is*; this section explains why standard training is defenceless against it. The explanation is sharper than a general appeal to brittleness: it identifies the exact premise that fails and the exact guarantee that lapses with it. Read this section as the formal counterpart of Lecture 1 Section 5.1, which stated the IID bound without examining what happens when its assumption is withdrawn.

**DRL (deep representation learning) models are traditionally trained using a learning rule called Empirical Risk Minimization (ERM).** The vulnerability to domain shift is a direct consequence of that rule, and the argument is built up through four quantities: true risk, approximation error, empirical risk, and estimation error.

### 3.1 Hypothesis Space and Expected (True) Risk

![[03 - Attachments/Screenshot 2026-09-19 at 11.41.00 PM.png]]
Let $\mathcal{H}$ be the **hypothesis space** — **we can choose any model $f_\theta(\cdot)$ within this space.** The training data is $\{x_i, y_i\}_{i=1}^{N} \sim P(X,Y)$, fed through the usual encoder-plus-head architecture to produce $\hat{Y}$.

**For any choice $f_\theta(\cdot)$, we aim to reduce the expected risk.** The **expected risk associated with the choice $f_\theta(\cdot)$** is

$$
\boxed{R_{\text{True}}(f_\theta) = \mathbb{E}_{P(X,Y)}\Big[\underbrace{L\big(f_\theta(X), Y\big)}_{\text{loss measuring how well } f_\theta \text{ approximates the } X\to Y \text{ relationship}}\Big]}
$$

The best model *within the hypothesis space* is then

$$
\boxed{f_\theta^{*} = \operatorname*{arg\,min}_\theta R_{\text{True}}(f_\theta)}
$$

i.e. **the model which gives minimum $R_{\text{True}}$ among all $f_\theta \in \mathcal{H}$.**

Three distinct models must be kept apart, and the plot of $R_{\text{True}}$ against "all possible hypotheses" shows all three:
![[03 - Attachments/Screenshot 2026-09-19 at 11.53.26 PM.png]]

| Symbol | Meaning | Risk |
|---|---|---|
| $f_\theta^{\text{best}}(\cdot)$ | the **best model overall** — over *all* possible hypotheses, not just those in $\mathcal{H}$ | $R_{\min}$ |
| $f_\theta^{*}(\cdot)$ | the best model **inside $\mathcal{H}$** | $R_{\min}^{*}$ |
| $\hat{f}_\theta(\cdot)$ | the model **ERM actually returns** (Section 3.3) | — |

$$
R_{\min} = R_{\text{True}}\big(f_\theta^{\text{best}}\big) = \min_{\text{all } f_\theta} R_{\text{True}}(f_\theta), \qquad\qquad
R_{\min}^{*} = R_{\text{True}}\big(f_\theta^{*}\big) = \min_{f_\theta \in \mathcal{H}} R_{\text{True}}(f_\theta).
$$

### 3.2 Approximation Error

The price paid for restricting attention to $\mathcal{H}$ at all:

$$
\boxed{\text{Approximation Error} = AE = R_{\min}^{*} - R_{\min}}
$$

This is **due to model misspecification** — your architecture family simply may not contain the truly best function. Note $AE \ge 0$ always, and it is fixed the moment you choose an architecture.

### 3.3 Empirical Risk and the ERM Rule

**Since we do not have $P(X,Y)$, we cannot find $f_\theta^{*}(\cdot)$ either.** We therefore fall back on the **empirical risk** — the Monte Carlo substitution of Lecture 2 applied to the risk:

$$
\boxed{R_{\text{emp}}(f_\theta) = \frac{1}{N}\sum_{i=1}^{N} L\big(f_\theta(x_i), y_i\big) = \text{average loss on data}}
$$

and minimize *that* instead. This is the **ERM rule**:

$$
\boxed{\hat{f}_\theta(\cdot) = \operatorname*{arg\,min}_\theta R_{\text{emp}}(f_\theta)}
$$

### 3.4 Estimation Error and Excess Risk

The price paid for minimizing the empirical rather than the true risk:

$$
\boxed{\text{Estimation Error} = EE = R_{\text{emp}}\big(\hat{f}_\theta\big) - R_{\text{True}}\big(f_\theta^{*}\big)}
$$

Adding the two error sources gives the total gap between what you get and the best achievable:

$$
\boxed{\text{Excess Risk} = R_{\text{excess}}\big(\hat{f}_\theta\big) = AE + EE = \underbrace{\Big[R_{\text{True}}(f_\theta^{*}) - R_{\text{True}}(f_\theta^{\text{best}})\Big]}_{\text{due to model misspecification}} + \underbrace{\Big[R_{\text{emp}}(\hat{f}_\theta) - R_{\text{True}}(f_\theta^{*})\Big]}_{\text{how well ERM approximates } f_\theta^{*}}}
$$

And the practical consequence, boxed in the notes:

> **$EE$ is the only thing that we can reduce, given that $\mathcal{H}$ has been fixed.**

Once the architecture is chosen, $AE$ is locked in; all of training is an attempt to shrink $EE$.

### 3.5 When Does the Estimation Error Vanish?

**It can be shown that $EE \to 0$ as $N \to \infty$ if and only if:**

1. **New data points are IID.**
2. **The capacity of $\hat{f}_\theta(\cdot)$ is NOT too large; otherwise we overfit.**

with capacity being exactly the quantity defined in Lecture 1, now written in terms of the parameter matrices $\theta_\ell$:

$$
\boxed{\text{Capacity of } \hat{f}_\theta(\cdot) = \beta\sqrt{S}, \qquad \beta = \prod_{\ell=1}^{L}\lVert \theta_\ell\rVert_2, \qquad S = \sum_{\ell=1}^{L}\frac{\lVert\theta_\ell\rVert_F^2}{\lVert\theta_\ell\rVert_2^2}}
$$

**Condition 1 is the entire problem.** ERM's guarantee is conditional on the IID assumption, and domain shift is by definition a violation of it. A model can be trained perfectly, with $EE$ driven to zero on the source distribution, and still fail arbitrarily badly on a shifted target — not because training went wrong, but because the guarantee never covered that case. **This is why models are prone to domain shift.**

*Intuition.* The decomposition is worth holding as a chain of prices: $AE$ is the price of choosing an architecture, fixed once chosen; $R_{\text{emp}}$ is what training actually minimizes; $EE$ is the price of using a finite sample in place of a distribution, and the only term training can reduce. Domain shift does not increase any of these terms — it invalidates the theorem connecting them. The subtlety, and the reason the failure evades ordinary practice, is that every quantity a practitioner can measure is computed on source-distributed data, so a model that is failing on the target looks indistinguishable from one that is not. Nothing in the training loop is wrong, which is precisely why nothing in the training loop can detect it.

### 3.6 Why $EE \to 0$ Even at Low Capacity — the Margin View of Cross-Entropy

A follow-up question: *why does $EE \to 0$ even if the model capacity is low?* The answer comes from re-reading the cross-entropy loss in the ratio form derived in Lecture 1. For an example $\{x_i, y_i\}$ with true label $j$:

$$
\mathcal{L}_{\text{CE}_i} = \log\left(1 + \sum_{k \neq j}\frac{e^{z_i^{\top}\psi_k}}{e^{z_i^{\top}\psi_j}}\right) = \log\left(1 + \sum_{k\neq j}\exp\big[\psi_k^{\top}z_i - \psi_j^{\top}z_i\big]\right)
$$

Factoring out the negative sign exposes a **margin** inside the exponential:

$$
\boxed{\mathcal{L}_{\text{CE}_i} = \log\left(1 + \sum_{k\neq j}\exp\Big[-\underbrace{\big(\psi_j^{\top}z_i - \psi_k^{\top}z_i\big)}_{m_{jk}(z_i)\;=\;\text{margin of class } j \text{ from all other } k\neq j \text{ classes}}\Big]\right)}
$$

Therefore

$$
\mathcal{L}_{\text{CE}_i} = 0 \quad \text{when} \quad m_{jk}(z_i) \to \infty \;\Longrightarrow\; \psi_j^{\top}z_i \text{ is very high and/or } \psi_k^{\top}z_i \text{ is very low.}
$$

$$
\boxed{\Rightarrow \text{ We maximize the correlation between } z_i \text{ and } \psi_j.}
$$

The mechanism, drawn as a pipeline: $x_i \to g_\theta(\cdot) \to z_i \in \mathbb{R}^{d\times1} \to \psi_j^\top z_i \to s_j \to \text{Softmax}$, where $s_j$ is the **true class logit**.

So cross-entropy does not need enormous capacity to reach near-zero loss — it only needs to find *any* direction $z_i$ strongly correlated with $\psi_j$. And that is precisely where the trouble starts: **nothing in the objective says which direction.** Section 4 is about what the network picks.

---

## 4. Spurious Correlations and the Feature Taxonomy

### 4.1 Spurious Correlations

> **Spurious correlations:** increased interaction / dependency between **domain-dependent latent dimensions** and **task labels.**

Formally, the latent $z$ contains some **domain-dependent dimensions** $z_{dd}$ — carrying information like **background, shadows, small objects, etc.** — and the failure mode is

$$
\text{Corr}\big(z_{dd},\, \psi_j\big) = \text{very high.}
$$

Because Section 3.6 showed that cross-entropy is satisfied by *any* high-correlation direction, and domain-dependent features are often the easiest such direction to find:

> **Models often find shortcuts, hence they prioritize learning spurious correlations.**

### 4.2 The Cows-and-Camels Example

| | Cows | Camels |
|---|---|---|
| **Source domain $\mathcal{D}_S$** | Cows on grass | Camels in desert |
| **Target domain $\mathcal{D}_T$** | Cows in **desert** | Camels on **grass** |

The model is trained on $\mathcal{D}_S$, producing $f_\theta(\cdot)$. **If the model learned to associate**

$$
\text{Grass} \longrightarrow \text{cows}, \qquad\qquad \text{Desert} \longrightarrow \text{camels},
$$

— a spurious correlation, since background is not what makes an animal a cow — **then the performance decreases on $\mathcal{D}_T$**, where the backgrounds have been swapped. The shortcut that was perfectly predictive in-domain becomes actively *anti*-predictive out-of-domain.

### 4.3 The Causal Graph of Latent Factors

The relationships among the latent factors are drawn as a small causal graph:
![[03 - Attachments/Pasted image 20260922000636.png|544]]
$$
\text{Domain} \longrightarrow z_{DD}, \qquad \text{Domain} \longrightarrow z_{DI}
$$
$$
z_{DD} \longrightarrow z_{NC}, \qquad z_{DI} \longrightarrow z_{NC}, \qquad z_{DI} \longrightarrow z_{C}
$$
$$
z_{NC} \longrightarrow (X \leftrightarrow Y), \qquad z_{C} \longrightarrow (X \leftrightarrow Y)
$$

with the subscripts reading: $DD$ = domain-dependent, $DI$ = domain-invariant, $NC$ = non-causal, $C$ = causal. The structural point is that the **domain** node influences both the domain-dependent *and* the domain-invariant branches, but only the domain-invariant branch feeds a genuinely **causal** factor $z_C$. Everything reaching $X \leftrightarrow Y$ through $z_{NC}$ is correlational.

### 4.4 The Feature Taxonomy — Causality × Domain Dependence

Crossing the two axes — *is the feature causal for $Y$?* and *is it domain-invariant?* — gives four quadrants. This $2\times2$ <mark style="background: #ABF7F7A6;">table is the organizing framework for the whole module:</mark>

| | **Domain-Invariant Features** | **Domain-Dependent Features** |
|---|---|---|
| **Causal for $Y$** | **Label-informative and stable across domains.** These are the features we want. *Ex.:* object shape. | **Causal but entangled**, in the sense that the content signal is there, but $z = g_\theta(\cdot)$ shifts with domains (lighting, viewpoint, etc.). *Ex.:* an edge/shape feature whose contrast depends on the camera. |
| **Non-causal for $Y$** | **Benign / irrelevant**, in the sense that they are stable across domains but **not informative for the label**. *Ex.:* a border or watermark in images. | **Spurious:** non-causal **and** domain-dependent, as they correlate with the label **only because of domain bias**. *Ex.:* background, sky color. $\Rightarrow$ **IID accuracy $\uparrow$, OOD accuracy $\downarrow$.** |

Three of the four quadrants are worth reading carefully:

- The **top-left** quadrant is the target of all representation learning — this is what "learn domain-invariant features" means precisely.
- The **top-right** quadrant is the *recoverable* problem: the signal is genuinely there, it just arrives wearing a domain-specific costume. Alignment methods (Section 6) target this quadrant.
- The **bottom-right** quadrant is the dangerous one. Note the signature: **IID accuracy goes up while OOD accuracy goes down.** A model exploiting spurious features looks *better* on a standard validation split, which is exactly why the problem goes undetected.
- The **bottom-left** quadrant is harmless — a watermark on every image in every domain misleads no one, because it correlates with nothing.

---

## 5. Domain Adaptation

### 5.1 Definition and Goal

> **Domain Adaptation (DA):** improve the generalization of a model which has access to **labelled data from source domains** and **unlabelled data from the target domain.**

$$
\underbrace{\mathcal{D}_{S_1}\{x_i, y_i\},\; \mathcal{D}_{S_2}\{x_i,y_i\},\;\ldots}_{\text{source domains (labelled)}} \;\;\text{and}\;\; \underbrace{\mathcal{D}_T\{x_i\}}_{\text{target domain (unlabelled)}} \;\longrightarrow\; \boxed{\text{Train } f_\theta(\cdot)}
$$

**Goal: good performance in prediction of labels for $\mathcal{D}_T$.**

### 5.2 The Four DA Settings

**There are four different settings of DA categories**, distinguished by how the label/category spaces of source and target relate.

**(1) Closed-Set DA (CSDA).** The **label space / category space across $\mathcal{D}_{S_i}$ and $\mathcal{D}_T$ are consistent** — both contain the same $K$ classes $\{C_1, C_2, \ldots, C_K\}$. **We only have covariate shift:**

$$
P_{S_i}(X,Y) \neq P_T(X,Y) \quad \text{due to} \quad P_{S_i}(x) \neq P_T(x) \;\; \forall i.
$$

**(2) Partial-Set DA (PSDA).** The **category space of $\mathcal{D}_T$ is a subset of the category space of $\mathcal{D}_S$** (considering only one source domain for simplicity). If $\mathcal{D}_S$ has $\{C_1, C_2, C_3\}$ and $\mathcal{D}_T$ has $\{C_1, C_2\}$, then $C_3$ is a **private category** of the source. Here we have **both covariate and semantic shifts**:

$$
P_S(x) \neq P_T(x) \quad\text{and}\quad P_S(Y\mid x) \neq P_T(Y\mid x).
$$

**(3) Open-Set DA (OSDA).** The reverse containment — the **category space of $\mathcal{D}_S$ is a subset of $\mathcal{D}_T$**. If $\mathcal{D}_S$ has $\{C_1, C_2\}$ and $\mathcal{D}_T$ has $\{C_1, C_2, C_3\}$, then $C_3$ is a **private category** of the target. **Again** both shifts are present.

**(4) Universal DA (UDA).** **Private categories can exist in both $\mathcal{D}_S$ and $\mathcal{D}_T$** — e.g. $\mathcal{D}_S = \{C_1, C_2, C_3, C_4\}$ and $\mathcal{D}_T = \{C_2, C_3, C_5\}$, so $C_1, C_4$ are source-private and $C_5$ is target-private. Both shifts are present.

Summarizing the four:

| Setting | Category relation | Shifts present |
|---|---|---|
| **CSDA** | $\mathcal{C}_S = \mathcal{C}_T$ | Covariate only |
| **PSDA** | $\mathcal{C}_T \subset \mathcal{C}_S$ | Covariate + semantic |
| **OSDA** | $\mathcal{C}_S \subset \mathcal{C}_T$ | Covariate + semantic |
| **UDA** | Private categories on both sides | Covariate + semantic |

### 5.3 The DA Upper-Bound Theorem

> **Theorem.** For a model $f_\theta(\cdot)$, the loss on $\mathcal{D}_T$, written $L_T(f_\theta)$, is upper-bounded as $$\boxed{L_T(f_\theta) \;\le\; L_S(f_\theta) \;+\; \text{Divergence}\big(P_S(x),\, P_T(x)\big) \;+\; \min(a, b)}$$ with $$a = \mathbb{E}_{P_S}\Big[\big|P_S(Y\mid x) - P_T(Y\mid x)\big|\Big] \quad \text{(average on } P_S),$$$$b = \mathbb{E}_{P_T}\Big[\big|P_S(Y\mid x) - P_T(Y\mid x)\big|\Big] \quad \text{(average on } P_T).$$

A note attached in the margin, worth taking seriously:

> **Note: this is a control statement, not a comparison of $L_S$ and $L_T$.**

That is, the theorem is not saying "target loss exceeds source loss." It is saying that target loss is *controlled by* three quantities you can act on — which is what makes it an engineering recipe rather than a mere observation. The accompanying sketch shows the bound sitting above the true $L_T(f_\theta)$ curve with the annotation: **reduce the bound, and $L_T$ will be small.**

Reading the three terms:

- **$L_S(f_\theta)$** — the source loss. You minimize this by ordinary supervised training on the labelled source data.
- **$\text{Divergence}(\cdot)$** — **a divergence measure between $P_S(x,Y)$ and $P_T(x,Y)$ to indicate how different these distributions are.** This is the whole toolkit of Lecture 2 (KL (Kullback–Leibler), JS (Jensen–Shannon), MMD, TVD (total variation distance), Wasserstein), and reducing it is *marginal alignment*.
- **$\min(a,b)$** — the **conditional distribution alignment term**: the **smallest possible difference that is achievable by aligning $\mathcal{D}_T$ and $\mathcal{D}_S$.** It answers *how well do label distributions align, given $x$?* and satisfies
$$
\min(a,b) \to 0 \quad \text{as} \quad P_S(Y\mid x) = P_T(Y\mid x).
$$

The three terms map exactly onto the two shift types of Section 2: the divergence term addresses covariate shift, and $\min(a,b)$ addresses semantic shift. In CSDA, where by definition $P_S(Y\mid x) = P_T(Y\mid x)$, the third term vanishes and only the first two matter — which is why CSDA is tractable.

*Intuition.* The structure is the IID bound of Lecture 1 Section 5.2 with one term added. Where that bound read *training loss plus a generalization penalty*, this one reads *source loss plus a penalty for the domains differing*, and the new middle term is precisely the price of the assumption Lecture 1 was permitted to make for free. Setting $P_S = P_T$ collapses it and recovers the familiar case.

The bound also explains the shape of every method in Section 6. Two of the three terms are actionable — minimize source loss by ordinary supervised training, minimize divergence by making the *features* of the two domains indistinguishable — and since the divergence is measured on $z$ rather than on $x$, the encoder can be pushed to discard whatever distinguishes the domains. That is the common mechanism of the entire section; the families differ only in which instrument from Lecture 2 measures the divergence.

One caution, which [[ATML_Lecture_5_2 - Domain Generalization]] develops: minimizing the marginal divergence is not sufficient. The bound constrains $L_T$ from above, so a small right-hand side guarantees a small target loss, but aligning the marginals $P_S(x)$ and $P_T(x)$ can be achieved by a mapping that superimposes source class 1 onto target class 2 — the marginals match while the conditionals are destroyed. The third term is what forbids this, and it is assumed zero in CSDA rather than enforced.

> **In this course, we focus on CSDA.** The remaining sections cover the few mainstreams for CSDA models.

---

## 6. Mainstream CSDA Methods

Section 5.3 reduced domain adaptation to a single actionable instruction: minimize the divergence between source and target feature distributions while retaining source accuracy. What remains is the choice of instrument, and the four families below are four answers, each importing a measurement from an earlier lecture. Section 6.1 uses MMD from Lecture 2 Section 6.2; Section 6.2 uses a learned adversarial witness, the mechanism of the GAN (generative adversarial network) discriminator in Lecture 4 Section 1; Section 6.3 uses the contrastive loss of Lecture 3 Section 3; Section 6.5 uses the multimodal alignment of CLIP (Contrastive Language–Image Pre-training). The progression is also one of increasing class-awareness, since each family addresses a limitation the previous one exposed.

### 6.1 Measure-Based — Deep Adaptation Network (DAN)

> **Measure-based methods: minimize $\text{Divergence}(P_S, P_T)$ by adding some adaptive metric to the loss function, to promote learning of invariant features across $\mathcal{D}_S$ and $\mathcal{D}_T$.**

This attacks the second term of the bound directly and explicitly.

**Architecture.** The network is split into three stages:

![[03 - Attachments/Screenshot 2026-09-25 at 11.08.36 AM.png]]
$$
x \;\longrightarrow\; \underbrace{\text{shared layers}}_{\text{one copy, both domains}} \;\longrightarrow\; \underbrace{\text{alignment layers}}_{\text{indices } \ell_1 \ldots \ell_2,\; \text{S and T branches}} \;\longrightarrow\; \underbrace{\text{task-specific (fully-connected) layers}}_{} \;\longrightarrow\; \hat{y}
$$

In the alignment region, each layer exists as a **source layer** and a **target layer** connected by an alignment penalty $\mathcal{L}_{\text{align}}$ — the source branch $\{S\}$ and target branch $\{T\}$ run in parallel from index $\ell_1$ through index $\ell_2$.

**Deep Adaptation Network (DAN).** The objective:

$$
\boxed{\mathcal{L}_{\text{Total}} = \underbrace{\mathcal{L}_{\text{CE}}}_{\text{on } \mathcal{D}_S} \;+\; \lambda\sum_{\ell = \ell_1}^{\ell_2} d^2\big(\mathcal{D}_S^{\ell},\, \mathcal{D}_T^{\ell}\big)}
$$

where $d^2(\mathcal{D}_S^\ell, \mathcal{D}_T^\ell)$ is the **distance between the feature representations of the source and target layers at index $\ell$**, and

$$
d(\cdot,\cdot) = \text{distance measure between distributions, e.g. MMD, CORAL, WD, etc.}
$$

Note that the cross-entropy term uses **only source data** — the target contributes nothing but its unlabelled features to the alignment penalty, which is exactly what "unlabelled target" means operationally. Note also that alignment is applied at *several* layers ($\ell_1$ through $\ell_2$) rather than only at the final representation: shift accumulates through depth, so it is corrected through depth.

### 6.2 Adversarial-Based — Domain Discriminator and the Gradient Reversal Layer (DANN)

> **Adversarial-based methods: implicitly learn the divergence $(P_S, P_T)$ through a domain discriminator.**

Where measure-based methods *specify* a divergence (MMD, CORAL (correlation alignment), …), adversarial methods *learn* one — the same move GANs make, and with the same machinery.

**Architecture.** The feature extractor's latent feeds two heads:
![[03 - Attachments/Screenshot 2026-09-25 at 11.12.52 AM.png|590]]
$$
x \to \boxed{\text{Feature Extractor } g_{\theta_f}(\cdot)} \xrightarrow{\;z\;}
\begin{cases}
\boxed{\text{Label Prediction Head } h_{\theta_y}(z)} \to \text{class label prediction}\\[1ex]
\boxed{\text{Domain Discriminator } D_{\theta_d}(z)} \to \text{domain label } \langle S, T\rangle = ?
\end{cases}
$$

**Three parameter groups, three objectives.**

**$\theta_f$ and $\theta_y$** — find the values that **minimize the DANN (domain-adversarial neural network) supervised loss**:

$$
\boxed{\mathcal{L}_{\text{sup}}(\theta_f, \theta_y) = \mathcal{L}_{\text{CE}}\Big(h_{\theta_y}\big(g_{\theta_f}(x_i)\big),\, y_i\Big), \qquad i = 1, 2, \ldots, N_S}
$$

where $N_S$ is the **number of source-domain samples** (only source data has labels).

**$\theta_d$** — the domain discriminator's target output is

$$
D_{\theta_d}\big(g_{\theta_f}(x)\big) = \begin{cases} 1 & \text{if } x \in \mathcal{D}_S \\ 0 & \text{if } x \in \mathcal{D}_T\end{cases}
$$

**The domain discriminator $D_{\theta_d}(\cdot)$ should be able to accurately classify $x \in \mathcal{D}_S$ and $x\in\mathcal{D}_T$ through BCE (binary cross-entropy) loss minimization:**

$$
\boxed{\mathcal{L}_{\text{domain}}(\theta_f, \theta_d) = \mathbb{E}_{P_S(x)}\Big[\log D_{\theta_d}\big(g_{\theta_f}(x)\big)\Big] + \mathbb{E}_{P_T(x)}\Big[1 - \log D_{\theta_d}\big(g_{\theta_f}(x)\big)\Big]}
$$

> **[Possible source inconsistency]** — the second term is written as $\mathbb{E}_{P_T}[1 - \log D(\cdot)]$. The standard binary cross-entropy form (and the one consistent with the GAN value function of Lecture 4) is $\mathbb{E}_{P_T}[\log(1 - D(\cdot))]$ — the $1-$ belongs inside the logarithm. The professor's form is reproduced as written; use $\log(1 - D(\cdot))$ in implementation.

**The adversarial turn.** **If we allow the domain discriminator to reduce $\mathcal{L}_{\text{domain}}$, then the latent $z$ generated for source-domain $x_S$ and target-domain $x_T$ will be totally different. We do not want this** — we want

$$
\boxed{g_{\theta_f}(x_S) \approx g_{\theta_f}(x_T)} \;\Longrightarrow\; \textbf{domain invariance.}
$$

**In order to do this, the feature extractor must fool the discriminator as much as it can** $\Rightarrow$ **$g_{\theta_f}(\cdot)$ should *maximize* $\mathcal{L}_{\text{domain}}(\cdot)$.**

Hence the combined objective for $\theta_f$ carries a negative sign on the domain term:

$$
\boxed{\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{sup}}(\theta_f, \theta_y) - \lambda\,\mathcal{L}_{\text{domain}}(\theta_f, \theta_d)}
$$

and the three update rules are:

$$
\tag{1} \theta_f \leftarrow \theta_f - \alpha\Big[\nabla_{\theta_f}\mathcal{L}_{\text{sup}} - \lambda\,\nabla_{\theta_f}\mathcal{L}_{\text{domain}}\Big]
$$
$$
\tag{2} \theta_y \leftarrow \theta_y - \alpha\,\nabla_{\theta_y}\mathcal{L}_{\text{sup}}
$$
$$
\tag{3} \theta_d \leftarrow \theta_d - \alpha\,\nabla_{\theta_d}\mathcal{L}_{\text{domain}}
$$

**Implementing (1) — the Gradient Reversal Layer.** The awkward part of update (1) is the minus sign: standard autograd will happily compute $\nabla_{\theta_f}\mathcal{L}_{\text{domain}}$, but we need its *negative*. The trick is a **Gradient Reversal Layer (GRL)** inserted **between the feature-extractor output and the domain-discriminator input**:

$$
x \to \boxed{\text{Feature Extractor}} \xrightarrow{\; z = g_{\theta_f}(\cdot)\;} \boxed{\text{GRL}} \to \boxed{\text{Domain Discriminator}} \to u = D_{\theta_d}(z) \to \mathcal{L}_{\text{domain}}
$$

**Forward pass:** $\;\text{GRL} = \text{Identity}$ — it does nothing at all.

**Backward pass:** the chain rule would ordinarily give

$$
\nabla_{\theta_f}\mathcal{L}_{\text{domain}} = \frac{\partial}{\partial u}\mathcal{L}_{\text{domain}} \cdot \frac{\partial u}{\partial z}\cdot\frac{\partial z}{\partial \theta_f},
$$

but **while passing through the GRL it becomes**

$$
\boxed{\nabla_{\theta_f}\mathcal{L}_{\text{domain}} = -\lambda\left(\frac{\partial z}{\partial\theta_f}\right)\left(\frac{\partial u}{\partial z}\right)\left(\frac{\partial}{\partial u}\mathcal{L}_{\text{domain}}\right)}
$$

So the GRL is a layer that is invisible going forward and multiplies by $-\lambda$ coming back. This is a genuinely elegant piece of engineering: the entire min-max structure is implemented inside a single autograd node, so the whole system trains with one ordinary gradient-descent loop rather than the alternating two-phase procedure GANs require (Lecture 4, Section 1.6).

### 6.3 Contrastive-Based — Prototypes, Pseudo-Labels, and Cross-Domain Contrastive Loss

> **Contrastive-based methods: explicitly pull together features of the same classes across domains, and push apart different classes, using the contrastive InfoNCE loss. For $\mathcal{D}_T$, use periodically refreshed pseudo-labels.**

Note what this fixes relative to Sections 6.1–6.2: both of those align the *marginals* $P_S(x)$ and $P_T(x)$ without any regard to class identity, so they can happily align a source "cow" with a target "camel." Contrastive methods align *class-conditionally*.

The procedure has four steps.

**Step ① — Train on the source.** Get $\{\theta, \psi\}$ using $\mathcal{D}_S$ samples with the ordinary supervised loss:

$$
\mathcal{L}_{\text{sup}}(\theta, \psi) = \frac{1}{N_S}\sum_i \text{CE}\big(\hat{y}_i,\, y_i\big),
$$

over the pipeline $\mathcal{D}_S\{x_i,y_i\} \to g_\theta(\cdot) \to z \to h_\psi(\cdot)$.

**Step ② — Get source-class prototypes in the latent space.** For each class $c \in \{1, 2, \ldots, K\}$, average the $\ell_2$-normalized source features of that class:

$$
\boxed{\mu_c = \frac{1}{N_c}\sum_{i\,:\,y_i = c} \frac{f_\theta(x_i)}{\lVert f_\theta(x_i)\rVert_2} \qquad \text{— class prototype for class } c}
$$

Drawn as $K$ latent clusters $C_1, \ldots, C_K$ with their centroids $\mu_1, \ldots, \mu_K$. The normalization before averaging means these prototypes live on the unit sphere, so "similarity" downstream is angular.

**Step ③ — Pseudo-label the target with K-means.** **Use K-means to get pseudo-labels for $\mathcal{D}_T$ samples, after K-means has converged.** The **K-means centroids are initialized as the $\mu_c$** from Step ②, **with total number of clusters $= K =$ number of classes in $\mathcal{D}_S$.** Initializing from source prototypes is what transfers the label semantics: a target cluster inherits the class of the source prototype it formed around.

**Step ④ — Cross-domain contrastive loss.** The pairing rule:

- **Take an anchor from $\mathcal{D}_S$**, where anchor $a \in C_i$.
- **Take positive samples (having the same class) from $\mathcal{D}_T$.**
- **Everything else is negative samples.**

Crucially the anchor and its positives come from *different domains* — that is the whole point, and it is what makes the loss produce domain-invariant, class-discriminative features rather than merely class-discriminative ones.

The source-to-target direction:

$$
\boxed{\mathcal{L}_{S2T}(a) = -\log \frac{\sum_p \exp\big(\text{sim}(z_a, z_p)\big)}{\sum_k \exp\big(\text{sim}(z_a, z_k)\big)}}
$$

with the conditions: **$a \in \mathcal{D}_S$, $p \in \mathcal{D}_T$, and their class labels are the same**; and **$k$ does not contain samples from the anchor class.**

The target-to-source direction is symmetric, with $a \in \mathcal{D}_T$ and $p \in \mathcal{D}_S$:

$$
\boxed{\mathcal{L}_{T2S}(a) = -\log\frac{\sum_p \exp\big(\text{sim}(z_a, z_p)\big)}{\sum_k \exp\big(\text{sim}(z_a, z_k)\big)}}
$$

Averaging over both anchor sets $A_S$ and $A_T$ gives the **cross-domain contrastive (CDC) loss**:

$$
\boxed{\mathcal{L}_{\text{CDC}} = \frac{1}{|A_S|}\sum_{a \in A_S}\mathcal{L}_{S2T}(a) \;+\; \frac{1}{|A_T|}\sum_{a\in A_T}\mathcal{L}_{T2S}(a)}
$$

and the full objective is

$$
\boxed{\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CDC}} + \mathcal{L}_{\text{sup}}}
$$

The two-direction structure is the same symmetrization used by CLIP (Lecture 3, Section 3.5) — there it was image↔text, here it is source↔target.

### 6.4 Pseudo-Label Generation

**Pseudo-label generation assigns labels to unlabelled $\mathcal{D}_T$ samples. This can be done in many ways.**

**Method ① — Direct transfer.** Train $f_\theta(\cdot)$ on source data, then run target data through the same $f_\theta(\cdot)$ and take its predictions as pseudo-labels. **This approach is equivalent to KNN (k-nearest neighbours) clustering as described above** (Section 6.3, Steps ②–③).

**Method ② — Multi-classifier consensus.** **Train 2 or 3 classifiers having different architectures and use consensus or divergence-minimization in their decisions to get pseudo-labels.** Architecture: one shared encoder $g_\theta(\cdot)$ feeding several heads $H_1, H_2, H_u$. The algorithm:

1. Train $g_\theta(\cdot)$, $H_1$, $H_2$, $H_u$ on $\mathcal{D}_S$.
2. Get labels for $\mathcal{D}_T$ using $g_\theta(\cdot)\circ H_u$, producing $\mathcal{D}_T^{PL}$ — **the target domain with pseudo-labels.**
3. Form new data: $\mathcal{D}_N = \mathcal{D}_S \cup \mathcal{D}_T^{PL}$.
4. **For $L$ iterations:**
   - Train $g_\theta(\cdot)$, $H_1$, $H_2$ using $\mathcal{D}_N$.
   - Train $g_\theta(\cdot)$ and $H_u$ using $\mathcal{D}_T^{UL}$ (the unlabelled target).
5. **end**

The asymmetry between $\{H_1, H_2\}$ and $H_u$ is deliberate: the first two are trained on the combined labelled-plus-pseudo-labelled set while $H_u$ is refreshed against the raw unlabelled target, so the heads do not collapse into agreeing with each other's mistakes.

### 6.5 Multimodality-Based — Prompt Learning in CLIP Style

> **Multimodality-based methods: learn domain-specific text prompts, to be used as a dynamic classifier in CLIP style.**

**Prompt structure.** For prompt learning, the prompt is composed of three parts:

$$
\boxed{[\,\text{Domain-agnostic text}\,]\;[\,\text{Domain-specific text}\,]\;[\,\text{Class Label}\,]}
$$

Concretely: *"An image of"* $\;\;|\;\;$ *"Painting"* $\;\;|\;\;$ *"Dog"*. In learnable-token form:

$$
t_K^{d} = \underbrace{[v]_1, [v]_2, \ldots, [v]_{M_1}}_{\text{domain-agnostic context}}\;\;\underbrace{[d]_1, [d]_2, \ldots, [d]_{M_2}}_{\text{domain-specific}}\;\;\underbrace{[\text{class}]_K}_{\text{class label}}
$$

where $t_K^d$ is the **text prompt for class $K$**, $M_1$ = **number of context tokens**, and $M_2$ = **number of domain-specific tokens**. Only the tokens are learned; the encoders stay frozen.

Two prompts exist per class — $t_K^{S}$ for $\mathcal{D}_S$ and $t_K^{T}$ for $\mathcal{D}_T$ — differing in their domain-specific segment:

$$
[d]_i^{S} \neq [d]_j^{T} \qquad\Longrightarrow\qquad \boxed{\text{2 domains } \mathcal{D}_S, \mathcal{D}_T \text{ with } K \text{ categories have } 2K \text{ prompts to learn}}
$$

**Prompt tokens are learned over time** by gradient descent:

$$
\boxed{t_K^{d} \leftarrow t_K^{d} - \alpha\,\nabla\mathcal{L}_{\text{total}}\big(t_K^{d}\big), \qquad d \in \{S, T\}}
$$

**Two pre-trained encoders** (both frozen, marked with $*$ in the diagram): a **text encoder** $g(\cdot)$ taking $t_K^{S}$ to a text embedding, and an **image encoder** $f(\cdot)$ taking $x_i^{S}$ to an image embedding. This gives **$2K$ embeddings** — one per prompt — of which $t_K^{S}$ is the **domain-matching prompt** for a source image.

**Classification.** The class posterior is a CLIP-style softmax over similarities, but with the denominator summing over **both** domains' prompts:

$$
\boxed{P\big(\hat{y}_i^{s} = K \mid x_i^{s},\, t_K^{s}\big) = \frac{\exp\Big(\text{sim}\big(g(t_K^{s}),\, f(x_i^{s})\big)\Big)}{\displaystyle\sum_{d \in \{S,T\}}\sum_{j=1}^{K}\exp\Big(\text{sim}\big(g(t_j^{d}),\, f(x_i^{s})\big)\Big)}}
$$

The **sum over both domain prompts** in the denominator is the mechanism that ties the two domains together: a source image must not only match its own class prompt, it must out-score the *target* domain's prompts for all other classes too. **Similarly for $\mathcal{D}_T$ samples, after pseudo-labels** are assigned.

**Losses.** On the labelled source:

$$
\boxed{\mathcal{L}_S = -\frac{1}{N_S}\sum_{i=1}^{N_S}\log P\big(\hat{y}_i^{s} = y_i^{s}\big)}
$$

On the pseudo-labelled target, with a confidence gate:

$$
\boxed{\mathcal{L}_T = -\frac{1}{N_T}\sum_{i=1}^{N_T}\log \underbrace{P\big(\hat{y}_i^{T} = PL_i^{T} \ge \tau\big)}_{\text{prob. that the } i\text{-th sample's prediction matches its pseudo-label}}}
$$

**Usually, only if $P(\hat{y}_i^{T}) \ge \tau$ is the sample counted** $\Rightarrow$ **$\tau$ is the confidence threshold, and it removes pseudo-label noise.** Combining:

$$
\boxed{\mathcal{L}_{\text{Total}} = \mathcal{L}_T + \mathcal{L}_S}
$$

### 6.6 Insights on Prompt Learning

**A lot of work is happening in "prompt learning" / "prompt tuning."** Some insights follow from splitting the prompt into a **shared part** and a **domain-specific part**:

$$
\underbrace{[\;\;\;]}_{\text{shared part } t_K^{SH}}\;\underbrace{[\;\;\;]}_{\text{specific to } S \text{ or } T:\; t_K^{d},\; d\in\{S,T\}}\;\underbrace{[\;\;\;]}_{[\text{class}]_K}
$$

**The domain-specific tokens** are updated by their own domain's loss only:

$$
t_K^{S} \leftarrow t_K^{S} - \eta\,\nabla_t \mathcal{L}_S, \qquad\qquad t_K^{T} \leftarrow t_K^{T} - \eta\,\nabla_t\mathcal{L}_T
$$

**The shared token** receives gradients from *both*, weighted:

$$
\boxed{t_K^{SH} \leftarrow t_K^{SH} - \eta\Big(\nabla_t\mathcal{L}_T + \lambda\,\nabla_S\mathcal{L}_S\Big)}
$$

and **when there are more than one source domain**, the source contributions sum:

$$
\boxed{t_K^{SH} \leftarrow t_K^{SH} - \eta\Big(\nabla_t\mathcal{L}_T + \lambda\sum_i \nabla_{S_i}\mathcal{L}_{S_i}\Big)}
$$

These are **multi-objective gradients**, and the highlighted warning is:

> **Conflict must be resolved first, before adding.**

with the accompanying reasoning:

> **Since $t_K^{SH}$ is shared across domains, its learned embeddings should also be similar across domains, so there should be no conflict in the gradients.**

That is, gradient conflict between the domain terms is *diagnostic*: if the shared token is genuinely capturing domain-invariant semantics, the per-domain gradients ought to point in compatible directions. Conflict signals that the "shared" part is being pulled into doing domain-specific work.

**Why learn source prompts at all?** A natural objection: **we have domain-information and labels for the source domain — why are we learning its corresponding prompts?** Two answers:

1. **If $t_K^{S}$ were fixed, then $t_K^{SH}$ would try to do both domain-invariant *and* source-specific fitting** — just to fit the labelled source. The shared token would be contaminated by source specifics, defeating its purpose.
2. **Further, if $t_K^{S}$ were frozen, they may be miscalibrated to the feature extractor / dataset**, since **they are part of the class prototypes in CLIP-style inference**:

$$
P\big(y = K \mid x^{s},\, t_K^{s}\big) = \frac{\exp\Big(\text{sim}\big(f(x^{s}),\, g(t_K)\big)\Big)}{\displaystyle\sum_{d\in\{S,T\}}\sum_{j=1}^{K}\exp\Big(\text{sim}\big(f(x^{s}),\, g(t_j)\big)\Big)}
$$

In other words, the text embeddings *are* the classifier weights (the $\psi_k$ of Lecture 1). A frozen prompt is a frozen class prototype, and there is no reason a prototype inherited from generic pre-training should be well-placed relative to *this* encoder on *this* dataset.

---

## 7. Open Problems and Challenges in DA

**Measure-based:**

1. **Focus is on global alignment $P_S(x) \leftrightarrow P_T(x)$. This does not ensure class-conditional alignment**, particularly **when classes overlap differently across domains.** $\Rightarrow$ **conditional class-aware methods are needed.**
2. **Concept / semantic shift is not handled directly.**

**Adversarial alignment:**

1. **The min-max game over-aligns and erodes the discrimination power of the model.** Pushed too far, domain-invariance is achieved by discarding information the classifier needed.
2. **A single discriminator on marginals still wrongly aligns conditionals** — the same class-blindness problem as above, arrived at by a different route.

**Contrastive-based:**

1. **Pseudo-label noise corrupts positive / negative samples.** A mislabelled target sample is not merely unhelpful; it is actively used as a *positive* for the wrong class.
2. **Strong contrastive separation can conflict with domain alignment.** Sketched as two class clusters $C_1$ and $C_2$ separated by distance $d_{c_1c_2}$, each containing $\mathcal{D}_1$ and $\mathcal{D}_2$ samples: **increasing $d_{c_1c_2}$ may compromise the closeness of $\mathcal{D}_1$ and $\mathcal{D}_2$ samples.** The two objectives pull on the same geometry.

**Prompt learning:**

1. **Prompts are brittle and ambiguous:** a **small $\Delta t_K$ produces large swings in predictions.**
2. **Pseudo-label dependence persists.**
3. **A frozen image encoder might produce image embeddings which are inconsistent with the text embeddings corresponding to learned prompts** — the two modalities drift apart because only one side is being adapted.

**Cross-cutting challenges:**

- **PSDA, OSDA, UDA: very little work** has been done. Almost all effort targets CSDA.
- **Source-free DA:** **no access to $\mathcal{D}_S$ data — rather, a model trained on this data is available.** The consequences:
  - **It is not possible to directly align $P_S(x)$ and $P_T(x)$**, since one of the two distributions is simply absent.
  - **There is no label guidance for $\mathcal{D}_T$ samples. This causes a drift in the model during training, called Unknown-Model-Deviation.**
  - **Research is very small** in this area.

> **[Unclear in source]** — the first source-free bullet is written as "align $P_S(x)$ and $P_T(Y)$"; comparing an input marginal against a label marginal does not match the alignment framing used throughout, so $P_T(x)$ is used here.

---

## Looking Ahead

The lecture reduces to one inequality and one caveat. The inequality is Section 5.3's bound, $L_T \le L_S + \text{Divergence} + \min(a,b)$, whose middle term every method of Section 6 attacks with an instrument borrowed from an earlier lecture. The caveat is that marginal alignment is not conditional alignment: making $P_S(z)$ and $P_T(z)$ indistinguishable in aggregate permits a solution that matches the two domains class-for-class incorrectly, which is why the progression from measure-based to adversarial to contrastive methods is a progression in class-awareness rather than merely in sophistication.

[[ATML_Lecture_5_2 - Domain Generalization]] removes the assumption that made all of this possible. With no target data available at training time, the divergence term cannot be evaluated at all, and the bound must be restated over the convex hull of the source domains instead. Every family in the next lecture is recognizable as a response to that single deletion: alignment methods align the sources to one another in the absence of a target, learning strategies seek solutions that are stable under the perturbation an unseen domain represents, and disentanglement methods attempt to separate what is causal from what is domain-specific so that only the former is relied upon.

---

## Numerical Problems

> [!example]- Problem 1 — Decomposing excess risk
> A model family $\mathcal{H}$ is chosen for a task. The best possible model over all hypotheses achieves $R_{\min} = 0.05$; the best model inside $\mathcal{H}$ achieves $R_{\min}^{*} = 0.12$; ERM returns a model with $R_{\text{emp}}(\hat{f}_\theta) = 0.18$. Compute $AE$, $EE$, and the excess risk, and state which component could be reduced by switching to a larger architecture.
>
> *Solution.* Applying the definitions of Sections 3.2 and 3.4, the approximation error is $AE = R_{\min}^* - R_{\min} = 0.12 - 0.05 = 0.07$, and the estimation error is $EE = R_{\text{emp}}(\hat f_\theta) - R_{\text{True}}(f_\theta^*) = 0.18 - 0.12 = 0.06$. Summing,
>
> $$
> \boxed{AE = 0.07, \qquad EE = 0.06, \qquad R_{\text{excess}} = 0.13}
> $$
>
> Only $AE$ would be reduced by enlarging $\mathcal{H}$, since $AE$ measures model misspecification; $EE$ is the component reducible *within* a fixed $\mathcal{H}$ (by more data or lower capacity).
>
> *Reasonableness check:* both components are non-negative as they must be, and their sum equals $R_{\text{emp}}(\hat f_\theta) - R_{\min} = 0.18 - 0.05 = 0.13$, confirming the decomposition telescopes correctly. Note the trade-off: enlarging $\mathcal{H}$ to shrink $AE$ raises capacity, which per Section 3.5 works against $EE$.

> [!example]- Problem 2 — Margin and cross-entropy
> A 4-class classifier produces logits $\psi_k^\top z_i$ of $(1.0,\, 4.0,\, 0.5,\, 2.0)$ for an example whose true class is $j = 2$. Compute the three margins $m_{jk}(z_i)$ and the resulting $\mathcal{L}_{\text{CE}_i}$ in nats. Then determine the loss if the true-class logit were raised to $8.0$.
>
> *Solution.* Using $m_{jk}(z_i) = \psi_j^\top z_i - \psi_k^\top z_i$ from Section 3.6 with $\psi_j^\top z_i = 4.0$, the margins are $m_{21} = 3.0$, $m_{23} = 3.5$, and $m_{24} = 2.0$. Substituting into $\mathcal{L}_{\text{CE}_i} = \log\big(1 + \sum_{k\neq j} e^{-m_{jk}}\big)$, the exponentials are $e^{-3.0}\approx 0.0498$, $e^{-3.5}\approx 0.0302$, and $e^{-2.0}\approx 0.1353$, summing to $0.2153$. Then $\log(1.2153) \approx 0.1950$.
>
> $$
> \boxed{m = (3.0,\, 3.5,\, 2.0), \qquad \mathcal{L}_{\text{CE}_i} \approx 0.195 \text{ nats}}
> $$
>
> Raising the true logit to $8.0$ gives margins $(7.0, 7.5, 6.0)$, exponentials $0.000912 + 0.000553 + 0.00248 = 0.00394$, and $\log(1.00394) \approx 0.00393$.
>
> $$
> \boxed{\mathcal{L}_{\text{CE}_i} \approx 0.0039 \text{ nats — a 50-fold reduction}}
> $$
>
> *Reasonableness check:* the loss falls toward zero as the margins grow, exactly as Section 3.6 predicts ($\mathcal{L}_{\text{CE}_i} = 0$ when $m_{jk}\to\infty$). Note that a mere 4-unit increase in one logit collapses the loss almost entirely — which is precisely why cross-entropy can be satisfied by a low-capacity model that has found *any* strongly-correlated direction, spurious or not.

> [!example]- Problem 3 — Reading the DA bound
> A CSDA model achieves source loss $L_S(f_\theta) = 0.10$. An MMD estimate of $\text{Divergence}(P_S(x), P_T(x))$ gives $0.25$. Since the setting is closed-set, compute the bound on $L_T(f_\theta)$. Then recompute if the setting were partial-set with $a = 0.30$ and $b = 0.18$.
>
> *Solution.* In CSDA the label spaces are consistent, so $P_S(Y\mid x) = P_T(Y\mid x)$ and the conditional alignment term vanishes: $\min(a,b) = 0$ per Section 5.3. The bound is therefore $L_T \le 0.10 + 0.25 + 0$.
>
> $$
> \boxed{L_T(f_\theta) \le 0.35 \quad \text{(CSDA)}}
> $$
>
> For the partial-set case, the conditional term no longer vanishes and contributes $\min(0.30,\, 0.18) = 0.18$, giving $L_T \le 0.10 + 0.25 + 0.18$.
>
> $$
> \boxed{L_T(f_\theta) \le 0.53 \quad \text{(PSDA)}}
> $$
>
> *Reasonableness check:* the PSDA bound is looser, which is correct — semantic shift adds a source of target error that no amount of marginal alignment can remove, and this is exactly why the course restricts attention to CSDA. Note also that the bound uses $\min(a,b)$, not $a$ or $b$ individually, so only the *cheaper* of the two alignment directions is charged.

> [!example]- Problem 4 — DAN objective
> A DAN aligns layers $\ell_1 = 6$ through $\ell_2 = 8$ with $\lambda = 0.5$. The measured MMD distances are $d(\mathcal{D}_S^6, \mathcal{D}_T^6) = 0.4$, $d(\cdot^7) = 0.3$, and $d(\cdot^8) = 0.2$, and the source cross-entropy is $\mathcal{L}_{\text{CE}} = 0.35$. Compute $\mathcal{L}_{\text{Total}}$.
>
> *Solution.* The penalty in Section 6.1 uses the *squared* distances, so the three terms are $0.4^2 = 0.16$, $0.3^2 = 0.09$, and $0.2^2 = 0.04$, summing to $0.29$. Scaling by $\lambda = 0.5$ gives $0.145$, and adding the source cross-entropy,
>
> $$
> \mathcal{L}_{\text{Total}} = 0.35 + 0.5(0.29).
> $$
>
> $$
> \boxed{\mathcal{L}_{\text{Total}} = 0.495}
> $$
>
> *Reasonableness check:* forgetting to square the distances would give $0.5(0.9) = 0.45$ instead of $0.145$ — a threefold error, and the most common slip on this formula. Note also that the deepest aligned layer contributes least, consistent with shift being progressively corrected through the alignment block.

> [!example]- Problem 5 — Gradient reversal
> At some training step, $\frac{\partial \mathcal{L}_{\text{domain}}}{\partial u} = 0.8$, $\frac{\partial u}{\partial z} = 0.5$, and $\frac{\partial z}{\partial \theta_f} = 2.0$, with $\lambda = 0.1$. Compute the gradient that reaches $\theta_f$ with and without the GRL, and state the direction each pushes the feature extractor.
>
> *Solution.* Without the GRL, the ordinary chain rule of Section 6.2 gives $0.8 \times 0.5 \times 2.0 = 0.8$. With the GRL, the same product is multiplied by $-\lambda$: $-0.1 \times 0.8 = -0.08$.
>
> $$
> \boxed{\text{without GRL: } +0.8 \quad\text{(minimizes } \mathcal{L}_{\text{domain}}\text{)}; \qquad \text{with GRL: } -0.08 \quad\text{(maximizes it)}}
> $$
>
> Without the GRL, gradient descent on $\theta_f$ would make the latent *more* domain-separable — the opposite of what DA needs. With the GRL, the sign flips, so the update drives $g_{\theta_f}(x_S) \approx g_{\theta_f}(x_T)$, i.e. domain invariance.
>
> *Reasonableness check:* the magnitude also shrinks by the factor $\lambda = 0.1$, which is intentional — $\lambda$ controls how hard the adversarial pressure competes against the supervised loss, and is typically ramped up over training so the classifier is not destabilized early on.

> [!example]- Problem 6 — Class prototype and confidence-gated target loss
> Three source images of class $c$ produce unnormalized features $(3,4)$, $(0,5)$, and $(6,8)$. Compute the prototype $\mu_c$. Separately, in a prompt-learning run with $N_T = 4$ target samples, the probabilities assigned to their pseudo-labels are $(0.95,\, 0.60,\, 0.88,\, 0.40)$ with threshold $\tau = 0.7$. Compute $\mathcal{L}_T$ over the retained samples.
>
> *Solution.* Following Section 6.3, each feature is $\ell_2$-normalized *before* averaging: $(3,4)$ has norm $5$ giving $(0.6, 0.8)$; $(0,5)$ has norm $5$ giving $(0, 1)$; $(6,8)$ has norm $10$ giving $(0.6, 0.8)$. Averaging the three,
>
> $$
> \boxed{\mu_c = \tfrac13\big[(0.6,0.8) + (0,1) + (0.6,0.8)\big] = (0.4,\; 0.8\overline{6})}
> $$
>
> For the target loss, the threshold $\tau = 0.7$ retains only the samples with probabilities $0.95$ and $0.88$; the other two are discarded as noisy pseudo-labels. Averaging the negative logs over the retained samples, $-\log(0.95) \approx 0.0513$ and $-\log(0.88) \approx 0.1278$, so $\mathcal{L}_T = \tfrac12(0.0513 + 0.1278)$.
>
> $$
> \boxed{\mathcal{L}_T \approx 0.0896 \text{ nats (2 of 4 samples retained)}}
> $$
>
> *Reasonableness check:* note that $(3,4)$ and $(6,8)$ normalize to the *same* unit vector — they differ only in magnitude, which normalization deliberately discards, so the prototype is a mean of directions rather than of magnitudes. On the loss side, including the two rejected samples would have raised $\mathcal{L}_T$ to roughly $0.36$ — the threshold is doing real work, which is exactly the pseudo-label-noise removal described in Section 6.5.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Domain | $\mathcal{D}_i \triangleq \{P_i(x),\, P_i(y\mid x),\, \mathcal{L}\}$ |
| Task | $T_j \triangleq \{P_i(x),\, P_i(y\mid x),\, \mathcal{L}_j\}$, $j = 1,\ldots,T$ |
| Domain shift | $P_S(X,Y) \neq P_T(X,Y)$ |
| Factorization | $P(X,Y) = P(X)P(Y\mid X)$ |
| Covariate shift | $P_T(x) \neq P_S(x)$ but $P_T(Y\mid X) = P_S(Y\mid X)$ |
| Semantic / concept shift | $P_T(Y\mid x) \neq P_S(Y\mid x)$ |
| Expected (true) risk | $R_{\text{True}}(f_\theta) = \mathbb{E}_{P(X,Y)}[L(f_\theta(x), Y)]$ |
| Best model in $\mathcal{H}$ | $f_\theta^* = \arg\min_\theta R_{\text{True}}(f_\theta)$ |
| Approximation error | $AE = R_{\min}^* - R_{\min}$ |
| Empirical risk | $R_{\text{emp}}(f_\theta) = \frac1N\sum_{i=1}^N L(f_\theta(x_i), y_i)$ |
| ERM rule | $\hat{f}_\theta(\cdot) = \arg\min_\theta R_{\text{emp}}(f_\theta)$ |
| Estimation error | $EE = R_{\text{emp}}(\hat f_\theta) - R_{\text{True}}(f_\theta^*)$ |
| Excess risk | $R_{\text{excess}}(\hat f_\theta) = AE + EE$ |
| $EE \to 0$ conditions | (1) new data IID; (2) capacity $\beta\sqrt{S}$ not too large |
| Capacity | $\beta = \prod_{\ell=1}^{L}\lVert\theta_\ell\rVert_2$, $S = \sum_{\ell=1}^{L}\frac{\lVert\theta_\ell\rVert_F^2}{\lVert\theta_\ell\rVert_2^2}$ |
| CE in margin form | $\mathcal{L}_{\text{CE}_i} = \log\big(1+\sum_{k\neq j}\exp[-m_{jk}(z_i)]\big)$, $m_{jk}(z_i) = \psi_j^\top z_i - \psi_k^\top z_i$ |
| Spurious correlation signature | $\text{Corr}(z_{dd}, \psi_j)$ very high; IID accuracy $\uparrow$, OOD $\downarrow$ |
| DA bound theorem | $L_T(f_\theta) \le L_S(f_\theta) + \text{Divergence}(P_S(x), P_T(x)) + \min(a,b)$ |
| Conditional alignment terms | $a = \mathbb{E}_{P_S}[\lvert P_S(Y\mid x) - P_T(Y\mid x)\rvert]$; $b = \mathbb{E}_{P_T}[\cdot]$ |
| DAN objective | $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CE}} + \lambda\sum_{\ell=\ell_1}^{\ell_2} d^2(\mathcal{D}_S^\ell, \mathcal{D}_T^\ell)$ |
| DANN supervised loss | $\mathcal{L}_{\text{sup}}(\theta_f,\theta_y) = \mathcal{L}_{\text{CE}}(h_{\theta_y}(g_{\theta_f}(x_i)), y_i)$ |
| DANN domain loss | $\mathcal{L}_{\text{domain}} = \mathbb{E}_{P_S(x)}[\log D_{\theta_d}(g_{\theta_f}(x))] + \mathbb{E}_{P_T(x)}[1 - \log D_{\theta_d}(g_{\theta_f}(x))]$ |
| DANN total (for $\theta_f$) | $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{sup}} - \lambda\mathcal{L}_{\text{domain}}$ |
| Domain invariance target | $g_{\theta_f}(x_S) \approx g_{\theta_f}(x_T)$ |
| GRL forward / backward | forward: identity; backward: $\nabla_{\theta_f}\mathcal{L}_{\text{domain}} = -\lambda\left(\frac{\partial z}{\partial\theta_f}\right)\left(\frac{\partial u}{\partial z}\right)\left(\frac{\partial\mathcal{L}_{\text{domain}}}{\partial u}\right)$ |
| Class prototype | $\mu_c = \frac{1}{N_c}\sum_{i:y_i=c}\frac{f_\theta(x_i)}{\lVert f_\theta(x_i)\rVert_2}$ |
| Cross-domain contrastive (S2T) | $\mathcal{L}_{S2T}(a) = -\log\frac{\sum_p \exp(\text{sim}(z_a,z_p))}{\sum_k \exp(\text{sim}(z_a,z_k))}$ |
| CDC loss | $\mathcal{L}_{\text{CDC}} = \frac{1}{\lvert A_S\rvert}\sum_{a\in A_S}\mathcal{L}_{S2T}(a) + \frac{1}{\lvert A_T\rvert}\sum_{a\in A_T}\mathcal{L}_{T2S}(a)$ |
| Contrastive DA total | $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{CDC}} + \mathcal{L}_{\text{sup}}$ |
| Prompt structure | $t_K^d = [v]_1\ldots[v]_{M_1}\,[d]_1\ldots[d]_{M_2}\,[\text{class}]_K$; $2K$ prompts for 2 domains |
| Prompt-based posterior | $P(\hat y_i^s = K\mid x_i^s, t_K^s) = \frac{\exp(\text{sim}(g(t_K^s), f(x_i^s)))}{\sum_{d\in\{S,T\}}\sum_{j=1}^K \exp(\text{sim}(g(t_j^d), f(x_i^s)))}$ |
| Prompt losses | $\mathcal{L}_S = -\frac{1}{N_S}\sum_i \log P(\hat y_i^s = y_i^s)$; $\mathcal{L}_T$ gated at $\tau$; $\mathcal{L}_{\text{Total}} = \mathcal{L}_T + \mathcal{L}_S$ |
| Shared-token update | $t_K^{SH}\leftarrow t_K^{SH} - \eta\big(\nabla_t\mathcal{L}_T + \lambda\sum_i\nabla_{S_i}\mathcal{L}_{S_i}\big)$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the four DA settings** — **"C-P-O-U"**, read by what the target has relative to the source: **C**losed (same), **P**artial (target has fewer), **O**pen (target has more), **U**niversal (each has its own private classes). Only **C** has covariate shift alone; the other three add semantic shift.

**Mnemonic for the four CSDA mainstreams** — **"MACM"**: **M**easure-based, **A**dversarial-based, **C**ontrastive-based, **M**ultimodality-based. The first two align *marginals*; the second two align *classes*.

**Quick-reference intuition per topic**

- **Domain vs. task:** a domain fixes $\{P(x), P(y\mid x)\}$; a task is a choice of loss on top of it. Same data, different loss = different task. Different data = different domain.
- **DA vs. DG in one line:** DA gets unlabelled target data at training time; DG gets nothing.
- **Covariate vs. semantic shift:** $P(X,Y) = P(X)P(Y\mid X)$ — covariate shift moves the first factor, semantic shift moves the second. Near OOD = semantic only; Far OOD = both.
- **The ERM story in four quantities:** $R_{\text{True}}$ (want it, can't compute it) $\to$ $AE$ (price of choosing $\mathcal{H}$, fixed once you pick an architecture) $\to$ $R_{\text{emp}}$ (what you actually minimize) $\to$ $EE$ (price of using a finite sample — **the only thing you can reduce once $\mathcal{H}$ is fixed**).
- **Why ERM breaks under shift:** $EE \to 0$ requires *IID new data*. Domain shift violates the premise, so the guarantee simply does not apply. The model isn't broken; the theorem never covered the case.
- **The margin form of CE:** $\mathcal{L}_{\text{CE}_i} = \log(1 + \sum_{k\neq j}e^{-m_{jk}})$. Loss $\to 0$ as margins $\to \infty$, which needs only *some* direction correlated with $\psi_j$ — the objective never specifies *which*. That freedom is where shortcuts enter.
- **Spurious correlation signature:** $\text{Corr}(z_{dd}, \psi_j)$ high; **IID accuracy up, OOD accuracy down.** A spurious model looks *better* on a standard validation split — which is why the failure hides.
- **The $2\times2$ feature table:** want top-left (causal + invariant, e.g. object shape); can recover top-right (causal but domain-entangled, e.g. camera-dependent contrast); ignore bottom-left (benign, e.g. a watermark on everything); fear bottom-right (spurious, e.g. background, sky colour).
- **Cows and camels:** grass$\to$cow and desert$\to$camel is perfectly predictive on the source and actively *anti*-predictive on a target where the backgrounds are swapped.
- **The DA bound's three levers:** source loss (train well), marginal divergence (align $P(x)$), and $\min(a,b)$ (align $P(Y\mid x)$). In CSDA the third is zero by definition — which is exactly why CSDA is the tractable case.
- **"Control statement, not comparison":** the theorem does not claim $L_T > L_S$; it says $L_T$ is *controlled by* three reducible quantities.
- **Measure vs. adversarial, one line:** measure-based *specifies* the divergence (MMD, CORAL, WD); adversarial *learns* it with a discriminator.
- **DAN gotcha:** the penalty uses $d^2$, not $d$, and it is applied at **several** layers $\ell_1\ldots\ell_2$, not just the last one.
- **The GRL in one sentence:** identity forward, multiply by $-\lambda$ backward — the whole min-max game packed into one autograd node, so DANN trains with a single ordinary loop (unlike GANs, which need two alternating phases).
- **Sign check for DANN:** $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{sup}} - \lambda \mathcal{L}_{\text{domain}}$. The feature extractor **maximizes** the domain loss (fools the discriminator); the discriminator **minimizes** it.
- **Why contrastive DA exists:** marginal alignment is class-blind — it can align a source cow with a target camel. Contrastive DA aligns *class-conditionally* by pairing a source anchor with same-class target positives.
- **Prototype gotcha:** normalize each feature **before** averaging. $\mu_c$ is a mean of *directions*, not of magnitudes.
- **K-means initialization matters:** seeding the target clustering with the source prototypes $\mu_c$ is what transfers label semantics across the domain gap.
- **Prompt learning:** the prompt is `[domain-agnostic][domain-specific][class]`; only the tokens are learned, both encoders stay frozen. Two domains × $K$ classes = $2K$ prompts, and the softmax denominator sums over **both** domains' prompts.
- **Text embeddings ARE the class prototypes:** $g(t_K)$ plays the role of $\psi_k$ from Lecture 1. That is why freezing source prompts is a bad idea — you'd be freezing your classifier weights at values calibrated for some other encoder and dataset.
- **$\tau$ exists to kill pseudo-label noise:** target samples below the confidence threshold are simply not counted.
- **Gradient conflict is diagnostic:** if the shared prompt token's per-domain gradients fight, the "shared" part is being pulled into domain-specific work. Resolve conflict *before* adding.
- **The recurring open problem, stated once:** almost every method aligns **marginals** while what actually matters is **conditionals**. Measure-based, adversarial — both fail the same way.
- **Source-free DA:** you get the trained model but not the source data, so you cannot align anything directly, and the model drifts during training — **Unknown-Model-Deviation**.

**Recall checklist**

- [ ] Can you write the formal definitions of a domain and a task, and say precisely what differs between two tasks in the same domain?
- [ ] Can you name the three types of learning problems with an example dataset for each?
- [ ] Can you state the one-sentence difference between DA and DG?
- [ ] Can you define covariate shift and semantic shift in terms of $P(X,Y) = P(X)P(Y\mid X)$, and say which combination gives Near vs. Far OOD?
- [ ] Can you define $R_{\text{True}}$, $R_{\text{emp}}$, $AE$, $EE$, and excess risk, and say which one is reducible once $\mathcal{H}$ is fixed?
- [ ] Can you state the two conditions for $EE \to 0$, and explain why the first one is the reason models break under domain shift?
- [ ] Can you derive the margin form of cross-entropy and explain what it implies about shortcuts?
- [ ] Can you define a spurious correlation and give the cows-and-camels example?
- [ ] Can you reproduce the $2\times2$ feature taxonomy with an example in each cell?
- [ ] Can you name the four DA settings, their category-space relations, and which shifts each has?
- [ ] Can you state the DA bound theorem, name all three terms, and say which vanishes in CSDA?
- [ ] Can you write the DAN objective, including the squared distance and the layer range?
- [ ] Can you write DANN's three update rules and say which sign is flipped and why?
- [ ] Can you describe the GRL's forward and backward behaviour?
- [ ] Can you list the four steps of contrastive-based DA and write $\mathcal{L}_{\text{CDC}}$?
- [ ] Can you describe both pseudo-label generation methods, including the multi-head loop?
- [ ] Can you write the three-part prompt structure, say how many prompts are learned, and write the CLIP-style posterior with its two-domain denominator?
- [ ] Can you give both reasons why source prompts are learned rather than frozen?
- [ ] Can you name at least one open problem for each of the four method families, plus the source-free DA challenge?

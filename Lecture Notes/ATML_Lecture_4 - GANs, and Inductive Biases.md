---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Module I - Notes.pdf]]'
tags:
- course/cs6304
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]] · [[ATML_Lecture_5_1 - Domain Adaptation]] →
Concepts: [[Generative Adversarial Network]], [[Inductive Bias]]
%% nav:end %%

# ATML — Lecture 4: GANs, and Inductive Biases in Deep Representation Learning

## Where This Lecture Sits

*Continues from [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models]], which left the generative family at the VAE (variational autoencoder) — explicit likelihood, smooth latent, blurry samples. Section 1 here completes that family with the opposite design, and the rest of the lecture turns to the second lever of the generalization bound.*

GANs (generative adversarial networks) trade explicit likelihood for sharp samples at the cost of mode collapse — the reverse-KL half of Lecture 2 Section 5.4. Sections 3–6 then take up **inductive bias**, the second route to a tighter bound named in Lecture 1 Section 5.4: a model whose form already encodes a true assumption about the data does not need excess capacity to fit it. The property-wise biases of Section 6 supply the group-theoretic vocabulary Lecture 1 Section 3 deferred.

## Topic Inventory

- 1. Generative Adversarial Networks
  - 1.1 Setup — Removing the Encoder
  - 1.2 The Discriminator, the Generator, and the Equilibrium
  - 1.3 The Min-Max Game
  - 1.4 Where $V(G,D)$ Comes From — Binary Cross-Entropy
  - 1.5 Reading the Value Function — Who Wins When
  - 1.6 Training GANs in Two Phases
  - 1.7 Saturating vs. Non-Saturating GANs
  - 1.8 Mode Collapse and Vanishing Gradients
- 2. VAE Latent vs. GAN Latent
- 3. Inductive Bias
  - 3.1 Definition
  - 3.2 Strong Bias vs. Weak Bias
- 4. Semantic Biases
  - 4.1 Shape Bias and Texture Bias
  - 4.2 The Shape/Texture Landscape Across Models
  - 4.3 The Cue-Conflict Problem
  - 4.4 Style Transfer and Stylized ImageNet
  - 4.5 Color Bias and Colored MNIST
  - 4.6 Are Semantic Biases Good or Bad?
- 5. Architectural Biases
  - 5.1 Locality Bias and the Receptive Field
  - 5.2 Receptive Field and Depth
  - 5.3 Pooling and the Receptive Field
  - 5.4 The Effective Receptive Field
  - 5.5 Locality Bias vs. Global Context — CNN vs. ViT
  - 5.6 A Concrete Contrast on a Toy Matrix
  - 5.7 Computational Scaling
  - 5.8 Combining Both, Method 1 — Weighted Sum of Values
  - 5.9 Combining Both, Method 2 — Post- and Pre-Normalization Combination
  - 5.10 Combining Both, Method 3 — Convolutional Modulation of Values
  - 5.11 Hybrid Architectures
- 6. Property-Wise Biases
  - 6.1 Translation Invariance
  - 6.2 Translation Equivariance
  - 6.3 Permutation Invariance
  - 6.4 Permutation Equivariance
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Generative Adversarial Networks

Lecture 3 Section 5.6 attributed VAE blurriness to two causes: MSE (mean squared error) averages over plausible reconstructions, and the smooth prior limits how sharply the decoder commits. Both are consequences of having an explicit likelihood to maximize. The GAN removes that objective entirely, replacing "assign high probability to the data" with "produce samples a classifier cannot distinguish from data" — a criterion that never requires a density and therefore never has to average.

### 1.1 Setup — Removing the Encoder

The defining architectural move of a GAN, relative to the VAE of Lecture 3, is stated in one line: **remove the encoder.** The consequence is immediate and important — **the generator $G$ does not know anything about $X$.** It never sees a real image; it only ever receives a latent sample and a signal about how convincing its output was.

$$
z \sim P_\theta(z) \;\longrightarrow\; \boxed{G \;\text{(Generator)}} \;\longrightarrow\; \hat{x} = G(z)
$$

Because there is no reconstruction target, the training signal must come from somewhere else. It comes from **adversarial feedback**, which answers the question: *"Does this $\hat{x}$ look like real $X$ or not?"*

That question is answered by a second network, the **discriminator** $D$, which receives both real and generated samples:

$$
\hat{x} \;\longrightarrow\; \boxed{D \;\text{(Discriminator)}}, \qquad\qquad P(x) \to x \;\longrightarrow\; \boxed{D}
$$

and outputs the pair $\langle D(x),\, D(\hat{x})\rangle$, where $D(\cdot)$ is the **probability assigned by $D$ to its input for being real**.

### 1.2 The Discriminator, the Generator, and the Equilibrium

The discriminator is doing plain **binary classification** with labels

$$
\hat{x} = \text{Fake} \;\Rightarrow\; \text{label} = 0, \qquad\qquad x = \text{Real} \;\Rightarrow\; \text{label} = 1,
$$

so its prediction targets are $D(\hat{x}) = 0$ and $D(x) = 1$.

The two networks' goals are directly opposed:

- **$D$ should produce a high $D(x)$, close to $1$, and a low $D(\hat{x})$, close to $0$.**
- **$G$ should make $D(\hat{x})$ close to $1$** — i.e. **fool $D$**. A perfect $G$ produces $\hat{x}$ such that $D(\hat{x}) \approx 1$.

**At equilibrium**, neither can improve:

$$
D(x) = D(\hat{x}) = 0.5 \;\Longrightarrow\; D \text{ is maximally confused} \;\Longrightarrow\; \boxed{P(\hat{x}) = P(x)}
$$

That final implication is the whole point of the construction: if a well-trained discriminator genuinely cannot distinguish generated from real samples, the two distributions must have converged. The adversarial game is a way of minimizing a distributional divergence — specifically the Jensen–Shannon divergence of Lecture 2 — without ever writing down a density.

*Intuition.* The discriminator is a *learned* test function, which places the GAN squarely in the integral-probability-metric template of Lecture 2 Section 6.1: two distributions are close when no admissible witness can separate them. Where MMD (maximum mean discrepancy) fixes its witness family in advance through a kernel, the GAN trains its witness by gradient descent, which is what makes it powerful and also what makes it unstable — the metric being minimized changes at every step, so the generator is descending a moving landscape rather than a fixed one. This is the structural reason GAN training requires the careful alternation of Section 1.6 while VAE training does not.

### 1.3 The Min-Max Game

The whole training procedure is compressed into a single saddle-point objective:

$$
\boxed{\min_{G}\;\max_{D}\; V(G,D) = \mathbb{E}_{P(x)}\big[\log D(x)\big] + \mathbb{E}_{P(z)}\big[\log\big(1 - D(G(z))\big)\big]}
$$

$D$ maximizes it (getting better at telling real from fake) while $G$ minimizes it (getting better at fooling $D$).

### 1.4 Where $V(G,D)$ Comes From — Binary Cross-Entropy

The value function is not arbitrary; it is exactly the negative binary cross-entropy of the discriminator's classification problem. Setting up that problem,

$$
\begin{cases} x \to y = 1 \\ \hat{x} \to y = 0 \end{cases} \;\longrightarrow\; \boxed{D} \;\longrightarrow\; \hat{y} \qquad \text{(binary classification)},
$$

the standard **binary cross-entropy** loss is

$$
\mathcal{L}_{\text{BCE}} = -y\log\hat{y} - (1-y)\log(1-\hat{y}).
$$

Substituting the two cases — the real branch ($y=1$, $\hat y = D(x)$) kills the second term, and the fake branch ($y=0$, $\hat y = D(G(z))$) kills the first — gives

$$
\mathcal{L}_{\text{BCE}} = \underbrace{-\log\big(D(x)\big)}_{y=1} \;\underbrace{-\log\big(1 - D(G(z))\big)}_{y=0}.
$$

Therefore

$$
V(G,D) = -\mathcal{L}_{\text{BCE}} \;\Longrightarrow\; V(G,D) = \mathbb{E}_{P(x)}\big[\log D(x)\big] + \mathbb{E}_{P(z)}\big[\log(1 - D(G(z)))\big],
$$

which is exactly the min-max objective of Section 1.3. **The GAN objective is BCE with a sign flip** — nothing more exotic than that.

### 1.5 Reading the Value Function — Who Wins When

Because $D(\cdot) \in [0,1]$, both logarithms are non-positive, so

$$
V(G,D) \;\le\; 0.
$$

Its maximum is attained at

$$
V(G,D)\big|_{\max} = 0 \qquad \text{when } D(x) = 1 \text{ and } D(\hat{x}) = D(G(z)) = 0,
$$

which means the **discriminator wins** — it classifies every sample perfectly.

**When does $G$ win?** When **$D$ is maximally uncertain**, i.e. $D(\cdot) = 0.5$ always. Then $V = \log(0.5) + \log(0.5) = -2\log 2 \approx -1.386$, the minimum attainable when $D$ is playing optimally — the equilibrium of Section 1.2.

### 1.6 Training GANs in Two Phases

Training alternates between two phases, because the two networks cannot be updated with the same gradient step.

**Phase 1 — $D$ is trained.** Draw latents $\{z^1, \ldots, z^m\}$, push them through $G$ to obtain fakes $\{\hat{x}^1, \ldots, \hat{x}^m\}$ with **label $= 0$**, and pair them with a batch of reals $\{x^1, \ldots, x^m\}$ with **label $= 1$**. Feed both to $D$ and minimize

$$
\boxed{\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^{m}\Big[\log D(x^i) + \log\big(1 - D(\hat{x}^i)\big)\Big]}
$$

with the update
$$
\theta_D^{k} = \theta_D^{k-1} - \eta_D \nabla \mathcal{L}_D.
$$

**Phase 2 — $G$ is trained.** Get a **new** batch of latents $\{z^1, \ldots, z^m\}$, generate $\{\hat{x}^1, \ldots, \hat{x}^m\}$, and pass them through the **now-updated, frozen** discriminator $D^{*}$ (parameters $\theta_D^{*}$) to obtain $D(\hat{x}^i)$. Minimize

$$
\boxed{\mathcal{L}_G = -\frac{1}{m}\sum_{i=1}^{m}\log\big(D(\hat{x}^i)\big)}
$$

— this is the **non-saturating** generator loss — with the update
$$
\theta_G^{k} = \theta_G^{k-1} - \eta_G \nabla \mathcal{L}_G.
$$

Note that the generator's gradient flows *through* the frozen discriminator: $D$ acts as a learned, differentiable critic standing in for the loss function we cannot write down.

### 1.7 Saturating vs. Non-Saturating GANs

Two variants of the generator objective exist, and the difference is not cosmetic:

- **Saturating GAN:** for training $G$, **maximizes $\log\big(1 - D(\hat{x}^i)\big)$** — the literal second term of the min-max value function. This **leads to problems**.
- **Non-saturating GAN:** for training $G$, **minimizes $-\log\big(D(\hat{x}^i)\big)$** — the form used in Phase 2 above.

The reason the saturating form fails: early in training the generator is bad, so $D(\hat{x}) \approx 0$, and at that point $\log(1 - D(\hat{x})) \approx \log 1 = 0$ has a nearly **flat** gradient. The generator receives almost no learning signal exactly when it needs the most. The non-saturating form, $-\log D(\hat{x})$, has a *steep* gradient in the same regime — it blows up as $D(\hat{x}) \to 0$ — so the generator gets a strong push precisely when it is doing badly. Both objectives have the same fixed point; they differ only in gradient magnitude, and that difference is what makes GANs trainable at all.

### 1.8 Mode Collapse and Vanishing Gradients

**Mode collapse.** If the **discriminator is not good enough**, the **generator will stop learning after learning only a few modes**. The picture drawn is a multimodal $P(x)$ on the left collapsing to a single sharp peak on the right: many distinct latents $z_1, \ldots, z_m$ all map to the **same $\hat{x}$**. The generator has found one output convincing enough to fool a weak discriminator, and has no incentive to explore the rest of the distribution.

**Why it persists — vanishing gradients.** Mode collapse **persists due to vanishing gradients**. Consider

$$
\nabla_G \,\mathbb{E}_{P(x)}\big[\log D(G(z))\big].
$$

**If $D(G(z)) \approx 0$**, the generator **has learned some shortcut in the loss landscape**, and $\nabla_G$ becomes **small**, giving **unstable updates**. The generator is stuck: it cannot escape the collapsed mode because there is no gradient pointing out of it.

**Fixes** named in the notes: **InfoGAN** (a GAN that maximizes mutual information between part of its latent code and the output), **W-GAN** (Wasserstein GAN), and others. The W-GAN in particular replaces the JS-divergence-flavoured objective with the Wasserstein distance of Lecture 2 — precisely because, as established there, Wasserstein remains informative (finite, with usable gradients) even when the two distributions have disjoint support, whereas KL (Kullback–Leibler) and JS do not.

---

## 2. VAE Latent vs. GAN Latent

The two generative approaches produce latent spaces with opposite characters, and the contrast is worth memorizing as a unit:

|                         | **VAE Latent**       | **GAN Latent**        |     |
| ----------------------- | -------------------- | --------------------- | --- |
| Smoothness              | **Smooth**           | **Sharp**             |     |
| Interpretability        | **Interpretable**    | **Non-interpretable** |     |
| Distributional behavior | **Support-matching** | **Mode-seeking**      |     |

This maps directly onto the KL analysis of Lecture 2. The VAE's ELBO (evidence lower bound) contains an explicit $D_{\mathrm{KL}}(q_\psi(z\mid x)\,\|\,P(z))$ prior-matching term, which is a *support-covering* pressure — it forces the aggregate posterior to spread out and cover the prior, yielding a smooth, traversable latent (and, as a side effect, blurry samples). The GAN has no such term; its adversarial objective rewards producing samples the discriminator accepts, which is a *mode-seeking* pressure — yielding sharp samples and, in the failure case, mode collapse.

**Smoothness and sharpness are two ends of the same trade-off, not two independent defects.**

---

## 3. Inductive Bias

Sections 1 and 2 close the generative thread. The remainder of the lecture returns to the generalization bound of Lecture 1 Section 5.2 and to the second of the three levers listed in Section 5.4 there. The observation motivating it is that capacity is expensive only when it is *needed*: a model whose architecture already encodes a true assumption about the data can fit that data with a smaller $\beta\sqrt{S}$ than one that must discover the assumption from scratch. Inductive bias is the name for such built-in assumptions, and the three sections following classify them.

### 3.1 Definition

> **Inductive bias:** the set of (explicit or implicit) **assumptions** made by a machine learning algorithm in order to perform **induction** — that is, to **generalize a finite set of observations (the training data) into a general model of the domain**.

Two classical examples make the idea concrete:

- **SVM (support vector machine):** the assumption of **maximum margin** — among all separating hyperplanes, prefer the one furthest from the data.
- **KNN (k-nearest neighbours):** the assumption that **nearest neighbours** share labels — locality in feature space implies locality in label space.

**Intuitively.** Training data enters a learning algorithm which, in principle, has an **infinite** hypothesis space available (Model 1, Model 2, …, Model $\infty$); it must return *one* model, which then makes predictions on unseen data. **From the available infinitely many possible hypotheses, which model does the learning algorithm choose?** The answer to that question *is* the inductive bias. **The right choice leads to better generalization** — which connects directly to Lecture 1: inductive bias was listed there as one of the three ways to control the complexity/margin ratio.

### 3.2 Strong Bias vs. Weak Bias

- **Strong bias:** the set of assumptions is **relatively small** — the algorithm is willing to consider only a narrow slice of hypothesis space.
- **Weak bias:** the set of assumptions is **relatively large** — the algorithm will entertain almost anything.

The two boxed conclusions in the notes, one for each regime:

> **If a learning algorithm has strong bias, it is more likely to find a good model in low-data settings. But in rich-data settings, it may lead to constraints that harm generalization.** — Example: **CNN** (convolutional neural network).

> **If a learning algorithm has weak bias, it can find a better optimum if enough data is provided. However, it performs worse in low-data settings.** — Example: **ViT** (vision transformer).

This is the single most important trade-off in the module, and it recurs in every subsequent section: strong bias is a substitute for data, and data is a substitute for strong bias.

*Intuition.* Read in the terms of Lecture 1 Section 5.2, a strong bias shrinks the effective hypothesis space and therefore the complexity numerator of the bound, tightening it — but only if the assumption is *true of the data*. A false assumption shrinks the space in the wrong direction, excluding the good hypotheses along with the bad, and no amount of data repairs it, since the architecture cannot express what it has been built to exclude. This is why the strong/weak comparison is a genuine trade-off rather than a preference: bias converts data into generalization at a favourable rate when correct, and forecloses generalization entirely when not. Sections 4 through 6 can each be read as asking of a particular bias whether the assumption it encodes actually holds.

---

## 4. Semantic Biases

Section 3 treated inductive bias as a design choice. The first class of biases, however, is not designed at all: it is *acquired* during training, and consists of which features a model comes to prefer when several would serve. Since nothing in cross-entropy specifies which separating direction to use — the freedom identified at the end of Lecture 1 Section 7.4 — the preference is determined by what is easiest to extract, not by what is most reliable. Measuring that preference is therefore a diagnostic rather than a specification, and it is the first place in this course where a model's failure under distribution shift becomes predictable in advance.

**Semantic bias** asks: *how much does a model rely on semantics for making predictions?*

### 4.1 Shape Bias and Texture Bias

**Shape bias** measures the **reliance of a model on object shape**:

$$
\boxed{\text{SB} = \frac{\#\text{ of correct shape recognitions}}{\#\text{ of correct recognitions}}, \qquad 0 \le \text{SB} \le 1}
$$

**Texture bias** measures the **reliance of a model on object texture**:

$$
\boxed{\text{TB} = \frac{\#\text{ of correct texture recognitions}}{\#\text{ of correct recognitions}}, \qquad 0 \le \text{TB} \le 1}
$$

Both are ratios of counts of *correct* recognitions, so both are bounded in $[0,1]$; the denominator being "correct recognitions" (rather than all recognitions) means these metrics describe *how a model succeeds*, not *how often*.

### 4.2 The Shape/Texture Landscape Across Models

The measurement is made on a **5-class classification problem** with five shape categories, and results are plotted on a chart with two opposing axes: **SB** running right-to-left along the top ($0 \to 0.2 \to 0.6 \to 1$) and **TB** running left-to-right along the bottom ($0 \to 0.4 \to 0.8 \to 1$). Each of the five shape categories occupies one row, and each model type is plotted with its own marker:
![[03 - Attachments/Pasted image 20260914180301.png|539]]

| Model | Marker | Approximate position |
|---|---|---|
| **Humans** | red triangle | far left — highest shape bias, lowest texture bias |
| **ViTs** | between the triangle and the star | shape-biased, but less so than humans |
| **CLIP** (Contrastive Language–Image Pre-training) | blue star | intermediate |
| **CNNs** | cyan circle | far right — highest texture bias, lowest shape bias |

The headline finding: **humans classify by shape; CNNs classify by texture**, with ViTs and CLIP sitting in between. That gap is the empirical starting point for everything in Sections 4.3–4.4.

### 4.3 The Cue-Conflict Problem

The **cue-conflict problem** asks: *how do deep nets handle conflicting visual cues from texture and shape when classifying images?*

The **dataset** is constructed so the two cues disagree: an image is built with the **texture of elephant skin BUT the shape of a cat**. Consequently, **texture suggests the class elephant, while shape suggests the class cat.** The model's answer reveals its bias directly:

- **Texture-biased model** $\Rightarrow$ answers **Elephant**.
- **Shape-biased model** $\Rightarrow$ answers **Cat**.

This is an elegant diagnostic because it requires no interpretability tooling — the prediction itself is the measurement.

### 4.4 Style Transfer and Stylized ImageNet

**Style transfer** is a method to **change the texture of an image while keeping its shape or structure intact** — precisely the operation needed to build cue-conflict images at scale.

**Stylized ImageNet** (a 1.2-million-image, 1000-class image benchmark) is a **variant of the ImageNet dataset where images have been modified using style transfer**, so that texture is randomized while shape is preserved. Training on it removes texture as a reliable predictor, forcing the network to fall back on shape:

$$
\text{Stylized ImageNet} \;\longrightarrow\; \boxed{\text{CNN}_1} \quad \text{(training)} \qquad\Longrightarrow\qquad \boxed{\text{CNN}_1 \text{ will be more robust towards spurious correlations}}
$$

This is the watermark lesson of Lecture 1 in a different costume: if a cue is made unreliable during training, the model stops using it, and OOD (out-of-distribution) robustness improves.

### 4.5 Color Bias and Colored MNIST

**Color bias** measures **how sensitive or responsive a model is to color as opposed to actual features.**

**How is it measured?** By constructing a dataset where color and label are deliberately confounded:

- **MNIST (a dataset of handwritten digits):** 10 classes, greyscale images, output label = digit ($0$–$9$).
- **Colored MNIST:** 10 classes, colored images, output label = the pair $\{$digit, color$\}$, with each digit assigned a fixed color: $(0, \text{Brown})$, $(1, \text{Yellow})$, $(2, \text{Pink})$, $(3, \text{Green})$, …, $(9, \text{Red})$.

**Protocol.** Train a model on the colored images. At test time, query the trained model twice on the same input — once for the **digit** and once for the **color** — and compare the two accuracies:

$$
\text{Train:}\quad \text{Image} \to \boxed{\text{Model}} \to \{\text{color},\, \text{digit}\}
$$
$$
\text{Test:}\quad \text{Image} \to \boxed{\text{Trained Model}} \to \text{Digit}, \qquad \text{Image} \to \boxed{\text{Trained Model}} \to \text{Color} \qquad \Rightarrow \text{accuracy comparison}
$$

If color accuracy substantially exceeds digit accuracy, the model has taken the color shortcut — it has learned "brown means 0" rather than learning digit shape. Because the color–digit mapping is perfectly correlated in training, color is the *easier* feature, and the model will prefer it unless something prevents this.

### 4.6 Are Semantic Biases Good or Bad?

**It depends on the model's reliance on them.** Five worked comparisons show that no single semantic attribute is universally the right one:

| # | Comparison | Which cue discriminates |
|---|---|---|
| 1 | House vs. Zebra | **Texture** differs |
| 2 | Zebra vs. Car (a car with zebra-stripe paint) | **Shape** differs (texture is now misleading) |
| 3 | Chair vs. Table (same wood material) | Texture is the **same**; **shape** differs |
| 4 | Traffic light: (Red=ON, Yellow=OFF, Green=OFF) vs. (Red=OFF, Yellow=OFF, Green=ON) | **Color** is more useful than shape and texture |
| 5 | Glass of water vs. glass of milk vs. glass of coke | **Color** is more useful than shape and texture |

Cases 1–3 show shape and texture trading places depending on the pair. Cases 4–5 show situations where shape and texture are *identical* by construction and colour carries the entire signal — a traffic light has the same geometry in every state, and a glass has the same geometry regardless of contents.

The conclusion:

> **Semantic attributes (shape, color, texture) are not themselves bad. But if they cause spurious correlations, then OOD generalization will be bad.**

The problem was never the attribute; it was the *unreliable correlation* between the attribute and the label.

---

## 5. Architectural Biases

Section 4 closed on the observation that the problem is never the attribute but the unreliable correlation — which leaves open why a model gravitated toward one attribute rather than another. Part of that answer is architectural: an operator that can only see a $3\times3$ neighbourhood cannot represent shape until many layers have composed, whereas texture is available in the first layer. The CNN's texture bias is thus not an arbitrary preference but a consequence of what its receptive field makes cheap, which is the subject of this section.

**Architectural biases** are **implicit inductive biases induced in a model as a result of its architecture.** The two most common are **(a) locality bias / receptive fields** and **(b) global context** — and the two are related, so they are treated together.

### 5.1 Locality Bias and the Receptive Field

**Locality bias:** the model **focuses on local patterns more than on global context.**

**Receptive field (RF):** the **small localized regions of the input which affect a unit in subsequent layers** constitute the receptive field of that unit.

Two contrasting cases illustrate the definition:

- **Fully-connected layer:** the RF of a unit is **all the neurons in the previous layer** — i.e. the whole input. An FC layer has *no* locality bias. (Drawn with a small example where the unit sees all inputs; another sketch shows a unit with $\text{RF} = 3$.)
- **Convolution:** the RF of a unit is the small kernel window only — sketched on a grid as a single output cell fed by a $2\times2$ window, giving $\text{RF} = 4$.

### 5.2 Receptive Field and Depth

**In a CNN, as we go deeper, the RF of units increases linearly: each extra layer increases the RF by the kernel size.** The consequence is a hierarchy of feature scale, drawn as a chain:

$$
\underbrace{\square}_{\text{patterns over a small area, e.g. edges}} \to \square \to \square \to \cdots \to \underbrace{\square}_{\text{patterns over a larger area, e.g. shapes}}
$$

Early layers see edges; later layers, with their larger RFs, see shapes. This is the standard feature hierarchy, and it is a *consequence* of the RF arithmetic rather than something explicitly designed in.

### 5.3 Pooling and the Receptive Field

**Sub-sampling or pooling increases the RF multiplicatively** — a much faster growth than depth alone provides. The worked example: let the current $\text{RF} = 9$; take a $6\times6$ feature map through a **$2\times2$ max-pool with stride $2$**, giving a $3\times3$ feature map. Each neuron there now **corresponds to a larger region in the original input**:

$$
\text{RF} = 9 \times \underbrace{4}_{2\times2} = 36.
$$

**However, excessive sub-sampling can lead to loss of important spatial information/features.** Pooling buys receptive field at the cost of spatial resolution, and the trade has a limit.

### 5.4 The Effective Receptive Field

The theoretical RF overstates what a unit actually uses. **The effective area within the theoretical RF is only a fraction of it, called the Effective Receptive Field (ERF).**

$$
\text{ERF is distributed as a Gaussian with its peak at the center of the theoretical RF.}
$$

So influence decays smoothly outward from the center: the pixels at the edge of a theoretical RF contribute far less than those near the middle. This is why simply stacking layers to grow the theoretical RF does not automatically deliver true global context.

### 5.5 Locality Bias vs. Global Context — CNN vs. ViT

Two boxed conclusions in the notes state the trade-off in full.

> **Locality bias is beneficial for tasks like object detection and local image recognition, where understanding of local features and their spatial relationships is critical. But it limits long-range dependencies or global context of the image. $\Rightarrow$ CNNs have strong locality bias $\to$ they perform well in low-data settings.**

> **In contrast, Transformers like ViTs have no inherent locality bias, due to self-attention instead of convolution. ViTs cover global context earlier in the network. However, they may or may not capture local features. $\Rightarrow$ Transformers have weak bias $\Rightarrow$ they require a lot of data to capture patterns effectively.**

This is exactly the strong-vs-weak-bias trade-off of Section 3.2, now grounded in a specific architectural mechanism: convolution *is* the strong assumption; self-attention *is* the absence of one.

### 5.6 A Concrete Contrast on a Toy Matrix

Two $5\times5$ binary matrices are constructed to have opposite structure:

- $A$ contains **three distinct regions of local patterns** (blocks of ones clustered together).
- $B$ contains a **global pattern**: its **diagonal elements are important, and the connections between them are also essential** — an X-shaped pattern of ones.
![[03 - Attachments/Screenshot 2026-09-21 at 6.47.21 PM.png|434]]
**What convolution does.** With the filter $W = \begin{bmatrix}1&1\\1&1\end{bmatrix}$ and **no padding**, the convolution $A * W$ gives

$$
A * W = \begin{bmatrix} 4 & 2 & 0 & 0 \\ 4 & 2 & 0 & 0 \\ 2 & 4 & 2 & 0 \\ 2 & 4 & 2 & 0 \\ 0 & 2 & 2 & 0 \end{bmatrix}
$$

and the annotation reads: **local patterns are enhanced.** The high values ($4$) sit exactly where the local blocks are; distant relationships are invisible to the operation.

> **[Unclear in source]** — the printed result is a $5\times4$ matrix, whereas a valid (no-padding) $2\times2$ convolution on a $5\times5$ input would give $4\times4$. The matrix is reproduced as written; the point being made — that convolution amplifies local blocks and nothing else — is unaffected.

**What self-attention does.** For a ViT, self-attention computes

$$
D = \text{Attention weights} \times V, \qquad \text{Attention weights} = \text{Softmax}(\text{Attention scores}), \qquad \text{Attention scores} = QK^{\top},
$$

so
$$
D = \text{Softmax}\big(QK^{\top}\big)V.
$$

For simplicity assume $Q = K = V = B$. Then

$$
D = \text{Softmax}\big(BB^{\top}\big)B = \begin{bmatrix} 0.8 & 0.1 & 0.1 & 0.1 & 0.8 \\ 0.1 & 1 & 0.1 & 1 & 0.1 \\ 0.1 & 0.1 & 1 & 0.1 & 0.1 \\ 0.1 & 1 & 0.1 & 1 & 0.1 \\ 0.8 & 0.1 & 0.1 & 0.1 & 0.8 \end{bmatrix}
$$

with the annotation: **distant relationships are also enhanced.** The corners — positions $(1,5)$ and $(5,1)$, maximally far apart — carry large values ($0.8$), which no local operator could produce. This is the mechanism behind "no inherent locality bias" made visible in a single matrix.

### 5.7 Computational Scaling

Positioned on an axis running from **global context** (ViT end) to **locality bias** (CNN end):

**ViT:**
- Local patterns — **may or may not** be captured.
- **Excellent** global patterns.
- Requires **a lot of data**.
- **Attention scales quadratically**, $O(H^2 \times W^2)$, due to matrix multiplication.

**CNN:**
- **Excellent local pattern identification in low-data settings.**
- **Bad at global pattern identification.**
- **Convolution scales linearly** with respect to input size, $O(H_1 \times W_1)$.

The quadratic scaling of attention is a direct consequence of the $N\times N$ attention matrix from Lecture 3: with $N = HW$ tokens, forming $QK^\top$ costs $O(N^2) = O(H^2W^2)$.

**Ideally, we seek a model combining the good properties of both.** The following three sections give three ways to do it.

### 5.8 Combining Both, Method 1 — Weighted Sum of Values

Both convolution and attention can be written in one common form: a **weighted sum of values**. For convolution, let $L(i)$ denote the **local neighbourhood of the $i$-th pixel**. Then, for one channel,

$$
\boxed{y_i = \sum_{j \in L(i)} x_j \odot w_{i-j}}
$$

**Convolution relies on a fixed kernel $W$ to get information from $L(i)$.** Written as an operator, $x \circledast W = Y$ — this is a **depthwise** convolution when applied per channel. Doing it for **all channels** makes $\underline{y}_i$ a **vector**, where **every channel has a different $W$ kernel**.

### 5.9 Combining Both, Method 2 — Post- and Pre-Normalization Combination

**Attention in the same notation.** Let $\underline{x}_i$ be the **query** (the $i$-th feature vector), $G$ the **global set** of positions, and $\underline{x}_j$ the **values**. Then

$$
\underline{y}_i = \sum_{j\in G} \text{Softmax}\big(\text{score}(\underline{x}_i, \underline{x}_j)\big)\,\underline{x}_j,
$$

where the weight on the $\ell$-th value is

$$
\text{weight}_1 = \text{Softmax}\big(\text{score}(\underline{x}_i, \underline{x}_1)\big) = \frac{\exp\big(\underline{x}_i^{\top}\underline{x}_1\big)}{\sum_{k\in G}\exp\big(\underline{x}_i^{\top}\underline{x}_k\big)}, \qquad \text{score}(\cdot) = \underbrace{\underline{x}_i^{\top}}_{\text{query}}\underbrace{\underline{x}_j}_{\text{key}}.
$$

Therefore

$$
\underline{y}_i = \sum_{j\in G} \underbrace{\frac{\exp(\underline{x}_i^{\top}\underline{x}_j)}{\sum_{k\in G}\exp(\underline{x}_i^{\top}\underline{x}_k)}}_{A_{i,j}}\,\underline{x}_j \;=\; \sum_{j \in G} A_{i,j}\,\underline{x}_j.
$$

**The side-by-side comparison** is the heart of this section:

| Attention: $\;y_i = \sum_{j\in G} A_{i,j}\,x_j$ | Convolution: $\;y_i = \sum_{j\in L(i)} w_{i-j}\,x_j$ |
|---|---|
| **Input dependent** — weights depend on the actual content | **Input independent** — weights are fixed after training |
| $\text{RF} = G$ (global) | **Relative shift $(i-j)$ matters** between $y_i$ and $x_j$ → **translation equivariance** |
| | $\text{RF} = L(i)$ (local) |

**We can combine them in two ways.**

**1 — Post-normalization combination.** Add the relative positional term *outside* the softmax:

$$
\boxed{y_i^{\text{Post}} = \sum_{j\in G}\big(A_{i,j} + w_{i-j}\big)\,x_j}
$$

Here $w_{i-j}$ represents a **relative positional encoding, added to the Softmax output to bias the computed attention, where the bias depends on $i-j$.**

**2 — Pre-normalization combination.** Add the relative positional term *inside* the softmax:

$$
\boxed{y_i^{\text{Pre}} = \sum_{j\in G}\frac{\exp\big(\underline{x}_i^{\top}\underline{x}_j + w_{i-j}\big)}{\sum_{k\in G}\exp\big(\underline{x}_i^{\top}\underline{x}_k + w_{i-k}\big)}\,x_j}
$$

Here **$w_{i-j}$ is added inside the softmax to bias the attention weights based on relative position.**

The practical difference: in the post-normalization form the positional term is added to an already normalized weight, so the resulting weights no longer sum to one; in the pre-normalization form it biases the logits and normalization is preserved.

### 5.10 Combining Both, Method 3 — Convolutional Modulation of Values

**Convolutional Modulation (CM)** replaces the attention matrix entirely with a convolution, while keeping the multiplicative gating structure of attention.

**Self-attention, for reference.** Input $X: N \times C$ with $N = HW$ and $C$ = number of channels; $Q, K, V: N\times C$:

$$
\underbrace{\text{Attention}(x) = A V}_{N\times C}, \qquad \underbrace{A = \text{Softmax}(QK^{\top})}_{N\times N}, \qquad Q = W_QX,\; K = W_KX,\; V = W_VX.
$$

**Convolutional modulation.** Same input $X: N\times C$, but:

$$
\boxed{
\begin{aligned}
V &= W_2 X && \to N\times C \\
A &= \text{DWConv}_{K\times K}(W_1 X) \quad (C \text{ kernels}) && \to N\times C \\
Z &= A \odot V && \to N\times C
\end{aligned}
}
$$

The crucial structural differences: $A$ is now $N \times C$ rather than $N\times N$ (so the quadratic cost disappears), it is produced by a **depthwise convolution** with $C$ kernels rather than by a token-token similarity, and it is combined with $V$ by an **elementwise product** $\odot$ rather than a matrix product. The result keeps attention's content-dependent gating while inheriting convolution's linear scaling and locality bias.

Reading the two block diagrams side by side: in CM, $X$ ($HW\times C$) branches through $W_1$ (producing $A: HW\times C$) and $W_2$ (producing $V$), which are then combined by $\odot$ into $Z$. In self-attention, $X$ branches through $W_Q$, $W_K$, $W_V$; $Q$ and $K$ multiply into $A: HW\times HW$, which then multiplies $V$ to give $\text{Attention}(X): HW \times C$.

> **[Possible source inconsistency]** — the "different ways" list on these pages is numbered (1) Weighted Sum of Values … (3) Convolutional Modulation, and the following block is *also* numbered (3) Hybrid Architectures. The combination schemes of Section 5.9 are almost certainly the intended item (2); the numbering used in this document reflects that reading.

### 5.11 Hybrid Architectures

The final route to combining CNN and ViT properties is architectural rather than operator-level. Three families are given, each with named exemplars.

**(a) Architectural integration** — how the two backbone types are wired together:

| Sub-type | Structure | Examples |
|---|---|---|
| **Parallel** | $x$ splits into a CNN branch and a ViT branch, whose outputs are summed | **Conformer** |
| **Serial** | $x \to \text{CNN} \to \text{ViT}$ | **CoAtNet**, **CMT** |
| **Hierarchical** | repeated CNN/ViT stages at successive resolutions | **ViTVAE**, **CvT**, **DiNAT** |

**(b) Feature fusion** — where in the pipeline the two feature types are merged:

| Sub-type | Structure | Examples |
|---|---|---|
| **Early** | fuse at the stem, before the main blocks | **LeViT**, **MaskFormer** |
| **Late** | fuse near the head, after separate processing | **ViTMatte** |

**(c) Attention mechanism** — modifying attention itself rather than the backbone:

| Sub-type | Structure | Examples |
|---|---|---|
| **MHSA** | insert multi-head self-attention into a CNN backbone | **BoT** (Bottleneck Transformer) |
| **X-Attention** | $x \to \text{CNN} \to$ channel attention and spatial attention branches, recombined | **DANet**, **CBAM**, **HBCT** |

---

## 6. Property-Wise Biases

Section 5 compared convolution and attention by what they can *compute* — receptive field, cost, input dependence. This section compares them by what they *preserve*, which is a sharper and more portable criterion, since a symmetry the operator respects is an assumption no amount of data can dislodge. This is also the group-theoretic measurement family that Lecture 1 Section 3 named as the third lens and deferred: invariance and equivariance are exactly the two ways an operator can respect a group acting on its input.

**Property-wise biases** are biases that arise **due to the mathematical properties of the convolution ($\circledast$) and self-attention (SA) operations** themselves — not from data, and not from how blocks are wired, but from the algebra of the operators.

### 6.1 Translation Invariance

**Translation** is a geometric operation that **shifts all pixels in a given direction by the same distance**:
$$
T_v(x) = x + v.
$$

A model $f(\cdot)$ is **translation-invariant** if

$$
\boxed{f(x + \delta) = f(x) \quad \text{for any translation vector } \delta}
$$

- **CNNs achieve translation invariance to some extent through pooling layers**, which **select the most important features from different spatial locations** — a max-pool returns the same value whether the activating feature sat at the left or the right of its window.
- **ViTs do not possess translation invariance, due to fixed-size patches and positional embeddings.** The **content of each patch changes after translation**, resulting in a **different output**: shifting an image by a few pixels re-cuts every patch boundary, so the token sequence is genuinely different.

### 6.2 Translation Equivariance

A model $f(\cdot)$ is **equivariant** to another function $g(\cdot)$ if

$$
f\big(g(x)\big) = g\big(f(x)\big) \qquad\Longrightarrow\qquad \boxed{f(x + \delta) = f(x) + \delta}
$$

That is, transforming the input and then applying the model gives the same result as applying the model and then transforming the output. Invariance destroys the transformation; equivariance *carries it through*.

*Intuition, and why the distinction is not pedantic.* <mark style="background: #ABF7F7A6;">The right question for any layer is what it should do with information it is not being asked about. Invariance discards it</mark>: a classifier should return "cat" regardless of where the cat sits, so position is noise at the output. <mark style="background: #ABF7F7A6;">Equivariance preserves it</mark> in a usable form: a segmentation layer must know that a cat shifted right produces a mask shifted right, so position must survive the layer rather than being destroyed by it. <mark style="background: #ABF7F7A6;">The two are therefore appropriate at different depths of the same network,</mark> which is exactly the CNN's design — <mark style="background: #FFF3A3A6;">convolutions are equivariant and carry spatial information forward, pooling progressively converts that equivariance into invariance</mark>, and by the classifier head the position is gone. Discarding too early is the failure mode: an architecture that pools away location in its first layers cannot represent spatial relations later, because the information no longer exists to be recovered.

- **CNNs are translation equivariant, provided there are no pooling layers.** Here $f(\cdot) =$ convolution and $g(\cdot) =$ translation, and the commuting diagram — $x \to f \to g$ versus $x \to g \to f$ — gives equal results **if and only if there is no pooling.** (Pooling is exactly what converts equivariance into partial invariance, per Section 6.1.)
- **ViTs do not maintain translation equivariance, due to fixed-size patches**, but **it can be achieved by modifying the design of ViTs.**

### 6.3 Permutation Invariance

A transformation $T_\pi(\cdot): \mathbb{R}^{d\times n} \to \mathbb{R}^{d\times n}$ is called a **spatial permutation** if

$$
T_\pi(x) = \underbrace{X}_{d\times n}\underbrace{P_\pi}_{n\times n}, \qquad P_\pi = \big[\underline{e}_{\pi(1)} \;\; \underline{e}_{\pi(2)} \;\; \cdots \;\; \underline{e}_{\pi(n)}\big],
$$

where each $\underline{e}_i$ is a **one-hot vector with its $i$-th element equal to $1$**. For example,

$$
P_\pi = \begin{bmatrix} 0&1&0 \\ 0&0&1 \\ 1&0&0 \end{bmatrix}.
$$

An operator $A$ is **permutation-invariant (P.I.)** if

$$
\boxed{A\big(T_\pi(x)\big) = A(x)}
$$

The findings:

- **Convolution ($\circledast$) is NOT P.I.**
- **Self-attention is NOT P.I.**
- **Self-attention with a learnable Query matrix IS P.I.** Here "learnable" means $Q$ is a free parameter rather than a projection of the input: $Q \neq W_Q X$. Writing this operator $A_Q$,

$$
A_Q\big(T_\pi(x)\big) = W_v\,T_\pi(x)\cdot\text{Softmax}\Big(\big(W_K T_\pi(x)\big)^{\top}\cdot Q\Big),
$$

and using the orthogonality of permutation matrices,

$$
P_\pi P_\pi^{\top} = I \;\Longrightarrow\; P_\pi^{-1} = P_\pi^{\top},
$$

the permutation cancels and

$$
A_Q\big(T_\pi(x)\big) = A_Q(x).
$$

The intuition: when the query is a fixed learned vector rather than derived from the input tokens, the operator is aggregating over the token set without regard to its ordering — a set operation, not a sequence operation.

- **ViTs are NOT permutation-invariant with respect to the order of input patches** — precisely because of the positional embeddings added in Lecture 3, Section 2.6. Without them, the ViT *would* be permutation-invariant, which is exactly why they are needed.

### 6.4 Permutation Equivariance

An operator $A$ is **permutation-equivariant (P.E.)** if

$$
\boxed{T_\pi\big(A(x)\big) = A\big(T_\pi(x)\big)}
$$

The findings:

- **Convolution ($\circledast$) is NOT P.E.**
- **Self-attention IS P.E.** The derivation, writing the standard self-attention operator as $A_s$:

$$
A_s\big(T_\pi(x)\big) = W_v\,T_\pi(x)\cdot\text{Softmax}\Big(\big(W_K T_\pi(x)\big)^{\top}\cdot W_Q T_\pi(x)\Big)
$$

Substituting $T_\pi(x) = XP_\pi$ and expanding, the permutation matrices appear on both sides of the softmax argument, where the transpose pair $P_\pi^\top(\cdot)P_\pi$ acts as a simultaneous row and column permutation of the attention matrix — which the softmax commutes with, since softmax is applied elementwise after a row-wise normalization that permutes along with its rows:

$$
= W_v X P_\pi \cdot \text{Softmax}\Big(P_\pi^{\top}\big(W_K X\big)^{\top}\cdot W_Q X\Big)P_\pi
$$

$$
= W_v X\,\text{Softmax}\Big(\big(W_K X\big)^{\top}\cdot W_Q X\Big)P_\pi
$$

$$
= T_\pi\big(A_s(x)\big). \qquad \blacksquare
$$

The conclusion in plain terms: **self-attention treats its input as a set, not a sequence.** Permuting the tokens permutes the outputs identically and changes nothing else. This is the mathematical reason positional embeddings are not optional in a transformer — without them the architecture is literally incapable of distinguishing token orderings.

---

## Looking Ahead

The lecture's two halves reach the same conclusion from opposite directions. Sections 1–2 show that a generative model's sample quality is determined by which divergence its objective implicitly minimizes; Sections 3–6 show that a discriminative model's generalization is determined by which assumptions its architecture implicitly encodes. In both cases the behaviour worth predicting follows from a structural choice made before any data is seen.

Two threads carry into Module II. The first is diagnostic: Section 4 established that a model prefers whichever cue is easiest to extract, and that this preference is invisible on an in-distribution test set — the precondition for the spurious-correlation analysis that opens [[ATML_Lecture_5_1 - Domain Adaptation]]. The second is the adversarial mechanism of Section 1, which reappears in an unexpected role: where the GAN's discriminator distinguishes real from generated samples, the domain discriminator of Lecture 5.1 Section 6.2 distinguishes source from target features, and the generator's fooling objective becomes the feature extractor's alignment objective. The min-max structure is the same; only the two distributions being matched have changed.

---

## Numerical Problems

> [!example]- Problem 1 — GAN value function at three operating points
> Compute $V(G,D)$ for (a) a perfect discriminator, $D(x) = 1$ and $D(G(z)) = 0$; (b) the equilibrium, $D(x) = D(G(z)) = 0.5$; and (c) an intermediate state, $D(x) = 0.8$, $D(G(z)) = 0.3$. Use natural logarithms.
>
> *Solution.* Applying $V(G,D) = \mathbb{E}_{P(x)}[\log D(x)] + \mathbb{E}_{P(z)}[\log(1-D(G(z)))]$ from Section 1.3 pointwise:
>
> (a) $V_{(a)} = \log(1) + \log(1-0) = 0 + 0 = 0$. (Discriminator wins - the maximum)
> (b) $V_{(b)} = \log(0.5) + \log(0.5) = -0.6931 - 0.6931=-1.387$. (Equilibrium, generator wins)
> (c) $V_{(c)} = \log(0.8) + \log(0.7) = -0.2231 - 0.3567=-0.590$.
>
> *Reasonableness check:* all three values are $\le 0$, as required by Section 1.5, and they order correctly — the intermediate state sits between the discriminator's best case ($0$) and the equilibrium ($-1.386$), consistent with a discriminator that is doing well but not perfectly.

> [!example]- Problem 2 — Saturating vs. non-saturating generator gradients
> Early in training the discriminator is confident, $D(\hat{x}) = 0.05$. Compare the magnitude of $\left|\frac{\partial}{\partial D(\hat{x})}\right|$ for the saturating objective $\log(1-D(\hat{x}))$ and the non-saturating objective $-\log D(\hat{x})$.
>
> *Solution.* Differentiating each with respect to $u = D(\hat x)$: for the saturating form, $\frac{d}{du}\log(1-u) = \frac{-1}{1-u}$, whose magnitude at $u = 0.05$ is $\frac{1}{0.95} \approx 1.053$. For the non-saturating form, $\frac{d}{du}\big(-\log u\big) = \frac{-1}{u}$, whose magnitude at $u = 0.05$ is $\frac{1}{0.05} = 20$.
>
> $$
> \boxed{\left|\text{saturating}\right| \approx 1.05, \qquad \left|\text{non-saturating}\right| = 20, \qquad \text{ratio} \approx 19\times}
> $$
>
> *Reasonableness check:* the non-saturating gradient is nearly twenty times larger exactly in the regime where the generator is losing badly, which is the quantitative content of Section 1.7. As $D(\hat x) \to 0$ the saturating gradient tends to $1$ while the non-saturating one diverges — the generator that needs help most gets it only in the second formulation.

> [!example]- Problem 3 — Receptive field growth through depth and pooling
> A CNN applies three $3\times3$ convolutions (stride 1, no pooling), then a $2\times2$ max-pool with stride $2$, then two more $3\times3$ convolutions. Compute the theoretical RF after each stage, using the rules of Sections 5.2–5.3 (each conv layer adds the kernel size; pooling multiplies).
>
> *Solution.* Treating the RF as a linear side-length and applying the additive rule for convolution — an extra $3\times3$ layer extends the RF by $2$ on each side, i.e. $\text{RF} \to \text{RF} + (F-1)$ — the first three layers give $\text{RF} = 3$, then $5$, then $7$. The $2\times2$ stride-2 pool multiplies the linear RF by $2$, giving $\text{RF} = 14$. The final two convolutions each add $(F-1)\times\text{stride factor} = 2\times2 = 4$ in input pixels, giving $18$ then $22$.
>
> $$
> \boxed{3 \to 5 \to 7 \;\xrightarrow{\text{pool}}\; 14 \to 18 \to 22 \text{ (linear); area } 22^2 = 484 \text{ pixels}}
> $$
>
> *Reasonableness check:* the single pooling layer contributes more RF growth than the three convolutions that preceded it — which is precisely the multiplicative-vs-additive distinction of Section 5.3. The ERF (Section 5.4) is a Gaussian-weighted fraction of this $22\times22$ window, so the *usable* context is considerably smaller than $484$ pixels.

> [!example]- Problem 4 — Attention vs. convolution cost
> For a $224\times224$ feature map, compare the cost of a convolution ($O(H_1 W_1)$) against self-attention over per-pixel tokens ($O(H^2W^2)$). Then repeat with $16\times16$ patch tokens.
>
> *Solution.* Applying the scalings of Section 5.7, the convolution cost is proportional to $224 \times 224 = 50{,}176$. Per-pixel attention has $N = 50{,}176$ tokens, so its cost is proportional to $N^2 = 50{,}176^2 \approx 2.52\times10^{9}$ — a ratio of about $50{,}176:1$. With $16\times16$ patches, $N = (224/16)^2 = 196$, so attention costs $196^2 = 38{,}416$.
>
> $$
> \boxed{\text{per-pixel attention} \approx 2.52\times10^{9} \;\text{vs}\; \text{conv} \approx 5.0\times10^{4}; \quad \text{patch attention} \approx 3.8\times10^{4}}
> $$
>
> *Reasonableness check:* patchifying brings attention *below* the convolution cost, which is the entire engineering reason ViTs operate on $16\times16$ patches rather than pixels. It also shows why the quadratic term only bites when $N$ is large — the asymptotic penalty is real but the constant matters enormously in practice.

> [!example]- Problem 5 — Shape bias and texture bias from cue-conflict counts
> A model is evaluated on $500$ cue-conflict images. It produces $400$ recognitions matching one of the two constituent classes; of these, $120$ match the shape class and $280$ match the texture class. Compute SB and TB, and classify the model against the landscape of Section 4.2.
>
> *Solution.* Applying the definitions of Section 4.1, the denominator for both is the number of correct recognitions, $400$. Then $\text{SB} = 120/400 = 0.30$ and $\text{TB} = 280/400 = 0.70$.
>
> $$
> \boxed{\text{SB} = 0.30, \quad \text{TB} = 0.70, \quad \text{SB} + \text{TB} = 1}
> $$
>
> *Reasonableness check:* both lie in $[0,1]$ as required, and they sum to $1$ because on cue-conflict images every correct recognition must match either the shape class or the texture class. An SB of $0.30$ against a TB of $0.70$ places this model firmly on the CNN end of the chart in Section 4.2 — far from the human position, which sits near $\text{SB} \approx 1$.

> [!example]- Problem 6 — Verifying permutation equivariance on a $3\times3$ example
> Let $P_\pi = \begin{bmatrix}0&1&0\\0&0&1\\1&0&0\end{bmatrix}$ be the permutation matrix from Section 6.3. Verify that $P_\pi P_\pi^{\top} = I$, and determine which permutation of columns $T_\pi(X) = XP_\pi$ performs when $X = [\,c_1 \;\; c_2 \;\; c_3\,]$.
>
> *Solution.* Computing $P_\pi P_\pi^\top$: each row of $P_\pi$ is a distinct standard basis vector, so the $(i,j)$ entry of the product is the inner product of rows $i$ and $j$, which is $1$ when $i = j$ and $0$ otherwise.
>
> $$
> \boxed{P_\pi P_\pi^{\top} = I \;\Longrightarrow\; P_\pi^{-1} = P_\pi^{\top}}
> $$
>
> For the column action, the $j$-th column of $XP_\pi$ is $X$ times the $j$-th column of $P_\pi$. Column 1 of $P_\pi$ is $(0,0,1)^\top$, selecting $c_3$; column 2 is $(1,0,0)^\top$, selecting $c_1$; column 3 is $(0,1,0)^\top$, selecting $c_2$.
>
> $$
> \boxed{XP_\pi = [\,c_3 \;\; c_1 \;\; c_2\,] \quad \text{— a cyclic right-shift of the tokens}}
> $$
>
> *Reasonableness check:* orthogonality is what makes the cancellation in Section 6.3 work, and it holds for every permutation matrix, so the P.I. result for the learnable-query operator is general rather than specific to this $\pi$. Applying the same $P_\pi$ three times returns the original ordering, consistent with a 3-cycle.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| GAN generator | $\hat{x} = G(z)$, $z \sim P_\theta(z)$ |
| GAN labels | $x$ real $\to y=1$; $\hat{x}$ fake $\to y=0$ |
| GAN equilibrium | $D(x) = D(\hat{x}) = 0.5 \Rightarrow P(\hat{x}) = P(x)$ |
| Min-max game | $\min_G\max_D V(G,D) = \mathbb{E}_{P(x)}[\log D(x)] + \mathbb{E}_{P(z)}[\log(1-D(G(z)))]$ |
| Binary cross-entropy | $\mathcal{L}_{\text{BCE}} = -y\log\hat{y} - (1-y)\log(1-\hat{y})$ |
| Value function origin | $V(G,D) = -\mathcal{L}_{\text{BCE}}$, and $V(G,D)\le 0$ |
| Discriminator loss (Phase 1) | $\mathcal{L}_D = -\frac1m\sum_i[\log D(x^i) + \log(1-D(\hat{x}^i))]$ |
| Generator loss (Phase 2, non-saturating) | $\mathcal{L}_G = -\frac1m\sum_i \log D(\hat{x}^i)$ |
| Parameter updates | $\theta_D^k = \theta_D^{k-1} - \eta_D\nabla\mathcal{L}_D$; $\theta_G^k = \theta_G^{k-1}-\eta_G\nabla\mathcal{L}_G$ |
| Saturating GAN | $G$ maximizes $\log(1 - D(\hat{x}^i))$ — leads to problems |
| Mode-collapse gradient | $\nabla_G\mathbb{E}_{P(x)}[\log D(G(z))] \to$ small when $D(G(z))\approx 0$ |
| Shape bias | $\text{SB} = \frac{\#\text{ correct shape recognitions}}{\#\text{ correct recognitions}}$, $0\le\text{SB}\le1$ |
| Texture bias | $\text{TB} = \frac{\#\text{ correct texture recognitions}}{\#\text{ correct recognitions}}$, $0\le\text{TB}\le1$ |
| RF and depth | Each extra conv layer increases RF **linearly** by the kernel size |
| RF and pooling | Pooling increases RF **multiplicatively**: RF $= 9 \times 4 = 36$ for a $2\times2$ pool |
| Effective RF | ERF $\subset$ theoretical RF, Gaussian-distributed with peak at the centre |
| Convolution scaling | $O(H_1\times W_1)$ — linear |
| Attention scaling | $O(H^2\times W^2)$ — quadratic |
| Convolution as weighted sum | $y_i = \sum_{j\in L(i)} x_j\odot w_{i-j}$ — input independent, RF $= L(i)$ |
| Attention as weighted sum | $y_i = \sum_{j\in G} A_{i,j}x_j$, $A_{i,j} = \frac{\exp(x_i^\top x_j)}{\sum_{k\in G}\exp(x_i^\top x_k)}$ — input dependent, RF $= G$ |
| Post-normalization combination | $y_i^{\text{Post}} = \sum_{j\in G}(A_{i,j} + w_{i-j})x_j$ |
| Pre-normalization combination | $y_i^{\text{Pre}} = \sum_{j\in G}\frac{\exp(x_i^\top x_j + w_{i-j})}{\sum_{k\in G}\exp(x_i^\top x_k + w_{i-k})}x_j$ |
| Self-attention (CM comparison) | $\text{Attention}(x) = AV$, $A = \text{Softmax}(QK^\top)$, $A: N\times N$ |
| Convolutional modulation | $V = W_2X$; $A = \text{DWConv}_{K\times K}(W_1X)$ ($C$ kernels); $Z = A\odot V$, all $N\times C$ |
| Translation | $T_v(x) = x + v$ |
| Translation invariance | $f(x+\delta) = f(x)$ |
| Equivariance (general) | $f(g(x)) = g(f(x))$ |
| Translation equivariance | $f(x+\delta) = f(x) + \delta$ |
| Spatial permutation | $T_\pi(x) = XP_\pi$, $P_\pi = [e_{\pi(1)}\cdots e_{\pi(n)}]$, $P_\pi P_\pi^\top = I$ |
| Permutation invariance | $A(T_\pi(x)) = A(x)$ |
| Permutation equivariance | $T_\pi(A(x)) = A(T_\pi(x))$ |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the three bias types** — **"S-A-P: Semantic, Architectural, Property-wise."** **S**emantic = what cues the model uses (shape/texture/colour); **A**rchitectural = what the architecture makes easy (locality/global context); **P**roperty-wise = what the operator's algebra forces (invariance/equivariance).

**Quick-reference intuition per topic**

- **GAN in one line:** delete the encoder, replace the reconstruction loss with a learned critic, and train two networks against each other until the critic cannot tell real from fake.
- **The GAN objective is just BCE:** $V(G,D) = -\mathcal{L}_{\text{BCE}}$. If you can write BCE you can write the min-max game.
- **Who wins when:** $V = 0$ means $D$ wins perfectly; $V = -2\log 2 \approx -1.386$ with $D(\cdot) = 0.5$ means $G$ wins. $V$ is always $\le 0$.
- **Two phases, not one:** train $D$ on real+fake with fixed $G$; then train $G$ through a frozen $D$ on a *fresh* batch of latents.
- **Saturating vs. non-saturating:** same fixed point, different gradient magnitudes. Saturating ($\max \log(1-D(\hat x))$) goes flat when $G$ is bad; non-saturating ($\min -\log D(\hat x)$) goes steep. Always use non-saturating.
- **Mode collapse:** a weak $D$ lets $G$ get away with one mode; then $D(G(z))\approx 0$ makes $\nabla_G$ vanish, so $G$ cannot escape. Fixes: InfoGAN, W-GAN.
- **VAE vs. GAN latent:** smooth + interpretable + support-matching versus sharp + non-interpretable + mode-seeking. Same forward/reverse-KL dichotomy from Lecture 2, wearing architectures.
- **Inductive bias:** which hypothesis the algorithm picks out of infinitely many. SVM picks max margin; KNN picks nearest neighbours; CNN picks local patterns; ViT picks nothing in particular.
- **Strong vs. weak bias:** strong bias wins in low data, loses in rich data (CNN); weak bias loses in low data, wins in rich data (ViT). **Strong bias is a substitute for data.**
- **Shape vs. texture:** humans are shape-biased, CNNs are texture-biased, ViTs and CLIP sit in between. Cue-conflict images (elephant texture, cat shape) measure this directly — the answer *is* the measurement.
- **Stylized ImageNet:** style transfer randomizes texture while keeping shape, so texture stops being predictive and the CNN is forced onto shape → more robust to spurious correlations.
- **Colored MNIST:** confound colour with digit in training, then test digit accuracy against colour accuracy. A big gap means the model took the colour shortcut.
- **Semantic biases aren't inherently bad:** traffic lights and glasses of liquid make colour the *only* useful cue. The problem is spurious correlation, not the attribute.
- **RF arithmetic:** depth adds (linear, by kernel size); pooling multiplies. Excessive pooling costs spatial information.
- **ERF:** the *usable* receptive field is a Gaussian-weighted centre fraction of the theoretical one. Stacking layers to grow theoretical RF doesn't automatically buy global context.
- **Conv vs. attention as one formula:** both are $y_i = \sum_j (\text{weight}) \cdot x_j$. Conv weights are **input-independent** and depend on the relative shift $(i-j)$ (hence translation equivariance) with RF $= L(i)$; attention weights are **input-dependent** with RF $= G$.
- **Post- vs. pre-normalization:** post adds $w_{i-j}$ *outside* the softmax (weights no longer sum to 1); pre adds it *inside* (normalization preserved).
- **Convolutional modulation:** replace $A: N\times N$ from $QK^\top$ with $A: N\times C$ from a depthwise conv, and replace matrix multiply with elementwise $\odot$. Kills the quadratic cost, keeps the gating.
- **Invariance vs. equivariance:** invariance *destroys* the transformation ($f(x+\delta) = f(x)$); equivariance *carries it through* ($f(x+\delta) = f(x)+\delta$).
- **The four-way property table (memorize this):** CNNs are translation-**equivariant** without pooling and gain partial translation-**invariance** with pooling; ViTs have neither, because fixed patches and positional embeddings break both. Convolution is neither P.I. nor P.E.; self-attention is **P.E. but not P.I.** — unless the query is a free learned parameter, in which case it becomes **P.I.**
- **Why positional embeddings are mandatory:** self-attention is permutation-equivariant, i.e. it treats its input as a *set*. Without positional embeddings a transformer literally cannot distinguish token orderings.

**Recall checklist**

- [ ] Can you write the GAN min-max objective and derive it from BCE?
- [ ] Can you state the equilibrium condition and what it implies about $P(\hat{x})$ and $P(x)$?
- [ ] Can you write $\mathcal{L}_D$ and $\mathcal{L}_G$ and describe both training phases?
- [ ] Can you explain why the saturating generator loss fails, in terms of gradients?
- [ ] Can you explain mode collapse and why vanishing gradients make it persist? Can you name two fixes?
- [ ] Can you give the three-property contrast between VAE and GAN latents?
- [ ] Can you define inductive bias, give the SVM and KNN examples, and state the strong/weak trade-off with its two exemplar architectures?
- [ ] Can you write SB and TB and place humans, CNNs, ViTs, and CLIP on the shape/texture chart?
- [ ] Can you describe a cue-conflict image and predict what a texture-biased vs. shape-biased model answers?
- [ ] Can you explain what Stylized ImageNet does and why it improves robustness?
- [ ] Can you describe the Colored MNIST protocol and what a colour-vs-digit accuracy gap means?
- [ ] Can you give two examples where colour is the *right* cue?
- [ ] Can you compute RF growth through convolutions and pooling, and define the ERF?
- [ ] Can you state the two boxed CNN-vs-ViT conclusions on locality bias and data requirements?
- [ ] Can you write conv and attention in the shared weighted-sum form and list the three differences between them?
- [ ] Can you write both the post- and pre-normalization combinations and say where $w_{i-j}$ goes?
- [ ] Can you write the three lines of convolutional modulation and say why it avoids the quadratic cost?
- [ ] Can you name the three hybrid-architecture families with one exemplar each?
- [ ] Can you define translation invariance and equivariance, and say which of CNN/ViT has which and why?
- [ ] Can you define a spatial permutation matrix, state $P_\pi P_\pi^\top = I$, and say which operators are P.I. and which are P.E.?
- [ ] Can you reproduce the three-line proof that self-attention is permutation-equivariant?

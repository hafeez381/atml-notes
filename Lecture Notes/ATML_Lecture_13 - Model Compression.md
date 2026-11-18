---
type: lecture
course: "[[_CS6304 ATML|CS6304 ATML]]"
source:
  - "[[ATML Lec 20.pdf]]"
  - "[[ATML Lec 21.pdf]]"
  - "[[ATML Lec 22 (1).pdf]]"
  - "[[ATML Lec 22 (2).pdf]]"
  - "[[ATML Lec 23 (1).pdf]]"
  - "[[ATML Lec 23 (2).pdf]]"
tags:
  - atml
  - course/cs6304
  - model-compression
  - pruning
  - quantization
title: ATML — Lecture 13 — Model Compression
lecture: 13
aliases:
  - Model Compression
  - ATML Lecture 13
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning]]
%% nav:end %%

# ATML — Lecture 13 — Model Compression

*[[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning|Lecture 12]] spent more compute at inference to get better answers from a fixed model. This lecture goes the other way: it makes the model itself cheaper to store and run, while losing as little accuracy as possible. It starts from why large models are expensive and why a smaller model can still generalize (double descent), frames compression as a trade-off between rate and distortion, and then works through the four families: pruning, quantization, low-rank approximation and knowledge distillation, ending with feature-level and LLM distillation.*

There are no professor's notes for this lecture yet. The sources are your class notes [[ATML Lec 20.pdf|ATML Lecture 20]] (pages 3–5, from "Model Compression" on), [[ATML Lec 21.pdf|ATML Lecture 21]] (10 pages), and two sets of Lecture 22 notes: the paper notebook [[ATML Lec 22 (1).pdf|Lecture 22 (1)]] (4 pages, 12 Nov 2026) and the Notability [[ATML Lec 22 (2).pdf|Lecture 22 (2)]] (11 pages). [[#1. Why Compress a Model|§1]]–[[#2. Compression as Rate–Distortion|§2]] come from Lecture 20, [[#3. Pruning|§3]] and [[#4.1 Mapping reals to integers|§4.1]]–[[#4.5 What quantization touches, and integer arithmetic|§4.5]] from Lecture 21, and [[#4.6 Quantize and dequantize along the network|§4.6]]–[[#6. Knowledge Distillation|§6]] from Lecture 22. [[#6.9 Recap — what soft labels add, and why forward KL averages|§6.9]]–[[#7. Distillation for LLMs|§7]] come from two sets of Lecture 23 notes (17 Nov 2026): your friend's paper notebook [[ATML Lec 23 (1).pdf|Lecture 23 (1)]] (4 pages) and your Notability [[ATML Lec 23 (2).pdf|Lecture 23 (2)]] (8 pages). Where two sets differ, both readings are given, and where one set captured something better it is used. All figures are redrawn in TikZ from your notes.

> [!warning] Notation in this lecture
> Here $D$ is the **number of parameters** and $N$ the **number of training samples**. [[ATML_Lecture_7 - LLM Foundations#5. Scaling Laws|Lecture 7's scaling laws]] use the opposite letters ($N$ for parameters, $D$ for data). This note keeps your class notation.

## Topic Inventory

- 1. Why Compress a Model
  - 1.1 The model and its size
  - 1.2 The generalization gap
  - 1.3 Double descent
  - 1.4 Compute and memory grow with size
- 2. Compression as Rate–Distortion
  - 2.1 Rate and distortion
  - 2.2 A layer-level proxy for distortion
  - 2.3 The rate–distortion objective
  - 2.4 Four families of compression
- 3. Pruning
  - 3.1 Masks, sparsity and density
  - 3.2 Pruning as constrained optimization
  - 3.3 How to group weights
  - 3.4 How to score groups
  - 3.5 Activation-aware pruning (Wanda)
  - 3.6 When to prune
  - 3.7 The Lottery Ticket Hypothesis
- 4. Quantization
  - 4.1 Mapping reals to integers
  - 4.2 Scale, zero-point and levels
  - 4.3 Asymmetric quantization
  - 4.4 Symmetric quantization
  - 4.5 What quantization touches, and integer arithmetic
  - 4.6 Quantize and dequantize along the network
  - 4.7 When to quantize — QAT and the straight-through estimator
  - 4.8 Post-training quantization and activation awareness
  - 4.9 When quantization struggles — the outlier problem
  - 4.10 Mixed precision, combining methods, and architecture search
- 5. Low-Rank Approximation
  - 5.1 Replacing W by two thin matrices
  - 5.2 More rank from the same parameters — the Hadamard product
- 6. Knowledge Distillation
  - 6.1 Teacher, student and the undistilled baseline
  - 6.2 Logit matching and the combined loss
  - 6.3 Why the ρ² factor — distillation becomes logit MSE
  - 6.4 Hard labels vs soft labels
  - 6.5 Strict vs lenient teacher
  - 6.6 Shaping the teacher's distribution — temperature and label smoothing
  - 6.7 The accuracy gap and dark knowledge
  - 6.8 Decoupled knowledge distillation (DKD)
  - 6.9 Recap — what soft labels add, and why forward KL averages
  - 6.10 Gradient conflict between the CE and KD terms
  - 6.11 Resolving the conflict — gradient harmonization and momentum
  - 6.12 Why logits lose structure — the task head
  - 6.13 FitNets — feature-level distillation
  - 6.14 Contrastive Representation Distillation (CRD)
  - 6.15 Adversarial distillation and other divergences
  - 6.16 What knowledge gets distilled?
- 7. Distillation for LLMs
  - 7.1 Which prefix does the student see?
  - 7.2 Off-policy distillation and exposure bias
  - 7.3 On-policy distillation
  - 7.4 When to use LLM distillation
- Numerical Problems
- Formula Summary Sheet
- Summary

---

## 1. Why Compress a Model

### 1.1 The model and its size

The model is the usual pipeline of [[ATML_Lecture_1 - Representation and Generalization Theory#2.1 Pipeline Components and Naming|Lecture 1 §2.1]]: an encoder with parameters $\theta$ maps the input $x$ to a representation $z$, and a task head maps $z$ to the prediction $y$. All the parameters are stacked into one vector, $\theta \in \mathbb R^{D \times 1}$, so $D$ is the parameter count.

![[03 - Attachments/ATML_L13_setup.svg|640]]

*(a) The model whose size $D$ we want to cut. (b) The layer-level comparison used to measure the damage ([[#2.2 A layer-level proxy for distortion|§2.2]]).*

As $D$ grows, three things happen, and each is a reason to compress: the generalization gap behaves in a surprising way ([[#1.2 The generalization gap|§1.2]]–[[#1.3 Double descent|1.3]]), and compute and memory both grow ([[#1.4 Compute and memory grow with size|§1.4]]).

### 1.2 The generalization gap

The **generalization gap** is the difference between the loss on the whole data distribution and the loss on the training set:

$$\Delta = \underbrace{\mathbb E_{P(x, y)}\big[\ell(f_\theta(x), y)\big]}_{\text{actual risk}} \;-\; \underbrace{\frac1N\sum_{i=1}^{N}\ell\big(f_\theta(x_i), y_i\big)}_{\text{empirical risk}}$$

The **actual risk** is the theoretical average loss over the population $P(x, y)$, which we cannot compute. The **empirical risk** is its estimate from the $N$ training samples, which is what training minimizes. A large $\Delta$ means the model fits the training set much better than new data. [[ATML_Lecture_1 - Representation and Generalization Theory#5.1 The IID Setting and the Generalization Bound|Lecture 1 §5.1]] bounded this gap by a complexity term; the classical reading is that more parameters mean more complexity and a larger gap.

### 1.3 Double descent

The **double descent phenomenon** says the classical reading is only half the story.

![[03 - Attachments/ATML_L13_double_descent.svg|640]]

*Train (green) and test (red) loss against parameter count $D$, on a log scale. The test loss rises to a peak at $D \approx N$ and then falls again in the interpolation region.*

Reading the curve from left to right:
- **Under-parameterized ($D < N$).** The test loss first falls, then rises as the model starts **overfitting**. Here there are more equations (samples) than unknowns (parameters), and SGD (stochastic gradient descent) finds the **least-squares solution**, the best fit that cannot hit every point.
- **Interpolation point ($D \approx N$).** The model has just enough parameters to fit every training sample exactly. There is essentially one solution that does so, and it is forced to bend to the noise, so the test loss peaks.
- **Interpolation region ($D \gg N$).** The test loss falls again. Now the system $A\theta = y$ has more unknowns than equations, so it **has infinitely many solutions**. SGD started from small weights picks a particular one, the **minimum-norm solution**:
$$\hat x = A^{+}y$$
where $A^{+}$ is the **Moore–Penrose pseudo-inverse**; for a full-row-rank $A$, $A^{+} = A^\top(AA^\top)^{-1}$. *(Expanding on why this helps:)* among all the solutions that fit the data, the minimum-norm one is the smoothest, with the smallest weights, so it generalizes better than the forced fit at the peak. Adding parameters gives SGD more room to find such a solution.

Your notes mark CNNs and ViTs (Vision Transformers) as living in the region up to the interpolation point.

**Where compression fits.** Your notes mark two compression moves on the curve. A **deployed or trained model** sitting on the rising part, just before the peak, can be compressed back to the minimum of the first descent: fewer parameters and *lower* test loss. A model far out in the interpolation region can be compressed leftwards along the flat tail, losing many parameters for almost no change in loss. In both cases a smaller model need not be worse.

### 1.4 Compute and memory grow with size

The other two effects of a larger $D$ are costs:
- **FLOPs grow with $D$:** each forward pass does work proportional to the number of parameters (about $2D$ FLOPs per input, as in [[ATML_Lecture_7 - LLM Foundations#5.3 Chinchilla laws|Lecture 7 §5.3]]).
- **Storage and memory grow with $D$:** memory is needed to store the **weights**, the **activations** (intermediate outputs) and, during training, the **optimizer** states (Adam keeps two extra numbers per parameter).

Compression reduces these costs. The next section makes "how much cheaper" and "how much worse" precise.

---

## 2. Compression as Rate–Distortion

### 2.1 Rate and distortion

**Rate–distortion theory** measures a compressed model $\hat\theta$ with two numbers.

The **rate** is how many bits the model takes:

$$\text{Rate} = (\#\text{ non-zero parameters}) \times (\text{bits per parameter})$$

The **distortion** is how much worse the compressed model is:

$$\text{Distortion} = L(\hat\theta) - L(\theta)$$

the loss of the compressed model minus the loss of the original. It **should ideally be zero or negative**: negative is possible, as the left compression move of [[#1.3 Double descent|§1.3]] shows. *(Expanding on the name:)* rate–distortion theory comes from information theory ([[ATML_Lecture_2 - Information Theory and Divergences#3. Information Theory|Lecture 2 §3]]), where it asks how few bits can describe a signal within a given error.

The rate formula has two factors, and the compression families of [[#2.4 Four families of compression|§2.4]] differ in which one they cut.

### 2.2 A layer-level proxy for distortion

$L(\hat\theta)$ requires evaluating the whole compressed model on data, which is expensive to do for every candidate. So distortion is **approximated via a proxy measure**, layer by layer. For a linear layer (an MLP layer) with weight $W$, the original computes $Wx$ and the compressed one $\hat Wx$. The proxy is how far apart the two outputs are:

$$\big\lVert \hat Wx - Wx \big\rVert_F$$

where $\lVert\cdot\rVert_F$ is the **Frobenius norm**, the square root of the sum of squared entries (for a batch of inputs stacked as columns of $x$, the outputs are matrices). If every layer's output barely changes, the final loss barely changes.

### 2.3 The rate–distortion objective

Compression chooses $\hat W$ to balance the two:

$$\boxed{\min_{\hat W}\; \text{Rate} + \beta\,(\text{Distortion})} \qquad\qquad \text{(1)}$$
^eq-1

where $\beta > 0$ sets the exchange rate: a large $\beta$ protects accuracy, a small $\beta$ favours a smaller model. Your notes link this to **information bottleneck theory**, the trade-off of [[ATML_Lecture_1 - Representation and Generalization Theory#3. Representation Theory — What Makes a Representation Good?|Lecture 1 §3]] and [[ATML_Lecture_2 - Information Theory and Divergences#3.5 Mutual Information|Lecture 2 §3.5]] between keeping what matters for the task and discarding everything else. There it is applied to the representation $z$; here it is applied to the weights.

### 2.4 Four families of compression

Four families solve [[#^eq-1|eq. (1)]]. They **differ in how they select $\hat W$**:

| Family | What it does to $W$ | Which part of the rate it cuts |
|---|---|---|
| **Pruning** ([[#3. Pruning\|§3]]) | sets many weights to exactly 0, $\lVert \hat W\rVert_0 \ll \lVert W\rVert_0$ | the number of non-zero parameters |
| **Quantization** ([[#4. Quantization\|§4]]) | stores each weight in fewer bits | the bits per parameter |
| **Low-rank approximation** ([[#5. Low-Rank Approximation\|§5]]) | replaces $W$ by a product $BA$ of two thin matrices | the parameter count |
| **Knowledge distillation** ([[#6. Knowledge Distillation\|§6]]), offline and online | trains a small student to imitate a large teacher | the parameter count, by changing the architecture |

Your Lecture 21 notes open with "model compression has 2 techniques" and treat pruning and quantization; Lecture 22 finishes quantization and adds the other two. In short:
- **Low-rank approximation** writes $W \approx BA$ with $B \in \mathbb R^{d_o \times r}$, $A \in \mathbb R^{r \times d_i}$ and small $r$, cutting $d_od_i$ parameters to $r(d_o + d_i)$. It is the same factorization as [[ATML_Lecture_10 - Fine-Tuning#8.1 The low-rank update|LoRA in Lecture 10 §8.1]], applied to the weight itself instead of to its update.
- **Knowledge distillation** trains the student on the teacher's output distribution, the loss of [[ATML_Lecture_10 - Fine-Tuning#3.5 Distillation-based fine-tuning|Lecture 10 §3.5]]. *(Expanding on offline vs online:)* offline distillation trains the teacher first and then the student; online distillation trains both together.

---

## 3. Pruning

### 3.1 Masks, sparsity and density

Pruning removes weights by multiplying the weight matrix elementwise by a **mask** $M$ of zeros and ones:

$$\hat W = W \odot M$$

where $\odot$ is the elementwise (Hadamard) product. A 1 keeps a weight and a 0 removes it.

![[03 - Attachments/ATML_L13_mask.svg|560]]

*A $4\times4$ mask keeps 9 of 16 weights.*

> [!note] Mask entries
> Your notes write $M \in [0, 1]$. A pruning mask is binary, $M \in \{0, 1\}$, since each weight is either kept or removed. A mask with values in between would rescale weights rather than prune them.

The number of weights kept is counted with the **$\ell_0$ "norm"**, $\lVert W\rVert_0$, the number of non-zero entries. Pruning looks for $\hat W$ with

$$\lVert \hat W\rVert_0 \ll \lVert W\rVert_0$$

The amount of pruning is described by two fractions that sum to 1. With $D$ the total number of parameters:

$$\alpha = \text{sparsity} = \frac{\#\text{ parameters removed}}{D}, \qquad 1 - \alpha = \text{density} = \frac{\#\text{ parameters retained}}{D}$$

Both lie in $[0, 1]$ and can be written as percentages; your example has sparsity $\alpha = 90\%$ and density $1 - \alpha = 10\%$, the actual ratio of parameters kept.

> [!warning] Possible source inconsistency
> Your notes define $\alpha$ as "# of parameters retained / total" and then also call $1 - \alpha$ "# of parameters retained". Only one can hold. The example ($\alpha = 90\%$, density $10\%$) and the constraint in [[#3.2 Pruning as constrained optimization|§3.2]] both need $\alpha$ to be the fraction **removed**, as written above.

### 3.2 Pruning as constrained optimization

With masks $M^{(\ell)}$ for each layer $\ell$, pruning is the problem

$$\boxed{\min_{\{M^{(\ell)}\},\, W}\; \mathbb E_{P(x, y)}\Big[L\big(f_{\hat W}(x), y\big)\Big] \quad \text{s.t.} \quad \hat W = W \odot M, \quad \sum_\ell \big\lVert M^{(\ell)}\big\rVert_0 \le (1 - \alpha)\sum_\ell \big\lVert W^{(\ell)}\big\rVert_0} \qquad\qquad \text{(2)}$$
^eq-2

The objective is the usual expected loss (your notes assume a supervised task), now computed with the masked weights. The constraint says the masks together keep at most a fraction $1 - \alpha$ of the weights. [[#Numerical Problems|Problems 1–3]] work through it.

**The problem is NP-hard.** Choosing which $(1-\alpha)D$ of the $D$ weights to keep means searching over

$$\binom{D}{(1-\alpha)D}$$

possible masks, and no algorithm is known that finds the best one in polynomial time. *(Expanding on the size:)* even $D = 100$ at $\alpha = 0.9$ gives $\binom{100}{10} \approx 1.7\times10^{13}$ masks; a real network has billions of weights.

So pruning is **approximated with a two-step process**:
1. **identify groups** of weights that could be pruned together;
2. **score each group** and **discard the groups with the lowest score**.

This raises three questions, answered in [[#3.3 How to group weights|§3.3]]–[[#3.6 When to prune|3.6]]: how to group, how to score, and when to prune.

### 3.3 How to group weights

**Unstructured pruning** makes **every individual weight its own group**. It is **simple and was widely used before LLMs**. It reaches high sparsity with little loss, but the zeros are scattered, so the matrix keeps its shape and standard hardware gets little speed-up without special sparse kernels.

**Structured pruning** groups weights along the network's structure. In a CNN (convolutional neural network), a **group is a filter, a layer or a channel**: removing a filter deletes a whole slice of the weight tensor, so the layer becomes genuinely smaller and runs faster on ordinary hardware. The cost is that whole filters are a coarser choice, so the same sparsity usually loses more accuracy.

![[03 - Attachments/ATML_L13_grouping.svg|720]]

*(a) Unstructured: individual weights removed anywhere. (b) Structured: one whole filter removed. (c) Layer-wise: the global sparsity split into one budget per layer.*

**Layer-wise grouping (sparsity).** Instead of one global ranking, a **scheme** divides the overall sparsity $\alpha$ into per-layer sparsities $\alpha_1, \alpha_2, \dots, \alpha_L$. Then $\alpha_1$ is used to prune layer 1, $\alpha_2$ to prune layer 2, and so on, and the pruning inside each layer can be structured or unstructured. *(Expanding on why:)* layers differ in how much they tolerate pruning; the first and last layers are usually more sensitive, so a scheme can prune them less.

> [!warning] Possible source inconsistency
> Your notes write the condition as $\sum_\ell \alpha_\ell = \alpha$. Sparsities are fractions, so they do not add up across layers. The overall sparsity is the parameter-weighted average, $\sum_\ell \frac{D_\ell}{D}\alpha_\ell = \alpha$, where $D_\ell$ is layer $\ell$'s parameter count ([[#Numerical Problems|Problem 3]]).

### 3.4 How to score groups

The score estimates how much the loss would suffer if a group were removed. Removing a weight $\theta_j$ means setting it to 0, a change of $\delta\theta_j = -\theta_j$. *(Expanding on where all three scores come from:)* a Taylor expansion of the loss around the trained weights gives

$$\Delta L \approx \frac{\partial L}{\partial\theta_j}\,\delta\theta_j + \frac12 H_{jj}\,\delta\theta_j^2 = -\frac{\partial L}{\partial\theta_j}\,\theta_j + \frac12 H_{jj}\,\theta_j^2$$

where $H_{jj} = \partial^2 L/\partial\theta_j^2$ is a diagonal entry of the **Hessian**, the matrix of second derivatives. Keeping zero, one or two terms gives the three orders of scoring.

**Zero-order scoring (magnitude).** Use the weight alone: the score is $\lvert\theta_{ij}\rvert$, and the **lower it is, the lower the score**. For a group, sum the magnitudes in the unstructured case, or use the squared norm $\lVert\theta_j\rVert_2^2$ of the group (e.g. a filter) in the structured case. It ignores the loss entirely: a small weight is assumed to matter little.

**First-order scoring.** **Bring in the loss** through its gradient:

$$\text{score}_{ij} = \Big\lvert \frac{\partial L}{\partial\theta_{ij}}\cdot\theta_{ij}\Big\rvert$$

The gradient measures the **sensitivity of the loss with respect to $\theta_{ij}$**, so the product estimates the first-order change in loss from removing the weight. This is the score of **SNIP** (Single-shot Network Pruning based on connection sensitivity, Lee et al., 2019). A related group score is the **gradient norm** $\lVert \partial L/\partial\theta_j\rVert_2$.

**Second-order scoring.** At a well-trained minimum the gradient is close to zero, so the first-order term says little. The second-order term uses the curvature:

$$\text{score}_j = \frac12 H_{jj}\,\theta_j^2$$

**assuming a diagonal Hessian** (ignoring interactions between weights, which makes the Hessian affordable). $H_{jj}$ says **how sensitive the curvature of the loss is with respect to $\theta_j$**: a weight in a flat direction can move to 0 cheaply. **Lower sensitivity → remove it.** This is **Optimal Brain Damage** (LeCun et al., 1989). Your notes add a caveat: **curvature can be due to spurious things**, so a high $H_{jj}$ does not always mean the weight is important.

**Optimal Brain Surgeon** (Hassibi and Stork, 1993) goes one step further. It uses the full Hessian to **prune a weight and then adjust the remaining weights** to **recover** what was lost. *(Expanding on the adjustment:)* when weight $q$ is removed, the other weights move by $\delta\theta = -\frac{\theta_q}{[H^{-1}]_{qq}}H^{-1}e_q$, where $e_q$ is the $q$-th unit vector, and the loss increase is $\frac{\theta_q^2}{2[H^{-1}]_{qq}}$. Your notes say this will be studied in detail with **SparseGPT**, which applies the same idea layer by layer to LLMs.

| Order | Score | Uses | Weakness |
|---|---|---|---|
| Zero | $\lvert\theta_{ij}\rvert$, or $\lVert\theta_j\rVert_2^2$ per group | weights only | ignores the loss and the inputs |
| First | $\lvert(\partial L/\partial\theta_{ij})\,\theta_{ij}\rvert$ (SNIP) | gradient | gradient is near 0 at a minimum |
| Second | $\frac12 H_{jj}\theta_j^2$ (OBD); OBS also adjusts the rest | curvature | Hessian is expensive; curvature can be spurious |

### 3.5 Activation-aware pruning (Wanda)

All three scores of [[#3.4 How to score groups|§3.4]] look at the weights and the loss, but not at the **inputs** the weights multiply. Your notes show why that is a problem with a single neuron, **assuming no bias**:

![[03 - Attachments/ATML_L13_activation_aware.svg|420]]

*One neuron with two inputs. What reaches the next layer is $x_1\theta_1 + x_2\theta_2$, not the weights alone.*

1. If $\lvert\theta_1\rvert < \lvert\theta_2\rvert$, magnitude pruning removes $\theta_1$.
2. But if $\lvert x_1\rvert \gg \lvert x_2\rvert$, so that $\lvert x_1\theta_1\rvert \gg \lvert x_2\theta_2\rvert$, then $\theta_1$'s contribution to the **activation is much bigger**, and it is **propagated to later layers**. The right weight to prune is $\theta_2$.

**Activation-aware pruning** therefore scores each weight by its size times the size of its input. **Wanda** (Pruning by **W**eights **and a**ctivations, Sun et al., ICLR 2024) uses

$$\text{score}_{ij} = \lvert\theta_{ij}\rvert\cdot\lVert X_j\rVert_2$$

where $\lVert X_j\rVert_2$ is the norm of input feature $j$ over a small batch of calibration data. Your notes record that it **outperformed all of them** (the earlier scores). *(Expanding on why this matters for LLMs:)* LLMs have a few input features with very large activations, so magnitude alone prunes the wrong weights. Wanda needs no gradients and no retraining, only one forward pass to measure the activations.

### 3.6 When to prune

The third question is when pruning happens relative to training.

![[03 - Attachments/ATML_L13_when_to_prune.svg|560]]

*The three schedules. PAT is the most common.*

1. **PBT, pruning before training:** take the initial weights $\theta_0$, prune them, then train the sparse network. SNIP ([[#3.4 How to score groups|§3.4]]) was designed for this: its gradient score can be computed at initialization from one batch.
2. **PDT, pruning during training:** alternate training and pruning in a loop, then **fine-tune at the end** to recover accuracy.
3. **PAT, pruning after training (the most common):** $\theta_0 \to$ pre-train $\to$ prune $\to$ fine-tune. The pruning decision uses a fully trained network, which is when the scores of [[#3.4 How to score groups|§3.4]] are most reliable.

### 3.7 The Lottery Ticket Hypothesis

PBT asks whether a sparse network can be trained from the start. The **Lottery Ticket Hypothesis** (Frankle and Carbin, 2019) says yes, in a specific sense: **there exists a sparse network within a dense network that reaches as much accuracy as the full model**, when trained from the same initial weights.

The procedure that finds it loops over train, prune and **rewind**:

![[03 - Attachments/ATML_L13_lottery_ticket.svg|640]]

*Iterative magnitude pruning with rewinding. Each loop prunes more and restarts the survivors from their original initial values.*

1. Take a dense network $\theta_{\text{dense}}$ and **initialise** it to $\theta_0$ (Gaussian noise or another initialization scheme).
2. **Train** it to get $\theta$, reaching $X\%$ validation/test accuracy.
3. **Prune** to get $\hat\theta = \theta \odot M$. In your sketch only two connections survive.
4. **Rewind:** reset the surviving weights to **their initial values in $\theta_0$** ($\theta \to \theta_0$), keeping the mask.
5. **Train again**, and repeat. **At every loop, increase the sparsity** until the target $\alpha$ is reached (e.g. 97%).

The output of the loop is $\hat\theta$ at sparsity $\alpha$, with $Y\%$ accuracy. Usually $X\% \le Y\%$: the sparse ticket matches or beats the dense network, and only in rare cases is $X > Y$. The **lottery ticket** is the pair $(\theta_0, M)$: the initial values and the mask together. The rewinding step is essential: the same mask with fresh random weights does not train as well, so the "winning ticket" is a lucky combination of structure and starting values.

**The catch-22.** Finding the ticket requires training the dense network first:

$$\theta_{\text{Dense}} \longrightarrow \boxed{\text{Algorithm}} \longrightarrow \{\theta_0, M\}$$

Could an algorithm find the ticket in polynomial time *without* training the dense network? Your notes record that **none exists**, while the existence of tickets has been **shown empirically on every network** tried. So the hypothesis explains why large networks train well (they contain many lucky sub-networks), but it does not yet give a cheaper way to train.

Your notes add that it was the **first result that let people think about mechanistic interpretability**: if a small sub-network does the work, one can ask what that sub-network computes.

---

## 4. Quantization

Pruning cut the number of non-zero parameters. **Quantization**, the second method, cuts the other factor of the rate ([[#2.1 Rate and distortion|§2.1]]): the bits per parameter.

### 4.1 Mapping reals to integers

Quantization replaces real numbers (32- or 16-bit floats) with integers of $b$ bits. It defines a mapping $T$ from the real line to the integers that sends each point $A, B, C$ to $A', B', C'$:

![[03 - Attachments/ATML_L13_quant_map.svg|560]]

*(a) An affine map keeps ratios of distances. (b) The real range $[x_{\min}, x_{\max}]$ mapped onto $2^b$ integer levels, spaced by the scale $s$.*

$T$ is an **affine transformation** (a scaling plus a shift), which preserves ratios of distances along the line:

$$\frac{CA}{BA} = \frac{C'A'}{B'A'}$$

Using $T$ one defines **linear quantization** $\mathbb R \to \mathbb Z$, real to integer. Your notes' example sends $-3.82 \mapsto -2$.

> [!note] The $-3.82$ example
> Plain rounding would give $-4$. The value $-2$ comes from first dividing by a scale, e.g. $s = 2$: $\text{round}(-3.82/2) = \text{round}(-1.91) = -2$. [[#4.2 Scale, zero-point and levels|§4.2]] makes the scale explicit.

### 4.2 Scale, zero-point and levels

The real values $x \in \mathbb R$ are **given**: weights, activations, neuron biases, etc. They range from $x_{\min}$ to $x_{\max}$. The integers $q \in \mathbb Z$, from $q_{\min}$ to $q_{\max}$, are **unknown and have to be found**.

**Levels.** The number of integer levels depends on how many bits $b$ are used: $2^b$ levels. The range depends on the integer type:

| Type | Range | $b = 8$ |
|---|---|---|
| signed | $[-2^{b-1},\ 2^{b-1} - 1]$ | int8: $[-128, 127]$ |
| unsigned | $[0,\ 2^b - 1]$ | uint8: $[0, 255]$ |

Your notes write int8 as $[-128, 128]$; the top value is $127$, since $-128$ to $127$ is exactly $2^8 = 256$ levels.

**Quantizing.** Each real $x$ is mapped to

$$\boxed{q = \text{round}\Big(\frac{x}{s}\Big) + z} \qquad\qquad \text{(3)}$$
^eq-3

with two parameters to find:
- the **scale** $s \in \mathbb R$, the step size: one integer step covers $s$ in real units;
- the **zero-point** (zero-offset) $z \in \mathbb Z$, the integer that the real 0 maps to. It lets the integer range be placed anywhere relative to 0.

*(Expanding on the reverse map and clamping:)* the real value is recovered approximately by **dequantization**, $\hat x = s(q - z)$, and the error $x - \hat x$ is at most $s/2$ for values inside the range. Values outside $[x_{\min}, x_{\max}]$ are **clamped** to $q_{\min}$ or $q_{\max}$. A smaller $s$ means finer steps but a narrower range, so choosing the range is a trade-off between rounding error and clipping error.

The design choices are the number of bits and how to choose $(s, z)$. There are two standard ways to choose them.

### 4.3 Asymmetric quantization

**Asymmetric** quantization is used for **activations**, the outputs of neurons. Activations are often lopsided around 0: after a leaky ReLU, for example, the range has a long positive side and only a short negative one. The zero-point gives the flexibility to decide how many of the integer levels go to negatives.

![[03 - Attachments/ATML_L13_sym_asym.svg|600]]

*(a) A leaky-ReLU output has a lopsided range, so 0 maps to some integer $z$ near one end. (b) Weights are roughly centred, so the range is made symmetric and $z = 0$.*

Choose $s$ and $z$ so that the ends of the real range land exactly on the ends of the integer range. *(The derivation, which your notes ask for in full:)* without the rounding, [[#^eq-3|eq. (3)]] is the affine map $q = x/s + z$. Requiring $x_{\min} \mapsto q_{\min}$ and $x_{\max} \mapsto q_{\max}$ gives two equations:

$$q_{\min} = \frac{x_{\min}}{s} + z, \qquad q_{\max} = \frac{x_{\max}}{s} + z$$

Subtracting the first from the second eliminates $z$: $q_{\max} - q_{\min} = (x_{\max} - x_{\min})/s$, so

$$\boxed{s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}}$$

Substituting $s$ back into the first equation and rounding, since $z$ must be an integer:

$$\boxed{z = \text{round}\Big(q_{\min} - \frac{x_{\min}}{s}\Big)}$$

Rounding $z$ is what makes the real 0 map to an exact integer, so zeros (e.g. ReLU outputs, padding) are represented with no error.

### 4.4 Symmetric quantization

**Symmetric** quantization is used for **weights**, which are roughly centred on 0. It fixes the zero-point at 0 and makes the range symmetric:

$$\boxed{z = 0, \qquad s = \frac{\max\big(\lvert x_{\min}\rvert, \lvert x_{\max}\rvert\big)}{q_{\max}}}$$

The larger of the two ends sets the scale, so the whole range $[-\max\lvert x\rvert, \max\lvert x\rvert]$ fits. The cost is that if the values are lopsided, part of the integer range on the shorter side is never used ([[#Numerical Problems|Problem 7]]). The benefit is that $z = 0$ removes the zero-point terms from the integer arithmetic of [[#4.5 What quantization touches, and integer arithmetic|§4.5]], which makes weight matrix multiplication cheaper.

| | Asymmetric ([[#4.3 Asymmetric quantization\|§4.3]]) | Symmetric ([[#4.4 Symmetric quantization\|§4.4]]) |
|---|---|---|
| Used for | activations | weights |
| Zero-point | $z = \text{round}(q_{\min} - x_{\min}/s)$ | $z = 0$ |
| Scale | $(x_{\max} - x_{\min})/(q_{\max} - q_{\min})$ | $\max(\lvert x_{\min}\rvert, \lvert x_{\max}\rvert)/q_{\max}$ |
| Fits lopsided ranges | yes | wastes levels |

### 4.5 What quantization touches, and integer arithmetic

The two methods differ in scope. **Pruning only touches the weights. Quantization touches the activations as well.** That is why [[#4.3 Asymmetric quantization|§4.3]] and [[#4.4 Symmetric quantization|§4.4]] need two rules.

Quantizing both is what allows **integer arithmetic**: nothing has to be done in floating point, and the only arithmetic is the matrix multiplication, done on integers. *(Expanding on why the scales can be pulled out:)* with symmetric weights ($z_w = 0$) and, for simplicity, a zero-point-free activation, $W \approx s_w Q_w$ and $x \approx s_x q_x$, so

$$Wx \approx s_w s_x\,(Q_w q_x)$$

The product $Q_wq_x$ is an integer matrix multiplication, which hardware does much faster and with less energy than floating point, and the two scales are applied once at the end. A non-zero activation zero-point adds a correction term that can be precomputed.

### 4.6 Quantize and dequantize along the network

Lecture 22 restates the goal before going further: **pruning reduces the number of parameters, quantization reduces the number of bits per parameter**, for example FP32 → INT8. Both shrink the rate of [[#2.1 Rate and distortion|§2.1]], which your notes now read as a cost:

$$\text{Rate} \;\approx\; \text{compute cost} \;\approx\; (\#\text{ params}) \times (\#\text{ bits per param}).$$

The task is to **minimize the rate given a level of distortion**. This is the constrained form of [[#^eq-1|eq. (1)]]: fix how much distortion you will accept and make the rate as small as possible; $\beta$ in eq. (1) is the Lagrange multiplier that turns this constraint into a penalty.

Quantization is then two operations, labelled (A) and (B) in your notes:
- **(A) Quantize:** find $(s, z)$ such that $q = \text{round}(x/s) + z$, [[#^eq-3|eq. (3)]]. This happens in the forward pass and is what makes it fast.
- **(B) Dequantize:** given $q$, $s$ and $z$, recover $\hat x = s(q - z)$ ([[#4.2 Scale, zero-point and levels|§4.2]]).

Both apply to every layer: **we quantize both weights and activations** ([[#4.5 What quantization touches, and integer arithmetic|§4.5]]). The number-line picture of (A) and (B), with $2^b$ levels between $q_{\min}$ and $q_{\max}$, is [[#4.1 Mapping reals to integers|§4.1]]'s figure, and the asymmetric and symmetric ranges are [[#4.3 Asymmetric quantization|§4.3]]–[[#4.4 Symmetric quantization|§4.4]].

![[03 - Attachments/ATML_L13_quant_pipeline.svg|760]]

*Integer weights and activations flow through the layers (A); before the LM head they are dequantized back to real numbers (B), so the softmax works on real logits. Different attention heads may use different bit-widths.*

**Why dequantize at all?** Your notes answer: we have to dequantize in the **LM head**, because "we don't want quantization to meddle with the affair of softmax". *(Expanding on this:)* the softmax exponentiates the logits, so it is sensitive to their exact differences and scale; integers that still carry an unknown $s$ and $z$ are not logits. Dequantizing restores real-valued logits, so the output probabilities and the loss are computed in floating point.

**Granularity.** In an attention head the weights may go from FP16 to INT4, while another head might use a different bit-width; the choice can be made at the granularity of individual heads. *(Expanding on this:)* the same idea applies to the quantization parameters: one $(s, z)$ per tensor, per output channel or per small group of weights. Finer granularity costs a few extra stored scales but lets each group fit its own range, which matters for [[#4.9 When quantization struggles — the outlier problem|§4.9]].

### 4.7 When to quantize — QAT and the straight-through estimator

Quantization can happen during training or after it, the same timing question [[#3.6 When to prune|§3.6]] asked for pruning. The first option is **QAT**, **Quantization-Aware Training** (your Notability notes write "Quantization Aware Tuning"; both names are used, since QAT is often a fine-tuning phase): the model is trained with quantization in the forward pass, so the weights learn to tolerate it.

**The problem.** Take the activations $H$ of some layer, quantize them to $\hat H$, and continue the forward pass to the loss:

![[03 - Attachments/ATML_L13_qat_ste.svg|720]]

*Forward: $H$ is rounded by the staircase quantizer to $\hat H$ and passed on to the loss. Backward: the staircase has zero slope, so the straight-through estimator (STE) replaces it with the identity and passes $\partial L/\partial \hat H$ straight to $H$.*

In your example $H = \begin{bmatrix}1.1 & 2.2\\ -1.7 & 3.6\end{bmatrix}$ becomes $\hat H = \begin{bmatrix}1 & 2\\ -2 & 2\end{bmatrix}$. The first three entries are plain rounding with $s = 1$; $3.6 \mapsto 2$ means the range is also clipped at $2$ (round to 4, then clamp to $q_{\max} = 2$).

To reach the gradient for $H$, and through it every earlier parameter, backpropagation needs the derivative of the quantizer. **The derivative does not work** at a staircase: it is 0 on every flat step and undefined at every jump, so the chain rule would pass back zero everywhere and nothing before the quantizer would learn.

The fix is the **straight-through estimator (STE)**: in the backward pass, pretend the quantizer was the identity,

$$\frac{\partial \hat H}{\partial H} := 1 \quad\Longrightarrow\quad \frac{\partial L}{\partial H} = \frac{\partial L}{\partial \hat H} \qquad\qquad\text{(4)}$$
^eq-4

so the gradient passes "straight through" the rounding. The forward pass still uses $\hat H$, so the loss sees the quantization error and training adapts to it. *(Expanding on this: the STE is from Bengio et al., 2013. A common variant passes gradient 1 only inside $[x_{\min}, x_{\max}]$ and 0 for clipped values such as the 3.6, since moving them slightly would not change $\hat H$.)*

### 4.8 Post-training quantization and activation awareness

The second option is **PTQ**, **Post-Training Quantization**: train in full precision, then quantize the finished model, usually with a small calibration set to measure the activation ranges. It needs no retraining, which is why it is the usual choice for large LLMs, at the cost of more distortion at low bit-widths than QAT.

Your notes flag **activation-aware quantization** under PTQ to read about. *(Expanding on this:)* the best-known method is **AWQ** (Activation-aware Weight Quantization; Lin et al., 2024). It observes that a small fraction of weight channels, about 1%, matter most, and that they are found by the size of their **input activations**, not of the weights themselves. AWQ scales those channels up before quantizing so they lose less precision. This is the same insight as Wanda in pruning ([[#3.5 Activation-aware pruning (Wanda)|§3.5]]): a weight's importance is its size times the size of what it multiplies.

### 4.9 When quantization struggles — the outlier problem

Quantization targets bits per parameter. When will it fail, or struggle? Take the weights of one layer:

![[03 - Attachments/ATML_L13_outlier.svg|720]]

*One weight of $10^6$ among values in $[-1, 4]$ sets $x_{\max}$, so a single quantization step spans about $10^6/(2^b-1)$ and the other eleven weights all fall into the first step.*

The **scale would change**: $s$ is computed from $x_{\max}$ ([[#4.3 Asymmetric quantization|§4.3]]), so the single value $10^6$ stretches the step size until it **dominates all levels**. Most of the inputs sit in a tiny region near the bottom of the range, while $x_{\max}$ comes directly from the outlier. This is the **outlier problem**.

Your paper notes ask whether the outlier affects $z$ as well as $s$. It does for asymmetric quantization, because $z = \text{round}(q_{\min} - x_{\min}/s)$ depends on $s$: a huge $s$ shrinks $x_{\min}/s$ towards 0, so $z$ collapses towards $q_{\min}$ ([[#Numerical Problems|Problem 13]] works it through: $z$ goes from 51 to 0). For symmetric quantization $z = 0$ by design, so only $s$ is affected, but that is enough: with $s = 10^6/127$ every other weight rounds to $q = 0$.

*(Expanding on the usual fixes:)* clip the range below the outlier and accept a large error on that one value; keep outliers in higher precision (LLM.int8() keeps outlier feature dimensions in FP16); use finer granularity ([[#4.6 Quantize and dequantize along the network|§4.6]]) so the outlier only spoils its own group; or move the outlier's magnitude into the other operand before quantizing, as AWQ ([[#4.8 Post-training quantization and activation awareness|§4.8]]) and SmoothQuant do.

### 4.10 Mixed precision, combining methods, and architecture search

**Mixed precision.** We cannot use INT4 for all layers, as they have different sensitivities: some layers tolerate 4 bits, others lose too much. How to match the bits to the layers is, in your notes' words, an **open problem**, an active research question.

**Quantization vs pruning.** Your notes record the class's view that **quantization is much, much better than pruning**. *(Expanding on why:)* quantization keeps the matrix dense, so it maps directly onto integer hardware and gives a real speed-up and memory cut (8-bit is a 4× cut from FP32, often with little loss), while unstructured pruning rarely speeds anything up ([[#3.3 How to group weights|§3.3]]).

**Why not both?** Your paper notes ask why we can't use both pruning and quantization at a time. We can, and they cut different factors of the rate ([[#Numerical Problems|Problem 8]]); *Deep Compression* (Han et al., 2016) pruned, quantized and then entropy-coded weights. The difficulty is choosing the combination: for every layer or head, whether to prune, how much, and how many bits. The joint search space is huge.

That search is what **Network Architecture Search (NAS)** automates. *(The standard name is Neural Architecture Search.)* It searches over everything: bits per head, pruning or not. Your notes add to read about **NAS engines**, and that some models are deployed only with it on **edge devices** (phones, embedded boards) whose memory and latency budgets force such searches. *(Expanding on this:)* hardware-aware NAS adds the device's measured latency or memory to the search objective, which is [[#^eq-1|eq. (1)]] with rate measured on the actual hardware.

---

## 5. Low-Rank Approximation

Pruning and quantization keep $W$'s shape and change its entries. The third compression technique, which Lecture 22 gives a quick mention, replaces the matrix itself with something smaller.

### 5.1 Replacing W by two thin matrices

Start from the LoRA picture of [[ATML_Lecture_10 - Fine-Tuning#8.1 The low-rank update|Lecture 10 §8.1]]: a frozen $W \in \mathbb R^{m\times n}$ with a parallel low-rank branch $\Delta W = BA$, $A \in \mathbb R^{r\times n}$, $B \in \mathbb R^{m\times r}$. For compression, **drop $W$** (remove it) and keep only the branch, so the layer computes $x \mapsto BAx$.

![[03 - Attachments/ATML_L13_lowrank.svg|720]]

*(a) Start from the LoRA layer and remove $W$. (b) What remains is $x \to A \to B$, a rank-$r$ layer. (c) Two rank-$r$ branches combined with a Hadamard (element-wise) product reach rank up to $r^2$.*

- **Rate:** $mn$ parameters become $r(m + n)$.
- **Capacity:** $r$ represents the capacity of the (low-rank) model: $\text{rank}(BA) \le r$. As $r$ goes up, memory goes up and capacity goes up.
- **Cost:** degradation, the distortion of [[#2.1 Rate and distortion|§2.1]], since $BA$ can only approximate $W$. *(Expanding on the best choice:)* for a fixed $r$, the approximation with the smallest $\lVert W - \hat W\rVert_F$ keeps the top $r$ singular values of $W$ (the Eckart–Young theorem), so a truncated SVD gives the starting $A$ and $B$.

> [!warning] Possible source inconsistency
> Your Notability notes write "increase rate at cost of distortion". Low-rank approximation **lowers** the rate (fewer parameters) and pays in distortion; the intended meaning is probably "increase compression".

### 5.2 More rank from the same parameters — the Hadamard product

Digging deeper into linear algebra gives a theorem:

> **The Hadamard product of two matrices of rank $\bar r$ has rank at most $\bar r^2$.**

*(The proof, which the notes state without:)* write $X = \sum_{a=1}^{\bar r} u_a v_a^\top$ and $Y = \sum_{b=1}^{\bar r} s_b t_b^\top$ as sums of rank-one terms. The Hadamard product multiplies entry by entry, and $(u v^\top)\odot(s t^\top) = (u\odot s)(v\odot t)^\top$, so

$$X \odot Y = \sum_{a=1}^{\bar r}\sum_{b=1}^{\bar r} (u_a\odot s_b)(v_a\odot t_b)^\top,$$

a sum of $\bar r^2$ rank-one matrices, hence rank at most $\bar r^2$. It is an upper bound; for generic matrices it is usually reached when $\bar r^2 \le \min(m, n)$.

How to combine this with $(B_1A_1)\odot(B_2A_2)$? Instead of one pair of matrices, use two rank-$r$ branches and multiply their outputs element-wise, $\hat W = (B_1A_1)\odot(B_2A_2)$ (panel (c)). This answers your notes' question, **how to get more rank with fewer parameters**:

| | Parameters | Maximum rank |
|---|---|---|
| One branch, rank $2r$ | $2r(m + n)$ | $2r$ |
| Two rank-$r$ branches, Hadamard | $2r(m + n)$ | $r^2$ |

For $r = 8$ the same budget gives rank 64 instead of 16. *(Expanding on the name: this is the low-rank Hadamard product of FedPara (Hyeon-Woo et al., 2022), also used as "LoHa" for fine-tuning.)*

---

## 6. Knowledge Distillation

Low-rank approximation still works inside the original architecture. Knowledge distillation drops that constraint: it trains a separate, much smaller network to behave like the large one. Your notes call it a fascinating new subject.

### 6.1 Teacher, student and the undistilled baseline

Take an image input $x$ for image classification. The **teacher** $T$ is a much bigger model (billions of parameters); the **student** $S$ is a much smaller one (thousands). Each produces logits, which a softmax (SM) with **temperature** $\rho$ turns into probabilities:

$$p_i = \frac{e^{v_i/\rho}}{\sum_j e^{v_j/\rho}} \ \text{(teacher, logits } v_i), \qquad q_i = \frac{e^{z_i/\rho}}{\sum_j e^{z_j/\rho}} \ \text{(student, logits } z_i).$$

$q_i$ says which class the student predicts. **$T$ remains frozen; $S$ is trained and used at inference.** That is the compression: the deployed model is $S$.

![[03 - Attachments/ATML_L13_kd_setup.svg|720]]

*The frozen teacher and the trained student see the same $x$; their temperature softmaxes give $p_i$ and $q_i$, compared by $L_{\text{DIST}}$, while the student also fits the true label through $L_{\text{CE}}$. Bottom: the undistilled baseline trains $S$ on $L_{\text{CE}}$ alone.*

The **undistilled student** is the baseline: the same small network trained on its own with the usual cross-entropy $L_{\text{CE}}$ against the labels. Distillation has to beat it to be worth anything.

### 6.2 Logit matching and the combined loss

**Logit matching** uses the KL divergence between the teacher's and the student's probabilities ([[ATML_Lecture_2 - Information Theory and Divergences#5.2 Forward KL Divergence|Lecture 2 §5.2]]), and trains the student on a combined loss:

$$\boxed{L_T = \lambda\,L_{\text{CE}} + (1-\lambda)\,L_{\text{DIST}}}, \qquad L_{\text{DIST}} = \rho^2\,\mathrm{KL}(p\,\Vert\,q) \qquad\qquad\text{(5)}$$
^eq-5

$\lambda \in [0, 1]$ controls how much the student relies on the true labels versus on the teacher's distribution.

> [!warning] Possible source inconsistency
> Both sets of notes write $L_{\text{DIST}} = -\rho^2\,\mathrm{KL}(p\Vert q)$. With the minus sign, minimizing the loss would push the student *away* from the teacher. The minus belongs to the cross-entropy form, $-\rho^2\sum_i p_i\log q_i$, which differs from $\rho^2\,\mathrm{KL}(p\Vert q)$ only by the teacher's entropy, a constant for the student; the two give the same gradients.

### 6.3 Why the ρ² factor — distillation becomes logit MSE

Your notes ask why the $\rho^2$, answer with $\partial L_{\text{DIST}}/\partial z_i \approx (z_i - v_i)$, and then ask what loss would have this derivative. *(The derivation below fills in the steps the notes skip.)*

The derivative of the cross-entropy $-\sum_j p_j \log q_j$ with respect to a student logit is $\tfrac1\rho(q_i - p_i)$: the usual softmax gradient, divided by $\rho$ because the logits enter as $z_i/\rho$. The teacher's entropy does not depend on $z$, so with the $\rho^2$ factor

$$\frac{\partial L_{\text{DIST}}}{\partial z_i} = \rho\,(q_i - p_i).$$

At high temperature $e^{z_i/\rho} \approx 1 + z_i/\rho$, so for $K$ classes and logits with zero mean (subtracting a constant from all logits does not change a softmax), $q_i \approx \tfrac1K(1 + z_i/\rho)$ and $p_i \approx \tfrac1K(1 + v_i/\rho)$. Substituting,

$$\frac{\partial L_{\text{DIST}}}{\partial z_i} \approx \rho\cdot\frac{z_i - v_i}{K\rho} = \frac{1}{K}(z_i - v_i).$$

This is the gradient of $\frac{1}{2K}\sum_i (z_i - v_i)^2$, the **MSE between logits**: at high temperature distillation **is logit matching**, which gives the method its name. The $\rho^2$ is there because without it the gradient would scale as $1/\rho^2$ and vanish as $\rho$ grows; with it, the distillation term keeps the same weight relative to $L_{\text{CE}}$ whatever temperature is chosen. *(These are Hinton et al.'s 2015 arguments.)*

### 6.4 Hard labels vs soft labels

Is there a better way to train the student than its own labels? Take three classes: $C_1$ cat $= [1, 0, 0]$, $C_2$ dog $= [0, 1, 0]$, and $C_3$ fox $= [0, 0, 1]$, where a fox looks similar to both a cat and a dog.

Give a cat image to both $T$ and $S$. The **hard label** (one-hot, hard-coded) puts 1 on cat and 0 elsewhere, and an undistilled student is trained on exactly that distribution. Nothing in it says that a dog is more similar to a cat than to a fox: every wrong class is equally wrong. The teacher's output after the softmax is a **soft label**: for the cat image, $(0.8, 0.1, 0.1)$; for a dog image, $(0.2, 0.6, 0.2)$ in your Notability notes or $(0.1, 0.6, 0.3)$ in the paper notes. Those small numbers carry the structure: which classes the teacher thinks are close. For the fox, whose arrow your notes leave blank, a good teacher would put noticeable mass on both cat and dog.

![[03 - Attachments/ATML_L13_kd_labels.svg|780]]

*(a) A hard label says nothing about similarity. (b) The teacher's soft labels for a cat and a dog image. (c) A strict teacher is nearly one-hot; a lenient one spreads probability. (d) Label smoothing moves $\varepsilon$ from the top class to the others.*

Do we need to make the soft labels equal to the hard labels? No: if the teacher output a one-hot vector, the KL term would **collapse** into the cross-entropy against the label (the teacher's entropy is 0), so it would carry **no information in terms of information theory** beyond what the label already gives ([[ATML_Lecture_2 - Information Theory and Divergences#3.3 Entropy as Average Information Content|Lecture 2 §3.3]]). **We want soft labels and a lenient teacher, because they provide structural similarity between classes.**

### 6.5 Strict vs lenient teacher

A **strict teacher** is nearly one-hot: a sharp peak, entropy close to 0 (your paper notes: $T_1$ giving $0.8$ to one class). A **lenient teacher** spreads its probability: a wider curve, higher entropy ($T_2$ giving about $0.5$ to the top class). The lenient teacher is **preferred**, because it teaches **inter-class similarity**: it provides more information on the structure between classes.

There is a limit. A **very lenient** teacher outputs a pure uniform distribution, which is **bad**: uniform probabilities say every class is equally likely and carry no information either. The useful teacher sits between the two extremes.

### 6.6 Shaping the teacher's distribution — temperature and label smoothing

How do we make the teacher lenient? Two ways from class, out of hundreds of techniques:

1. **The temperature $\rho$** of the softmax ([[#6.1 Teacher, student and the undistilled baseline|§6.1]]). A larger $\rho$ divides all logits by more, shrinking their differences, so the distribution flattens; $\rho \to \infty$ gives the uniform distribution and $\rho \to 0$ the one-hot argmax. *(Your paper notes place "using SM temperature of student" next to the curves; in standard distillation the same $\rho$ is applied to both teacher and student, as in [[#^eq-5|eq. (5)]].)*
2. **LSR**, the **Label Smoothing Regularizer**. Take the teacher's distribution, e.g. $(0.05, 0.15, 0.8)$, take $\varepsilon$ off the top class, leaving $0.8 - \varepsilon$, and spread it over the others. *(Expanding on the standard form:)* label smoothing mixes a distribution $y$ with the uniform one, $y_{\text{LS}} = (1-\varepsilon)\,y + \varepsilon/K$.

> [!note] Unclear in source
> Your Notability notes end this list with "get rid of KL div." Read here as a third direction: some methods replace the KL divergence altogether. [[#6.8 Decoupled knowledge distillation (DKD)|§6.8]] keeps KL but splits it in two, which follows from the problem in [[#6.7 The accuracy gap and dark knowledge|§6.7]].

### 6.7 The accuracy gap and dark knowledge

Compare the accuracies. The teacher is best, the distilled student is next, and the undistilled student is lowest: distillation helps, but **there is always a gap between the teacher's accuracy and what the student attains**.

![[03 - Attachments/ATML_L13_kd_gap.svg|720]]

*(a) Teacher > distilled student > undistilled student. (b) A student that matches only the teacher's main mode ignores the smaller bump, the teacher's dark knowledge.*

Your notes give two reasons:
1. **Capacity.** The student has far fewer parameters, so the capacity it shares with the teacher is limited; some of what the teacher represents simply does not fit.
2. **The KL divergence is mode-seeking.** KL focuses on the region of the main mode and ignores the rest; what it ignores is **dark knowledge** (DK), the information in the teacher's small probabilities on the non-target classes. The teacher might not be able to teach the stripes of a tiger: the small probability it gives "tiger" for a cat image is exactly the kind of detail that gets lost.

> [!warning] Possible source inconsistency
> In [[ATML_Lecture_2 - Information Theory and Divergences#5.3 Forward KL Is Mode-Covering (Support-Seeking)|Lecture 2 §5.3]], the forward $\mathrm{KL}(p\Vert q)$ used in [[#^eq-5|eq. (5)]] is **mode-covering**; the reverse $\mathrm{KL}(q\Vert p)$ is the mode-seeking one ([[ATML_Lecture_2 - Information Theory and Divergences#5.4 Reverse KL Divergence Is Mode-Seeking|§5.4]]). The effect your notes describe is real, but it comes from a different mechanism: when the teacher is confident, the KL term is dominated by the target class and the non-target part is down-weighted. [[#6.8 Decoupled knowledge distillation (DKD)|§6.8]] makes this exact. The Lecture 23 recap ([[#6.9 Recap — what soft labels add, and why forward KL averages|§6.9]]) states the standard property: forward KL makes the student mode-covering.

How do we identify dark knowledge and teach it to the student? There are many techniques; the class focuses on one.

### 6.8 Decoupled knowledge distillation (DKD)

**Decoupled Knowledge Distillation** (DKD; Zhao et al., 2022) **decouples the mode of the distribution from the rest of the distribution**. If we do not, KL focuses on matching the mode. So separate the bars: one KL for the **target class** against everything else, and one for how the **non-target classes** are distributed among themselves.

The problem is that the separated pieces **don't sum to 1**, so they are not distributions and KL cannot be applied to them directly. The fix is to **introduce hypothetical classes**:
- **TCKD** (Target-Class KD) merges all non-target classes into one hypothetical class "rest", giving a two-class distribution $b = (p_t,\ 1 - p_t)$. In your paper example the teacher gives $(0.8, 0.2)$ and the student $(0.7, 0.3)$.
- **NCKD** (Non-target-Class KD) renormalizes the non-target probabilities among themselves, $\hat p_i = p_i/(1 - p_t)$ for $i \ne t$, so they sum to 1 again.

![[03 - Attachments/ATML_L13_dkd.svg|720]]

*Teacher and student outputs (target class in orange) are split into a binary target-vs-rest distribution (TCKD) and a renormalized distribution over the non-target classes (NCKD).*

*(The decomposition, which shows why plain KD suppresses dark knowledge:)* since $p_i = (1 - p_t)\hat p_i$ for $i \ne t$ (and the same for $q$),

$$\mathrm{KL}(p\Vert q) = p_t\log\frac{p_t}{q_t} + \sum_{i\ne t}(1-p_t)\hat p_i\left[\log\frac{1-p_t}{1-q_t} + \log\frac{\hat p_i}{\hat q_i}\right],$$

and $\sum_{i\ne t}\hat p_i = 1$ collapses the middle term, giving

$$\boxed{\mathrm{KL}(p\Vert q) = \underbrace{\mathrm{KL}(b^T\Vert b^S)}_{\text{TCKD}} + (1 - p_t)\,\underbrace{\mathrm{KL}(\hat p\Vert \hat q)}_{\text{NCKD}}} \qquad\qquad\text{(6)}$$
^eq-6

The non-target part is multiplied by $1 - p_t$. A confident teacher ($p_t \to 1$) therefore switches NCKD, the dark knowledge, almost off: that is the "focus on the mode" your notes describe. DKD removes the coupling by weighting the two terms independently, $L_{\text{DKD}} = \gamma_{\text{TC}}\,\mathrm{TCKD} + \gamma_{\text{NC}}\,\mathrm{NCKD}$, with $\gamma_{\text{NC}}$ no longer tied to $p_t$. *(The weights are called $\alpha$ and $\beta$ in the DKD paper; renamed here because $\alpha$ is sparsity in [[#3.1 Masks, sparsity and density|§3.1]].)*

### 6.9 Recap — what soft labels add, and why forward KL averages

Lecture 23 opened by restating the setup of [[#6.1 Teacher, student and the undistilled baseline|§6.1]]–[[#6.4 Hard labels vs soft labels|§6.4]], with $i$ the label of the true class. Matching the teacher's and student's probabilities $p_i$ and $q_i$ as **soft labels**, with $D_{\mathrm{KL}}(p\Vert q)$, is equivalent at high temperature to **matching logits**, $\text{MSE}(z_i, v_i)$ ([[#6.3 Why the ρ² factor — distillation becomes logit MSE|§6.3]]). The loss is your friend's $L_T = \alpha L_{\text{CE}} + (1-\alpha)L_{\text{KD}}$, which is [[#^eq-5|eq. (5)]] with $\lambda$ written $\alpha$ and $L_{\text{DIST}}$ written $L_{\text{KD}}$; your notes drop the $\alpha$ on the CE term. The two terms carry different information:
- $L_{\text{CE}}$ uses **hard labels** (one-hot encoded), which only say that $x \in$ class $i$.
- $L_{\text{KD}}$ uses the teacher's **soft labels**, a full distribution, which also says how similar $x$ is to every other class: $\text{similarity}(x, c_j)$ for $j \ne i$.

The second plot in your notes is that extra information: a distribution peaked at the target $i$ with mass spread over the other classes, the **dark knowledge** of [[#6.7 The accuracy gap and dark knowledge|§6.7]]. The **capacity gap** of §6.7 is restated as: the student has fewer parameters than the teacher.

**The problem with the original KD.** The divergence is the **forward** KL. If the teacher's distribution has **two modes**, a student that cannot represent both is pushed to be **mode-covering**: it spreads over both modes and puts its peak on the **average**, between them, where the teacher has little mass. This is the mode-covering behaviour of forward KL from [[ATML_Lecture_2 - Information Theory and Divergences#5.3 Forward KL Is Mode-Covering (Support-Seeking)|Lecture 2 §5.3]]. It settles the inconsistency flagged in §6.7: the class's recap uses the standard property.

![[03 - Attachments/ATML_L13_forward_kl_avg.svg|700]]

*(a) Forward $\mathrm{KL}(p\Vert q)$ heavily penalizes $q \approx 0$ wherever $p > 0$, so a single-mode student must cover both teacher modes and centres itself on their average. (b) Dark knowledge: besides the target $i$, the soft label has mass on every class $j \ne i$, which encodes $\text{similarity}(x, c_j)$.*

### 6.10 Gradient conflict between the CE and KD terms

DKD ([[#6.8 Decoupled knowledge distillation (DKD)|§6.8]]) **decouples the weights**: instead of $\alpha$ and $1 - \alpha$, the two terms get independent weights, which your notes write as $\alpha$ and $\beta$ (here $\lambda$ and $\mu$, since $\alpha$ is sparsity in §3.1):

$$L_T = \lambda L_{\text{CE}} + \mu L_{\text{KD}}, \qquad \nabla L_T = \lambda\,\nabla L_{\text{CE}} + \mu\,\nabla L_{\text{KD}}.$$

The issue is in the gradient: **the two gradients can cancel each other out.** The class measures this **conflict** by the cosine similarity of the two gradients,

$$\text{conflict} = \cos(\nabla L_{\text{CE}}, \nabla L_{\text{KD}}) = \frac{\nabla L_{\text{CE}}\cdot\nabla L_{\text{KD}}}{\lVert\nabla L_{\text{CE}}\rVert\,\lVert\nabla L_{\text{KD}}\rVert}, \qquad\qquad\text{(7)}$$
^eq-7

and in practice it is found to be **$\approx 0$ or $< 0$**: the gradients are orthogonal or point against each other.

**Why they conflict.** The one-hot label has **0 in the non-target classes; the soft labels don't.** *(The derivation below makes this concrete.)* With $\rho = 1$, the gradient of each term with respect to the student's logits is the familiar softmax gradient: $\partial L_{\text{CE}}/\partial z = q - y$ and $\partial L_{\text{KD}}/\partial z = q - p$. On a non-target class $j$, CE's gradient is $q_j > 0$, so gradient descent lowers $z_j$ towards probability 0. KD's gradient is $q_j - p_j$, which is negative whenever the teacher gives class $j$ more mass than the student, so it raises $z_j$. The two terms push the same logit in **opposite directions** on every such class. On the target class, CE always raises $z_t$ ($q_t - 1 < 0$), while KD lowers it once the student is more confident than the teacher ($q_t > p_t$). A student sitting between the one-hot label and the teacher is pulled both ways at once. (The cosine is usually measured over the parameter gradients; these are the logit gradients mapped back through the network, so the conflict carries over.)

**What it costs.** The update is $\theta_S^{(i+1)} \leftarrow \theta_S^{(i)} - \eta\,\nabla\text{loss}$. When $\nabla_{\text{KD}}$ and $\nabla_{\text{CE}}$ point apart, their sum is short (the red arrow in your notes), so each step makes little progress: **convergence becomes slow**. [[#Numerical Problems|Problem 18]] works an example with cosine $-0.97$.

### 6.11 Resolving the conflict — gradient harmonization and momentum

How to resolve the conflict? The class gave two approaches.

**① Gradient harmonization with the Gram–Schmidt procedure.** Make the cosine similarity **at least $\approx 0$**. When $g_{\text{KD}}\cdot g_{\text{CE}} < 0$, remove from $g_{\text{KD}}$ its component along $g_{\text{CE}}$, one step of Gram–Schmidt orthogonalization (your friend's sketch marks the result "ortho"):

$$g_{\text{KD}}^{\perp} = g_{\text{KD}} - \frac{g_{\text{KD}}\cdot g_{\text{CE}}}{\lVert g_{\text{CE}}\rVert^2}\,g_{\text{CE}}, \qquad g_{\text{KD}}^{\perp}\cdot g_{\text{CE}} = 0. \qquad\qquad\text{(8)}$$
^eq-8

The update then uses $\lambda g_{\text{CE}} + \mu g_{\text{KD}}^{\perp}$: the distillation signal no longer undoes the task signal, and what remains of it points in a direction the CE term does not care about. *(Expanding on this: the same projection is "gradient surgery", PCGrad (Yu et al., 2020), used for conflicting tasks in multi-task learning; when the cosine is already $\ge 0$, nothing is changed.)* [[#Numerical Problems|Problem 19]] applies it.

**② Maintain an average velocity (momentum).** Instead of following each step's gradient, keep a running average: put a **low-pass filter over a moving window** of gradients. The **exponential moving average (EMA)**

$$v_t = \beta_m\,v_{t-1} + (1 - \beta_m)\,g_t, \qquad \theta_{t+1} = \theta_t - \eta\,v_t \qquad\qquad\text{(9)}$$
^eq-9

becomes the **velocity** $v_t$, and it **provides momentum**. Here $\beta_m \in [0, 1)$ is the momentum coefficient (written $\beta_m$ to keep it apart from DKD's $\beta$); the window it averages over is about $1/(1 - \beta_m)$ steps. Components of the gradient that flip direction from step to step, such as the tug of war between CE and KD, average out, while the consistent direction accumulates. Your friend's note says the EMA **counters drift**: the little zig-zag arrows become one steady arrow $v_t$. *(Expanding on this: the Distillation-Oriented Trainer, DOT (Zhao et al., 2023, from the DKD authors), uses this idea with separate momenta for the CE and KD gradients, a larger one for the KD term.)* [[#Numerical Problems|Problem 20]] shows the filtering.

![[03 - Attachments/ATML_L13_grad_conflict.svg|760]]

*(a) Conflicting gradients: the angle exceeds $90^\circ$, so their sum is short and steps are slow. (b) Gram–Schmidt removes the part of $g_{\text{KD}}$ that opposes $g_{\text{CE}}$, leaving $g_{\text{KD}}^{\perp}$ orthogonal to it. (c) Momentum: the EMA of zig-zagging gradients is a steady velocity $v_t$.*

### 6.12 Why logits lose structure — the task head

Both fixes keep distilling at the **logits**. The class then asked whether the logits are the right place at all. Follow the teacher on an image $x$:

$$x \;\to\; T \;\to\; h \in \mathbb R^{d} \;\to\; \text{task head} \;\to\; v \in \mathbb R^{K} \;\to\; p \in \mathbb R^{K}$$

$h$ is the **latent of $T$**, a $d \times 1$ vector (your notes write it $z$; renamed here because $z$ is the student's logits in §6.1). The **task head** is an MLP, so the teacher's logits are $v = \text{MLP}(h)$, a $K \times 1$ vector with $K$ the number of classes and $d \gg K$.

For the latent to be useful, we **need its dimensions to be independent of each other**: the hope is that $h_1, \dots, h_d$ hold **disentangled semantic information**. The MLP **mixes this information again** to produce the $K$ logits, connecting every latent dimension to every logit. After that mixing, the **inter-class similarity structure is destroyed**: the $K$ numbers no longer say which semantic factors made two classes similar. Recovering it from the logits is an **open problem**. It suggests distilling **before the head**, from the features themselves.

![[03 - Attachments/ATML_L13_task_head.svg|620]]

*The task head maps the $d$-dimensional latent $h$ to $K \ll d$ logits by mixing every dimension into every logit, so logit matching sees only the mixed result.*

### 6.13 FitNets — feature-level distillation

**FitNets** (Romero et al., 2015) distil at the **feature level**. Pass $x$ through both networks (drawn in your notes as stacks of Transformer blocks) and match an intermediate feature map of the teacher, $F_T$, with one of the student, $F_S$, by mean squared error, $\text{MSE}(F_T, F_S)$, in addition to the usual outputs $p$ and $q$.

**Problem: dimension mismatch.** $F_T$ is the **bigger tensor**, since the teacher is wider. So put a **regressor head** on the student to upsample its features, for example a **one-layer MLP**:

$$L_{\text{FitNets}} = \text{MSE}\big(F_T,\ \text{Reg}(F_S)\big). \qquad\qquad\text{(10)}$$
^eq-10

Your friend notes the regressor **can be upscaling or downscaling** (one could instead shrink $F_T$). The remaining problem: **we don't know the ideal regressor.** *(Expanding on why this matters: $\text{Reg}$ is trained together with the student. A powerful regressor can match $F_T$ on its own without the student's features learning anything; a weak one cannot even express the right mapping. The loss value does not say which happened.)* The next step is therefore to **distil semantics directly**, without asking features to match coordinate by coordinate.

### 6.14 Contrastive Representation Distillation (CRD)

**CRD** (Tian, Krishnan and Isola, MIT, 2020) distils the representation with a contrastive objective. $x \sim$ class $i$ goes through both networks, giving representations $h_T$ and $h_S$ (your notes: $z_T$, $z_S$). The teacher and student representations of **the same $x$** form the **positive pair**; the teacher's pairing with images **not in class $i$** gives the **negatives** $h^-$. Over a batch of $N$ (one positive and the rest negatives),

$$L_{\text{contrastive}} = \frac{\text{sim}(h_T, h_S)}{\text{sim}(h_T, h_S) + \sum_{k=2}^{N}\text{sim}(h_T, h_k^{-})}. \qquad\qquad\text{(11)}$$
^eq-11

(Your notes write the first term of the denominator as "$\,''\,$", a ditto for the numerator.) The similarity **contains a temperature**, ignored above; in the usual form $\text{sim}(a, b) = \exp\!\big(\cos(a, b)/\tau\big)$. [[#^eq-11|Eq. (11)]] is a ratio to **maximize**, close to 1 when the student's representation is near the teacher's and far from the negatives. Training minimizes its negative log, the **InfoNCE** loss of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.2 The Contrastive Pipeline and the InfoNCE Loss|Lecture 3 §3.2]]. The student is no longer asked to copy the teacher's coordinates, only to put the same images close together and different ones apart, which is the semantic structure. [[#Numerical Problems|Problem 22]] computes it at two temperatures.

The number of negatives is large: **on the order of 65K** in your notes. *(Expanding on this: CRD keeps the negatives in a memory bank of past embeddings rather than one batch; I recall its default as 16,384, so 65K may be a configuration mentioned in class. Treat the exact figure as unconfirmed.)*

**Ranking from class:** $\text{CRD} \gg \text{FitNets} > \text{logit matching}$.

![[03 - Attachments/ATML_L13_feature_kd.svg|820]]

*(a) FitNets: a regressor lifts the student's feature $F_S$ to the teacher's size and MSE matches it to $F_T$. (b) CRD: pull the teacher–student pair of the same $x$ together and push the teacher's embedding away from $N - 1$ negatives of other classes. (c) Adversarial distillation: a discriminator tries to tell teacher from student representations; when it is fooled ($D = 0.5$) the two match.*

### 6.15 Adversarial distillation and other divergences

**Adversarial distillation** brings in a **discriminator** "Disc" that receives the teacher's and the student's representations and tries to tell them apart. The student is trained to fool it. The representations are the same when the discriminator is **fooled maximally**, outputting probability **0.5**: the GAN equilibrium of [[ATML_Lecture_4 - GANs, and Inductive Biases#1.2 The Discriminator, the Generator, and the Equilibrium|Lecture 4 §1.2]], here with the teacher's representations as the "real" data (the same device as the domain discriminator in [[ATML_Lecture_5_1 - Domain Adaptation#6.2 Adversarial-Based — Domain Discriminator and the Gradient Reversal Layer (DANN)|Lecture 5.1 §6.2]]). Your notes record that, in specific scenarios, this was shown to be **almost similar to FitNets, empirically**.

> [!note] Unclear in source
> "Proved in specific scenario that this is almost similar to FitNets, empirically" mixes a proof and an empirical finding. Read here as: in the settings studied, adversarial distillation performs about the same as FitNets.

**Other divergences.** People have tried **optimal transport**, the **Wasserstein distance** $W_D(p, q)$, instead of KL, or the **Sinkhorn** distance; your friend adds that it could be any other such distance. These are [[ATML_Lecture_2 - Information Theory and Divergences#7.3 The Optimal Transport Linear Program and the Wasserstein Distance|Lecture 2 §7.3]] and [[ATML_Lecture_2 - Information Theory and Divergences#7.5 Entropic Regularization, Sinkhorn Distance, and the Sinkhorn Algorithm|§7.5]]. *(Expanding on why one would: KL compares the two distributions class by class and ignores how classes relate; OT uses a cost between classes, so moving mass from "cat" to "tiger" can cost less than from "cat" to "truck", which builds class similarity into the loss.)*

### 6.16 What knowledge gets distilled?

A **2023 paper** asks **what knowledge gets distilled**, which your friend marks as a paper to read. Its finding as noted in class: the knowledge transferred lies in a **different direction than the accuracy-maximizing direction**. The student inherits more than the teacher's accuracy. This is a **new research direction**.

The class example is **adversarial attacks**: for a cat image, the teacher looks at a particular region for its decision, and **so does the student**. *(Expanding on this: the paper is Ojha et al., "What Knowledge Gets Distilled in Knowledge Distillation?", NeurIPS 2023. It reports that students inherit teachers' localization of evidence, their vulnerability to the same adversarial perturbations, and some of their biases, even when that is not needed for accuracy.)*

---

## 7. Distillation for LLMs

Everything in §6 assumed one prediction per input. A language model makes a prediction at **every token**, conditioned on the tokens before it, and that raises a new question.

### 7.1 Which prefix does the student see?

Take $(x, y) \sim$ dataset, a prompt and its response. Both teacher and student are **autoregressive** ([[ATML_Lecture_7 - LLM Foundations#1.2 Autoregressive (decoder-only) models|Lecture 7 §1.2]]), so at each position $t$ they give next-token distributions $p(\cdot \mid y_{<t})$ and $q(\cdot \mid y_{<t})$: **distributions for every token**, matched per token with a divergence as in §6.2.

What problems can occur? The key one is **what token to feed next to the student** to train it, that is, where the prefix $y_{<t}$ comes from. Three sources of information are available:
1. the **dataset**;
2. the **teacher's predictions**;
3. the **student's predictions**.

This choice **defines two areas** of LLM distillation.

### 7.2 Off-policy distillation and exposure bias

In **off-policy distillation** the next token, and so the prefix, comes from **the teacher or the dataset**. This is teacher forcing ([[ATML_Lecture_7 - LLM Foundations#4.2 Worked example and teacher forcing|Lecture 7 §4.2]]): the student is always conditioned on good text it did not write.

**Problem: exposure bias.** *(Expanding on this:)* at inference the student conditions on **its own** earlier tokens. Once it makes a mistake, it is in a prefix unlike anything it was trained on, and errors compound, because training never exposed it to its own outputs.

### 7.3 On-policy distillation

In **on-policy distillation** the prefix comes from the **student's own samples**: the student generates, and the teacher provides the target distribution at every token of that student-written text. The student is trained on the prefixes it will actually see at inference, which removes exposure bias. *(Expanding on this: GKD, generalized knowledge distillation (Agarwal et al., 2024), trains on student-generated sequences, and MiniLLM (Gu et al., 2024) pairs on-policy sampling with the reverse KL, which is mode-seeking ([[ATML_Lecture_2 - Information Theory and Divergences#5.4 Reverse KL Divergence Is Mode-Seeking|Lecture 2 §5.4]]) and so avoids the averaging of §6.9.)*

![[03 - Attachments/ATML_L13_llm_kd.svg|760]]

*(a) Off-policy: the prefix comes from the teacher or the dataset, and the student and teacher distributions are matched on it per token; the student never sees its own mistakes (exposure bias). (b) On-policy: the student samples the next token itself, so it is trained on the prefixes it will produce at inference.*

### 7.4 When to use LLM distillation

The class listed three use cases:
1. **Initialize the student**, $\theta^{(0)}$: your notes have "init S with $\theta^{(0)}$ → 1–2 iterations of teacher".
2. **Prune → fine-tune on a stronger signal**: after pruning ([[#3. Pruning|§3]]), recover accuracy by fine-tuning against the teacher's soft distributions rather than hard labels.
3. **Post-training distillation**: bring in a strong model, "Fable 5.2" in your notes, to align a model such as DeepSeek.

> [!note] Unclear in source
> Use case 1 is only a fragment in both sets. Read here as warm-starting the student by distilling from the teacher for one or two iterations before ordinary training, but the intended meaning of "1–2 iterations of teacher" is not recorded.

Your notes end with: **generalization happens during the RL alignment stage.** *(Expanding on this: distillation, like SFT, teaches the student to imitate; the RL stage of [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.3 RLVR: a verifier in place of the learned reward model|Lecture 9 §1.3]], with rewards on its own outputs, is where the model learns behaviour beyond what it imitated.)*

![[03 - Attachments/ATML_L13_kd_map.svg|780]]

*How the distillation methods of §6–§7 relate. From plain KD, logit-level work refines the loss (DKD) and its optimization (gradient conflict fixes). Because the task head mixes the latent, feature-level methods distil before it (FitNets, CRD, adversarial), and other divergences (OT, Sinkhorn) replace KL. LLMs add the choice of prefix (off- vs on-policy) and a set of use cases.*


---

## Numerical Problems

> [!example]- Problem 1 — the sparsity budget
> A network has three layers with 1,000, 4,000 and 5,000 weights. With sparsity $\alpha = 0.9$, how many weights can be kept in total under [[#^eq-2|eq. (2)]]'s constraint?
>
> $\sum_\ell\lVert W^{(\ell)}\rVert_0 = 10{,}000$, so $\sum_\ell\lVert M^{(\ell)}\rVert_0 \le (1 - 0.9) \times 10{,}000$:
>
> $$\boxed{\text{at most } 1{,}000 \text{ weights kept}}$$

> [!example]- Problem 2 — why exact pruning is NP-hard
> How many masks keep 3 of 10 weights? How many keep 10% of 100?
>
> $$\binom{10}{3} = \frac{10\cdot9\cdot8}{3\cdot2\cdot1} = 120, \qquad \binom{100}{10} \approx \boxed{1.73\times10^{13}}$$
>
> Checking each mask would mean evaluating the loss $1.7\times10^{13}$ times for a 100-weight model, which is why the two-step approximation of [[#3.2 Pruning as constrained optimization|§3.2]] is used.

> [!example]- Problem 3 — layer-wise sparsity
> In Problem 1's network, a scheme prunes layer 1 at $\alpha_1 = 0.5$ (it is sensitive) and layers 2 and 3 at the same $\alpha'$. What must $\alpha'$ be for overall sparsity 0.9?
>
> The overall sparsity is the parameter-weighted average: $\frac{1000}{10000}(0.5) + \frac{9000}{10000}\alpha' = 0.9$, so $0.05 + 0.9\alpha' = 0.9$:
>
> $$\boxed{\alpha' = 0.944}$$
>
> The naive sum $\sum_\ell\alpha_\ell = 0.5 + 0.944 + 0.944 = 2.39$ is not 0.9, which is why the condition must be a weighted average ([[#3.3 How to group weights|§3.3]]).

> [!example]- Problem 4 — magnitude vs Wanda
> In the neuron of [[#3.5 Activation-aware pruning (Wanda)|§3.5]], $\theta_1 = 0.1$, $\theta_2 = 0.5$, and typical inputs are $\lvert x_1\rvert = 10$, $\lvert x_2\rvert = 0.5$. Which weight does each method prune, and how much output does each removal lose?
>
> Magnitude: $\lvert\theta_1\rvert = 0.1 < 0.5$, so it prunes $\theta_1$. Wanda: $\lvert\theta_1\rvert\lvert x_1\rvert = 1.0$ and $\lvert\theta_2\rvert\lvert x_2\rvert = 0.25$, so it prunes $\theta_2$. Removing $\theta_1$ loses a contribution of $1.0$; removing $\theta_2$ loses $0.25$.
>
> $$\boxed{\text{magnitude prunes } \theta_1 \text{ (loses 1.0)}; \ \text{Wanda prunes } \theta_2 \text{ (loses 0.25)}}$$

> [!example]- Problem 5 — Optimal Brain Damage
> Three weights have values $\theta = (0.4, -0.1, 0.3)$ and diagonal Hessian entries $H = (0.5, 20, 2)$. Rank them by OBD score and by magnitude.
>
> OBD scores $\frac12H_{jj}\theta_j^2$: $\frac12(0.5)(0.16) = 0.04$, $\frac12(20)(0.01) = 0.10$, $\frac12(2)(0.09) = 0.09$. Magnitudes: $0.4, 0.1, 0.3$.
>
> $$\boxed{\text{OBD prunes } \theta_1 \ (0.04); \ \text{magnitude prunes } \theta_2 \ (0.1)}$$
>
> $\theta_2$ is the smallest weight but sits in a sharply curved direction, so OBD keeps it; $\theta_1$ is the largest but the loss is flat along it.

> [!example]- Problem 6 — asymmetric uint8 quantization
> Activations range over $[-1, 3]$. Find $s$ and $z$ for uint8, quantize $x = 1.0$, and dequantize it.
>
> $q_{\min} = 0$, $q_{\max} = 255$. Then $s = (3 - (-1))/255 = 0.015686$ and $z = \text{round}(0 - (-1)/0.015686) = \text{round}(63.75) = 64$. For $x = 1.0$: $q = \text{round}(1.0/0.015686) + 64 = \text{round}(63.75) + 64 = 128$. Dequantizing: $\hat x = 0.015686 \times (128 - 64) = 1.0039$.
>
> $$\boxed{s = 0.01569, \ z = 64, \ q = 128, \ \hat x = 1.0039}$$
>
> The error, $0.0039$, is below $s/2 = 0.0078$, as it must be. Check the ends: $x = -1 \mapsto \text{round}(-63.75) + 64 = 0$ and $x = 3 \mapsto \text{round}(191.25) + 64 = 255$.

> [!example]- Problem 7 — symmetric int8 quantization
> Weights range over $[-0.6, 0.9]$. Find $s$, quantize $w = -0.5$, and find how many integer levels go unused.
>
> $s = \max(0.6, 0.9)/127 = 0.0070866$, $z = 0$. Then $q = \text{round}(-0.5/0.0070866) = \text{round}(-70.56) = -71$, and $\hat w = -71 \times 0.0070866 = -0.5031$. The most negative weight maps to $\text{round}(-0.6/0.0070866) = -85$, so the levels $-128$ to $-86$ are never used.
>
> $$\boxed{s = 0.00709, \ q = -71, \ \hat w = -0.5031, \ 43 \text{ unused levels}}$$

> [!example]- Problem 8 — the rate of a compressed LLM
> A 7B-parameter model is stored in fp16. Compare its rate with int4 quantization, and with 50% pruning plus int8.
>
> fp16: $7\times10^9 \times 16 = 1.12\times10^{11}$ bits $= 14$ GB. int4: $7\times10^9 \times 4 = 2.8\times10^{10}$ bits $= 3.5$ GB. Pruned and int8: $3.5\times10^9 \times 8 = 2.8\times10^{10}$ bits $= 3.5$ GB.
>
> $$\boxed{14 \text{ GB} \to 3.5 \text{ GB in both cases}}$$
>
> Both routes cut the rate by 4×, through different factors of [[#2.1 Rate and distortion|§2.1]]'s product (ignoring the cost of storing the mask's positions, which unstructured pruning also has to pay).

> [!example]- Problem 9 — lottery-ticket rounds
> Each round of the lottery-ticket loop prunes 20% of the remaining weights. How many rounds reach at least 97% sparsity?
>
> After $n$ rounds the density is $0.8^n$. We need $0.8^n \le 0.03$, so $n \ge \ln 0.03/\ln 0.8 = 15.7$:
>
> $$\boxed{16 \text{ rounds}}$$
>
> Each round includes a full retraining, which is the catch-22's cost ([[#3.7 The Lottery Ticket Hypothesis|§3.7]]).

> [!example]- Problem 10 — the generalization gap
> A model has training loss 0.05 and test loss 0.12. After compression the training loss is 0.07 and the test loss 0.11. Compute the gap and the distortion before and after (use test loss for $L$).
>
> Before: $\Delta = 0.12 - 0.05 = 0.07$. After: $\Delta = 0.11 - 0.07 = 0.04$. Distortion $= L(\hat\theta) - L(\theta) = 0.11 - 0.12 = -0.01$.
>
> $$\boxed{\Delta: 0.07 \to 0.04, \quad \text{distortion} = -0.01}$$
>
> A negative distortion is the left compression move of [[#1.3 Double descent|§1.3]]: the smaller model generalizes better.

> [!example]- Problem 11 — the minimum-norm solution
> Solve $\theta_1 + \theta_2 = 2$ (one sample, two parameters) with the pseudo-inverse, and compare with the solution $(2, 0)$.
>
> $A = [1\ \ 1]$, so $AA^\top = 2$ and $A^{+} = A^\top(AA^\top)^{-1} = \tfrac12[1\ \ 1]^\top$. Then $\hat\theta = A^{+}\cdot 2 = (1, 1)$, with norm $\sqrt2 = 1.414$, while $(2, 0)$ has norm 2.
>
> $$\boxed{\hat\theta = (1, 1), \ \lVert\hat\theta\rVert = 1.414 < 2}$$
>
> Both fit the data exactly; the pseudo-inverse picks the one with the smallest weights ([[#1.3 Double descent|§1.3]]).

> [!example]- Problem 12 — QAT with the straight-through estimator
> As in [[#4.7 When to quantize — QAT and the straight-through estimator|§4.7]], $H = \begin{bmatrix}1.1 & 2.2\\ -1.7 & 3.6\end{bmatrix}$ is quantized with $s = 1$, $z = 0$ and the range clipped to $[-2, 2]$. The backward pass delivers $\partial L/\partial \hat H = \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0.5\end{bmatrix}$. Find $\hat H$, the true gradient through the quantizer, and $\partial L/\partial H$ under the plain STE and the clipped STE.
>
> Rounding gives $(1, 2, -2, 4)$ and clipping $4 \mapsto 2$, so $\hat H = \begin{bmatrix}1 & 2\\ -2 & 2\end{bmatrix}$. The quantizer is flat at each of these inputs, so the true gradient is $0$ everywhere. The plain STE ([[#^eq-4|eq. (4)]]) copies the incoming gradient; the clipped STE also zeroes the entry that was clipped (3.6 lies outside $[-2, 2]$):
>
> $$\boxed{\text{true: } 0; \quad \text{STE: } \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0.5\end{bmatrix}; \quad \text{clipped STE: } \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0\end{bmatrix}}$$

> [!example]- Problem 13 — the outlier moves $s$ and $z$
> A layer's weights lie in $[-1, 4]$. Quantize asymmetrically to uint8, then repeat after one weight becomes $10^6$ ([[#4.9 When quantization struggles — the outlier problem|§4.9]]). Where does the weight $4$ land in each case? Repeat the second case with symmetric int8.
>
> Without the outlier: $s = 5/255 = 0.01961$ and $z = \text{round}(0 + 1/0.01961) = 51$; the weight 4 gives $q = \text{round}(204) + 51 = 255$. With it: $s = (10^6 + 1)/255 = 3921.6$ and $z = \text{round}(1/3921.6) = \text{round}(0.00026) = 0$; the weight 4 gives $q = \text{round}(0.001) + 0 = 0$, so $\hat w = 0$. Symmetric int8: $s = 10^6/127 = 7874$, and every $\lvert w\rvert \le 4$ has $\lvert w\rvert/s < 0.001$, so it rounds to 0.
>
> $$\boxed{z: 51 \to 0, \quad s: 0.0196 \to 3922, \quad \text{all eleven ordinary weights} \mapsto 0}$$
>
> The outlier changes both $s$ and $z$ (the question in your paper notes), and wipes out every other weight.

> [!example]- Problem 14 — low-rank and Hadamard parameter counts
> A $4096\times4096$ weight is replaced by a rank-64 factorization. What is the compression ratio? With the parameters of a rank-16 factorization, what maximum rank does the Hadamard construction of [[#5.2 More rank from the same parameters — the Hadamard product|§5.2]] reach?
>
> Dense: $4096^2 = 16{,}777{,}216$. Rank 64: $64(4096 + 4096) = 524{,}288$, a ratio of $32$. A rank-16 factorization costs $16 \times 8192 = 131{,}072$ parameters; spent as two rank-8 branches, it has the same count and rank up to $8^2 = 64$.
>
> $$\boxed{32\times \text{ smaller}; \quad \text{rank } 64 \text{ instead of } 16 \text{ for } 131{,}072 \text{ parameters}}$$

> [!example]- Problem 15 — temperature makes a teacher lenient
> A teacher's logits are $v = (4, 2, 0)$. Compute $p$ and its entropy at $\rho = 1$ and $\rho = 4$.
>
> At $\rho = 1$: $e^4, e^2, e^0 = 54.60, 7.389, 1$, total $62.99$, so $p = (0.867, 0.117, 0.016)$ and $H = -\sum p_i\ln p_i = 0.441$ nats. At $\rho = 4$ the logits become $(1, 0.5, 0)$: $e^1, e^{0.5}, 1 = 2.718, 1.649, 1$, total $5.367$, so $p = (0.506, 0.307, 0.186)$ and $H = 1.020$ nats.
>
> $$\boxed{\rho = 1:\ (0.867, 0.117, 0.016),\ H = 0.44; \qquad \rho = 4:\ (0.506, 0.307, 0.186),\ H = 1.02}$$
>
> The maximum, for the uniform distribution, is $\ln 3 = 1.099$: at $\rho = 4$ the teacher is lenient ([[#6.5 Strict vs lenient teacher|§6.5]]) but still ranks the classes.

> [!example]- Problem 16 — distillation at high temperature is logit matching
> With $K = 3$, $\rho = 10$, teacher logits $v = (2, 0, -2)$ and student logits $z = (1, 0, -1)$ (both zero-mean), compare the exact gradient $\rho(q_i - p_i)$ with the approximation $(z_i - v_i)/K$ of [[#6.3 Why the ρ² factor — distillation becomes logit MSE|§6.3]].
>
> $p = \text{softmax}(v/10) = (0.402, 0.329, 0.269)$ and $q = \text{softmax}(z/10) = (0.367, 0.332, 0.301)$. Exact: $10(q - p) = (-0.346, 0.033, 0.313)$. Approximation: $\tfrac13(-1, 0, 1) = (-0.333, 0, 0.333)$.
>
> $$\boxed{\text{exact } (-0.346, 0.033, 0.313) \approx \text{MSE gradient } (-0.333, 0, 0.333)}$$
>
> Both push $z_1$ up and $z_3$ down towards the teacher's more spread-out logits; the approximation improves as $\rho$ grows.

> [!example]- Problem 17 — the DKD decomposition
> With four classes and target $t = 1$, the teacher gives $p = (0.8, 0.06, 0.08, 0.06)$ and the student $q = (0.7, 0.1, 0.08, 0.12)$. Compute $\mathrm{KL}(p\Vert q)$, TCKD and NCKD, and check [[#^eq-6|eq. (6)]].
>
> Directly, $\mathrm{KL} = 0.8\ln\frac{0.8}{0.7} + 0.06\ln\frac{0.06}{0.1} + 0.08\ln 1 + 0.06\ln\frac{0.06}{0.12} = 0.1068 - 0.0306 + 0 - 0.0416 = 0.0346$. TCKD on $b^T = (0.8, 0.2)$, $b^S = (0.7, 0.3)$: $0.8\ln\frac87 + 0.2\ln\frac23 = 0.1068 - 0.0811 = 0.0257$. NCKD on $\hat p = (0.3, 0.4, 0.3)$, $\hat q = (0.333, 0.267, 0.4)$: $0.3\ln0.9 + 0.4\ln1.5 + 0.3\ln0.75 = -0.0316 + 0.1622 - 0.0863 = 0.0443$.
>
> $$\boxed{0.0257 + (1 - 0.8)(0.0443) = 0.0346 = \mathrm{KL}(p\Vert q)}$$
>
> In plain KD the non-target mismatch, the larger of the two, enters with weight only $0.2$; DKD gives it its own weight $\gamma_{\text{NC}}$ ([[#6.8 Decoupled knowledge distillation (DKD)|§6.8]]).

> [!example]- Problem 18 — measuring gradient conflict
> A student gives $q = (0.6, 0.3, 0.1)$ on a three-class example with true class 1, so $y = (1, 0, 0)$; the teacher gives $p = (0.5, 0.4, 0.1)$. With $\rho = 1$, compute the logit gradients of $L_{\text{CE}}$ and $L_{\text{KD}}$, their cosine ([[#^eq-7|eq. (7)]]), and the length of the update with $\lambda = \mu = 0.5$ ([[#6.10 Gradient conflict between the CE and KD terms|§6.10]]).
>
> $g_{\text{CE}} = q - y = (-0.4, 0.3, 0.1)$ and $g_{\text{KD}} = q - p = (0.1, -0.1, 0)$. The dot product is $-0.04 - 0.03 + 0 = -0.07$, and the norms are $\sqrt{0.26} = 0.510$ and $\sqrt{0.02} = 0.141$, so $\cos = -0.07/(0.510 \times 0.141) = -0.971$. The combined gradient is $0.5g_{\text{CE}} + 0.5g_{\text{KD}} = (-0.15, 0.10, 0.05)$, of length $0.187$.
>
> $$\boxed{\cos(g_{\text{CE}}, g_{\text{KD}}) = -0.97, \quad \lVert\text{update}\rVert = 0.187 < 0.255 = \lVert 0.5\,g_{\text{CE}}\rVert}$$
>
> The student is more confident on the target than the teacher ($0.6 > 0.5$), so KD pushes the target logit down while CE pushes it up, and the KD half of the update undoes part of the CE half.

> [!example]- Problem 19 — Gram–Schmidt harmonization
> Apply [[#^eq-8|eq. (8)]] to Problem 18's gradients and give the harmonized update.
>
> $\frac{g_{\text{KD}}\cdot g_{\text{CE}}}{\lVert g_{\text{CE}}\rVert^2} = \frac{-0.07}{0.26} = -0.2692$, so $g_{\text{KD}}^{\perp} = (0.1, -0.1, 0) + 0.2692\,(-0.4, 0.3, 0.1) = (-0.0077, -0.0192, 0.0269)$. Check: $g_{\text{KD}}^{\perp}\cdot g_{\text{CE}} = 0.0031 - 0.0058 + 0.0027 = 0$. The update $0.5g_{\text{CE}} + 0.5g_{\text{KD}}^{\perp} = (-0.204, 0.140, 0.063)$.
>
> $$\boxed{g_{\text{KD}}^{\perp} = (-0.0077, -0.0192, 0.0269), \quad \text{update} = (-0.204, 0.140, 0.063)}$$
>
> The CE part of the step is now intact, and only the KD direction that CE does not oppose remains.

> [!example]- Problem 20 — momentum filters a zig-zag
> The gradient along one coordinate alternates, $g_t = 0.2 + (-1)^t$ (so $-0.8, 1.2, -0.8, \dots$), with a true average of 0.2. With $\beta_m = 0.9$ and $v_0 = 0$ in [[#^eq-9|eq. (9)]], compute $v_1$ to $v_4$ and the size of $v$'s long-run oscillation.
>
> $v_1 = 0.1(-0.8) = -0.08$; $v_2 = 0.9(-0.08) + 0.1(1.2) = 0.048$; $v_3 = 0.0432 - 0.08 = -0.0368$; $v_4 = -0.0331 + 0.12 = 0.0869$. In steady state $v_t = 0.2 + A(-1)^t$; substituting, $A(-1)^t = -0.9A(-1)^{t} + 0.1(-1)^t$, so $A = 0.1/1.9 = 0.053$.
>
> $$\boxed{v_t \to 0.2 \pm 0.053, \ \text{versus raw gradients } 0.2 \pm 1}$$
>
> The oscillation shrinks by a factor of $(1 - \beta_m)/(1 + \beta_m) = 1/19$ while the consistent component passes unchanged: the low-pass filter of [[#6.11 Resolving the conflict — gradient harmonization and momentum|§6.11]].

> [!example]- Problem 21 — the FitNets regressor
> The student's hint features are 384-dimensional per token and the teacher's 768. How many parameters does a one-layer linear regressor $\text{Reg}$ ([[#^eq-10|eq. (10)]]) need?
>
> A $384 \to 768$ linear layer has $384 \times 768$ weights and $768$ biases:
>
> $$\boxed{294{,}912 + 768 = 295{,}680 \text{ parameters}}$$
>
> These are trained with the student and thrown away afterwards; their capacity is the "ideal regressor" question of [[#6.13 FitNets — feature-level distillation|§6.13]].

> [!example]- Problem 22 — the CRD ratio and temperature
> The teacher–student positive pair has cosine 0.8, and three negatives have cosines $0.1, -0.2, 0.3$. With $\text{sim} = \exp(\cos/\tau)$, compute [[#^eq-11|eq. (11)]] and the loss $-\ln(\cdot)$ at $\tau = 0.1$ and $\tau = 1$.
>
> At $\tau = 0.1$: $e^{8} = 2981.0$ and the negatives give $e^{1} + e^{-2} + e^{3} = 2.718 + 0.135 + 20.086 = 22.94$, so the ratio is $2981.0/3003.9 = 0.9924$ and the loss $0.0077$. At $\tau = 1$: $e^{0.8} = 2.226$ against $1.105 + 0.819 + 1.350 = 3.274$, so the ratio is $2.226/5.499 = 0.405$ and the loss $0.905$.
>
> $$\boxed{\tau = 0.1:\ 0.992,\ L = 0.008; \qquad \tau = 1:\ 0.405,\ L = 0.905}$$
>
> A low temperature sharpens the differences between similarities, so the same geometry is scored as almost solved; with $\tau = 1$ the negatives still claim most of the denominator.

> [!example]- Problem 23 — the fooled discriminator
> In adversarial distillation ([[#6.15 Adversarial distillation and other divergences|§6.15]]), the discriminator outputs $D(r)$, the probability that representation $r$ came from the teacher. Its optimum is $D^*(r) = \frac{p_T(r)}{p_T(r) + p_S(r)}$. What is $D^*$ when the student's representations match the teacher's, and what is the discriminator's binary cross-entropy $-\ln D(r_T) - \ln(1 - D(r_S))$ there?
>
> If $p_S = p_T$, then $D^* = p_T/(2p_T) = 0.5$ everywhere, and the loss is $-\ln 0.5 - \ln 0.5 = 2\ln 2$:
>
> $$\boxed{D^* = 0.5, \quad \text{BCE} = 2\ln 2 = 1.386}$$
>
> This is the equilibrium of [[ATML_Lecture_4 - GANs, and Inductive Biases#1.2 The Discriminator, the Generator, and the Equilibrium|Lecture 4 §1.2]]: the discriminator can do no better than a coin flip.

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Generalization gap | $\Delta = \mathbb E_{P(x,y)}[\ell] - \frac1N\sum_i\ell(f_\theta(x_i), y_i)$ | [[#1.2 The generalization gap\|§1.2]], [[ATML_Lecture_1 - Representation and Generalization Theory#5.1 The IID Setting and the Generalization Bound\|L1 §5.1]] |
| Minimum-norm solution | $\hat x = A^{+}y$, $A^{+} = A^\top(AA^\top)^{-1}$ | [[#1.3 Double descent\|§1.3]] |
| Rate | $(\#\text{ non-zero params})\times(\text{bits per param})$ | [[#2.1 Rate and distortion\|§2.1]] |
| Distortion | $L(\hat\theta) - L(\theta)$, ideally $\le 0$ | [[#2.1 Rate and distortion\|§2.1]] |
| Layer proxy | $\lVert\hat Wx - Wx\rVert_F$ | [[#2.2 A layer-level proxy for distortion\|§2.2]] |
| Rate–distortion objective ([[#^eq-1\|eq. (1)]]) | $\min_{\hat W}\text{Rate} + \beta\,\text{Distortion}$ | [[#2.3 The rate–distortion objective\|§2.3]] |
| Low-rank approximation | $W \approx BA$, $r(d_o + d_i)$ params | [[#2.4 Four families of compression\|§2.4]], [[ATML_Lecture_10 - Fine-Tuning#8.1 The low-rank update\|L10 §8.1]] |
| Pruning mask | $\hat W = W\odot M$, $M\in\{0,1\}$ | [[#3.1 Masks, sparsity and density\|§3.1]] |
| Sparsity and density | $\alpha = \#\text{removed}/D$, $1 - \alpha = \#\text{retained}/D$ | [[#3.1 Masks, sparsity and density\|§3.1]] |
| Pruning problem ([[#^eq-2\|eq. (2)]]) | $\min\mathbb E[L(f_{\hat W}(x), y)]$ s.t. $\sum_\ell\lVert M^{(\ell)}\rVert_0 \le (1-\alpha)\sum_\ell\lVert W^{(\ell)}\rVert_0$ | [[#3.2 Pruning as constrained optimization\|§3.2]] |
| Search size | $\binom{D}{(1-\alpha)D}$ | [[#3.2 Pruning as constrained optimization\|§3.2]] |
| Layer-wise sparsity | $\sum_\ell\frac{D_\ell}{D}\alpha_\ell = \alpha$ | [[#3.3 How to group weights\|§3.3]] |
| Taylor estimate | $\Delta L \approx -\frac{\partial L}{\partial\theta_j}\theta_j + \frac12H_{jj}\theta_j^2$ | [[#3.4 How to score groups\|§3.4]] |
| Zero-order score | $\lvert\theta_{ij}\rvert$; group $\lVert\theta_j\rVert_2^2$ | [[#3.4 How to score groups\|§3.4]] |
| First-order (SNIP) | $\lvert(\partial L/\partial\theta_{ij})\,\theta_{ij}\rvert$; grad-norm $\lVert\partial L/\partial\theta_j\rVert_2$ | [[#3.4 How to score groups\|§3.4]] |
| Second-order (OBD) | $\frac12H_{jj}\theta_j^2$ | [[#3.4 How to score groups\|§3.4]] |
| OBS update | $\delta\theta = -\frac{\theta_q}{[H^{-1}]_{qq}}H^{-1}e_q$, cost $\frac{\theta_q^2}{2[H^{-1}]_{qq}}$ | [[#3.4 How to score groups\|§3.4]] |
| Wanda | $\lvert\theta_{ij}\rvert\cdot\lVert X_j\rVert_2$ | [[#3.5 Activation-aware pruning (Wanda)\|§3.5]] |
| Lottery ticket | $(\theta_0, M)$; rewind survivors to $\theta_0$ | [[#3.7 The Lottery Ticket Hypothesis\|§3.7]] |
| Affine map | $CA/BA = C'A'/B'A'$ | [[#4.1 Mapping reals to integers\|§4.1]] |
| Integer ranges | signed $[-2^{b-1}, 2^{b-1}-1]$; unsigned $[0, 2^b - 1]$ | [[#4.2 Scale, zero-point and levels\|§4.2]] |
| Quantize ([[#^eq-3\|eq. (3)]]) | $q = \text{round}(x/s) + z$ | [[#4.2 Scale, zero-point and levels\|§4.2]] |
| Dequantize | $\hat x = s(q - z)$, error $\le s/2$ | [[#4.2 Scale, zero-point and levels\|§4.2]] |
| Asymmetric | $s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$, $z = \text{round}(q_{\min} - x_{\min}/s)$ | [[#4.3 Asymmetric quantization\|§4.3]] |
| Symmetric | $z = 0$, $s = \max(\lvert x_{\min}\rvert, \lvert x_{\max}\rvert)/q_{\max}$ | [[#4.4 Symmetric quantization\|§4.4]] |
| Integer matmul | $Wx \approx s_ws_x(Q_wq_x)$ | [[#4.5 What quantization touches, and integer arithmetic\|§4.5]] |
| Rate as cost | $\text{Rate} \approx \#\text{params}\times\#\text{bits/param}$, minimized at a given distortion | [[#4.6 Quantize and dequantize along the network\|§4.6]], [[#2.1 Rate and distortion\|§2.1]] |
| Straight-through estimator ([[#^eq-4\|eq. (4)]]) | $\partial\hat H/\partial H := 1$, so $\partial L/\partial H = \partial L/\partial\hat H$ | [[#4.7 When to quantize — QAT and the straight-through estimator\|§4.7]] |
| Outlier scale | $s \approx x_{\max}/(2^b - 1)$ set by one value; $z = \text{round}(q_{\min} - x_{\min}/s) \to q_{\min}$ | [[#4.9 When quantization struggles — the outlier problem\|§4.9]] |
| Low-rank compression | $\hat W = BA$, $mn \to r(m+n)$ params, rank $\le r$ | [[#5.1 Replacing W by two thin matrices\|§5.1]], [[ATML_Lecture_10 - Fine-Tuning#8.1 The low-rank update\|L10 §8.1]] |
| Hadamard rank bound | $\text{rank}(X\odot Y) \le \text{rank}(X)\,\text{rank}(Y)$; $(B_1A_1)\odot(B_2A_2)$ reaches $r^2$ with $2r(m+n)$ params | [[#5.2 More rank from the same parameters — the Hadamard product\|§5.2]] |
| Temperature softmax | $p_i = e^{v_i/\rho}/\sum_j e^{v_j/\rho}$, $q_i = e^{z_i/\rho}/\sum_j e^{z_j/\rho}$ | [[#6.1 Teacher, student and the undistilled baseline\|§6.1]], [[#6.6 Shaping the teacher's distribution — temperature and label smoothing\|§6.6]] |
| Distillation loss ([[#^eq-5\|eq. (5)]]) | $L_T = \lambda L_{\text{CE}} + (1-\lambda)\rho^2\,\mathrm{KL}(p\Vert q)$ | [[#6.2 Logit matching and the combined loss\|§6.2]], [[ATML_Lecture_10 - Fine-Tuning#3.5 Distillation-based fine-tuning\|L10 §3.5]] |
| High-temperature gradient | $\partial L_{\text{DIST}}/\partial z_i = \rho(q_i - p_i) \approx (z_i - v_i)/K$, i.e. logit MSE | [[#6.3 Why the ρ² factor — distillation becomes logit MSE\|§6.3]] |
| Label smoothing | $y_{\text{LS}} = (1-\varepsilon)y + \varepsilon/K$ | [[#6.6 Shaping the teacher's distribution — temperature and label smoothing\|§6.6]] |
| DKD decomposition ([[#^eq-6\|eq. (6)]]) | $\mathrm{KL}(p\Vert q) = \mathrm{KL}(b^T\Vert b^S) + (1-p_t)\,\mathrm{KL}(\hat p\Vert\hat q)$, $\hat p_i = p_i/(1-p_t)$ | [[#6.8 Decoupled knowledge distillation (DKD)\|§6.8]] |
| DKD loss | $\gamma_{\text{TC}}\,\mathrm{TCKD} + \gamma_{\text{NC}}\,\mathrm{NCKD}$ | [[#6.8 Decoupled knowledge distillation (DKD)\|§6.8]] |
| Decoupled weights | $L_T = \lambda L_{\text{CE}} + \mu L_{\text{KD}}$, $\nabla L_T = \lambda\nabla L_{\text{CE}} + \mu\nabla L_{\text{KD}}$ | [[#6.10 Gradient conflict between the CE and KD terms\|§6.10]] |
| Logit gradients ($\rho = 1$) | $\partial L_{\text{CE}}/\partial z = q - y$, $\partial L_{\text{KD}}/\partial z = q - p$ | [[#6.10 Gradient conflict between the CE and KD terms\|§6.10]] |
| Gradient conflict ([[#^eq-7\|eq. (7)]]) | $\cos(\nabla L_{\text{CE}}, \nabla L_{\text{KD}})$, observed $\approx 0$ or $< 0$ | [[#6.10 Gradient conflict between the CE and KD terms\|§6.10]] |
| Gram–Schmidt harmonization ([[#^eq-8\|eq. (8)]]) | $g_{\text{KD}}^{\perp} = g_{\text{KD}} - \frac{g_{\text{KD}}\cdot g_{\text{CE}}}{\lVert g_{\text{CE}}\rVert^2}g_{\text{CE}}$ | [[#6.11 Resolving the conflict — gradient harmonization and momentum\|§6.11]] |
| EMA momentum ([[#^eq-9\|eq. (9)]]) | $v_t = \beta_m v_{t-1} + (1-\beta_m)g_t$, $\theta_{t+1} = \theta_t - \eta v_t$ | [[#6.11 Resolving the conflict — gradient harmonization and momentum\|§6.11]] |
| Task head | $v = \text{MLP}(h)$, $h \in \mathbb R^d \to v \in \mathbb R^K$, $d \gg K$ | [[#6.12 Why logits lose structure — the task head\|§6.12]] |
| FitNets ([[#^eq-10\|eq. (10)]]) | $\text{MSE}(F_T, \text{Reg}(F_S))$ | [[#6.13 FitNets — feature-level distillation\|§6.13]] |
| CRD ([[#^eq-11\|eq. (11)]]) | $\frac{\text{sim}(h_T,h_S)}{\text{sim}(h_T,h_S) + \sum_{k=2}^N\text{sim}(h_T,h_k^-)}$, $\text{sim} = e^{\cos/\tau}$ | [[#6.14 Contrastive Representation Distillation (CRD)\|§6.14]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.2 The Contrastive Pipeline and the InfoNCE Loss\|L3 §3.2]] |
| Optimal discriminator | $D^* = p_T/(p_T + p_S) = 0.5$ when matched | [[#6.15 Adversarial distillation and other divergences\|§6.15]], [[ATML_Lecture_4 - GANs, and Inductive Biases#1.2 The Discriminator, the Generator, and the Equilibrium\|L4 §1.2]] |

---

## Summary

- **Why compress ([[#1. Why Compress a Model|§1]])**
  - Here $D$ is the parameter count and $N$ the number of samples, the reverse of Lecture 7.
  - The generalization gap is actual risk minus empirical risk.
  - Double descent: test loss falls, peaks at the interpolation point $D \approx N$, and falls again for $D \gg N$, where SGD picks the minimum-norm solution $A^{+}y$.
  - A model before the peak can be compressed to lower test loss; a model in the interpolation region can lose many parameters for little loss.
  - FLOPs and memory (weights, activations, optimizer states) grow with $D$.

- **Rate–distortion ([[#2. Compression as Rate–Distortion|§2]])**
  - Rate = non-zero parameters × bits per parameter; distortion = $L(\hat\theta) - L(\theta)$, ideally zero or negative.
  - Distortion is approximated per layer by $\lVert\hat Wx - Wx\rVert_F$.
  - Compression minimizes rate + $\beta$·distortion, an information-bottleneck trade-off.
  - Pruning cuts non-zero parameters, quantization cuts bits per parameter; low-rank ($W \approx BA$) and distillation shrink the parameter count.

- **Pruning ([[#3. Pruning|§3]])**
  - $\hat W = W\odot M$ with a binary mask; sparsity $\alpha$ is the fraction removed, density $1 - \alpha$ the fraction kept.
  - Exact pruning searches $\binom{D}{(1-\alpha)D}$ masks and is NP-hard, so group weights and discard the lowest-scoring groups.
  - Unstructured groups are single weights (simple, high sparsity, little speed-up); structured groups are filters, channels or layers (real speed-up, coarser); layer-wise schemes split $\alpha$ per layer as a weighted average.
  - Scores: magnitude (zero-order), SNIP $\lvert g\theta\rvert$ (first-order), OBD $\frac12H_{jj}\theta^2$ (second-order, diagonal Hessian); OBS also adjusts the other weights.
  - Wanda scores $\lvert\theta\rvert\cdot\lVert X\rVert$, because a small weight on a large input can matter more than a large weight on a small input.
  - Prune before (PBT), during (PDT) or after training (PAT, the most common).
  - Lottery tickets: a sparse sub-network, rewound to its initial weights, trains to the dense accuracy; finding it still needs the dense training.

- **Quantization ([[#4. Quantization|§4]])**
  - An affine map from reals to $2^b$ integer levels: $q = \text{round}(x/s) + z$, recovered as $s(q - z)$.
  - Asymmetric (activations): $s = (x_{\max} - x_{\min})/(q_{\max} - q_{\min})$, $z = \text{round}(q_{\min} - x_{\min}/s)$.
  - Symmetric (weights): $z = 0$, $s = \max\lvert x\rvert/q_{\max}$; simpler arithmetic, but lopsided ranges waste levels.
  - Pruning touches only weights; quantization also touches activations, which makes integer-only matrix multiplication possible.
  - Quantize (A) through the layers, dequantize (B) before the LM head so the softmax sees real logits; bit-widths can differ per head.
  - QAT trains with quantization in the forward pass; rounding has zero gradient, so the STE treats it as the identity in the backward pass.
  - PTQ quantizes a trained model; activation-aware methods (AWQ) protect weights whose inputs are large, like Wanda in pruning.
  - One outlier sets $s$ (and, for asymmetric, $z$), so every ordinary value collapses into a few levels; fixes include clipping, higher-precision outliers and finer granularity.
  - Mixed precision (which bits for which layer) is an open problem; quantization usually beats pruning in practice; NAS searches bits and pruning jointly, e.g. for edge devices.

- **Low-rank approximation ([[#5. Low-Rank Approximation|§5]])**
  - Drop $W$ and keep $BA$: $mn \to r(m+n)$ parameters; $r$ is the capacity, more $r$ means more memory and less distortion.
  - $\text{rank}(X\odot Y) \le \text{rank}X\cdot\text{rank}Y$, so $(B_1A_1)\odot(B_2A_2)$ reaches rank $r^2$ for the parameters of a rank-$2r$ product.

- **Knowledge distillation ([[#6. Knowledge Distillation|§6]])**
  - A frozen large teacher, a small trained student that is deployed; the baseline is the undistilled student trained on $L_{\text{CE}}$ only.
  - $L_T = \lambda L_{\text{CE}} + (1-\lambda)\rho^2\mathrm{KL}(p\Vert q)$; the notes' minus sign belongs only to the cross-entropy form.
  - $\rho^2$ keeps the gradient size independent of temperature; at high $\rho$ the gradient is $\approx (z_i - v_i)/K$, so distillation becomes logit MSE.
  - Soft labels carry similarity between classes that one-hot labels lack; a one-hot teacher adds no information.
  - A lenient teacher (higher entropy) is better than a strict one, but a uniform teacher is useless; shape it with temperature or label smoothing.
  - Teacher > student > undistilled student; the gap comes from limited capacity and lost dark knowledge on non-target classes.
  - DKD splits KL into TCKD (target vs a hypothetical "rest" class) and NCKD (renormalized non-target classes); plain KD weights NCKD by $1 - p_t$, so a confident teacher hides it, and DKD weights the two separately.
  - Hard labels say only $x \in$ class $i$; soft labels add $\text{similarity}(x, c_j)$. Forward KL makes a one-mode student cover a two-mode teacher, peaking at their average.
  - With independent weights, $\nabla L_{\text{CE}}$ and $\nabla L_{\text{KD}}$ conflict (cosine $\approx 0$ or $< 0$): one-hot targets push non-target logits to 0, soft labels pull them up, so updates are short and convergence slow.
  - Fixes: Gram–Schmidt removes the opposing component of $g_{\text{KD}}$ (cosine $\to 0$); momentum (EMA, a low-pass filter) keeps the average direction.
  - The task head $v = \text{MLP}(h)$ mixes $d \gg K$ latent dimensions, destroying inter-class structure in the logits (open problem), so distil features instead.
  - FitNets: $\text{MSE}(F_T, \text{Reg}(F_S))$ with a regressor for the size mismatch, but the ideal regressor is unknown.
  - CRD: InfoNCE between teacher and student embeddings of the same $x$ against many negatives (~65K in class). Ranking: CRD ≫ FitNets > logit matching.
  - Adversarial distillation (discriminator fooled at 0.5) performs about like FitNets; OT/Wasserstein or Sinkhorn can replace KL.
  - What is distilled goes beyond accuracy (2023): students copy where teachers look, including adversarial weaknesses.

- **LLM distillation ([[#7. Distillation for LLMs|§7]])**
  - Per-token distributions; the key choice is where the prefix comes from (dataset, teacher or student).
  - Off-policy (teacher or dataset prefixes) suffers exposure bias; on-policy trains on the student's own samples. Your notes swap the labels; your friend's (used here) are standard.
  - Uses: initialize the student, prune then fine-tune on the teacher's signal, post-training distillation from a stronger model; generalization comes in the RL alignment stage.

---
type: lecture
course: "[[_CS6304 ATML|CS6304 ATML]]"
source:
  - "[[ATML Lec 20.pdf]]"
  - "[[ATML Lec 21.pdf]]"
  - "[[ATML Lec 22 (1).pdf]]"
  - "[[ATML Lec 22 (2).pdf]]"
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

*[[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning|Lecture 12]] spent more compute at inference to get better answers from a fixed model. This lecture goes the other way: it makes the model itself cheaper to store and run, while losing as little accuracy as possible. It starts from why large models are expensive and why a smaller model can still generalize (double descent), frames compression as a trade-off between rate and distortion, and then works through the four families: pruning, quantization, low-rank approximation and knowledge distillation.*

There are no professor's notes for this lecture yet. The sources are your class notes [[ATML Lec 20.pdf|ATML Lecture 20]] (pages 3–5, from "Model Compression" on), [[ATML Lec 21.pdf|ATML Lecture 21]] (10 pages), and two sets of Lecture 22 notes: the paper notebook [[ATML Lec 22 (1).pdf|Lecture 22 (1)]] (4 pages, 12 Nov 2026) and the Notability [[ATML Lec 22 (2).pdf|Lecture 22 (2)]] (11 pages). [[#1. Why Compress a Model|§1]]–[[#2. Compression as Rate–Distortion|§2]] come from Lecture 20, [[#3. Pruning|§3]] and [[#4.1 Mapping reals to integers|§4.1]]–[[#4.5 What quantization touches, and integer arithmetic|§4.5]] from Lecture 21, and [[#4.6 Quantize and dequantize along the network|§4.6]]–[[#6. Knowledge Distillation|§6]] from Lecture 22. Where the two Lecture 22 sets differ, both readings are given. All figures are redrawn in TikZ from your notes.

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
> In [[ATML_Lecture_2 - Information Theory and Divergences#5.3 Forward KL Is Mode-Covering (Support-Seeking)|Lecture 2 §5.3]], the forward $\mathrm{KL}(p\Vert q)$ used in [[#^eq-5|eq. (5)]] is **mode-covering**; the reverse $\mathrm{KL}(q\Vert p)$ is the mode-seeking one ([[ATML_Lecture_2 - Information Theory and Divergences#5.4 Reverse KL Divergence Is Mode-Seeking|§5.4]]). The effect your notes describe is real, but it comes from a different mechanism: when the teacher is confident, the KL term is dominated by the target class and the non-target part is down-weighted. [[#6.8 Decoupled knowledge distillation (DKD)|§6.8]] makes this exact.

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

---

## Numerical Problems

**Problem 1 — the sparsity budget.** A network has three layers with 1,000, 4,000 and 5,000 weights. With sparsity $\alpha = 0.9$, how many weights can be kept in total under [[#^eq-2|eq. (2)]]'s constraint?

$\sum_\ell\lVert W^{(\ell)}\rVert_0 = 10{,}000$, so $\sum_\ell\lVert M^{(\ell)}\rVert_0 \le (1 - 0.9) \times 10{,}000$:

$$\boxed{\text{at most } 1{,}000 \text{ weights kept}}$$

**Problem 2 — why exact pruning is NP-hard.** How many masks keep 3 of 10 weights? How many keep 10% of 100?

$$\binom{10}{3} = \frac{10\cdot9\cdot8}{3\cdot2\cdot1} = 120, \qquad \binom{100}{10} \approx \boxed{1.73\times10^{13}}$$

Checking each mask would mean evaluating the loss $1.7\times10^{13}$ times for a 100-weight model, which is why the two-step approximation of [[#3.2 Pruning as constrained optimization|§3.2]] is used.

**Problem 3 — layer-wise sparsity.** In Problem 1's network, a scheme prunes layer 1 at $\alpha_1 = 0.5$ (it is sensitive) and layers 2 and 3 at the same $\alpha'$. What must $\alpha'$ be for overall sparsity 0.9?

The overall sparsity is the parameter-weighted average: $\frac{1000}{10000}(0.5) + \frac{9000}{10000}\alpha' = 0.9$, so $0.05 + 0.9\alpha' = 0.9$:

$$\boxed{\alpha' = 0.944}$$

The naive sum $\sum_\ell\alpha_\ell = 0.5 + 0.944 + 0.944 = 2.39$ is not 0.9, which is why the condition must be a weighted average ([[#3.3 How to group weights|§3.3]]).

**Problem 4 — magnitude vs Wanda.** In the neuron of [[#3.5 Activation-aware pruning (Wanda)|§3.5]], $\theta_1 = 0.1$, $\theta_2 = 0.5$, and typical inputs are $\lvert x_1\rvert = 10$, $\lvert x_2\rvert = 0.5$. Which weight does each method prune, and how much output does each removal lose?

Magnitude: $\lvert\theta_1\rvert = 0.1 < 0.5$, so it prunes $\theta_1$. Wanda: $\lvert\theta_1\rvert\lvert x_1\rvert = 1.0$ and $\lvert\theta_2\rvert\lvert x_2\rvert = 0.25$, so it prunes $\theta_2$. Removing $\theta_1$ loses a contribution of $1.0$; removing $\theta_2$ loses $0.25$.

$$\boxed{\text{magnitude prunes } \theta_1 \text{ (loses 1.0)}; \ \text{Wanda prunes } \theta_2 \text{ (loses 0.25)}}$$

**Problem 5 — Optimal Brain Damage.** Three weights have values $\theta = (0.4, -0.1, 0.3)$ and diagonal Hessian entries $H = (0.5, 20, 2)$. Rank them by OBD score and by magnitude.

OBD scores $\frac12H_{jj}\theta_j^2$: $\frac12(0.5)(0.16) = 0.04$, $\frac12(20)(0.01) = 0.10$, $\frac12(2)(0.09) = 0.09$. Magnitudes: $0.4, 0.1, 0.3$.

$$\boxed{\text{OBD prunes } \theta_1 \ (0.04); \ \text{magnitude prunes } \theta_2 \ (0.1)}$$

$\theta_2$ is the smallest weight but sits in a sharply curved direction, so OBD keeps it; $\theta_1$ is the largest but the loss is flat along it.

**Problem 6 — asymmetric uint8 quantization.** Activations range over $[-1, 3]$. Find $s$ and $z$ for uint8, quantize $x = 1.0$, and dequantize it.

$q_{\min} = 0$, $q_{\max} = 255$. Then $s = (3 - (-1))/255 = 0.015686$ and $z = \text{round}(0 - (-1)/0.015686) = \text{round}(63.75) = 64$. For $x = 1.0$: $q = \text{round}(1.0/0.015686) + 64 = \text{round}(63.75) + 64 = 128$. Dequantizing: $\hat x = 0.015686 \times (128 - 64) = 1.0039$.

$$\boxed{s = 0.01569, \ z = 64, \ q = 128, \ \hat x = 1.0039}$$

The error, $0.0039$, is below $s/2 = 0.0078$, as it must be. Check the ends: $x = -1 \mapsto \text{round}(-63.75) + 64 = 0$ and $x = 3 \mapsto \text{round}(191.25) + 64 = 255$.

**Problem 7 — symmetric int8 quantization.** Weights range over $[-0.6, 0.9]$. Find $s$, quantize $w = -0.5$, and find how many integer levels go unused.

$s = \max(0.6, 0.9)/127 = 0.0070866$, $z = 0$. Then $q = \text{round}(-0.5/0.0070866) = \text{round}(-70.56) = -71$, and $\hat w = -71 \times 0.0070866 = -0.5031$. The most negative weight maps to $\text{round}(-0.6/0.0070866) = -85$, so the levels $-128$ to $-86$ are never used.

$$\boxed{s = 0.00709, \ q = -71, \ \hat w = -0.5031, \ 43 \text{ unused levels}}$$

**Problem 8 — the rate of a compressed LLM.** A 7B-parameter model is stored in fp16. Compare its rate with int4 quantization, and with 50% pruning plus int8.

fp16: $7\times10^9 \times 16 = 1.12\times10^{11}$ bits $= 14$ GB. int4: $7\times10^9 \times 4 = 2.8\times10^{10}$ bits $= 3.5$ GB. Pruned and int8: $3.5\times10^9 \times 8 = 2.8\times10^{10}$ bits $= 3.5$ GB.

$$\boxed{14 \text{ GB} \to 3.5 \text{ GB in both cases}}$$

Both routes cut the rate by 4×, through different factors of [[#2.1 Rate and distortion|§2.1]]'s product (ignoring the cost of storing the mask's positions, which unstructured pruning also has to pay).

**Problem 9 — lottery-ticket rounds.** Each round of the lottery-ticket loop prunes 20% of the remaining weights. How many rounds reach at least 97% sparsity?

After $n$ rounds the density is $0.8^n$. We need $0.8^n \le 0.03$, so $n \ge \ln 0.03/\ln 0.8 = 15.7$:

$$\boxed{16 \text{ rounds}}$$

Each round includes a full retraining, which is the catch-22's cost ([[#3.7 The Lottery Ticket Hypothesis|§3.7]]).

**Problem 10 — the generalization gap.** A model has training loss 0.05 and test loss 0.12. After compression the training loss is 0.07 and the test loss 0.11. Compute the gap and the distortion before and after (use test loss for $L$).

Before: $\Delta = 0.12 - 0.05 = 0.07$. After: $\Delta = 0.11 - 0.07 = 0.04$. Distortion $= L(\hat\theta) - L(\theta) = 0.11 - 0.12 = -0.01$.

$$\boxed{\Delta: 0.07 \to 0.04, \quad \text{distortion} = -0.01}$$

A negative distortion is the left compression move of [[#1.3 Double descent|§1.3]]: the smaller model generalizes better.

**Problem 11 — the minimum-norm solution.** Solve $\theta_1 + \theta_2 = 2$ (one sample, two parameters) with the pseudo-inverse, and compare with the solution $(2, 0)$.

$A = [1\ \ 1]$, so $AA^\top = 2$ and $A^{+} = A^\top(AA^\top)^{-1} = \tfrac12[1\ \ 1]^\top$. Then $\hat\theta = A^{+}\cdot 2 = (1, 1)$, with norm $\sqrt2 = 1.414$, while $(2, 0)$ has norm 2.

$$\boxed{\hat\theta = (1, 1), \ \lVert\hat\theta\rVert = 1.414 < 2}$$

Both fit the data exactly; the pseudo-inverse picks the one with the smallest weights ([[#1.3 Double descent|§1.3]]).

**Problem 12 — QAT with the straight-through estimator.** As in [[#4.7 When to quantize — QAT and the straight-through estimator|§4.7]], $H = \begin{bmatrix}1.1 & 2.2\\ -1.7 & 3.6\end{bmatrix}$ is quantized with $s = 1$, $z = 0$ and the range clipped to $[-2, 2]$. The backward pass delivers $\partial L/\partial \hat H = \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0.5\end{bmatrix}$. Find $\hat H$, the true gradient through the quantizer, and $\partial L/\partial H$ under the plain STE and the clipped STE.

Rounding gives $(1, 2, -2, 4)$ and clipping $4 \mapsto 2$, so $\hat H = \begin{bmatrix}1 & 2\\ -2 & 2\end{bmatrix}$. The quantizer is flat at each of these inputs, so the true gradient is $0$ everywhere. The plain STE ([[#^eq-4|eq. (4)]]) copies the incoming gradient; the clipped STE also zeroes the entry that was clipped (3.6 lies outside $[-2, 2]$):

$$\boxed{\text{true: } 0; \quad \text{STE: } \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0.5\end{bmatrix}; \quad \text{clipped STE: } \begin{bmatrix}0.3 & -0.1\\ 0.2 & 0\end{bmatrix}}$$

**Problem 13 — the outlier moves $s$ and $z$.** A layer's weights lie in $[-1, 4]$. Quantize asymmetrically to uint8, then repeat after one weight becomes $10^6$ ([[#4.9 When quantization struggles — the outlier problem|§4.9]]). Where does the weight $4$ land in each case? Repeat the second case with symmetric int8.

Without the outlier: $s = 5/255 = 0.01961$ and $z = \text{round}(0 + 1/0.01961) = 51$; the weight 4 gives $q = \text{round}(204) + 51 = 255$. With it: $s = (10^6 + 1)/255 = 3921.6$ and $z = \text{round}(1/3921.6) = \text{round}(0.00026) = 0$; the weight 4 gives $q = \text{round}(0.001) + 0 = 0$, so $\hat w = 0$. Symmetric int8: $s = 10^6/127 = 7874$, and every $\lvert w\rvert \le 4$ has $\lvert w\rvert/s < 0.001$, so it rounds to 0.

$$\boxed{z: 51 \to 0, \quad s: 0.0196 \to 3922, \quad \text{all eleven ordinary weights} \mapsto 0}$$

The outlier changes both $s$ and $z$ (the question in your paper notes), and wipes out every other weight.

**Problem 14 — low-rank and Hadamard parameter counts.** A $4096\times4096$ weight is replaced by a rank-64 factorization. What is the compression ratio? With the parameters of a rank-16 factorization, what maximum rank does the Hadamard construction of [[#5.2 More rank from the same parameters — the Hadamard product|§5.2]] reach?

Dense: $4096^2 = 16{,}777{,}216$. Rank 64: $64(4096 + 4096) = 524{,}288$, a ratio of $32$. A rank-16 factorization costs $16 \times 8192 = 131{,}072$ parameters; spent as two rank-8 branches, it has the same count and rank up to $8^2 = 64$.

$$\boxed{32\times \text{ smaller}; \quad \text{rank } 64 \text{ instead of } 16 \text{ for } 131{,}072 \text{ parameters}}$$

**Problem 15 — temperature makes a teacher lenient.** A teacher's logits are $v = (4, 2, 0)$. Compute $p$ and its entropy at $\rho = 1$ and $\rho = 4$.

At $\rho = 1$: $e^4, e^2, e^0 = 54.60, 7.389, 1$, total $62.99$, so $p = (0.867, 0.117, 0.016)$ and $H = -\sum p_i\ln p_i = 0.441$ nats. At $\rho = 4$ the logits become $(1, 0.5, 0)$: $e^1, e^{0.5}, 1 = 2.718, 1.649, 1$, total $5.367$, so $p = (0.506, 0.307, 0.186)$ and $H = 1.020$ nats.

$$\boxed{\rho = 1:\ (0.867, 0.117, 0.016),\ H = 0.44; \qquad \rho = 4:\ (0.506, 0.307, 0.186),\ H = 1.02}$$

The maximum, for the uniform distribution, is $\ln 3 = 1.099$: at $\rho = 4$ the teacher is lenient ([[#6.5 Strict vs lenient teacher|§6.5]]) but still ranks the classes.

**Problem 16 — distillation at high temperature is logit matching.** With $K = 3$, $\rho = 10$, teacher logits $v = (2, 0, -2)$ and student logits $z = (1, 0, -1)$ (both zero-mean), compare the exact gradient $\rho(q_i - p_i)$ with the approximation $(z_i - v_i)/K$ of [[#6.3 Why the ρ² factor — distillation becomes logit MSE|§6.3]].

$p = \text{softmax}(v/10) = (0.402, 0.329, 0.269)$ and $q = \text{softmax}(z/10) = (0.367, 0.332, 0.301)$. Exact: $10(q - p) = (-0.346, 0.033, 0.313)$. Approximation: $\tfrac13(-1, 0, 1) = (-0.333, 0, 0.333)$.

$$\boxed{\text{exact } (-0.346, 0.033, 0.313) \approx \text{MSE gradient } (-0.333, 0, 0.333)}$$

Both push $z_1$ up and $z_3$ down towards the teacher's more spread-out logits; the approximation improves as $\rho$ grows.

**Problem 17 — the DKD decomposition.** With four classes and target $t = 1$, the teacher gives $p = (0.8, 0.06, 0.08, 0.06)$ and the student $q = (0.7, 0.1, 0.08, 0.12)$. Compute $\mathrm{KL}(p\Vert q)$, TCKD and NCKD, and check [[#^eq-6|eq. (6)]].

Directly, $\mathrm{KL} = 0.8\ln\frac{0.8}{0.7} + 0.06\ln\frac{0.06}{0.1} + 0.08\ln 1 + 0.06\ln\frac{0.06}{0.12} = 0.1068 - 0.0306 + 0 - 0.0416 = 0.0346$. TCKD on $b^T = (0.8, 0.2)$, $b^S = (0.7, 0.3)$: $0.8\ln\frac87 + 0.2\ln\frac23 = 0.1068 - 0.0811 = 0.0257$. NCKD on $\hat p = (0.3, 0.4, 0.3)$, $\hat q = (0.333, 0.267, 0.4)$: $0.3\ln0.9 + 0.4\ln1.5 + 0.3\ln0.75 = -0.0316 + 0.1622 - 0.0863 = 0.0443$.

$$\boxed{0.0257 + (1 - 0.8)(0.0443) = 0.0346 = \mathrm{KL}(p\Vert q)}$$

In plain KD the non-target mismatch, the larger of the two, enters with weight only $0.2$; DKD gives it its own weight $\gamma_{\text{NC}}$ ([[#6.8 Decoupled knowledge distillation (DKD)|§6.8]]).

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

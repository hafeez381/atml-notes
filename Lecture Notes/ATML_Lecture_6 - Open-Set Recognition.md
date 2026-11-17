---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[OSR Notes.pdf]]'
tags:
- atml
- course/cs6304
- open-set-recognition
- ood-detection
title: ATML — Lecture 6 — Open-Set Recognition
lecture: 6
aliases:
- OSR
- ATML Lecture 6
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_5_2 - Domain Generalization]] · [[ATML_Lecture_6_Quiz_Prep|Quiz Prep L1–6]] →
Concepts: [[Open-Set Recognition]]
%% nav:end %%

# ATML — Lecture 6: Open-Set Recognition

## Where This Lecture Sits

*Continues from [[ATML_Lecture_5_1 - Domain Adaptation]] and [[ATML_Lecture_5_2 - Domain Generalization]], which studied covariate shift under a fixed label space. This lecture takes up the other kind of shift that Lecture 5.1 Section 2.3 named and set aside — semantic shift, where test inputs belong to classes the model has never seen.*

The question is no longer how to stay accurate when $P(x)$ moves, but what a classifier should do when the correct answer is *none of the classes it knows*. The source (six photographed pages) is cropped at the page edge on pages 2, 3 and 5; each affected spot is flagged where it occurs.

## Topic Inventory

- 1. The Open-Set Recognition Setting
  - 1.1 From Covariate Shift to Semantic Shift
  - 1.2 Notation
  - 1.3 The Closed-Set Model Cannot Say "None of These"
  - 1.4 Unknownness Score and Rejection Threshold
  - 1.5 The OSR Goal
- 2. Terminology and Types of Unknowns
  - 2.1 Four Related Problems
  - 2.2 Near and Far Unknowns
- 3. Formal Definition of OSR
  - 3.1 The Classifier
  - 3.2 The Detector
- 4. Two Questions and Two Design Principles
  - 4.1 Where Should the Classifier Say "Known"? — Open-Space Risk
  - 4.2 The Open-Space Risk Objective
  - 4.3 The Optimal Detector — the Likelihood-Ratio Test
  - 4.4 The Two Design Principles
- 5. What Can We Do? Four Families of Methods
  - 5.1 Post-hoc Scoring
  - 5.2 Maximum Softmax Probability (MSP)
  - 5.3 Maximum Logit Score (MLS)
  - 5.4 Energy Score
  - 5.5 Mahalanobis Distance
  - 5.6 How the Four Scores Relate
  - 5.7 Good Closed-Set Classifier (GCSC)
  - 5.8 Placeholder Learning — PROSER
  - 5.9 Geometry Reshaping — Reciprocal Points
- 6. Why Closed-Set Models Fail on OSR
  - 6.1 The Symptom: False Confidence
  - 6.2 Mechanism 1: Cross-Entropy Says Nothing About Unknowns
  - 6.3 Mechanism 2: Unbounded Decision Regions
  - 6.4 Mechanism 3: Generic Discriminative Features
  - 6.5 Shortcuts and Spurious Correlations
- 7. OSR Metrics
  - 7.1 Detector Metrics and Classifier Metrics
  - 7.2 The Running Example
  - 7.3 Closed-Set Accuracy (CSA)
  - 7.4 AUROC
  - 7.5 TPR, FPR, FRR and FPR@95TPR
  - 7.6 AUROC versus FPR@95TPR, and OSCR
- Looking Ahead
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. The Open-Set Recognition Setting

### 1.1 From Covariate Shift to Semantic Shift

The professor opens by contrasting OSR with the two settings of Module II. In **UDA (unsupervised domain adaptation) and DG** (domain generalization), the **label space stays fixed** and **only $P(x)$ changes**; the goal is to keep $f_\theta$ accurate under that change. Every method in Lectures 5.1 and 5.2 assumed that a test input, however unfamiliar its style, still belongs to one of the training classes.

OSR removes that assumption. At test time some inputs come from classes that never appeared in training, so the right output for them is not any training label. This is **semantic shift** in the sense of Lecture 5.1 Section 2.3: what has changed is the *meaning* of the input, not merely its appearance.

### 1.2 Notation

In OSR we have:

- $\mathcal{X}$, the **input space**;
- $\mathcal{Y}_K = \{1, 2, \ldots, K\}$, the $K$ **known classes** seen in training;
- $\mathcal{Y}_U$, the **unknown classes**, with the assumption $\mathcal{Y}_U \cap \mathcal{Y}_K = \emptyset$ — the two label sets are disjoint, which is precisely what makes this a semantic shift;
- $P_K$, the **distribution of known-class data**;
- $P_U$, the **distribution of unknown-class data**.

$P_K$ is what the training set samples from. $P_U$ is never observed during training, and as Section 4.3 shows, that single fact shapes every method in the lecture.

### 1.3 The Closed-Set Model Cannot Say "None of These"

The model is the pipeline of Lecture 1 Section 2: an encoder produces a feature, a linear head produces logits, and softmax turns logits into class probabilities.

$$
x \;\longrightarrow\; \boxed{g_\theta(x)} \;\xrightarrow{\;h\,:\,d\times 1\;}\; \boxed{\text{Linear layer: } z = Wh + b} \;\xrightarrow{\;z\,:\,K\times 1 \text{ (logits)}\;}\; \boxed{\text{Softmax}} \;\longrightarrow\; P_K(x)
$$

The feature $h \in \mathbb{R}^{d}$ is the output of the encoder $g_\theta$, the logits $z \in \mathbb{R}^{K}$ have one entry per known class, and the softmax output — written $P_K(x)$ in the notes, with components $p_k(x) = e^{z_k}/\sum_{j} e^{z_j}$ — is a distribution over the $K$ known classes only.

The professor's margin note states the problem exactly: the softmax outputs **sum to 1, so the model has no way to say "none of these"**. Every input, including a photograph of something the model has never seen, is forced to distribute all of its probability mass among the $K$ known classes. Softmax normalizes away the one thing an unknown input should convey, namely that none of the classes fits well.

### 1.4 Unknownness Score and Rejection Threshold

Since the classifier cannot express "none of these", OSR adds a separate quantity that can. Define:

- $u(x)$, the **unknownness score**. A higher value means $x$ looks more like an unknown.
- $\tau$, the **rejection threshold**, which turns the score into a decision:

$$
\boxed{u(x) \le \tau \;\Rightarrow\; x \in \text{Known} \;\to\; \text{Accept}, \qquad u(x) > \tau \;\Rightarrow\; x \in \text{Unknown} \;\to\; \text{Reject}}
$$

The convention *larger $u$ = more unknown* is used throughout the lecture and matters when reading every score in Section 5: each is written with a sign chosen so that it grows for unknowns.

### 1.5 The OSR Goal

The notes highlight the goal: **make the model abstain under semantic shift while staying accurate on $\mathcal{Y}_K$**. Both halves are required. A model that rejects everything abstains perfectly but is useless, and a model that never rejects is simply a closed-set classifier.

The failure being guarded against is drawn as a pipeline: a wolf image enters $f_\theta$ and comes out labelled "dog", given that wolf $\notin \mathcal{Y}_K$. This is the canonical **OSR failure** — not a wrong known class, but a confident known class for something that is not known at all.

---

## 2. Terminology and Types of Unknowns

Section 1 defined the problem in isolation. In practice OSR sits among several closely related problems that are often confused, and the distinctions matter because they change what the model is trained on and what it is asked.

### 2.1 Four Related Problems

| # | Problem | Trained on | Question asked at test time |
|---|---|---|---|
| 1 | **Anomaly / outlier detection** | one class only (e.g. "cat") | *Is this a cat?* Flag anything else. |
| 2 | **OOD (out-of-distribution) detection** | $P_K$ | *Is $x$ from $P_K$?* |
| 3 | **OSR** | $P_K$ and $\mathcal{Y}_K$ | *Which known class does $x$ belong to, or is it unknown?* |
| 4 | **Open-world recognition** | $P_K$ and $\mathcal{Y}_K$ | Detect unknowns, then collect them, label them, learn them, and continue. |

Anomaly detection takes a **one-class view**: train on "cat" only and flag anything else. OOD detection uses a multi-class training set but asks only a binary question, whether $x$ comes from $P_K$. OSR asks the binary question *and* the $K$-way question together, which is why it needs both a classifier and a detector (Section 3). Open-world recognition extends OSR into a loop: rejected unknowns are collected, labelled, and added as new known classes, and the process continues. *(Expanding on this: OOD detection and OSR use near-identical scoring machinery — the scores of Section 5 come from the OOD literature — but OOD benchmarks typically treat any dataset shift as "out", while OSR specifically targets new classes, i.e. semantic shift.)*

### 2.2 Near and Far Unknowns

Not all unknowns are equally hard, and the notes distinguish two types.

A **near unknown** shares shape, texture, or context with $\mathcal{Y}_K$ objects. The example is a known dog and an unknown wolf: $\text{Dog} \in \mathcal{Y}_K \leftrightarrow \text{wolf} \in \mathcal{Y}_U$. A wolf activates almost every feature a dog does, so it lands close to the dog class in feature space.

A **far unknown** shares little with any known class. With $\mathcal{Y}_K = \{\text{car}, \text{bus}, \text{pick-up}\}$ and $\mathcal{Y}_U = \{\text{bowl}\}$, a bowl has nothing in common with vehicles.

Far unknowns are comparatively easy, since they tend to produce weak activations everywhere. Near unknowns are the real test of an OSR method, because they are exactly the inputs a closed-set classifier maps confidently to a known class — the wolf-to-dog failure of Section 1.5.

---

## 3. Formal Definition of OSR

With the vocabulary in place, the notes define an OSR system as two cooperating components: a classifier that handles the known classes, and a detector that decides whether to trust it.

### 3.1 The Classifier

The classifier is an ordinary closed-set model — an encoder parameterized by $\theta$ followed by a head producing logits $z_k(x)$ — trained to minimize the **closed-set risk**
$$
\boxed{R_{\text{closed}}(\theta) = \mathbb{E}_{(x,y)\sim P_K}\big[\ell(f_\theta(x), y)\big]}
$$

with prediction
$$
\hat{y}(x) = \operatorname*{arg\,max}_{k \in \mathcal{Y}_K} z_k(x).
$$

The expectation is over $P_K$ only. This is the ERM (empirical risk minimization) objective of Lecture 5.1 Section 3 restricted to known data, and Section 6.2 returns to what that restriction implies.

### 3.2 The Detector

The detector computes $u(x)$ and gates the classifier's output:

$$
\boxed{\hat{y}(x) = \begin{cases} \operatorname*{arg\,max}_{k\in\mathcal{Y}_K} z_k(x) & u(x) \le \tau \;\to\; \text{Accept} \\[4pt] \text{Unknown} & u(x) > \tau \;\to\; \text{Reject} \end{cases}}
$$

The output space is therefore $\mathcal{Y}_K \cup \{\text{Unknown}\}$, which is the "none of these" option softmax could not provide. The two components can share a network — $u$ is often computed from the same logits — but they answer different questions, and Section 7 evaluates them with different metrics.

---

## 4. Two Questions and Two Design Principles

The definition says what an OSR system outputs but not how to build a good one. The notes approach that through two questions: one geometric, about where acceptance should be allowed, and one statistical, about what the ideal detector would compute. Each produces a design principle.

### 4.1 Where Should the Classifier Say "Known"? — Open-Space Risk

The first question is: **geometrically, where should the classifier be allowed to say "known"?** The answer is the **open-space risk** design principle: **the acceptance region should not extend arbitrarily far beyond the training data.**

![[03 - Attachments/ATML_L6_openspace.jpg|600]]

The sketch contrasts two models over the same four classes (circles, crosses, stars, triangles). The **closed-set model** partitions the whole feature space into four cells: every point belongs to some class, including points far from any training sample. The **open-set model** instead draws a bounded region around each class's samples and leaves the rest of the space unclaimed. $S_O$ is the **entire feature space**, and the unclaimed remainder is the **open space** $O$, the region where unknowns may lie.

Samples in $O$ carry the risk of being classified as "knowns", and that risk is the **open-space risk**. The more samples in $O$ that are incorrectly identified as known, the higher the open-space risk.

> [!warning] Partly cut off in source
> The notes write "Create an open space where unknowns may lie: $O = \ldots$", but the right edge of the page is cropped and the definition of $O$ is lost. The standard definition (Scheirer et al., 2013, which introduced open-space risk) is used here: $O = S_O \setminus \bigcup_{i} B_r(x_i)$, the feature space minus balls of radius $r$ around the training samples. This matches the sketch, where $O$ is everything outside the drawn class regions.

### 4.2 The Open-Space Risk Objective

Open-space risk is quantified as the fraction of the classifier's acceptance that falls in open space:

$$
\boxed{R_O(f_\theta) = \frac{\int_{O} f_\theta(x)\,dx}{\int_{S_O} f_\theta(x)\,dx} = \frac{\text{Accepted mass in } O}{\text{Total accepted mass}}}
$$

Here $f_\theta(x) \ge 0$ plays the role of a **recognition function** — how strongly the model accepts $x$ as known (in the simplest case, $1$ if accepted and $0$ otherwise). Note that this is a different use of $f_\theta$ from the logit-producing classifier of Section 3.1. The ratio is $0$ if the model accepts nothing outside the support of the training data, and approaches $1$ if almost all acceptance lies in unexplored space. A closed-set model, which accepts everywhere, has an open-space risk close to $1$, since open space is most of $S_O$.

The training objective balances this against ordinary accuracy:

$$
\boxed{\min_\theta \; R_O(f_\theta) + \lambda\, R_{\text{sup}}(f_\theta)}
$$

The first term means **do not accept too much from unexplored space**; the second, the supervised risk, means **classify the closed set correctly**; and $\lambda$ trades them off. Minimizing $R_O$ alone is trivial (accept nothing), and minimizing $R_{\text{sup}}$ alone recovers the closed-set model, so the two terms together encode both halves of the OSR goal of Section 1.5.

### 4.3 The Optimal Detector — the Likelihood-Ratio Test

The second question: **if we actually knew what unknown data looked like, what would be the optimal detector?** The answer is the **likelihood-ratio test (LRT)**:

$$
\boxed{u^*(x) = \log\frac{P_U(x)}{P_K(x)}} \qquad \begin{cases} u^*(x) > \tau \to \text{Reject} \\ u^*(x) \le \tau \to \text{Accept} \end{cases}
$$

The score compares how likely $x$ is under the unknown distribution against how likely it is under the known one, and rejects when the unknown explanation wins by more than $\tau$. *(The professor states optimality directly; the justification is the Neyman–Pearson lemma: among all tests that accept known data at a given rate, thresholding the likelihood ratio minimizes the rate at which unknowns are accepted. No other score can do better at any operating point.)*

**But $P_U(x)$ is always unknown.** It is unobserved by definition — no unknown classes appear in training — so the optimal detector cannot be computed. What the LRT provides is a target: every practical score is an attempt to approximate it with only $P_K$ available.

### 4.4 The Two Design Principles

The two answers together give the lecture's two design principles:

1. **Control unsupported acceptance regions** — from Section 4.1: do not let the model accept inputs far from its training data.
2. **Construct a proxy for the unavailable LRT** — from Section 4.3: since $P_U$ is unknown, build a score that behaves like $\log P_U/P_K$ using what is available.

Every method in Section 5 can be read as pursuing one or both. Post-hoc scores and GCSC build proxies for the LRT from a fixed classifier; placeholder learning simulates $P_U$ so a proxy can be learned; geometry reshaping bounds the acceptance regions directly.

---

## 5. What Can We Do? Four Families of Methods

> [!note] Partly cut off in source
> The left edge of this page is cropped. The heading reads "…can we do?" and the score labels read "…P:", "…S:", "…nergy:"; these are reconstructed as "What can we do?", "MSP", "MLS" and "Energy", consistent with the formulas beside them.

Section 4 established what an ideal detector would look like and why it cannot be built. The notes now answer the practical question of what can be done instead, in four families of increasing intervention: leave the classifier alone and score it, train it better, give it an explicit unknown class, or reshape its geometry.

### 5.1 Post-hoc Scoring

**Post-hoc scoring** keeps $f_\theta(\cdot)$ fixed and defines $u(x)$ from quantities the trained network already produces — **logits, features, activations, etc.** No retraining is needed, so these scores apply to any existing classifier. The four scores below differ only in which quantity they read and how.

### 5.2 Maximum Softmax Probability (MSP)

$$
\boxed{u_{\text{MSP}}(x) = 1 - \max_k p_k(x)}
$$

MSP measures **how dominant the winning logit is relative to all other logits**; the quantity $\max_k p_k$ is the model's **high relative confidence**. A confident prediction gives a small $u_{\text{MSP}}$ and is accepted.

The notes' example shows the limitation. With logits $z = [10, 0, 0]^\top$, softmax gives $p \approx [1, 0, 0]^\top$, so $u_{\text{MSP}} \approx 0$ and the input is accepted. But the conclusion drawn is: **high relative confidence $\ne$ high evidence of being known**. Softmax is invariant to adding a constant to every logit, so <mark style="background: #ABF7F7A6;">MSP sees only the *differences* between logits, never their absolute size</mark>. An input that weakly activates every class but one slightly more than the rest can receive the same MSP as an input that strongly activates one class. MSP answers "how sure is the model which known class this is?" rather than "how sure is the model this is known at all?".

### 5.3 Maximum Logit Score (MLS)

$$
\boxed{u_{\text{MLS}}(x) = -\max_k z_k(x)}
$$

MLS reads the **absolute level of the strongest raw logit**, before softmax discards scale. The sign convention is explained in a side note: a large $\max_k z_k$ means *known*, so its negation $-\max_k z_k$ is large for *unknowns*, which is the orientation $u$ requires.

The notes then show MLS's own blind spot. With $z_A = [6, 5.9, 0]^\top$ and $z_B = [6, 0, 0]^\top$, both inputs have the **same MLS**, $u_{\text{MLS}} = -6$, **although $A$ is highly ambiguous** <mark style="background: #ABF7F7A6;">between its first two classes.</mark> <mark style="background: #FFF3A3A6;">MLS looks only at the top logit and ignores what the rest are doing</mark>. MSP would separate these two cases (Numerical Problem 1); MLS cannot. The two scores fail in complementary ways, one blind to absolute scale and the other to the competition.

### 5.4 Energy Score

$$
\boxed{u_{\text{energy}}(x) = -\log\Big(\sum_k e^{z_k(x)}\Big)}
$$

The energy score measures the **aggregate level of all logits**. Its argument is the log-sum-exp of Lecture 1 Section 7.3, the normalizer of the softmax, which is a smooth version of the maximum: it grows with the largest logit but also with how many logits are large.

The notes' example: with $z_A = [6, 0, 0]^\top$ and $z_B = [5.5, 5.5, 5.5]^\top$, **$B$ has lower energy and looks more known, even though all classes are similarly activated.** Numerically $u_{\text{energy}}(A) \approx -6.005$ and $u_{\text{energy}}(B) \approx -6.599$. Input $B$ does not favour any class — MSP would give it the maximal uncertainty $u_{\text{MSP}} = 2/3$ — yet energy treats it as more familiar than $A$, because the sum over three strong logits exceeds one slightly stronger logit. <mark style="background: #FFF3A3A6;">Energy rewards total activation, which is its strength against far unknowns</mark> (weak everywhere) and its weakness against inputs that excite several classes at once. *(Expanding on this: the standard energy score, Liu et al. 2020, includes a temperature, $-T\log\sum_k e^{z_k/T}$; the notes use $T = 1$.)*

### 5.5 Mahalanobis Distance

The first three scores read logits. The Mahalanobis score reads the feature $h$ instead, measuring the **distance from the nearest known-class feature distribution**:
$$
\boxed{u_{\text{MAH}}(x) = \min_{c\in\mathcal{Y}_K} (h - \mu_c)^\top \Sigma^{-1} (h - \mu_c)}
$$

Here $\mu_c$ is the mean training feature of class $c$ and $\Sigma$ the covariance of the training features. *(Expanding on this: in the standard method, Lee et al. 2018, each class is modelled as a Gaussian with its own mean and a covariance shared by all classes, both estimated from training features; the score is the squared Mahalanobis distance of Lecture 2 Section 2.3 to the nearest class Gaussian.)* <mark style="background: #FFF3A3A6;">Where the logit scores ask how strongly the head responds, Mahalanobis asks whether the feature lies inside a known class's ellipsoid</mark> — a density-based proxy for $P_K(x)$ in the denominator of the LRT.

<mark style="background: #ABF7F7A6;">Its failure mode, noted in the source: an unknown "fox" feature lands close to the known "dog" cluster. A near unknown (Section 2.2) sits inside a known class's ellipsoid, and no distance to class means can separate it.</mark> This is not a <mark style="background: #FFF3A3A6;">flaw of</mark> the distance but of the <mark style="background: #FFF3A3A6;">features</mark>: if the <mark style="background: #FFF3A3A6;">encoder</mark> represents a fox and a dog with the same features, every score computed from those features inherits the confusion.

### 5.6 How the Four Scores Relate

*(This subsection is an addition connecting the three logit scores; it is not in the notes.)* The three logit scores are closer than they look. Writing $m = \max_k z_k$ and $L = \log\sum_k e^{z_k}$, the definitions give $u_{\text{MLS}} = -m$ and $u_{\text{energy}} = -L$, and since $\max_k p_k = e^{m}/e^{L}$,

$$
u_{\text{MSP}} = 1 - e^{\,u_{\text{energy}} - u_{\text{MLS}}}.
$$

Because $m \le L \le m + \log K$, energy and MLS never differ by more than $\log K$: $u_{\text{MLS}} - \log K \le u_{\text{energy}} \le u_{\text{MLS}}$. The gap between them is $0$ when one logit dominates and $\log K$ when all logits are equal, and MSP is a function of that gap alone. So MLS measures absolute strength, MSP measures only relative dominance, and energy mixes the two. In the notes' energy example, the gap for $z_B = [5.5,5.5,5.5]$ is exactly $\log 3 \approx 1.099$, the maximum possible, which is why MSP rates $B$ as maximally uncertain while energy rates it as more known than $A$.

| Score | Reads | Measures | Blind to | Better on |
|---|---|---|---|---|
| MSP | softmax | relative dominance of winner | absolute logit scale | near unknowns (detects low relative confidence) |
| MLS | top logit | absolute strength of winner | competition between classes | far unknowns (weak absolute top logit) |
| Energy | all logits | aggregate activation | ambiguity when many logits are high | far unknowns (weak everywhere → low total activation) |
| Mahalanobis | features | distance to nearest class Gaussian | near unknowns inside a class ellipsoid | far unknowns (far from all class Gaussians) |

### 5.7 Good Closed-Set Classifier (GCSC)

The post-hoc scores all assume a fixed classifier, and Section 5.5 closed on the observation that a <mark style="background: #FFF3A3A6;">score can only be as good as the features beneath it</mark>. The second family acts on that observation directly: **a good closed-set classifier, with MLS, is what we need.**

The mechanism given in the notes: training a stronger classifier **removes shortcuts in the representation**, which **forces the features to carry the true identity of the object rather than nuisance statistics**. If the features encode what a dog actually is, a wolf will match less of it, and the MLS of a wolf will fall. *(Expanding on this: this is the finding of Vaze et al. 2022, "Open-Set Recognition: A Good Closed-Set Classifier Is All You Need". Their recipe trains a standard architecture much longer, with stronger augmentation, label smoothing, and a better learning-rate schedule, and scores with MLS rather than MSP. On standard benchmarks this simple baseline matched or beat purpose-built OSR methods.)*

The notes attach two qualifications:

- **There is a strong correlation between CSA and AUROC** (area under the ROC (receiver operating characteristic) curve) — closed-set accuracy and OSR detection quality rise together across models. **But the correlation is empirical, and correlation $\ne$ object identity.** Higher accuracy is evidence of better features, not proof of them.
- **Improving CSA rewards any direction that raises accuracy.** The example is a **texture cue that separates cat from dog but not from leopard**. Such a cue improves closed-set accuracy on a cat/dog task while doing nothing to reject an unknown leopard, which shares the cat's texture — in fact it actively pulls the leopard toward "cat".

The second point connects to two earlier lectures. It is the freedom identified in Lecture 1 Section 7.4, that cross-entropy never specifies *which* separating direction to use, and the texture bias of [[ATML_Lecture_4 - GANs, and Inductive Biases]] Section 4, now appearing as an OSR failure rather than a robustness failure. GCSC works to the extent that "better classifier" means "better object features"; when accuracy is bought with shortcuts, OSR does not improve.

### 5.8 Placeholder Learning — PROSER

GCSC improves the features but still gives the model no explicit place to put unknowns. The third family does exactly that. **Placeholder learning (PROSER)**: **manufacture unknown-like data points using known data points, and give the classifier somewhere to put them.** This attacks the missing $P_U$ of Section 4.3 by simulating it.

![[03 - Attachments/ATML_L6_proser.jpg|700]]

**Architecture.** The input $x \in y_j$ passes through the encoder to the feature $h$ and the logits $z$, and the softmax now has **$K+1$ outputs**: the $K$ known classes plus a dummy class $K+1$ standing for "unknown". A branch leaves an intermediate layer of the encoder, passes through **Manifold Mixup**, and produces an extra logit $z_u$ that joins $z$ at the softmax.

**Manufacturing unknowns.** Take two samples from different known classes, $x_i \in y_i$ and $x_j \in y_j$, and obtain their hidden representations $h_i, h_j$. Then **interpolate the hidden representations**:

$$
\boxed{h_u = \lambda h_i + (1-\lambda) h_j, \qquad \lambda \sim \text{Beta}(2,2)}
$$

The example is $h_u = 0.45\,h_{\text{cat}} + 0.55\,h_{\text{truck}}$, which is labelled as the **$K+1$ unknown class**.

![[03 - Attachments/ATML_L6_mixup.jpg|450]]

The sketch shows why this works: the interpolated point lies on the path between the cat cluster and the truck cluster, in the empty region between them. That is exactly open space in the sense of Section 4.1, and labelling it "unknown" teaches the model that the gaps between classes are not to be claimed by any known class. Mixing at an intermediate layer, rather than on pixels, matters: a pixel-space blend of a cat and a truck is an unnatural double exposure, while a blend of hidden representations lands in the space where the classifier actually makes decisions. The $\text{Beta}(2,2)$ distribution has mean $0.5$ and concentrates $\lambda$ away from $0$ and $1$, so interpolants usually sit genuinely between the classes rather than on top of one of them.

**The loss.**

$$
\boxed{L_{\text{cls}} = \ell_{\text{ce}}\big([z, z_u],\, y_j\big) + \beta\,\ell_{\text{ce}}\big([z_{\setminus y_j}, z_u],\, y_{K+1}\big)}
$$

The first term is ordinary cross-entropy over all $K+1$ logits with the true label $y_j$: the **correct class wins**. The second term removes the true class's logit — written $z_{\setminus y_j}$, the logits with the true class $y_j$ **masked** — and applies cross-entropy with target $K+1$: **the dummy $K+1$ class should win once the true class is masked.** The weight $\beta$ balances the two.

The second term is the subtle one. It trains the dummy logit to be the *runner-up* for every known sample: once the true class is removed, "unknown" must outrank every wrong known class. The dummy logit therefore sits just below the true class everywhere, which makes it a learned, input-adaptive threshold. For a known input, the true class clears it; for an unknown input, no known class is strong enough to clear it, and the dummy wins. The class $K+1$ then serves directly as the unknownness signal.

> [!note] Fuller picture
> *(Expanding beyond the notes.)* The original method (Zhou, Ye and Zhan, 2021, "Learning Placeholders for Open-Set Recognition") has two separate components, which the notes' diagram combines. **Classifier placeholders** add a dummy classifier head whose logit is trained by the masked second term above on ordinary known samples. **Data placeholders** are the manifold-mixup interpolants, trained with a separate cross-entropy loss toward class $K+1$ and added with their own weight. Both push toward the same outcome — class $K+1$ absorbs open space — from two directions: one sets the threshold, the other supplies explicit examples of what lies beyond it.

### 5.9 Geometry Reshaping — Reciprocal Points

Placeholder learning simulates unknowns; the fourth family changes the geometry so that unknowns have somewhere natural to fall. **Geometry reshaping** means: **bound known regions and model what a class is NOT.**

![[03 - Attachments/ATML_L6_rpl_point.jpg|650]]

Each class $j$ gets a **reciprocal point** $P_j$, a learned point that **represents everything that is not $y_j$**. The known samples of class $j$ occupy a region $\in y_j$ around features $h_j$, and the distance $d(h_j, P_j)$ from the reciprocal point is the **evidence for class $j$**: **the larger $d(h_j, P_j)$, the larger the evidence for class $j$**.

This inverts the usual prototype picture of Lecture 1 Section 7.5. There, each class had a prototype $\psi_j$ and a sample was assigned to the class whose prototype it was *closest* to. Here, each class has an anti-prototype, and a sample is assigned to the class whose anti-prototype it is *farthest* from. *(Expanding on this: this is Reciprocal Point Learning, Chen et al. 2020. The logit for class $k$ is a distance $d(h, P_k)$, so cross-entropy pushes each sample away from its own class's reciprocal point.)*

The loss has a classification term and a bounding term:

$$
\boxed{L_{\text{RPL}} = \ell_{\text{ce}}(z, y) + \lambda\big(d(h_j, P_j) - R\big)^2, \qquad d(h_j, P_j) = \lVert h_j - P_j\rVert_2^2}
$$

The first term pushes samples away from their reciprocal point to raise the evidence for the correct class. On its own it would push forever, since evidence grows without bound with distance. The second term stops that: it penalizes any deviation of $d(h_j, P_j)$ from the learned radius $R$, so **when $d(\cdot) \approx R$, the push from this term stops.** Known samples are held on a shell at distance about $R$ from their reciprocal point rather than flung arbitrarily far out.

![[03 - Attachments/ATML_L6_rpl_plot.jpg|550]]

The feature-space sketch shows the result in two dimensions. Classes 1, 2 and 3 sit on the outside, while the reciprocal points $P^1, P^2, P^3$ cluster together in the middle, inside a bounded circle — the region that is *not* any class. **Possible unknowns** fall into that central region. An unknown has no reason to be far from any reciprocal point, so it lands near all of them, has low evidence for every class, and is rejected. This is the first design principle of Section 4.4 realized geometrically: the bounding term keeps known classes on a shell of radius $R$, so the acceptance regions stay bounded and the central open space is left for unknowns. *(A natural unknownness score is then $u(x) = -\max_k d(h, P_k)$: low maximum evidence means unknown.)*

---

## 6. Why Closed-Set Models Fail on OSR

> [!note] Partly cut off in source
> The left edge of this page is cropped. The heading is reconstructed as "Why closed-set models fail on OSR?", "…chanisms:" as "Mechanisms", the three mechanism numbers "1-, 2-, 3-" from the visible pattern, and "…eneric Discriminative Features" as "Generic Discriminative Features".

Section 5 presented the remedies; this section, which comes next in the professor's notes, gives the diagnosis they respond to. Reading it now also explains, in hindsight, why each family takes the form it does.

### 6.1 The Symptom: False Confidence

**The symptom is always the same: false confidence.** A closed-set model rarely fails on an unknown by being uncertain; it fails by assigning a high score to a known class, as in the wolf-to-dog example of Section 1.5. The notes identify three mechanisms behind it.

### 6.2 Mechanism 1: Cross-Entropy Says Nothing About Unknowns

**Cross-entropy says nothing about unknowns.** The closed-set risk of Section 3.1 is an expectation over $P_K$ alone. No term in it is evaluated on unknown inputs, so training places no constraint whatsoever on the model's output for them. Whatever the model does on unknowns is an accident of how it generalizes the known-class boundaries. This is the formal reason the LRT of Section 4.3 is unavailable and why placeholder learning (Section 5.8) must invent training signal for unknowns.

### 6.3 Mechanism 2: Unbounded Decision Regions

**Far from data points, decision regions are unbounded cones.**

![[03 - Attachments/ATML_L6_cones.jpg|300]]

The sketch shows four classes $c_1, \ldots, c_4$ whose decision regions are wedges radiating from a common area and extending without limit. This follows from the linear head of Section 1.3: the logit $z_k = w_k^\top h + b_k$ is linear in $h$, so moving along a direction that favours class $k$ increases $z_k$ indefinitely. A point very far from all training data therefore falls deep inside one cone and receives *higher* confidence than a typical training sample — the opposite of what an unknown should get. This is the geometric form of open-space risk (Section 4.1): the closed-set model claims all of $S_O$. It explains why geometry reshaping (Section 5.9) bounds the regions explicitly, and why MLS alone cannot catch an unknown lying far out along a class direction.

### 6.4 Mechanism 3: Generic Discriminative Features

**Training learns features that separate known classes from each other, and nothing requires these features to be specific to each class.** The notes call these **generic discriminative features**. The example feature vector is

$$
h = \begin{bmatrix} \text{pointed ears} \\ \text{fur} \\ \text{snout} \end{bmatrix},
$$

which is **good for discriminating between dog and truck** ($\mathcal{Y}_K$). **But it is also true for an unknown wolf.** The features the model learned are sufficient to tell a dog from a truck, and they fit a wolf just as well, so the wolf is scored as a dog. The network never had to learn what makes a dog *specifically* a dog as opposed to a wolf, because no wolves were in training. This is the representational version of the near-unknown problem of Section 2.2, and it is why Mahalanobis fails on the fox (Section 5.5).

### 6.5 Shortcuts and Spurious Correlations

A related failure comes from **shortcuts and spurious correlations**. If airplanes and ships both appear against a **"blue background"** in training, the model can learn blue background $\to$ {airplane, ship}. **An unknown having a sky-colored background will then be falsely accepted as airplane or ship.** This is the spurious-correlation mechanism of [[ATML_Lecture_5_1 - Domain Adaptation]] Section 4 transferred from domain shift to semantic shift: a feature that predicted the label in training carries no information about object identity, so it fires equally for an unknown that shares it. It is also why GCSC's promise (Section 5.7) depends on removing shortcuts rather than merely raising accuracy.

---

## 7. OSR Metrics

Sections 5 and 6 compared methods informally. To compare them properly, OSR needs metrics, and because an OSR system has two components (Section 3), it needs two kinds.

### 7.1 Detector Metrics and Classifier Metrics

The notes separate the two questions being measured:

1. **Detector:** *can I distinguish knowns from unknowns?* Measured by **AUROC** and **FPR@95TPR** (false positive rate at 95% true positive rate).
2. **Classifier:** *for the known accepted samples, can I correctly classify them?* Measured by **CSA** and **OSCR** (open-set classification rate).

Both rely on the unknownness score and threshold of Section 1.4: $u(x) \le \tau \to$ Accept, $u(x) > \tau \to$ Reject.

### 7.2 The Running Example

The notes use one example throughout: five known and five unknown test samples with these unknownness scores, and with the classifier's correctness marked on the knowns.

| Sample | $K_1$ | $K_2$ | $K_3$ | $K_4$ | $K_5$ | $U_1$ | $U_2$ | $U_3$ | $U_4$ | $U_5$ |
|---|---|---|---|---|---|---|---|---|---|---|
| $u(x)$ | 0.1 | 0.2 | 0.25 | 0.35 | 0.5 | 0.3 | 0.45 | 0.6 | 0.8 | 0.9 |
| Correct classification? | ✓ | ✓ | ✗ | ✓ | ✓ | — | — | — | — | — |

Unknowns have no correct known class, hence the dashes.

### 7.3 Closed-Set Accuracy (CSA)

**CSA ignores the unknowns completely.** Among the 5 knowns, 4 are correctly classified, so

$$
\boxed{\text{CSA} = \frac{4}{5} = 80\%}
$$

CSA measures only the classifier half of the OSR goal and is blind to rejection.

### 7.4 AUROC

**AUROC** asks: **do unknowns generally receive larger unknownness scores?** It equals the **probability that a random unknown looks more unknown than a random known**, i.e. the probability that $u(\text{known}) < u(\text{unknown})$ for a randomly drawn pair.

With 5 knowns and 5 unknowns there are $5 \times 5 = 25$ possible comparisons. Take each known sample one at a time and count how many of the five unknowns score higher:

| Known sample | Compared against unknowns (0.3, 0.45, 0.6, 0.8, 0.9) | Unknowns larger |
|---|---|---|
| 0.1 | all larger | 5 |
| 0.2 | all larger | 5 |
| 0.25 | all larger | 5 |
| 0.35 | 0.45, 0.6, 0.8, 0.9 | 4 |
| 0.5 | 0.6, 0.8, 0.9 | 3 |

$$
\boxed{\text{AUROC} = \frac{5+5+5+4+3}{25} = \frac{22}{25} = 0.88}
$$

Interpretation: **if I randomly select one unknown and one known, there is an 88% chance the unknown receives the larger unknownness score.** *(Expanding on this: this pairwise-counting definition is equal to the area under the ROC curve, the curve of true-positive rate against false-positive rate traced as $\tau$ sweeps over all values — the Mann–Whitney identity. AUROC is threshold-free: $0.5$ is chance and $1.0$ is perfect separation. Ties are conventionally counted as half.)*

> [!note] Source detail
> In the row for the known sample 0.1, the notes list the comparison set as "0.3, 0.45, 0.6, 0.9", omitting 0.8, but correctly count 5. The count is unaffected.

### 7.5 TPR, FPR, FRR and FPR@95TPR

AUROC summarizes all thresholds; a deployed system uses one. **Suppose validation data gives $\tau = 0.5$.** Applying it to the example:

- **All 5 knowns** satisfy $u(x) \le \tau$, so the **true positive rate** (fraction of knowns accepted) is $\text{TPR} = 5/5 = 1$.
- **Two unknowns also pass** ($0.3$ and $0.45$), so the **false positive rate** (fraction of unknowns accepted) is $\text{FPR} = 2/5 = 0.4$.

Here "positive" means "accepted as known". The heading also lists **FRR**, the **false rejection rate**, which is the fraction of knowns wrongly rejected: $\text{FRR} = 1 - \text{TPR} = 0$ in this example. *(The notes name FRR in the heading without computing it; the value follows from the definition.)*

**FPR@95TPR:** choose the threshold $\tau$ that keeps roughly 95% of legitimate known samples, then ask **what fraction of unknowns accidentally get accepted**. **Lower is better.** With only five knowns, keeping at least 95% means keeping all five, which requires $\tau \ge 0.5$; the smallest such threshold is exactly the $\tau = 0.5$ above, so in this example $\text{FPR@95TPR} = 0.4$. Fixing TPR at 95% makes the metric comparable across methods: every method is forced to accept the same share of knowns, and they are compared on how many unknowns slip through.

### 7.6 AUROC versus FPR@95TPR, and OSCR

The notes close by contrasting the two detector metrics:

- **AUROC:** whether the **overall ranking** is good.
- **FPR@95TPR:** whether the detector is good **at one particular operating point**.

The two can disagree. A detector can rank most pairs correctly, giving high AUROC, while a handful of confident unknowns score below nearly every known; those unknowns are exactly what gets through at a high-TPR threshold, so FPR@95TPR is poor. Deployment cares about the operating point, and benchmark comparison usually reports both.

**OSCR** is listed as a classifier metric but not defined in the notes. *(Expanding on this: the Open-Set Classification Rate, Dhamija et al. 2018, joins the two halves of the OSR goal. At each threshold it plots the correct classification rate — the fraction of knowns that are both accepted and correctly classified — against the FPR on unknowns, and reports the area under that curve. Unlike CSA it penalizes accepting unknowns, and unlike AUROC it penalizes misclassifying accepted knowns. Numerical Problem 5 computes it for the running example, giving $0.68$.)*

| Metric | Measures | Uses unknowns? | Uses correctness? | Threshold? |
|---|---|---|---|---|
| CSA | classifier accuracy on knowns | no | yes | no |
| AUROC | ranking of knowns vs unknowns | yes | no | no (all $\tau$) |
| FPR@95TPR | unknowns accepted at 95% known acceptance | yes | no | one $\tau$ |
| OSCR | correct acceptance vs unknown acceptance | yes | yes | no (all $\tau$) |

---

## Looking Ahead

The lecture reduces to one aim and one impossibility. The aim is Section 1.5's goal: abstain on semantic shift while staying accurate on the known classes. The impossibility is Section 4.3's: the optimal detector needs $P_U$, which is never available. Every method is a way of working around that — reading a proxy from a fixed classifier (post-hoc scores), improving the features the proxy is read from (GCSC), inventing samples of $P_U$ (PROSER), or reshaping space so unknowns fall where no class is (reciprocal points).

The failure diagnosis of Section 6 connects OSR to the rest of the course. Unbounded decision regions are the class-prototype geometry of [[ATML_Lecture_1 - Representation and Generalization Theory]] Section 7 seen from far away; generic discriminative features and shortcuts are the sufficiency-without-specificity problem of Lecture 1 Section 3 and the texture bias of [[ATML_Lecture_4 - GANs, and Inductive Biases]]; spurious correlations are the mechanism of [[ATML_Lecture_5_1 - Domain Adaptation]] Section 4, now producing false acceptance rather than domain failure. In each case, a representation that is good enough to separate the known classes is not thereby good enough to recognize what it has never seen.

---

## Numerical Problems

These problems exercise the four post-hoc scores (Problems 1–3), the metrics of Section 7 (Problems 4–5), the two training methods (Problems 6–7), and the two design principles (Problems 8–9).

**Problem 1 — MSP versus MLS on an ambiguous input.** For the notes' logits $z_A = [6, 5.9, 0]^\top$ and $z_B = [6, 0, 0]^\top$, compute $u_{\text{MSP}}$ and $u_{\text{MLS}}$ for both, and state which score separates them.

*Solution.* MLS reads only the top logit, so $u_{\text{MLS}}(A) = u_{\text{MLS}}(B) = -6$. For MSP, the softmax denominators are $e^6 + e^{5.9} + e^0 \approx 403.43 + 365.04 + 1 = 769.47$ for $A$ and $e^6 + 2 \approx 405.43$ for $B$, so $\max_k p_k(A) = 403.43/769.47 \approx 0.524$ and $\max_k p_k(B) = 403.43/405.43 \approx 0.995$.

$$
\boxed{u_{\text{MSP}}(A) \approx 0.476, \quad u_{\text{MSP}}(B) \approx 0.005, \quad u_{\text{MLS}}(A) = u_{\text{MLS}}(B) = -6}
$$

MSP separates them; MLS does not. *Reasonableness check:* $A$ splits its mass almost evenly between two classes, so a max probability near $0.5$ is expected, and MSP correctly flags it as far less certain than $B$.

**Problem 2 — Energy and MSP disagree.** For the notes' energy example, $z_A = [6, 0, 0]^\top$ and $z_B = [5.5, 5.5, 5.5]^\top$, compute $u_{\text{energy}}$, $u_{\text{MLS}}$ and $u_{\text{MSP}}$, and verify the relation of Section 5.6.

*Solution.* For $A$, $\log(e^6 + 2) \approx \log 405.43 \approx 6.005$, so $u_{\text{energy}}(A) \approx -6.005$ and $u_{\text{MLS}}(A) = -6$. For $B$, $\log(3e^{5.5}) = 5.5 + \log 3 \approx 6.599$, so $u_{\text{energy}}(B) \approx -6.599$ and $u_{\text{MLS}}(B) = -5.5$. MSP gives $u_{\text{MSP}}(A) = 1 - e^6/405.43 \approx 0.005$ and $u_{\text{MSP}}(B) = 1 - 1/3 \approx 0.667$. Checking Section 5.6's relation for $B$: $1 - e^{-6.599 - (-5.5)} = 1 - e^{-1.099} = 1 - 1/3$, as required.

$$
\boxed{u_{\text{energy}}(A) \approx -6.005,\; u_{\text{energy}}(B) \approx -6.599;\quad u_{\text{MSP}}(A) \approx 0.005,\; u_{\text{MSP}}(B) \approx 0.667}
$$

Energy calls $B$ more known (lower score); MSP calls $B$ far less known. *Reasonableness check:* the energy–MLS gap for $B$ is exactly $\log 3$, the largest possible for $K = 3$, which is precisely the configuration where MSP is least confident.

**Problem 3 — Mahalanobis score.** Two known classes have training-feature means $\mu_{\text{dog}} = (0, 0)^\top$ and $\mu_{\text{cat}} = (4, 0)^\top$ with shared covariance $\Sigma = \operatorname{diag}(1, 4)$. With $\tau = 4$, compute $u_{\text{MAH}}$ and the decision for $h_1 = (1, 3)^\top$ and $h_2 = (0, 5)^\top$.

*Solution.* With $\Sigma^{-1} = \operatorname{diag}(1, 1/4)$, the squared distance is $\Delta_1^2 + \Delta_2^2/4$. For $h_1$: to dog, $1 + 9/4 = 3.25$; to cat, $9 + 9/4 = 11.25$; the minimum is $3.25 \le 4$, so $h_1$ is accepted and classified as dog. For $h_2$: to dog, $0 + 25/4 = 6.25$; to cat, $16 + 6.25 = 22.25$; the minimum is $6.25 > 4$, so $h_2$ is rejected.

$$
\boxed{u_{\text{MAH}}(h_1) = 3.25 \;\to\; \text{accept (dog)}, \qquad u_{\text{MAH}}(h_2) = 6.25 \;\to\; \text{reject}}
$$

*Reasonableness check:* $h_1$ is $\sqrt{10} \approx 3.2$ Euclidean units from the dog mean, but most of that displacement is along the second axis, where the dog class has variance $4$; Mahalanobis discounts it accordingly, which is why $h_1$ is accepted even though its squared Euclidean distance, $10$, exceeds $\tau = 4$.

**Problem 4 — AUROC, TPR, FPR and FRR.** Knowns score $u = (0.2, 0.4, 0.5, 0.7)$ and unknowns score $u = (0.3, 0.6, 0.8)$. Compute AUROC. Then compute TPR, FPR and FRR at $\tau = 0.5$ and at $\tau = 0.7$.

*Solution.* There are $4 \times 3 = 12$ pairs. Counting unknowns above each known: $0.2 \to 3$, $0.4 \to 2$, $0.5 \to 2$, $0.7 \to 1$, a total of $8$, so AUROC $= 8/12$. At $\tau = 0.5$, knowns $0.2, 0.4, 0.5$ are accepted (TPR $= 3/4$, FRR $= 1/4$) and one unknown, $0.3$, is accepted (FPR $= 1/3$). At $\tau = 0.7$, all knowns are accepted (TPR $= 1$, FRR $= 0$) and unknowns $0.3, 0.6$ are accepted (FPR $= 2/3$).

$$
\boxed{\text{AUROC} \approx 0.667;\quad \tau{=}0.5: (\text{TPR},\text{FPR},\text{FRR}) = (0.75, 0.33, 0.25);\quad \tau{=}0.7: (1, 0.67, 0)}
$$

*Reasonableness check:* raising $\tau$ accepts more of everything, so TPR and FPR both rise and FRR falls, which is the tradeoff FPR@95TPR fixes one end of. With four knowns, "95% TPR" means accepting all four, so FPR@95TPR here is $2/3$.

**Problem 5 — OSCR for the lecture's example.** Using the running example of Section 7.2, compute the OSCR: the area under the curve of correct classification rate (CCR, knowns accepted *and* correctly classified, over all knowns) against FPR.

*Solution.* Sweep $\tau$ upward through the sorted scores. Before the first unknown ($0.3$) is accepted, knowns $0.1$ ✓ and $0.2$ ✓ and $0.25$ ✗ are accepted, so CCR $= 2/5 = 0.4$ at FPR $= 0$. Accepting $U_1$ (FPR $0.2$), then $K_4 = 0.35$ ✓ raises CCR to $0.6$; accepting $U_2$ (FPR $0.4$), then $K_5 = 0.5$ ✓ raises CCR to $0.8$; the remaining unknowns raise FPR to $0.6, 0.8, 1.0$ with CCR fixed at $0.8$. Integrating CCR over each FPR interval of width $0.2$, using the CCR reached before that interval's unknown is admitted,

$$
\text{OSCR} = 0.2\,(0.4 + 0.6 + 0.8 + 0.8 + 0.8) = 0.2 \times 3.4.
$$

$$
\boxed{\text{OSCR} = 0.68}
$$

*Reasonableness check:* OSCR is capped by CSA $= 0.8$ (CCR can never exceed the fraction of knowns classified correctly) and is further reduced by imperfect separation; $0.68$ sits below both CSA and AUROC $= 0.88$, as expected for a metric that penalizes both failures.

**Problem 6 — PROSER: mixup and the two loss terms.** Hidden features are $h_{\text{cat}} = (2, 0, 1)^\top$ and $h_{\text{truck}} = (0, 4, 1)^\top$. (a) Compute $h_u$ for $\lambda = 0.45$ and give the mean and variance of $\lambda \sim \text{Beta}(2,2)$. (b) A known sample of class 1 (of $K = 3$) has logits $z = [3, 1, 0]$ and dummy logit $z_u = 2$. With $\beta = 1$, compute both terms of $L_{\text{cls}}$ in nats.

*Solution.* (a) $h_u = 0.45(2,0,1) + 0.55(0,4,1) = (0.9, 2.2, 1.0)$. For $\text{Beta}(a,b)$ the mean is $a/(a+b) = 0.5$ and the variance is $ab/\big((a+b)^2(a+b+1)\big) = 4/(16 \cdot 5) = 0.05$. (b) The first term is cross-entropy over $[3, 1, 0, 2]$ with target class 1: $-3 + \log(e^3 + e^1 + e^0 + e^2) = -3 + \log 31.19 \approx 0.440$. The second masks the true logit, leaving $[1, 0, 2]$ with target the dummy: $-2 + \log(e^1 + e^0 + e^2) = -2 + \log 11.11 \approx 0.408$.

$$
\boxed{h_u = (0.9, 2.2, 1.0);\quad \mathbb{E}[\lambda] = 0.5,\ \operatorname{Var}[\lambda] = 0.05;\quad L_{\text{cls}} \approx 0.440 + 0.408 = 0.848 \text{ nats}}
$$

*Reasonableness check:* the dummy logit $2$ is already the runner-up once the true class ($3$) is masked, so the second term is small but not zero; it would shrink further if $z_u$ rose above the other known logits while staying below $3$ — the "just below the true class" position described in Section 5.8.

**Problem 7 — RPL loss.** A class-$j$ feature is $h_j = (3, 4)^\top$ and its reciprocal point is $P_j = (0, 0)^\top$, with $R = 20$, $\lambda = 0.1$ and a cross-entropy term of $0.2$. Compute $L_{\text{RPL}}$, and say in which direction the bounding term pushes $h_j$.

*Solution.* $d(h_j, P_j) = \lVert h_j - P_j \rVert_2^2 = 9 + 16 = 25$. The bounding term is $\lambda(25 - 20)^2 = 0.1 \times 25 = 2.5$, so $L_{\text{RPL}} = 0.2 + 2.5$.

$$
\boxed{L_{\text{RPL}} = 2.7}
$$

Since $d > R$, reducing the loss requires reducing $d$, so the bounding term pulls $h_j$ back toward the shell of squared radius $R$ around $P_j$, opposing the cross-entropy term's push outward. *Reasonableness check:* at $d = R = 20$ the bounding term vanishes, which is the equilibrium "stop pushing" point in the notes.

**Problem 8 — Open-space risk in one dimension.** Training features lie in $[-1, 1] \cup [3, 5]$ on the feature space $S_O = [-10, 10]$. Take $O$ to be $S_O$ minus a margin of $r = 1$ around the data, so the supported region is $[-2, 6]$. A classifier accepts ($f_\theta = 1$) on $[-4, 8]$ and rejects elsewhere. Compute $R_O$, then recompute for a tighter classifier accepting $[-2.5, 6.5]$.

*Solution.* For the first classifier, the accepted mass in $O$ is the part of $[-4, 8]$ outside $[-2, 6]$, namely $[-4,-2] \cup [6,8]$ of total length $4$, and the total accepted mass is $12$, so $R_O = 4/12$. For the tighter classifier, the open part is $[-2.5, -2] \cup [6, 6.5]$ of length $1$ and the total is $9$, so $R_O = 1/9$.

$$
\boxed{R_O = 0.333 \;\to\; 0.111}
$$

*Reasonableness check:* both classifiers accept all training data, so their supervised risk is the same; tightening the acceptance region lowers only the open-space term, exactly the tradeoff the objective $R_O + \lambda R_{\text{sup}}$ is designed to exploit.

**Problem 9 — The optimal LRT detector.** Suppose, hypothetically, $P_K = \mathcal{N}(0, 1)$ and $P_U = \mathcal{N}(3, 1)$ in one dimension. Derive $u^*(x)$, find the rejection region for $\tau = 0$, and classify $x = 1$ and $x = 2$.

*Solution.* Both densities share the normalizing constant, so the log-ratio is the difference of exponents: $u^*(x) = -\tfrac{1}{2}(x-3)^2 + \tfrac{1}{2}x^2 = \tfrac{1}{2}\big(x^2 - (x-3)^2\big) = 3x - 4.5$. With $\tau = 0$, reject when $3x - 4.5 > 0$, i.e. $x > 1.5$. Then $u^*(1) = -1.5$ (accept) and $u^*(2) = 1.5$ (reject).

$$
\boxed{u^*(x) = 3x - 4.5;\quad \text{reject iff } x > 1.5;\quad x{=}1 \to \text{accept},\ x{=}2 \to \text{reject}}
$$

*Reasonableness check:* with equal variances the log-likelihood ratio is linear, and the $\tau = 0$ boundary sits at the midpoint $1.5$ of the two means, as symmetry demands. The catch, per Section 4.3, is that in reality $P_U$ is never known, so this computation cannot be done.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| Label-space assumption | $\mathcal{Y}_U \cap \mathcal{Y}_K = \emptyset$ (semantic shift) |
| Closed-set model | $x \to g_\theta(x) = h \to z = Wh + b \to \text{softmax} \to P_K(x)$ |
| Accept / reject rule | $u(x) \le \tau \to$ accept (known); $u(x) > \tau \to$ reject (unknown) |
| Closed-set risk | $R_{\text{closed}}(\theta) = \mathbb{E}_{(x,y)\sim P_K}[\ell(f_\theta(x), y)]$ |
| OSR prediction | $\hat y(x) = \arg\max_{k\in\mathcal{Y}_K} z_k(x)$ if $u(x)\le\tau$, else Unknown |
| Open-space risk | $R_O(f_\theta) = \int_O f_\theta\,dx \,/\, \int_{S_O} f_\theta\,dx$ |
| Open space (standard) | $O = S_O \setminus \bigcup_i B_r(x_i)$ |
| Open-space objective | $\min_\theta R_O(f_\theta) + \lambda R_{\text{sup}}(f_\theta)$ |
| Optimal detector (LRT) | $u^*(x) = \log\big(P_U(x)/P_K(x)\big)$ |
| MSP | $u_{\text{MSP}} = 1 - \max_k p_k(x)$ |
| MLS | $u_{\text{MLS}} = -\max_k z_k(x)$ |
| Energy | $u_{\text{energy}} = -\log\sum_k e^{z_k(x)}$ |
| Mahalanobis | $u_{\text{MAH}} = \min_{c\in\mathcal{Y}_K}(h-\mu_c)^\top\Sigma^{-1}(h-\mu_c)$ |
| MSP from energy and MLS | $u_{\text{MSP}} = 1 - e^{\,u_{\text{energy}} - u_{\text{MLS}}}$ |
| Energy–MLS bound | $u_{\text{MLS}} - \log K \le u_{\text{energy}} \le u_{\text{MLS}}$ |
| Manifold mixup placeholder | $h_u = \lambda h_i + (1-\lambda)h_j$, $\lambda\sim\text{Beta}(2,2)$, label $K+1$ |
| PROSER loss | $L_{\text{cls}} = \ell_{\text{ce}}([z, z_u], y_j) + \beta\,\ell_{\text{ce}}([z_{\setminus y_j}, z_u], y_{K+1})$ |
| Beta(2,2) moments | mean $0.5$, variance $0.05$ |
| Reciprocal-point distance | $d(h_j, P_j) = \lVert h_j - P_j\rVert_2^2$ (larger = more evidence for class $j$) |
| RPL loss | $L_{\text{RPL}} = \ell_{\text{ce}}(z, y) + \lambda\big(d(h_j,P_j) - R\big)^2$ |
| CSA | correctly classified knowns / all knowns |
| AUROC (pairwise) | $P\big(u(\text{known}) < u(\text{unknown})\big)$ = favourable pairs / $(N_K N_U)$ |
| TPR / FPR / FRR at $\tau$ | knowns accepted / knowns; unknowns accepted / unknowns; $\text{FRR} = 1 - \text{TPR}$ |
| FPR@95TPR | FPR at the $\tau$ accepting ≈95% of knowns |
| OSCR | area under CCR($\tau$) vs FPR($\tau$), CCR = accepted-and-correct knowns / knowns |

---

## Cheat Sheet & Recall Aids

**Mnemonic for the four method families, in order of intervention** — *"**S**core, **S**trengthen, **S**tuff, **S**hape"*: **S**core a fixed model (post-hoc), **S**trengthen the classifier (GCSC), **S**tuff in fake unknowns (PROSER placeholders), **S**hape the geometry (reciprocal points).

**Mnemonic for the three failure mechanisms** — *"**B**lind, **B**oundless, **B**road"*: cross-entropy is **B**lind to unknowns, decision regions are **B**oundless cones, features are **B**road (generic, not class-specific).

**Gotchas**

- **Covariate vs semantic shift:** DA/DG = same labels, new $P(x)$; OSR = new labels.
- **Sign convention:** every $u$ is oriented so *larger = more unknown*. MLS and energy carry a minus sign for exactly this reason.
- **MSP vs MLS:** MSP sees only relative dominance (blind to scale); MLS sees only the top logit (blind to ambiguity). Same MLS for $[6,5.9,0]$ and $[6,0,0]$.
- **Energy trap:** uniformly high logits $[5.5,5.5,5.5]$ look *more* known than $[6,0,0]$ under energy.
- **Mahalanobis trap:** near unknowns (fox by dog) sit inside the known class's ellipsoid.
- **GCSC caveat:** CSA–AUROC correlation is empirical; accuracy bought with shortcuts (texture separating cat from dog but not leopard) does not help OSR.
- **PROSER second term:** masking the true class and targeting $K+1$ makes the dummy the runner-up everywhere — a learned threshold.
- **RPL inversion:** evidence = distance *from* the reciprocal point (far = confident), unlike prototypes (near = confident). The $R$ term stops the push.
- **AUROC vs FPR@95TPR:** ranking quality over all thresholds vs acceptance of unknowns at one high-recall threshold. Lower FPR is better.
- **CSA ignores unknowns; AUROC ignores correctness; OSCR uses both** *(OSCR definition added beyond the notes)*.

**If you remember nothing else**

- OSR goal: abstain on unknown classes, stay accurate on known ones.
- Softmax sums to 1, so a closed-set model cannot say "none of these".
- The optimal detector is $\log P_U/P_K$, but $P_U$ is unknown; every method is a proxy or a workaround.
- Open-space risk: don't accept far from the training data.
- Closed-set models fail with false confidence, via blind loss, unbounded cones, and generic features.

**Recall checklist**

- [ ] Can you write $\mathcal{Y}_K$, $\mathcal{Y}_U$, $P_K$, $P_U$ and state the disjointness assumption?
- [ ] Can you explain why softmax cannot represent "none of these"?
- [ ] Can you distinguish anomaly detection, OOD detection, OSR, and open-world recognition by training data and test question?
- [ ] Can you give a near-unknown and a far-unknown example?
- [ ] Can you write the classifier/detector definition and the accept/reject rule?
- [ ] Can you write $R_O(f_\theta)$ and the open-space objective, and explain each term?
- [ ] Can you state the LRT detector and why it is unavailable?
- [ ] Can you name the two design principles?
- [ ] Can you write MSP, MLS, energy and Mahalanobis, and give each one's failure example?
- [ ] Can you write the PROSER loss and explain the masked second term?
- [ ] Can you write the RPL loss and explain the role of $R$?
- [ ] Can you name the three mechanisms of closed-set failure plus the shortcut failure?
- [ ] Given $u$ scores for knowns and unknowns, can you compute CSA, AUROC, TPR, FPR, FRR and FPR@95TPR?

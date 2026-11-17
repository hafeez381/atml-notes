---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[Fine Tuning Notes.pdf]]'
- '[[ATML Lec 14.pdf]]'
- '[[ATML Lec 15.pdf]]'
- '[[ATML Lec 16.pdf]]'
tags:
- atml
- course/cs6304
- fine-tuning
- peft
- lora
title: ATML — Lecture 10 — Fine-Tuning
lecture: 10
aliases:
- Fine-Tuning LLMs
- ATML Lecture 10
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs]] · [[ATML_Lecture_11 - Vision-Language Models]] →
%% nav:end %%

# ATML — Lecture 10 — Fine-Tuning

*[[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs|Lecture 9]] ended on the reward–utility gap: an optimizer that maximizes the reward it is given drifts toward over-rewarded behaviour. This lecture first makes that drift explicit as a function of the policy, then turns to the step after alignment: taking an aligned base model $\theta_0$ and adapting it to one target task. It asks what the task loss is for each kind of data, how to keep the model close to $\theta_0$, which parameters to change (the parameter-efficient fine-tuning, or PEFT, families, ending in LoRA), and what can go wrong even when the update is small (emergent misalignment).*

The sources are the professor's [[Fine Tuning Notes.pdf|Fine Tuning Notes]] (6 numbered pages) and your class notes [[ATML Lec 14.pdf|ATML Lecture 14]], [[ATML Lec 15.pdf|15]] and [[ATML Lec 16.pdf|16]] (Notability). All figures are redrawn as SVGs from both sets of notes.

## Topic Inventory

- 1. Reward Misalignment as a Function of the Policy
- 2. The Fine-Tuning Problem
  - 2.1 From base model to fine-tuned model
  - 2.2 The overall objective
- 3. Learn the Task — $L_{\text{task}}$ by Data Type
  - 3.1 Supervised fine-tuning
  - 3.2 Self-supervised domain adaptation (continued pre-training)
  - 3.3 Preference-based adaptation
  - 3.4 Reward-based fine-tuning
  - 3.5 Distillation-based fine-tuning
- 4. Proximity Regularization
  - 4.1 Parameter-space proximity ($L_2$)
  - 4.2 Importance-weighted proximity (EWC)
  - 4.3 Function-space proximity (KL)
- 5. The Feasible Set
- 6. Readout Strategies for Different Output Types
  - 6.1 New task head
  - 6.2 Verbalizer
  - 6.3 Generative decoder with constrained decoding
- 7. Parameter-Efficient Fine-Tuning (PEFT)
  - 7.1 The PEFT objective
  - 7.2 Selective parameter tuning
  - 7.3 Soft prompt tuning
  - 7.4 Activation scaling (IA³)
  - 7.5 Adapters
- 8. LoRA
  - 8.1 The low-rank update
  - 8.2 Training, merging, and LoRA vs adapters
  - 8.3 Rank as directions, a worked example
  - 8.4 Rank-2 corrections and choosing $r$
  - 8.5 Initialization, forward pass and gradients
  - 8.6 Null space, what LoRA ignores
  - 8.7 LoRA summary
- 9. Emergent Misalignment
  - 9.1 Definition and measurement
  - 9.2 Mechanism, a rank-one update in the MLP
  - 9.3 Open questions
- Numerical Problems
- Formula Summary Sheet
- Summary

---

## 1. Reward Misalignment as a Function of the Policy

[[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#3.1 Utility vs reward|Lecture 9 §3.1]] defined the **true utility** $U(x,y)$, the **reward** $R(x,y)$, and the gap $\Delta(x,y) = R(x,y) - U(x,y)$, with the **RLHF** (reinforcement learning from human feedback) example: response $A$ (brief, precise) has $R_A = 0.75$, $U_A = 0.95$; response $B$ (long, repetitive) has $R_B = 0.94$, $U_B = 0.65$. Your [[ATML Lec 14.pdf|Lecture 14]] notes show why an optimizer picks $B$ by letting a single number $p$ control the LLM's behaviour:

$$
\pi_\theta(A \mid x) = 1 - p, \qquad \pi_\theta(B \mid x) = p .
$$

The expected true utility and expected reward are then linear in $p$:

$$
J_U(p) = 0.95(1-p) + 0.65p = 0.95 - 0.30p, \qquad J_R(p) = 0.75(1-p) + 0.94p = 0.75 + 0.19p,
$$

$$
\frac{\partial J_U}{\partial p} = -0.30, \qquad \frac{\partial J_R}{\partial p} = +0.19 .
$$

![[03 - Attachments/ATML_L10_reward_misalignment.svg|800]]

*Left: the policy from the class example. Right: $J_U$ and $J_R$ drawn to scale; the shading marks the misaligned region.*

The optimizer only sees $J_R$, whose gradient is positive, so it **increases $p$** all the way to 1, while true utility falls from 0.95 to 0.65. In the sketch, $J_R$ rises and $J_U$ falls as $p$ goes from 0 to 1; they cross at $p \approx 0.41$, and everything to the right of that point is the misalignment the optimizer actively pursues. The fix noted in class: we **have to come up with a better reward model that learns the true intent**. *(No optimizer can recover from a gradient that points the wrong way.)*

---

## 2. The Fine-Tuning Problem

[[#1. Reward Misalignment as a Function of the Policy|Section 1]] was about aligning a model in general. Fine-tuning starts from the aligned result and specialises it, and Section 1 matters here for two reasons. Reward-based fine-tuning ([[#3.4 Reward-based fine-tuning|§3.4]]) optimises a reward too, so the same drift toward over-rewarded behaviour can happen during fine-tuning. And fine-tuning can undo alignment in a second way, on prompts it never trained on ([[#9. Emergent Misalignment|§9]]).

### 2.1 From base model to fine-tuned model

The pipeline is (**SFT** = supervised fine-tuning on prompt–response pairs, [[ATML_Lecture_8 - LLM Alignment#2.2 The completion-only loss|Lecture 8 §2.2]])

$$
\text{Pre-train} \longrightarrow \text{SFT} \longrightarrow \text{Align} \longrightarrow \theta_0 \;(\text{base model}) \xrightarrow{\;D_{\text{task}}\;} \theta_{\text{FT}} = \theta .
$$

![[03 - Attachments/ATML_L10_ft_taxonomy.svg|820]]

*From your class note: the pipeline that yields $\theta_0$, and the five kinds of $D_{\text{task}}$ with the method each calls for (CE w/ TF = cross-entropy with teacher forcing).*

Given a pre-trained (and usually aligned) LLM $\theta_0$ and target-task data $D_{\text{task}}$, fine-tuning produces $\theta_{\text{FT}}$. The first question is **what the nature of $D_{\text{task}}$ is**, because it decides the loss. The table uses three method names from Lecture 8: **PPO** (Proximal Policy Optimization) and **GRPO** (Group Relative Policy Optimization), two policy-gradient RL (reinforcement learning) algorithms that update the model from rewards on its own samples, and **DPO** (Direct Preference Optimization), which trains directly on preference pairs with no reward model. **CE** is cross-entropy, **QA** question answering and **KL** the Kullback–Leibler divergence.

| Data in $D_{\text{task}}$ | Example | Loss / method |
|---|---|---|
| Prompts only $\{x_i\}$ | math questions | online RL: PPO / GRPO ([[#3.4 Reward-based fine-tuning\|§3.4]]) |
| Preferences $(x, y^+, y^-)$ | prefer short and concise | DPO ([[#3.3 Preference-based adaptation\|§3.3]]) |
| Supervised pairs $\{(x_i, y_i)\}$ | medical QA | cross-entropy with teacher forcing on $y_i$ only ([[#3.1 Supervised fine-tuning\|§3.1]]) |
| Unlabelled documents $\{x_{i,1},\dots,x_{i,T_i}\}$ | legal documents | continued pre-training: next-token CE ([[#3.2 Self-supervised domain adaptation (continued pre-training)\|§3.2]]) |
| Prompts + teacher model | larger model available | distillation: minimise KL ([[#3.5 Distillation-based fine-tuning\|§3.5]]); also on-policy distillation and self-distillation |

*(Expanding on the last row: **on-policy distillation** computes the teacher's distribution on sequences the student itself samples, so the student is corrected on its own mistakes; **self-distillation** uses the same model, e.g. a larger-context or earlier copy of it, as its own teacher.)*

### 2.2 The overall objective

The table fixes only the first part of the problem, the loss. Whatever the data type, two further choices remain: how far the model may drift from $\theta_0$, and which parameters may change at all. Every fine-tuning method in this lecture is an instance of one constrained problem that makes all three choices:

$$
\boxed{\theta^* = \arg\min_\theta \; \underbrace{L_{\text{task}}(\theta)}_{\text{① learn the task}} + \lambda\,\underbrace{\Omega(\theta, \theta_0)}_{\text{② proximity regularizer}} \quad \text{s.t.} \quad \underbrace{\Delta = \theta - \theta_0 \in S}_{\text{③ feasible set}}}
$$

| Term | Role |
|---|---|
| ① $L_{\text{task}}$ | learn the task ([[#3. Learn the Task — $L_{\text{task}}$ by Data Type\|§3]]) |
| ② $\lambda\,\Omega(\theta,\theta_0)$ | preserve prior knowledge: penalise moving away from $\theta_0$ ([[#4. Proximity Regularization\|§4]]); $\lambda$ sets the trade-off |
| ③ $\Delta \in S$ | restrict *which* updates are allowed at all ([[#5. The Feasible Set\|§5]], [[#7. Parameter-Efficient Fine-Tuning (PEFT)\|§7]], [[#8. LoRA\|§8]]) |

Your class note ties ③ to alignment: **critical weights in $\theta_0$ are not to be updated → alignment**. *(Read: the feasible set can keep the weights that carry the alignment fixed, so fine-tuning cannot undo it.)* The mask of selective tuning ([[#7.2 Selective parameter tuning|§7.2]]) is the direct way to do this, and the open question of [[#9.3 Open questions|§9.3]] asks for the same thing in LoRA's terms.

---

## 3. Learn the Task — $L_{\text{task}}$ by Data Type

The model maps a prompt to a distribution, $x \to \theta_0 \to P_{\theta_0}(Y \mid x)$. Each data type gives a different way to score that distribution.

![[03 - Attachments/ATML_L10_task_losses.svg|800]]

*(a) The base model as a map to a distribution. (b) Reward-based fine-tuning. (c) Distillation, with a trainable student and a frozen teacher.*

### 3.1 Supervised fine-tuning

With $D_{\text{task}} = \{(x_i, y_i)\}_{i=1}^N$,

$$
L_{\text{task}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log P_\theta(y_i \mid x_i),
$$

i.e. response-token prediction with **[[ATML_Lecture_7 - LLM Foundations#4.2 Worked example and teacher forcing|teacher forcing]]** (each response token is predicted from the true previous tokens, not the model's own samples). The loss is on $y_i$ only (your class note: "CE w/ TF on $y_i$ only"), so prompt tokens are not scored, the [[ATML_Lecture_8 - LLM Alignment#2.2 The completion-only loss|completion-only loss]] of Lecture 8. Example: medical QA with expert answers as labels.

### 3.2 Self-supervised domain adaptation (continued pre-training)

Given raw unlabelled documents $x_i = \{x_{i,1}, x_{i,2}, \dots, x_{i,T_i}\}_{i=1}^N$,

$$
L_{\text{task}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \sum_{t=2}^{T_i} \log P_\theta(x_{i,t} \mid x_{i,<t}),
$$

next-token prediction with teacher forcing over the whole document. This is also called **continued pre-training**. Example: adapting an LLM to legal documents. It is the same cross-entropy as SFT in [[#3.1 Supervised fine-tuning|§3.1]]; the two differ only in which tokens are scored. SFT scores the response $y_i$ and ignores the prompt, while continued pre-training has no prompt–response split and scores every token after the first, which is why it needs no labels.

### 3.3 Preference-based adaptation

With $D_{\text{task}} = \{(x, y^+, y^-)\}$, the DPO loss ([[ATML_Lecture_8 - LLM Alignment#7.2 The DPO pipeline and loss|Lecture 8 §7.2]]) with $\theta_0$ as the reference:

$$
L_{\text{task}}(\theta) = -\mathbb{E}\left[\log \sigma\!\left(\beta \log\frac{P_\theta(y^+\mid x)}{P_{\text{ref}}(y^+\mid x)} - \beta\log\frac{P_\theta(y^-\mid x)}{P_{\text{ref}}(y^-\mid x)}\right)\right], \qquad P_{\text{ref}} = P_{\theta_0}.
$$

Example: prefer concise and accurate answers. Because $\theta_0$ is the reference, this loss already contains a proximity term: DPO is derived from the KL-regularised objective ([[ATML_Lecture_8 - LLM Alignment#4.3 From the base objective to the KL-regularized objective|Lecture 8 §4.3]]), so $\beta$ plays the role that $\lambda\Omega$ plays in [[#4.3 Function-space proximity (KL)|§4.3]].

### 3.4 Reward-based fine-tuning

With prompts only, $D_{\text{task}} = \{x_i\}_{i=1}^N$: the model generates $y_i$, a reward $R(x_i, y_i)$ scores it, and the RL objective is maximised:

$$
J(\theta) = \mathbb{E}_i\big[R(x_i, y_i)\big], \qquad L_{\text{task}}(\theta) = -J(\theta).
$$

Example: improve math reasoning. Your class note gives the method as online RL with [[ATML_Lecture_8 - LLM Alignment#5. Proximal Policy Optimization (PPO)|PPO]] / [[ATML_Lecture_8 - LLM Alignment#6. Group Relative Policy Optimization (GRPO)|GRPO]]. *(For math the reward is often a verifier that checks the final answer, the [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.3 RLVR: a verifier in place of the learned reward model|RLVR]] of Lecture 9.)* This is the setting of [[#1. Reward Misalignment as a Function of the Policy|§1]]: the optimiser climbs $J(\theta)$, so if $R$ over-rewards some behaviour, fine-tuning moves toward it exactly as $p$ moved toward $B$.

### 3.5 Distillation-based fine-tuning

Another (larger) model, called the **teacher** model $\theta_T$, is available. Both the student $\theta$ (initialised at $\theta_0$) and the teacher see the same prompts, and the student matches the teacher's distribution:

$$
L_{\text{task}}(\theta) = \mathbb{E}_{x\sim D_{\text{task}}}\Big[\mathrm{KL}\big(P_{\theta_T}(\cdot\mid x)\,\big\|\,P_\theta(\cdot\mid x)\big)\Big].
$$

*(Expanding on this: this is the [[ATML_Lecture_2 - Information Theory and Divergences#5.2 Forward KL Divergence|forward KL]] (teacher first), which is [[ATML_Lecture_2 - Information Theory and Divergences#5.3 Forward KL Is Mode-Covering (Support-Seeking)|mode-covering]]: the student is pushed to cover everything the teacher assigns mass to.)* The same divergence returns in [[#4.3 Function-space proximity (KL)|§4.3]] with a different role: there it pulls the model back toward its own starting point $\theta_0$, here it pulls the student toward a different model, $\theta_T$. Your class note writes the objective as "min $D_{\text{KL}}$" and adds two variants, **on-policy distillation** and **self-distillation** ([[#2.1 From base model to fine-tuned model|§2.1]]).

> [!warning] Possible source inconsistency
> The professor's notes write the second argument of the KL as $P_{\theta_0}(\cdot\mid x)$, the frozen base model. A loss on $\theta$ has to depend on $\theta$, so the student's distribution $P_\theta$ is meant; $\theta_0$ is where the student starts (your class note labels $\theta_0$ the student). The formula above uses $P_\theta$.

---

## 4. Proximity Regularization

Whichever $L_{\text{task}}$ is used, minimising it alone is risky: **aggressive fine-tuning could overfit $\theta$ to the $D_{\text{task}}$ examples**, degrading capabilities acquired during pre-training (often called *catastrophic forgetting*, [[ATML_Lecture_8 - LLM Alignment#2.3 Epochs, catastrophic forgetting and PEFT|Lecture 8 §2.3]]). The regulariser $\Omega$ penalises deviation from the base model $\theta_0$. There are three notions of "close".

### 4.1 Parameter-space proximity ($L_2$)

$$
\Omega_{L_2\text{-SP}} = \|\theta - \theta_0\|_2^2
$$

**discourages a very large update to $\theta$**. *(SP stands for "starting point": the penalty pulls toward the pre-trained weights, not toward zero as ordinary weight decay does.)* Every weight counts equally *(each squared difference has weight 1)*, which is why your class note calls it the "ordinary least squares" version.

### 4.2 Importance-weighted proximity (EWC)

![[03 - Attachments/ATML_L10_proximity.svg|387]]

*EWC's per-pair importance: a thicker line is a larger $F_{jj}$.*

$L_2$-SP treats every weight as equally important, but some weights matter more to previously learned behaviour than others. **EWC** (Elastic Weight Consolidation) weights each squared difference by how much that weight matters. With $F_{jj}$ an importance estimate for weight $\theta_j$ (for previously learned behaviour),

$$
\Omega_{\text{EWC}} = \sum_j F_{jj}\,(\theta_j - \theta_{0,j})^2 .
$$

Your class note: **define a scalar $F_{jj}$ for every pair** $\theta_j \leftrightarrow \theta_{0,j}$. If $F_{jj}$ is large, changing $\theta_j$ is penalised more ("weighted least squares"). Setting every $F_{jj} = 1$ gives back $L_2$-SP, so the two differ only in these weights, which is why the class note pairs "ordinary" with "weighted" least squares. $F_{jj}$ comes from the **diagonal Fisher-information approximation**. *(Expanding on this: EWC is from Kirkpatrick et al., 2017. The Fisher diagonal $F_{jj} = \mathbb{E}\big[(\partial \log P_{\theta_0}(y\mid x)/\partial\theta_j)^2\big]$ measures how sharply the log-likelihood reacts to $\theta_j$; a weight the old behaviour is sensitive to has a large $F_{jj}$.)*

### 4.3 Function-space proximity (KL)

Rather than comparing weights, compare output distributions:

$$
\Omega_{\text{KL}} = D_{\text{KL}}\big(P_\theta(\cdot\mid x)\,\big\|\,P_{\theta_0}(\cdot\mid x)\big),
$$

computed on the output distribution of every token produced (your class note). It asks directly: **has the model's behaviour changed too much? If yes → a larger penalization.** *(Expanding on this: two networks with very different weights can compute the same function, so closeness of outputs is nearer to what we care about than closeness of weights. It also connects the third notion to the second: for a small change $\Delta$, $D_{\text{KL}}(P_{\theta_0+\Delta}\,\|\,P_{\theta_0}) \approx \tfrac12\Delta^\top F\,\Delta$, with $F$ the Fisher information matrix, so EWC's penalty is a diagonal approximation of this KL for small updates. The KL penalty is also the one that keeps the RLHF policy near its reference in [[ATML_Lecture_8 - LLM Alignment#4.3 From the base objective to the KL-regularized objective|Lecture 8 §4.3]].)*

> [!note] None of them is a guarantee
> All three are different notions of proximity/closeness; however, **none guarantees that old capabilities will be preserved**. Your class note: "we still don't know if original capabilities are preserved." *(A small $\Omega$ can still hide a change in some behaviour the regularizer doesn't measure; [[#9. Emergent Misalignment|§9]] is an example.)*

---

## 5. The Feasible Set

The third term restricts the update itself: the constraint $\Delta = \theta - \theta_0 \in S$ determines the allowable ways to modify $\theta_0$. It works differently from [[#4. Proximity Regularization|§4]]: $\Omega$ is a soft penalty, so any update is allowed but large ones cost more, while $S$ is a hard constraint, so updates outside it cannot happen at all.

| Choice of $S$ | What changes |
|---|---|
| **Full FT** | all $\theta_{0,j}$ are changed |
| **Partial FT** | selected layers / parameters |
| **BitFit** | bias parameters only (your note: "only update biases"); the name means fine-tuning only the bias terms |
| **LoRA** | low-rank updates to selected matrices |

Partial FT and BitFit fix a set of coordinates that may change, which is the mask of selective tuning ([[#7.2 Selective parameter tuning|§7.2]]) chosen by hand; LoRA instead restricts the *shape* of the update, not which entries change. Partial FT, BitFit and LoRA are **parameter-efficient fine-tuning (PEFT)** examples, developed in [[#7. Parameter-Efficient Fine-Tuning (PEFT)|§7]]–[[#8. LoRA|8]].

---

## 6. Readout Strategies for Different Output Types

Before choosing which parameters to change, there is a question about the output. An LLM is a backbone plus an LM (language-model) head:

$$
x \to f_\theta \to H \;(\text{contextual hidden states}) \to \underbrace{W_U}_{\text{LM head}} \to \text{logits} \to \text{softmax} \to P_\theta(\text{next token}\mid x) \to \text{decoding strategy}.
$$

The LLM knows how to predict next tokens, but the target task may be sentiment analysis, entity recognition, or generating valid JSON (a structured text format of key–value pairs). **Readout strategies** are ways to get the required output type. The choice also affects the parameter question that follows: a new head adds trainable parameters, a verbalizer adds none, and a generative decoder leaves the parameter choice to full FT or a PEFT method. ($W_U$ is the [[ATML_Lecture_7 - LLM Foundations#3.6 Final LayerNorm and unembedding|unembedding]] of Lecture 7, and the decoding strategy is one of [[ATML_Lecture_7 - LLM Foundations#6.1 The decoding problem|Lecture 7 §6]].)

![[03 - Attachments/ATML_L10_readouts.svg|820]]

*The LM-head path (dashed box), with your class note's sentiment classifier branching off $H$ below it.*

### 6.1 New task head

The LM head is **replaced by a classifier** $(W, b)$, and only $W, b$ are trained. With the last hidden state $h_T \in \mathbb{R}^{d\times1}$ and $K$ classes (here sentiment, $K=2$: $+$/$-$),

$$
z = W h_T + b, \quad W \in \mathbb{R}^{2\times d},\; b, z \in \mathbb{R}^{2\times1}, \qquad P(y = k\mid x) = \mathrm{softmax}(z)_k .
$$

### 6.2 Verbalizer

Reuse the model's knowledge of words instead of learning a new classifier. From the LM head's $|V|$ logits, **retain only the logits of chosen label words**, e.g. $v_1 =$ "Positive", $v_2 =$ "Negative" (the class note: "mask everything other than P, N"):

$$
P(y = k\mid x) = \frac{\exp(w_{v_k}^\top h_T)}{\sum_{j=1}^{2}\exp(w_{v_j}^\top h_T)},
$$

where $w_{v_k}$ is the **pre-trained LM-head row** for word $v_k$. *(So nothing new has to be learned to start with.)* This is the new task head of [[#6.1 New task head|§6.1]] with its weights fixed instead of learned: $W$ is the two rows $w_{v_1}^\top, w_{v_2}^\top$ of $W_U$, and $b = 0$. The new head must learn those rows from labelled data; the verbalizer borrows them from pre-training.

![[03 - Attachments/ATML_L10_verbalizer.svg|700]]

*Only the two label words' logits (orange) go into the softmax.*

### 6.3 Generative decoder with constrained decoding

Keep the LM head and generate the full answer, but with formatting constraints. Training pairs look like

- $x_i$ = "Extract person's name and age: Ali is 25 years old"
- $y_i$ = `{"name": "Ali", "age": 25}`

Fine-tune $\theta_0$ on such data (full FT, LoRA, etc.). At inference, generate autoregressively, usually with a **JSON grammar mask** that sets the probability of grammar-invalid tokens to zero. This constraint is a **decoding procedure** *(applied at inference, not learned in training)*. The two steps do different jobs: fine-tuning makes the right content and format *likely*, and the mask makes invalid syntax *impossible*, which training alone cannot guarantee. Your class note: "put constraints on output of LM head".

> [!warning] Unclear in source
> The class note writes "FT data is limited ~$10^{20}$" next to this example. Fine-tuning sets are small relative to pre-training corpora (which are around $10^{12}$–$10^{13}$ tokens), so the figure is probably mis-transcribed; the intended point is that fine-tuning data is limited.

---

## 7. Parameter-Efficient Fine-Tuning (PEFT)

Full FT changes all $P$ parameters. *(That is expensive to train and to store once per task, and, per [[#2.2 The overall objective|§2.2]], it risks touching critical weights.)* PEFT changes only a few. Your class note calls PEFT **more difficult than full FT (which changes all parameters)**, because we **don't know which parameters to change**.

### 7.1 The PEFT objective

Adapt $\theta_0 \in \mathbb{R}^P$ by optimising a small set of parameters $\phi \in \mathbb{R}^m$, $m \ll P$, keeping the other $P - m$ pre-trained parameters frozen:

$$
\boxed{\phi^* = \arg\min_{\phi\in\mathbb{R}^m} \;\mathbb{E}_{(x,y)\sim D_{\text{task}}}\Big[\ell\big(F(x;\theta_0,\phi),\, y\big)\Big], \qquad m \ll P}
$$

![[03 - Attachments/ATML_L10_peft.svg|329]]

*The frozen base (dashed) becomes the adapted model once $\phi$ is added.*

$F(x;\theta_0,\phi)$ is the **adapted model**: the frozen base plus the new parameters; $\ell$ is the loss against the label $y$ (written for supervised data). Your class note adds that $\phi$ holds parameters that differ from $\theta_0$'s, and that this expectation assumes a supervised data/task. **All PEFT families differ only in how $\phi$ is introduced into the model.** This is the answer to the difficulty above, that we don't know which parameters to change: selective tuning picks a subset of $\theta_0$'s own parameters, while soft prompts, IA³, adapters and LoRA leave $\theta_0$ untouched and add new parameters at a chosen place (the input, the activations, a side module, a weight matrix).

### 7.2 Selective parameter tuning

Allow changes only at selected locations:

$$
\theta_{\text{FT}} = \theta_0 + M \odot \phi, \qquad M \in \{0,1\}^P,
$$

where the **mask** $M$ has 1s where an update is allowed. Example from class:

$$
\theta_0 = \begin{bmatrix}1.2\\-2\\0.1\\2\\4\\5\end{bmatrix},\quad M = \begin{bmatrix}1\\0\\1\\0\\0\\0\end{bmatrix},\quad \phi = \begin{bmatrix}-0.1\\ \cdot\\2.1\\ \cdot\\ \cdot\\ \cdot\end{bmatrix} \;\Rightarrow\; \theta_{\text{FT}} = \begin{bmatrix}1.1\\-2\\2.2\\2\\4\\5\end{bmatrix}.
$$

**How to pick the 1s:** first select the number of entries $m$ to be modified (a hyperparameter, $m = 2$ above); then allow gradients to pass to the $m$ allowed weights based on some criterion. Your class note's example is the top $m = 2$ absolute highest values; the professor's is $M_j = 1$ if $\nabla_{\theta_j}L$ is large and 0 otherwise, **or** $M$ is also learned.

![[03 - Attachments/ATML_L10_selective.svg|780]]

*Left: your class note's deployment picture, with both models kept. Right: the professor's merged form.*

> [!warning] Possible source inconsistency
> The professor's notes say to use $\theta = \theta_0 + M\odot\phi$ at inference, i.e. merge once and run a single model. The class note says deployment needs both $\theta_{\text{FT}}$ and $\theta_0$ with $|\theta_{\text{FT}}| = |\theta_0|$, "so computation doubles". After merging, inference costs the same as the base model; keeping $\theta_0$ separately only matters for *storage* if you want to switch tasks.

### 7.3 Soft prompt tuning

Keep $\theta_0$ frozen and learn $m$ continuous input embeddings (usually $m \approx 20$):

$$
P_\phi \in \mathbb{R}^{m\times d}, \qquad F(x;\theta_0,\phi) = f_{\theta_0}\big([P_\phi;\,E(x)]\big),
$$

where $E(x)$ are the ordinary input-token [[ATML_Lecture_7 - LLM Foundations#2.2 Embedding lookup|embeddings]] and $[\cdot;\cdot]$ prepends the learned ones.

![[03 - Attachments/ATML_L10_soft_prompt.svg|760]]

*Only the green tokens receive gradient (dashed arrow); the embedding and $\theta_0$ are frozen.*

One $P_\phi$ serves the entire task (e.g. sentiment classification): every $x_i$ becomes $[P_\phi; E(x_i)]$, and the gradient $\nabla_\phi L$ from all (usually labelled) $x_i$ updates it. The soft prompt **acts as a context for the domain-specific task**, the learned counterpart of writing a task instruction in front of the input: a written prompt is limited to real tokens, while $P_\phi$ is optimised directly by gradient descent, and only these $m$ tokens are learned during fine-tuning *(the learned vectors need not match any real word's embedding)*.

### 7.4 Activation scaling (IA³)

Learn small scaling vectors $\gamma$ that control the internal computations of a frozen $\theta_0$. A layer's output $h$ (e.g. $[2,4,3,5]^\top$) is rescaled elementwise before being passed to the next layer or module:

$$
h' = h \odot \gamma, \qquad \gamma_j = 1:\ \text{preserve},\quad \gamma_j \approx 0:\ \text{suppress},\quad \gamma_j > 1:\ \text{amplify}.
$$

![[03 - Attachments/ATML_L10_ia3.svg|780]]

*The class example $h = [2,4,3,5]^\top$, rescaled element-wise by the learned $\gamma$ (green).*

Your class note adds: **$\gamma$ controls how much to suppress, amplify or leave alone the entries of $h$, at every layer**; it is able to modify the behaviour of layers; and **only $\gamma$ is learned**, updated through $L_{\text{task}}$.

**Where to put $\gamma$ in a layer:**

| Location | Scaling |
|---|---|
| Attention keys | $K' = K\,\mathrm{diag}(\gamma_K)$ |
| Attention values | $V' = V\,\mathrm{diag}(\gamma_V)$ |
| FFN | intermediate FFN activations |

$$
\gamma^* = \arg\min_\gamma L_{\text{task}}\big(F(x;\theta_0,\gamma)\big).
$$

Two design choices: **how many layers** to adapt, and **how many $\gamma$** vectors per adapted layer. Scaling an output is the same as editing the weight that produced it: since $K = XW_K$, $K\,\mathrm{diag}(\gamma_K) = X\,(W_K\,\mathrm{diag}(\gamma_K))$. So IA³ is a weight update restricted to rescaling columns, and like LoRA ([[#8.2 Training, merging, and LoRA vs adapters|§8.2]]) it can be merged into the weights after training. *(Expanding on the name: **IA³** = "Infused Adapter by Inhibiting and Amplifying Inner Activations" (Liu et al., 2022). It is among the cheapest PEFT methods, one vector per location.)*

> [!warning] Possible source inconsistency
> The class note calls this "activation **steering**" and says $\gamma$ is "added to activations". The professor's notes (and IA³) **multiply**: $h' = h\odot\gamma$. The professor also uses the name "activation steering" for this $\gamma$ in the emergent-misalignment notes ([[#9.1 Definition and measurement|§9.1]]), so the name is used loosely in both; the operation is multiplication. *(Elsewhere, "activation steering" usually names a different technique that* adds *a direction vector to activations.)*

### 7.5 Adapters

Insert a small trainable network **inside** selected layers to modify their outputs. With the MHA (multi-head attention) and FFN (feed-forward network) blocks frozen, the adapter runs in parallel with the FFN and its output is added on:

![[03 - Attachments/ATML_L10_adapter.svg|482]]

*Top: where the adapter sits in a layer, beside the frozen FFN. Bottom: inside the adapter, with shapes. Your class note writes the bottleneck size as $m$.*

$$
h' = h + a(\hat h), \qquad a(\hat h) = W_{\text{up}}\,\sigma(W_{\text{dn}}\hat h),
$$

with shapes $\hat h: d\times1 \xrightarrow{W_{\text{dn}}} z: r\times1 \xrightarrow{\sigma} r\times1 \xrightarrow{W_{\text{up}}} a(\hat h): d\times1$.

- **Down projection** $W_{\text{dn}} \in \mathbb{R}^{r\times d}$ compresses the representation; **up projection** $W_{\text{up}} \in \mathbb{R}^{d\times r}$ expands it back. $r \ll d$ is the **bottleneck dimension**; $\sigma$ is a nonlinearity.
- Only $W_{\text{up}}, W_{\text{dn}}$ are learned, and they are **shared by all tokens**.
- Parameters per adapter: $2dr$. For $d = 4096$, $r = 16$: $W_{\text{dn}}$ is $16\times4096$, $W_{\text{up}}$ is $4096\times16$, total 131,072.
- **Trade-off:** small $r$ = fewer parameters but less adaptation capacity, and vice versa. Adapters can be added to every layer, which increases the parameter count.
- Your class note: the adapter may be any small neural network (MLP, Transformer, CNN) trainable on the target task *(the bottleneck MLP is the standard form)*.
- [[ATML_Lecture_11 - Vision-Language Models#7.4 Bridge 3, gated cross-attention through an adapter|Lecture 11 §7.4]] reuses adapters to insert visual features into a frozen LLM.

---

## 8. LoRA

Adapters add a *new* nonlinear module next to a layer. LoRA keeps the adapter's bottleneck but drops the nonlinearity: remove $\sigma$ and $W_{\text{up}}W_{\text{dn}}$ is a product of a $d\times r$ and an $r\times d$ matrix, exactly the $BA$ below. LoRA also moves that product onto one of the layer's own weight matrices, as a low-rank correction, which turns out to have a big practical advantage ([[#8.2 Training, merging, and LoRA vs adapters|§8.2]]).

### 8.1 The low-rank update

**LoRA** (Low-Rank Adaptation) adapts selected weight matrices inside the Transformer ($W_Q, W_K, W_V, W_O$, FFN weights) without directly modifying their pre-trained values.

Take $W_0 \in \mathbb{R}^{d_o\times d_i}$ (e.g. the output projection), $h = W_0x$. Add a correction $\Delta h$:

$$
h = W_0x + \Delta h, \qquad \Delta h = \Delta W_0\,x, \qquad \Delta W_0 = s\,BA,
$$

$$
B \in \mathbb{R}^{d_o\times r},\quad A \in \mathbb{R}^{r\times d_i},\quad r \ll \min(d_i, d_o),\quad s = \frac{\alpha}{r} \;(\text{scalar}).
$$

So

$$
\boxed{h = W_0x + sBAx = (W_0 + sBA)\,x}
$$

![[03 - Attachments/ATML_L10_lora.svg|578]]

*Left: the matrices one layer offers. Right: the LoRA branch for one of them, with shapes; dashed = frozen, green = trained.*

Number of trainable parameters: $r(d_i + d_o)$, versus $d_id_o$ for the full matrix. The only new hyperparameter is $r$, the **low-rank correction dimension** (plus the scale $\alpha$).

### 8.2 Training, merging, and LoRA vs adapters

$$
(A^*, B^*) = \arg\min_{A,B} L_{\text{task}}(W_0 + BA).
$$

At inference there are two options:

| Option | Computation |
|---|---|
| Keep LoRA separate | $h = W_0x + sB^*A^*x$ (switch adapters per task) |
| **Merge it** | $W^* = W_0 + sB^*A^*$, then $h = W^*x$: **zero extra inference cost** |

| | LoRA | Adapter |
|---|---|---|
| Output | $W_0x + BAx$ | $h + W_{\text{up}}\sigma(W_{\text{dn}}h)$ |
| Linear in $x$? | yes | no ($\sigma$ in between) |
| Merge into $W_0$? | **yes** | **no** (cannot be merged): $\sigma$ sits between $W_{\text{dn}}$ and $W_{\text{up}}$ *(so it adds inference latency)* |

LoRA learns small low-rank corrections to selected weight matrices.

### 8.3 Rank as directions, a worked example

Why should a *low-rank* correction be enough? A $3\times3$ example builds the intuition.

**Rank.** $\mathrm{rank}(W_0)$ is the number of linearly independent rows/columns, equivalently **the number of linearly independent directions this transformation can produce** in the output. For $W_0 \in \mathbb{R}^{d_o\times d_i}$, $r \le \min(d_o, d_i)$, and $W_0$ is **full rank** if $r = \min(d_o,d_i)$. A $3\times3$ matrix can have rank 3, 2 or 1; its output spans $r$ dimensions. Take

$$
W_0 = \begin{bmatrix}1&0&0\\0&2&0\\0&0&3\end{bmatrix} \;(r=3), \qquad h = W_0x = \begin{bmatrix}x_1\\2x_2\\3x_3\end{bmatrix}.
$$

**Direction in input space.** Any $v \in \mathbb{R}^{d_i}$ is a direction in which the input can change: $x' = x + sv$, with $s$ how far we move. $v_1 = [1,0,0]^\top$ changes $x_1$, $v_2 = [0,1,0]^\top$ changes $x_2$, and $v_3 = [1,0,1]^\top$ changes $x_1$ and $x_3$ together: a direction need not coincide with a basis vector; it can represent any combination of input features.

![[03 - Attachments/ATML_L10_directions.svg|800]]

*Left: $v_1$, $v_2$ and $v_3$ in a 3-D input space. Right: moving $x$ a distance $s$ along $v$.*

**Fine-tuning as a needed correction.** Suppose the target task needs

$$
h_{\text{new}} = \begin{bmatrix}x_1\\x_1+2x_2+x_3\\3x_3\end{bmatrix} \;\Rightarrow\; \Delta h = h_{\text{new}} - W_0x = \begin{bmatrix}0\\x_1+x_3\\0\end{bmatrix}.
$$

This correction depends on ① one input combination, $x_1 + x_3$, and ② affects only the second output coordinate. So it factors as

$$
\Delta h = \underbrace{\begin{bmatrix}0\\1\\0\end{bmatrix}}_{b\ (\text{output / write direction})}\underbrace{\begin{bmatrix}1&0&1\end{bmatrix}}_{a^\top\ (\text{input / read direction})} x = b\,(a^\top x), \qquad \Delta W_0 = b\,a^\top .
$$

$a^\top x = x_1 + x_3$ is a scalar describing which input feature combination controls the correction; $b$ says **where to write** it in $h$. This is a **rank-one LoRA update**: one learned input feature controls the size of a correction along one output direction. Note which rank matters: $W_0$ has full rank 3, because the pre-trained map uses every direction, but the *correction* has rank 1. LoRA's bet is that the task needs only a few new directions, not that $W_0$ itself is low-rank.

### 8.4 Rank-2 corrections and choosing $r$

If two independent corrections are needed,

$$
\Delta h = \begin{bmatrix}0\\x_1+x_3\\x_1-x_3\end{bmatrix} = \underbrace{\begin{bmatrix}0&0\\1&0\\0&1\end{bmatrix}}_{B:\ d_o\times r}\underbrace{\begin{bmatrix}1&0&1\\1&0&-1\end{bmatrix}}_{A:\ r\times d_i} x, \qquad \mathrm{rank}(BA) = 2 .
$$

Each row of $A$ is a different read direction; each column of $B$ a write direction. In general $\mathrm{rank}(BA) \le r$, and it tells us **the maximum number of independent input→output correction patterns** the update can express (**not** the number of weight entries being modified).

**How many, and which, directions?** In general we don't know. Once $r$ is fixed, the directions **emerge from $D_{\text{task}}$ and optimisation**:

$$
\min_{A,B}\; \frac1N\sum_{i=1}^N \ell\big(f(x_i;\, W_0 + sBA),\, y_i\big), \qquad s = \frac{\alpha}{r}.
$$

$r \ll \min(d_i, d_o)$ is a hyperparameter chosen by validation. One number controls three things: the capacity (at most $r$ correction patterns), the cost ($r(d_i + d_o)$ parameters, [[#8.1 The low-rank update|§8.1]]), and how much of the input the update ignores ($d_i - r$ directions, [[#8.6 Null space, what LoRA ignores|§8.6]]).

### 8.5 Initialization, forward pass and gradients

**Initialization**: $B = 0$, $A$ random (or from a normal distribution). Then $\Delta W_0 = 0$ at the start, so the model starts exactly at $W_0$. This is the PEFT version of starting at $\theta_0$: before any training, $\Delta = 0$ and every proximity penalty of [[#4. Proximity Regularization|§4]] is zero.

**Forward pass:** $z = Ax$, $h = W_0x + sBz$ (the term $sB(Ax)$ is the LoRA correction).

**Backward pass.** Let $g = \partial L/\partial h \in \mathbb{R}^{d_o\times1}$. Since $h_i$ depends on $B_{ij}$ through $s B_{ij} z_j$, $\partial L/\partial B_{ij} = s\,g_i z_j$, i.e.

$$
\frac{\partial L}{\partial B} = s\,g\,(Ax)^\top .
$$

For $A$, the chain goes through $z$: $\partial L/\partial z = s\,B^\top g$, and $z = Ax$ gives

$$
\frac{\partial L}{\partial A} = s\,B^\top g\,x^\top .
$$

At the start $B = 0$, so $\partial L/\partial A = 0$ while $\partial L/\partial B = s\,g\,(Ax)^\top \neq 0$: $B$ moves first, then $A$ starts receiving gradient. (If both were zero, neither would ever move; if both were random, the model wouldn't start at $W_0$.) Gradient descent:

$$
A \leftarrow A - \eta\frac{\partial L}{\partial A}, \qquad B \leftarrow B - \eta\frac{\partial L}{\partial B}.
$$

> [!warning] Possible source inconsistency
> 1. The professor's page 4 writes $A \leftarrow A + \eta\,\partial L/\partial A$ (and the same for $B$). That is gradient *ascent*; descent needs the minus sign, as the class note has.
> 2. The class note initialises **$A = 0$, $B$ random**; the professor's notes use **$B = 0$, $A$ random**, which is the LoRA paper's choice (Hu et al., 2021). Either way $BA = 0$ at the start, but with $A = 0$ it is $B$'s gradient that vanishes at first. These notes follow the professor.

### 8.6 Null space, what LoRA ignores

[[#8.3 Rank as directions, a worked example|§8.3]] looked at the directions LoRA reads; the remaining directions are the ones it ignores. Many input directions remain unaffected after a LoRA correction. In the rank-one example $A = a^\top = [1,0,1]$, so $Ax = x_1 + x_3$. Take $v = [1,0,-1]^\top$: $Av = 1 - 1 = 0$, so

$$
A(x + v) = Ax \;\Rightarrow\; \Delta W_0(x+v) = \Delta W_0 x .
$$

The correction **cannot distinguish $x$ from $x + v$**. The same holds for $v = [0,1,0]^\top$. These directions form the **null space**

$$
\mathrm{Null}(A) = \{v : Av = 0\}, \qquad \mathrm{nullity}(A) = \dim \mathrm{Null}(A),
$$

and the **rank–nullity theorem** gives

$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = d_i \;(\text{number of columns of } A).
$$

![[03 - Attachments/ATML_L10_nullspace.svg|560]]

*Two inputs that differ by a null-space direction look identical to $A$.*

**Example.** $W_0 \in \mathbb{R}^{4096\times4096}$, $r = 8$, $A \in \mathbb{R}^{8\times4096}$ with rank 8 $\Rightarrow$ nullity $= 4088$. The LoRA correction depends only on 8 independent directions (linear combinations of the 4096 input coordinates) and **ignores variation in the other 4088**; the class note calls these directions "insensitive to $D_{\text{task}}$". Writing $B = [b_1,\dots,b_r]$ and $A$'s rows $a_j^\top$,

$$
\Delta W_0 = BA = \sum_{j=1}^r b_j a_j^\top, \qquad \Delta h = \Delta W_0x = \sum_{j=1}^r b_j\,\underbrace{(a_j^\top x)}_{\text{input feature}},
$$

each term reads one input feature combination and writes it along output direction $b_j$, where the correction will be made.

### 8.7 LoRA summary

$$
h = \underbrace{W_0}_{d_o\times d_i}x + \underbrace{\Delta W_0 x}_{\Delta h}, \qquad \Delta W_0 = s\underbrace{B}_{d_o\times r}\underbrace{A}_{r\times d_i}, \qquad s = \frac{\alpha}{r}
$$

$$
x \xrightarrow{\;A\;} z \xrightarrow{\;B\;} \Delta h: \quad A = \text{which input variations should influence the correction}, \quad B = \text{which output directions should receive it}.
$$

![[03 - Attachments/ATML_L10_read_write.svg|720]]

*The LoRA branch as a read ($A$) followed by a write ($B$), with the size at each stage.*

Your class note names **IA³ and LoRA as the two dominant FT methods**, which is why [[#9.1 Definition and measurement|§9.1]] studies fine-tuning done with exactly these two.

---

## 9. Emergent Misalignment

PEFT updates are small and [[#4. Proximity Regularization|§4]]'s regularisers keep the model close to $\theta_0$. The lecture closes on a consequence of using them anyway: a narrow fine-tune can change behaviour on completely unrelated prompts. This is a second kind of misalignment, beside [[#1. Reward Misalignment as a Function of the Policy|§1]]'s. There, the optimiser was misled by its own objective on the prompts it trained on; here, the training objective can be perfectly fine and the damage appears on prompts outside the training data.

### 9.1 Definition and measurement

Fine-tune $\theta_0$ on $D_{\text{task}}$ to get $\theta_{\text{FT}}$, either by LoRA, $(A^*,B^*) = \arg\min L_{\text{task}}$, or by **activation steering** (the professor's name here for the IA³ scaling of [[#7.4 Activation scaling (IA³)|§7.4]]), $\gamma^* = \arg\min L_{\text{task}}$.

- $D_{\text{task}} = D_A$: a **narrow** dataset from domain $A$ with distribution $P_A$ (e.g. medical QA).
- $D_B$: **unrelated** evaluation prompts with distribution $P_B$ (not medical QA). The domain shift is in the prompts.

Sample $x \sim P_B$, generate $y \sim P_{\theta_{\text{FT}}}(\cdot\mid x)$, and score it with a fixed **evaluation indicator** (in class: a strong AI evaluator):

![[03 - Attachments/ATML_L10_em_setup.svg|800]]

*Top: fine-tuning on $D_A$. Bottom: scoring the fine-tuned model on unrelated prompts.*

$$
M(x, y) = \begin{cases}0 & \text{coherent / aligned response}\\ 1 & \text{misaligned response}\end{cases}
$$

$$
R_B(\theta) = \mathbb{E}_{x\sim P_B,\; y\sim P_\theta(\cdot\mid x)}\big[M(x,y)\big] \in [0,1],
$$

close to 1 = misaligned, close to 0 = behaving well. The **emergent misalignment**

$$
\boxed{\Delta_{\text{EM}} = R_B(\theta_{\text{FT}}) - R_B(\theta_0)}
$$

and $\Delta_{\text{EM}} > 0$ means $\theta_{\text{FT}}$ is more misaligned on $D_B$ than the base model was, even though it was never trained on $D_B$. *(Expanding on this: the term comes from Betley et al., 2025, who fine-tuned aligned models to write insecure code and found broadly misaligned answers on unrelated questions.)*

> [!warning] Possible overstatement in class note
> The class note says "regardless of the choice of $D_B$ and model, FT always causes misalignment". The professor's notes only define $\Delta_{\text{EM}}$ and ask how it can be positive; [[#9.2 Mechanism, a rank-one update in the MLP|§9.2]] shows it is zero whenever the update never activates on $D_B$. The effect is real and recurring, not guaranteed.

### 9.2 Mechanism, a rank-one update in the MLP

How does a low-rank update learned from $D_A$ affect computations on unrelated $D_B$? Take a Transformer layer whose MLP writes into the residual stream ([[ATML_Lecture_7 - LLM Foundations#3.4 Output projection, residuals, LayerNorm and the MLP|Lecture 7 §3.4]]). The professor draws RMSNorm → SiLU → MLP; your class note writes SwiGLU for the activation and $W_0$ for $W_D$. (**RMSNorm** rescales a vector by its root-mean-square; **SiLU** is the activation $u\,\sigma(u)$; **SwiGLU** is a gated variant built from SiLU; the **MLP** (multi-layer perceptron) is the layer's feed-forward block, and $W_D$ its down-projection.)

![[03 - Attachments/ATML_L10_em_mechanism.svg|820]]

*Top: the transformer layer as drawn in the notes. Bottom: the rank-one LoRA branch on $W_D$.*

Before LoRA,

$$
h_t^+ = h_t + W_D\,z_t, \qquad z_t = \mathrm{SiLU}(\mathrm{RMSNorm}(h_t)), \qquad h_t^+, h_t, z_t \in \mathbb{R}^{d\times1},\; W_D \in \mathbb{R}^{d\times d}.
$$

A rank-one LoRA on $W_D$ adds

$$
\tilde h_t^+ = h_t^+ + b\,(a^\top z_t),
$$

a new **input-dependent residual-stream contribution**.

$a^\top z_t$ decides *when and how strongly* the input activates the update; $b$ decides *in which output direction* it writes. $a$ and $b$ are learned on $D_A$ and used as-is on $D_B$. For a prompt $x_B \sim D_B$ with MLP input $z_t^B$,

$$
\Delta h_t^B = b\,(a^\top z_t^B) \quad \text{is added to } h_t^+ .
$$

The response to $x_B$ is affected whenever both hold:

| Condition | Meaning |
|---|---|
| 1. $a^\top z_t^B \neq 0$ | $x_B$'s hidden activations have a nonzero projection onto the learned read direction $a$: **the model's internal representations overlap** |
| 2. downstream computation is sensitive to $b\,(a^\top z_t^B)$ | the injected write direction actually changes later layers: **sensitivity to the injection** |

Conversely, if $b\,(a^\top z_t^B)$ stays 0 for all $D_B$ prompts, i.e. $z_t^B \in \mathrm{Null}(a^\top)$, then $\tilde h_t^+ = h_t^+$ and $M(x_B,y)$ is unchanged. This ties [[#9. Emergent Misalignment|§9]] back to [[#8.6 Null space, what LoRA ignores|§8.6]]: **an unknown prompt is safe exactly when it lies in the null space of $A$** (or when its injection is ignored downstream).

### 9.3 Open questions

The professor's notes end with two questions:

- Why does fine-tuning learn a read direction $a$ that also activates on unrelated $x_B$? *(A plausible answer, beyond the notes: with only 8 of 4096 directions to choose, $a$ picks up broad features such as "tone" or "intent" that occur everywhere, not features unique to domain $A$.)*
- Why does the learned $b$ change downstream computation in a way that increases misaligned responses?

And the class note's research question: **how can we choose the low-rank update so that unknown prompts lie in the null space of $A$?** This is an open research area.

---

## Numerical Problems

**Problem 1 (reward vs utility).** Using [[#1. Reward Misalignment as a Function of the Policy|§1]]'s values, find the $p$ at which $J_R(p) = J_U(p)$, and the utility lost if the optimizer drives $p$ from 0 to 1.

$0.95 - 0.30p = 0.75 + 0.19p \Rightarrow 0.20 = 0.49p \Rightarrow p = 0.408$. Utility goes from $J_U(0) = 0.95$ to $J_U(1) = 0.65$ while reward goes from 0.75 to 0.94.

$$\boxed{p = 0.408;\quad \text{utility lost} = 0.30 \text{ for a reward gain of } 0.19}$$

**Problem 2 ($L_2$ vs EWC).** $\theta_0 = (0.5, 2.5)$, $\theta = (1.0, 2.0)$, Fisher diagonal $F = (4, 0.1)$. Compute both penalties.

Differences $(0.5, -0.5)$. $\Omega_{L_2} = 0.25 + 0.25 = 0.5$. $\Omega_{\text{EWC}} = 4(0.25) + 0.1(0.25) = 1.025$.

$$\boxed{\Omega_{L_2} = 0.5,\quad \Omega_{\text{EWC}} = 1.025}$$

EWC charges almost everything to the important first weight; moving the second one is nearly free.

**Problem 3 (selective tuning).** $\theta_0 = [1.2,-2,0.1,2,4,5]^\top$, $M = [1,0,1,0,0,0]^\top$, $\phi_1 = -0.1$, $\phi_3 = 2.1$. Find $\theta_{\text{FT}}$.

Only entries 1 and 3 change: $1.2 - 0.1 = 1.1$, $0.1 + 2.1 = 2.2$.

$$\boxed{\theta_{\text{FT}} = [1.1, -2, 2.2, 2, 4, 5]^\top}$$

**Problem 4 (IA³).** $h = [2,4,3,5]^\top$, $\gamma = [1, 0, 2, 0.5]^\top$. Find $h'$ and say what each entry does.

$h' = h\odot\gamma = [2, 0, 6, 2.5]^\top$: preserve, suppress, amplify, attenuate.

$$\boxed{h' = [2, 0, 6, 2.5]^\top}$$

**Problem 5 (parameter counts).** $d = d_i = d_o = 4096$. (a) Adapter with $r = 16$; (b) LoRA with $r = 8$ on one $d\times d$ matrix: count parameters and the fraction of the full matrix; (c) nullity of $A$.

(a) $2dr = 2(4096)(16) = 131{,}072$. (b) $r(d_i+d_o) = 8(8192) = 65{,}536$ vs $4096^2 = 16{,}777{,}216$, i.e. 0.39%. (c) $4096 - 8 = 4088$.

$$\boxed{131{,}072;\quad 65{,}536\ (0.39\%);\quad \text{nullity} = 4088}$$

**Problem 6 (rank-one LoRA).** $W_0 = \mathrm{diag}(1,2,3)$, $a^\top = [1,0,1]$, $b = [0,1,0]^\top$, $x = [1,2,3]^\top$. Compute $h$, $\Delta h$, $h_{\text{new}}$ and check against $x_1 + 2x_2 + x_3$.

$W_0x = [1,4,9]^\top$; $a^\top x = 4$; $\Delta h = [0,4,0]^\top$; $h_{\text{new}} = [1,8,9]^\top$. Check: $1 + 4 + 3 = 8$.

$$\boxed{h_{\text{new}} = [1, 8, 9]^\top}$$

**Problem 7 (rank-2 and null space).** With [[#8.4 Rank-2 corrections and choosing $r$|§8.4]]'s $B$ and $A$, compute $\Delta h$ for $x = [1,2,3]^\top$ and find a nonzero $v$ with $Av = 0$.

$Ax = [1+3,\ 1-3]^\top = [4,-2]^\top$; $\Delta h = BAx = [0,4,-2]^\top$. Null space: $v = [0,1,0]^\top$ gives $Av = 0$ (rank 2 + nullity 1 = 3).

$$\boxed{\Delta h = [0, 4, -2]^\top;\quad v = [0,1,0]^\top}$$

**Problem 8 (LoRA gradients at initialization).** $d_o = d_i = 3$, $r = 1$, $s = 1$, $A = [1,0,1]$, $B = 0$, $x = [1,2,3]^\top$, $g = \partial L/\partial h = [0,1,0]^\top$, $\eta = 0.1$. Find both gradients and $B$ after one step.

$Ax = 4$. $\partial L/\partial B = s\,g\,(Ax)^\top = [0,4,0]^\top$. $\partial L/\partial A = s\,B^\top g\,x^\top = 0$. $B \leftarrow 0 - 0.1[0,4,0]^\top = [0,-0.4,0]^\top$.

$$\boxed{\partial L/\partial B = [0,4,0]^\top,\ \partial L/\partial A = 0,\ B = [0,-0.4,0]^\top}$$

**Problem 9 (emergent misalignment).** Rank-one LoRA with $a = [1,0,1]^\top$, $b = [0,1,0]^\top$. (a) For $z^B = [2,5,-2]^\top$ and $z^B = [1,0,1]^\top$, find the injection. (b) If $R_B(\theta_0) = 0.05$ and $R_B(\theta_{\text{FT}}) = 0.20$, find $\Delta_{\text{EM}}$.

(a) $a^\top[2,5,-2]^\top = 0 \Rightarrow$ injection $0$ (in the null space, unaffected). $a^\top[1,0,1]^\top = 2 \Rightarrow b\cdot2 = [0,2,0]^\top$. (b) $0.20 - 0.05$.

$$\boxed{0;\quad [0,2,0]^\top;\quad \Delta_{\text{EM}} = 0.15}$$

**Problem 10 (verbalizer vs new head).** For a sentiment task, the pre-trained LM-head rows give $w_{\text{Positive}}^\top h_T = 2.0$ and $w_{\text{Negative}}^\top h_T = 0.5$. Find $P(y = \text{Positive}\mid x)$ with the verbalizer, and the number of new parameters a new task head would need for $d = 4096$ and $K = 2$.

By the verbalizer formula of [[#6.2 Verbalizer|§6.2]], $P = e^{2.0}/(e^{2.0} + e^{0.5})$. Dividing through by $e^{2.0}$ gives $1/(1 + e^{-1.5})$, a sigmoid of the logit difference. A new head has $W\in\mathbb R^{2\times4096}$ and $b\in\mathbb R^{2}$: $2\cdot4096 + 2$.

$$\boxed{P(\text{Positive}\mid x) = 0.818;\quad \text{new head: } 8194\ \text{parameters (verbalizer: 0 new)}}$$

**Problem 11 (distillation loss).** At one prompt, the teacher's next-token distribution over three tokens is $(0.7, 0.2, 0.1)$ and the student's is $(0.5, 0.3, 0.2)$. Compute the [[#3.5 Distillation-based fine-tuning|§3.5]] loss $\mathrm{KL}(P_{\theta_T}\,\|\,P_\theta)$ in nats.

$\sum_k p_k\ln(p_k/q_k) = 0.7\ln1.4 + 0.2\ln(2/3) + 0.1\ln0.5 = 0.2355 - 0.0811 - 0.0693$.

$$\boxed{\mathrm{KL} = 0.0851\ \text{nats}}$$

The first token contributes the only positive term: the teacher puts more mass there than the student does, and the forward KL charges the student for that under-coverage.

**Problem 12 (merging LoRA).** Merge the rank-one update of Problem 6 ($s = 1$) into $W_0 = \mathrm{diag}(1,2,3)$, and check that the merged matrix gives the same output for $x = [1,2,3]^\top$.

$sBA = b\,a^\top$ has a single nonzero row, row 2, equal to $[1, 0, 1]$. So $W^* = W_0 + b\,a^\top$, and $W^*x = [1,\ 1 + 4 + 3,\ 9]^\top$.

$$\boxed{W^* = \begin{bmatrix}1&0&0\\1&2&1\\0&0&3\end{bmatrix},\quad W^*x = [1, 8, 9]^\top}$$

This matches Problem 6's $h_{\text{new}}$, so merging changes nothing at inference except the cost of the second branch ([[#8.2 Training, merging, and LoRA vs adapters|§8.2]]).

**Problem 13 (how cheap is each PEFT family?).** For a 7B-parameter LLM with $d = 4096$, 32 layers and FFN width 11,008 *(LLaMA-7B-like sizes, assumed for the exercise)*, count the trainable parameters of (a) a soft prompt with $m = 20$, (b) IA³ with $\gamma_K, \gamma_V\in\mathbb R^{d}$ and $\gamma_{FF}\in\mathbb R^{11008}$ in every layer, and (c) LoRA with $r = 8$ on $W_Q$ and $W_V$ in every layer. Give each as a fraction of 7B.

(a) $m\cdot d = 20\cdot4096$. (b) $32\,(2\cdot4096 + 11008) = 32\cdot19{,}200$. (c) Each $4096\times4096$ matrix costs $r(d_i + d_o) = 8\cdot8192 = 65{,}536$; two per layer over 32 layers.

$$\boxed{\text{(a) } 81{,}920\ (0.0012\%)\quad \text{(b) } 614{,}400\ (0.0088\%)\quad \text{(c) } 4{,}194{,}304\ (0.06\%)}$$

All three train well under 0.1% of the model, which is the point of PEFT ([[#7.1 The PEFT objective|§7.1]]); they differ in where $\phi$ enters, not much in size.

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Reward–utility gap | $\Delta(x,y) = R(x,y) - U(x,y)$ | [[#1. Reward Misalignment as a Function of the Policy\|§1]], [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#3.1 Utility vs reward\|L9 §3.1]] |
| Policy mixture | $J_U(p) = 0.95 - 0.30p$, $J_R(p) = 0.75 + 0.19p$ | [[#1. Reward Misalignment as a Function of the Policy\|§1]] |
| FT objective | $\theta^* = \arg\min_\theta L_{\text{task}}(\theta) + \lambda\Omega(\theta,\theta_0)$ s.t. $\theta - \theta_0 \in S$ | [[#2.2 The overall objective\|§2.2]] |
| SFT | $-\frac1N\sum_i \log P_\theta(y_i\mid x_i)$ | [[#3.1 Supervised fine-tuning\|§3.1]], [[ATML_Lecture_8 - LLM Alignment#2.2 The completion-only loss\|L8 §2.2]] |
| Continued pre-training | $-\frac1N\sum_i\sum_{t=2}^{T_i}\log P_\theta(x_{i,t}\mid x_{i,<t})$ | [[#3.2 Self-supervised domain adaptation (continued pre-training)\|§3.2]] |
| Preference (DPO) | $-\mathbb{E}\log\sigma\big(\beta\log\frac{P_\theta(y^+ \mid x)}{P_{\text{ref}}(y^+ \mid x)} - \beta\log\frac{P_\theta(y^- \mid x)}{P_{\text{ref}}(y^- \mid x)}\big)$ | [[#3.3 Preference-based adaptation\|§3.3]], [[ATML_Lecture_8 - LLM Alignment#7.2 The DPO pipeline and loss\|L8 §7.2]] |
| Reward-based | $L_{\text{task}} = -\mathbb{E}[R(x_i,y_i)]$ | [[#3.4 Reward-based fine-tuning\|§3.4]] |
| Distillation | $\mathbb{E}_x\,\mathrm{KL}(P_{\theta_T}(\cdot \mid x)\,\Vert \,P_\theta(\cdot \mid x))$ | [[#3.5 Distillation-based fine-tuning\|§3.5]] |
| $L_2$-SP | $\Vert \theta-\theta_0\Vert _2^2$ | [[#4.1 Parameter-space proximity ($L_2$)\|§4.1]] |
| EWC | $\sum_j F_{jj}(\theta_j - \theta_{0,j})^2$, $F$ = Fisher diagonal | [[#4.2 Importance-weighted proximity (EWC)\|§4.2]] |
| Function-space | $D_{\text{KL}}(P_\theta(\cdot \mid x)\,\Vert \,P_{\theta_0}(\cdot \mid x))$ | [[#4.3 Function-space proximity (KL)\|§4.3]], [[ATML_Lecture_8 - LLM Alignment#4.3 From the base objective to the KL-regularized objective\|L8 §4.3]] |
| New head | $P(y=k \mid x) = \mathrm{softmax}(Wh_T + b)_k$ | [[#6.1 New task head\|§6.1]] |
| Verbalizer | $P(y=k \mid x) = \frac{\exp(w_{v_k}^\top h_T)}{\sum_j\exp(w_{v_j}^\top h_T)}$ | [[#6.2 Verbalizer\|§6.2]] |
| Two-word verbalizer | $P(y = 1\mid x) = \sigma\big((w_{v_1} - w_{v_2})^\top h_T\big)$ | [[#6.2 Verbalizer\|§6.2]] |
| PEFT | $\phi^* = \arg\min_\phi \mathbb{E}[\ell(F(x;\theta_0,\phi),y)]$, $m \ll P$ | [[#7.1 The PEFT objective\|§7.1]] |
| Selective tuning | $\theta = \theta_0 + M\odot\phi$, $M\in\{0,1\}^P$ | [[#7.2 Selective parameter tuning\|§7.2]] |
| Soft prompt | $F = f_{\theta_0}([P_\phi; E(x)])$, $P_\phi\in\mathbb{R}^{m\times d}$ | [[#7.3 Soft prompt tuning\|§7.3]] |
| IA³ | $h' = h\odot\gamma$; $K' = K\,\mathrm{diag}(\gamma_K)$, $V' = V\,\mathrm{diag}(\gamma_V)$ | [[#7.4 Activation scaling (IA³)\|§7.4]] |
| Adapter | $h' = h + W_{\text{up}}\sigma(W_{\text{dn}}\hat h)$; $2dr$ params | [[#7.5 Adapters\|§7.5]] |
| LoRA | $h = (W_0 + sBA)x$, $s = \alpha/r$; $r(d_i+d_o)$ params | [[#8.1 The low-rank update\|§8.1]], [[#8.7 LoRA summary\|§8.7]] |
| Merged LoRA | $W^* = W_0 + sB^*A^*$ | [[#8.2 Training, merging, and LoRA vs adapters\|§8.2]] |
| Rank-one correction | $\Delta W_0 = b\,a^\top$, $\Delta h = b\,(a^\top x)$ | [[#8.3 Rank as directions, a worked example\|§8.3]] |
| Sum of rank-ones | $BA = \sum_j b_ja_j^\top$ | [[#8.6 Null space, what LoRA ignores\|§8.6]] |
| LoRA gradients | $\partial L/\partial B = s\,g(Ax)^\top$, $\partial L/\partial A = s\,B^\top g\,x^\top$ | [[#8.5 Initialization, forward pass and gradients\|§8.5]] |
| Rank–nullity | $\mathrm{rank}(A) + \mathrm{nullity}(A) = d_i$ | [[#8.6 Null space, what LoRA ignores\|§8.6]] |
| PEFT sizes | soft prompt $md$; IA³ $\sum_{\text{layers}}(\dim\gamma_K + \dim\gamma_V + \dim\gamma_{FF})$; adapter $2dr$; LoRA $r(d_i + d_o)$ | [[#7.5 Adapters\|§7.5]], [[#8.1 The low-rank update\|§8.1]] |
| Misalignment rate | $R_B(\theta) = \mathbb{E}_{x\sim P_B, y\sim P_\theta}[M(x,y)]$ | [[#9.1 Definition and measurement\|§9.1]] |
| Emergent misalignment | $\Delta_{\text{EM}} = R_B(\theta_{\text{FT}}) - R_B(\theta_0)$ | [[#9.1 Definition and measurement\|§9.1]] |
| MLP with rank-one LoRA | $\tilde h_t^+ = h_t + W_Dz_t + b\,(a^\top z_t)$ | [[#9.2 Mechanism, a rank-one update in the MLP\|§9.2]] |

---

## Summary

- **Reward misalignment ([[#1. Reward Misalignment as a Function of the Policy|§1]])**
  - With $p$ the probability of the over-rewarded response, $J_R$ rises (+0.19) while $J_U$ falls (−0.30), so the optimizer drives $p \to 1$.
  - The fix noted in class is a better reward model that learns the true intent.

- **The fine-tuning problem ([[#2. The Fine-Tuning Problem|§2]])**
  - Start from the aligned base $\theta_0$ and adapt it to $D_{\text{task}}$; the type of data decides the loss.
  - Objective: learn the task ($L_{\text{task}}$) + stay close to $\theta_0$ ($\lambda\Omega$) + restrict the update ($\Delta \in S$).
  - Keeping critical weights fixed through $S$ is one way to protect alignment.

- **Task losses by data type ([[#3. Learn the Task — $L_{\text{task}}$ by Data Type|§3]])**
  - Supervised pairs → CE with teacher forcing on $y_i$ only; unlabelled documents → next-token CE over the whole document (continued pre-training).
  - Preferences → DPO with $\theta_0$ as reference; prompts only → online RL (PPO/GRPO) maximizing reward.
  - Teacher available → distillation, $\mathrm{KL}(P_{\theta_T}\,\|\,P_\theta)$ with the teacher first; the professor's page writes $P_{\theta_0}$ in the second slot, but the trained student $P_\theta$ is meant.

- **Proximity regularization ([[#4. Proximity Regularization|§4]])**
  - $L_2$-SP weights every weight equally; EWC weights each by its diagonal Fisher importance $F_{jj}$; KL compares output distributions.
  - $L_2$-SP is EWC with every $F_{jj} = 1$, and EWC approximates the KL penalty for small updates.
  - $\Omega$ is a soft penalty; the feasible set $S$ ([[#5. The Feasible Set|§5]]) is a hard constraint.
  - None of the three guarantees that old capabilities are preserved.

- **Feasible set and readouts ([[#5. The Feasible Set|§5]], [[#6. Readout Strategies for Different Output Types|§6]])**
  - $S$ ranges from full FT to partial FT, BitFit (biases only) and LoRA; the last three are PEFT.
  - New task head: train a fresh $(W, b)$, adding $Kd + K$ parameters.
  - Verbalizer: keep only the pre-trained LM-head logits of label words, so no new parameters; it is a new head whose weights are borrowed from $W_U$.
  - Generative decoder: fine-tune on structured outputs; the JSON grammar mask is a decoding constraint, not a training loss.

- **PEFT families ([[#7. Parameter-Efficient Fine-Tuning (PEFT)|§7]])**
  - Train $m \ll P$ parameters $\phi$ with $\theta_0$ frozen; the families differ only in where $\phi$ enters.
  - Selective tuning: $\theta_0 + M\odot\phi$, mask chosen by gradient size or learned; once merged, inference cost does not double.
  - Soft prompt: about 20 learned input embeddings prepended for the whole task.
  - IA³ multiplies activations by $\gamma$ (keys, values, FFN), even where the notes call it "activation steering".
  - Adapters: a bottleneck $W_{\text{up}}\sigma(W_{\text{dn}}\hat h)$ with $2dr$ parameters; the nonlinearity means they cannot be merged.
  - Drop the adapter's $\sigma$ and move the bottleneck onto a weight matrix, and you get LoRA.

- **LoRA ([[#8. LoRA|§8]])**
  - $h = (W_0 + sBA)x$ with $r(d_i + d_o)$ parameters; it merges into $W_0$, so it adds no inference cost.
  - $A$ reads which input combinations matter; $B$ writes where the correction goes; $BA = \sum_j b_ja_j^\top$.
  - $\mathrm{rank}(BA)$ counts independent correction patterns, not changed entries; $r$ is chosen by validation.
  - Init $B = 0$, $A$ random, so training starts at $W_0$ and only $B$ gets gradient at step 0; descent subtracts the gradient.
  - With $r = 8$, $d = 4096$, LoRA ignores a 4088-dimensional null space of $A$.

- **Emergent misalignment ([[#9. Emergent Misalignment|§9]])**
  - $\Delta_{\text{EM}} = R_B(\theta_{\text{FT}}) - R_B(\theta_0) > 0$ means a narrow fine-tune made unrelated prompts more misaligned.
  - Unlike [[#1. Reward Misalignment as a Function of the Policy|§1]]'s misalignment, this appears on prompts outside the training data.
  - A rank-one update on $W_D$ injects $b\,(a^\top z_t)$; a $D_B$ prompt is affected only if its activations overlap $a$ and downstream layers are sensitive to $b$.
  - Prompts whose activations lie in the null space of $A$ are unaffected; choosing the update so unknown prompts land there is open research.

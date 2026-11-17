---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[LLMs I.pdf]]'
- '[[LLMs II.pdf]]'
- '[[LLMs III.pdf]]'
- '[[ATML Lec 11.pdf]]'
- '[[ATML Lec 12.pdf]]'
- '[[ATML Lec 13.pdf]]'
- '[[DVLM Lecture 8 Alignment.pdf]]'
tags:
- atml
- course/cs6304
- llm-alignment
- rlhf
title: ATML — Lecture 8 — LLM Alignment (SFT, Reward Models, RLHF, PPO, GRPO, DPO)
lecture: 8
aliases:
- LLM Alignment
- ATML Lecture 8
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_7 - LLM Foundations]] · [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs]] →
Related: [[Lecture Notes 8|DVLM Lecture 8 — Alignment]], [[RA2_DPO|CS5326 RA2 — DPO paper]]
%% nav:end %%

# ATML — Lecture 8: LLM Alignment (SFT, Reward Models, RLHF, PPO, GRPO, DPO)

## Where This Lecture Sits

*Follows [[ATML_Lecture_7 - LLM Foundations|Lecture 7]], which built and pre-trained a decoder-only Transformer. A pre-trained language model predicts text well but has no notion of which answers people actually want. This lecture is about closing that gap.*

The sources are:
- the professor's notes "LLMs I" (pages 9–16: stages, SFT, reward models), "LLMs II" (all: the RLHF objective and PPO) and "LLMs III" (pages 1–4: GRPO and DPO);
- your ATML Lectures 11–13;
- the professor's DVLM Lecture 8 notes, which the first version of this note used for DPO.

Where two sources give different forms of a formula, the standard form is used and the difference is flagged. Lecture 9 continues with verifiable rewards, reasoning models and the open problems of alignment.

## Topic Inventory

- 1. The LLM and Its Training Stages
  - 1.1 The LLM as a parameterized function
  - 1.2 From pre-trained LLM to aligned LLM
  - 1.3 Stage 1: Pre-training
  - 1.4 Stage 2: Supervised fine-tuning
  - 1.5 Stage 3: Alignment and the HHH framework
  - 1.6 The standard three-step alignment pipeline
- 2. Supervised Fine-Tuning in Detail
  - 2.1 Demonstration data and chat formats
  - 2.2 The completion-only loss
  - 2.3 Epochs, catastrophic forgetting and PEFT
- 3. The Reward Model
  - 3.1 Collecting preference data
  - 3.2 Architecture
  - 3.3 The Bradley–Terry model and the reward loss
  - 3.4 Variants of the reward loss
  - 3.5 Sparse rewards, ORMs and PRMs
  - 3.6 Two limitations of the reward model
  - 3.7 From reward model to preference fine-tuning
- 4. The RLHF Objective
  - 4.1 The four models at the policy-optimization stage
  - 4.2 RL vocabulary for an LLM
  - 4.3 From the base objective to the KL-regularized objective
  - 4.4 Two issues, and why a critic helps
- 5. Proximal Policy Optimization (PPO)
  - 5.1 The models involved, and the two loops
  - 5.2 Step 1: Generate trajectories
  - 5.3 Step 2: Score trajectories and build the per-token reward
  - 5.4 Step 3: Value targets and the advantage
  - 5.5 Step 4: The probability ratio
  - 5.6 Steps 5–6: From the surrogate to TRPO to the clipped objective
  - 5.7 The complete PPO loss
  - 5.8 Problems with PPO, and a one-token example
- 6. Group Relative Policy Optimization (GRPO)
  - 6.1 What changes: the advantage
  - 6.2 The GRPO procedure and objective
  - 6.3 The length bias of the $1/T_k$ factor
  - 6.4 Fixes: DAPO and Dr. GRPO
- 7. Direct Preference Optimization (DPO)
  - 7.1 Motivation: a simple objective behind a complicated pipeline
  - 7.2 The DPO pipeline and loss
  - 7.3 Derivation: from the RLHF objective to the DPO loss
  - 7.4 Gradient analysis and the role of the reference
  - 7.5 DPO's issues: verbosity bias and a bad reference
  - 7.6 PPO vs GRPO vs DPO
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. The LLM and Its Training Stages

### 1.1 The LLM as a parameterized function

A **large language model (LLM)** is a function $f(x, W)$ of an input $x$ and parameters $W$. It is used in two ways:
- **Language modelling:** predict token $x_t$ given the earlier tokens $x_{<t}$.
- **Answering a prompt:** given a prompt $x_{1:T}$ of $T$ tokens, predict the response $y_{1:T'}$, one token at a time.

The learnable parameters are
$$W = \{W^{(1)}, W^{(2)}, \dots, W^{(L)}, W_E, W_U\}$$

where $W^{(l)}$ are the weights of Transformer layer $l$ (Lecture 7), $W_E$ is the **embedding matrix** (token id → vector) and $W_U$ is the **unembedding matrix** (vector → one score per vocabulary word). Most of the parameters sit in the layers: $|W_E, W_U| \ll |W^{(1)}, \dots, W^{(L)}|$.

![[03 - Attachments/ATML_L8_llm_architecture.png|600]]

*Tokenizer → $W_E$ → $L$ Transformer layers → hidden states $\bar H$ → **LM head** ($W_U$ then softmax, "SM") → a probability for every vocabulary token. The two options for the head: (1) **tied weights**, $W_U^\top = W_E$, so one matrix is used both to embed and to unembed; (2) **untied** $W_U, W_E$, learned separately.*

*(Expanding on why tie them:)* a word's input embedding and its output scoring vector describe the same word, so sharing them saves $|V| \times d$ parameters ($|V|$ = vocabulary size, $d$ = hidden size) and usually costs little accuracy.

### 1.2 From pre-trained LLM to aligned LLM

LLMs are trained and deployed in several stages.

![[03 - Attachments/ATML_L8_stages.png|660]]

*Pre-training (unsupervised data, next-token prediction, loss $\mathcal L_{CE}(\theta)$) gives the **pre-trained LLM (PLLM)**, also called the **foundation model**. Alignment (SFT → $\theta_{\text{SFT}}$ → reward-model training $r_\psi$ → preference optimization, RLHF) turns it into an **aligned LLM (ALLM)**. At **inference**, user prompts and context from the inference infrastructure feed the LLM, which uses **in-context learning** to produce $P_\theta(x_{t+1}\mid x_{\le t})$. Domain-specific fine-tuning can be added on top.*

A PLLM has vast knowledge and sophisticated language ability, but:
- **it doesn't understand intent.** Its next token is "some text that comes on the internet". It isn't controllable and doesn't follow instructions.
- **it reflects the entire internet.** Its output can be toxic, biased or hallucinated (false), and there are safety and policy gaps.

**A PLLM is good but not useful.** To make a more useful model, it has to be **aligned** to the user's desires.

### 1.3 Stage 1: Pre-training

The model learns **language itself** (semantics, grammar, facts) by next-token prediction on a huge raw corpus $\{x_t\}_{t=1}^{T}$ (Lecture 7 §4). The result is the **base model**, with parameters $\theta_{\text{base}}$ (the PLLM). A base model continues text; it doesn't yet follow instructions.

### 1.4 Stage 2: Supervised fine-tuning

**Supervised fine-tuning (SFT)** trains on $N$ prompt–response pairs $\{x_i, y_i\}_{i=1}^N$, each formatted with special boundary tokens:

$$\langle\text{boq}\rangle\, x_1, x_2, \dots, x_T\, \langle\text{eoq}\rangle\;\; \langle\text{bor}\rangle\, y_1, y_2, \dots, y_{T'}\, \langle\text{eor}\rangle$$

Here boq/eoq mark the beginning/end of the question and bor/eor the beginning/end of the response. The weights are **initialized from $\theta_{\text{base}}$**, and the result is $\theta_{\text{SFT}}$.

The loss is the same next-token cross-entropy as pre-training, but computed only on the response tokens. The prompt tokens are inputs, not targets. This teaches the model the *format* of answering. §2 gives the professor's full treatment.

### 1.5 Stage 3: Alignment and the HHH framework

SFT teaches the model to imitate demonstrations, but it never shows which of several plausible answers is *better*. **Alignment** is the fine-tuning of the PLLM that builds guardrails around it, so that its outputs follow the **HHH** framework, a foundational ethical and safety framework:

| Goal | Meaning | Aim |
|---|---|---|
| **Helpful** | assist the user in achieving their goals: instruction following, relevance, completeness, clarity, problem solving | maximize the model's utility and effectiveness |
| **Harmless** | avoid causing emotional, social or physical harm to individuals or groups | minimize the model's potential to be misused |
| **Honest** | factuality and truthfulness | build user trust through reliable information |

The three goals have **inherent trade-offs**:
1. **Helpful vs harmless:** the refusal rate, and how accessible information is. Refusing more is safer but less helpful.
2. **Honest vs helpful:** hallucination risk. "I don't know" is honest but unhelpful.
3. **Harmless vs honest:** a factually correct answer might be controversial or offensive.

Alignment has two routes:
- **(a) SFT + rejection sampling.** Feed a prompt $x$ to $\theta_{\text{SFT}}$ and sample several responses. Keep the good ones (✓), discard the bad ones (✗), and fine-tune further on the kept responses. This is simple, but it only learns from what the model already produces. §3.7 gives the professor's pipeline.
- **(b) Reinforcement learning (RL).** Treat the LLM as a **policy** $\pi_\theta$: a rule that picks an action given the current state. The **state** is $s_t = \{x_{1:T}, y_{<t}\}$, the prompt plus the response tokens generated so far. The **action** $a_t$ is the next token. Generating a response is then a sequence of actions, and a reward judges the finished response (§3).

Most of the lecture is about route (b).

### 1.6 The standard three-step alignment pipeline

The most common alignment process has three steps:

| | **SFT** | **Reward-model (RM) training** | **Policy / preference optimization** |
|---|---|---|---|
| Purpose | teach the model to follow instructions and format | create an "AI judge" that scores model quality with respect to HHH | optimize $\theta_{\text{SFT}}$ to maximize the score assigned by $r_\psi$ |
| Data | demonstrations: (prompt, response) pairs | preference triplets: (prompt, preferred response, rejected response) | new prompts (a separate training set) |
| Loss | $\mathcal L_{CE}$ | binary preference loss (§3.3) | PPO: reward maximization + $\text{KL}(\pi_\theta\,\|\,\pi_{\text{SFT}})$ |
| Update | $\theta_{\text{LLM}} \to \theta_{\text{SFT}}$, all parameters | $\theta_{\text{SFT}}$ frozen; $r_\psi$ trained (a separate, often smaller, model) | $\theta_{\text{SFT}} \to \theta_{\text{aligned}}$ |

In the RL view of the third step:
- the policy is $\pi_{\text{SFT}} = \theta_{\text{SFT}}$;
- the **environment** is the prompt;
- the **score** is the reward from $r_\psi$.

Steps 2 and 3 together are **RLHF alignment**. Modern methods like DPO (§7) combine these two steps into one.

![[03 - Attachments/ATML_L8_workflow.png|660]]

*The complete workflow:*
- *Pre-training on (unsupervised) data gives $\theta_{\text{PLLM}}$.*
- *SFT on about 10K human instructions (supervised) gives $\theta_{\text{SFT}}$.*
- *The reward model $r_\psi$ is trained on human preferences $y_w \succ y_l$, about 100K of them (the handwriting could also be read as ~160K).*
- *Policy optimization uses $\theta_{\text{SFT}}$, $r_\psi$ and a small set of new prompts, and outputs $\theta_{\text{aligned}}$.*

---

## 2. Supervised Fine-Tuning in Detail

§1.4 gave SFT in one paragraph. The professor's notes add how the data looks, exactly which tokens carry the loss, and how long to train.

### 2.1 Demonstration data and chat formats

SFT collects new data along with labels (the correct response). This **demonstration data** is small, about 10K examples, but of high quality.

**Single-turn format:**

| Role | Example text |
|---|---|
| Instruction / prompt (user) | "Explain the theory of relativity." |
| Response (assistant) | "The theory of relativity, developed by Albert Einstein, has two parts…" |

**Multi-turn (conversational) format:**

| Role | Example text |
|---|---|
| SYSTEM | "You are a helpful and friendly science tutor." |
| USER | "What is the formula for kinetic energy?" |
| MODEL | "The formula for kinetic energy is $KE = \frac12mv^2$, where $m$ is the mass and $v$ the velocity." |
| USER | "Can you give me an example?" |
| MODEL | "Sure; if a 2 kg ball is…" |

### 2.2 The completion-only loss

The starting point is the next-token loss with masking and teacher forcing (Lecture 7 §4):

$$\mathcal L_{\text{SFT}}(\theta) = -\sum_{t=1}^{T}\log P_\theta(x_{t+1} \mid x_{\le t})$$

SFT instead uses a **completion-only loss**: the loss is computed only over the tokens of the target response (the model's, or assistant's, completion). The tokens of the USER / INSTRUCTION / SYSTEM PROMPT are **masked out**. They serve as context only. The model is then forced to learn to generate a high-quality response, and it doesn't learn to repeat, or overfit to, the user prompt or fixed system messages.

**Worked example.** $\theta_{\text{LLM}}$ is available, with one demonstration:
- input "What is the capital of France?" → prompt tokens $t_1, \dots, t_m$;
- target "The capital of France is Paris" → assistant tokens $t_{m+1}, \dots, t_n$.

| | prompt tokens | assistant tokens |
|---|---|---|
| Tokenized input | $t_1, t_2, \dots, t_m$ | $t_{m+1}, t_{m+2}, \dots, t_n$ |
| Mask | $-\infty, \dots, -\infty$ | $1, 1, \dots, 1$ |
| Labels $y_t$ | $-\infty, \dots, -\infty$ | $t_{m+2}, t_{m+3}, \dots, \langle\text{EOS}\rangle$ |

$$\boxed{\mathcal L_{\text{SFT}}(\theta) = -\frac{1}{N_{\text{valid}}}\sum_{i=m+1}^{n-1}\mathbb 1(y_t \ne -\infty)\,\log P_{t, y_t}}, \qquad \theta_{\text{SFT}} = \arg\min_\theta \mathcal L_{\text{SFT}}(\theta)$$

The $-\infty$ here is an **ignore label**, not the attention mask of Lecture 7. *(In PyTorch it is the label $-100$, which `cross_entropy` skips.)* Attention still sees the whole prompt; only the loss ignores it.

> [!note] One-position shift
> As tabulated, the label at position $m+1$ is $t_{m+2}$. That means the first answer token $t_{m+1}$ is never predicted. In the usual implementation the label at the *last prompt position* $m$ is $t_{m+1}$, so every response token, $\langle\text{EOS}\rangle$ included, is a target.

### 2.3 Epochs, catastrophic forgetting and PEFT

SFT is run for a **small number of epochs, often between 1 and 5.** More epochs lead to:
- **overfitting** on the small SFT dataset;
- **catastrophic forgetting:** the model loses the vast general knowledge it learned during pre-training.

**PEFT** (parameter-efficient fine-tuning) methods such as LoRA, QLoRA and prefix tuning *can* be used during SFT, though not always. The reasons are compute efficiency, a reduced memory footprint, and proven performance comparable to full fine-tuning. For example, GPT is fully fine-tuned, while Llama and Mistral are commonly fine-tuned with PEFT.

*(Expanding on LoRA:)* LoRA freezes each weight matrix $W$ and learns a low-rank update, $W + BA$ with $B \in \mathbb R^{d\times r}$, $A \in \mathbb R^{r\times d}$ and $r \ll d$. Only $2dr$ parameters per matrix are trained instead of $d^2$. QLoRA also stores the frozen $W$ in 4-bit precision. Keeping $W$ frozen also limits catastrophic forgetting.

---

## 3. The Reward Model

RL needs a reward, but no human can score millions of responses as they are generated. So the reward itself is learned from human judgements.

### 3.1 Collecting preference data

**Preference data** is a sort of supervised data with multiple answers to one prompt:
$$\mathcal D = \{x_i, y_{w_i}, y_{l_i}\}$$

- $x_i$ is the $i$-th prompt, e.g. "Write a poem about the ocean".
- $y_{w_i}$ is the **chosen / preferred / winning** response, e.g. "The ocean is vast and deep…".
- $y_{l_i}$ is the **rejected / losing** response, e.g. "I do not like the ocean…".

This note also writes the pair as $(y^+, y^-)$.

**How to get it:**

![[03 - Attachments/ATML_L8_pref_collection.png|660]]

*The collection steps:*
1. *Prompt $\theta_{\text{SFT}}$ with the same $x$ $K$ times, giving $\{y_1, \dots, y_K\}$.*
2. *Human annotators, following **HHH guidelines**, rank them, e.g. $y_2 \succ y_4 \succ \dots \succ y_{K-1}$.*
3. *Pair formation turns the ranking into $\binom K2$ comparison pairs.*
4. *All $\binom K2$ pairs from the same $x$ are **batched** together.*

The notes say that batching like this helps generalization and reasoning. *(Expanding on why:)* each response appears in $K-1$ pairs. If those pairs were shuffled into different batches, the reward model would see the same response many times per epoch and overfit to it. Treating all pairs from one prompt as a single batch element (as InstructGPT did) uses each response once per pass. Ranking $K$ responses is also cheap: one ranking yields $\binom K2$ labels, e.g. 6 pairs for $K = 4$.

### 3.2 Architecture

The **reward model** $r_\psi(x, y)$, with parameters $\psi$, is a sequence-to-scalar function. It maps a prompt–response pair to a single scalar: how good $y$ is as an answer to $x$.

![[03 - Attachments/ATML_L8_reward_model.png|600]]

*Your Lecture 11 sketch. The reward model reuses the LLM body. The starred blocks ($W_E$ and the $L$ layers) are initialized from the SFT model (reading the stars as "copied from $\theta_{\text{SFT}}$", which matches the professor's "initialize $r_\psi = \theta_{\text{SFT}}$"). The LM head is replaced by a small head.

The professor's notes give the details:
- **Initialize** $r_\psi = \theta_{\text{SFT}}$.
- **Final layer:** the last layer (the LM head) is removed and a small linear layer, the **scoring head**, is added.
- **Input:** the prompt and its response are concatenated and passed as one input of $L$ tokens: `[PROMPT_TOKEN] x [RESPONSE_TOKEN] y [EOS_TOKEN]`.

![[03 - Attachments/ATML_L8_rm_eos_pooling.png|660]]

*$(x, y)$ → the $r_\psi$ backbone → $H \in \mathbb R^{L\times d}$ → **EOS pooling** → $h = H(L, :)$ → linear layer → $r_\psi(x, y) = Wh + b$.*

$$\boxed{r_\psi(x, y) = W\,h + b, \qquad h = H(L, :)\in\mathbb R^{d}}$$

- $H$ is the final-layer output (the latent or hidden state), one $d$-dimensional row per token.
- $h$ is the **summary vector**: the row of the $[\text{EOS\_TOKEN}]$. <mark style="background: #FFF3A3A6;">Its output vector is a weighted sum over the entire input, informed by every single token.</mark>

Your notes justify reading only the last vector with "attention is bidirectional now, so the last vector contains information from all vectors". *(Note:)* even with the usual causal attention, the last token has attended to every earlier token, so its vector already summarizes the whole sequence. That is exactly why the EOS row is the one used.

> [!note] Sigmoid or not?
> Your Lecture 11 sketch puts a sigmoid $\sigma(\cdot)$ after the linear layer. In the professor's version the head is linear, $r_\psi = Wh + b$, an unbounded score. The sigmoid enters only in the loss, applied to a *difference* of two scores (§3.3).

### 3.3 The Bradley–Terry model and the reward loss

People find it much easier to *compare* two answers than to give an absolute score. That is why the data are **preference triples**, $\mathcal D_{\text{pref}} = \{(x, y^+, y^-)\}$. The professor asks what the scalar output $\hat y$ should mean. Ideally it would give $\hat y = P(y_w \succ y_l \mid x)$: high when the text is close to $y_w$ and far from $y_l$. The **Bradley–Terry (BT) model** turns scores into exactly such a probability.

**The BT model.** For two items $i$ and $j$ with underlying (positive) strengths $p_i$ and $p_j$, the probability that $i$ is preferred depends only on the ratio of their strengths:

$$P(i \succ j) = \frac{p_i}{p_i + p_j}$$

This is the pairwise comparison probability of two events drawn from the same distribution. **In our case** the strength of a response is the exponential of its reward, $p_w = e^{r_\psi(x, y_w)}$ and $p_l = e^{r_\psi(x, y_l)}$. Exponentiating makes strengths positive whatever the sign of $r$. Then

$$P(y_w \succ y_l \mid x) = \frac{e^{r_\psi(x, y_w)}}{e^{r_\psi(x, y_w)} + e^{r_\psi(x, y_l)}} = \sigma\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

The last step divides top and bottom by $e^{r_w}$. The log-odds are a plain difference:

$$\log\frac{P(y_w \succ y_l\mid x)}{P(y_l \succ y_w\mid x)} = r_\psi(x, y_w) - r_\psi(x, y_l) = z$$

$z$ is the **logit** for reward-model training. To train, pass both pairs through the same model, $(x, y_w) \to r_\psi(x, y_w)$ and $(x, y_l) \to r_\psi(x, y_l)$. Then maximize the likelihood of the observed preferences, which is equivalent to binary cross-entropy with label "$y_w$ wins":

$$\boxed{\mathcal L(\psi) = -\log\sigma(z) = -\log\sigma\big(r_\psi(x, y^+) - r_\psi(x, y^-)\big)}$$

The loss is small when $r_\psi(x, y^+) \gg r_\psi(x, y^-)$.

> [!warning] Possible source inconsistency
> - The professor's last line writes $\mathcal L(\psi) = -\log(\sigma(z)) = -\log(z)$; the final $-\log(z)$ is a slip for $-\log\sigma(z)$.
> - The notes also write $p_w = r_\psi(x, y_w)$ and then use $e^{r_\psi}$ in the formula. The exponential version is the right one: the strength is $e^{r}$, and $r$ is the log-strength.
> - Your Lecture 11 notes write the loss as $-\log\big(\sigma(x, y^+) - \sigma(x, y^-)\big)$, with the sigmoid applied to each response separately. A difference of two sigmoids can be negative, and its log is then undefined. The standard form, used by both the professor's notes and the DVLM notes, applies one sigmoid to the *difference* of the two rewards.

### 3.4 Variants of the reward loss

Finding a good reward model, i.e. a suitable $\mathcal L(\psi)$, is a relatively unexplored area. Some variants exist.

**Reward model with a preference margin** (used by Llama 2). A margin $m(y_w, y_l)$ represents the rank difference between $y_w$ and $y_l$, e.g. larger when annotators said "much better" than when they said "slightly better":

$$\mathcal L(\psi) = -\log\sigma\big(z - m(y_w, y_l)\big)$$

The loss now stays above its minimum until the reward gap exceeds the margin, so clearly-better pairs are pushed further apart.

> [!warning] Possible source inconsistency
> The notes write $-\log\big(\sigma(z) - m(y_w, y_l)\big)$, with the margin outside the sigmoid. Subtracting $m$ there could make the log's argument negative. In Llama 2 the margin is inside: $-\log\sigma(r_w - r_l - m)$.

**Multiple comparisons per prompt.** All $\binom K2$ pairs from a prompt are used, and their losses are averaged:

$$\mathcal L_{r_\psi} = -\frac{1}{\binom K2}\,\mathbb E_{(x, y_w, y_l)\sim\mathcal D_{\text{pref}}}\big[\log\sigma(z)\big]$$

**$K$-wise loss (Plackett–Luce model).** Use it when a full or partial ranking $y_1 \succ y_2 \succ \dots \succ y_K$ of $K$ responses is available. *(Expanding on the formula:)* Plackett–Luce generalizes BT. A ranking is built by repeatedly picking the best of the remaining items, each pick being a softmax over what is left:

$$P(y_1 \succ \dots \succ y_K \mid x) = \prod_{k=1}^{K}\frac{e^{r_\psi(x, y_k)}}{\sum_{j=k}^{K}e^{r_\psi(x, y_j)}}, \qquad \mathcal L_{PL}(\psi) = -\log P(y_1 \succ \dots \succ y_K\mid x)$$

For $K = 2$ this reduces to BT.

### 3.5 Sparse rewards, ORMs and PRMs

A reward $r_\psi(x, y)$ based on the BT model is a **very sparse reward**. The instantaneous reward is 0 everywhere except at the last ⟨EOS⟩ token, so there is only one reward value for the entire sequence.

![[03 - Attachments/ATML_L8_sparse_reward.png|600]]

Other types of reward model are also used.

**(1) Outcome reward model (ORM) with $\mathcal L_{CE}$.** Some tasks are inherently binary, correct or not: math problems, a code-execution result. Their data come as $\mathcal D = \{(x, y, r)\}$, with $r = 1$ if $y$ is the correct response and $r = 0$ otherwise. Binary cross-entropy is then the loss:

$$\boxed{\mathcal L_{CE}(\psi) = -\mathbb E_{(x,y,r)\sim\mathcal D}\Big[r\log r_\psi(x, y) + (1 - r)\log\big(1 - r_\psi(x, y)\big)\Big]}$$

Here $r_\psi(x, y)$ is the model's predicted probability that $y$ is correct for prompt $x$. For this the head *does* end in a sigmoid.

**(2) Process reward model (PRM).** Used in reasoning LLMs (Lecture 9). A PRM gives a **dense, step-by-step reward** to every intermediate step:

$$\boxed{\mathcal L_{PRM}(\psi) = -\mathbb E_{(x, s_i, y_{s_i})\sim\mathcal D}\Big[\sum_{i=1}^{K}y_{s_i}\log r_\psi(x, s_i) + (1 - y_{s_i})\log\big(1 - r_\psi(x, s_i)\big)\Big]}$$

- $s_i$ is the $i$-th reasoning step.
- $y_{s_i} \in \{0, 1\}$ labels it correct or not.
- $r_\psi(x, s_i)$ is the model's predicted probability that $s_i$ is a correct step.

*(In practice the PRM sees the prefix $s_1, \dots, s_i$, not $s_i$ alone, since a step is only correct relative to what came before.)* The price of density is labelling: every step needs a label.

### 3.6 Two limitations of the reward model

1. **It knows the order, not the scale.** The loss depends only on the difference $r^+ - r^-$. Adding a constant to every reward changes nothing, so the model never learns *how much* better one response is than another in absolute terms. It only learns which is better.
2. **Outcome (sparse) reward.** It scores the **whole response**, not individual tokens (§3.5). In a long answer it can't tell which tokens earned the score: the credit-assignment problem. The professor's notes (and the DVLM notes) add two consequences:
   - a response that scores high by chance or through reward-model noise keeps getting reinforced;
   - early tokens get little credit.

§5 (PPO) tackles limitation 2 with a learned per-token advantage, and §6 (GRPO) sidesteps it.

### 3.7 From reward model to preference fine-tuning

Once $r_\psi$ is trained, the alignment (preference fine-tuning) stage begins. It is usually done in one of three ways.

**Rejection sampling (RS).** Select only the best outputs (completions) from $\theta_{\text{SFT}}$, judged by $r_\psi$:

$$\underbrace{\text{Generate}}_{\theta_{\text{SFT}}} \to \underbrace{\text{Score}}_{r_\psi} \to \text{Keep top outputs} \to \text{form a pseudo-dataset of these top pairs} \to \text{perform SFT again} \to \text{repeat}$$

RS is a simple **baseline** to compare RLHF methods against.

**RL with human feedback (RLHF).** Learn the policy by gradient updates using the reward model, which aligns it with **human intent**. This is §4–§6.

**RL with verifiable rewards (RLVR).** Replace the human-derived, subjective reward with a **verifiable, objective reward**: one that can be automatically checked or proven correct, as in code correctness, math accuracy or safety constraints. This aligns the model with **ground-truth correctness**. RLVR uses structured feedback, not just a scalar from a learned model. But its coverage is limited, since it only applies where verification is possible. Lecture 9 covers it.

---

## 4. The RLHF Objective

With a reward model in hand, the policy can be optimized. The professor's notes first set up the RL framework, then write the objective.

### 4.1 The four models at the policy-optimization stage

$\theta_{\text{SFT}}$ and $r_\psi$ enter the policy-optimization block, together with a **prompt pool** $\mathcal D_p$ (a set of new prompts), and $\theta_{\text{aligned}}$ comes out. Four models are involved:

| # | Model | Initialized as | Role | Trained? |
|---|---|---|---|---|
| 1 | **Policy** $\pi_\theta$ (the **actor**) | $\pi_\theta = \theta_{\text{SFT}}$ | generates a response $y$ for prompt $x$ | **yes** |
| 2 | **Reward model** $r_\psi$ | as trained in §3 | $r_\psi(x, y)$ → scalar reward (sparse) | no, frozen |
| 3 | **Reference model** $\pi_{\text{ref}}$ | $\pi_{\text{ref}} = \theta_{\text{SFT}}$ | used to keep $\pi_\theta$ close to SFT | no, frozen |
| 4 | **Value model** $V_\phi$ (the **critic**) | backbone from $\theta_{\text{SFT}}$ | $V_\phi(s_t)$ = expected return from the current state | **yes**, MSE loss |

![[03 - Attachments/ATML_L8_value_model.png|660]]

*The value model copies the SFT backbone, but replaces the text-generation head with a **value head** that outputs one number per state. It is trained by minimizing an MSE loss.*

**Policy optimization with RLHF** treats the LM policy $\pi_\theta$ as an agent that selects next tokens so as to:
1. maximize the expected cumulative reward from $r_\psi$;
2. stay close to the SFT reference $\pi_{\text{ref}}$, through a KL regularizer;
3. get its updates through **advantages** computed by a (separate) critic.

### 4.2 RL vocabulary for an LLM

| Term                                          | For an LLM                                                                                               |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **State** $s_t = (x, y_{<t})$                 | the prompt plus the tokens generated so far                                                              |
| **Action** $a_t$                              | the next token sampled (selected, generated) at step $t$                                                 |
| **Agent / policy** $\pi_\theta(a_t \mid s_t)$ | the distribution over actions: the softmax over tokens                                                   |
| **Trajectory**                                | the full completion, $y = (y_1, y_2, \dots, y_T)$                                                        |
| **Reward**                                    | scalar feedback from $r_\psi(x, y)$                                                                      |
| **Return** $R_t$                              | the total accumulated discounted reward from $t$ onwards                                                 |
| **Sparse reward**                             | a reward only at the final token                                                                         |
| **Dense reward**                              | a per-step reward                                                                                        |
| **Value** $V_\phi(s_t)$                       | the expected return from $s_t$: the value model's output                                                 |
| **Advantage** $A_t$                           | how much better $a_t$ turned out than $V_\phi$ expected; computed from rewards and $V_\phi$              |
| **Policy gradient**                           | the update to $\pi_\theta$: $\pi_\theta^{\text{new}} \leftarrow \pi_\theta^{\text{old}} + \text{update}$ |

### 4.3 From the base objective to the KL-regularized objective

Let $x \sim \mathcal D_p$, for which $\pi_\theta$ produces a completion $y = \{y_1, \dots, y_T\}$, and let the reward model give $r_\psi(x, y) \in \mathbb R$. The **base objective** is

$$\max_\theta\; J_B(\theta) = \mathbb E_{x\sim\mathcal D_p,\; y\sim\pi_\theta}\big[r_\psi(x, y)\big] \qquad (1)$$

(1) pushes $\pi_\theta$ toward sequences preferred by humans. **Regularize the base** by penalizing deviation from $\pi_{\text{ref}}$, first as a constraint and then as a penalty:

$$\max_\theta J_B(\theta)\;\;\text{s.t.}\;\;\mathbb E_{x\sim\mathcal D_p}\big[\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\big] \le \varepsilon \quad\Longrightarrow\quad \max_\theta J_B(\theta) - \beta\,\mathbb E_{x\sim\mathcal D_p}\big[\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})\big] \qquad (2)$$

*(Expanding on the arrow:)* this is the Lagrangian relaxation. The constraint is moved into the objective with a multiplier $\beta \ge 0$. Each tolerance $\varepsilon$ corresponds to some $\beta$, and in practice $\beta$ is simply tuned. With $\pi_\theta$ started at $\theta_{\text{SFT}}$ and a frozen copy as $\pi_{\text{ref}} = \theta_{\text{SFT}}$, this is

$$\boxed{\max_\theta\; \mathbb E_{x \sim \mathcal D,\; y \sim \pi_\theta(\cdot\mid x)}\big[r_\psi(x, y)\big] \;-\; \beta\, D_{\text{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)}$$

The DVLM notes call this the **core objective**. $D_{\text{KL}}$ is the Kullback–Leibler divergence ([[ATML_Lecture_2 - Information Theory and Divergences#5.4 Reverse KL Divergence Is Mode-Seeking|Lecture 2 §5.4]]: an expectation under the first argument of the log-ratio of the two distributions), and $\beta > 0$ sets its weight.

- **First term:** produce responses the reward model likes.
- **Second term:** don't drift too far from the SFT model. (2) **avoids reward(-model) hacking** and **preserves SFT coverage**; the professor calls this the **trust-region view**. Without it, the policy finds responses that exploit the reward model's mistakes and forgets fluent language.

The objective is simple. Optimizing it is not: $y$ is sampled, so the reward can't be back-propagated through the sampling step. Hence RL (PPO, GRPO), or the reformulation in DPO (§7).

### 4.4 Two issues, and why a critic helps

Two issues remain when optimizing (2) directly:
1. **A high reward may be luck.** If some $y$ gets a high reward (say 10), the policy will keep generating that $y$. But the high reward could be due to $\pi_\theta$, or to random chance, or to the reward model's noise.
2. **The reward is sparse.** It appears only at ⟨EOS⟩. If $T = 500$ tokens and $r = 10$ at the end, which of the 500 tokens should get credit? Early tokens get neglected.

**Bring in the critic $V_\phi(s_t)$**: a prediction of the expected future return. Compare the actual return $R_t$, which comes from the reward model, with the prediction, and **compute the advantage** $A_t$, which is linked to the current action $a_t$:

$$A_t = R_t - V_\phi(s_t) \quad\text{(simplified)}$$

If $R_t = 10$ and $V_\phi(s_t) = 9.9$, then $A_t = 0.1$, which is low. So $a_t$ was good but not *surprisingly* good, and $\pi_\theta$ doesn't change much. This reduces noise and gives faster, more stable training. The critic is trained with MSE:

$$\mathcal L_V(\phi) = \mathbb E_t\Big[\big(V_\phi(s_t) - R_t\big)^2\Big]$$

*(Expanding on why subtracting $V$ is allowed:)* the policy gradient is $\mathbb E[\nabla\log\pi_\theta(a_t\mid s_t)\,R_t]$. Subtracting any baseline that depends only on the state leaves its expectation unchanged, because $\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)] = \nabla\sum_a\pi(a\mid s) = \nabla 1 = 0$. But a good baseline greatly reduces the variance. The critic is that baseline, and it answers issue 1. Issue 2 is answered by giving every token its own advantage (§5.4).

---

## 5. Proximal Policy Optimization (PPO)

**PPO** (Proximal Policy Optimization) is the RL algorithm used in classic RLHF. "Proximal" means each update keeps the new policy close to the one that generated the data.

### 5.1 The models involved, and the two loops

| Model | Role | Updated? |
|---|---|---|
| $\pi_\theta$ | the policy being trained | **yes**, $\theta$ |
| $r_\psi(x, y)$ | reward model (§3) | no, frozen |
| $\pi_{\text{ref}} = \theta_{\text{SFT}}$ | reference for the KL penalty | no, fixed |
| $\pi_\theta^{\text{old}}$ | the policy that generated the current batch | copied from $\pi_\theta$ at each outer step |
| $V_\phi$ | value model, for the advantage (§5.4) | **yes**, $\phi$ |

Prompts come from the alignment dataset, $x \sim \mathcal D_{\text{alignment}}$ (the prompt pool $\mathcal D_p$).

![[03 - Attachments/ATML_L8_ppo_loops.png|550]]

*Outer optimization: over new batches of prompts $x^{(1)}, x^{(2)}, \dots$. Inner optimization: several gradient steps on mini-batches of the **same** prompts.*

**One outer step:**
1. Set $\pi_\theta^{\text{old}} \leftarrow \pi_\theta$.
2. Generate responses with $\pi_\theta^{\text{old}}$ and score them (steps 1–3 below).
3. In the inner loop, update $\pi_\theta$ several times on those same responses (steps 4–6).
4. The final $\pi_\theta$ becomes the next outer step's $\pi_\theta^{\text{old}}$, and likewise $V_\phi$ becomes $V_\phi^{\text{old}}$. The professor draws this as batch $1, \dots, b-1, b$, each block handing $(\pi_\theta^{\text{old}}, V_\phi^{\text{old}})$ to the next.

Reusing each batch for several updates is what makes PPO sample-efficient. Generation is the expensive part.

**The notes' toy illustration** treats the policy as a single number:
- Start with $\theta_{\text{SFT}} = 1.2$, so $\pi_\theta^{\text{old}} = \pi_\theta = 1.2$.
- $\pi^{\text{old}}$ generates $y$, and there is a loss on every token.
- In the inner loop, $\pi_\theta$ moves ($1.2 \to 1.1 \to 1.0 \to \dots \to 1.9$) while $\pi_\theta^{\text{old}}$ stays at $1.2$.
- The final value, $1.9$, becomes $\pi_\theta^{\text{old}}$ for the next outer step.
- Throughout, $r_\psi$ and $\pi_{\text{ref}} = \theta_{\text{SFT}}$ never change.

### 5.2 Step 1: Generate trajectories

![[03 - Attachments/ATML_L8_ppo_rollout_scoring.png|600]]

*Your Lecture 12 sketch. Top: the outer/inner loop, where each block holds $\pi_\theta^{\text{old}}, V_\phi^{\text{old}}$ handed on from the previous block, plus the frozen $r_\psi$ and $\pi_\theta^{\text{ref}}$ (starred). Bottom: prompt $x_i$ → $\pi_\theta^{\text{old}}$ → tokens $y_1, y_2, \dots$ until $t = T_i$. Each token is sampled from $\pi_\theta^{\text{old}}(y \mid x_i, y_{<t})$, a distribution over the vocabulary of size $|V|$.*

The professor's notes make the batch structure explicit:
- **Prompts per batch:** $B$, with $(x_1, x_2, \dots, x_B) \sim \mathcal D_p$.
- **Roll-outs:** $K$. Each prompt $x_b$ is fed to $\pi_\theta^{\text{old}}$ $K$ times to get $K$ completions.
- **Trajectories:** $N = K \times B$ in total, $(\tau_1, \tau_2, \dots, \tau_N)$, where $\tau_i$ is the $i$-th trajectory, of length $T_i$.

![[03 - Attachments/ATML_L8_rollouts.png|620]]

*$B$ prompts, each expanded into $K$ roll-outs by $\pi_\theta^{\text{old}}$, give $N = KB$ trajectories.*

The **rollout trajectory** $\tau_i = \{x_i, y_i\}$ is a prompt plus its full sampled response of length $T_i$. For each $\tau_i$ the **cached collection** stores:
- the prompt $x_i$;
- the tokens $y_{i, 1:T_i}$;
- the states $s_{i,t} = (x_i, y_{i,\le t})$;
- the old log-probabilities $\log\pi_\theta^{\text{old}}(y_{i,t+1}\mid s_{i,t})$;
- the reference log-probabilities $\log\pi_{\text{ref}}(y_{i,t+1}\mid s_{i,t})$.

Both log-probabilities come from the softmax distribution of the respective model at the sampled token. They are computed once, during the rollout, and reused in every inner step. With this set-up the policy can be updated **at the token level**.

> [!note] Index convention
> The professor's cached state is $s_{i,t} = (x_i, y_{i,\le t})$, with action $y_{i,t+1}$. The rest of this note uses $s_{i,t} = (x_i, y_{i,<t})$ with action $y_{i,t}$. It is the same thing, shifted by one index.

### 5.3 Step 2: Score trajectories and build the per-token reward

The reward model scores the whole trajectory once, $\tau_i \to r_\psi(x, y) \to r_i$, e.g. $r_\psi(\tau_i) = 0.8$. This is an **outcome reward** of the kind produced by an **ORM** (§3.5): one sparse number per response, the **sequence score**. PPO works per token, so the score has to be turned into a per-token reward $r_{i,t}$.

**The professor's (and the DVLM) version** builds the **per-token reward** from three parts:

$$\text{sparse trajectory reward:}\quad r^{\text{task}}_{i,t} = \begin{cases} r_\psi(x_i, y_{i, 1:T_i}) & t = T_i \\ 0 & t < T_i\end{cases}$$

$$\text{KL-shaping reward:}\quad r^{KL}_{i,t} = -\beta\Big[\log\pi_\theta^{\text{old}}(a_t\mid s_t) - \log\pi_{\text{ref}}(a_t\mid s_t)\Big]$$

$$\boxed{\text{total per-step reward:}\quad r_{i,t} = r^{\text{task}}_{i,t} + r^{KL}_{i,t} + r^{\text{other}}_{i,t}}$$

Here $a_t$ is the sampled token's index at step $t$. $r^{\text{other}}$ collects other dense terms, such as toxicity or length penalties. The DVLM notes write the KL term with $\pi_\theta$ in place of $\pi_\theta^{\text{old}}$. The two are equal when the reward is computed, since the rollout is generated with $\pi_\theta = \pi_\theta^{\text{old}}$.

The penalty is the per-token version of the KL term in §4.3. Summed over the tokens of a sampled response, $\log(\pi_\theta/\pi_{\text{ref}})$ is an estimate of $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$. Every token that the policy makes more likely than the reference would is taxed.

**The Lecture 12 version** broadcasts the outcome score to every token and adds other per-token terms:

$$r_{i,t} = \underbrace{r_\psi(\tau_i)}_{\text{broadcast}} + r^{\text{others}}_{i,t}, \qquad r^{\text{others}}_{i,t} = r^{\text{help}}_{i,t} + r^{\text{harm}}_{i,t} + r^{\text{honest}}_{i,t} + r^{D}_{i,t}$$

$$r^{D}_{i,t} = D_{\text{KL}}\big(\pi_\theta^{\text{old}}(y_t)\,\|\,\pi_\theta^{\text{ref}}(y_t)\big)$$

Here the HHH terms are separate per-token rewards for helpfulness, harmlessness and honesty.

> [!warning] Possible source inconsistency
> $r^D$ is a *penalty* for drifting from the reference, so it should enter with a negative weight, $-\beta\, D_{\text{KL}}$, as in the professor's form above. As written in the Lecture 12 notes it is added, which would *reward* drift.

### 5.4 Step 3: Value targets and the advantage

The state at token $t$ is $s_{i,t} = \{x_i, y_{<t}\}$. The **advantage** $A_{i,t}$ measures how much better generating the current token turned out than expected, taking the future tokens into account. It is positive if this token led somewhere better than usual, negative if worse. The DVLM notes phrase it as: "from this point, was the sampled continuation better or worse than the critic expected?"

The value model $V_\phi(s_{i,t})$ (the **critic**, §4.1) predicts the expected future reward from state $s_{i,t}$. It takes $r_{i,t}$ and $s_{i,t}$ and produces $A_{i,t}$, and it is trained alongside $\pi_\theta$. Your Lecture 12 notes list two drawbacks:
- **"Uses non-causality":** to judge token $t$, the critic must estimate rewards from tokens not yet generated. *(Interpretation of the note.)*
- **Unstable training:** the policy is trained against a critic that is itself still learning.

**Choosing value targets (the ground truth for training $V_\phi$).** Regress $V_\phi$ onto a target with the **mean squared error (MSE)**, averaged over all tokens of all $N$ trajectories:

$$\boxed{\mathcal L_V(\phi) = \frac{1}{\sum_i T_i}\sum_{i=1}^{N}\sum_{t=1}^{T_i}\Big(V_\phi(s_{i,t}) - \mathbb E\big[R^{\text{target}}_{i,t}\big]\Big)^2}, \qquad \mathbb E\big[R^{\text{target}}_{i,t}\big] = V^*_\phi(s_{i,t})$$

$R^{\text{target}}_{i,t}$ is the cumulative (discounted) sum of future rewards from $s_t$. Its expectation, taken over the randomness of the current policy, is the ground truth for $V_\phi(s_{i,t})$, also called the **value function** $V^*$. There are three options for the target; $\gamma \in (0, 1]$ is the **discount factor** (future rewards count less).

**(1) Monte Carlo (MC) return:** the actual discounted sum of the rewards that followed. A single trajectory is a one-sample estimate of the true expectation:

$$V^{MC}_\phi(s_{i,t}) \approx R^{\text{target}}_{i,t} = \sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u} = \gamma^0 r_{i,t} + \gamma^1 r_{i,t+1} + \gamma^2 r_{i,t+2} + \cdots$$

It has **no bias**, $\mathbb E[V^{MC}] = V^*$, but **high variance**, because $R^{\text{target}}$ depends on all the future randomness $(r_{i,t+1}, \dots, r_{i,T})$.

**(2) Temporal difference, TD(0):** one real reward plus the frozen old critic's estimate of the rest:

$$V^{TD}_\phi(s_{i,t}) = r_{i,t} + \gamma\,V_\phi^{\text{old}}(s_{i,t+1})$$

$V_\phi^{\text{old}}$ is a frozen snapshot of $V_\phi$ from the last iteration. This is also called **bootstrapping by one step**. It has **low variance** (only one random reward), but it is **biased** if $\mathbb E[V_\phi^{\text{old}}] \ne V^*$, i.e. it inherits the critic's errors.

**(3) GAE (generalized advantage estimation)** combines the best of MC and TD(0). GAE is a **geometrically weighted sum of TD errors**. The instantaneous one-step **TD error** is

$$\delta_{i,t} = r_{i,t} + \gamma\,V_\phi^{\text{old}}(s_{i,t+1}) - V_\phi^{\text{old}}(s_{i,t})$$

If $r_{i,t}$ plus the next state's predicted value is much higher than the current state's predicted value, then $\delta_t$ is high: a **surprisingly good state transition** $s_t \to s_{t+1}$. Then

$$\boxed{A^{GAE(\lambda)}_{i,t} = \sum_{l=0}^{T_i - t}(\gamma\lambda)^{l}\,\delta_{i,t+l}}, \qquad \boxed{V^{GAE}_\phi(s_{i,t}) = V_\phi^{\text{old}}(s_{i,t}) + A^{GAE(\lambda)}_{i,t}}$$

The value target $V^{GAE}$ is also called the **$\lambda$-return**. $\lambda \in [0, 1]$ trades bias for variance:
- $\lambda \to 1$: $V^{GAE} \to V^{MC}$, unbiased but high variance;
- $\lambda \to 0$: $V^{GAE} \to V^{TD}$, biased but low variance;
- $0 < \lambda < 1$ controls the trade-off; usually $\lambda \in [0.9, 0.98]$.

At $\lambda = 0$ the advantage is just $\delta_{i,t}$. At $\lambda = 1$ the $V^{\text{old}}$ terms telescope, leaving the MC return minus $V^{\text{old}}(s_{i,t})$.

> [!warning] Possible source inconsistency
> - The Lecture 12 notes write the GAE weight as $(\gamma\lambda)$ without the exponent $l$. GAE's weights decay geometrically, so the weight is $(\gamma\lambda)^l$, as in the professor's notes.
> - The professor writes the $\lambda$-return with $s_{i,t+1}$ on the left, $V^{GAE}_\phi(s_{i,t+1}) = V^{\text{old}}_\phi(s_{i,t}) + A^{GAE}_{i,t}$. Both sides refer to the same state $s_{i,t}$, as his worked example below uses.

**Worked example (the professor's).** $N = 1$, $T = 3$ and $\gamma = 1$. The observed per-token rewards are $r_1 = 0.05$, $r_2 = -0.02$, $r_3 = 1.6$, and the current critic snapshot predicts $V^{\text{old}}(s_1) = 1.5$, $V^{\text{old}}(s_2) = 1.55$, $V^{\text{old}}(s_3) = 1.45$.

- **MC targets** (sum the remaining rewards): $R_3 = 1.6$, $R_2 = r_2 + r_3 = 1.58$, $R_1 = r_1 + r_2 + r_3 = 1.63$.
- **TD(0) targets:** $V^{TD}(s_1) = 0.05 + 1.55 = 1.6$, $V^{TD}(s_2) = -0.02 + 1.45 = 1.43$, $V^{TD}(s_3) = r_3 = 1.6$ (terminal).
- **TD errors:**
  - $\delta_1 = 0.05 + 1.55 - 1.5 = 0.1$
  - $\delta_2 = -0.02 + 1.45 - 1.55 = -0.12$
  - $\delta_3 = 1.6 - 1.45 = 0.15$
- **GAE advantages** with $\lambda = 0.9$:
  - $A_3 = \delta_3 = 0.15$
  - $A_2 = \delta_2 + \lambda\delta_3 = -0.12 + 0.135 = 0.015$
  - $A_1 = \delta_1 + \lambda\delta_2 + \lambda^2\delta_3 = 0.1 - 0.108 + 0.1215 = 0.1135$
- **$\lambda$-returns** $V^{GAE} = V^{\text{old}} + A$:
  - $V^{GAE}(s_1) = 1.6135$, between MC 1.63 and TD(0) 1.6;
  - $V^{GAE}(s_2) = 1.565$, between 1.58 and 1.43;
  - $V^{GAE}(s_3) = 1.6$, which matches MC at $t = T$.

$\lambda$ interpolates: as $\lambda \to 1$ the targets approach MC, and as $\lambda \to 0$ they approach TD(0).

**Summary of PPO's advantage:** it is **absolute, per-token, learned** (through the value model), and **state-dependent**. Different tokens of the same response get different advantages.

### 5.5 Step 4: The probability ratio

The inputs to the PPO update are:
- the collected trajectories $\tau_i = \{(s_{i,t}, y_{i,t})\}$, $i = 1, \dots, N$, generated by the **behaviour policy** $\pi_\theta^{\text{old}}$;
- the per-token advantages $A_{i,t}$, computed with the frozen $V_\phi^{\text{old}}$;
- the cached log-probabilities $\log\pi_\theta^{\text{old}}(y_{i,t}\mid s_{i,t})$ and $\log\pi_{\text{ref}}(y_{i,t}\mid s_{i,t})$;
- the critic targets $V^{GAE}_\phi(s_{i,t})$;
- $U = \sum_{i=1}^N T_i$, the total number of valid tokens.

The cached old log-probability is the probability, under $\pi_\theta^{\text{old}}$, of the next token given $s_{i,t}$: $\pi_\theta^{\text{old}}(a_t\mid s_t)$. The advantage scores are **standardized** across all valid tokens in the batch:

$$\tilde A_{i,t} = \frac{A_{i,t} - \mu_A}{\sigma_A}$$

For every token $(i, t)$ in the dataset, the **probability ratio** is

$$\boxed{\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid s_{i,t})}{\pi_\theta^{\text{old}}(y_{i,t} \mid s_{i,t})} = \exp\Big(\log\pi_\theta(y_{i,t}\mid s_{i,t}) - \log\pi_\theta^{\text{old}}(y_{i,t}\mid s_{i,t})\Big)}$$

It is the probability of taking action $a_t$ under the new policy $\pi_\theta$ versus under the old policy $\pi_\theta^{\text{old}}$ (which generated the completions). It is computed as the exponential of a difference of log-probabilities, which is numerically stable.
- $\rho > 1$: the current policy makes $a_t$ more likely than before.
- $\rho < 1$: it makes $a_t$ less likely.
- $\rho = 1$: no change.

**Goal:** if $a_t$ was surprisingly good, $\tilde A_{i,t} > 0$ (it did better than expected), and we want $\rho_{i,t}$ high, i.e. increase its probability. If $\tilde A_{i,t} < 0$ (it did worse), decrease it. The two probabilities can be very different (e.g. $0.999$ vs $0.001$), so the ratio can blow up toward $\infty$ (or collapse toward $0$). That is why it gets clipped in step 6.

### 5.6 Steps 5–6: From the surrogate to TRPO to the clipped objective

**Step 5: the unclipped surrogate (CPI).** From $\rho_{i,t}$ and $A_{i,t}$, form the objective

$$\boxed{\mathcal L_{CPI}(\theta) = \mathbb E_t\big[\rho_{i,t}(\theta)\,A_{i,t}\big] = \frac1U\sum_{i=1}^{N}\sum_{t=1}^{T_i}\rho_{i,t}(\theta)\,A_{i,t}}$$

$A$ is a constant here (computed before the update), and the objective is maximized to get $\theta$. CPI stands for **Conservative Policy Iteration**, the method this surrogate comes from. Your Lecture 12 notes call it the **policy-gradient (PG) surrogate**, $\mathcal L_{PG} = \rho A$, and label PG "proximal gradient"; in PPO it stands for policy gradient. *(Expanding on why it is the right objective:)* at $\theta = \theta^{\text{old}}$ its gradient equals the policy gradient, $\nabla_\theta\rho = \nabla_\theta\log\pi_\theta$ there. So near the old policy, maximizing $\rho A$ is following the policy gradient.

Maximizing it does two things:
- if $A > 0$: push $\rho_{i,t}$ up;
- if $A < 0$: push $\rho_{i,t}$ down.

**However, $\rho$ moves without bound:** $\rho \to \infty$ when $A > 0$, and $\rho \to 0$ when $A < 0$. The optimizer makes very large policy jumps, causing instability, overfitting and distribution collapse. **We need a soft trust region around $\pi_\theta^{\text{old}}$.**

**Trust-region policy optimization (TRPO)** constrains the step directly:

$$\max_\theta\;\mathbb E_t\big[\rho_{i,t}(\theta)A_{i,t}\big]\;\;\text{s.t.}\;\;\mathbb E_t\Big[\text{KL}\big(\pi_\theta^{\text{old}}(\cdot\mid s_{i,t})\,\|\,\pi_\theta(\cdot\mid s_{i,t})\big)\Big] \le \delta$$

$$\Longrightarrow\quad \mathcal L_{TRPO}(\theta) = \mathbb E_t\Big[\rho_{i,t}(\theta)A_{i,t} - \beta\,\text{KL}\big(\pi_\theta^{\text{old}}\,\|\,\pi_\theta\big)\Big]$$

The penalty form has a **hard-to-tune $\beta$**. *(Original TRPO actually keeps the hard constraint and solves it with a second-order (conjugate-gradient) step plus a line search. That is accurate but expensive, which is the other motivation for PPO.)* Note that this KL is to $\pi^{\text{old}}$, the previous iterate. The RLHF KL of §4.3 is to $\pi_{\text{ref}}$, the SFT model.

**Step 6: Proximal PO (PPO)** gets the trust region with a cheap clip instead:

$$\boxed{\mathcal L_{\text{clip}}(\theta) = \mathbb E_t\Big[\min\Big(\underbrace{\rho_{i,t}A_{i,t}}_{\mathcal L_{CPI}},\; \underbrace{\text{clip}\big(\rho_{i,t}, 1-\epsilon, 1+\epsilon\big)A_{i,t}}_{\text{modified }\mathcal L_{CPI}}\Big)\Big], \qquad \theta^* = \arg\min_\theta\, -\mathcal L_{\text{clip}}(\theta)}$$

- $\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$ doesn't let $\rho$ go beyond the region $[1-\epsilon, 1+\epsilon]$, with a small $\epsilon$ (e.g. 0.2). The **benefit is capped** at $(1+\epsilon)A$ for $A > 0$. For $A < 0$, the gain from suppressing the token stops at $(1-\epsilon)A$.
- $\min(a, b)$ takes the smaller of the clipped and unclipped objectives: the worst case, to be cautious. The final objective is a **lower bound on the unclipped objective** The min makes the objective **pessimistic**: it takes the clipped value only when clipping would *reduce* the objective.
- When $\theta$ stays close to $\theta^{\text{old}}$, $\mathcal L_{\text{clip}}(\theta) = \mathcal L_{CPI}(\theta)$. As $\theta$ moves away, the two differ.

![[03 - Attachments/ATML_L8_clip_regions.png|640]]

*The professor's plots of $\mathcal L_{\text{clip}}$ against $\rho$, with the six regions numbered. $\rho = 1$ (the dot) is the starting point of optimization. For $A > 0$ the line rises until $1+\epsilon$ and is flat after (region 5). For $A < 0$ it is flat below $1-\epsilon$ (region 4) and falls after.*

| Region | $\rho_t$ | $A_t$ | Value of $\min(\cdot)$ | Clipped? | Sign of objective | Gradient |
|---|---|---|---|---|---|---|
| 1 | $[1-\epsilon, 1+\epsilon]$ | $+$ | $\rho_tA_t$ | no | $+$ | ✓ |
| 2 | $[1-\epsilon, 1+\epsilon]$ | $-$ | $\rho_tA_t$ | no | $-$ | ✓ |
| 3 | $< 1-\epsilon$ | $+$ | $\rho_tA_t$ | no | $+$ | ✓ |
| 4 | $< 1-\epsilon$ | $-$ | $(1-\epsilon)A_t$ | **yes** | $-$ | **0** |
| 5 | $> 1+\epsilon$ | $+$ | $(1+\epsilon)A_t$ | **yes** | $+$ | **0** |
| 6 | $> 1+\epsilon$ | $-$ | $\rho_tA_t$ | no | $-$ | ✓ |

*(The handwritten table's right-hand columns drift down by about half a row. Read row by row, they give exactly this table, which is also the standard result.)*

How to read it:
- The gradient is zero only in regions 4 and 5, where the token has *already* moved far enough in the direction its advantage wants.
- In regions 3 and 6 the policy has moved the *wrong* way. The min keeps the unclipped line, so the full gradient pulls it back, and a move that makes things worse is never hidden.
- Flat parts have zero gradient, so a token whose ratio has already moved past the bound stops pushing.

![[03 - Attachments/ATML_L8_ppo_clip_objective.png|650]]

*The same clipped objective (blue) drawn against the ratio. For $A > 0$, the gain stops once $\rho > 1 + \epsilon$: no extra reward for making a good token even more likely in one step. For $A < 0$, it stops once $\rho < 1 - \epsilon$: no extra reward for suppressing a bad token further.*

> [!warning] Possible source inconsistency
> 1. The Lecture 12 notes write $\min\big(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\big)$, without the $A$ multiplying the clip term. The professor's notes and the DVLM notes multiply both terms by $A$.
> 2. The hand-drawn loss sketches in Lecture 12 don't match the standard shape. As a loss to minimize ($-\mathcal L^{\text{clip}}$):
>    - for $A > 0$ it decreases up to $1 + \epsilon$ and is flat after;
>    - for $A < 0$ it is flat below $1 - \epsilon$ and increases after.
>
>    The plots above show the objective (maximized), which is the mirror image.
> 3. The red note in Lecture 12 marks $\rho = 1$ as "$\pi_\theta^{\text{old}} = \pi_\theta^{\text{ref}}$". $\rho = 1$ means $\pi_\theta = \pi_\theta^{\text{old}}$. The two coincide with $\pi_{\text{ref}}$ only at the very first step, when all three equal $\theta_{\text{SFT}}$.

### 5.7 The complete PPO loss

The full loss trains the actor and the critic together:

$$\boxed{\mathcal L_{PPO}(\theta, \phi) = -\mathcal L_{\text{clip}}(\theta) + \mathcal L_V(\phi) + \mathcal L_{KL}(\theta) - \mathcal L_{ENT}(\theta)}$$

| Term | Formula | Role |
|---|---|---|
| $\mathcal L_V(\phi)$ | $\frac1U\sum_{i=1}^N\sum_{t=1}^{T_i}\big(V_\phi(s_{i,t}) - V^{GAE}_\phi(s_{i,t})\big)^2$ | trains the critic toward the $\lambda$-returns |
| $\mathcal L_{KL}(\theta)$ *(optional)* | $\frac1U\sum_{i,t}\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ | monitors (and penalizes) the per-token KL to the SFT reference |
| $\mathcal L_{ENT}(\theta)$ *(optional)* | $\frac1U\sum_{i,t}H\big(\pi_\theta(\cdot\mid s_{i,t})\big)$ | rewards entropy; discourages premature collapse of each token's distribution |

*(Expanding on this:)* implementations weight the value and entropy terms with coefficients, e.g. $c_1\mathcal L_V - c_2\mathcal L_{ENT}$. The KL to $\pi_{\text{ref}}$ is put **either** into the per-token reward (§5.3) **or** into the loss, usually not both.

This is **RLHF with PPO**, a leading policy-gradient method: an **actor–critic policy gradient with a clipped trust-region surrogate**.

**Full-batch vs sequence-minibatch.** The average $\frac1U\sum_{i=1}^N\sum_{t=1}^{T_i}$ runs over every token of every trajectory: **full-batch PPO**. In practice each inner step uses a mini-batch over a subset of the trajectories: **sequence-minibatch PPO**.

![[03 - Attachments/ATML_L8_ppo_pipeline.png|640]]

*The PPO data flow:*
- *$x$ → $\pi_\theta$ → $(x, y)$.*
- *$\pi_{\text{ref}}$ and $r_\psi$ score $(x, y)$ (the KL-shaped reward); $V_\phi$ gives the values.*
- *GAE combines them into the advantage $A$, and also returns targets to $V_\phi$.*
- *$A$ back-propagates into $\pi_\theta$.*

### 5.8 Problems with PPO, and a one-token example

**Problems with PPO:**
- $V_\phi$ can be inaccurate, which destabilizes the actor's updates.
- The value head, GAE and $V_\phi$ cost more compute and memory: the critic is a second LLM-sized network.

**Worked example (the professor's).** Prompt $x$: "What is 2+2?" The LLM so far has produced $y$ = "The answer is". Get the logits for the last token produced by the policy, apply the softmax, and look up the probability of the token "4" (the correct answer):

| Probability of "4" | $\pi_\theta^{\text{old}}$ | $\pi_\theta$ |
|---|---|---|
| | 0.1 | 0.15 |

So $\rho_t = 0.15 / 0.1 = 1.5$. The raw reward and advantage are:

| $r_\psi$ reward | $V_\phi$ prediction (expected reward) | Advantage |
|---|---|---|
| $+1$ (correct) | 0.4 | $1 - 0.4 = 0.6$ |
| $-1$ (wrong) | 0.2 | $-1 - 0.2 = -1.2$ |

For the correct answer:
- **raw score:** $\rho_tA_t = 1.5 \times 0.6 = 0.9$;
- **clipped ratio:** $\text{clip}(\rho_t, 0.8, 1.2) = 1.2$ (with $\epsilon = 0.2$);
- **clipped score:** $1.2 \times 0.6 = 0.72$;
- **final score:** since $A_t \ge 0$, $\min(\text{raw}, \text{clipped}) = 0.72$.

$$\text{loss} = -\text{final score} + \beta\,\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$$

> [!warning] Possible source inconsistency
> 1. The notes write the clip as $\text{clip}(\rho_tA_t, 0.8, 1.2)$. The clip applies to the ratio $\rho_t$ alone, which is what the computed value 1.2 uses.
> 2. The notes then say "else (if $A_t < 0$), final score $= \max(\text{raw}, \text{clipped})$". In terms of *scores* PPO always takes the min. For the wrong answer, $\min(1.5 \times (-1.2), 1.2 \times (-1.2)) = \min(-1.8, -1.44) = -1.8$: the unclipped value, so the full penalty applies (region 6). The max is correct only in terms of *ratios*: for $A < 0$, $\min(\rho A, \text{clip}(\rho)A) = A\cdot\max(\rho, \text{clip}(\rho))$.

---

## 6. Group Relative Policy Optimization (GRPO)

PPO's advantage needs a value model as large as the LLM itself, which is costly, unstable and must look ahead (§5.4, §5.8). **GRPO** (Group Relative Policy Optimization) replaces the $V_\phi(\cdot)$ model with a **group (per-prompt) baseline**, computed from multiple roll-outs of the same prompt. This changes what the advantage means.

### 6.1 What changes: the advantage

| | PPO | GRPO |
|---|---|---|
| Reference point | absolute | **relative** to sibling responses |
| Granularity | per token | **per completion**: every token of a response gets the same advantage |
| Source | learned (value model) | **not learned**: computed from rewards |
| State dependence | depends on $s_{i,t}$ | **state-independent** |

The question changes from "how good is this token compared with what the critic expected?" to "among sibling completions sampled from the same prompt, was this one above or below the group average?" The group baseline **replaces the token-level critic advantage** $A_{i,t} = V^{GAE}_\phi(s_{i,t}) - V^{\text{old}}_\phi(s_{i,t})$ of §5.4.

### 6.2 The GRPO procedure and objective

**Set-up (one training step):**
1. Take a batch of $B$ prompts $\{x_b\}_{b=1}^B$.
2. For each prompt, **sample $K$ roll-outs** from the old policy: $\{y_{b,k}\}_{k=1}^K \sim \pi_\theta^{\text{old}}(\cdot\mid x_b)$, completion $k$ having length $T_k$.
3. **Score each completion** with $r_\psi(x, y)$: $r_{b,k} = r_\psi(x_b, y_{b,k})$.
4. **Compute the group-relative advantage.** For prompt $x_b$, build the group baseline (the group mean) and subtract it:
$$\boxed{\mu_b = \frac1K\sum_{k=1}^K r_{b,k}, \qquad A_{b,k} = r_{b,k} - \mu_b}$$

$A_{b,k}$ is the advantage of the $k$-th roll-out in group $b$, called the **per-prompt relative signal / advantage**.

**Intuition:**
- The **winners** in a group, all roll-outs with reward $\ge \mu_b$, get a positive advantage.
- The **losers** get a negative one.

![[03 - Attachments/ATML_L8_grpo_illustration.png|660]]

*One prompt $x_b$ gives $K = 4$ completions of different lengths. $r_\psi$ scores each, the group computation turns the scores into advantages $A_1, \dots, A_4$, and each $A_k$ is **broadcast to all tokens** of completion $k$: $A_{k,t} = A_k$.*

> [!note] Mean-only vs normalized advantage
> The professor's GRPO subtracts the group mean only. Your Lecture 12 notes (and the original GRPO paper) also divide by the group's standard deviation:
> $$A_{k,t} = \tilde r_k = \frac{r_k - \bar r}{\sigma(r)}$$
> Dividing by $\sigma$ keeps the advantages on a comparable scale across prompts of different difficulty. But it also up-weights prompts whose rewards barely differ, which is one reason Dr. GRPO (§6.4) drops it. Lecture 12 writes the index as $i$ and, as in §5.6, leaves $A$ off the clip term. The index is renamed $k$ here so it doesn't clash with the trajectory index $i$ of PPO, and $A$ is restored on the clip term.

**The GRPO loss for one prompt** is PPO's clipped surrogate with the sequence-level advantage, plus a per-token KL penalty to the SFT reference:

$$\boxed{\mathcal L_{GRPO}(\theta) = -\frac1K\sum_{k=1}^{K}\frac{1}{T_k}\sum_{t=1}^{T_k}\underbrace{\Big[\min\Big(\rho_{k,t}A_{k,t},\;\text{clip}(\rho_{k,t}, 1-\epsilon, 1+\epsilon)A_{k,t}\Big) - \beta\,\text{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})\Big]}_{\ell_{k,t}}}, \qquad \rho_{k,t} = \frac{\pi_\theta(\cdot\mid s_{k,t})}{\pi_\theta^{\text{old}}(\cdot\mid s_{k,t})}$$

Here $k$ is the completion index (PPO used $i$, the trajectory index) and $t$ the token index. The sequence-level advantage $A_{k,t} = A_k$ is broadcast to all tokens of the completion. **Each sequence contributes its mean token loss** (the $1/T_k$ factor), and all sequences are then averaged (the $1/K$). Putting the KL in the loss, rather than in the reward as PPO does, is the usual choice in GRPO. The DVLM version writes the same thing as an objective to maximize: $\mathbb E_x\big[\frac1K\sum_k\frac1{T_k}\sum_t\min(\cdot)\big] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$.

![[03 - Attachments/ATML_L8_grpo_pipeline.png|660]]

*GRPO's data flow:*
- *$x$ → $\pi_\theta$ → $K$ completions $y_1, \dots, y_K$.*
- *$r_\psi$ (with $\pi_{\text{ref}}$ for the KL) scores them to $r_1, \dots, r_K$.*
- *The group computation turns the scores into $A_1, \dots, A_K$, which update $\pi_\theta$.*
- *The KL to $\pi_{\text{ref}}$ feeds back directly.*

**Models needed:**
- **PPO** needs $\pi^{\text{SFT}}_\theta$ (the reference), $\pi^{\text{new}}_\theta$, $V_\phi(s)$ and $r_\psi$.
- **GRPO** crosses out $V_\phi(s)$.

**Why it works:** the group mean is a free baseline. Subtracting it does what the critic was for, telling good samples from typical ones, but it costs $K$ samples instead of a second network.

### 6.3 The length bias of the $1/T_k$ factor

The $1/T_k$ factor causes the **length bias** (a lack of length invariance) of GRPO. Even if two completions have the same reward (the same semantic quality), their contributions to the loss are not the same, because each token's weight is proportional to $A_k/T_k$:
- **short sequence:** $T_k$ is small, so the per-token weight $A_k/T_k$ is large in magnitude;
- **long sequence:** $T_k$ is large, so the per-token weight is small.

**Impact:**
- If $A_k > 0$ (a good completion), short correct answers get a stronger per-token push than long ones.
- If $A_k < 0$ (a bad completion), short wrong answers get a stronger per-token penalty than long ones. **A long incorrect response is under-penalized**, and the model starts rambling.

So GRPO tends to produce increasingly long sequences, especially for incorrect outputs: an **optimization bias in GRPO**.

### 6.4 Fixes: DAPO and Dr. GRPO

**DAPO** switches to **token-level normalization**. All tokens in the group are averaged together, with no per-sequence $1/T_k$ factor:

$$\mathcal L_{DAPO}(\theta) \propto -\frac{1}{\sum_{k=1}^K T_k}\sum_{k}\sum_{t}\ell_{k,t}$$

Every token now has the same weight $A_k / \sum_j T_j$, whatever the length of its own sequence. A long wrong answer is penalized in proportion to its length.

**Dr. GRPO** divides by a **constant** $T$ instead, usually $T = T_{\max}$ over all sequences:

$$\mathcal L_{\text{Dr.GRPO}}(\theta) \propto -\frac{1}{T_{\max}}\frac1K\sum_{k}\sum_{t}\ell_{k,t}$$

*(The professor writes the constant as $L$; it is renamed $T_{\max}$ here, since $L$ already means the number of layers and the loss.)* *(Expanding on this:)* Dr. GRPO ("GRPO Done Right") also removes the division by the group's standard deviation. That is the mean-only advantage the professor uses in §6.2.

---

## 7. Direct Preference Optimization (DPO)

### 7.1 Motivation: a simple objective behind a complicated pipeline

RLHF solves the simple core objective of §4.3 with a very complicated pipeline. It has two main bottlenecks:
1. **The reward model $r_\psi(x, y)$:** a separate network to train, with the limitations of §3.6.
2. **Getting the advantage:** a value model (PPO) or many rollouts (GRPO), plus sampling during training.

DPO asks: **can the policy be optimized directly on the preference data, with no reward model and no RL?** The pipeline becomes

$$\theta_{\text{PLLM}} \to \text{SFT} \to \theta_{\text{SFT}} = \pi_{\text{ref}} \to \underbrace{\text{directly optimize } \pi_\theta}_{\text{on } \mathcal D_{Pr}:\,(x, y^+, y^-)} \to \theta_{\text{aligned}}$$

Here $y^+$ is preferred over $y^-$ by a human annotator. **The idea:**
- skip the reward model $r_\psi(\cdot)$ and the RL loop;
- use $\mathcal D_{Pr}$ directly as a supervised signal for training;
- **treat $\theta_{\text{SFT}}$ as a hidden reward model**.

The DVLM notes put it as widening the probability gap between the winning and the rejected response directly, measured relative to $\pi_{\text{ref}}$. Your Lecture 13 notes call it **offline RL**: there is **no sampling from the LM policy** during training.

### 7.2 The DPO pipeline and loss

![[03 - Attachments/ATML_L8_dpo_pipeline.png|620]]

*Each triple $(x, y^+, y^-)$ goes through both the frozen reference model and the policy model. The four log-probabilities form two log-ratios, $a$ for the winner and $b$ for the loser. The loss is $-\log\sigma(a - b)$, and its gradient updates only $\pi_\theta$.*

Start from the reward-model loss of §3.3, $\mathcal L_{RM}(\psi) = -\mathbb E_{(x, y^+, y^-)}\big[\log\sigma(r_\psi(x, y^+) - r_\psi(x, y^-))\big]$. DPO directly integrates $y^+ \succ y^-$ into its loss by putting a **KL-style log-ratio** in place of each reward:

$$a = z_1 = \beta\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)}, \qquad b = z_2 = \beta\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}$$

$$\boxed{\mathcal L_{\text{DPO}}(\theta) = -\,\mathbb E_{(x, y^+, y^-)\sim\mathcal D_{\text{pref}}}\Big[\log\sigma(a - b)\Big]}$$

$\sigma(z_1 - z_2)$ has the **same structure as the reward-model sigmoid**: $a$ and $b$ play the role of $r_\psi(x, y^+)$ and $r_\psi(x, y^-)$, and §7.3 shows this is no coincidence. The loss **increases the log-ratio in favour of $y^+$, relative to what $\pi_{\text{ref}}$ does, but not infinitely**: once $a - b$ is large the sigmoid saturates and the push fades.

A response's log-probability is the sum of its tokens' log-probabilities, $\log\pi(y\mid x) = \sum_t \log\pi(y_t\mid x, y_{<t})$. So each term costs one forward pass and no sampling.

> [!warning] Possible source inconsistency
> The professor's first line writes $\mathcal L_{DPO}(\theta) = \mathbb E\big[\sigma(z_1 - z_2)\big]$, without the $-\log$. The very next line, the per-pair loss, has it: $\ell(\theta) = -\log\sigma(z_1 - z_2)$. The loss is the negative log-likelihood, as above.

**Rewriting with policy and reference log-ratios.** For any single pair, regroup the four log-probabilities:

$$\ell(\theta) = -\log\sigma(z_1 - z_2) = -\log\sigma\Big(\beta\big[\underbrace{\log\pi_\theta(y^+\mid x) - \log\pi_\theta(y^-\mid x)}_{\Delta_\theta\text{: policy log-ratio}}\big] - \beta\big[\underbrace{\log\pi_{\text{ref}}(y^+\mid x) - \log\pi_{\text{ref}}(y^-\mid x)}_{\Delta_{\text{ref}}\text{: reference correction}}\big]\Big)$$

$$\boxed{\ell(\theta) = -\log\sigma(z), \qquad z = \beta\,(\Delta_\theta - \Delta_{\text{ref}})}$$

$\Delta_{\text{ref}}$ is constant with respect to $\theta$. To decrease $\ell$, increase $z$, so that $\sigma(z) \to 1$, $\log(1) \to 0$ and $\mathcal L_{DPO} \to \min$.

### 7.3 Derivation: from the RLHF objective to the DPO loss

*(Neither set of notes derives the loss; this derivation fills in why it is right.)*

**(i) The RLHF objective has a closed-form optimum.** For a fixed prompt $x$, write the §4.3 objective over distributions $\pi(\cdot\mid x)$:

$$\max_\pi\; \sum_y \pi(y\mid x)\,r(x, y) - \beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

Dividing by $-\beta$ and folding $r$ into the log turns it into a minimization of $\sum_y\pi\log\frac{\pi}{\pi_{\text{ref}}\exp(r/\beta)}$. That is a KL divergence (up to a constant) to the normalized distribution

$$\boxed{\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x, y)}{\beta}\Big), \qquad Z(x) = \sum_y\pi_{\text{ref}}(y\mid x)\exp\!\Big(\frac{r(x, y)}{\beta}\Big)}$$

A KL divergence is minimized (at 0) exactly when the two distributions match, so the optimal policy is the reference reweighted by $e^{r/\beta}$. $Z(x)$ sums over every possible response, so it can't be computed. That is why RLHF can't just use this formula and needs RL.

**(ii) Invert it: the reward is a log-ratio.** Take logs and solve for $r$:

$$r(x, y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

So every reward function corresponds to an optimal policy, and the reward can be written in terms of that policy.

**(iii) Substitute into Bradley–Terry; $Z$ cancels.** The BT probability (§3.3) depends only on a difference of rewards for the same prompt $x$, so the uncomputable $\beta\log Z(x)$ cancels:

$$P(y^+\succ y^-\mid x) = \sigma\Big(\beta\log\frac{\pi^*(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)} - \beta\log\frac{\pi^*(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}\Big)$$

**(iv) Fit the policy by maximum likelihood.** Parameterize $\pi^*$ by the policy $\pi_\theta$ and maximize the likelihood of the human preferences. The negative log-likelihood is exactly $\mathcal L_{\text{DPO}}$. The policy *is* the reward model: $\hat r_\theta(x, y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ is called the **implicit reward**, and the KL constraint to $\pi_{\text{ref}}$ is built in through $\beta$. This is the precise sense in which DPO "treats $\theta_{\text{SFT}}$ as a hidden reward model" (§7.1).

**What the gradient does:**

$$\nabla_\theta\mathcal L_{\text{DPO}} = -\beta\,\mathbb E\Big[\sigma(b - a)\big(\nabla_\theta\log\pi_\theta(y^+\mid x) - \nabla_\theta\log\pi_\theta(y^-\mid x)\big)\Big]$$

It raises the winner's likelihood and lowers the loser's. Each pair is weighted by $\sigma(b - a)$, which is large when the model currently ranks the pair the *wrong* way round. Pairs it already gets right contribute little.

### 7.4 Gradient analysis and the role of the reference

The professor works the same gradient out per log-probability. By the chain rule through $z = \beta(\Delta_\theta - \Delta_{\text{ref}})$, with $\frac{d}{dz}[-\log\sigma(z)] = \sigma(z) - 1$:

$$\frac{\partial\ell}{\partial\log\pi_\theta(y^+\mid x)} = \frac{\partial\ell}{\partial z}\cdot\frac{\partial z}{\partial\Delta_\theta}\cdot\frac{\partial\Delta_\theta}{\partial\log\pi_\theta(y^+\mid x)} = (\sigma(z) - 1)\times\beta\times1 = \boxed{\beta\,(\sigma(z) - 1)}$$

$$\frac{\partial\ell}{\partial\log\pi_\theta(y^-\mid x)} = \boxed{-\beta\,(\sigma(z) - 1)}$$

The second has the opposite sign because $\Delta_\theta$ contains $-\log\pi_\theta(y^-\mid x)$. Since $\sigma(b - a) = 1 - \sigma(z)$, these are the weights in the gradient of §7.3.

**During gradient descent:**

$$\log\pi_\theta \leftarrow \log\pi_\theta - \eta\,\frac{\partial\ell}{\partial\log\pi_\theta}$$

**When the model does not prefer $y^+$ over $y^-$**, $z$ is small or negative ($\sigma(z) \lesssim 0.5$), so $\sigma(z) - 1 \le -0.5 < 0$.
- **Good completion $y^+$:** $\partial\ell/\partial\log\pi_\theta(y^+\mid x) = \beta(\sigma(z) - 1) < 0$, so $\log\pi_\theta(y^+\mid x)$ **increases**.
- **Bad completion $y^-$:** the derivative is $> 0$, so $\log\pi_\theta(y^-\mid x)$ **decreases**.

**The SFT reference's role.** Look at $z = \beta(\Delta_\theta - \Delta_{\text{ref}})$. Suppose $\pi_\theta$ prefers $y^+$ over $y^-$ *at least as strongly* as $\pi_{\text{ref}}$ does, with room to spare. Then $\Delta_\theta - \Delta_{\text{ref}}$ is large and positive, so:
- $\sigma(z) \approx 1$;
- $\sigma(z) - 1 \approx 0$;
- the gradients are about 0.

That is **regularization**: the policy is not pushed further once it beats the reference by a margin. **So $\pi_\theta$ has to prefer $y^+$ more than $\pi_{\text{ref}}$ does to lower the loss.** The reference itself never receives or creates a gradient; it only sets the bar the policy's preference is measured against. *(The notes' phrase "when it starts doing PO, no push from $\pi_{\text{ref}}$" is read here this way.)*

**Your Lecture 13 numbers** ($\beta = 1$):
1. $\Delta_\theta = \Delta_{\text{ref}} = 0.4$, so $z = 0$, $\sigma(z) = 0.5$ and the loss is $\ln 2 \approx 0.7$. This is the value at initialization.
2. $\Delta_\theta = 1$ and $\Delta_{\text{ref}} = 0.4$, so $z = 0.6$, $\sigma(z) = 0.65$ and the loss is $0.44$ (your notes round it to 0.4). Increasing $z$ lowers the loss.

### 7.5 DPO's issues: verbosity bias and a bad reference

**(1) Verbosity bias: DPO produces overly long answers.** A sequence's log-probability is a sum over its tokens,

$$\log\pi_\theta(y\mid x) = \sum_{t=1}^{T_y}\log\pi_\theta(y_t\mid x, y_{<t})$$

so it scales roughly with the length $T_y$. If $T_{y^+} > T_{y^-}$, the two KL terms in DPO sum over different numbers of tokens. That difference can **over- or under-estimate** the preference signal $y^+ \succ y^-$. *(Expanding on this:)* the same small per-token shift in the log-ratio adds up to a bigger $z$ for a longer winner (Problem 14). Preference data also often favour longer answers. So DPO learns that length itself pays.

Fixes:
- **SamPO:** down-sample tokens so that both responses contribute the same number of per-token terms, making the contribution **length-neutral**. *(The down-sampled KL is computed over an equal number of randomly chosen tokens from each response.)*
- **SimPO:** use the **averaged** log-probability, and go **reference-free**. *(Expanding on this:)* its implicit reward is $\frac{\beta}{|y|}\log\pi_\theta(y\mid x)$, with a target margin $\gamma$:
$$\mathcal L_{\text{SimPO}} = -\log\sigma\Big(\frac{\beta}{|y^+|}\log\pi_\theta(y^+\mid x) - \frac{\beta}{|y^-|}\log\pi_\theta(y^-\mid x) - \gamma\Big)$$
  Averaging removes the length effect. Dropping $\pi_{\text{ref}}$ also halves the memory.

**(2) A bad $\pi_{\text{ref}}$.** The KL regularization pulls everything toward a **bad anchor**: if the SFT model is poor, DPO keeps the policy near it. Fixes *(one-line glosses added)*:
- **IPO** (Identity Preference Optimization): replaces $-\log\sigma$ with a squared loss toward a fixed target margin. DPO can push the log-ratio gap without limit on near-deterministic preferences and overfit; IPO's squared loss caps it.
- **ORPO** (Odds-Ratio Preference Optimization): reference-free. It adds an odds-ratio penalty to the plain SFT loss, merging SFT and alignment into one stage, so there is no anchor to inherit.
- **α-DPO:** makes the reference adaptive (an $\alpha$-weighted mix involving the policy itself) to soften dependence on a fixed, possibly poor, $\pi_{\text{ref}}$.

**Models needed:** DPO keeps $\pi^{\text{SFT}}_\theta$ (as $\pi_{\text{ref}}$) and $\pi^{\text{new}}_\theta$, and crosses out both $V_\phi(s)$ and $r_\psi$. Its loop: $x \to \pi_\theta \to \Delta_\theta$; together with $\pi_{\text{ref}}$ and $\mathcal D_{Pr}$ that gives $\mathcal L_{DPO}(\theta)$, which updates $\pi_\theta$.

### 7.6 PPO vs GRPO vs DPO

| | PPO | GRPO | DPO |
|---|---|---|---|
| Models (Lecture 13) | $\pi_\theta, \pi^{\text{old}}_\theta, \pi_{\text{ref}}, r_\psi, V_\phi$ | $\pi_\theta, \pi^{\text{old}}_\theta, r_\psi, \pi_{\text{ref}}$ | $\pi_\theta, \pi_{\text{ref}}$ |
| Reward model | yes, frozen | yes (or a rule-based verifier) | **no**: implicit in $\pi_\theta$ |
| Value model | yes | **no** | **no** |
| Sampling during training | yes | yes, $K$ per prompt | **no**: offline on fixed $\mathcal D_{\text{pref}}$ |
| Advantage | learned, per token | group-relative, per completion | none: pairwise log-ratio gap |
| KL to $\pi_{\text{ref}}$ | shaped into the per-token reward | added to the loss | built into the loss through $\beta$ |
| Models in memory | 4 ($\pi_\theta, \pi_{\text{ref}}, r_\psi, V_\phi$) | 3 | 2 ($\pi_\theta, \pi_{\text{ref}}$) |
| Length issue | — | $1/T_k$ length bias (§6.3) | verbosity bias (§7.5) |
| Main weakness | complex, unstable critic | needs many rollouts per prompt | can't explore beyond the dataset's responses; bad-anchor risk |

*(The last row is added beyond the notes.)* ($\pi^{\text{old}}_\theta$ is a snapshot of $\pi_\theta$, so it needs no extra parameters at training time, only the cached log-probabilities.)

---

## Numerical Problems

**Problem 1: reward-model loss and shift invariance.** $r_\psi(x, y^+) = 2.0$ and $r_\psi(x, y^-) = 0.5$. Find $P(y^+\succ y^-)$ and the loss, then repeat with both rewards shifted by $+10$.

The difference is $1.5$, so $P = \sigma(1.5) = 1/(1 + e^{-1.5}) = 1/1.2231$ and the loss is $-\ln 0.8176$. Shifting both rewards leaves the difference at $1.5$.

$$\boxed{P = 0.818,\ \mathcal L = 0.201\ \text{(unchanged after the shift)}}$$

This is limitation 1 of §3.6: only the difference is learned.

**Problem 2: PPO clipping.** With $\epsilon = 0.2$, find $\mathcal L^{\text{clip}}$ and say whether the token still pushes.

| Case | Region | $\rho A$ | $\text{clip}(\rho)\,A$ | min | Pushes? |
|---|---|---|---|---|---|
| $A = +2$, $\rho = 1.5$ | 5 | $3.0$ | $1.2 \times 2 = 2.4$ | $2.4$ | no: clipped, flat |
| $A = +2$, $\rho = 0.7$ | 3 | $1.4$ | $0.8 \times 2 = 1.6$ | $1.4$ | yes: still raises $\rho$ |
| $A = -1$, $\rho = 0.6$ | 4 | $-0.6$ | $0.8 \times (-1) = -0.8$ | $-0.8$ | no: clipped, flat |
| $A = -1$, $\rho = 1.4$ | 6 | $-1.4$ | $1.2 \times (-1) = -1.2$ | $-1.4$ | yes: still lowers $\rho$ |

$$\boxed{2.4,\ 1.4,\ -0.8,\ -1.4}$$

Clipping only stops the "already improved enough" direction.

**Problem 3: value targets and GAE.** A 3-token response has a sparse reward $r = [0, 0, 1]$, with $\gamma = 1$ and $\lambda = 0.5$. The old critic predicts $V(s_1) = 0.5$, $V(s_2) = 0.6$, $V(s_3) = 0.8$, and $V(s_4) = 0$ (terminal). Find the MC and TD(0) targets at $s_1$ and the GAE advantages.

- **Targets at $s_1$:** MC $= 0 + 0 + 1 = 1$; TD(0) $= 0 + 0.6 = 0.6$.
- **TD errors:**
  - $\delta_1 = 0 + 0.6 - 0.5 = 0.1$
  - $\delta_2 = 0 + 0.8 - 0.6 = 0.2$
  - $\delta_3 = 1 + 0 - 0.8 = 0.2$
- **GAE** ($\gamma\lambda = 0.5$):
  - $A_1 = 0.1 + 0.5(0.2) + 0.25(0.2) = 0.25$
  - $A_2 = 0.2 + 0.5(0.2) = 0.3$
  - $A_3 = 0.2$

$$\boxed{V^*_{\text{MC}} = 1,\ V^*_{\text{TD(0)}} = 0.6;\quad A = [0.25, 0.30, 0.20]}$$

Even though only the last token was rewarded, every token gets credit through the critic.

**Problem 4: GRPO advantages.** $K = 4$ completions score $r = [0.9, 0.4, 0.6, 0.1]$. Find the advantages.

$\bar r = 0.5$, and the deviations are $[0.4, -0.1, 0.1, -0.4]$. Then $\sigma = \sqrt{(0.16 + 0.01 + 0.01 + 0.16)/4} = \sqrt{0.085} = 0.2915$.

$$\boxed{A = [1.37,\ -0.34,\ 0.34,\ -1.37]\ \text{(normalized)};\quad [0.4, -0.1, 0.1, -0.4]\ \text{(the professor's mean-only form)}}$$

The advantages sum to zero: half the group is pushed up and half down, whatever the absolute reward level.

**Problem 5: DPO loss.** $\beta = 0.1$. $\log\pi_\theta(y^+) = -12$ and $\log\pi_{\text{ref}}(y^+) = -14$; $\log\pi_\theta(y^-) = -15$ and $\log\pi_{\text{ref}}(y^-) = -13$. Find $\mathcal L_{\text{DPO}}$, and the loss at initialization ($\pi_\theta = \pi_{\text{ref}}$).

$a = 0.1(-12 + 14) = 0.2$ and $b = 0.1(-15 + 13) = -0.2$, so $a - b = 0.4$. Then $\sigma(0.4) = 1/(1 + e^{-0.4}) = 0.5987$, and the loss is $-\ln 0.5987$. At initialization $a = b = 0$, giving $-\ln\sigma(0) = \ln 2$.

$$\boxed{\mathcal L = 0.513;\quad \mathcal L_{\text{init}} = \ln 2 = 0.693}$$

**Problem 6: KL-shaped token reward.** The last token has $\pi_\theta(y_T) = 0.5$ and $\pi_{\text{ref}}(y_T) = 0.25$, with $\beta = 0.1$ and $r_\psi = 0.8$. An earlier token has $\pi_\theta = \pi_{\text{ref}}$. Find both shaped rewards.

Last token: $0.8 - 0.1\ln 2 = 0.8 - 0.0693$. Earlier token: $0 - 0.1\ln 1 = 0$.

$$\boxed{r_T = 0.731,\quad r_{t<T} = 0}$$

A token is taxed only to the extent the policy has moved away from the reference on it.

**Problem 7: Bradley–Terry from strengths.** Two responses have rewards $r_w = 1.1$ and $r_l = 0$. Find their BT strengths, $P(y_w \succ y_l)$ and the reward loss.

The strengths are $p_w = e^{1.1} = 3.004$ and $p_l = e^0 = 1$, so $P = 3.004/4.004$. Equivalently, $\sigma(1.1) = 0.750$. The loss is $-\ln 0.750$.

$$\boxed{P = 0.750,\quad \mathcal L = 0.288}$$

A reward gap of $\ln 3$ means "three times as strong", i.e. preferred 3 times out of 4.

**Problem 8: pairs and margins.** (a) $K = 4$ responses are ranked $y_2 \succ y_4 \succ y_1 \succ y_3$. List the comparison pairs. (b) A pair has $r_w - r_l = 1.5$ and a Llama-2 margin $m = 1.0$. Compare the loss with and without the margin.

(a) $\binom42 = 6$ pairs: $(y_2, y_4), (y_2, y_1), (y_2, y_3), (y_4, y_1), (y_4, y_3), (y_1, y_3)$, winner first. (b) Without the margin, $-\ln\sigma(1.5) = 0.201$. With it, $-\ln\sigma(1.5 - 1.0) = -\ln 0.6225$.

$$\boxed{6\text{ pairs};\quad \mathcal L = 0.201 \to 0.474 \text{ with the margin}}$$

The margin keeps the loss, and so the gradient, high until the gap exceeds $m$.

**Problem 9: Plackett–Luce.** Three responses with rewards $[2, 1, 0]$ are ranked $y_1 \succ y_2 \succ y_3$. Find the ranking probability and the $K$-wise loss.

$P = \frac{e^2}{e^2 + e + 1}\cdot\frac{e}{e + 1} = \frac{7.389}{11.107}\cdot\frac{2.718}{3.718} = 0.665 \times 0.731$, and the loss is $-\ln 0.486$.

$$\boxed{P = 0.486,\quad \mathcal L_{PL} = 0.721}$$

The second factor is just the BT probability of $y_2$ over $y_3$: PL chains BT-like choices.

**Problem 10: the professor's 2+2 example, wrong answer.** With $\pi_\theta^{\text{old}}(\text{"4"}) = 0.1$ and $\pi_\theta(\text{"4"}) = 0.15$ as before, suppose the reward is $-1$ and $V_\phi = 0.2$. Find the final score for $\rho = 1.5$, and again for $\rho = 0.5$.

$A = -1 - 0.2 = -1.2$.
- **$\rho = 1.5$:** raw $= -1.8$; clipped $= 1.2 \times (-1.2) = -1.44$; $\min = -1.8$. Not clipped (region 6), so the full gradient applies.
- **$\rho = 0.5$:** raw $= -0.6$; clipped $= 0.8 \times (-1.2) = -0.96$; $\min = -0.96$. Clipped (region 4), so the gradient is 0.

$$\boxed{-1.8\ (\rho = 1.5),\qquad -0.96\ (\rho = 0.5)}$$

Lowering a bad token's probability is rewarded only down to $1 - \epsilon$.

**Problem 11: GRPO's length bias, and the fixes.** One prompt has $K = 2$ correct completions, both with $A = +1$, of lengths $T_1 = 10$ and $T_2 = 100$. Find the weight each *token* receives under GRPO, DAPO and Dr. GRPO ($T_{\max} = 100$).

- **GRPO** $\frac1K\cdot\frac{A}{T_k}$: $\frac12\cdot\frac1{10} = 0.05$ and $\frac12\cdot\frac1{100} = 0.005$.
- **DAPO** $\frac{A}{\sum T_k}$: $\frac1{110} = 0.0091$ for both.
- **Dr. GRPO** $\frac1K\cdot\frac{A}{T_{\max}}$: $\frac12\cdot\frac1{100} = 0.005$ for both.

$$\boxed{\text{GRPO: } 0.05 \text{ vs } 0.005\ (10\times\text{ gap});\quad \text{DAPO: } 0.0091;\quad \text{Dr. GRPO: } 0.005}$$

With $A = -1$, the same 10× gap means the long wrong answer is penalized 10× less per token under GRPO.

**Problem 12: mean-only vs normalized advantage, and a dead group.** (a) Rewards $[1, 0, 1, 0]$. (b) Rewards $[0, 0, 0, 0]$. Find the advantages.

(a) $\mu = 0.5$, so the mean-only advantages are $[0.5, -0.5, 0.5, -0.5]$. With $\sigma = 0.5$ the normalized ones are $[1, -1, 1, -1]$. (b) Every reward equals the mean, so every advantage is 0.

$$\boxed{(a)\ [\pm0.5] \text{ or } [\pm1];\quad (b)\ A = 0 \Rightarrow \text{no gradient}}$$

A group where every roll-out fails (or every one succeeds) teaches nothing. This matters for verifiable rewards (Lecture 9).

**Problem 13: DPO gradient.** For $\beta = 1$ and the two cases of your Lecture 13 notes, find $\partial\ell/\partial\log\pi_\theta(y^+)$.

$\partial\ell/\partial\log\pi_\theta(y^+) = \beta(\sigma(z) - 1)$:
- case 1, $z = 0$: $0.5 - 1 = -0.5$;
- case 2, $z = 0.6$: $0.646 - 1 = -0.354$.

$$\boxed{-0.5\ (\mathcal L = 0.693),\qquad -0.354\ (\mathcal L = 0.437)}$$

As the policy's preference grows beyond the reference's, the push weakens: the sigmoid saturates.

**Problem 14: verbosity bias.** With $\beta = 0.1$, the policy raises every token of $y^+$ by $0.02$ nats relative to $\pi_{\text{ref}}$ and lowers every token of $y^-$ by $0.02$. Here $T_{y^-} = 10$. Find $z$ when $T_{y^+} = 100$, and when $T_{y^+} = 10$.

$\Delta_\theta - \Delta_{\text{ref}} = 0.02\,T_{y^+} + 0.02\,T_{y^-}$:
- $T_{y^+} = 100$: $2 + 0.2 = 2.2$, so $z = 0.22$;
- $T_{y^+} = 10$: $0.2 + 0.2 = 0.4$, so $z = 0.04$.

$$\boxed{z = 0.22 \text{ vs } 0.04}$$

The same per-token change earns a 5.5× larger margin for the longer winner. That is the verbosity bias. SimPO's length-averaged reward would give both the same margin.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| LLM parameters | $W = \{W^{(1)}, \dots, W^{(L)}, W_E, W_U\}$; tied: $W_U^\top = W_E$ |
| RL state / action | $s_t = \{x_{1:T}, y_{<t}\}$, $a_t$ = next token |
| SFT completion-only loss | $-\frac{1}{N_{\text{valid}}}\sum_{t}\mathbb 1(y_t \ne -\infty)\log P_{t,y_t}$ |
| Preference pairs per prompt | $\binom K2$ |
| RM head | $r_\psi(x,y) = Wh + b$, $h = H(L,:)$ (EOS row) |
| BT strengths | $P(i \succ j) = p_i/(p_i + p_j)$, $p = e^{r}$ |
| Bradley–Terry | $P(y^+\succ y^-\mid x) = \sigma(r_\psi(x, y^+) - r_\psi(x, y^-))$; log-odds $= r_w - r_l$ |
| Reward-model loss | $-\log\sigma(r_\psi(x, y^+) - r_\psi(x, y^-))$ |
| Margin loss (Llama 2) | $-\log\sigma(z - m(y_w, y_l))$ |
| Plackett–Luce | $P = \prod_k e^{r_k}/\sum_{j\ge k}e^{r_j}$ |
| ORM (BCE) | $-\mathbb E[r\log r_\psi + (1-r)\log(1-r_\psi)]$ |
| PRM | $-\mathbb E[\sum_i y_{s_i}\log r_\psi(x,s_i) + (1-y_{s_i})\log(1-r_\psi(x,s_i))]$ |
| Base objective | $\max_\theta\mathbb E_{x,y\sim\pi_\theta}[r_\psi(x,y)]$ |
| RLHF objective | $\max_\theta\mathbb E[r_\psi(x, y)] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ |
| Critic loss (simple) | $\mathbb E_t[(V_\phi(s_t) - R_t)^2]$ |
| Per-token reward | $r_{i,t} = r^{\text{task}}_{i,t} + r^{KL}_{i,t} + r^{\text{other}}_{i,t}$; $r^{\text{task}}$ only at $T_i$; $r^{KL} = -\beta[\log\pi^{\text{old}} - \log\pi_{\text{ref}}]$ |
| Value loss | $\frac{1}{\sum_iT_i}\sum_i\sum_t(V_\phi(s_{i,t}) - V^*(s_{i,t}))^2$ |
| MC target | $V^{MC} = \sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}$ (unbiased, high variance) |
| TD(0) target | $V^{TD} = r_{i,t} + \gamma V^{\text{old}}_\phi(s_{i,t+1})$ (low variance, biased) |
| TD error | $\delta_{i,t} = r_{i,t} + \gamma V^{\text{old}}_\phi(s_{i,t+1}) - V^{\text{old}}_\phi(s_{i,t})$ |
| GAE | $A^{\text{GAE}}_{i,t} = \sum_{l=0}^{T_i - t}(\gamma\lambda)^l\delta_{i,t+l}$; $\lambda \in [0.9, 0.98]$ typical |
| $\lambda$-return | $V^{GAE}(s_{i,t}) = V^{\text{old}}(s_{i,t}) + A^{GAE}_{i,t}$ |
| Standardized advantage | $\tilde A = (A - \mu_A)/\sigma_A$ over all valid tokens |
| Probability ratio | $\rho_{i,t} = \exp(\log\pi_\theta - \log\pi^{\text{old}}_\theta)$ |
| CPI surrogate | $\mathcal L_{CPI} = \frac1U\sum_i\sum_t\rho_{i,t}A_{i,t}$, $U = \sum_iT_i$ |
| TRPO | $\max\mathbb E[\rho A]$ s.t. $\mathbb E[\text{KL}(\pi^{\text{old}}\|\pi_\theta)] \le \delta$; penalty $\mathbb E[\rho A - \beta\text{KL}]$ |
| PPO clipped objective | $\min(\rho A,\ \text{clip}(\rho, 1-\epsilon, 1+\epsilon)A)$; minimize $-\mathbb E[\cdot]$ |
| Full PPO loss | $-\mathcal L_{\text{clip}} + \mathcal L_V + \mathcal L_{KL} - \mathcal L_{ENT}$ |
| GRPO advantage | $A_{b,k} = r_{b,k} - \mu_b$ (prof.); $(r_k - \bar r)/\sigma(r)$ (normalized) |
| GRPO loss | $-\frac1K\sum_k\frac1{T_k}\sum_t[\min(\rho A, \text{clip}(\rho)A) - \beta\text{KL}]$ |
| DAPO | $-\frac{1}{\sum_kT_k}\sum_k\sum_t\ell_{k,t}$ |
| Dr. GRPO | $-\frac{1}{T_{\max}}\frac1K\sum_k\sum_t\ell_{k,t}$ |
| Optimal RLHF policy | $\pi^*(y\mid x) = \frac1{Z(x)}\pi_{\text{ref}}(y\mid x)e^{r(x, y)/\beta}$ |
| Implicit reward | $r(x, y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$ |
| DPO loss | $-\log\sigma(a - b)$, $a = \beta\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)}$, $b = \beta\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}$ |
| DPO, $\Delta$ form | $-\log\sigma(z)$, $z = \beta(\Delta_\theta - \Delta_{\text{ref}})$, $\Delta = \log\pi(y^+) - \log\pi(y^-)$ |
| DPO gradients | $\partial\ell/\partial\log\pi_\theta(y^\pm) = \pm\beta(\sigma(z) - 1)$ |
| SimPO | $-\log\sigma\big(\frac{\beta}{\lvert y^+\rvert}\log\pi_\theta(y^+) - \frac{\beta}{\lvert y^-\rvert}\log\pi_\theta(y^-) - \gamma\big)$ |

---

## Cheat Sheet & Recall Aids

**Mnemonics**
- **Three stages: "Learn, Imitate, Prefer":** pre-train (language) → SFT (answer format) → alignment (HHH preferences).
- **HHH trade-offs: "Help–Harm = refusals; Honest–Help = 'I don't know'; Harm–Honest = uncomfortable truths".**
- **Four RLHF models: "Actor, Judge, Anchor, Critic":** $\pi_\theta$, $r_\psi$, $\pi_{\text{ref}}$, $V_\phi$.
- **PPO's six steps: "Generate, Score, Advantage, Ratio, Surrogate, Clip" ("Good Students Always Read Slides Carefully").**
- **Surrogate lineage: CPI → TRPO → PPO** (unbounded → KL constraint → clip).
- **Advantage types:** PPO = **ALPS** (Absolute, Learned, Per-token, State-dependent); GRPO = **RNPI** (Relative, Not learned, Per-completion, state-Independent).
- **DPO fixes: "SS for length, IOA for the anchor":** SamPO and SimPO for verbosity; IPO, ORPO and α-DPO for a bad reference.

**Gotchas**
- SFT loss is **completion-only**: prompt labels are ignored ($-\infty$ / $-100$); attention still sees the prompt.
- The reward model learns *order*, not *scale*: only $r^+ - r^-$ enters the loss. BT strength is $e^{r}$.
- One sigmoid of the reward **difference**, not a difference of sigmoids. The margin goes **inside** the sigmoid.
- The RM head is linear, $Wh + b$, read from the **EOS** row. A sigmoid appears only for ORM/PRM (probability of correctness).
- The KL term is a **penalty**: subtract $\beta\log(\pi_\theta/\pi_{\text{ref}})$.
- $\rho$ compares $\pi_\theta$ with $\pi^{\text{old}}$ (the sampler), **not** with $\pi_{\text{ref}}$. TRPO's KL is to $\pi^{\text{old}}$; RLHF's KL is to $\pi_{\text{ref}}$.
- The clip multiplies **$A$** too, and the $\min$ is pessimistic. The gradient is zero only in regions 4 and 5.
- In PPO, scores always take the **min**. "Max" is true only for ratios when $A < 0$.
- MC is unbiased with high variance; TD(0) is biased with low variance. GAE weights are $(\gamma\lambda)^l$: $\lambda = 0$ gives TD(0), $\lambda = 1$ gives MC.
- GRPO: every token in a completion shares one advantage, and the advantages in a group sum to 0. A group with identical rewards gives zero gradient.
- GRPO's $1/T_k$ under-penalizes **long wrong** answers. DAPO normalizes per token; Dr. GRPO divides by a constant.
- DPO has no reward model, no critic and no sampling, but $\pi_{\text{ref}}$ stays: the log-ratios need it. The professor's first line omits the $-\log$.
- At DPO initialization the loss is $\ln 2$ for every pair, and $\partial\ell/\partial\log\pi(y^+) = -\beta/2$.

**If you remember nothing else**
- RLHF = maximize reward − $\beta$·KL to the SFT model.
- PPO = clipped ratio × learned per-token advantage, plus a value loss (and optional KL and entropy terms).
- GRPO = PPO with the critic replaced by a group-mean baseline. Mind the $1/T_k$ length bias.
- DPO = the same objective solved in closed form, which turns it into a BT loss on policy log-ratios relative to the reference.

**Recall checklist**
- [ ] Draw the stage diagram (PLLM → SFT → RM → RLHF → ALLM → inference); list the PLLM's shortcomings and the HHH trade-offs.
- [ ] Fill in the three-step table (data, loss, update) and the workflow data sizes.
- [ ] Write the SFT chat formats and the completion-only masked loss.
- [ ] Describe preference collection ($K$ samples → ranking → $\binom K2$ pairs → batching).
- [ ] Draw the RM with EOS pooling. Derive the BT sigmoid from strengths, write the RM loss and its three variants, and the ORM and PRM losses.
- [ ] Write the base and KL-regularized objectives, and the two issues the critic addresses.
- [ ] List PPO's models (which are frozen?) and explain the outer/inner loops and $N = KB$.
- [ ] Write the per-token reward, the MC/TD(0)/GAE targets and the $\lambda$-return; redo the professor's GAE example.
- [ ] Write $\rho$, CPI, TRPO and the clipped objective; fill in the six-region table; write the full PPO loss.
- [ ] Compute GRPO advantages; explain the length bias and the DAPO and Dr. GRPO fixes.
- [ ] Derive $\pi^*$, invert it to an implicit reward, show $Z$ cancels; write the $\Delta$ form and the per-log-prob gradients; explain the reference's role.
- [ ] Explain DPO's verbosity and bad-anchor issues and name their fixes.

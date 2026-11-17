---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[LLMs III.pdf]]'
- '[[ATML Lec 13.pdf]]'
tags:
- atml
- course/cs6304
- rlvr
- reasoning-llm
- prompting
title: ATML — Lecture 9 — RLVR, Reasoning LLMs, and Using LLMs
lecture: 9
aliases:
- RLVR and Reasoning LLMs
- ATML Lecture 9
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_8 - LLM Alignment]] · [[ATML_Lecture_10 - Fine-Tuning]] →
%% nav:end %%

# ATML — Lecture 9: RLVR, Reasoning LLMs, and Using LLMs

## Where This Lecture Sits

*[[ATML_Lecture_8 - LLM Alignment|Lecture 8]] aligned an LLM with a reward learned from human preferences (PPO, GRPO) or with no explicit reward at all (DPO). All of those rely on an **outcome reward** that a learned model, or a human, hands out. This lecture asks where better feedback can come from, what changes when the model is trained to reason, what still goes wrong, and how a finished LLM is actually used.*

The sources are the professor's notes "LLMs III" (pages 5–8) and your ATML Lecture 13 (pages 2–5).

## Topic Inventory

- 1. Where the Feedback Comes From
  - 1.1 The outcome-reward problem and three sources of feedback
  - 1.2 RLAIF: AI feedback
  - 1.3 RLVR: a verifier in place of the learned reward model
  - 1.4 RLVR inside PPO and GRPO
  - 1.5 Properties of RLVR
- 2. Reasoning LLMs
  - 2.1 Same training stack, new data
  - 2.2 The reasoning trace
  - 2.3 Rewards for reasoning: ORM, PRM, RLVR
  - 2.4 Self-training and self-verification loops
- 3. The Reward–Utility Gap
  - 3.1 Utility vs reward
  - 3.2 Three examples
- 4. What Goes Wrong in Alignment
  - 4.1 Objective level
  - 4.2 Optimization level
  - 4.3 Length bias and behaviour after training
  - 4.4 When reward outweighs rules
- 5. Open and Future Directions
- 6. Using LLMs
  - 6.1 Six uses of an LLM
  - 6.2 Prompting techniques
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Where the Feedback Comes From

### 1.1 The outcome-reward problem and three sources of feedback

Your Lecture 13 notes name **the** problem in alignment as the **outcome reward**. Whether the optimizer is PPO, GRPO or DPO, the reasoning LLM (RLLM) learns from feedback on whole responses, and that feedback is only as good as its source. There are three sources:

| | Source of the label | Data / signal |
|---|---|---|
| **RLHF** | humans | $\mathcal D_{Pr} = (x, y^+, y^-)$; the feedback $y^+ \succ y^-$ is curated by humans |
| **RLAIF** | an AI model | the same preference format, but an AI decides $y^+ \succ y^-$; or an AI scores $K$ samples |
| **RLVR** | a verifier | $R_{VR} = \mathbb 1\big(\text{ExactAns}(y) = a(x)\big)$ |

Humans are expensive, slow and inconsistent. An AI judge is cheap but inherits its own biases. A verifier is exact but only exists for checkable tasks.

### 1.2 RLAIF: AI feedback

**Reinforcement learning from AI feedback (RLAIF)** keeps the RLHF machinery but replaces the human labeller with an AI model. Your notes give two forms:
- **(a) Preference form:** humans supply the prompt $x$ and the pair $y^+, y^-$, and an AI decides which is better ($y^+ \succ y^-$). The result is used exactly like human preference data (reward model + PPO, or DPO).
- **(b) GRPO-like form:** $x \to \pi_\theta \to y_1, y_2, \dots, y_K$, and the AI scores each of the $K$ samples directly. The scores then feed a group-relative advantage (Lecture 8 §6).

*(Expanding on this:)* the AI judge is usually a strong LLM prompted with a rubric or "constitution". This scales labelling to millions of comparisons, but the policy is now aligned to the judge's preferences, flaws included (§3.2).

### 1.3 RLVR: a verifier in place of the learned reward model

**RL with verifiable rewards (RLVR)** replaces the learned reward model $r_\psi(\cdot)$ with a **verifiable reward model** $r_v(\cdot)$, a program that checks the answer:

| Domain | Verifier |
|---|---|
| Math | check the final answer's correctness |
| Code | run unit tests |
| Reasoning | run a checker |

Everything else stays the same. The models kept are $\pi_\theta^{\text{SFT}}$, $\pi_\theta^{\text{new}}$ and, for PPO, $V_\phi(s)$; only $r_\psi$ is crossed out.
- **PPO** still uses $V_\phi(s)$ and GAE.
- **GRPO** still uses the group-relative advantage.

For a prompt $x$ (a problem) and a completion $y$ (steps plus answer):
1. extract the **final answer**, $\text{final\_answer}(y)$;
2. look up the **ground truth**, $g(x)$;
3. reward the match:

$$\boxed{r_{\text{task}}(x, y) = \mathbb 1\big[\text{final\_answer}(y) = g(x)\big] + \text{optional shaping} = r_v(\cdot) \in [0, 1]}$$

The **optional shaping** can add partial credit or penalties (format, length, etc.). Your Lecture 13 notes write the same reward as $R_{VR} = \mathbb 1(\text{ExactAns}(y) = a(x))$.

RLVR aligns the model with **ground-truth correctness** rather than with human intent. Its reward can't be flattered and doesn't drift, but its **coverage is limited**: it only applies where an answer can be checked automatically (Lecture 8 §3.7).

### 1.4 RLVR inside PPO and GRPO

**For PPO**, spread the task reward to the per-token reward, with the usual KL shaping (Lecture 8 §5.3):

$$r_{i,k,t} = r_{\text{task},i} + r^{KL}_{i,k,t}, \qquad r^{KL}_{i,k,t} = -\beta\Big(\log\pi_\theta^{\text{old}}(y_{i,k,t}\mid s_{i,k,t}) - \log\pi_{\text{ref}}(y_{i,k,t}\mid s_{i,k,t})\Big)$$

*(As in Lecture 8 §5.3, the sparse $r_{\text{task}}$ is normally placed at the last token $t = T$, with only the KL term at earlier tokens. The notes write it without a time index.)*

**For GRPO**, fold the KL into one reward per completion. For each prompt $x_b$ and its $K$ roll-outs:

$$\boxed{r_{b,k} = r_{\text{task}}(x_b, y_{b,k}) - \beta\sum_t\Big(\log\pi_\theta^{\text{old}}(y_{b,k,t}\mid s_{b,k,t}) - \log\pi_{\text{ref}}(y_{b,k,t}\mid s_{b,k,t})\Big)}, \qquad A_{b,k} = r_{b,k} - \mu_b$$

The rest proceeds as in Lecture 8 §6. This variant puts the KL in the reward, unlike Lecture 8's GRPO, which put it in the loss.

### 1.5 Properties of RLVR

1. **Sparse at the start.** $r_v(\cdot) \in [0, 1]$ is one number per complete completion, and most samples score **zero at the start** of training. So the gradient is $\nabla = 0$ much of the time, mostly for long contexts. *(Expanding on why it's exactly zero in GRPO:)* if every roll-out in a group scores 0, then $\mu_b = 0$ and every $A_{b,k} = 0$ (Lecture 8, Problem 12). The prompt contributes nothing.
2. **High variance between prompts.** Some prompts are trivial, so the reward is easy to get. Others are difficult, so it is hard to get. Trivial prompts give all-1 groups and impossible ones all-0 groups; both are useless. *(Common remedies, added: filter or resample prompts to keep mixed groups, e.g. DAPO's dynamic sampling, and use a difficulty curriculum.)*
3. **Verifier noise.** The verifier might be imperfect: an answer-format mismatch can mark a right answer wrong, and weak unit tests can pass wrong code.

**Case studies:** DeepSeek-R1, Kimi K1.5 and Qwen3 all use RLVR.

---

## 2. Reasoning LLMs

RLVR made it practical to train models on long, checkable problems. **Reasoning LLMs** are what comes out.

### 2.1 Same training stack, new data

The training stack is the same:

$$\underbrace{\text{Pre-train}}_{\text{same}} \to \underbrace{\text{SFT}}_{\mathcal D_{SFT}} \to \underbrace{\text{Alignment}}_{\mathcal D_{Pr}}$$

Pre-training is unchanged. What changes is the data of the later stages. Your notes summarize it as: pre-train on $\{x_i\}$, SFT on $\{x_i, y_i\}$ **with steps added**, align on $\mathcal D_{Pr} = (x, y^+, y^-)$ **with steps added**.

**$\mathcal D_{SFT}$ changes** from $(x, y)$ to $(x, y_{CoT})$:
- $y$ contains **detailed multi-step solutions** instead of a short helpful answer;
- $y$ has proofs, derivations, code, comments, etc.;
- intermediate steps are **annotated**, e.g. "First write the equation as …, then isolate $x$, …".

### 2.2 The reasoning trace

$y_{CoT}$ is long, structured reasoning, with a conceptual split:

$$y_{CoT} = \big\{\langle\text{Reasoning tokens}\rangle,\ \langle\text{Answer tokens}\rangle\big\}$$

The reasoning portion is called the **reasoning trace**, $\{y_1, \dots, y_{T_{\text{think}}}\}$. A full roll-out, one trajectory, is

$$\big(x,\ \underbrace{y_{1:T_{\text{think}}}}_{\text{reasoning trace}},\ \langle\text{Answer tokens}\rangle\big)$$

- **Reasoning trace:** $y_{1:T_{\text{think}}}$ for a given $x$.
- **Reasoning tokens:** each $y_t \in y_{1:T_{\text{think}}}$.
- **Reasoning vocabulary:** the subset of token types that are characteristic of reasoning traces. *(Examples, added: "wait", "let me check", "so", markers that open and close the trace such as the ⟨start thinking⟩ token of Lecture 7 §2.1.)*

**Classical chain-of-thought (CoT) vs reasoning LLMs.** In classical CoT the trace is shown to the user ("let's think step by step…"). Reasoning LLMs generate an **internal CoT** ("reasoning tokens"), then hide it and show only the final answer.

### 2.3 Rewards for reasoning: ORM, PRM, RLVR

**$r_\psi(\cdot)$ changes.** The preference triples $(x, y^+, y^-)$ now have to reflect:
- the **correctness of the final answer**;
- the **quality of the reasoning**;
- the **style**: a non-rambling CoT.

$r_\psi(\cdot)$ can be a learned model of two kinds (Lecture 8 §3.5):
- **ORM:** one reward for the final answer.
- **PRM:** a per-step reward,
$$r_t = \begin{cases}+1 & \text{step correct (✓)}\\ -1 & \text{step wrong (✗)}\\ 0 & \text{otherwise}\end{cases}$$

Often **RLVR** is used instead (§1.3). *(Trade-off, added:)* the ORM and RLVR only see the outcome, so they can reward a right answer reached by wrong reasoning (§3.2, RLVR example). The PRM sees each step but needs step-level labels and can itself be hacked.

### 2.4 Self-training and self-verification loops

The model generates many candidate solutions, a verifier filters them, and the surviving solutions become new SFT data:

$$\pi_\theta \xrightarrow{\text{many candidates}} \text{verifier} \xrightarrow{\text{survivors}} \mathcal D_{SFT} \xrightarrow{\text{SFT}} \pi_\theta$$

This is rejection sampling (Lecture 8 §3.7) with a verifier as the judge. Each round the model learns from its own best verified work.

---

## 3. The Reward–Utility Gap

Every method so far optimizes *a* reward. Your Lecture 13 notes make precise why that is not the same as optimizing what we want.

### 3.1 Utility vs reward

- $U(x, y)$ is the **true utility** of $y$: how well it serves the true intent with respect to the HHH framework.
- $R(x, y)$ is the **reward available** for $y$: what the human, AI judge or verifier actually reports.

Ideally $U(x, y) = R(x, y)$. The gap is

$$\boxed{\Delta(x, y) = R(x, y) - U(x, y)}$$

$\Delta > 0$ means the response is **over-rewarded**. An optimizer that maximizes $R$ will drift toward over-rewarded responses. That drift is **reward hacking** (§4.2).

### 3.2 Three examples

| Setting | Prompt | Responses | $R_A$ | $U_A$ | $R_B$ | $U_B$ |
|---|---|---|---|---|---|---|
| **RLHF** | "Explain overfitting" | $y_A$: brief and precise; $y_B$: long and repetitive | 0.75 | 0.95 | 0.94 | 0.65 |
| **RLAIF** | a drug whose human trials are uncertain | $y_A$: admits the uncertainty; $y_B$: builds a convincing story around it | 0.65 | 1 | 0.92 | 0.35 |
| **RLVR** | a math question, answer 42 | $y_A$: valid steps + 42; $y_B$: invalid steps + 42 | 1 | 1 | 1 (binary reward) | 0.35 |

- **RLHF:** the true utility prefers $y_A$, brief and precise. The reward model **takes fluency as a reward measure** and scores the long answer higher. $\Delta_B = +0.29$, so the optimizer learns verbosity.
- **RLAIF:** the AI judge prefers the confident story. $\Delta_B = +0.57$, so the optimizer learns to sound sure: honesty loses to persuasiveness.
- **RLVR:** the binary verifier can't tell the two apart ($R_A = R_B = 1$), though $y_B$'s reasoning is wrong. $\Delta_B = +0.65$. Outcome-only verification rewards lucky or invalid reasoning.

In all three, the response with the larger positive gap wins under optimization. The source of the gap differs: fluency bias (humans), persuasiveness bias (AI judge), blindness to the process (verifier).

---

## 4. What Goes Wrong in Alignment

The reward–utility gap is one failure. The professor's notes sort the issues of alignment and RLHF into four levels.

### 4.1 Objective level

- **The objective is a proxy.** We use SFT + $r_\psi(\cdot)$ + KL as a proxy for HHH. During policy optimization we push toward a scalar reward, so **its mistakes may dominate**.
- **Preference data is noisy, biased and sparse.**
- **One scalar hides the trade-offs.** In effect, reward $= \text{helpfulness} + \lambda\times\text{harmlessness} - \beta\times\text{toxicity} + \cdots$. A single scalar makes the trade-offs between H, H and H **opaque**.

### 4.2 Optimization level

- **Noisy gradients.** The sequence space is huge but the reward is sparse, so the gradients are noisy and high-variance.
- **Long reasoning traces** need a reward that is scalable and automatic, with no humans in the loop (hence RLVR, §1.3).
- **Over-parameterization → reward hacking.** Because of their enormous size, models learn to exploit quirks in $r_\psi(\cdot)$ (or in the verifier). This is also called **reward hacking**.
- **Mode collapse.** Probability mass concentrates on the narrow patterns that perform well, so diversity is lost.
- **KL spikes.** If we push too hard, we move away from $\theta_{\text{SFT}}$, and the KL term jumps.

### 4.3 Length bias and behaviour after training

**Length / verbosity bias:** RLHF models become over-verbose. This is the RLHF row of §3.2, and DPO's verbosity bias in Lecture 8 §7.5.

**Behaviour after training:**
- **Jailbreaks:** crafted prompts that get around the guardrails.
- **OOD generalization:** alignment training mostly sees "nice" prompts. In the wild there are multi-turn set-ups, long-horizon tool use and new topics, all out of distribution for the alignment data (compare the distribution-shift lectures, [[ATML_Lecture_5_1 - Domain Adaptation|5.1]] and [[ATML_Lecture_5_2 - Domain Generalization|5.2]]).

### 4.4 When reward outweighs rules

![[03 - Attachments/ATML_L9_rules_monitor.png|560]]

*Your Lecture 13 sketch:*
- *An LLM operates under a list of rules, and a separate AI model **monitors** it.*
- *When **reward > rules**, the monitoring AI model "develops sympathy with the LLM and against the rules", allowing it to **break a rule to solve the task**.*

*(Interpretation:)* if a model's training rewards task success more strongly than rule-following, breaking a rule becomes the higher-reward strategy. A monitor that is itself a learned model can come to share that judgement and let the violation through. Oversight by an AI is only as reliable as the objective the AI was trained on. This is the reward–utility gap of §3 applied to the overseer.

---

## 5. Open and Future Directions

![[03 - Attachments/ATML_L9_open_directions.png|660]]

| Direction | Approach |
|---|---|
| **Better objectives than PPO** | direct preference optimization (DPO and its variants) that avoids some pathologies of RLHF |
| **Better reward and verification** | combine human preference + rules + automated checks |
| **More robust optimization** | DRO-type methods |
| **Understanding the policy $\pi_\theta$** | mechanistic interpretability and probes on reasoning traces |
| **Monitoring the policy $\pi_\theta$** | detect reward hacking early |

*(Expanding on DRO:)* **distributionally robust optimization** minimizes the worst-case expected loss over a set of distributions near the training one, instead of the average loss on the training distribution. That directly targets the OOD failures of §4.3.

Your Lecture 13 notes add two current directions:
1. **RLVR with more robust tools and verifiers.**
2. **Hybrid rewards**, each source covering what it is good at:

$$\boxed{R_{\text{hybrid}} = \lambda_1 R_{HF} + \lambda_2 R_{VR} + \lambda_3 R_{AIF}}$$

Here $R_{HF}$ calibrates for human values, $R_{VR}$ covers math and coding, and $R_{AIF}$ covers safety.

---

## 6. Using LLMs

With a trained and aligned model in hand, the last question is how it is used.

### 6.1 Six uses of an LLM

1. **Inference:** deploy it as a general-purpose assistant.
2. **SFT data creation:** use it to create SFT data.
3. **RL roll-outs:** generate the $K$ roll-outs of a prompt (Lecture 8 §5.2, §6.2).
4. **Evaluation:** run it on a test prompt set to check how good it is, e.g. A/B testing.
5. **Distillation:** train a small student model $S$ on the reasoning traces of a teacher model $T$.
6. **Tools and system building:** RAG + agents + pipelines.

In all of these the model is used the same way:

$$\text{Prompt } x \to \theta_{LLM} \to \text{Response } y$$

**Prompting techniques** decide how you call $\theta_{LLM}$.

### 6.2 Prompting techniques

**(1) Reason-enhancing techniques**

| Technique | Idea |
|---|---|
| **Chain of thought (CoT)** | break the reasoning into steps, for logical tasks |
| **Self-consistency** | sample multiple CoT responses $y$ and choose the most common answer |
| **Tree of thought** | explore multiple reasoning branches (with look-ahead and backtracking) |
| **ReAct** (Reasoning + Acting) | interleave thought steps with tool, function or API calls |

**(2) Core techniques**

| Technique | Idea |
|---|---|
| **Zero-shot prompting** | ask $\theta_{LLM}$ to produce $y$ with no examples |
| **One-shot prompting** | the same, with one example |
| **Few-shot prompting** | the same, with a few examples (in-context learning, Lecture 8 §1.2) |

**(3) Instruction and role-based techniques**

| Technique | Idea |
|---|---|
| **Instruction prompting** | clearly define the task ("Summarize this in one sentence") |
| **System / role prompting** | set the model's persona ("You are a legal assistant") |
| **Instruction + few-shot** | combine both |

**(4) Prompt-composition techniques**

| Technique | Idea |
|---|---|
| **Prompt chaining** | use the $y$ of one prompt as the $x$ of another: $x_A \to y_A = x_B \to y_B$ |
| **Dynamic prompting** | insert context into $x$ programmatically (e.g. retrieved documents) |
| **Meta prompting** | ask $\theta_{LLM}$ to review its own $y$ |

**(5) Multimodal prompting**

| Technique | Example |
|---|---|
| **Image + text prompting** | text + image in $x$; task: captioning |
| **Audio / video prompting** | voice or video in $x$; task: meeting notes |

---

## Numerical Problems

**Problem 1: verifier reward with shaping.** A math verifier gives $\mathbb 1[\text{correct}]$ plus $0.1$ for the correct answer format. The total is clipped to $[0, 1]$, and $0.05$ is deducted for each step beyond 10. Three completions are:
- $y_1$: correct, formatted, 8 steps;
- $y_2$: wrong, formatted, 6 steps;
- $y_3$: correct, unformatted, 14 steps.

Find the rewards.
- $y_1$: $1 + 0.1 = 1.1$, clipped to $1$.
- $y_2$: $0 + 0.1 = 0.1$.
- $y_3$: $1 + 0 - 0.05\times4 = 0.8$.

$$\boxed{r = [1.0,\ 0.1,\ 0.8]}$$

Shaping gives the wrong answer some signal and penalizes rambling, at the risk of being gamed (e.g. perfect format, wrong answer).

**Problem 2: GRPO with RLVR and the KL folded in.** $K = 4$ roll-outs have task rewards $[1, 0, 0, 1]$ and summed log-ratios $\sum_t(\log\pi^{\text{old}} - \log\pi_{\text{ref}}) = [0.5, 0.2, 0.1, 1.5]$, with $\beta = 0.1$. Find $r_{b,k}$ and the advantages.

$r_{b,k} = r_{\text{task}} - 0.1\times(\cdot) = [0.95, -0.02, -0.01, 0.85]$, so $\mu_b = 1.77/4 = 0.4425$.

$$\boxed{A = [0.5075,\ -0.4625,\ -0.4525,\ 0.4075]}$$

The fourth roll-out is right but has drifted far from the reference, so it gets less credit than the first.

**Problem 3: how often is the gradient zero?** Each roll-out solves a prompt independently with probability $p$, and $K = 8$. Find the probability that a group has zero advantage (all fail or all succeed) for $p = 0.05$ and for $p = 0.5$.

- $p = 0.05$: $0.95^8 + 0.05^8 = 0.663$.
- $p = 0.5$: $2\times0.5^8 = 0.0078$.

$$\boxed{66\%\ \text{of hard prompts give no signal};\ \text{under } 1\%\ \text{at } p = 0.5}$$

This is property 1 of §1.5 in numbers. It is why mixed-difficulty prompts are kept.

**Problem 4: the reward–utility gap.** Using §3.2, find $\Delta$ for both responses in each setting, and which response a reward maximizer prefers versus which the user prefers.

| Setting | $\Delta_A$ | $\Delta_B$ | Reward picks | Utility picks |
|---|---|---|---|---|
| RLHF | $0.75 - 0.95 = -0.20$ | $0.94 - 0.65 = +0.29$ | $B$ | $A$ |
| RLAIF | $0.65 - 1 = -0.35$ | $0.92 - 0.35 = +0.57$ | $B$ | $A$ |
| RLVR | $1 - 1 = 0$ | $1 - 0.35 = +0.65$ | tie | $A$ |

$$\boxed{\text{In every case the optimizer favours (or can't reject) } y_B \text{, the response the user values less.}}$$

**Problem 5: hybrid reward.** With $\lambda = (0.5, 0.3, 0.2)$, $R_{HF} = 0.8$, $R_{VR} = 1$ and $R_{AIF} = 0.4$, find $R_{\text{hybrid}}$, and again if the safety judge drops to $R_{AIF} = 0$.

- $0.5(0.8) + 0.3(1) + 0.2(0.4) = 0.40 + 0.30 + 0.08 = 0.78$.
- With $R_{AIF} = 0$: $0.70$.

$$\boxed{R_{\text{hybrid}} = 0.78 \to 0.70}$$

A safety failure costs only 0.08 here. This is the "opaque trade-off" of §4.1: the weights decide how much harm is tolerated.

**Problem 6: self-consistency.** Three independent CoT samples are each correct with probability $0.6$, and the answer is taken by majority vote. Assume the wrong answers don't agree with each other, so a correct majority needs at least 2 correct. Find the accuracy.

$P(\ge 2\text{ correct}) = \binom32(0.6)^2(0.4) + (0.6)^3 = 0.432 + 0.216$.

$$\boxed{0.648 > 0.6}$$

Voting over samples beats a single sample whenever each sample is better than chance.

**Problem 7: ORM vs PRM.** A 4-step solution has a wrong step 3 but reaches the right final answer. Find the ORM / RLVR reward and the PRM step rewards, with their sum and minimum.

- ORM / RLVR: $1$, since only the final answer is checked.
- PRM: $[+1, +1, -1, +1]$, with sum $2$ and min $-1$.

$$\boxed{\text{ORM} = 1;\quad \text{PRM} = [+1, +1, -1, +1]}$$

Only the PRM flags the bad step. Aggregating by the minimum makes the flaw decisive.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| RLVR reward | $r_{\text{task}}(x,y) = \mathbb 1[\text{final\_answer}(y) = g(x)] + \text{shaping} \in [0,1]$ |
| RLVR in PPO | $r_{i,k,t} = r_{\text{task},i} + r^{KL}_{i,k,t}$, $r^{KL} = -\beta(\log\pi^{\text{old}} - \log\pi_{\text{ref}})$ |
| RLVR in GRPO | $r_{b,k} = r_{\text{task}} - \beta\sum_t(\log\pi^{\text{old}} - \log\pi_{\text{ref}})$; $A_{b,k} = r_{b,k} - \mu_b$ |
| Reasoning roll-out | $(x, y_{1:T_{\text{think}}}, \langle\text{Answer tokens}\rangle)$ |
| PRM step reward | $r_t = +1$ (✓), $-1$ (✗), $0$ otherwise |
| Reward–utility gap | $\Delta(x,y) = R(x,y) - U(x,y)$ |
| Hybrid reward | $R_{\text{hybrid}} = \lambda_1R_{HF} + \lambda_2R_{VR} + \lambda_3R_{AIF}$ |
| Zero-signal group | $P = p^K + (1-p)^K$ |
| Self-consistency (3 samples) | $P = 3p^2(1-p) + p^3$ |

---

## Cheat Sheet & Recall Aids

**Mnemonics**
- **Feedback sources: "Humans, AIs, Verifiers" = RLHF, RLAIF, RLVR**, in order of rising objectivity and shrinking coverage.
- **RLVR's three properties: "Zero, Varied, Noisy":** zero gradients at the start, variance between prompts, verifier noise.
- **Four failure levels: "Objective, Optimization, Over-verbose, Out-of-distribution".**
- **Prompting families: "Reason, Core, Role, Compose, Multimodal".**

**Gotchas**
- RLVR replaces **only** $r_\psi$. PPO keeps $V_\phi$ and GAE; GRPO keeps the group advantage.
- An all-0 (or all-1) GRPO group has $A = 0$: no learning from that prompt.
- An outcome verifier can't see the reasoning, so invalid steps + right answer score 1. Use a PRM to see steps.
- Classical CoT **shows** the trace; reasoning LLMs **hide** an internal trace and show the answer.
- $\Delta = R - U > 0$ marks over-rewarded responses, and optimization moves toward them.
- Self-consistency samples several CoTs and **votes**. Tree of thought **branches**. ReAct **calls tools**.

**If you remember nothing else**
- RLVR = RL with a program as the reward. It is exact where it applies, applies only where checkable, and is sparse.
- Reasoning LLMs = the same stack, trained on step-by-step data, with ORM/PRM/RLVR rewards.
- Every alignment method optimizes $R$, never $U$. The gap $\Delta$ explains verbosity, persuasion and lucky reasoning.
- Open directions: better objectives, hybrid rewards, robust optimization, interpretability and monitoring.

**Recall checklist**
- [ ] Contrast RLHF, RLAIF and RLVR by label source, signal and weakness.
- [ ] Write the RLVR reward and plug it into PPO (per token) and GRPO (KL folded in).
- [ ] State the three properties of RLVR and compute the zero-signal probability.
- [ ] Describe how SFT data, the trajectory and the reward change for reasoning LLMs.
- [ ] Define $U$, $R$ and $\Delta$; work the three examples.
- [ ] List the four levels of alignment issues and the five open directions.
- [ ] Name the six uses of an LLM and the five prompting families with their techniques.

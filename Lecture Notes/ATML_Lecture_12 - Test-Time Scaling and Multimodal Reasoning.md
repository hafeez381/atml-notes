---
type: lecture
course: "[[_CS6304 ATML|CS6304 ATML]]"
source:
  - "[[Test Time Scaling Notes.pdf]]"
  - "[[ATML Lec 19.pdf]]"
  - "[[ATML Lec 20.pdf]]"
tags:
  - atml
  - course/cs6304
  - test-time-scaling
  - reasoning
  - vlm
title: ATML — Lecture 12 — Test-Time Scaling and Multimodal Reasoning
lecture: 12
aliases:
  - Test-Time Scaling
  - ATML Lecture 12
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_11 - Vision-Language Models]] · → [[ATML_Lecture_13 - Model Compression]]
%% nav:end %%

# ATML — Lecture 12 — Test-Time Scaling and Multimodal Reasoning

*[[ATML_Lecture_7 - LLM Foundations#5. Scaling Laws|Lecture 7]] showed how loss falls as training scale grows, and listed inference-time laws as a newer direction ([[ATML_Lecture_7 - LLM Foundations#5.4 A roadmap, and scaling laws beyond pre-training|§5.4]]). [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#2. Reasoning LLMs|Lecture 9]] trained models that reason before answering, and [[ATML_Lecture_11 - Vision-Language Models#10.8 How to reason on images?|Lecture 11]] ended by asking how a VLM should reason on images. This lecture answers both threads. It first treats inference compute as a resource to scale, in four families (sequential, parallel, hybrid, latent), and then carries reasoning over to vision-language models: why it is harder there, the algorithms used, and reasoning in a VLM's hidden states.*

The sources are the professor's [[Test Time Scaling Notes.pdf|Test Time Scaling Notes]] (9 pages) and your class notes [[ATML Lec 19.pdf|ATML Lecture 19]] (5 pages) and [[ATML Lec 20.pdf|ATML Lecture 20]] (pages 1–3, up to "Model Compression", which is covered in [[ATML_Lecture_13 - Model Compression|Lecture 13]]). Pages 1–5 of the professor's notes give [[#1. Why Scale Compute at Test Time|§1]]–[[#4. Hybrid Scaling — Think Selectively|§4]], and pages 6–9 give [[#6. Reasoning in VLMs|§6]]–[[#8. Latent Reasoning in VLMs|§8]]. [[#5. Latent Scaling — Think Implicitly|§5]] comes from your Lecture 20 notes. All figures are redrawn in TikZ from both sets of notes.

## Topic Inventory

- 1. Why Scale Compute at Test Time
  - 1.1 From training-time scaling to inference compute
  - 1.2 Four design questions
  - 1.3 What counts as inference compute
  - 1.4 Four families of test-time scaling
- 2. Sequential Scaling — Think Longer
  - 2.1 The reasoning trace as intermediate context
  - 2.2 Where a single chain fails
  - 2.3 Plan-and-Solve
- 3. Parallel Scaling — Think Wider
  - 3.1 Sampling K trajectories
  - 3.2 Self-consistency (majority voting)
  - 3.3 Best-of-N with a verifier
  - 3.4 Self-aggregation
- 4. Hybrid Scaling — Think Selectively
  - 4.1 Solution coverage vs selection accuracy
  - 4.2 Outcome verification
  - 4.3 Process verification
  - 4.4 From verification to search
  - 4.5 Self-refinement
- 5. Latent Scaling — Think Implicitly
  - 5.1 The cost of explicit reasoning tokens
  - 5.2 Latent thoughts in an LLM
  - 5.3 What latent reasoning gives up
- 6. Reasoning in VLMs
  - 6.1 Reasoning as a distribution over chains
  - 6.2 Text-only vs multimodal reasoning
  - 6.3 Four challenges unique to VLMs
- 7. Algorithms for Multimodal Reasoning
  - 7.1 SFT on multimodal chains of thought
  - 7.2 RLVR with GRPO
  - 7.3 Tree search and MCTS
  - 7.4 RAG and tool use
  - 7.5 Latent reasoning
- 8. Latent Reasoning in VLMs
  - 8.1 Explicit vs latent chain of thought
  - 8.2 Layer-wise vertical refinement
  - 8.3 Token-wise horizontal latent thoughts
  - 8.4 How to get the latent vector
- Numerical Problems
- Formula Summary Sheet
- Summary

---

## 1. Why Scale Compute at Test Time

### 1.1 From training-time scaling to inference compute

Scaling laws ([[ATML_Lecture_7 - LLM Foundations#5.3 Chinchilla laws|Lecture 7 §5.3]]) say that a model's loss improves predictably as the model size $N$ and the training data $D$ grow, for a fixed compute budget $C$:

$$L(N, D) \propto L_0 + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}, \qquad C_{\text{train}} \propto ND$$

Here $L_0$ is the irreducible loss and $a, b, \alpha, \beta$ are fitted constants; this is the Chinchilla law of Lecture 7 with $E, A, B$ renamed. All of it is paid during training. A model such as o1 (OpenAI) or R1 (DeepSeek), shown in your notes as $x \to \theta^*_{\text{LLM}} \to y$, is already trained and aligned (pre-train + SFT + align), and its parameters $\theta^*$ are fixed. Your notes split the total compute into two parts:

$$C = C_{\text{train}} + C_{\text{inf}}$$

$C_{\text{inf}}$ is the compute spent when the model answers a query. Spending more of it moves cost from training to deployment.

The lecture's question is: **instead of training a larger model, can we make an existing model perform better by allocating more compute during inference?** The quantity of interest changes from the loss to the task accuracy, which now depends on inference compute as a third resource:

$$L(N, D) \longrightarrow \text{Accuracy}_{\text{task}}(N, D, C_{\text{inf}})$$

The **empirical observation** is that increasing $C_{\text{inf}}$ can **sometimes** outperform increasing $N$. *(Expanding on "sometimes": Snell et al. (2024) found that a smaller model with a well-chosen test-time strategy could beat a model about 14 times larger, but mainly on problems the small model could already solve some of the time. On problems far beyond the small model, extra test-time compute helped little and a larger model was better.)* This is **test-time scaling (TTS)**: improving answers by spending more computation per query, without changing $N$ or $D$.

![[03 - Attachments/ATML_L12_tts_setup.svg|360]]

*Your notes' view: a fixed model, with $C_{\text{inf}}$ as the extra input that test-time scaling turns up.*

Your notes list what the extra compute buys: generating more tokens, doing more in-depth reasoning, calling tools (computer vision or math tools), and running a verifier to check the response.

### 1.2 Four design questions

"How to do it?" breaks into four questions, which the professor writes as open questions with candidate answers:

![[03 - Attachments/ATML_L12_tts_taxonomy.svg|720]]

*(a) The four design questions with the professor's candidate answers. (b) The four families of [[#1.4 Four families of test-time scaling|§1.4]].*

| Question | What it decides | Answer from class |
|---|---|---|
| **What to scale?** | the quantity that grows with $C_{\text{inf}}$ | the length of the chain of thought (CoT), the number of samples for self-consistency, the number of tool calls |
| **How to scale?** | whether the parameters change | change the parameters with SFT or RL so the model learns to use more compute, or keep them fixed and run a search over responses |
| **Where to scale?** | which stage of answering gets the compute | reasoning time (producing the answer) or output evaluation (checking it) |
| **How well to scale?** | how a method is judged | accuracy (e.g. on VQA, visual question answering), efficiency (latency, FLOPs), controllability |

Your notes phrase the last question as "is there any better way to scale?". The professor's list also has **scaling** as a criterion. *(Expanding on the two less obvious criteria: **controllability** asks whether the amount of compute can be set in advance, for example with a thinking budget, and gives predictable behaviour. **Scaling** asks whether accuracy keeps improving as $C_{\text{inf}}$ grows, or levels off.)*

### 1.3 What counts as inference compute

$C_{\text{inf}}$ is not one number. The professor lists its forms: **generated tokens, model evaluations (forward passes), sampled solutions, verifier calls, latency, FLOPs, tool calls, etc.** Methods that use the same FLOPs can differ a lot in latency: ten samples generated in parallel take about as long as one, while ten rounds of revision take ten times as long.

### 1.4 Four families of test-time scaling

For a query $x$, the four main ways to answer "what to scale" give four families, each with a one-phrase summary from your notes:

| Family | Summary | What grows |
|---|---|---|
| **Sequential** ([[#2. Sequential Scaling — Think Longer\|§2]]) | think longer | the length of one response |
| **Parallel** ([[#3. Parallel Scaling — Think Wider\|§3]]) | think wider | the number of independent responses |
| **Hybrid** ([[#4. Hybrid Scaling — Think Selectively\|§4]]) | think selectively: check, revise, verify | samples plus a verifier that decides what to keep or expand |
| **Latent** ([[#5. Latent Scaling — Think Implicitly\|§5]]) | think implicitly: scale the LLM's latents | computation in hidden states, not in tokens |

The families build on each other. Parallel scaling repeats sequential sampling $K$ times. Hybrid scaling adds a judge to parallel scaling, so it can pick among samples or prune them early. Latent scaling keeps the extra computation but moves it out of the token space.

---

## 2. Sequential Scaling — Think Longer

The first family spends the extra compute on one longer answer.

### 2.1 The reasoning trace as intermediate context

Instead of mapping $x$ directly to $y$, the model first writes a **reasoning trace** $r$ and then the answer:

![[03 - Attachments/ATML_L12_sequential.svg|380]]

*Top: direct answer. Bottom: the same model writes a trace $r$ first, so $y$ is generated after it.*

The trace is generated token by token, so its probability factorizes by the chain rule:

$$P_\theta(r \mid x) = \prod_{t=1}^{T_r} P_\theta(r_t \mid x, r_{<t})$$

where $T_r$ is the trace length and $r_{<t}$ are the trace tokens before position $t$. Each trace token **provides intermediate context** that **influences subsequent tokens**, including the answer. Your notes put it as: the trace gives the model intermediate context before it starts generating the response, as a single, longer response. This is the [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#2.2 The reasoning trace|reasoning trace of Lecture 9 §2.2]].

The answer's probability is then a sum over all possible traces:

$$P_\theta(y \mid x) = \sum_r P_\theta(r, y \mid x) = \sum_r P_\theta(r \mid x)\,P_\theta(y \mid x, r)$$

*(Expanding on why this matters:)* generating one trace and reading its answer draws a single sample from this sum. The answer of one trace can be wrong even when the most probable answer under $P_\theta(y \mid x)$ is right. [[#3.2 Self-consistency (majority voting)|§3.2]] uses exactly this, by estimating the sum from many traces.

Sequential reasoning is **induced via prompting or SFT** (your notes): a "think step by step" prompt ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques|Lecture 9 §6.2]]) or fine-tuning on data with written-out steps ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#2.1 Same training stack, new data|Lecture 9 §2.1]]). The compute knob is $T_r$: each extra token costs one more forward pass.

### 2.2 Where a single chain fails

Ordinary CoT **greedily follows one trajectory**, and it can fail if it selected:
- a **poor decomposition** of the problem;
- a **missed step**;
- calculations **performed in the incorrect order**.

Your notes name the underlying problem: **a single wrong step will propagate further.** Every later token is conditioned on the earlier ones ($r_{<t}$ above), so an early mistake becomes part of the context the rest of the chain builds on, and the model does not go back to fix it.

### 2.3 Plan-and-Solve

**Plan-and-Solve** attacks the decomposition failure. The model:
1. first generates a **plan** $P$, deciding which steps are necessary before attempting them;
2. then constructs reasoning steps based on $P$.

By the chain rule, the joint probability of plan, trace and answer factorizes in that order:

$$P_\theta(P, r, y \mid x) = P_\theta(P \mid x)\,P_\theta(r \mid x, P)\,P_\theta(y \mid x, P, r)$$

The professor's question is: **a wrong initial plan?** It **contaminates the entire trajectory**, because both the trace and the answer are conditioned on $P$. Plan-and-Solve moves the critical early choice into the plan, but the run is still one path, so the propagation problem of [[#2.2 Where a single chain fails|§2.2]] remains. That is the motivation for the next family.

---

## 3. Parallel Scaling — Think Wider

### 3.1 Sampling K trajectories

In single-path (serial) generation, early choices strongly influence subsequent reasoning. So instead of one long attempt, **sample $K$ trajectories in parallel**:

$$\big\{(r^{(k)}, y^{(k)})\big\}_{k=1}^{K} \sim P_\theta(r, y \mid x)$$

This **increases the number of candidate solutions instead of the length of one candidate solution**. A bad early choice now ruins one sample, not the whole answer.

The samples must differ, or $K$ copies add nothing (greedy decoding would return $K$ identical traces). To ensure **diversity**, use:
- different sampling **seeds**;
- different **decoding settings**;
- the softmax **temperature** $\tau$ (written $\rho$ in [[ATML_Lecture_7 - LLM Foundations#6.3 Sampling with temperature|Lecture 7 §6.3]]).

Your notes put it as: the samples are stochastic through the decoding strategy. Parallel scaling then needs a rule for turning $K$ answers into one. The three rules below differ in what does the choosing: the samples' agreement, an external verifier, or the LLM itself.

![[03 - Attachments/ATML_L12_parallel.svg|640]]

*One model, $K$ diverse samples, and the three ways to merge them: vote ([[#3.2 Self-consistency (majority voting)|§3.2]]), verifier ([[#3.3 Best-of-N with a verifier|§3.3]]), or feed them back to the LLM ([[#3.4 Self-aggregation|§3.4]]).*

### 3.2 Self-consistency (majority voting)

**Successful candidate solutions may converge on the same answer, while unsuccessful paths may disagree.** **Self-consistency**, also called **majority voting**, uses that. Estimate how often each answer appears among the $K$ samples:

$$\hat P_K(y \mid x) = \frac{1}{K}\sum_{k=1}^{K}\mathbb 1\big(y^{(k)} = y\big)$$

and predict the most frequent one:

$$\boxed{\hat y_{sc} = \arg\max_y \hat P_K(y \mid x)} \qquad\qquad \text{(1)}$$
^eq-1

$\mathbb 1(\cdot)$ is the indicator function, 1 when its condition holds and 0 otherwise.

> [!warning] Possible source inconsistency
> The professor's notes write the indicator as $\mathbb 1(y^{(k)} = \text{correct})$. At test time the correct answer is unknown, and $\hat P_K(y \mid x)$ must count samples that equal the candidate $y$, so the indicator is $\mathbb 1(y^{(k)} = y)$, as above. Your notes write the estimate as $\hat p_\theta(y \mid x, r)$.

*(Expanding on why voting works:)* $\hat P_K(y \mid x)$ is a Monte Carlo estimate of the answer marginal $P_\theta(y \mid x) = \sum_r P_\theta(r \mid x)P_\theta(y \mid x, r)$ from [[#2.1 The reasoning trace as intermediate context|§2.1]]. Majority voting therefore picks the most probable answer of the whole distribution rather than the answer of one trace. Wrong paths tend to disagree because there are many ways to be wrong and usually one right answer, so wrong answers split their votes.

**The problems:**
- **The majority can be systematically wrong.** If the model has a consistent misconception, most traces make the same mistake and the vote picks it with confidence. More samples don't help, since the estimate just converges to the wrong mode.
- **It doesn't look at the reasoning traces at all** (your notes). A correct answer can be hallucinated at the end of wrong reasoning and still be counted as a good vote.
- *(Added:)* the answers must be comparable for equality, which works for numbers or multiple choice but not for free-form text.

This is the self-consistency prompting technique of [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques|Lecture 9 §6.2]], now seen as a way to spend compute.

### 3.3 Best-of-N with a verifier

**Best-of-N (BoN)** replaces the vote with a judge. Score each roll-out with a **verifier or reward model** $V_\phi$, with its own parameters $\phi$, and select the highest-scoring one:

$$\boxed{y_{\text{BoN}} = y^{(k^*)}, \qquad k^* = \arg\max_k V_\phi\big(x, r^{(k)}, y^{(k)}\big)} \qquad\qquad \text{(2)}$$
^eq-2

The professor lists three kinds of verifier:

| Verifier | What it scores | Earlier lecture |
|---|---|---|
| **Outcome-based** | the final answer only | ORM, [[ATML_Lecture_8 - LLM Alignment#3.5 Sparse rewards, ORMs and PRMs\|L8 §3.5]]; programmatic checker, [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.3 RLVR: a verifier in place of the learned reward model\|L9 §1.3]] |
| **Outcome + reasoning steps** | the answer and the steps that led to it | PRM, [[ATML_Lecture_8 - LLM Alignment#3.5 Sparse rewards, ORMs and PRMs\|L8 §3.5]] |
| **General response preference** | overall quality, as a human would rank it | Bradley–Terry reward model, [[ATML_Lecture_8 - LLM Alignment#3.3 The Bradley–Terry model and the reward loss\|L8 §3.3]] |

Your notes call $V_\phi$ a **ranking mechanism** and give an example that needs no separate model: the **sum of log-probabilities** of the response under the LLM itself.

Unlike self-consistency, BoN reads each sample on its own, so it can pick a correct minority answer. The cost is that **BoN is vulnerable to $V_\phi$ errors**. *(Expanding on why more samples can make this worse:)* with more samples, there is more chance that some wrong response happens to score highly. BoN then finds the verifier's mistakes, the inference-time version of reward hacking ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#3.1 Utility vs reward|Lecture 9 §3.1]]).

### 3.4 Self-aggregation

**Self-aggregation** makes the LLM the judge. Feed all $K$ roll-outs back to the LLM and have it produce the answer, conditioned on all of them:

$$\hat y_{SA} = P_\theta\Big(y \;\Big|\; x, \big\{r^{(i)}, y^{(i)}\big\}_{i=1}^{K}\Big)$$

Your notes say it has the model pick the best roll-out. Unlike voting, the model can read the traces; unlike BoN, it needs no separate verifier. The problems:
- **additional cost:** the aggregation prompt contains all $K$ traces, a long context, on top of the $K$ generations (your notes: multiple forward passes);
- **aggregation can introduce new errors**, since the aggregator is the same fallible model;
- **majority traces can overwhelm the LLM during aggregation**: your notes say majority voting will dominate and the aggregator collapses onto it, which brings back the first problem of [[#3.2 Self-consistency (majority voting)|§3.2]].

---

## 4. Hybrid Scaling — Think Selectively

All three rules of [[#3. Parallel Scaling — Think Wider|§3]] depend on choosing well. Your notes call this the **biggest problem of the parallel mechanism**: a faulty selection mechanism gives no improvement, no matter how much we increase the number of roll-outs. **Generating multiple answers is not sufficient unless we can identify which answer is correct.** The hybrid family, "think more selectively", combines generation with **verification, feedback and search**.

### 4.1 Solution coverage vs selection accuracy

Two different abilities decide the result:

| | Ability | Question |
|---|---|---|
| **Solution coverage** | generation | does the search discover the correct answer $y^*$ among the $K$ samples? |
| **Selection accuracy** | discrimination | are we able to identify $y^*$ among them? |

![[03 - Attachments/ATML_L12_hybrid.svg|560]]

*(a) Coverage is a property of the samples; selection is a property of the verifier. (b) Process verification ([[#4.3 Process verification|§4.3]]): the verifier sees growing prefixes of one trajectory.*

**Increasing $K$ improves coverage but not selection accuracy**, which motivates using some verification. *(Expanding with the standard formula:)* if each sample is correct with probability $p$, the chance that at least one of $K$ is correct is

$$\text{coverage}(K) = 1 - (1 - p)^K$$

which rises towards 1 as $K$ grows (this is the pass@$K$ metric). The final accuracy is about coverage times the chance that the selector picks $y^*$ when it is present. If the selector is poor, that second factor caps the accuracy no matter how large $K$ is ([[#Numerical Problems|Problem 5]]).

### 4.2 Outcome verification

**Outcome verification** is best-of-$K$ with a verifier $V_\phi$, the rule of [[#^eq-2|eq. (2)]] read as the selection step:

$$k^* = \arg\max_{k = 1, \dots, K} V_\phi\big(x, r^{(k)}, y^{(k)}\big), \qquad \hat y = y^{(k^*)}$$

Your notes write the outcome-based verifier with the answer only, $V_\phi(x, y^{(k)})$. Including $r^{(k)}$ lets the verifier look at how the answer was reached, which leads to the next idea.

### 4.3 Process verification

An outcome verifier says whether the answer is wrong but not **where the first incorrect step occurred**. **Process verification** scores each step. Split a trajectory into steps $r = (r_1, \dots, r_S)$, where each $r_s$ is a **complete reasoning step** (a sentence or line, not a token). The **process verifier** scores every prefix:

$$V_s = V_\phi(x, r_{\le s}), \qquad s = 1, 2, \dots, S$$

$V_s$ is the **estimated validity of the trajectory through step $s$**. As your figure shows, the verifier's first input is $(x, r_1)$, its second $(x, r_1, r_2)$, and so on: each step is judged in the context of everything before it. This is the PRM of [[ATML_Lecture_8 - LLM Alignment#3.5 Sparse rewards, ORMs and PRMs|Lecture 8 §3.5]] used at test time.

The step scores are combined into one trajectory score. The professor's example is the product:

$$V_{\text{process}}(x, r) = \prod_{s=1}^{S} V_s$$

Your notes write $V_{\text{PRM}}(x, r) = \sum\log V_s$. *(Expanding on the difference:)* if each $V_s$ is the probability that step $s$ is valid given a valid prefix, the product is the probability that every step is valid, and one bad step pulls the whole score down. Summing $\log V_s$ is the same ranking as the product; summing the raw $V_s$ is not, since it can reward a long chain with one broken step. Taking the minimum over steps is another common choice.

### 4.4 From verification to search

Process verification makes a further step possible. **If we can score intermediate states, there is no need to generate full solutions independently** and judge them only at the end. Instead, **search over partial reasoning trajectories**, dropping bad prefixes early.

Define the **state** after $t$ steps as the query plus the steps so far, $S_t = \{x, r_1, \dots, r_t\}$. From a state, sample several candidate next steps, indexed by $j$:

$$r_{t+1}^{(j)} \sim P_\theta(r_{t+1} \mid S_t), \qquad S_{t+1}^{(j)} = \big(S_t, r_{t+1}^{(j)}\big)$$

The candidates differ because of the stochasticity of the decoding strategy (your notes: the softmax temperature). The professor's example, for $S_t$ = "Solve $2x + 4 = 10$":
- $j = 1$: subtract 4 from both sides first;
- $j = 2$: divide both sides by 2 first;
- $j = 3$: rearrange as $2x = 6$.

Each leads to a different $S_{t+1}^{(j)}$. All three are valid here; in harder problems some candidates are wrong, and the verifier's job is to rank them. Score every candidate state and keep the best $K$:

$$V_\phi\big(S^{(j)}_{t+1}\big) \;\longrightarrow\; \text{top-}K \;\longrightarrow\; B_{t+1}$$

With $K = 2$ in the professor's example, $B_{t+1} = \{S_{t+1}^{(1)}, S_{t+1}^{(3)}\}$. Your notes' example keeps $B_{t+1} = \{S_{t+1}^{(2)}, S_{t+1}^{(9)}\}$ out of $J = 10$ candidates. $K$ is the **beam width**.

![[03 - Attachments/ATML_L12_beam_search.svg|720]]

*One expansion step from a single state: $J$ candidate next steps, scored by $V_\phi$, with the best $K$ kept.*

In actual **beam search**, we maintain $K$ states at time $t$. Each state generates $J$ candidate continuations, giving $KJ$ successor states, and we then select the best $K$ states from the entire candidate pool. This is the beam search of [[ATML_Lecture_7 - LLM Foundations#6.4 Beam search|Lecture 7 §6.4]] lifted from tokens to reasoning steps: there, partial sequences were ranked by cumulative log-probability, while here a verifier ranks them. Each depth costs $KJ$ step generations and $KJ$ verifier calls ([[#Numerical Problems|Problem 7]]).

### 4.5 Self-refinement

Search keeps the best of many attempts. **Self-refinement** improves one attempt instead: **instead of accepting the first solution, the LLM is asked to review its own answer, identify mistakes, and produce an improved solution, and this is repeated.** The professor's example, for $x$ = "What is 15% of 240?":

1. **$S_1$, generate an initial solution:** $r^{(0)}$: $240 \times 0.15 = 32$ (the first state and first reasoning trace).
2. **$S_2$, self-critique:** send $x$ and $r^{(0)}$ back to the LLM, asking it to check for mistakes. It returns feedback $f^{(0)}$: "The multiplication is incorrect. Since 10% of 240 is 24 and 5% is 12, the sum should be 36."
3. **$S_3$, revise the solution:** the model receives its original solution and the feedback $f^{(0)}$ and generates a new solution, $r^{(1)}$: $240 \times 0.15 = 36$.

![[03 - Attachments/ATML_L12_self_refine.svg|520]]

*One round of self-refinement. The revised solution becomes the input of the next round.*

In general, round $t$ alternates two conditional generations with the same model:

$$f^{(t)} \sim P_\theta\big(f \mid x, r^{(t)}\big), \qquad r^{(t+1)} \sim P_\theta\big(r \mid x, r^{(t)}, f^{(t)}\big)$$

The first generates feedback conditioned on $x$ and the current solution $r^{(t)}$; the second generates the revised solution using $x$, $r^{(t)}$ and $f^{(t)}$. Self-refinement is hybrid because the LLM acts as its own verifier, with text feedback instead of a scalar score, and is sequential across rounds. It is the meta prompting of [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques|Lecture 9 §6.2]] applied in a loop. *(Expanding on its limit:)* it only helps when the model can recognise its own error; a model that cannot may leave the error in place, or talk itself out of a correct answer.

---

## 5. Latent Scaling — Think Implicitly

The first three families all spend compute by writing tokens. Your Lecture 20 notes review those three and then turn to the fourth family, which spends compute without writing tokens.

### 5.1 The cost of explicit reasoning tokens

Inside the LLM, the query $x$ ($T \times d$, where $T$ is the total number of tokens in the query and $d$ the hidden size) passes through layers $1, \dots, L$, and the task head (the LM head) maps the last layer's states to $y$. Every layer outputs a $T \times d$ matrix of hidden states; your notes write these as $h_1, \dots, h_L$, and this note writes $Z_\ell$ to match the professor's $Z_L$ in [[#8. Latent Reasoning in VLMs|§8]].

![[03 - Attachments/ATML_L12_latent_llm.svg|560]]

*Top: the layer stack, every layer $T \times d$. Bottom: the token positions of a reasoning model, query then trace then answer.*

After the ⟨think⟩ token, the LLM generates the first token of the reasoning trace at position $T + 1$ and continues up to position $\tilde T$. It generates ⟨/think⟩ (your notes: ⟨EndThink⟩) at $\tilde T$ and then produces $y$.

**Problem:** the model has to **decode all reasoning traces**. Each trace token requires a search over the softmax vocabulary, about 300k entries even for a small model (your notes), and sampling then turns the whole $d$-dimensional hidden state into one token ID. This **motivates latent reasoning**. *(Expanding on what is lost:)* a token from a 300k vocabulary carries at most $\log_2 300{,}000 \approx 18$ bits, while the hidden state it came from has $d$ continuous numbers. Writing a token throws away whatever the model "considered" but did not commit to ([[#Numerical Problems|Problem 10]]).

### 5.2 Latent thoughts in an LLM

**Latent reasoning** skips the decoding. Instead of turning the hidden state at position $T + 1$ into a token, the model keeps it as a vector, a **latent thought**, and feeds it back as the next input. In your notes' notation, the latent thought is a function of the input and all layers' states at the earlier positions:

$$h_L^{(T+1)} = f\big(x, h_\ell^{(1:T)}\big)$$

where $h_\ell^{(t)}$ is the hidden state of layer $\ell$ at position $t$. *(Expanding with a named method: Coconut, "Chain of Continuous Thought" (Hao et al., 2024), feeds the last hidden state back as the next input embedding, in place of the embedding of a sampled token.)*

![[03 - Attachments/ATML_L12_latent_thought_llm.svg|560]]

*Latent thoughts: the hidden vector itself goes to the next position, and a latent reward model can score candidate thoughts.*

Your notes give two ways to get the latent thought:
1. **Average over layers:** $$h_L^{(T+1)} = \frac{1}{L}\sum_{\ell=1}^{L} h_\ell^{(T+1)}$$which mixes shallow and deep representations instead of using only the last layer.
2. **Score candidates with a latent reward model:** produce candidate thoughts $h_L^{(c)(T+1)}$ and pass each through a **latent reward model** that outputs a reward, keeping the best. This is a verifier ([[#4.2 Outcome verification|§4.2]]) that reads hidden vectors instead of text. 

The latent positions run from $T + 1$ to $\tilde T$, after which your notes write a token ⟨bor⟩.

> [!note] Unclear in source
> ⟨bor⟩ is not defined in your notes. Given its position (right after the latent span, where ⟨/think⟩ sat in [[#5.1 The cost of explicit reasoning tokens|§5.1]]), it most likely marks the beginning of the response.

### 5.3 What latent reasoning gives up

Your notes name the price: **TTS methods cannot be applied to latent reasoning, because the stochasticity of decoding is lost.** Parallel scaling ([[#3. Parallel Scaling — Think Wider|§3]]) relies on sampling different tokens to get different traces; a deterministic hidden-state loop returns the same thoughts every time, so $K$ runs give $K$ copies. Verifiers that read text ([[#4. Hybrid Scaling — Think Selectively|§4]]) have no text to read. The latent reward model of [[#5.2 Latent thoughts in an LLM|§5.2]] is one attempt to bring verification back. The professor's notes add a second cost, in [[#8.1 Explicit vs latent chain of thought|§8.1]]: latent chains are not interpretable.

---

## 6. Reasoning in VLMs

The rest of the lecture applies these ideas to vision-language models. This answers the question [[ATML_Lecture_11 - Vision-Language Models#10.8 How to reason on images?|Lecture 11 §10.8]] left open.

### 6.1 Reasoning as a distribution over chains

A **reasoning VLM** takes a multimodal input $x = x_{\text{text}} \cup x_{\text{img}}$ and produces a chain of thought $(r_1, \dots, r_K)$ followed by a final answer $a$:

$$P_\theta\big((r_1, \dots, r_K), a \mid x\big) = \prod_t P_\theta(y_t \mid x, y_{<t})$$

where $y$ is the full output token sequence, CoT then answer, and $K$ here counts reasoning steps (not samples as in [[#3. Parallel Scaling — Think Wider|§3]]). This is the factorization of [[#2.1 The reasoning trace as intermediate context|§2.1]], with an image in the condition.

### 6.2 Text-only vs multimodal reasoning

For a **text-only** LLM, the input and output share a **single homogeneous token space**, and the model only needs $P_\theta(y \mid x)$. The professor's example is $x$ = "What is the area of a right triangle with legs 3 and 4?", for which

$$y = \tfrac12 \times 3 \times 4 = 6$$

which is **trivial**.

**However, when we have multiple modalities, the same question becomes hard.** Ask "What is the area of the triangle in the figure?" and attach the drawing, and the model has to:
1. **perceive the image and locate the triangle**;
2. **ground the visual shape to geometry** (recognize it as a right triangle with two legs);
3. **estimate metric lengths**, which is **structurally impossible for a text-only LLM**;
4. **extract numerical values**, then reason;
5. **compose these steps in a coherent chain**.

![[03 - Attachments/ATML_L12_text_vs_vlm.svg|560]]

*(a) Text-only: the numbers are in the prompt. (b) Multimodal: the numbers are in the image, across a semantic gap from the text.*

Your notes list the same steps (perceive and locate, ground, extract numerics, compose reasoning between text and image) and name the core problem: **grounding the text and its tokens to its image.** Both $x_T$ and $x_V$ become tokens for the LLM ([[ATML_Lecture_11 - Vision-Language Models#9.1 The pipeline and its two jobs|Lecture 11 §9.1]]), but nothing guarantees that a word token is linked to the right image tokens. The professor marks this as the **semantic gap**. The five steps line up with the failure types of [[ATML_Lecture_11 - Vision-Language Models#8.1 Five failure types|Lecture 11 §8.1]]: steps 1 and 3 are perception, step 2 grounding, and step 5 composition and reasoning.

### 6.3 Four challenges unique to VLMs

A VLM maps an image and a text to a **reasoning response** $R$:

$$\text{VLM}: X_{\text{img}} \times X_{\text{text}} \to R$$

Four challenges arise that a text-only reasoner does not face.

1. **Cross-modal grounding.** The model must **resolve co-references** across modalities. In "What is its color?", "it" refers to some object in the image, and the model must find which one.
2. **Conflicts.** The modalities can disagree: the image shows a smiling person while the text says "…sad…". The model must decide which source to trust.
3. **Hallucination amplification.** The model **generates plausible but visually unsupported text**. Usually the **language prior overwrites the visual evidence**:
$$P_\theta(a \mid x_{\text{text}}) \gg P_\theta(a \mid x_{\text{img}})$$
The answer is driven far more by what the text makes likely than by what the image shows. *(Expanding on "amplification":)* in a long chain, a hallucinated step becomes context for every later step ([[#2.2 Where a single chain fails|§2.2]]), so reasoning can make a small hallucination worse rather than correct it.
4. **Multi-hop reasoning.** Visual questions require **chained reasoning steps that alternate between modalities**, forming a **directed acyclic reasoning graph**: steps are nodes, an edge means one step uses another's result, and no step depends on itself.

![[03 - Attachments/ATML_L12_vlm_challenges.svg|560]]

*(a) A conflict between image and text. (b) A multi-hop chain whose steps alternate between reading the image and reasoning.*

In the professor's chain, $r_1 \xrightarrow{\text{extract}} r_2 \xrightarrow{\text{compare}} r_3 \xrightarrow{\text{look up}} a$: extract a value from the image, compare it with something, then look up a fact to answer.

---

## 7. Algorithms for Multimodal Reasoning

The professor lists five families of algorithms. The first two change the parameters ("how to scale: SFT or RL" in [[#1.2 Four design questions|§1.2]]). The next two spend compute at inference, through search and tools. The fifth moves reasoning into the latent space.

### 7.1 SFT on multimodal chains of thought

**SFT + MCoT (multimodal chain of thought):** train the VLM on **human-annotated reasoning chains** with the supervised loss

$$\mathcal L_{\text{SFT}} = -\log P_\theta(r, a \mid x) = -\sum_{t=1}^{T}\log P_\theta(y_t \mid x, y_{<t})$$

where $y = (r, a)$ has $T$ tokens. This is the completion-only loss of [[ATML_Lecture_8 - LLM Alignment#2.2 The completion-only loss|Lecture 8 §2.2]] with an image in $x$, applied to reasoning data as in [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#2.1 Same training stack, new data|Lecture 9 §2.1]]. Its cost is the annotation: someone has to write the chains.

### 7.2 RLVR with GRPO

**RLVR** ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.3 RLVR: a verifier in place of the learned reward model|Lecture 9 §1.3]]) removes the need for written chains. Its ingredients are the policy $\pi_\theta$, a **rule-based reward** in $\{0, 1\}$, **GRPO**, and a reference policy $\pi_{\text{ref}}$. The reward comes from **verifiers** suited to visual tasks:
1. **geometry solvers**;
2. **math checkers**;
3. **structured output parsers** (checking that the output has the required format, e.g. valid JSON or a bounding box).

![[03 - Attachments/ATML_L12_vlm_rlvr.svg|640]]

*One GRPO step for a VLM: $G$ roll-outs, verifier rewards, group-relative advantage, KL to $\pi_{\text{ref}}$.*

The query $q$ is a text plus visual embedding. $\pi_\theta$ produces $G$ roll-outs $o_1, \dots, o_G$, each is rewarded by the verifiers, and each roll-out's **relative advantage** is its reward minus the group's mean ([[ATML_Lecture_8 - LLM Alignment#6.1 What changes: the advantage|Lecture 8 §6.1]]). GRPO then updates $\pi_\theta$ with those advantages ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.4 RLVR inside PPO and GRPO|Lecture 9 §1.4]]). The $G$ roll-outs are the parallel samples of [[#3.1 Sampling K trajectories|§3.1]], used here for training rather than for answering.

### 7.3 Tree search and MCTS

**Tree search / MCTS (Monte Carlo Tree Search)** organizes the search of [[#4.4 From verification to search|§4.4]] as a tree:
- each **reasoning step is a node**, and its children are candidate next steps;
- a **process reward model (PRM)** is used **to expand and prune the tree**: promising nodes get children, weak ones are cut;
- the search **backtracks when a reasoning path becomes unpromising**, returning to an earlier node to try a different branch;
- some nodes are **visual subgoals** (shaded in the figure), steps that read or locate something in the image.

![[03 - Attachments/ATML_L12_mcts.svg|500]]

*A reasoning tree: the PRM decides which nodes to expand, and the search can back up out of a failing branch.*

The **value of a path** $\tau$ sums the PRM's scores of its prefixes, discounted by depth:

$$V(\tau) = \sum_{k} \gamma^{k}\,R_{\text{PRM}}(r_1, \dots, r_k)$$

where $R_{\text{PRM}}(r_1, \dots, r_k)$ is the PRM's score of the first $k$ steps and $\gamma \in (0, 1]$ is a **discount factor**. *(Expanding on $\gamma$:)* with $\gamma < 1$, later steps count for less, which favours paths that are good early.

*(Expanding on MCTS:)* each iteration of MCTS **selects** a path from the root by balancing high value against rarely-visited branches, **expands** a leaf with new candidate steps, **evaluates** the new node (here with the PRM), and **backs up** the value to every node on the path. Compared with the beam search of [[#4.4 From verification to search|§4.4]], the tree keeps every explored branch, so it can return to an alternative that beam search would already have dropped.

### 7.4 RAG and tool use

Some steps of [[#6.2 Text-only vs multimodal reasoning|§6.2]], such as reading text in an image or measuring a length, are hard for the VLM itself. **RAG + tool usage** hands them to other systems. The VLM policy $\pi_\theta^{\text{VLM}}$ exchanges information with:
- **external tools:** an **OCR** (optical character recognition) **engine** that reads text in images, a **Python interpreter**, **knowledge graphs**, **depth estimators**;
- **RAG** (retrieval-augmented generation), which retrieves documents and returns them as context.

![[03 - Attachments/ATML_L12_rag_tools.svg|460]]

*The VLM queries tools and a retriever, and their outputs become part of its input.*

The answer is conditioned on the query and on everything retrieved and returned:

$$a = \pi_\theta\big(x, \text{Retrieve}(x), \text{Tool}_1(x), \dots, \text{Tool}_m(x)\big)$$

Tool calls are one of the forms of $C_{\text{inf}}$ in [[#1.3 What counts as inference compute|§1.3]]. A depth estimator directly addresses the "estimate metric lengths" step that a VLM finds hard. This is the ReAct pattern of [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques|Lecture 9 §6.2]], reasoning interleaved with tool calls.

### 7.5 Latent reasoning

The fifth algorithm is **latent reasoning: reason in a unified latent space**, where image and text are both hidden vectors rather than one being turned into words. It is the VLM version of [[#5. Latent Scaling — Think Implicitly|§5]], and the professor develops it in the next section.

---

## 8. Latent Reasoning in VLMs

### 8.1 Explicit vs latent chain of thought

![[03 - Attachments/ATML_L12_explicit_vs_latent.svg|640]]

*(a) Explicit CoT: generated words are fed back as tokens. (b) Latent CoT: hidden states are fed back instead.*

In **explicit CoT**, the LLM takes text and image tokens $x$, opens with ⟨think⟩, generates words that are fed back one at a time, and ends with the answer ⟨a⟩. The chain is **human-readable**. In **latent CoT**, the loop runs through hidden states that are never decoded, and only the answer ⟨a⟩ is emitted. The chain is **non-interpretable**: we can no longer read or check the reasoning.

The professor sets up the notation for doing latent reasoning. The input $x$ is a $T \times d$ matrix whose rows are the token embeddings: ⟨boi⟩, the image tokens, ⟨eoi⟩, then the text tokens up to ⟨eos⟩ (the image markers of [[ATML_Lecture_11 - Vision-Language Models#10.4 Image understanding with token IDs|Lecture 11 §10.4]]). The model's blocks map it to the last layer's **hidden state**

$$Z_L = \begin{bmatrix} z_L^{(0)} \\ \vdots \\ z_L^{(T)} \end{bmatrix} \in \mathbb R^{T \times d}$$

which the **LM head** turns into token probabilities. There are two ways to give the model more computation in this space: loop over depth (vertical) or grow the sequence (horizontal).

### 8.2 Layer-wise vertical refinement

**Layer-wise vertical** reasoning **refines the hidden state $Z_L$ over $N$ steps by feeding it as input to the VLM** again:

$$H^{(1)} = Z_L = \text{VLM}(x), \qquad H^{(n+1)} = \text{VLM}\big(H^{(n)}\big), \qquad y = \text{LMHead}\big(H^{(N)}\big)$$

Every $H^{(n)}$ is $T \times d$, so the sequence length never changes.

![[03 - Attachments/ATML_L12_latent_vertical.svg|640]]

*Vertical: the same VLM is applied $N$ times to its own output before the LM head reads it.*

*(Expanding on what this buys:)* running the same $L$ layers $N$ times makes the network effectively $NL$ layers deep with no new parameters, so the extra compute goes into depth. This idea is studied as looped or recurrent-depth Transformers.

### 8.3 Token-wise horizontal latent thoughts

**Token-wise horizontal** reasoning instead **generates intermediate latent thoughts along the sequence dimension**. The VLM maps $x$ to $Z_L$, a **latent thought** $h_i$ is formed from it, and $h_i$ is appended to the input for the next round, for $i = 1, 2, \dots, N$:

![[03 - Attachments/ATML_L12_latent_horizontal.svg|560]]

*Horizontal: each round appends $K$ latent vectors, so the input grows.*

Each $h_i$ has $K$ rows, $h_i \in \mathbb R^{K \times d}$, where $K$ is the number of **additional latent thought vectors** per round. The **sequence length grows over $i$**: $T \to T + K \to T + 2K \to \cdots$. Each row $h \in \mathbb R^{d \times 1}$ is a **latent thought vector**: a **dense, continuous vector that encodes the visual features relevant to the text query**, and that **should represent successive reasoning steps**.

The two directions spend compute differently:

| | Vertical ([[#8.2 Layer-wise vertical refinement\|§8.2]]) | Horizontal ([[#8.3 Token-wise horizontal latent thoughts\|§8.3]]) |
|---|---|---|
| What grows | depth: $N$ passes through the VLM | length: $T + NK$ positions |
| Sequence length | fixed at $T$ | grows by $K$ per step |
| Counterpart in token space | none (one forward pass per token) | sequential scaling, [[#2. Sequential Scaling — Think Longer\|§2]] |

Horizontal reasoning is the VLM form of the latent thoughts in [[#5.2 Latent thoughts in an LLM|§5.2]], with $K$ vectors per step instead of one.

### 8.4 How to get the latent vector

The last question is what function produces $h_i$. The professor lists five options:

1. **Some function of $Z_L$:** $h_i = f(Z_L)$, the general form.
2. **Some function of $I_V$, the input vision embeddings:** $h_i = f(I_V)$. This uses the raw visual features before the language model has processed them.
3. **Cross-attention:** $h_i = \text{CrossAttn}\big(Q = Z_L^{\text{text}}, K = Z_L^{\text{img}}, V = Z_L^{\text{img}}\big)$, with queries from the hidden states at text positions and keys and values from those at image positions. The result is **text-conditioned image content**: the parts of the image the question asks about. This is the mechanism of LLIP and the Q-Former in Lecture 11 ([[ATML_Lecture_11 - Vision-Language Models#4.3 LLIP, caption-conditioned image embeddings|§4.3]], [[ATML_Lecture_11 - Vision-Language Models#7.3 Bridge 2, bottleneck resampler (Q-Former)|§7.3]]).
4. **A chain of visual thoughts:** the hidden states at image positions go through an **MoE** (mixture of experts, a set of specialist sub-networks with a router choosing among them), whose experts produce different kinds of visual information: **2D appearance, 3D geometry, edge structure, spatial layout**.
5. **Reconstruction:** the hidden states at text positions are used to **reconstruct image ROIs** (regions of interest), and the result is used to **generate the latent thought**. *(Expanding on why:)* asking the thought to rebuild the relevant image region forces it to carry real visual detail, which works against the language-prior hallucination of [[#6.3 Four challenges unique to VLMs|§6.3]].

> [!note] Unclear in source
> The professor's page numbers these options 1, 2, 2, 4, 5. The second "2" (cross-attention) is option 3.

![[03 - Attachments/ATML_L12_latent_vector.svg|600]]

*(a) Option 3, cross-attention from text to image states. (b) Option 4, MoE experts for different visual features. (c) Option 5, thoughts formed by reconstructing image regions.*

Options 3–5 differ in what makes the latent thought visual: attention to image states, experts for visual properties, or a reconstruction target. All of them inherit the limits of [[#5.3 What latent reasoning gives up|§5.3]]: the thoughts cannot be read, and without sampling, the parallel and verification methods of [[#3. Parallel Scaling — Think Wider|§3]]–[[#4. Hybrid Scaling — Think Selectively|§4]] do not apply directly.

---

## Numerical Problems

**Problem 1 — inference cost of thinking longer.** A forward pass costs about $2N$ FLOPs per generated token. Compare a 7B model that writes a 4,000-token reasoning trace with a 70B model that answers directly in 500 tokens.

The 7B model costs $2 \times 7\times10^9 \times 4000 = 5.6\times10^{13}$ FLOPs. The 70B model costs $2 \times 7\times10^{10} \times 500 = 7.0\times10^{13}$ FLOPs.

$$\boxed{5.6\times10^{13} \text{ vs } 7.0\times10^{13} \text{ FLOPs}}$$

The small model thinking eight times longer still uses 20% less compute per query. Whether it is also more accurate is the empirical question of [[#1.1 From training-time scaling to inference compute|§1.1]].

**Problem 2 — Plan-and-Solve.** For a query, $P_\theta(P \mid x) = 0.8$, $P_\theta(r \mid x, P) = 0.5$ and $P_\theta(y \mid x, P, r) = 0.9$. What is the joint probability of this plan, trace and answer?

By the chain rule of [[#2.3 Plan-and-Solve|§2.3]], $P_\theta(P, r, y \mid x) = 0.8 \times 0.5 \times 0.9$:

$$\boxed{0.36}$$

**Problem 3 — the answer marginal.** A model can produce three traces with $P_\theta(r_1 \mid x) = 0.5$, $P_\theta(r_2 \mid x) = 0.3$ and $P_\theta(r_3 \mid x) = 0.2$. Traces 1 and 3 end in answer A, trace 2 in answer B (each with probability 1). What is $P_\theta(A \mid x)$?

$$P_\theta(A \mid x) = \sum_r P_\theta(r \mid x)P_\theta(A \mid x, r) = 0.5 \times 1 + 0.3 \times 0 + 0.2 \times 1 = \boxed{0.7}$$

A single sampled trace gives B 30% of the time, even though A holds 70% of the probability. This is why voting over samples ([[#3.2 Self-consistency (majority voting)|§3.2]]) helps.

**Problem 4 — self-consistency.** Seven samples for "15% of 240" return $\{36, 32, 36, 24, 36, 32, 36\}$. Compute $\hat P_7$ and $\hat y_{sc}$.

Counting: 36 appears 4 times, 32 twice, 24 once. So $\hat P_7(36) = 4/7 = 0.571$, $\hat P_7(32) = 2/7 = 0.286$, $\hat P_7(24) = 1/7 = 0.143$. By [[#^eq-1|eq. (1)]]:

$$\boxed{\hat y_{sc} = 36, \quad \hat P_7(36) = 0.571}$$

**Problem 5 — coverage vs selection.** Each sample is correct with probability $p = 0.2$. (a) What is the coverage with $K = 10$? (b) If the verifier picks the correct answer 60% of the time when it is present, what is the final accuracy? (c) Compare with $K = 1$.

(a) $\text{coverage}(10) = 1 - 0.8^{10} = 1 - 0.1074 = 0.893$.

(b) Accuracy $\approx 0.893 \times 0.6 = 0.536$.

(c) With $K = 1$, there is nothing to select, so accuracy $= p = 0.2$.

$$\boxed{\text{coverage } 0.893, \quad \text{accuracy } 0.536 \text{ vs } 0.2}$$

As $K \to \infty$, coverage approaches 1 but accuracy approaches only 0.6: the verifier sets the ceiling ([[#4.1 Solution coverage vs selection accuracy|§4.1]]).

**Problem 6 — process verification.** Trajectory A has step scores $(0.9, 0.8, 0.95, 0.5)$; trajectory B has $(0.7, 0.7, 0.7, 0.7)$. Score both with the product rule, compare with a plain sum, and locate A's most likely first error.

Product: $V_A = 0.9 \times 0.8 \times 0.95 \times 0.5 = 0.342$; $V_B = 0.7^4 = 0.240$. Sum: $\Sigma_A = 3.15$, $\Sigma_B = 2.80$.

$$\boxed{V_A = 0.342 > V_B = 0.240}$$

Both rules prefer A here, but A's step 4 scores only 0.5, which marks it as the most likely first incorrect step. A minimum rule would prefer B ($0.7 > 0.5$). Which rule is right depends on whether one weak step should sink a trajectory ([[#4.3 Process verification|§4.3]]).

**Problem 7 — the cost of beam search.** Beam search with $K = 3$ beams and $J = 4$ continuations per beam runs for 5 steps. How many step generations and verifier calls does it make, and how many candidates are discarded per step?

Each step expands $K$ states into $KJ = 12$ candidates, all of which are generated and scored, and keeps 3. Over 5 steps:

$$\boxed{60 \text{ step generations}, \ 60 \text{ verifier calls}, \ 9 \text{ discarded per step}}$$

(This assumes the search starts from 3 states; starting from one state, the first step makes only $J = 4$ candidates.)

**Problem 8 — the value of a reasoning path.** A path's PRM prefix scores are $R_{\text{PRM}}(r_1) = 0.8$, $R_{\text{PRM}}(r_1, r_2) = 0.6$ and $R_{\text{PRM}}(r_1, r_2, r_3) = 0.9$, with $\gamma = 0.9$ and the sum starting at $k = 1$. Compute $V(\tau)$.

$$V(\tau) = 0.9 \times 0.8 + 0.9^2 \times 0.6 + 0.9^3 \times 0.9 = 0.72 + 0.486 + 0.656 = \boxed{1.862}$$

**Problem 9 — GRPO advantages for a VLM.** For one image question, $G = 4$ roll-outs get verifier rewards $\{1, 0, 0, 1\}$. Compute the relative advantages (mean-subtracted, as in [[ATML_Lecture_8 - LLM Alignment#6.1 What changes: the advantage|Lecture 8 §6.1]]). What happens if all four rewards are 1?

The mean is 0.5, so $A = r - 0.5$:

$$\boxed{A = (+0.5, -0.5, -0.5, +0.5)}$$

If all rewards are 1, the mean is 1 and every advantage is 0, so the question gives no gradient ([[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.5 Properties of RLVR|Lecture 9 §1.5]]).

**Problem 10 — what a token throws away.** A model has vocabulary size $|V| = 300{,}000$ and hidden size $d = 4096$ stored in 16-bit floats. Compare the information in one sampled token with the size of the hidden state it came from, and the LM-head cost per token ($2d|V|$ FLOPs).

A token carries at most $\log_2 300{,}000 = 18.2$ bits. The hidden state takes $4096 \times 16 = 65{,}536$ bits. The LM head costs $2 \times 4096 \times 300{,}000 = 2.46\times10^9$ FLOPs per token.

$$\boxed{18.2 \text{ bits vs } 65{,}536 \text{ bits}; \quad 2.46\times10^9 \text{ FLOPs per token}}$$

The hidden state isn't 65,536 bits of useful information, but the gap shows how much decoding can discard. That is the motivation for latent reasoning in [[#5.1 The cost of explicit reasoning tokens|§5.1]].

**Problem 11 — horizontal latent thoughts.** A VLM input has $T = 600$ tokens (576 image, 24 text). Horizontal reasoning appends $K = 8$ latent vectors per step for $N = 4$ steps. What is the final sequence length, and how many positions does vertical refinement with $N = 4$ use?

Horizontal: $T + NK = 600 + 4 \times 8 = 632$. Vertical keeps $T = 600$ but runs the VLM 4 times.

$$\boxed{632 \text{ positions (horizontal)}; \ 600 \text{ positions} \times 4 \text{ passes (vertical)}}$$

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Training scaling law | $L(N, D) \propto L_0 + a/N^{\alpha} + b/D^{\beta}$, $C_{\text{train}} \propto ND$ | [[#1.1 From training-time scaling to inference compute\|§1.1]], [[ATML_Lecture_7 - LLM Foundations#5.3 Chinchilla laws\|L7 §5.3]] |
| Compute split | $C = C_{\text{train}} + C_{\text{inf}}$ | [[#1.1 From training-time scaling to inference compute\|§1.1]] |
| TTS objective | $\text{Accuracy}_{\text{task}}(N, D, C_{\text{inf}})$ | [[#1.1 From training-time scaling to inference compute\|§1.1]] |
| Trace probability | $P_\theta(r \mid x) = \prod_{t=1}^{T_r}P_\theta(r_t \mid x, r_{<t})$ | [[#2.1 The reasoning trace as intermediate context\|§2.1]] |
| Answer marginal | $P_\theta(y \mid x) = \sum_r P_\theta(r \mid x)P_\theta(y \mid x, r)$ | [[#2.1 The reasoning trace as intermediate context\|§2.1]], [[#3.2 Self-consistency (majority voting)\|§3.2]] |
| Plan-and-Solve | $P_\theta(P, r, y \mid x) = P_\theta(P \mid x)P_\theta(r \mid x, P)P_\theta(y \mid x, P, r)$ | [[#2.3 Plan-and-Solve\|§2.3]] |
| Parallel samples | $\{(r^{(k)}, y^{(k)})\}_{k=1}^K \sim P_\theta(r, y \mid x)$ | [[#3.1 Sampling K trajectories\|§3.1]] |
| Vote share | $\hat P_K(y \mid x) = \frac1K\sum_k\mathbb 1(y^{(k)} = y)$ | [[#3.2 Self-consistency (majority voting)\|§3.2]] |
| Self-consistency ([[#^eq-1\|eq. (1)]]) | $\hat y_{sc} = \arg\max_y\hat P_K(y \mid x)$ | [[#3.2 Self-consistency (majority voting)\|§3.2]], [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#6.2 Prompting techniques\|L9 §6.2]] |
| Best-of-N ([[#^eq-2\|eq. (2)]]) | $y_{\text{BoN}} = y^{(k^*)}$, $k^* = \arg\max_k V_\phi(x, r^{(k)}, y^{(k)})$ | [[#3.3 Best-of-N with a verifier\|§3.3]], [[#4.2 Outcome verification\|§4.2]] |
| Self-aggregation | $\hat y_{SA} = P_\theta(y \mid x, \{r^{(i)}, y^{(i)}\}_{i=1}^K)$ | [[#3.4 Self-aggregation\|§3.4]] |
| Coverage (pass@$K$) | $1 - (1 - p)^K$ | [[#4.1 Solution coverage vs selection accuracy\|§4.1]] |
| Process verifier | $V_s = V_\phi(x, r_{\le s})$; $V_{\text{process}} = \prod_s V_s$ | [[#4.3 Process verification\|§4.3]], [[ATML_Lecture_8 - LLM Alignment#3.5 Sparse rewards, ORMs and PRMs\|L8 §3.5]] |
| Search state | $S_t = \{x, r_1, \dots, r_t\}$; $r^{(j)}_{t+1} \sim P_\theta(r_{t+1} \mid S_t)$; $S^{(j)}_{t+1} = (S_t, r^{(j)}_{t+1})$ | [[#4.4 From verification to search\|§4.4]] |
| Beam step | $V_\phi(S^{(j)}_{t+1}) \to \text{top-}K \to B_{t+1}$; $KJ$ candidates per step | [[#4.4 From verification to search\|§4.4]], [[ATML_Lecture_7 - LLM Foundations#6.4 Beam search\|L7 §6.4]] |
| Self-refinement | $f^{(t)} \sim P_\theta(f \mid x, r^{(t)})$, $r^{(t+1)} \sim P_\theta(r \mid x, r^{(t)}, f^{(t)})$ | [[#4.5 Self-refinement\|§4.5]] |
| Latent thought (LLM) | $h_L^{(T+1)} = f(x, h_\ell^{(1:T)})$; layer average $\frac1L\sum_\ell h_\ell^{(T+1)}$ | [[#5.2 Latent thoughts in an LLM\|§5.2]] |
| VLM reasoning | $P_\theta((r_1, \dots, r_K), a \mid x) = \prod_t P_\theta(y_t \mid x, y_{<t})$, $x = x_{\text{text}} \cup x_{\text{img}}$ | [[#6.1 Reasoning as a distribution over chains\|§6.1]] |
| Language-prior dominance | $P_\theta(a \mid x_{\text{text}}) \gg P_\theta(a \mid x_{\text{img}})$ | [[#6.3 Four challenges unique to VLMs\|§6.3]] |
| SFT on MCoT | $\mathcal L_{\text{SFT}} = -\sum_t\log P_\theta(y_t \mid x, y_{<t})$ | [[#7.1 SFT on multimodal chains of thought\|§7.1]], [[ATML_Lecture_8 - LLM Alignment#2.2 The completion-only loss\|L8 §2.2]] |
| Path value (MCTS) | $V(\tau) = \sum_k\gamma^kR_{\text{PRM}}(r_1, \dots, r_k)$ | [[#7.3 Tree search and MCTS\|§7.3]] |
| Tool-augmented answer | $a = \pi_\theta(x, \text{Retrieve}(x), \text{Tool}_1(x), \dots, \text{Tool}_m(x))$ | [[#7.4 RAG and tool use\|§7.4]] |
| Vertical refinement | $H^{(n+1)} = \text{VLM}(H^{(n)})$, $H^{(1)} = Z_L$, $y = \text{LMHead}(H^{(N)})$ | [[#8.2 Layer-wise vertical refinement\|§8.2]] |
| Horizontal thoughts | $h_i \in \mathbb R^{K \times d}$; length $T \to T + K \to T + 2K$ | [[#8.3 Token-wise horizontal latent thoughts\|§8.3]] |
| Cross-attention thought | $h_i = \text{CrossAttn}(Q = Z_L^{\text{text}}, K, V = Z_L^{\text{img}})$ | [[#8.4 How to get the latent vector\|§8.4]], [[ATML_Lecture_11 - Vision-Language Models#4.3 LLIP, caption-conditioned image embeddings\|L11 §4.3]] |

---

## Summary

- **Why test-time scaling ([[#1. Why Scale Compute at Test Time|§1]])**
  - Training scaling laws pay everything up front; TTS adds inference compute $C_{\text{inf}}$ to a fixed model, so $C = C_{\text{train}} + C_{\text{inf}}$.
  - More $C_{\text{inf}}$ can sometimes outperform a larger $N$, mainly on problems the model can already solve some of the time.
  - Four design questions: what to scale, how (SFT/RL or search), where (reasoning or evaluation), how well (accuracy, efficiency, controllability, scaling).
  - $C_{\text{inf}}$ is measured in tokens, forward passes, samples, verifier calls, latency, FLOPs and tool calls.
  - Four families: sequential (longer), parallel (wider), hybrid (selective), latent (implicit).

- **Sequential scaling ([[#2. Sequential Scaling — Think Longer|§2]])**
  - A reasoning trace before the answer gives the model intermediate context; it is induced by prompting or SFT.
  - The answer's probability is a sum over traces, and one trace is one sample from that sum.
  - A single chain can fail by poor decomposition, a missed step or a wrong order, and one wrong step propagates.
  - Plan-and-Solve plans first, but a wrong plan contaminates the whole trajectory.

- **Parallel scaling ([[#3. Parallel Scaling — Think Wider|§3]])**
  - Sample $K$ diverse trajectories (seeds, decoding settings, temperature) instead of one long one.
  - Self-consistency votes for the most frequent answer, estimating the answer marginal; it can be systematically wrong and ignores the traces.
  - Best-of-N picks the sample with the highest verifier score; it can pick a correct minority answer but is vulnerable to verifier errors.
  - Self-aggregation lets the LLM read all samples and answer; it costs more, can add errors and tends to collapse to the majority.

- **Hybrid scaling ([[#4. Hybrid Scaling — Think Selectively|§4]])**
  - Coverage (is $y^*$ among the samples) grows with $K$; selection accuracy (can we find it) does not, and caps the gain.
  - Outcome verification scores whole answers; process verification scores each prefix $V_\phi(x, r_{\le s})$ and finds the first bad step.
  - Step scores combine by product (or sum of logs); a plain sum can hide one broken step.
  - Scoring partial states turns verification into search: $K$ beams × $J$ continuations, keep the best $K$.
  - Self-refinement alternates critique $f^{(t)}$ and revision $r^{(t+1)}$ with the same model, and helps only if the model can spot its own error.

- **Latent scaling ([[#5. Latent Scaling — Think Implicitly|§5]])**
  - Explicit traces decode every token through a ~300k softmax and collapse a $d$-dimensional state into one ID.
  - Latent reasoning feeds hidden vectors back instead, e.g. the layer average or candidates scored by a latent reward model.
  - Without sampled tokens there is no decoding stochasticity, so parallel and text-verification TTS methods no longer apply.

- **Reasoning in VLMs ([[#6. Reasoning in VLMs|§6]])**
  - A reasoning VLM generates a chain and an answer from $x_{\text{text}} \cup x_{\text{img}}$.
  - A text-only problem is trivial when the numbers are in the prompt; with an image, the model must perceive, ground, measure, extract and compose.
  - The core problem is grounding text tokens to the image, across a semantic gap.
  - Four challenges: cross-modal grounding (co-references), conflicts between modalities, hallucination amplification ($P_\theta(a \mid x_{\text{text}}) \gg P_\theta(a \mid x_{\text{img}})$), multi-hop reasoning as a directed acyclic graph.

- **Algorithms ([[#7. Algorithms for Multimodal Reasoning|§7]])**
  - SFT on human-annotated multimodal chains, with the completion loss.
  - RLVR: rule-based 0/1 rewards from geometry solvers, math checkers and output parsers, optimized by GRPO with a reference policy.
  - Tree search / MCTS: a PRM expands and prunes a tree of steps, with backtracking and visual subgoals; path value $\sum_k\gamma^kR_{\text{PRM}}$.
  - RAG and tools (OCR, Python, knowledge graphs, depth estimators) supply what the VLM cannot do itself.
  - Latent reasoning reasons in a unified latent space.

- **Latent reasoning in VLMs ([[#8. Latent Reasoning in VLMs|§8]])**
  - Explicit CoT is human-readable; latent CoT is not.
  - Vertical: feed $Z_L$ back through the VLM $N$ times at fixed length $T$, then read it with the LM head.
  - Horizontal: append $K$ latent thought vectors per step, so the length grows $T \to T + K \to \cdots$; each vector should hold query-relevant visual features and successive reasoning steps.
  - The latent vector can come from $Z_L$, from the vision embeddings, from text-to-image cross-attention, from MoE visual experts, or from reconstructing image ROIs.

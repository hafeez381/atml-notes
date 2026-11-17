---
type: lecture
course: '[[_CS6304 ATML|CS6304 ATML]]'
source:
- '[[LLMs I.pdf]]'
- '[[ATML Lec 9.pdf]]'
- '[[ATML Lec 10.pdf]]'
tags:
- atml
- course/cs6304
- llm
- transformer
- scaling-laws
- decoding
title: ATML — Lecture 7 — LLM Foundations (Decoder-Only Transformer, Training Loss, Scaling Laws, Decoding)
lecture: 7
aliases:
- LLM Foundations
- ATML Lecture 7
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_6 - Open-Set Recognition]] · [[ATML_Lecture_8 - LLM Alignment]] →
Related: [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.7 The Transformer Layer|L3 Transformer layer]], [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention|CS5326 L6 — Decoding & KV cache]], [[RA1_Chinchilla_Cheatsheet|CS5326 RA1 — Chinchilla]]
%% nav:end %%

# ATML — Lecture 7: LLM Foundations

## Where This Lecture Sits

*Lecture 3 built the Transformer layer as an encoder for images (ViT) and contrastive models. This lecture turns the same layer into a **language model**: a decoder that predicts the next token. Lecture 8 then takes the trained model and aligns it with what people want.*

The material comes from the professor's notes "LLMs I" (pages 1–8) and your ATML Lectures 9 and 10. The professor writes token vectors as **rows** ($H \in \mathbb R^{T\times d}$); Lecture 3 wrote them as columns ($X \in \mathbb R^{D\times N}$). The two are transposes of each other, so $\text{softmax}(K^\top Q)$ in Lecture 3 becomes $\text{softmax}(QK^\top)$ here.

## Topic Inventory

- 1. Language Models and How They Generate
  - 1.1 Three families of Transformer language models
  - 1.2 Autoregressive (decoder-only) models
  - 1.3 Discrete diffusion LLMs
  - 1.4 Notation
- 2. From Text to Vectors
  - 2.1 Tokenization and special tokens
  - 2.2 Embedding lookup
  - 2.3 Positional embedding and the full pipeline
- 3. The Decoder-Only Transformer Layer
  - 3.1 Self-attention as a weighted sum of values
  - 3.2 Where the weights come from: queries and keys
  - 3.3 Multi-head causal self-attention
  - 3.4 Output projection, residuals, LayerNorm and the MLP
  - 3.5 The KV cache
  - 3.6 Final LayerNorm and unembedding
- 4. The Training Loss
  - 4.1 Cross-entropy at one position
  - 4.2 Worked example and teacher forcing
  - 4.3 Batching, padding and the loss mask
  - 4.4 Gradient of the loss with respect to the logits
  - 4.5 The softmax bottleneck
- 5. Scaling Laws
  - 5.1 Power laws
  - 5.2 Kaplan laws
  - 5.3 Chinchilla laws
  - 5.4 A roadmap, and scaling laws beyond pre-training
- 6. Decoding
  - 6.1 The decoding problem
  - 6.2 Greedy decoding
  - 6.3 Sampling with temperature
  - 6.4 Beam search
  - 6.5 Top-$k$ sampling
  - 6.6 Top-$p$ (nucleus) sampling
  - 6.7 What modern LLMs use
- Numerical Problems
- Formula Summary Sheet
- Cheat Sheet & Recall Aids

---

## 1. Language Models and How They Generate

### 1.1 Three families of Transformer language models

Transformer-based language models come in three types:

| Type | What it does | Example use |
|---|---|---|
| **Encoder-only** | Reads the whole text at once (bidirectional attention) and learns a representation of it | classification, retrieval, embeddings (e.g. BERT) |
| **Decoder-only** | Predicts the next token from all previous tokens (causal attention) | text generation, chat (e.g. GPT) |
| **Encoder–decoder** | Encodes a source sequence, then decodes a target sequence that attends to it | sequence-to-sequence tasks such as machine translation (e.g. T5) |

The course focuses on **decoder-only** models, the design behind GPT and most current LLMs.

### 1.2 Autoregressive (decoder-only) models

A decoder model is **autoregressive (AR)**: it predicts the conditional distribution of each token given all previous tokens. By the chain rule of probability, the probability of a whole sequence factorizes exactly into these conditionals:

$$\boxed{P_\theta(x_{1:T}) = \prod_{t=1}^{T} P_\theta(x_t \mid x_{<t})}$$

Each factor is the probability of generating token $x_t$ once the model has seen every earlier token. The professor writes the same thing shifted by one index, $P_\theta(x_{1:T}) = \prod_{t=1}^{T-1}P_\theta(x_{t+1}\mid x_{\le t})$: the model at position $t$ predicts token $x_{t+1}$. Both forms are used below.

Training is next-token prediction (NTP) with **teacher forcing** (§4.2): every position sees the true prefix $x_{\le t}$ and predicts $x_{t+1}$, so all $T$ positions are trained in one parallel pass.

![[03 - Attachments/ATML_L7_ar_llm.png|560]]

*Generation is a loop. Starting from $\langle\text{bos}\rangle$, the model produces one token; that token is fed back in as input (the coloured arrows) to produce the next, and so on.*

**Problem: generation is slow and not parallel.** Training is parallel because the true tokens are known in advance. At generation time token $t+1$ can't be computed until token $t$ exists, so a response of $T'$ tokens needs $T'$ sequential forward passes.

### 1.3 Discrete diffusion LLMs

Your Lecture 9 notes contrast AR models with **discrete diffusion LLMs (dLLMs)**, which attack exactly that bottleneck.

![[03 - Attachments/ATML_L7_dllm.png|480]]

*A **corruption process** replaces random tokens with $\langle\text{MASK}\rangle$ ("The cat sat on the mat" → "⟨MASK⟩ ⟨MASK⟩ sat on ⟨MASK⟩ mat"). The LLM is trained to restore the original sentence.*

*(Expanding on this:)* At generation time a dLLM starts from an all-mask sequence and fills in many positions per step, refining over several steps. It predicts several tokens in parallel, and each prediction can use context on both sides. The cost is that positions filled in the same step are predicted independently of each other, so dLLMs usually need several refinement steps to stay coherent.

### 1.4 Notation

| Symbol | Meaning |
|---|---|
| $V$ | vocabulary size (e.g. $V = 50{,}000$) |
| $T$ | sequence length |
| $B$ | batch size |
| $d$ | embedding (model) dimension |
| $h$ | number of attention heads |
| $d_h = d/h$ | per-head dimension |
| $t \in \{1, \dots, T\}$ | token index; $x_t$ is the $t$-th token |
| $l$, $L$ | layer index, total number of layers |
| $d_{ff}$ | feed-forward (MLP) width, typically $4d$ or $8d$ |

---

## 2. From Text to Vectors

The network works on vectors of real numbers, but its input is raw text. Three steps bridge that gap: tokenize, embed, and add position.

### 2.1 Tokenization and special tokens

The **tokenizer** turns a raw text sequence $X$ into a sequence of tokens $x_{1:T} = \{\langle\text{bos}\rangle, x_1, x_2, \dots, \langle\text{eos}\rangle\}$, using an algorithm such as **BPE** (byte-pair encoding) or **Unigram**.

- Each token $x_t \in \{0, 1, \dots, V-1\}$ is an integer **index** into the vocabulary (the "dictionary"). It is not a value: "token 3" says nothing about size, only which entry.
- For text generation the tokenizer **must be invertible**: generated token ids have to map back to text.
- Your notes add that tokenizers come from **source coding**: they remove redundancy in language. *(Expanding on this:)* BPE starts from characters or bytes and repeatedly merges the most frequent adjacent pair into a new token, so frequent strings become single tokens. This is the same principle as compression: common patterns get short codes.

**Special tokens** mark structure rather than words:

| Token | Role |
|---|---|
| $\langle\text{bos}\rangle$, $\langle\text{eos}\rangle$ | beginning / end of sequence |
| $\langle\text{bor}\rangle$, $\langle\text{eor}\rangle$ | beginning / end of a response (used in SFT, Lecture 8) |
| $\langle\text{PAD}\rangle$ | filler so sequences in a batch have equal length (§4.3) |
| $\langle\text{start thinking}\rangle$ | opens the hidden reasoning trace in reasoning LLMs (Lecture 9) |

### 2.2 Embedding lookup

The **embedding matrix** $W_E \in \mathbb R^{V\times d}$ is a learnable lookup table, initialized with random values: one row per vocabulary entry. Token $x_t$ selects its row:

$$E_t = W_E[x_t] \in \mathbb R^{d}$$

This maps a discrete integer ($x_t \in \mathbb Z$) to a continuous vector ($E_t \in \mathbb R^d$). Stacking the $T$ rows gives $E \in \mathbb R^{T\times d}$: the row index is the token position, the column index is the embedding dimension.

*(Expanding on this:)* A lookup is the same as multiplying a one-hot vector $\mathbf e_{x_t}^\top \in \mathbb R^{1\times V}$ by $W_E$, so the gradient reaches only the rows of tokens that actually appeared in the batch.

### 2.3 Positional embedding and the full pipeline

Attention treats its inputs as a set: permuting the tokens permutes the outputs in the same way and changes nothing else. Word order therefore has to be injected explicitly. A **positional embedding** $P \in \mathbb R^{T\times d}$ is added:

$$\boxed{H^{(0)} = E + P \in \mathbb R^{T\times d}}$$

$P$ is either **learnable** (one trained vector per position) or **RoPE** (rotary positional embedding). RoPE is not actually added to $E$: it rotates the query and key vectors inside attention by angles proportional to position, so the score $\langle q_t, k_u\rangle$ depends only on the offset $t - u$. See [[Lecture_03_From_Sinusoidal_Limitations_to_RoPE|CS5326 Lecture 3]] for the full derivation.

![[03 - Attachments/ATML_L7_pipeline.png|620]]

*The whole decoder-only LLM, from your Lecture 10 notes:*
- *Raw text → tokenizer (invertible) → tokens $x_{1:T}$.*
- *Embedding look-up table $W_E[\cdot]$ → $E$ ($T\times d$).*
- *Positional encoding → $H^{(0)} = E + P$.*
- *$L$ decoder-only Transformer layers → $\bar H$ ($T\times d$).*
- *Unembedding $W_U$ → logits $Z = \bar H W_U$ ($T\times V$).*
- *Row-wise softmax → $P_\theta(x_t \mid x_{<t})$, a distribution over $V$ categories at every position.*
- *Loss: $\ell_t = -\log\hat y_{t}$ at every position, using the ground-truth token, then averaged.*

---

## 3. The Decoder-Only Transformer Layer

The $L$ layers transform $H^{(0)}$ into $\bar H$:

$$H^{(0)} \to \text{Layer }1 \to H^{(1)} \to \cdots \to H^{(l-1)} \to \text{Layer }l \to H^{(l)} \to \cdots \to \text{Layer }L \to H^{(L)} \to \text{LN} \to \bar H$$

Every $H^{(l)}$ is $T\times d$: the layers never change the shape. Your Lecture 9 notes build attention up from scratch before the professor's full equations; the same order is followed here.

### 3.1 Self-attention as a weighted sum of values

Self-attention takes $T$ vectors $x_1, \dots, x_T \in \mathbb R^{d}$ and returns $T$ vectors $z_1, \dots, z_T \in \mathbb R^{d}$. Two steps define each $z_i$.

**(1) Compute a set of values.** Each input is projected to a **value** vector $v_i = W_V x_i$.

**(2) Write each output as a linear combination of the values:**

$$z_i = \sum_{j=1}^T a_{ij}\, v_j$$

Here $i$ is the output index and $j$ the value index. For $T = 3$:

$$z_1 = a_{11}v_1 + a_{12}v_2 + a_{13}v_3, \quad z_2 = a_{21}v_1 + a_{22}v_2 + a_{23}v_3, \quad z_3 = a_{31}v_1 + a_{32}v_2 + a_{33}v_3$$

In matrix form, with column vectors $V = [v_1, v_2, v_3] \in \mathbb R^{d\times T}$ and $A = [a_{ij}] \in \mathbb R^{T\times T}$:

$$Z = [z_1, z_2, z_3] = V A^\top \in \mathbb R^{d\times T}$$

The weight matrix has three properties:
1. It is a **square** $T\times T$ matrix: one weight for every (output, value) pair.
2. Each output's weights **sum to 1**: $\sum_j a_{ij} = 1$, which makes every column of $A^\top$ sum to 1. This is enforced by a **softmax**.
3. Each $a_{ij}$ is a **non-linear** function of the inputs. That non-linearity is what makes attention more than a fixed linear filter: the mixing weights change with the content.

### 3.2 Where the weights come from: queries and keys

How should $z_1$ decide $a_{11}, a_{12}, a_{13}$? Go back to the original inputs and make two further descriptions of each $x_i$:

$$q_i = W_Q x_i \quad\text{(query)}, \qquad k_i = W_K x_i \quad\text{(key)}$$

- The query $q_1$ is the question that output 1 asks: "what should $z_1$ look for in $x_1, x_2, x_3$?"
- The key $k_j$ is what $x_j$ offers in answer.

The match between them is the inner product $\langle q_1, k_j\rangle$, and the softmax over $j$ turns the three matches into the weights $(a_{11}, a_{12}, a_{13})$. Doing this for every query gives $A = \text{softmax}(QK^\top/\sqrt{d_k})$ in row form. That matches $Z = V\,\text{softmax}\big(K^\top Q/\sqrt{d_k} + M\big)$ in your column-form notes. The $\sqrt{d_k}$ keeps the scores at unit scale so the softmax doesn't saturate. Lecture 3 [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.9 Attention as Query–Key–Value Retrieval|§2.9]] gives the retrieval picture in full.

### 3.3 Multi-head causal self-attention

The professor's equations for layer $l$ use a **pre-LN** layout: LayerNorm comes *before* each sub-block. That differs from the post-LN order of Lecture 3 §2.7. First normalize:

$$\tilde H = \text{LN}\big(H^{(l-1)}\big) \in \mathbb R^{T\times d}$$

Then, for each head $i = 1, \dots, h$, project to queries, keys and values:

$$Q_i = \tilde H W_{Q,i}, \quad K_i = \tilde H W_{K,i}, \quad V_i = \tilde H W_{V,i} \;\in \mathbb R^{T\times d_h}, \qquad W_{Q,i}, W_{K,i}, W_{V,i} \in \mathbb R^{d\times d_h}$$

The attention matrix of head $i$ adds a **causal mask** $M$ before the softmax:

$$\boxed{A_i = \text{softmax}\!\Big(\frac{Q_i K_i^\top}{\sqrt{d_h}} + M\Big) \in \mathbb R^{T\times T}, \qquad M_{t,u} = \begin{cases} 0 & u \le t \\ -\infty & u > t\end{cases}}$$

Since $e^{-\infty} = 0$, position $t$ gives zero weight to every future position $u > t$. That is what makes the model a valid autoregressive predictor: it can't see the token it is being trained to predict. The softmax is row-wise, so every row of $A_i$ sums to 1.

Each head's output is $O_i = A_i V_i \in \mathbb R^{T\times d_h}$. The $h$ head outputs are concatenated:

$$O = \text{concat}(O_1, O_2, \dots, O_h) \in \mathbb R^{T\times d}$$

*(Expanding on why several heads:)* each head has its own projections, so different heads can attend for different reasons (syntax, coreference, position), each in a $d_h = d/h$ space. The total cost stays about the same as one head of width $d$.

### 3.4 Output projection, residuals, LayerNorm and the MLP

**Project back** with $W_O \in \mathbb R^{d\times d}$ and add the residual:

$$U = O\,W_O \in \mathbb R^{T\times d}, \qquad \boxed{H^{(l-1)}_{\text{attn}} = H^{(l-1)} + U}$$

**MLP** on the normalized result, with a second residual:

$$\hat H = \text{LN}\big(H^{(l-1)}_{\text{attn}}\big), \qquad \boxed{H^{(l)} = H^{(l-1)}_{\text{attn}} + \text{MLP}(\hat H)}$$

![[03 - Attachments/ATML_L7_residual.png|620]]

*Your Lecture 10 sketch of the **residual connections**:*
- *The highlighted path carries $H^{(l-1)}$ straight through.*
- *MHA adds $U^{(l-1)}$ to give $H_a^{(l-1)}$.*
- *LayerNorm → MLP adds the second update to give $H^{(l)}$.*
- *The MLP is one MLP applied to every token separately. Replacing it with many MLPs and routing each token to a few of them gives a **Mixture of Experts (MoE)**.*

Why this layout:
- **Residuals** make each sub-block learn an *update* to the running representation. They also give the gradient a direct path through all $L$ layers.
- **Attention** is the only part that mixes information *across* tokens. LayerNorm and the MLP act on each row (token) independently.
- **The MLP**, $\text{MLP}(\hat h) = W_2\,\phi(W_1\hat h)$ with $W_1 \in \mathbb R^{d_{ff}\times d}$, holds most of the layer's parameters.

### 3.5 The KV cache

During generation, each new token would naively recompute attention over the whole prefix. But the keys and values of earlier tokens never change: under the causal mask, token $u$'s $K$ and $V$ depend only on tokens $\le u$. So **each layer stores $K_i, V_i$ for the previous steps.** At step $t$, $x_t$ is projected to $Q_t, K_t, V_t$, and only $K_t, V_t$ are appended to the cache.

![[03 - Attachments/ATML_L7_kv_cache.png|600]]

*From your Lecture 9 notes. $x_1$ produces $(q_1, k_1, v_1)$ and $z_1 = \langle q_1, k_1\rangle v_1$, and $(k_1, v_1)$ goes into the cache. $x_2$ only computes $(q_2, k_2, v_2)$; $z_2 = \langle q_2, k_1\rangle v_1 + \langle q_2, k_2\rangle v_2$ reuses $k_1, v_1$ from the cache. The weights $\langle q, k\rangle$ are shown before the softmax for simplicity.*

*(Expanding on the cost:)* without the cache, generating token $t$ recomputes $K, V$ for all $t$ previous tokens. With it, each step projects one token and does one row of attention, $O(t\,d)$ instead of $O(t^2 d)$. The price is memory: $2 \times L \times T \times d$ numbers per sequence (Problem 5). [[Lecture_06_Decoding_KV_Cache_MQA_GQA_FlashAttention#4. KV Cache|CS5326 Lecture 6 §4]] covers this in depth.

### 3.6 Final LayerNorm and unembedding

After $L$ blocks, $H^{(L)} \in \mathbb R^{T\times d}$ goes through a last LayerNorm, $\bar H = \text{LN}(H^{(L)})$. **Unembedding** then turns each row into one score per vocabulary entry, the **logits** $Z \in \mathbb R^{T\times V}$. There are two options:

$$\underbrace{Z = \bar H\, W_E^\top}_{\text{tied embedding: reuse } W_E} \qquad\text{or}\qquad \underbrace{Z = \bar H\, W_U}_{\text{learn } W_U \in \mathbb R^{d\times V}}$$

In $Z$, each row is a token position and each column a vocabulary entry. A softmax over the vocabulary dimension (row-wise) turns each row into a probability distribution, $P \in \mathbb R^{T\times V}$, each row summing to 1:

$$\boxed{P_{t,v} = \frac{e^{Z_{t,v}}}{\sum_{v'=1}^{V} e^{Z_{t,v'}}}}$$

For fixed $t$, $v \in \{1, \dots, V\}$ runs along one row of $P$. Your Lecture 10 notes call $\bar H$ the **latent representation**. They also draw the head as "pass $h_1$ ($d\times1$) to a linear layer $W$ ($V\times d$, $V \approx 300$K there) → $z_1$ ($V\times1$) → softmax". That is the same unembedding applied to one row.

---

## 4. The Training Loss

With the forward pass defined, training needs a single number to minimize. It is the average next-token cross-entropy.

### 4.1 Cross-entropy at one position

At each position $t$:
- **input context** $= [x_1, x_2, \dots, x_t]$;
- **target label** $= x_{t+1}$.

Let $y_t$ be the true next token's **ID**. It is not a value but an index that tells us *which* probability in row $t$ to evaluate. The probability the model assigns to the true token is

$$P_{t, y_t} = \frac{e^{Z_{t, y_t}}}{\sum_{v'} e^{Z_{t, v'}}}$$

and the loss at position $t$ is its negative log:

$$\boxed{\ell_t = -\log P_{t,y_t} = -Z_{t, y_t} + \log\sum_{v'=1}^{V} e^{Z_{t, v'}} = -\log P_\theta(x_{t+1} \mid x_{\le t})}$$

This is the standard cross-entropy for a sample of class $j$, $\mathcal L_{CE} = -z_j + \log\sum_i e^{z_i}$ ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models|Lecture 3 §2.3]]).

### 4.2 Worked example and teacher forcing

Take $T = 4$ and $V = 6$, so $Z \in \mathbb R^{4\times6}$: each row is one time step, each column one vocabulary token. Let the true next-token indices be $y = [y_1, y_2, y_3, y_4] = [3, 5, 2, 4]$. At $t = 1$ the true next token has ID 3; at $t = 2$ it has ID 5; and so on. The loss picks the right logit from each row:

![[03 - Attachments/ATML_L7_loss_logits.png|430]]

*Row $t$ contributes the circled entry $Z_{t, y_t}$: $Z_{1,3}, Z_{2,5}, Z_{3,2}, Z_{4,4}$.*

$$\ell_1 = -Z_{1,3} + \log\textstyle\sum_{v'}e^{Z_{1,v'}}, \quad \ell_2 = -Z_{2,5} + \log\sum_{v'}e^{Z_{2,v'}}, \quad \ell_3 = -Z_{3,2} + \cdots, \quad \ell_4 = -Z_{4,4} + \cdots$$

The sequence loss averages over the positions that have a target:

$$\boxed{L = \frac{1}{T-1}\sum_{t=1}^{T-1}\ell_t = \frac{1}{T-1}\sum_{t=1}^{T-1}\Big(-Z_{t, y_t} + \log\sum_{v=1}^V e^{Z_{t,v}}\Big)}$$

> [!note] Counting positions
> A sequence of $T$ tokens has only $T-1$ next-token targets, since the last token has nothing after it. Hence the $\frac{1}{T-1}$. The example lists four targets for $T = 4$, which treats $T$ as the number of *predictions*. With four targets the average is over four terms. Either way: average over the positions that have a target.

**Teacher forcing:** when predicting the next token, the model is fed the true previous tokens, not its own earlier predictions. All $T-1$ predictions then depend only on known inputs, so the causal mask lets them be computed in one parallel pass. *(Expanding on this:)* the cost is **exposure bias**. At generation time the model conditions on its own outputs, which it never saw during training, so an early mistake can push it into unfamiliar contexts.

### 4.3 Batching, padding and the loss mask

In practice data is processed in batches, $b = 1, \dots, B$, so the logits and probabilities become $Z, P \in \mathbb R^{B\times T\times V}$. Every sequence in a batch must have the same length $T$, but real sentences don't:

| $b$ | Text | True length |
|---|---|---|
| 1 | "The cat sat on the mat" | 6 |
| 2 | "I love AI" | 3 |
| 3 | "Transformers are large language models" | 5 |

The **data collator / batch loader** pads every sequence to a common maximum length. Your notes: the tokenizer finds $T_{\max}$ and adds $\langle\text{PAD}\rangle$ tokens, so $X \in \mathbb R^{B\times T_{\max}\times d}$. With $T = 8$ and $\langle\text{bos}\rangle \ldots \langle\text{eos}\rangle$ around the text:

- $b=1$: [⟨bos⟩, The, cat, sat, on, the, mat, ⟨eos⟩]
- $b=2$: [⟨bos⟩, I, love, AI, ⟨eos⟩, ⟨PAD⟩, ⟨PAD⟩, ⟨PAD⟩]
- $b=3$: [⟨bos⟩, Transformers, are, large, language, models, ⟨eos⟩, ⟨PAD⟩]

Padding positions must not contribute to the loss. A **binary mask** excludes them:

$$m_{b,t} = \begin{cases}1 & \text{position } t \text{ of sequence } b \text{ holds a real token}\\ 0 & \text{it holds } \langle\text{PAD}\rangle\end{cases}$$

The batch loss averages over valid tokens only:

$$\boxed{L(\theta) = -\frac{1}{\sum_{b=1}^{B}\sum_{t=1}^{T-1} m_{b,t}}\sum_{b=1}^{B}\sum_{t=1}^{T-1} m_{b,t}\,\log P_\theta(x_{b,t+1} \mid x_{b,\le t})}$$

$N_{\text{valid}} = \sum_b\sum_t m_{b,t}$ is the count of valid tokens over the whole batch.

**Example ($B = 2$, $T = 5$).** The masks are $m_1 = [1,1,1,1,0]$ and $m_2 = [1,1,1,0,0]$, so $N_{\text{valid}} = 4 + 3 = 7$ and

$$L(\theta) = -\frac17\Big[\sum_{t=1}^{4}\log P_\theta(x_{1,t+1} \mid x_{1,\le t}) + \sum_{t=1}^{3}\log P_\theta(x_{2,t+1} \mid x_{2,\le t})\Big]$$

*(Why divide by $N_{\text{valid}}$ and not by $B\cdot T$:)* dividing by the padded size would make the loss, and so the gradient scale, depend on how much padding the batch happens to contain.

### 4.4 Gradient of the loss with respect to the logits

The professor states:

$$\boxed{\frac{\partial L}{\partial Z_{b,t,v}} = \frac{m_{b,t}}{N_{\text{valid}}}\Big(P_{b,t,v} - \mathbb 1[v = y_{b,t}]\Big)}$$

*(Derivation filled in:)* only the term $\ell_{b,t}$ depends on $Z_{b,t,\cdot}$, and it enters $L$ with weight $m_{b,t}/N_{\text{valid}}$. Differentiating $\ell = -Z_{y} + \log\sum_{v'}e^{Z_{v'}}$:
- the first term gives $-\mathbb 1[v = y]$;
- the log-sum-exp gives $e^{Z_v}/\sum_{v'}e^{Z_{v'}} = P_v$.

The gradient is "predicted minus target". Training pushes the true token's logit up by $1 - P_y$ and every other logit down by its probability. A masked position ($m = 0$) gets no gradient at all.

### 4.5 The softmax bottleneck

The output softmax in a decoder LLM operates along the $V$ dimension independently at every position. With $d = 4096$, $V = 50{,}000$, $T = 1024$ and $B = 8$:
- **the last layer alone has $d \times V \approx 200$ million parameters**;
- every token needs a 50,000-way softmax.

GPUs do this in parallel for millions of tokens at once using optimized kernels, and other approximation tricks are also used. *(Expanding on this:)* the logits tensor $B\times T\times V$ here holds $4.1\times10^8$ numbers, about 1.6 GB in fp32, for one batch. Common tricks:
- **fused softmax–cross-entropy** kernels that never materialize the full probability tensor;
- **chunking** over $T$;
- for very large vocabularies, sampled or hierarchical softmax.

The name "softmax bottleneck" is also used for a different, theoretical point. The logits $Z = \bar H W_U$ have rank at most $d$, so a $d$-dimensional model can't represent every possible $V$-way distribution when $V \gg d$.

---

## 5. Scaling Laws

Training a big model is expensive, so some dials have to be planned before and during training. **Scaling laws** are empirical **power-law** relationships showing how an LLM's loss, and many downstream metrics, improve as three things are scaled:
1. **model size** $N$ (number of parameters);
2. **training tokens** (data) $D$;
3. **compute** (FLOPs) $C$.

They answer two questions: how far scale alone can take us, and how to divide a compute budget $C$ between parameters $N$ and tokens $D$.

### 5.1 Power laws

"The test loss varies as a power law with respect to the number of parameters" means that a change in one quantity produces a *relative*, scale-invariant change in the other. For two quantities $x$ and $y$:

$$y = a\,x^{p}$$

Here $a$ and $p$ set the shape of the curve. The **inverse version**, $y = a(1/x)^p = a\,x^{-p}$, is the same law with a negative exponent. That is the loss case: loss falls as scale grows.

![[03 - Attachments/ATML_L7_power_law.png|640]]

*Left: $y = x^{0.5}$ ($a = 1$, $p = 0.5$) on normal axes. Middle: the same curve on log–log axes is a straight line. Bottom: the inverse version decays on normal axes and is again a line on log scale.*

Taking logs shows why a power law is a **linear trend on log scale**:

$$\log y = \log a + p\log x$$

This is a line with slope $p$. One quantity varies as a power of another, and the fitting form used is $f(x) = a\,x^{-k}$. For neural nets the loss is modelled as a sum of such terms, one per resource:

$$L(P_i) = \sum_{i=1}^{n}\alpha_i\, P_i^{-\beta_i}$$

Here $P_i$ is a performance factor ($N$, $D$, $C$, …), $\beta_i$ a scaling parameter and $\alpha_i$ a fitting parameter.

### 5.2 Kaplan laws

**The foundation (Kaplan et al., 2020):** loss $L$ scales as a power law in each factor when it is the one limiting the model.

![[03 - Attachments/ATML_L7_kaplan.png|600]]

$$L(N) = E + A\,N^{-\alpha_N} \;\Rightarrow\; L(N) \propto N^{-\alpha_N}, \qquad \alpha_N \approx 0.076$$
$$L(D) = E + B\,D^{-\alpha_D} \;\Rightarrow\; L(D) \propto D^{-\alpha_D}, \qquad \alpha_D \approx 0.095$$
$$L(C) = E + F\,C^{-\alpha_C} \;\Rightarrow\; L(C) \propto C^{-\alpha_C}$$

> [!warning] Possible source inconsistency
> The notes write the data law as $L(D) = E + B\,N^{-\alpha_D} \Rightarrow L(N) \propto D^{-\alpha_D}$. Both $N$s are slips for $D$: the data-limited law depends only on $D$.

$E$ is the **irreducible loss**, the entropy of language itself, which no model can beat. Only the excess $L - E$ follows the power law. *(Expanding on how small the exponents are:)* with $\alpha_N = 0.076$, doubling $N$ multiplies the reducible loss by $2^{-0.076} = 0.949$, only a 5% cut. Big gains need orders of magnitude.

**Combined law** when $N$ and $D$ vary together:

$$L(N, D) \propto \Big(\frac{1}{N^{\alpha_N}} + \frac{1}{D^{\alpha_D}}\Big)$$

**Optimal data-to-parameter ratio.** To avoid being bottlenecked by one factor, scale so that the penalty from too small an $N$ roughly equals the penalty from too small a $D$. That happens when the two terms balance:

$$\frac{1}{N^{\alpha_N}} \propto \frac{1}{D^{\alpha_D}} \;\Rightarrow\; D \propto N^{\alpha_N/\alpha_D} \;\Rightarrow\; \boxed{D \propto N^{0.74}}$$

The exponent is $0.076/0.095 \approx 0.8$ from the rounded values; Kaplan's fitted value is 0.74. It is **sub-linear**: data should grow more slowly than $N$.

**Compute scaling.** Under a fixed compute constraint $C \propto N\cdot D$, Kaplan found

$$N \propto C^{0.73}, \qquad D \propto C^{0.27}$$

The exponents sum to 1, as $C \propto ND$ requires. **Kaplan's conclusion: for a fixed compute budget, big models matter more than big data.** Most extra compute should go into parameters.

### 5.3 Chinchilla laws

**The unified view (Hoffmann et al., 2022):** the Chinchilla paper challenged Kaplan's optimal ratio $D \propto N^{0.74}$. It showed that Kaplan-style models were actually **under-trained**: too many parameters fed too few tokens. It proposed

$$\boxed{L(N, D) = E + \frac{A}{N^{\alpha_N}} + \frac{B}{D^{\alpha_D}}}, \qquad \alpha_N \approx \alpha_D \approx 0.34$$

with total training compute for given $N$ and $D$

$$\boxed{C \approx 6\,N\,D}$$

*(Expanding on the 6:)* each token costs about $2N$ FLOPs in the forward pass (one multiply and one add per parameter) and about $4N$ in the backward pass, so about $6N$ per training token. For a fixed cost, $N$ and $D$ are inversely related: $ND$ must stay constant.

**Minimize $L(N, D)$ subject to $C = 6ND$.** The optimum balances the two terms:

$$\frac{A}{N^{\alpha_N}} \approx \frac{B}{D^{\alpha_D}}$$

Since $\alpha_N \approx \alpha_D = \alpha$, this gives

$$\Big(\frac{D}{N}\Big)^{\alpha} \approx \frac{B}{A} = \text{const} \;\Rightarrow\; \boxed{D \propto N}$$

For a fixed $C$, getting a lower loss means scaling $N$ and $D$ at about the **same linear rate**. Substituting into $C = 6ND$ splits the compute evenly:

$$N \propto C^{0.5}, \qquad D \propto C^{0.5}$$

**Rule of thumb: budget about 20 training tokens per parameter.**

*(Expanding on the balance condition:)* with $D = C/(6N)$, $L(N) = E + AN^{-\alpha} + B(6N/C)^{\alpha}$. Setting $dL/dN = 0$ gives $\alpha A N^{-\alpha} = \alpha B (6N/C)^{\alpha}$, i.e. $A N^{-\alpha} = B D^{-\alpha}$, which is exactly the balance above. [[RA1_Chinchilla_Cheatsheet|CS5326 RA1]] covers the paper itself.

### 5.4 A roadmap, and scaling laws beyond pre-training

![[03 - Attachments/ATML_L7_roadmap.png|640]]

*Roadmap for the objective "develop an LLM", given what you have:*

| Sufficient data? | Sufficient compute? | Strategy |
|---|---|---|
| yes | yes | **scale data and model together** (Chinchilla law) ✓ |
| yes | no | **MoE scaling** ✗ |
| no | yes | **longer training** (more epochs over the data you have) ✗ |
| no | no | **data mixing scaling** ✗ |

*(Only the first route has the professor's ✓; the ✗ routes are compromises.)* Briefly:
- **MoE** gets more parameters for the same per-token compute, since each token uses only a few experts.
- **Longer training** repeats data, which gives diminishing returns after a few epochs.
- **Data mixing** chooses the proportions of different data sources to get the most from a limited corpus.

Scaling-law research has since expanded to other stages of the LLM:
- **fine-tuning / alignment-stage laws;**
- **inference-time laws** (more compute at test time, e.g. longer reasoning, lowers error);
- **multimodal LLM laws.**

---

## 6. Decoding

### 6.1 The decoding problem

A trained causal LLM with fixed $\theta$ defines $P_\theta(x_1, \dots, x_T)$. Selecting a sequence of tokens as the model generates them is called **decoding** or **sampling**. At each step $t$ the model gives a distribution $P_\theta(x_t \mid x_{<t})$ over the $|V|$ vocabulary entries, and decoding must decide which $x_t$ to choose.

**Running example.** Vocabulary $V = \{$"mat", "rug", "floor", "dog", "moon"$\}$, so $|V| = 5$. Context: $x_{\le t}$ = ["The", "cat", "sat", "on", "the"]. Predicted probabilities for $x_{t+1}$:

| token | mat | rug | floor | dog | moon |
|---|---|---|---|---|---|
| $P_\theta$ | 0.5 | 0.3 | 0.15 | 0.04 | 0.01 |

The goal is a balance: a high-probability sequence (fluent and coherent) that is also diverse and not repetitive. Your notes frame it as a trade-off between **fluency** (deterministic, but repetitive) and **innovation** (stochastic).

### 6.2 Greedy decoding

Always select the most probable token:

$$x_{t+1} = \arg\max_x P_\theta(x \mid x_{\le t})$$

In the example, $x_{t+1}$ = "mat". **Problem:** it is myopic. Each step is locally optimal, but the sequence can be globally sub-optimal: a slightly less likely word now might open a much more likely continuation.

### 6.3 Sampling with temperature

**Naive sampling** picks randomly from the whole distribution. This reflects the model's uncertainty: "mat" (0.5) and "rug" (0.3) are both good options, and they are picked about 50% and 30% of the time.

A **temperature** $\rho$ reshapes the distribution first, by dividing the logits: $P_\rho(v) \propto e^{z_v/\rho}$, equivalently $P_\rho(v) \propto P(v)^{1/\rho}$.
- $\rho = 1$: $P_\theta(x_t \mid x_{<t})$ is unchanged.
- $\rho < 1$ sharpens the distribution. $\rho = 0.5$ squares the probabilities and renormalizes: $[0.686, 0.247, 0.062, 0.004, 0.0003]$.
- $\rho \to 0$ puts all mass on the top token and **reduces to greedy**.
- $\rho > 1$ flattens it toward uniform.

> [!note] The $\rho = 0.5$ numbers
> The professor's notes give $[0.75, 0.2, 0.04, 0.008, 0.002]$ for $\rho = 0.5$. That shows the right direction (sharper), but the exact values from $P^{1/\rho}$ are the ones above.

**Problem:** sampling from the full distribution can still pick low-probability tokens. Across a long sequence, a few such picks can derail it.

### 6.4 Beam search

Beam search explores several promising sequences at once to find a globally better one. With **beam size $K$** it keeps the $K$ most probable partial generations. At step $\ell$ it:
1. generates all possible one-token extensions of the $K$ sequences;
2. selects the top-$K$ by **cumulative log-probability**,
$$\text{score}_i = \sum_{t=1}^{\ell}\log P_\theta(x_{i,t+1} \mid x_{i,\le t}), \qquad i = 1, \dots, K;$$
3. carries those $K$ forward to the next step.

![[03 - Attachments/ATML_L7_beam.png|520]]

*Beam search with $K = 3$ from "The cat". The surviving continuations (boxed) are kept at each step: "sat", "stood", "played", then "on", "up", "on", then "the", "to", "the".*

When $K = 1$ it reduces to greedy. **Problem:** beam-search text is less surprising. It tends to be repetitive and uninteresting. As LLMs got better, their lower-probability tokens became good choices too, and beam search throws those away. *(Expanding on this:)* human text is not the most probable text. Maximizing likelihood favours safe, generic continuations, which is why open-ended generation uses sampling and beam search survives mainly in tasks with one right answer, such as translation.

### 6.5 Top-$k$ sampling

Identify the $k$ most probable tokens, renormalize over them, and sample (naive sampling on the truncated set). In the example with $k = 3$, "mat", "rug" and "floor" sum to 0.95, so

$$0.5/0.95 = 0.526, \quad 0.3/0.95 = 0.316, \quad 0.15/0.95 = 0.158$$

"dog" and "moon" can never be picked. *(Weakness, added:)* $k$ is fixed. When the model is confident, $k$ tokens include junk; when it is unsure, $k$ may cut good options.

### 6.6 Top-$p$ (nucleus) sampling

Find the **smallest set** of most likely tokens whose cumulative probability is $\ge \hat p$, renormalize, and sample from it. With $\hat p = 0.9$:
- "mat" alone: $0.5 < \hat p$;
- "mat" + "rug": $0.5 + 0.3 = 0.8 < \hat p$;
- "mat" + "rug" + "floor": $0.5 + 0.3 + 0.15 = 0.95 \ge \hat p$, so **stop**.

The selected **nucleus** is ["mat", "rug", "floor"], which is then renormalized and sampled. Here it matches top-3. Unlike top-$k$, the set size **adapts**: a confident distribution gives a small nucleus, a flat one a large nucleus.

### 6.7 What modern LLMs use

Modern LLMs use a **combination** of temperature $\rho$, top-$k$ and top-$\hat p$, and the mix differs from model to model; GPT, for example, uses $\rho$ + top-$\hat p$. *(Expanding on the order:)* typically the temperature is applied to the logits first, then the top-$k$/top-$p$ truncation, then sampling.

---

## Numerical Problems

**Problem 1: cross-entropy at one position.** A position has logits $z = [2.0, 1.0, 0.5, -1.0]$ over a 4-token vocabulary. Find the probabilities and the loss when the true token is index 1, and when it is index 3.

The log-sum-exp is $\log(e^{2} + e^{1} + e^{0.5} + e^{-1}) = \log(7.389 + 2.718 + 1.649 + 0.368) = \log 12.124 = 2.495$. The probabilities are $e^{z_v - 2.495} = [0.610, 0.224, 0.136, 0.030]$. The loss is $\ell = -z_y + 2.495$:

$$\boxed{\ell(y{=}1) = 0.495, \qquad \ell(y{=}3) = 1.995}$$

Check: $-\ln 0.610 = 0.495$ ✓. Predicting a less likely token costs more.

**Problem 2: the logit gradient.** For the same position, $m = 1$, $N_{\text{valid}} = 4$ and true token index 1, find $\partial L/\partial z$.

By §4.4, $\partial L/\partial z = \frac{1}{4}(P - \mathbf e_1) = \frac14[0.610 - 1, 0.224, 0.136, 0.030]$:

$$\boxed{\partial L/\partial z = [-0.0975,\ 0.056,\ 0.034,\ 0.0075]}$$

The entries sum to 0: the push up on the true logit equals the total push down on the others.

**Problem 3: masked batch loss.** $B = 2$ with masks $m_1 = [1,1,1,1,0]$ and $m_2 = [1,1,1,0,0]$. The valid positions have true-token probabilities $[0.5, 0.25, 0.8, 0.1]$ for sequence 1 and $[0.6, 0.3, 0.9]$ for sequence 2. Find $L$, and the wrong value you'd get by averaging over all 10 positions (padding counted as loss 0).

The sum of $-\ln p$ is
- sequence 1: $0.693 + 1.386 + 0.223 + 2.303 = 4.605$;
- sequence 2: $0.511 + 1.204 + 0.105 = 1.820$;

for a total of $6.425$. With $N_{\text{valid}} = 7$:

$$\boxed{L = 6.425/7 = 0.918 \quad(\text{vs } 6.425/10 = 0.643 \text{ if padding is counted})}$$

Counting padding would understate the loss, and by an amount that depends on how much padding the batch has.

**Problem 4: the softmax bottleneck.** With $d = 4096$ and $V = 50{,}000$, how many parameters does an untied $W_U$ have, how many does tying save, and how large is the fp32 logits tensor for $B = 8$, $T = 1024$?

- $W_U$: $d \times V = 4096 \times 50{,}000 = 2.048\times10^8$ parameters. Tying reuses $W_E$, saving exactly that many.
- Logits: $B\,T\,V = 8 \times 1024 \times 50{,}000 = 4.096\times10^8$ floats $\times$ 4 bytes $= 1.64$ GB.

$$\boxed{\approx 205\text{M parameters saved by tying};\ \text{logits} \approx 1.6\text{ GB}}$$

**Problem 5: KV-cache memory.** A model has $L = 32$ layers and $d = 4096$, generating $T = 2048$ tokens in fp16 (2 bytes). How much memory does the KV cache need per sequence?

Each layer stores a key and a value vector of size $d$ per token: $2 \times L \times T \times d = 2 \times 32 \times 2048 \times 4096 = 5.37\times10^8$ numbers, times 2 bytes $= 1.07\times10^9$ bytes.

$$\boxed{\approx 1\text{ GiB per sequence}}$$

It grows linearly in $T$ and in the batch size, which is why long contexts are memory-bound.

**Problem 6: Kaplan vs Chinchilla.** Compute grows $10\times$. How should $N$ and $D$ grow under each law?

- **Kaplan:** $N \propto C^{0.73}$ gives $10^{0.73} = 5.37\times$, and $D \propto C^{0.27}$ gives $10^{0.27} = 1.86\times$.
- **Chinchilla:** $N, D \propto C^{0.5}$ gives $10^{0.5} = 3.16\times$ each.

$$\boxed{\text{Kaplan: } N\times5.4,\ D\times1.9; \quad \text{Chinchilla: } N\times3.2,\ D\times3.2}$$

Both products are $10\times$, as $C \propto ND$ requires.

**Problem 7: compute-optimal model size.** A budget of $C = 6\times10^{23}$ FLOPs. Using $C = 6ND$ and the 20-tokens-per-parameter rule, find $N$ and $D$.

$ND = 10^{23}$ and $D = 20N$, so $20N^2 = 10^{23}$ and $N = \sqrt{5\times10^{21}} = 7.07\times10^{10}$; then $D = 1.41\times10^{12}$.

$$\boxed{N \approx 71\text{B parameters},\ D \approx 1.4\text{T tokens}}$$

This is Chinchilla itself: 70B parameters on 1.4T tokens.

**Problem 8: temperature, top-$k$ and top-$p$.** For $P = [0.5, 0.3, 0.15, 0.04, 0.01]$, find the distribution at $\rho = 2$, and the nucleus for $\hat p = 0.75$.

At $\rho = 2$, $P^{1/2} = [0.707, 0.548, 0.387, 0.2, 0.1]$, which sums to $1.942$. For $\hat p = 0.75$: "mat" alone is $0.5 < 0.75$; "mat" + "rug" is $0.8 \ge 0.75$, so stop. Renormalize: $0.5/0.8$, $0.3/0.8$.

$$\boxed{\rho = 2:\ [0.364, 0.282, 0.199, 0.103, 0.051]; \quad \text{nucleus} = \{\text{mat } 0.625, \text{rug } 0.375\}}$$

A higher temperature lifts "moon" from 1% to 5%. A tighter $\hat p$ shrinks the nucleus.

**Problem 9: beam search beats greedy.** Step 1: $P(\text{A}) = 0.6$, $P(\text{B}) = 0.4$. Step 2: after A the best next token has probability 0.3; after B the best has 0.9. Compare greedy with beam search ($K = 2$).

Greedy picks A, then its best: $0.6 \times 0.3 = 0.18$, log-score $-1.71$. Beam keeps both, and B's best path scores $0.4 \times 0.9 = 0.36$, log-score $-1.02$.

$$\boxed{\text{beam: } 0.36 \text{ vs greedy: } 0.18}$$

Greedy's locally best first step led to a worse sequence.

---

## Formula Summary Sheet

| Concept | Formula |
|---|---|
| AR factorization | $P_\theta(x_{1:T}) = \prod_t P_\theta(x_t \mid x_{<t})$ |
| Embedding | $E_t = W_E[x_t]$, $W_E \in \mathbb R^{V\times d}$, $E \in \mathbb R^{T\times d}$ |
| Input to layer 1 | $H^{(0)} = E + P$ |
| Self-attention (column form) | $z_i = \sum_j a_{ij}v_j$, $Z = VA^\top$ |
| Head projections | $Q_i = \tilde HW_{Q,i}$, $K_i = \tilde HW_{K,i}$, $V_i = \tilde HW_{V,i} \in \mathbb R^{T\times d_h}$, $d_h = d/h$ |
| Causal attention | $A_i = \text{softmax}(Q_iK_i^\top/\sqrt{d_h} + M)$, $M_{t,u} = 0$ if $u\le t$, else $-\infty$ |
| Heads → output | $O_i = A_iV_i$; $O = \text{concat}(O_1..O_h)$; $U = OW_O$ |
| Pre-LN layer | $\tilde H = \text{LN}(H^{(l-1)})$; $H_{\text{attn}} = H^{(l-1)} + U$; $H^{(l)} = H_{\text{attn}} + \text{MLP}(\text{LN}(H_{\text{attn}}))$ |
| Final layer | $\bar H = \text{LN}(H^{(L)})$ |
| Logits | tied $Z = \bar HW_E^\top$; untied $Z = \bar HW_U$; $Z \in \mathbb R^{T\times V}$ |
| Softmax | $P_{t,v} = e^{Z_{t,v}}/\sum_{v'}e^{Z_{t,v'}}$ |
| Token loss | $\ell_t = -Z_{t,y_t} + \log\sum_{v'}e^{Z_{t,v'}}$ |
| Sequence loss | $L = \frac{1}{T-1}\sum_{t=1}^{T-1}\ell_t$ |
| Masked batch loss | $L = -\frac{1}{N_{\text{valid}}}\sum_b\sum_t m_{b,t}\log P_\theta(x_{b,t+1}\mid x_{b,\le t})$ |
| Logit gradient | $\partial L/\partial Z_{b,t,v} = \frac{m_{b,t}}{N_{\text{valid}}}(P_{b,t,v} - \mathbb 1[v = y_{b,t}])$ |
| KV-cache size | $2 \times L \times T \times d$ numbers per sequence |
| Power law | $y = ax^p$; $\log y = \log a + p\log x$ |
| Kaplan | $L(N) = E + AN^{-\alpha_N}$ ($\alpha_N \approx 0.076$); $L(D) = E + BD^{-\alpha_D}$ ($\alpha_D \approx 0.095$); $L(C) = E + FC^{-\alpha_C}$ |
| Kaplan optimum | $D \propto N^{0.74}$; $N \propto C^{0.73}$, $D \propto C^{0.27}$ |
| Chinchilla | $L(N,D) = E + A/N^{\alpha_N} + B/D^{\alpha_D}$, $\alpha_N \approx \alpha_D \approx 0.34$ |
| Training compute | $C \approx 6ND$ |
| Chinchilla optimum | $D \propto N$; $N, D \propto C^{0.5}$; ≈ 20 tokens per parameter |
| Greedy | $x_{t+1} = \arg\max_x P_\theta(x\mid x_{\le t})$ |
| Temperature | $P_\rho(v) \propto e^{z_v/\rho} \propto P(v)^{1/\rho}$ |
| Beam score | $\sum_{t=1}^{\ell}\log P_\theta(x_{i,t+1}\mid x_{i,\le t})$ |
| Top-$k$ / top-$p$ | renormalize over the top $k$ / the smallest set with cumulative prob $\ge \hat p$ |

---

## Cheat Sheet & Recall Aids

**Mnemonics**
- **Pipeline: "Tokens Enter Positions, Layers Unembed Softly":** Tokenize → Embed → add Position → $L$ Layers → Unembed → Softmax.
- **Layer, pre-LN: "Norm, Attend, Add; Norm, MLP, Add".**
- **Decoding ladder: "Greedy Takes Big Top Picks":** Greedy → Temperature → Beam → Top-$k$ → Top-$p$.

**Gotchas**
- A token is an **index**, not a value. $y_t$ picks which entry of row $t$ to read.
- The causal mask adds $-\infty$ **before** the softmax, so future weights become exactly 0.
- $T$ tokens give $T-1$ targets. With padding, divide by $N_{\text{valid}}$, never by $B\times T$.
- Logit gradient = **predicted − one-hot**, scaled by $m/N_{\text{valid}}$.
- Tied weights: $Z = \bar HW_E^\top$. That saves $V\cdot d$ parameters, about 200M at $d = 4096$, $V = 50$K.
- Kaplan says parameters matter more ($D \propto N^{0.74}$). Chinchilla says scale both equally ($D \propto N$, 20 tokens per parameter); Kaplan's models were **under-trained**.
- $C \approx 6ND$: 2 FLOPs per parameter per token forward, 4 backward.
- Temperature $\rho \to 0$ = greedy; beam $K = 1$ = greedy.
- Top-$k$ has a fixed set size; top-$p$ **adapts** to the model's confidence.
- The KV cache saves compute, costs memory, and works only because causal attention never changes old $K, V$.

**If you remember nothing else**
- A decoder LLM is $P_\theta(x_{t+1}\mid x_{\le t})$, trained with masked, teacher-forced cross-entropy.
- Attention = softmax-weighted values; the weights come from query–key inner products, masked to the past.
- Scaling: loss is a power law in $N$, $D$, $C$. Compute-optimal means growing $N$ and $D$ together.
- Decoding trades fluency (greedy/beam) against diversity (sampling with $\rho$, $k$, $p$).

**Recall checklist**
- [ ] Name the three Transformer families and why generation is sequential in AR models; contrast with dLLMs.
- [ ] Write every tensor shape from $x_{1:T}$ to $P$.
- [ ] Write one pre-LN layer's equations, including the causal mask.
- [ ] Explain the KV cache and compute its memory.
- [ ] Compute a token's cross-entropy from logits; write the masked batch loss and the logit gradient.
- [ ] Derive $D \propto N^{\alpha_N/\alpha_D}$ (Kaplan) and $D \propto N$ (Chinchilla); use $C = 6ND$ with 20 tokens per parameter.
- [ ] Apply greedy, temperature, beam, top-$k$ and top-$p$ to a given distribution.

---
type: lecture
course: "[[_CS6304 ATML|CS6304 ATML]]"
source:
  - "[[VLM Notes.pdf]]"
  - "[[ATML Lec 17.pdf]]"
  - "[[DVLM Lec 10.pdf]]"
tags:
  - atml
  - course/cs6304
  - vlm
  - clip
  - multimodal
  - vq-vae
title: ATML — Lecture 11 — Vision-Language Models
lecture: 11
aliases:
  - Vision-Language Models
  - ATML Lecture 11
textfold: md-atml-lecture-11-vision-language-models-24a772f9
---

%% nav:start %%
[[_CS6304 ATML|CS6304 ATML]] · ← [[ATML_Lecture_10 - Fine-Tuning]] · → [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning]]
%% nav:end %%

# ATML — Lecture 11 — Vision-Language Models

*[[ATML_Lecture_10 - Fine-Tuning|Lecture 10]] adapted a text-only LLM to new tasks with small, trainable add-ons such as [[ATML_Lecture_10 - Fine-Tuning#7.5 Adapters|adapters]]. This lecture adds a second modality, vision. It starts from the contrastive image–text model of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.4 CLIP — Contrastive Language–Image Pretraining|Lecture 3 (CLIP)]], studies the geometry that model learns and its variants (SigLIP, LLIP), then moves to models that fuse the modalities (FLAVA), generate from them (CoCa, Chameleon), or bolt a vision encoder onto a frozen LLM through a bridge, which is the adapter idea of [[ATML_Lecture_10 - Fine-Tuning#7.5 Adapters|Lecture 10 §7.5]] applied across modalities. After how VLMs fail, it opens up that modular stack (the ViT encoder, the two kinds of alignment, the three training stages and their limits) and ends with modality unification: turning images into discrete tokens so one next-token loss covers both understanding and generating images.*

The sources are the professor's [[VLM Notes.pdf|VLM Notes]] (11 pages; the same scans as DVLM Lectures 9 and 10, pages 1–5 and 6–11), and your class notes [[ATML Lec 17.pdf|ATML Lecture 17]] (3 pages) and [[DVLM Lec 10.pdf|DVLM Lecture 10]] (5 pages). [[#1. What a VLM Is|§1]]–[[#8. VLM Failure Modes|§8]] come from pages 1–5, and [[#9. Inside the Modular VLM — Vision Encoding and Alignment|§9]]–[[#10. Modality Unification — Images as Discrete Tokens|§10]] from pages 6–11. All figures are redrawn in TikZ from both sets of notes.

## Topic Inventory

- 1. What a VLM Is
  - 1.1 Definition and the three objectives
  - 1.2 Three families
- 2. Representational VLMs
  - 2.1 The dual-encoder pipeline
  - 2.2 The similarity matrix
  - 2.3 The symmetric InfoNCE loss
  - 2.4 Goal and inference
- 3. Geometry of the Shared Space
  - 3.1 Property 1, a cosine-similarity space on the unit sphere
  - 3.2 Property 2, the persistent modality gap
  - 3.3 Why the loss promotes the gap, alignment vs uniformity
  - 3.4 How big should the gap be
  - 3.5 The cone effect
- 4. Beyond Softmax — SigLIP and LLIP
  - 4.1 SigLIP, matching as binary classification
  - 4.2 When SigLIP is good, and its costs
  - 4.3 LLIP, caption-conditioned image embeddings
- 5. Fusion VLMs
  - 5.1 From matching to cross-modal interaction
  - 5.2 FLAVA's architecture
  - 5.3 The five-term FLAVA loss
  - 5.4 Inference and fine-tuning
- 6. Generative VLMs
  - 6.1 CoCa
  - 6.2 Chameleon and CM3leon
  - 6.3 Stable Diffusion and Imagen
- 7. Adaptive Modularity
  - 7.1 The pipeline
  - 7.2 Bridge 1, linear or MLP projector
  - 7.3 Bridge 2, bottleneck resampler (Q-Former)
  - 7.4 Bridge 3, gated cross-attention through an adapter
  - 7.5 Training stages
- 8. VLM Failure Modes
  - 8.1 Five failure types
  - 8.2 Which component fails
- 9. Inside the Modular VLM — Vision Encoding and Alignment
  - 9.1 The pipeline and its two jobs
  - 9.2 The ViT vision encoder
  - 9.3 Read-out for classification
  - 9.4 Why a VLM keeps every patch token
  - 9.5 Representational vs interface alignment
  - 9.6 Limitations of the stack
- 10. Modality Unification — Images as Discrete Tokens
  - 10.1 The idea
  - 10.2 The image tokenizer
  - 10.3 Training the tokenizer, five losses
  - 10.4 Image understanding with token IDs
  - 10.5 Image generation, from token IDs back to pixels
  - 10.6 Token-level autoregressive image generation
  - 10.7 Other families beyond raster-scan autoregression
  - 10.8 How to reason on images?
- Numerical Problems
- Formula Summary Sheet
- Summary

---

## 1. What a VLM Is

### 1.1 Definition and the three objectives

A **Vision-Language Model (VLM)** is a parameterized system that learns a **joint statistical interface between vision observations and text observations** to fulfil one or more of three objectives:

| # | Objective | Typical tasks |
|---|---|---|
| 1 | **Compare** image and text semantics | zero-shot classification, retrieval |
| 2 | **Generate text** conditioned on vision + a prompt | VQA, captioning |
| 3 | **Generate or edit images** conditioned on vision + a prompt | image/video generation |

**VQA** (Visual Question Answering) means answering a free-form question about an image ("What colour is the car?"). **Retrieval** means finding the best-matching caption for an image, or image for a caption, in a database. Your class notes summarise the purpose as "align vision and text modalities better" and draw the common case as a black box:

![[03 - Attachments/ATML_L11_vlm_box.svg|360]]

*Your class note's black-box view of the common case.*

### 1.2 Three families

The objectives map onto **three main VLM families**, which the rest of the lecture takes in turn:

| Family | What it does with the two modalities | Examples | Section |
|---|---|---|---|
| **Representational** | encodes each separately into one shared space and compares | CLIP, SigLIP, LLIP | [[#2. Representational VLMs\|§2]]–[[#4. Beyond Softmax — SigLIP and LLIP\|§4]] |
| **Fusion** | processes image and text tokens **jointly** for richer cross-modal interaction | FLAVA | [[#5. Fusion VLMs\|§5]] |
| **Generative** | **explicitly generates** text and/or images | CoCa, Chameleon, Stable Diffusion | [[#6. Generative VLMs\|§6]] |

The families line up with the objectives: representational models serve objective 1 directly, fusion models still mostly compare but at the level of words and regions, and generative models are the ones that serve objectives 2 and 3. [[#7. Adaptive Modularity|§7]] (adaptive modularity) is the practical recipe behind most current text-generating VLMs: connect a vision encoder to an existing LLM.

---

## 2. Representational VLMs

The simplest way to relate an image and a caption is to map both into one vector space and measure how close they land. This is the contrastive model of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.4 CLIP — Contrastive Language–Image Pretraining|Lecture 3 §3.4]] (**CLIP**, Contrastive Language–Image Pre-training), now written in this lecture's notation.

### 2.1 The dual-encoder pipeline

The data are $N$ image–text pairs $(x_i^V, x_j^T)$, $i, j = 1, \dots, N$, where superscript $V$ marks vision and $T$ text. Pair $(i, j)$ is a **positive pair** when $i = j$ (the image and its own caption) and a **negative pair** when $i\neq j$. Two encoders run in parallel, an image encoder $\theta_V$ and a text encoder $\theta_T$, and all trainable parameters are $\theta = \{\theta_V, \theta_T\}$:

$$x_i^V \xrightarrow{\ \theta_V\ } z_i^V,\qquad x_j^T \xrightarrow{\ \theta_T\ } z_j^T$$

A **projection and normalization** block maps both to the same dimension $d$ and rescales them to unit length, giving $v_i$ and $t_j$, which live in the **shared semantic space**. A **similarity computation** then scores every pair, and the scores feed a **bi-directional contrastive loss**.

![[03 - Attachments/ATML_L11_representational_vlm.svg|405]]

*Two separate encoders meet only in the shared space, where every image–text pair is scored.*

### 2.2 The similarity matrix

The similarity between image $i$ and text $j$ is a scaled dot product,

$$s_{ij} = \frac{v_i^\top t_j}{\tau}$$

where $\tau > 0$ is the **temperature**. A small $\tau$ sharpens the softmax that follows; in CLIP it is learned ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.5 CLIP Training|Lecture 3]]'s "scale" is $1/\tau$). Stacking all pairs gives the $N\times N$ matrix $S$, with **rows indexed by image $i$** and **columns by text $j$**:

$$S = \begin{bmatrix} s_{11} & s_{12} & \cdots & s_{1N}\\ s_{21} & s_{22} & \cdots & s_{2N}\\ \vdots & & \ddots & \vdots\\ s_{N1} & \cdots & & s_{NN}\end{bmatrix}$$

### 2.3 The symmetric InfoNCE loss

Each row of $S$ is an $N$-way classification problem for one image: which of the $N$ captions is the correct one? The **correct caption** ($j = i$) is the **positive** and the **other captions** are the **negatives**. For a given image, the denominator sums over all text captions:

$$L_{i2t}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ij})}$$

Each column is the same problem the other way round. For a given text prompt, the sum runs over all images:

$$L_{t2i}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s_{ii})}{\sum_{j=1}^{N}\exp(s_{ji})}$$

The **symmetric InfoNCE loss** averages the two:

$$\boxed{L_c(\theta) = \tfrac{1}{2}\big(L_{i2t}(\theta) + L_{t2i}(\theta)\big)}\qquad\qquad\text{(1)}$$
^eq-1

Each term is a softmax cross-entropy whose "classes" are the batch's captions (or images). This is the similarity form of cross-entropy from [[ATML_Lecture_1 - Representation and Generalization Theory#7.4 The Similarity-Based Form of Cross-Entropy|Lecture 1 §7.4]] and the InfoNCE loss of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.2 The Contrastive Pipeline and the InfoNCE Loss|Lecture 3 §3.2]]. Your notes put CLIP's question as "**which caption is best?**", answered with a softmax.

### 2.4 Goal and inference

**Goal:** pull matching pairs together and push non-matching pairs apart.

**Inference:** given an image, the prediction is the caption with the highest similarity:

$$\hat j = \arg\max_j\ v^\top t_j$$

For zero-shot classification the candidate "captions" are prompts built from the class names, as in [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.6 Zero-Shot Inference with CLIP|Lecture 3 §3.6]]. For retrieval they are a database of captions or images. $\tau$ can be dropped at inference because a positive scale doesn't change the argmax.

---

## 3. Geometry of the Shared Space

The loss in [[#^eq-1|eq. (1)]] only cares about dot products of unit vectors, so what it learns is best understood as an arrangement of points on a sphere. The lecture lists two properties of that arrangement and one related effect.

![[03 - Attachments/ATML_L11_geometry.svg|760]]

*(a) Image and text clouds on the unit sphere, separated by the modality gap. (b) Matched pairs pulled together (green) and mismatched pairs pushed apart (red), settling in a local minimum. (c) The cone effect, as narrow cones on the sphere and as the notes' off-centre double ellipsoid.*

### 3.1 Property 1, a cosine-similarity space on the unit sphere

The shared space is a **cosine-similarity space** with **normalized embeddings**. Since $\lVert v\rVert = \lVert t\rVert = 1$, the embeddings lie on the **unit sphere**, and $v^\top t = \cos\varphi$ where $\varphi$ is the angle between them. **Angles, not Euclidean distances, are the similarity measure.** *(Expanding on this: on the unit sphere the two carry the same ordering, since $\lVert v - t\rVert^2 = 2 - 2\cos\varphi$, so a smaller angle means a smaller distance. Normalization removes vector length as a way to cheat the loss: without it, the model could raise $s_{ii}$ just by inflating norms.)*

### 3.2 Property 2, the persistent modality gap

Even after training, the two modalities form **partly separate clouds** on the sphere (panel (a)). The angular separation between them is the **modality gap (MG)**. *("Persistent": training doesn't close it.)* The notes give three causes:

- **Information mismatch:** the information in $x^V$ is much greater than the information in $x^T$. *(A photo contains far more detail than any caption describes, so an image embedding can't sit exactly on top of its caption's.)*
- **Different inductive biases:** $\theta_V$ and $\theta_T$ are different architectures, e.g. a [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding|ViT]] (Vision Transformer) and a text Transformer ([[ATML_Lecture_4 - GANs, and Inductive Biases#5.5 Locality Bias vs. Global Context — CNN vs. ViT|Lecture 4 §5.5]] on how architecture shapes representations). *(So the two encoders map inputs into the space in systematically different ways.)*
- **$L_c(\theta)$ itself promotes the MG**, derived next.

### 3.3 Why the loss promotes the gap, alignment vs uniformity

Take only the image-to-text direction, $L_{i2t}(\theta)$, and write the average over the batch as an expectation. Splitting the logarithm of the ratio into a difference of logarithms and using $\log\exp(s_{ii}) = s_{ii}$:

$$\begin{aligned}
L_c(\theta) &= -\mathbb E\left[\log\frac{\exp(s_{ii})}{\sum_j\exp(s_{ij})}\right]\\
&= -\mathbb E\big[\log\exp(s_{ii})\big] + \mathbb E\Big[\log\sum_j\exp(s_{ij})\Big]\\
&= \underbrace{-\mathbb E[s_{ii}]}_{\text{alignment term}} + \underbrace{\mathbb E\Big[\log\sum_j\exp(s_{ij})\Big]}_{\text{uniformity term}}
\end{aligned}$$

- The **alignment term** maximizes the similarity between $x_i^V$ and $x_i^T$, the matched pair.
- The **uniformity term** maximizes the dissimilarity between $x_i^V$ and every $x_j^T$, **without any constraint within each modality**. It only sees cross-modal pairs, so nothing tells the images to spread among themselves or the texts among themselves.

These two **push the modalities in opposite directions**, so we mostly end up in a **local minimum** (panel (b)). One acceptable solution is two separate clouds: each matched pair is closer than any mismatched pair, which satisfies the loss, even though the clouds never merge. *(Expanding on this: the split is from Wang & Isola (2020), who showed that contrastive losses optimize "alignment" of positives and "uniformity" of the embedding distribution. Liang et al. (2022, "Mind the Gap") traced CLIP's modality gap to the cone effect at initialization ([[#3.5 The cone effect|§3.5]]), preserved by this loss, and showed that the temperature affects how large the gap stays.)*

### 3.4 How big should the gap be

The MG **should not be**:

- **too large**, which indicates **bad alignment**: matched pairs aren't actually close;
- **too small**, where both modalities have become **geometrically identical**, which suggests that **maybe one modality has collapsed**: text-only details or image-only details have been thrown away to make the clouds coincide.

**Better goal:** **shared semantics, with modality-specific details preserved.** The two clouds should agree on meaning while each keeps what only it can express. This links back to the information-mismatch cause in [[#3.2 Property 2, the persistent modality gap|§3.2]].

### 3.5 The cone effect

**Cone effect:** the **effective embedding space is restricted to a narrow cone**, and **different random initializations create different embedding cones** (panel (c)). Inside a cone, the embeddings have **non-zero mean directions**: their average points somewhere rather than at the origin. The notes draw this as an **off-centre double ellipsoid**, one elongated cloud per modality, each displaced from the centre. The resulting **anisotropic geometry** uses only a small patch of the sphere, unevenly in different directions. *(Expanding on this: two encoders initialized independently start in two different cones before any training, so the gap is present at step 0, and per [[#3.3 Why the loss promotes the gap, alignment vs uniformity|§3.3]] the contrastive loss has no term that forces the cones to merge. "Anisotropic" means the distribution is not the same in every direction.)*

---

## 4. Beyond Softmax — SigLIP and LLIP

CLIP's softmax makes every score depend on the whole batch. The professor asks whether there is any other way to match. One answer changes the question, the other changes the image embedding.

### 4.1 SigLIP, matching as binary classification

Instead of "which caption is best", ask of **each pair** "**does this $x^V$ match this $x^T$?**" with answer yes or no. **SigLIP** (Sigmoid Loss for Language–Image Pre-training) **replaces the softmax with a sigmoid**, turning each entry of $S$ into its own binary classification. With true labels

$$y_{ij} = \begin{cases}1 & i = j\\ 0 & \text{otherwise}\end{cases}$$

the loss is the **binary cross-entropy** over all $N^2$ entries:

$$\boxed{L_{\text{SigLIP}}(\theta) = -\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\Big[y_{ij}\log\sigma(s_{ij}) + (1-y_{ij})\log\big(1-\sigma(s_{ij})\big)\Big]}\qquad\qquad\text{(2)}$$
^eq-2

where $\sigma(u) = 1/(1+e^{-u})$ is the sigmoid. The professor's diagram labels this sigmoid block the **Image–Text Matching loss, $L_{\text{ITM}}$**, and your notes write the same formula as $L_{\text{ITM}}$. *(Expanding on this: the SigLIP paper computes the logit as $s_{ij} = t'\,v_i^\top t_j + b$ with a learnable scale and bias, initialized at $t' = 10$ and $b = -10$. [[#Numerical Problems|Problem 4]] shows why the bias matters.)*

![[03 - Attachments/ATML_L11_clip_vs_siglip.svg|760]]

*Left: SigLIP's sigmoid block. Right: the class sketch of zero-shot accuracy against batch size (qualitative; axis 66–80).*

| | CLIP | SigLIP |
|---|---|---|
| Question | "Is this caption **better than the others**?" | "Does this caption **match this image**? Y/N" |
| Normalization | softmax over a row / column | sigmoid per entry |
| Problems per batch | $2N$ problems, each $N$-way | $N^2$ binary problems |
| Loss | $L_c$, [[#^eq-1\|eq. (1)]] | $L_{\text{SigLIP}}$, [[#^eq-2\|eq. (2)]] |

### 4.2 When SigLIP is good, and its costs

**SigLIP is good for small $N$** (the plot above). **But:**

- **Class imbalance:** each batch has $N$ positives and $N^2 - N$ negatives, so almost every label is 0.
- **Reduced mutual information maximization.** *(Expanding on this: InfoNCE lower-bounds the mutual information between the paired views, $I \ge \log N - L$ ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.3 InfoNCE as a Lower Bound on Mutual Information|Lecture 3 §3.3]], [[ATML_Lecture_2 - Information Theory and Divergences#3.5 Mutual Information|MI in Lecture 2]]), and that bound comes from the $N$-way softmax. Independent binary problems don't give this bound, so the objective drives mutual information less directly.)*

Both costs show up in [[#3.3 Why the loss promotes the gap, alignment vs uniformity|§3.3]]'s split. SigLIP keeps an alignment term (the positive entries) but replaces CLIP's shared uniformity term, one log-sum-exp per row, with a separate push on each negative pair: the $N^2 - N$ separate pushes are the imbalance, and losing the shared normaliser is what weakens the mutual-information link.

*(Expanding on why SigLIP helps at small $N$: a softmax over a small batch is an easy $N$-way problem with few negatives, while the sigmoid's per-pair signal doesn't depend on how many other pairs share the row. The per-pair loss also needs no batch-wide normalization, which makes it cheaper to compute across devices. The SigLIP paper reports the sigmoid ahead at small batch sizes and the two roughly level once batches are large, with gains saturating around 32k.)*

### 4.3 LLIP, caption-conditioned image embeddings

**Motivation.** A single image contains a **multitude of visual concepts** that can be described by **many** captions or prompts. A photo of a dog on a beach could be captioned about the dog, the beach or the weather. A single fixed image embedding has to average over all of these. **LLIP** (Latent Language Image Pretraining) instead **makes the image embedding conditioned on the text embedding**.

![[03 - Attachments/ATML_L11_llip.svg|700]]

*The caption's embedding is the query, so each caption gets its own view of the image.*

The architecture inserts one **cross-attention** block ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.9 Attention as Query–Key–Value Retrieval|query–key–value attention]]) after $\theta_V$ with $Q = z_j^T$, so the image representation compared with caption $j$ is built by attending over the visual tokens with that caption's query. **LLIP:**

- **preserves fine visual details**, since a caption about a small object can attend to the tokens that hold it rather than relying on a global summary;
- **handles text ambiguity inherently**, since different captions of one image are allowed to pick out different embeddings of it.

*(Expanding on this: Lavoie et al. (2024) have the visual encoder output a few "mixture" tokens and use the caption to weight them. The cost is that $s_{ij}$ needs an image embedding per caption, so the $N\times N$ matrix needs $N^2$ conditioned embeddings. Each is a cheap weighted sum, but it isn't the single-pass dot product of CLIP.)*

LLIP is the first model in the lecture in which one modality attends to the other. It stops short of fusion ([[#5. Fusion VLMs|§5]]): the attention only shapes the image embedding, and the final comparison is still one dot product per pair.

---

## 5. Fusion VLMs

Representational models compare a whole image with a whole caption, which answers "does this match?" but not "**which object in the image does this word refer to?**". Answering the second question needs the two modalities to interact token by token.

### 5.1 From matching to cross-modal interaction

**Fusion VLMs** **jointly process image and text tokens** to learn **richer cross-modal interactions** than simple embedding alignment. A fusion module sees both token sequences at once, so a word's representation can depend on specific image regions and vice versa. This is what grounding a word in a region requires.

### 5.2 FLAVA's architecture

**FLAVA** (Foundational Language And Vision Alignment) **uses three models**: an image encoder $\theta_V$, a text encoder $\theta_T$ and a multimodal model $\theta_D$, which the notes call a **decoder model**. *(The FLAVA paper calls $\theta_D$ a multimodal **encoder**: a Transformer over the concatenated image and text hidden states.)*

![[03 - Attachments/ATML_L11_flava.svg|780]]

*Two masked streams feed $\theta_D$. The shaded unmasked branches feed $L_{\text{cont}}$ and $L_{\text{ITM}}$ and are the ones kept at inference. The right side lists text above image, the reverse of the notes, to keep the arrows apart.*

Two streams run through the same three models:

- **Masked image, full text** (left): mask patches of $x_i^V$ to get $\tilde x_i^V$, encode it to $\tilde z_i^V$, encode the full caption to $z_i^T$. Given $\tilde z_i^V$ and $z_i^T$, $\theta_D$ fills out $\tilde x_i^V$ to make it closer to $x_i^V$.
- **Full image, masked text** (right): encode the full image to $z_i^V$, mask words of $x_i^T$ to get $\tilde x_i^T$ and encode it to $\tilde z_i^T$. Given $z_i^V$ and $\tilde z_i^T$, $\theta_D$ fills out $\tilde x_i^T$ to make it closer to $x_i^T$.

Both fill-in tasks are **$L_{\text{MMM}}$** (masked multimodal modelling). The image is reconstructed with the help of the text and the text with the help of the image, so $\theta_D$ must learn which words correspond to which regions.

### 5.3 The five-term FLAVA loss

$$\boxed{L_{\text{FLAVA}}(\theta) = \lambda_1L_{\text{cont}} + \lambda_2L_{\text{MIM}} + \lambda_3L_{\text{MLM}} + \lambda_4L_{\text{ITM}} + \lambda_5L_{\text{MMM}}}\qquad\qquad\text{(3)}$$
^eq-3

| Term | Name | What it does | Batch |
|---|---|---|---|
| $L_{\text{cont}}$ | **contrastive** | CLIP's [[#^eq-1\|eq. (1)]] on the unmasked $z_i^V, z_i^T$; focuses on **global synchronization** | image + text pairs |
| $L_{\text{MIM}}$ | **masked image modelling** | predict masked patches from the rest of the image | image-only |
| $L_{\text{MLM}}$ | **masked language modelling** | predict masked words from the rest of the text ([[ATML_Lecture_7 - LLM Foundations#1.1 Three families of Transformer language models\|BERT-style]]) | text-only |
| $L_{\text{ITM}}$ | **image–text matching** | binary "do they match?" on the fused pair | image + text pairs |
| $L_{\text{MMM}}$ | **masked multimodal modelling** | mask the image and predict it using the text, and vice versa | image + text pairs |

The $\lambda_k$ weight the terms. The three batch types matter: unimodal data (images without captions, text without images) is far more plentiful than paired data, and $L_{\text{MIM}}$, $L_{\text{MLM}}$ let FLAVA use it. Your notes list the paired-data losses as (1) $L_{\text{CLIP}}$ on $z_i^V, z_j^T$, (2) $L_{\text{ITM}}$ on the same pair, and (3) fill-out given $\tilde z_i^V$ and $z_i^T$. *(Expanding on this: in FLAVA, ITM is a classifier on $\theta_D$'s fused output that predicts whether the pair is matched, with mismatched pairs made by swapping captions. It asks SigLIP's yes/no question ([[#4.1 SigLIP, matching as binary classification|§4.1]]), but after the modalities have interacted.)*

### 5.4 Inference and fine-tuning

At **inference**, the unmasked branches are used (green in the figure): the full image through $\theta_V$ and the full text through $\theta_T$. Masking is a training device only. For a downstream task, your notes draw the fused model feeding a task head:

![[03 - Attachments/ATML_L11_flava_head.svg|440]]

*Your notes' fine-tuning picture: the fused model's latent feeds a new task head.*

Your notes mark $\theta_D$ with a star, their convention (also used in [[#7.5 Training stages|§7.5]]) for a pre-trained module that is kept fixed; the figure draws it dashed. A new task head is trained on its latent, as in [[ATML_Lecture_10 - Fine-Tuning#6.1 New task head|Lecture 10 §6.1]].

---

## 6. Generative VLMs

Fusion models produce representations; a task head still has to turn them into an answer. **Generative-based VLMs** consider **explicit generation of text and/or images**. The notes give three branches.

![[03 - Attachments/ATML_L11_generative_vlms.svg|780]]

*(a) CoCa. (b) Chameleon and CM3leon, with a break token between image and text tokens. (c) The third branch, as far as the notes draw it.*

### 6.1 CoCa

**CoCa** (Contrastive Captioners) combines a representational and a generative objective. $\theta_V$ encodes $x^V$ and $\theta_T$ encodes $x^T$, with a contrastive loss $L_{\text{cont}}$ between them. A **multimodal text decoder** then **cross-attends** to the image features and produces **generated text**. *(Expanding on this: Yu et al. (2022) split the text decoder into unimodal layers, which give the text embedding for the contrastive loss, and multimodal layers, which cross-attend to the image and are trained with a captioning loss: [[ATML_Lecture_7 - LLM Foundations#4.1 Cross-entropy at one position|next-token cross-entropy]] on the caption. The total is $\lambda_{\text{con}}L_{\text{cont}} + \lambda_{\text{cap}}L_{\text{cap}}$.)* Like FLAVA ([[#5.3 The five-term FLAVA loss|§5.3]]), CoCa keeps CLIP's contrastive loss and adds a second objective; FLAVA's second objective fills in masked inputs, while CoCa's generates the caption, which is what makes CoCa generative.

### 6.2 Chameleon and CM3leon

$\theta_V$ plus an **image tokenizer/projector** turns $x^V$ into **discrete image tokens**, and a **text tokenizer** ([[ATML_Lecture_7 - LLM Foundations#2.1 Tokenization and special tokens|Lecture 7 §2.1]]) turns $x^T$ into text tokens. The two are concatenated into one sequence with a **break token** between them. A single **mixed-model autoregressive LM** ([[ATML_Lecture_7 - LLM Foundations#1.2 Autoregressive (decoder-only) models|Lecture 7 §1.2]]) predicts the next token whatever its modality. Generated text tokens are the **text output**; generated image tokens go through an **image detokenizer** to give the **image output**. *(Expanding on this: Chameleon (Meta, 2024) tokenizes a $512\times512$ image into 1024 tokens from an 8192-entry codebook, so images and text share one vocabulary and one Transformer; this is called **early fusion**. CM3leon (2023), "chameleon", is the earlier token-based model of the same kind. CM3 stands for Causally-Masked Multimodal Modelling.)* The notes put an asterisk on the mixed-model AR LM box without explaining it. How an image becomes discrete tokens in the first place is the subject of [[#10. Modality Unification — Images as Discrete Tokens|§10]].

### 6.3 Stable Diffusion and Imagen

The third branch lists **Stable Diffusion** and **Imagen**, drawn with an **AR (autoregressive) for text** block giving the **text output** and a **diffusion** block giving the **image output**.

> [!warning] Unclear in source
> The branch is drawn mostly with ditto marks, so its structure is only partly given. Stable Diffusion and Imagen themselves output only images. They are text-to-image **diffusion** models: an image is generated by iteratively denoising random noise, conditioned on a frozen text encoder (CLIP's text encoder for Stable Diffusion, T5, Google's text-to-text Transformer, for Imagen). Neither generates text. The "AR for text + diffusion for images" pair is read here as the general hybrid recipe, in which text is produced autoregressively and images by diffusion, rather than as a description of those two models.

**Backbones can be re-trained or frozen as well.** Any of these designs can start from pre-trained encoders and keep them fixed, which is the idea [[#7. Adaptive Modularity|§7]] turns into a recipe.

---

## 7. Adaptive Modularity

Training a generative VLM from scratch is expensive, and strong pre-trained vision encoders and LLMs already exist. **Adaptive modularity** connects them: keep a pre-trained LLM, add a vision encoder, and train a small **bridge** that turns visual features into something the LLM can read.

### 7.1 The pipeline

Text goes through the LLM's own **tokenizer** and [[ATML_Lecture_7 - LLM Foundations#2.2 Embedding lookup|embedding lookup]] to give $z^T$, one $d$-dimensional vector per token ($d\times1$ in your notes). The image goes through a **vision encoder** $\theta_V$ to give $z^V$. A **projection/bridging** module maps $z^V$ to $\tilde z^V$ in the LLM's input space. The LLM then reads $\tilde z^V$ alongside $z^T$ and outputs **text**. The LLM is **pre-trained and frozen**; **SFT ([[ATML_Lecture_10 - Fine-Tuning#3.1 Supervised fine-tuning|supervised fine-tuning]]), however, trains it**. A dotted arrow from $z^T$ into the bridge marks that the text can also steer the bridge.

![[03 - Attachments/ATML_L11_adaptive_modularity.svg|800]]

*Top: the pipeline; the dotted arrow lets the text steer the bridge. Bottom: the three bridges, by how many visual tokens reach the LLM.*

Your notes add the details:

- **$\theta_V$ is a CLIP-ViT used without any readout.** Instead of keeping only the [CLS] summary ([[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding|Lecture 3 §2.6]]), the bridge **uses all embeddings**: $z^V$ is an $(N_p+1)\times d_v$ matrix, $N_p$ patch tokens plus the [CLS] token. *(Your notes write $(N+1)\times d$; $N_p$ is used here so it isn't confused with the batch size $N$ of [[#2. Representational VLMs|§2]].)* Keeping every patch token preserves *where* things are, which a single pooled vector would lose.
- Each component maps to an ability: **$\theta_V$ ↔ perception**, **connector ↔ image/text alignment**, and **LLM ↔ grounding and composition**. [[#8.2 Which component fails|§8.2]] uses this map.

The bridge comes in three designs (MLP = multi-layer perceptron), compared below and detailed in [[#7.2 Bridge 1, linear or MLP projector|§7.2]]–[[#7.4 Bridge 3, gated cross-attention through an adapter|§7.4]].

| Bridge | How visual tokens enter the LLM | Tokens passed | LLM | Examples |
|---|---|---|---|---|
| Linear / MLP | projected token by token, prepended to the text | all $N_p+1$ | frozen in stage 1, trained in stages 2–3 ([[#7.5 Training stages\|§7.5]]) | LLaVA-NeXT, Mistral, Qwen2.5 |
| Bottleneck resampler | learnable queries compress them with a Q-Former | few ($K \ll N_p$) | frozen | BLIP family, MiniGPT-4 |
| Gated cross-attention | inserted **inside** LLM layers by cross-attention | none in the input sequence | frozen; train only the adapters | Llama 3.2, Flamingo |

### 7.2 Bridge 1, linear or MLP projector

A **linear map or a small MLP** is applied to each visual token separately, $\tilde z = W_2\,\sigma(W_1z)$ for a two-layer MLP with non-linearity $\sigma$. The projected tokens are placed in the LLM's input sequence next to the text tokens. Nothing is compressed: $N_p + 1$ visual tokens go in. Examples: **LLaVA-NeXT**, **Mistral**, **Qwen2.5**. *(Expanding on this: LLaVA (Large Language and Vision Assistant) uses a two-layer MLP from CLIP ViT-L/14 features into the LLM; at $336\times336$ pixels that is $24\times24 = 576$ visual tokens per image. "Mistral" and "Qwen2.5" refer to their vision variants, e.g. Pixtral and Qwen2.5-VL.)*

### 7.3 Bridge 2, bottleneck resampler (Q-Former)

A **Q-Former** (Querying Transformer) is used to **query $\theta_V$**. A small set of **learnable queries** cross-attends to the visual features, optionally together with the text $z^T$, and outputs a **small number of highly informative tokens**. The LLM reads these instead of all $N_p + 1$. Examples: the **BLIP family**, **MiniGPT-4**. The number of queries is how many embeddings we choose to represent the image. *(Expanding on this: BLIP-2 (Bootstrapping Language–Image Pre-training) uses 32 queries, so every image costs the LLM 32 tokens however many patches the encoder produced. [[#Numerical Problems|Problem 8]] measures the saving.)* The Q-Former uses the same mechanism as LLIP's cross-attention ([[#4.3 LLIP, caption-conditioned image embeddings|§4.3]]), visual tokens as keys and values, but its queries are learned vectors rather than a caption, so unless $z^T$ is fed in it summarises an image the same way for every prompt.

### 7.4 Bridge 3, gated cross-attention through an adapter

Here **visual features are inserted inside the LLM through cross-attention** layers placed between the LLM's own layers. The **queries come from the text** (the LLM's hidden states) and the **keys and values from vision**. The design rule is to **keep the LLM frozen and train only the adapter layers**, which is the adapter idea of [[ATML_Lecture_10 - Fine-Tuning#7.5 Adapters|Lecture 10 §7.5]]. Examples: **Llama 3.2**, **Flamingo**.

> [!warning] Unclear in source
> The line "# of queries = how many embeddings we choose to represent the image" sits under this column in the notes. In a gated cross-attention adapter, the queries come from the text, so their number is set by the text length. The line fits the Q-Former's learnable queries ([[#7.3 Bridge 2, bottleneck resampler (Q-Former)|§7.3]]), and is placed there.

*(Expanding on "gated": Flamingo multiplies each new cross-attention output by $\tanh(\alpha)$ with a learnable scalar $\alpha$ initialized to 0, $\;y = h + \tanh(\alpha)\,\text{xattn}(h, Z^V)$. At the start $\tanh(0) = 0$, so the frozen LLM behaves exactly as before, and the visual pathway is opened gradually during training. This is the same trick as LoRA's $B = 0$ initialization in [[ATML_Lecture_10 - Fine-Tuning#8.5 Initialization, forward pass and gradients|Lecture 10 §8.5]]: the new branch adds nothing at the start, so training begins from the pre-trained model.)*

### 7.5 Training stages

The stack is trained in **three stages**. Your ATML notes list them with $^*$ marking a **frozen** module, and the professor's [[VLM Notes.pdf|VLM Notes]] (page 8) draw the same three with a tick (trained) or a cross (frozen) beside each module:

![[03 - Attachments/ATML_L11_training_stages.svg|565]]

*Top: which modules train (green) and which stay frozen (dashed) in each stage. Bottom: the professor's summary box of the stack.*

| Stage | Data | Connector | LLM | Vision encoder |
|---|---|---|---|---|
| 1. Connector initialization | image–text pairs *(added)* | **train** | frozen$^*$ | frozen$^*$ |
| 2. SFT | SFT data (VQA-like) | **train** | **train** | frozen$^*$ |
| 3. Visual instruction fine-tuning | visual instruction data | **train** | **train** | frozen$^*$ |

- **Stage 1 (connector initialization)** trains only the connector, so it learns to map visual features to where the frozen LLM expects its inputs. This is the alignment ability of [[#7.1 The pipeline|§7.1]].
- **Stage 2 (SFT)** unfreezes the LLM and trains it together with the connector by [[ATML_Lecture_10 - Fine-Tuning#3.1 Supervised fine-tuning|supervised fine-tuning]] on **VQA-like** data: an image, a question about it and its answer. This is the "SFT, however, trains it" of the professor's diagram.
- **Stage 3 (visual instruction tuning)** trains the same modules as stage 2. What changes is the data.

Your notes ask **what type of data** visual instruction fine-tuning uses. *(Answer, expanding on the notes: image-grounded instruction–response pairs, such as multi-turn conversations about an image, detailed descriptions and reasoning questions. In LLaVA these were generated by a text-only LLM from each image's captions and object boxes. Stage 2's VQA-style data has short, fixed-format answers ("two", "red"). Stage 3 teaches the model to follow open-ended instructions and answer at length. In both, the loss is the [[ATML_Lecture_7 - LLM Foundations#4.3 Batching, padding and the loss mask|masked next-token cross-entropy]] on the response tokens only. In terms of [[#9.5 Representational vs interface alignment|§9.5]], stage 1 typically uses the image-grounded loss $L_{\text{IT}}$ (caption the image) and stages 2–3 the image–text-grounded $L_{\text{ITT}}$ (answer the prompt about the image). Some published recipes, such as LLaVA-1.5, merge stages 2 and 3 into one fine-tuning stage on a mixture of both kinds of data.)*

In the terms of [[ATML_Lecture_10 - Fine-Tuning#2.2 The overall objective|Lecture 10 §2.2]], each stage chooses a different feasible set $S$: which modules may change at all. The vision encoder stays frozen throughout, which is one of the limitations in [[#9.6 Limitations of the stack|§9.6]].

---

## 8. VLM Failure Modes

With the architecture in place, the lecture closes by listing how VLMs go wrong. The categories line up with the components of [[#7. Adaptive Modularity|§7]].

### 8.1 Five failure types

1. **Perception failure:** the model does not extract the right visual evidence. *Example:* missing a small object, misreading fine text.
2. **Grounding failure:** the model has the relevant visual evidence but attaches the question or phrase to the wrong object or region. *Example:* "the cup on the left" maps to the right cup; or it maps to the left object, which is not a cup.
3. **Structure/composition failure:** the model cannot preserve **object relations, counts or composed attributes**: *who has what? who is where relative to whom?* *Example:* "red", "cube" and "left of" are each known correctly, but "red cube left of the blue sphere" comes out wrong.
4. **Reasoning failure:** the model has the pieces of evidence but uses them in the wrong sequence.
5. **Trustworthiness failure:**
   - **hallucination:** the model states details not supported by the image;
   - **non-robustness:** performance drops when the image is blurry, noisy, etc.

### 8.2 Which component fails

Your notes' map from [[#7.1 The pipeline|§7.1]] assigns these to components:

| Component | Ability | Failure when it breaks |
|---|---|---|
| Vision encoder $\theta_V$ | perception | perception failure |
| Connector | image/text alignment | the LLM receives the right evidence in a form it can't relate to the words |
| LLM | grounding (one object and one query, e.g. "the cup is on the left side of the bottle") | grounding failure |
| LLM | composition (e.g. "red cube left of blue sphere": who has / is what, who is where relative to whom) | structure/composition failure |

*(Expanding on this: the map explains some design choices above. A pooled single-vector image embedding ([[#2. Representational VLMs|§2]]) loses the "where" information that grounding and composition need, which is why the bridge keeps all patch tokens ([[#7.1 The pipeline|§7.1]]) and why LLIP ([[#4.3 LLIP, caption-conditioned image embeddings|§4.3]]) and fusion models ([[#5. Fusion VLMs|§5]]) let the text attend to regions. A large modality gap or poor alignment ([[#3. Geometry of the Shared Space|§3]]) is one route to hallucination: the LLM fills in what the caption-like part of the embedding suggests rather than what the image shows.)*

---

## 9. Inside the Modular VLM — Vision Encoding and Alignment

The failure map of [[#8.2 Which component fails|§8.2]] blamed perception on the vision encoder and alignment on the connector. The professor's [[VLM Notes.pdf|VLM Notes]] (pages 6–11) open those two boxes: what the encoder computes, what "alignment" is trained to mean, and why the whole stack still falls short.

### 9.1 The pipeline and its two jobs

The professor redraws the [[#7.1 The pipeline|§7.1]] pipeline with the kind of token made explicit:

![[03 - Attachments/ATML_L11_vlm_pipeline.svg|640]]

*Text enters as discrete IDs and the image as continuous embeddings through the connector; dashed = frozen, green = trained.*

The **text tokens** are **discrete token IDs**, which the embedding turns into $U$. The **image/visual tokens** are **continuous embeddings** that come straight out of the encoder, so the connector maps them into $V$. $U$ and $V$ are the $z^T$ and $\tilde z^V$ of [[#7.1 The pipeline|§7.1]]. Your notes add the shapes, $Z^T\in\mathbb R^{T\times d}$ for $T$ text tokens and $Z^V\in\mathbb R^{T'\times d}$ for $T'$ visual tokens. They also add a legend: a black $^*$ marks a module that **remains frozen** (the vision encoder and the LLM) and a red $^*$ one that **is trained** (the projector, an MLP or a Q-Former). The vision encoder is labelled a **bidirectional encoder model**.

The pipeline has two jobs, which the rest of this section takes in turn:

1. **Vision encoding** ([[#9.2 The ViT vision encoder|§9.2]]–[[#9.4 Why a VLM keeps every patch token|§9.4]]). The **goal** is to **identify objects + attributes + relations (spatial)**. Your notes put it as "what is happening in $x^V$?": objects, their attributes and their relationships, e.g. circles and the colour of the circles. It is done by a **CNN or a ViT**, and the **ViT has become the de facto** choice.
2. **Vision–text alignment** ([[#9.5 Representational vs interface alignment|§9.5]]).

### 9.2 The ViT vision encoder

This is the ViT of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding|Lecture 3 §2.6]], in this lecture's notation. The professor writes $N$ for the number of patches; it is $N_p$ here, as in [[#7.1 The pipeline|§7.1]], to keep it apart from the batch size.

![[03 - Attachments/ATML_L11_vit_encoder.svg|820]]

*Top: from image to $Z_0$. Bottom: a classifier reads out one row, while a VLM passes every row to the connector.*

- **Patchification.** The image $x^V\in\mathbb R^{H\times W\times C}$ is cut into patches of **patch size** $P$, giving $N_p = HW/P^2$ patches. Each is flattened to $x_i\in\mathbb R^{P^2C\times1}$, $i = 1, \dots, N_p$.
- **Patch embedding.** A shared linear map gives $e_i = E\,x_i + b$, with $E\in\mathbb R^{d\times P^2C}$, $b\in\mathbb R^{d\times1}$ and $e_i\in\mathbb R^{d\times1}$. This is **the same as a convolution layer with kernel size $P$, stride $P$ and $d$ output channels**. *(Expanding on this: with stride equal to kernel size, each output position of the convolution sees exactly one patch and no patch is seen twice. Each of the $d$ filters is a $P\times P\times C$ block of weights, which is one row of $E$ reshaped. [[#Numerical Problems|Problem 12]] checks that the two parameter counts match.)*
- **[CLS] token.** A learnable token $e_{\text{CLS}}$ is prepended to $\{e_i\}$. It exists because **many token vectors must lead to one prediction**: the classifier reads its final state.
- **Positional embedding.** $Z_0 = [e_{\text{CLS}}, e_i] + E_{\text{pos}}$, of size $(N_p+1)\times d$. Attention ignores order, so position has to be added explicitly.
$Z_0$ then passes through **$L$ Transformer layers**. For $\ell = 1, \dots, L$:

$$z'_\ell = z_{\ell-1} + \text{MHA}\big(\text{LN}(z_{\ell-1})\big),\qquad z_\ell = z'_\ell + \text{MLP}\big(\text{LN}(z'_\ell)\big)$$

where LN is LayerNorm and MHA multi-head attention, with $\text{Attn}(X) = \text{softmax}\big(QK^\top/\sqrt{d_h}\big)V$ and **no causal mask**: every patch attends to every other, in every layer. That is what "bidirectional" means, in contrast to the causal LLM of [[ATML_Lecture_7 - LLM Foundations#3.3 Multi-head causal self-attention|Lecture 7 §3.3]]. *(Here LayerNorm comes before each sub-block ("pre-LN"), which is what ViT uses. [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.7 The Transformer Layer|Lecture 3 §2.7]] wrote it after; both orders are in use, and pre-LN trains more stably in deep stacks.)*

The output is $Z_L\in\mathbb R^{(N_p+1)\times d}$. Its first row, $z_L^{(0)}$, is the **final hidden state of the [CLS] token**; rows $z_L^{(1)}, \dots, z_L^{(N_p)}$ are the final states of the patches.

### 9.3 Read-out for classification

A plain ViT classifier needs one vector, so it ends in a **classification read-out**:

$$\hat y = W_h\,\text{LN}\big(z_L^{(0)}\big) + b_h,\qquad \hat y\in\mathbb R^{K\times1},\qquad \text{Loss} = -\log p\big(y\mid x^V\big)$$

where $K$ is the number of classes (in this subsection only; from [[#10. Modality Unification — Images as Discrete Tokens|§10]] on, $K$ is the codebook size) and $p$ is the softmax of $\hat y$, so the loss is the cross-entropy of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.3 From $H(Y \mid z)$ to Cross-Entropy and Negative Log-Likelihood|Lecture 3 §2.3]]. The single vector can be made in several ways:

1. **Use only $z_L^{(0)}$**, the [CLS] embedding, which contains a summary of all tokens' embeddings. It **provides a single global summary**.
2. **Mean pooling:** $\bar z = \frac1{N_p}\sum_{i=1}^{N_p} z_L^{(i)}$.
3. **Max pooling** *(the element-wise maximum over the patch rows)*.
4. **Attention pooling**, etc. *(Expanding on this: a learned query vector attends over the patch rows, and the softmax weights decide how much each patch contributes. [[#Numerical Problems|Problem 13]] computes all three pools on a small example.)*

### 9.4 Why a VLM keeps every patch token

**However, for using a ViT in a VLM we need fine-grained details**: "what is on the left of the cup?", "how many objects are on the table?" etc. **A single vector like $z_L^{(0)}$ tends to lose these spatial details.** So the VLM **uses all patch tokens**: the whole $Z_L$, $(N_p+1)\times d$, goes to the connector, and

$$V = \text{Connector}(Z_L)$$

with the connector being (1) an **MLP-style projector** ([[#7.2 Bridge 1, linear or MLP projector|§7.2]]), (2) a **Q-Former** ([[#7.3 Bridge 2, bottleneck resampler (Q-Former)|§7.3]]), etc. This is the "ViT without any readout" of [[#7.1 The pipeline|§7.1]]. It also explains a limit of [[#2. Representational VLMs|§2]]: a representational VLM compares one pooled vector per image, so it loses the same spatial detail, which is part of why grounding ([[#8.1 Five failure types|§8.1]]) needs more than a dual encoder. Your notes write it as $\tilde Z^V = Z_L \to \boxed{\text{Proj}} \to Z^V$, the **visual tokens**.

### 9.5 Representational vs interface alignment

The **connector translates visual features into "visual tokens"** that **trick the LLM into treating them as text**. The notes distinguish two kinds of alignment.

**Representational alignment:** a **matched image and text should have nearby representations**. Your notes' example is a cat image and its label. The losses are the image–text contrastive loss of [[#^eq-1|eq. (1)]], $L_{\text{cont}}(\theta) = \tfrac12\big[L_{i\to t}(\theta) + L_{t\to i}(\theta)\big]$, **and/or** the image–text matching loss $L_{\text{ITM}}(\theta)$ of [[#4.1 SigLIP, matching as binary classification|§4.1]] and [[#5.3 The five-term FLAVA loss|§5.3]].

**Interface alignment:** **visual embeddings become tokens for the LLM**. Two next-token losses train it. In **image–text-grounded text generation**, the LLM generates answer tokens conditioned on both $x^T$ and $V$:

$$\boxed{L_{\text{ITT}}(\theta) = -\sum_t \log P_\theta\big(y_t \mid y_{<t},\, x^T,\, V\big)}\qquad\qquad\text{(4)}$$
^eq-4

In **image-grounded text generation** there is no text prompt, only the image:

$$\boxed{L_{\text{IT}}(\theta) = -\sum_t \log P_\theta\big(y_t \mid y_{<t},\, V\big)}\qquad\qquad\text{(5)}$$
^eq-5

Your notes write $Z^V$ and $Z^T$ for $V$ and $x^T$ and group both as the **interface alignment loss**. The professor's summary box puts the stack together: $x^T\to\text{TE}\to U$ (TE = text embedding, VE = vision encoder) and $x^V\to\text{VE}\to\text{connector}\to V$, with the LLM modelling $P(y_t\mid y_{<t}, U, V)$.

The difference matters. Representational alignment asks for two embeddings to be **close**. Interface alignment only asks that the LLM can **use** $V$ to predict the right words; $V$ need not lie near any word embedding. *(Expanding on this: [[#^eq-5|eq. (5)]] is captioning, which suits stage 1 of [[#7.5 Training stages|§7.5]], and [[#^eq-4|eq. (4)]] is answering a prompt about the image, which suits stages 2 and 3. Both are the next-token cross-entropy of [[ATML_Lecture_7 - LLM Foundations#4.1 Cross-entropy at one position|Lecture 7 §4.1]], summed over the answer tokens $y_t$ only.)* The professor's notes then draw the three training stages, which are covered in [[#7.5 Training stages|§7.5]]. In practice the two kinds stack: the vision encoder is often a CLIP-ViT ([[#7.1 The pipeline|§7.1]]), already trained with representational alignment ([[#2. Representational VLMs|§2]]), and the connector then adds interface alignment on top.

### 9.6 Limitations of the stack

The professor lists three **limitations** of the vision encoder + connector + LLM stack, and your notes give a matching three:

1. **The LLM is text-trained**, and visual tokens turned into text-like tokens are **not sufficient to bridge the gap**. *(The LLM learned its weights on text alone; projected image features are a foreign input it was never pre-trained on.)*
2. **Where should fusion happen?** At the **token level**, at the **feature level**, or elsewhere (the notes leave a "?"). Your notes put this point as "**no guideline for the projector**": nothing in the design says what the connector's output should look like. *(Expanding on the "?": fusion can happen early, at the token level, with image tokens in the same sequence as text ([[#10. Modality Unification — Images as Discrete Tokens|§10]], Chameleon in [[#6.2 Chameleon and CM3leon|§6.2]]); in the middle, at the feature level, through cross-attention inside the layers ([[#7.4 Bridge 3, gated cross-attention through an adapter|§7.4]], FLAVA in [[#5. Fusion VLMs|§5]]); or late, by only comparing final embeddings (CLIP, [[#2. Representational VLMs|§2]]).)*
3. **The vision encoder is frozen**, so **all the burden is on the connector**. Your notes: **the vision encoder has only seen images**, so its features were never shaped by language. *(If the encoder misses a detail, no connector can recover it: this is the perception failure of [[#8.1 Five failure types|§8.1]].)*

All three come from joining two models that were trained apart. Modality unification removes the join.

---

## 10. Modality Unification — Images as Discrete Tokens

### 10.1 The idea

**Can we turn images into discrete tokens?** If an image becomes a sequence of IDs from a fixed vocabulary, just as text does, then **one unified loss function, next-token prediction, covers generating both text and images**. No connector is needed, and the LLM is trained on images from the start, which answers limitations 1 and 3 of [[#9.6 Limitations of the stack|§9.6]]. This is how the Chameleon-style models of [[#6.2 Chameleon and CM3leon|§6.2]] work; this section explains their image tokenizer.

### 10.2 The image tokenizer

An **image tokenizer** maps $x^V$ to $q_1, \dots, q_M$, the **discrete token IDs of the visual tokens**, with $q_i\in\{1, 2, \dots, K\}$. Inside it are three parts:

- an **encoder** $\mathcal E$ that **embeds $x^V$ into a latent**, one **latent patch feature** $h_i$ per position;
- a **codebook** $C = \{c_1, c_2, \dots, c_K\}$, $c_i\in\mathbb R^{d\times1}$, where **$K$ is the number of discrete codes** and **$d$ is the dimension of each code vector**;
- a **decoder** $D$ that **turns image tokens back into pixels**, $\tilde x^V$, using each token's vector from the codebook.

The quantizer **picks the nearest codebook entry** $c_k$ to $h_i$ and **returns its code index**:

$$q_i = \arg\min_{k\in\{1,2,\dots,K\}} \lVert h_i - c_k\rVert_2^2$$

The **code index $q_i$ becomes the image token**, which goes on to the LLM. Your notes name this model a **VQ-VAE (Vector-Quantized Variational AutoEncoder)**, with $z$ the **original latent**, $z^q$ the **quantized latent** that goes to the decoder, and the indices $q$ **saved separately**. *(Expanding on this: VQ-VAE (van den Oord et al., 2017) replaces the Gaussian latent of the [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.2 Prior, Posterior, and the VAE Architecture|VAE of Lecture 3]] with a choice among $K$ learned vectors. The encoder is deterministic, so there is no sampling step and no KL term to train.)*

On the page titled "Three Families of Image Tokenizers", the professor writes the same pipeline for a whole feature map, and this notation is used from here on. The input $x\in\mathbb R^{H\times W\times C}$ goes through $\mathcal E$ to a **continuous latent** $z\in\mathbb R^{h\times w\times d}$. This is **discretized** with the codebook $\{e_1, \dots, e_K\}$, $e_k\in\mathbb R^{d\times1}$, into the **discrete latent** $z^q$, and a generator $G$ (the decoder) maps $z^q$ to $\hat x$. For each position, $z_{ij}\in\mathbb R^{d\times1}$:

$$\boxed{q_{ij} = \arg\min_{k\in\{1,\dots,K\}} \lVert z_{ij} - e_k\rVert_2^2,\qquad z^q_{ij} = e_{q_{ij}}}\qquad\qquad\text{(6)}$$
^eq-6

For each $z_{ij}$ we **choose the nearest code**. The index $q_{ij}$ is the **token ID**, and $z^q\in\mathbb R^{h\times w\times d}$ goes **to the decoder**. ($h_i, c_k$ on the earlier pages are the same objects as $z_{ij}, e_k$ here.) Your notes draw $\mathcal E$ as a **CNN**: $x^V$ ($H\times W\times C$) becomes a final feature map $z$ ($h\times w\times d$), whose depth $d$ is the number of kernels in the last layer. Your notes call the lookup a **nearest-neighbour problem** that returns both $q_{ij}$ and $e_{q_{ij}}$.

![[03 - Attachments/ATML_L11_vq_tokenizer.svg|820]]

*The tokenizer during training; the index grid $q$ is what the LLM later sees.*

> [!warning] Unclear in source
> The page is headed "Three Families of Image Tokenizers", but it describes only one, vector quantization against a codebook. The three families listed at the end of the notes ([[#10.7 Other families beyond raster-scan autoregression|§10.7]]) are families of image *generators* that use discrete tokens, not tokenizers. The heading may have been meant for content the notes don't include.

### 10.3 Training the tokenizer, five losses

The tokenizer is **trained on an image dataset**, which gives $\mathcal E$, $D$ and the codebook. The professor's notes and yours give five losses:

| # | Loss | Formula | What it does |
|---|---|---|---|
| 1 | **Reconstruction** | $L_{\text{rec}} = \lVert x - \hat x\rVert_2^2$ | the decoded image should match the input pixel by pixel |
| 2 | **Codebook** | $L_{\text{codebook}} = \lVert \text{sg}(z) - z^q\rVert_2^2 = \sum_{i=1}^h\sum_{j=1}^w \lVert \text{sg}(z_{ij}) - z^q_{ij}\rVert_2^2$ | **updates only the codebook vectors**: the selected codebook entry is **brought closer** to the encoder output |
| 3 | **Commitment** | $L_{\text{commitment}} = \beta\,\lVert z - \text{sg}(z^q)\rVert_2^2$ | **updates only the encoder**: **forces the encoder output $z_{ij}$ to stay close to its assigned codebook vector**, so **$z_{ij}$ should commit to one code**. **$\beta$ is an important parameter** |
| 4 | **Perceptual** | $L_{\text{perceptual}} = \sum_\ell \lVert \phi_\ell(x) - \phi_\ell(\hat x)\rVert_1$ | compare images in the features of the **$\ell$-th layer of VGG** rather than in pixels |
| 5 | **Adversarial** | $L_{\text{adversarial}} = -\log \text{Disc}(x, \hat x)$ | a **discriminator** answers Y/N to "is this the real $x$ or the reconstructed $\hat x$?" |

**VGG** is a CNN image classifier (from Oxford's Visual Geometry Group) whose intermediate features $\phi_\ell$ serve as a measure of perceptual similarity. $\text{sg}(\cdot)$ is the **stop-gradient**: its argument is treated as a constant during backpropagation, so no gradient flows through it. That is how losses 2 and 3 can contain the same distance but update different things. In loss 2, $z$ is frozen, so only the code moves towards the encoder output. In loss 3, $z^q$ is frozen, so only the encoder output moves towards its code. *(Expanding on this: the code update works like one step of k-means, moving each code towards the encoder outputs assigned to it. Without the commitment term, the encoder outputs can keep drifting away from the codes, or keep switching between codes, faster than the codes can follow. A typical value is $\beta = 0.25$, which lets the codes move faster than the encoder. The $\arg\min$ in [[#^eq-6|eq. (6)]] has no gradient, so VQ-VAE uses the **straight-through estimator**: on the backward pass, the gradient arriving at $z^q$ is copied unchanged onto $z$, which is how $L_{\text{rec}}$ reaches the encoder.)*

For the adversarial loss, the professor notes that **when $\hat x = x$, the discriminator is maximally confused**: it can do no better than guess, outputting 0.5. This is the GAN game of [[ATML_Lecture_4 - GANs, and Inductive Biases#1.3 The Min-Max Game|Lecture 4 §1.3]], with the tokenizer as the generator. The perceptual loss is the one introduced in [[ATML_Lecture_2 - Information Theory and Divergences#4.1 The Two Families of Losses|Lecture 2 §4.1]]. *(Expanding on why there are five: VQ-VAE used only losses 1–3, and its reconstructions were blurry, the same averaging problem as the [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#5.6 Failure Modes: Posterior Collapse and Blurry Samples|VAE's blurry samples]]. VQ-GAN (Esser et al., 2021) added the perceptual and adversarial losses so that the decoded images keep sharp textures. These become the image tokens an LLM can later generate.)*

> [!warning] Possible source inconsistency
> 1. The professor writes the perceptual loss with an $L_1$ norm, $\lVert\cdot\rVert_1$. Your notes write a squared $L_2$ norm, $\lVert\cdot\rVert_2^2$, as does [[ATML_Lecture_2 - Information Theory and Divergences#4.1 The Two Families of Losses|Lecture 2]]. Both are used in practice; the table follows the professor.
> 2. The professor's notes write "(NTP with teacher forcing)" under "Train tokenizer". The tokenizer is trained with the five reconstruction-style losses above, not next-token prediction. Your notes attach NTP with teacher forcing to the **LLM** instead ([[#10.4 Image understanding with token IDs|§10.4]]), which is where it belongs.

### 10.4 Image understanding with token IDs

Once the tokenizer is trained, a VLM for **image understanding** uses only $\mathcal E$ and the codebook. An image becomes a short list of **token IDs**, e.g. $[3, 7, 7, 2]$, and these go to the LLM together with the text tokens. **No projector or connector is needed.** The IDs are **not human-interpretable**: code 7 is just whatever the training happened to put in entry 7.

The **combined token stream**, written $V_{1:T'}$, marks the image with special tokens ([[ATML_Lecture_7 - LLM Foundations#2.1 Tokenization and special tokens|Lecture 7 §2.1]]), $\langle\text{boi}\rangle$ (begin of image) and $\langle\text{eoi}\rangle$ (end of image):

$$V_{1:T'} = \big[\langle\text{boi}\rangle,\ 3,\ 7,\ 7,\ 2,\ \langle\text{eoi}\rangle,\ x_1, \dots, x_T\big]$$

Your notes write the sequence as $[\langle\text{boi}\rangle, q, \langle\text{eoi}\rangle, Z^T, \langle\text{eos}\rangle]$, ending in the end-of-sequence token. They add that the **LLM needs to be retrained**, by next-token prediction with [[ATML_Lecture_7 - LLM Foundations#4.2 Worked example and teacher forcing|teacher forcing]], because there is a **different vocabulary for image + text**. *(Expanding on this: the LLM's vocabulary grows by $K$ entries, and the embedding rows and output-layer rows for those $K$ codes start untrained. [[#Numerical Problems|Problem 15]] counts them.)*

![[03 - Attachments/ATML_L11_unified_tokens.svg|820]]

*(a) Understanding: one stream of IDs into the LLM. (b) Generation: IDs looked up in the codebook and decoded to pixels.*

### 10.5 Image generation, from token IDs back to pixels

Understanding and generation use the two halves of the same tokenizer: understanding needs only $\mathcal E$ and the codebook, to turn pixels into IDs, and generation needs only the codebook and $D$, to turn IDs into pixels. **When is the decoder used? For image generation**: a text prompt goes in and an image comes out. The prompt "Draw a red ball" becomes $[x_1, x_2, x_3, x_4]$, and $\langle\text{boi}\rangle$ is appended: $[x_1, x_2, x_3, x_4, \langle\text{boi}\rangle, ?, ?, ?, \dots]$. **The model starts generating image tokens after $\langle\text{boi}\rangle$.** The AR-LLM predicts them **autoregressively**:

$$P\big(z_1 \mid x_{1:4}, \langle\text{boi}\rangle\big) \to P\big(z_2 \mid x_{1:4}, \langle\text{boi}\rangle, z_1\big) \to \cdots$$

This gives $[\langle\text{boi}\rangle, z_1, z_2, z_3, z_4, \langle\text{eoi}\rangle]$, and **$z_1, \dots, z_4$ are token IDs**. (In this subsection and the next, the professor uses $z_i$ for generated IDs, not for the continuous latent of [[#10.2 The image tokenizer|§10.2]].) **For each token ID the decoder side looks up its code vector**: $z_1 = 5\to c_5$, $z_2 = 5\to c_5$, $z_3 = 1\to c_1$, $z_4 = 6\to c_6$. Arranged on the $2\times2$ grid, these are the **token embeddings** $\begin{bmatrix}c_5 & c_5\\ c_1 & c_6\end{bmatrix}$, and $D$ maps them to an image in **pixel space**.

### 10.6 Token-level autoregressive image generation

The professor gives two input settings for **(token-level) autoregressive image generation**:

1. **Input = text.** $x^T$ goes into a **causal** LLM, which outputs **visual token IDs** for $D$ to decode into an image, modelling $P_\theta(z_i\mid z_{<i}, x^T)$. Your notes' example prompt is "Generate an image of a cat". The LLM is **trained on image + text tokens**, and $D$ uses the codebook to decode.
2. **Input = text + image.** The input image goes through $\mathcal E$ and the image tokenizer with its codebook to give tokens, which join the text in $V_{1:T'}$. The causal LLM then generates new visual token IDs for $D$, modelling $P_\theta(z_i\mid z_{<i}, V_{1:T'})$. This is image editing: "make the ball blue". Your notes draw the LLM's vocabulary as covering both the text vocabulary and the codebook, and pass the input sequence $x = [\langle\text{boi}\rangle, \dots, \langle\text{eoi}\rangle, \dots, \langle\text{eos}\rangle]$ through a **masking scheme**.

![[03 - Attachments/ATML_L11_ar_generation.svg|700]]

*(a) Text in, image out. (b) Text and image in, image out, with the masking scheme from your notes.*

*(Expanding on the masking scheme, which the notes name but don't describe: with mixed sequences, two masks matter. The **attention mask** is causal, so each position sees only earlier tokens, image or text. The **loss mask** decides which positions are trained: for generation, only the output image tokens, not the prompt or the input image, as in [[ATML_Lecture_7 - LLM Foundations#4.3 Batching, padding and the loss mask|Lecture 7 §4.3]].)*

### 10.7 Other families beyond raster-scan autoregression

The autoregressive family (next-token prediction, NTP, with teacher forcing) **predicts image tokens in a fixed order (raster scan)**, row by row from the top-left, and **it is very slow**: an image of $h\times w$ tokens needs $hw$ sequential forward passes. **Other families use discrete image tokens but do not use AR:**

1. **Masked-token style generators.** *(Expanding on this: e.g. MaskGIT. Start with every image token masked, predict all of them in parallel, keep the most confident ones and re-mask the rest, and repeat for a fixed number of steps, typically 8–12. This is the masked-modelling idea of FLAVA's [[#5.3 The five-term FLAVA loss|$L_{\text{MIM}}$]] turned into a generator.)*
2. **Discrete diffusion.** *(Expanding on this: a forward process gradually corrupts the token grid, for example by masking or randomizing tokens, and a network learns to reverse it over several steps. It is the same idea as the [[ATML_Lecture_7 - LLM Foundations#1.3 Discrete diffusion LLMs|discrete diffusion LLMs of Lecture 7 §1.3]]. Stable Diffusion and Imagen ([[#6.3 Stable Diffusion and Imagen|§6.3]]) diffuse in a continuous space; discrete diffusion applies the same corrupt-and-reverse idea to token IDs.)*
3. **Progressive / hierarchical generators.** *(Expanding on this: generate a coarse token map first and then finer ones conditioned on it, e.g. VAR's "next-scale prediction". Each scale is predicted in parallel, so the number of sequential steps is the number of scales, not the number of tokens.)*

[[#Numerical Problems|Problem 16]] compares the number of sequential steps.

### 10.8 How to reason on images?

Your notes end with a VLM taking visual tokens $V$ and text $T$ as $X$ and returning $y$ = text, and ask: **how do we reason on images?** The source doesn't answer it.

![[03 - Attachments/ATML_L11_vlm_reasoning.svg|200]]

*Your notes' closing picture: visual tokens and text in, text out.*

 *(Answer, expanding on the notes: so far, the model maps $(V, T)$ straight to an answer, which is where the reasoning failures of [[#8.1 Five failure types|§8.1]] (right pieces, wrong sequence) come from. Reasoning means generating intermediate steps before the answer, as in the [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#2.2 The reasoning trace|reasoning traces of Lecture 9 §2.2]]. There are two routes. (1) A text chain of thought conditioned on the visual tokens: first state which objects are present, then their attributes and relations, then answer. It can be trained by SFT on such traces or by [[ATML_Lecture_9 - RLVR, Reasoning LLMs, and Using LLMs#1.3 RLVR: a verifier in place of the learned reward model|RLVR]] on visual questions with checkable answers, such as counting or chart reading. (2) Because a unified model can also emit image tokens ([[#10.6 Token-level autoregressive image generation|§10.6]]), the intermediate steps can themselves be visual, such as cropping or zooming into a region or drawing an auxiliary line, before the final text answer. [[ATML_Lecture_12 - Test-Time Scaling and Multimodal Reasoning#6. Reasoning in VLMs|Lecture 12 §6–8]] takes this up.)*

---

## Numerical Problems

**Problem 1 (CLIP loss and zero-shot prediction).** A batch of $N = 3$ pairs has cosine similarities $v_i^\top t_j$

$$\begin{bmatrix}0.8 & 0.2 & 0.1\\ 0.3 & 0.7 & 0.2\\ 0.1 & 0.4 & 0.6\end{bmatrix}$$

with $\tau = 0.1$. Compute $L_{i2t}$, $L_{t2i}$ and $L_c$, and the zero-shot prediction for each image.

Dividing by $\tau$ gives $S = \begin{bmatrix}8&2&1\\3&7&2\\1&4&6\end{bmatrix}$. Row $i$ contributes $-s_{ii} + \log\sum_j e^{s_{ij}}$. The row log-sum-exps are $8.0034$, $7.0247$ and $6.1328$, so the row losses are $0.0034$, $0.0247$ and $0.1328$, with mean $L_{i2t} = 0.0537$. The column log-sum-exps are $8.0076$, $7.0550$ and $6.0247$, so the column losses are $0.0076$, $0.0550$ and $0.0247$, with mean $L_{t2i} = 0.0291$. By [[#^eq-1|eq. (1)]], $L_c = \tfrac12(0.0537 + 0.0291)$. Every row's maximum is on the diagonal, so every image retrieves its own caption.

$$\boxed{L_{i2t} = 0.0537,\quad L_{t2i} = 0.0291,\quad L_c = 0.0414;\quad \hat j = (1, 2, 3)}$$

Image 3 has the largest loss because its caption is only 0.2 ahead of caption 2 in cosine. With $\tau = 1$ instead, the first row's loss alone is $0.7156$: a high temperature flattens the softmax, so even a correct ranking costs a lot.

**Problem 2 (alignment and uniformity).** For image 1 of Problem 1, split its $L_{i2t}$ term into the alignment and uniformity parts of [[#3.3 Why the loss promotes the gap, alignment vs uniformity|§3.3]], and give the softmax probabilities.

The alignment part is $-s_{11} = -8$ and the uniformity part is $\log(e^8 + e^2 + e^1) = 8.0034$. The softmax probabilities are $e^{s_{1j}}/e^{8.0034}$.

$$\boxed{-8 + 8.0034 = 0.0034;\quad p = (0.9966,\ 0.0025,\ 0.0009)}$$

The loss can be lowered either by raising $s_{11}$ (alignment) or by lowering $s_{12}$ and $s_{13}$ (uniformity). Nothing in either term refers to image–image or text–text similarity.

**Problem 3 (SigLIP loss).** A batch of $N = 2$ has logits $s = \begin{bmatrix}2 & -1\\ 0.5 & 3\end{bmatrix}$. Compute $L_{\text{SigLIP}}$.

The sigmoids are $\sigma(2) = 0.8808$, $\sigma(-1) = 0.2689$, $\sigma(0.5) = 0.6225$ and $\sigma(3) = 0.9526$. The diagonal entries are positives, costing $-\log\sigma$: $0.1269$ and $0.0486$. The off-diagonal entries are negatives, costing $-\log(1-\sigma)$: $0.3133$ and $0.9741$. By [[#^eq-2|eq. (2)]] the loss is the mean of the four.

$$\boxed{L_{\text{SigLIP}} = \tfrac14(0.1269 + 0.3133 + 0.9741 + 0.0486) = 0.3657}$$

The worst term is the negative pair $(2, 1)$ at logit 0.5: the model leans towards "match" for a non-matching pair. That pair is penalized on its own, without reference to the rest of row 2.

**Problem 4 (class imbalance and the SigLIP bias).** With $N = 32{,}768$, what fraction of SigLIP's labels are positive? Show that initializing the logit bias at $b = -10$ makes the initial prediction close to that fraction.

There are $N^2 - N = 1{,}073{,}709{,}056$ negatives and $N$ positives, so the positive fraction is $N/N^2 = 1/N = 3.05\times10^{-5}$. At initialization the cosines are near 0, so $s_{ij}\approx b = -10$ and $\sigma(-10) = 4.54\times10^{-5}$.

$$\boxed{1/N = 3.05\times10^{-5}\ \approx\ \sigma(-10) = 4.54\times10^{-5}}$$

Starting with the prior close to the true positive rate means the first steps aren't spent pushing a billion negatives down from $\sigma = 0.5$. That is how the paper counters the imbalance of [[#4.2 When SigLIP is good, and its costs|§4.2]].

**Problem 5 (the InfoNCE bound and batch size).** A CLIP run with $N = 32{,}768$ reaches $L_{i2t} = 2.0$ nats. What lower bound does this give on the image–text mutual information, and what is the most any $N = 256$ run could certify?

$I \ge \ln N - L = \ln 32768 - 2.0 = 10.397 - 2.0$. For $N = 256$ the bound can never exceed $\ln 256$, reached only at $L = 0$.

$$\boxed{I \ge 8.40\ \text{nats};\qquad N = 256:\ \text{bound} \le \ln 256 = 5.55\ \text{nats}}$$

This is the batch-size ceiling of [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.3 InfoNCE as a Lower Bound on Mutual Information|Lecture 3 §3.3]], and the reason the softmax loss wants large $N$ while SigLIP is preferred at small $N$.

**Problem 6 (angles, distances and the modality gap).** (a) Two unit embeddings have cosine 0.3; find their Euclidean distance. (b) In 3D, put three images at $v_k = (\cos30°\cos a_k,\ \cos30°\sin a_k,\ \sin30°)$ and their captions at $t_k = (\cos30°\cos a_k,\ \cos30°\sin a_k,\ -\sin30°)$, with $a_k = 0°, 120°, 240°$. Find the matched and mismatched cosines, and the gap $\lVert\mu_V - \mu_T\rVert$ between the two centroids.

(a) $\lVert v - t\rVert^2 = 2 - 2(0.3) = 1.4$, so the distance is $1.183$. (b) A matched pair shares $a_k$, so $v_k^\top t_k = \cos^2 30° - \sin^2 30° = 0.75 - 0.25 = 0.5$. A mismatched pair differs by $120°$ in $a$, so $v_k^\top t_m = 0.75\cos120° - 0.25 = -0.375 - 0.25 = -0.625$. The $\cos a_k$ and $\sin a_k$ terms average to zero over the three angles, so $\mu_V = (0, 0, 0.5)$ and $\mu_T = (0, 0, -0.5)$.

$$\boxed{\text{(a) } 1.183\qquad \text{(b) matched } 0.5,\ \text{mismatched } -0.625,\ \lVert\mu_V-\mu_T\rVert = 1.0}$$

Every matched pair beats every mismatched pair, so a contrastive loss is satisfied. Yet all images sit in the upper hemisphere and all captions in the lower one: the image cloud and the text cloud never meet. This is the persistent gap of [[#3.2 Property 2, the persistent modality gap|§3.2]]. It hides in a direction (here the third axis) that the ranking of pairs doesn't care about.

**Problem 7 (MLP projector size).** A two-layer MLP maps CLIP features of size $d_v = 1024$ to an LLM of width 4096 ($1024\to4096\to4096$, with biases). Count its parameters, and the visual tokens for a $336\times336$ image with $14\times14$ patches.

$1024\cdot4096 + 4096 = 4{,}198{,}400$ and $4096\cdot4096 + 4096 = 16{,}781{,}312$. The patch grid is $336/14 = 24$ per side, so $24^2 = 576$ patches.

$$\boxed{20{,}979{,}712\approx 21\text{M parameters};\quad 576\ \text{visual tokens}}$$

21M parameters is tiny next to a 7B LLM. This is why stage 1 of [[#7.5 Training stages|§7.5]] trains only the connector.

**Problem 8 (what the Q-Former saves).** A prompt has 100 text tokens. Compare the LLM's self-attention score count when the image enters as 576 MLP-projected tokens and when it enters as 32 Q-Former tokens.

With the MLP bridge the sequence has $676$ tokens, so there are $676^2 = 456{,}976$ scores per head per layer. With the Q-Former it has $132$, so $132^2 = 17{,}424$.

$$\boxed{456{,}976\ \text{vs}\ 17{,}424:\ \approx 26\times\ \text{fewer}}$$

The price is the bottleneck: 32 tokens must hold everything the question might need, which risks the perception failures of [[#8.1 Five failure types|§8.1]] for small details.

**Problem 9 (gated cross-attention).** A hidden state $h = (1, -2)$ receives a cross-attention output $(0.4, 0.6)$ through a $\tanh(\alpha)$ gate. Give the layer output at $\alpha = 0$ and at $\alpha = 0.5$. With 200 text tokens and 64 visual tokens, compare the cross-attention score count with self-attention over the concatenated sequence.

At $\alpha = 0$, $y = h = (1, -2)$, so the LLM is unchanged. At $\alpha = 0.5$, $\tanh(0.5) = 0.4621$, so $y = (1 + 0.1848,\ -2 + 0.2773)$. Cross-attention scores each text query against each visual key, $200\times64 = 12{,}800$. Self-attention over $264$ concatenated tokens gives $264^2 = 69{,}696$.

$$\boxed{y_{\alpha=0} = (1, -2),\quad y_{\alpha=0.5} = (1.185,\ -1.723);\quad 12{,}800\ \text{vs}\ 69{,}696\ (5.4\times)}$$

**Problem 10 (FLAVA loss and batches).** With $\lambda = (1, 0.5, 0.5, 1, 1)$ and component losses $L_{\text{cont}} = 2.1$, $L_{\text{MIM}} = 4.0$, $L_{\text{MLM}} = 3.2$, $L_{\text{ITM}} = 0.45$ and $L_{\text{MMM}} = 3.6$, compute $L_{\text{FLAVA}}$. Which terms can be computed on a batch of captionless images?

By [[#^eq-3|eq. (3)]], $L_{\text{FLAVA}} = 2.1 + 0.5(4.0) + 0.5(3.2) + 0.45 + 3.6$. A captionless image batch supports only $L_{\text{MIM}}$; the other four need text, and $L_{\text{cont}}$, $L_{\text{ITM}}$, $L_{\text{MMM}}$ need matched pairs.

$$\boxed{L_{\text{FLAVA}} = 9.75;\quad \text{image-only batch} \to L_{\text{MIM}}\ \text{only}}$$

**Problem 11 (early-fusion context budget).** A Chameleon-style model turns each $512\times512$ image into 1024 tokens and has a 4096-token context. How many whole images fit alongside a 200-token prompt?

$(4096 - 200)/1024 = 3.80$, so three images fit.

$$\boxed{3\ \text{images}}$$

Images dominate the context in early-fusion models, which is the same pressure that motivates the Q-Former's compression in [[#7.3 Bridge 2, bottleneck resampler (Q-Former)|§7.3]].

**Problem 12 (ViT tokens and the patch-embedding size).** A $224\times224\times3$ image goes through a ViT with $P = 16$ and $d = 768$. Find $N_p$, the length of each patch vector, the shape of $Z_0$, and the number of parameters in $E$ and $b$. Check that a convolution with kernel 16, stride 16 and 768 output channels has the same count.

$N_p = (224\cdot224)/16^2 = 196$. Each patch vector has $P^2C = 16\cdot16\cdot3 = 768$ entries. Adding [CLS] gives $Z_0\in\mathbb R^{197\times768}$. $E$ is $768\times768 = 589{,}824$ weights and $b$ adds 768. The convolution has 768 filters of $16\times16\times3 = 768$ weights each plus one bias per filter: $768\cdot768 + 768$.

$$\boxed{N_p = 196,\quad x_i\in\mathbb R^{768},\quad Z_0: 197\times768,\quad 590{,}592\ \text{parameters either way}}$$

**Problem 13 (read-out options).** After the last layer, a toy ViT with $d = 2$ has $z_L^{(0)} = (0.5, 0.1)$ and patch rows $(1, 0)$, $(0, 2)$, $(-1, 1)$. Give the [CLS], mean-pooled and max-pooled read-outs, and the attention-pooled one with query $q = (1, 1)$ (scores $q^\top z_L^{(i)}$, softmax weights).

[CLS] is $(0.5, 0.1)$. Mean over the three patches: $\big((1 + 0 - 1)/3,\ (0 + 2 + 1)/3\big) = (0, 1)$. Max per coordinate: $(1, 2)$. Attention scores are $1, 2, 0$, so the weights are $e^{1}, e^{2}, e^{0}$ over their sum $11.107$: $(0.2447, 0.6652, 0.0900)$. The pooled vector is $0.2447(1, 0) + 0.6652(0, 2) + 0.0900(-1, 1)$.

$$\boxed{\text{CLS } (0.5, 0.1),\quad \text{mean } (0, 1),\quad \text{max } (1, 2),\quad \text{attention } (0.155, 1.420)}$$

Each is one vector for the whole image. Whatever patch 2 held about *where* something was is now mixed in with the rest, which is why the VLM of [[#9.4 Why a VLM keeps every patch token|§9.4]] keeps all the rows.

**Problem 14 (vector quantization and its two losses).** A codebook has $e_1 = (0, 0)$, $e_2 = (1, 1)$, $e_3 = (2, 0)$. An encoder output is $z_{ij} = (0.9, 0.6)$. Find $q_{ij}$ and $z^q_{ij}$, the codebook and commitment losses for this cell with $\beta = 0.25$, and the gradient each sends (to the code and to $z_{ij}$).

The squared distances are $0.9^2 + 0.6^2 = 1.17$, $0.1^2 + 0.4^2 = 0.17$ and $1.1^2 + 0.6^2 = 1.57$, so by [[#^eq-6|eq. (6)]] $q_{ij} = 2$ and $z^q_{ij} = (1, 1)$. Both losses use the distance $0.17$: $L_{\text{codebook}} = 0.17$ and $L_{\text{commitment}} = 0.25\cdot0.17$. With $z$ frozen, the codebook loss's gradient on $e_2$ is $2(e_2 - z_{ij}) = (0.2, 0.8)$. With $z^q$ frozen, the commitment loss's gradient on $z_{ij}$ is $2\beta(z_{ij} - e_2) = (-0.05, -0.2)$.

$$\boxed{q_{ij} = 2,\ z^q_{ij} = (1, 1);\quad L_{\text{codebook}} = 0.17,\ L_{\text{commitment}} = 0.0425;\quad \nabla_{e_2} = (0.2, 0.8),\ \nabla_{z_{ij}} = (-0.05, -0.2)}$$

Gradient descent moves $e_2$ towards $z_{ij}$ and $z_{ij}$ towards $e_2$, and $\beta$ makes the encoder's step four times smaller than the code's.

**Problem 15 (token budget and vocabulary growth).** A tokenizer downsamples $256\times256\times3$ images by 16 in each direction and has $K = 8192$ codes. How many tokens and bits describe one image, and how does that compare with raw 8-bit pixels? With a 20-token prompt, how long is the sequence $[\langle\text{boi}\rangle, \text{image}, \langle\text{eoi}\rangle, \text{text}]$? If the LLM has width 4096, how many new parameters do the codes add to its input embedding and output layer?

$h = w = 256/16 = 16$, so $256$ tokens, each one of $8192 = 2^{13}$ codes, so 13 bits: $256\cdot13 = 3328$ bits. Raw pixels take $256\cdot256\cdot3\cdot8 = 1{,}572{,}864$ bits. The sequence has $1 + 256 + 1 + 20$ tokens. Each code needs an embedding row and an output row of size 4096: $2\cdot8192\cdot4096$.

$$\boxed{256\ \text{tokens},\ 3328\ \text{bits}\ (\approx473\times\ \text{smaller});\quad 278\ \text{tokens};\quad 67{,}108{,}864\ \text{new parameters}}$$

The compression is what makes images affordable as tokens. The new rows start untrained, which is why your notes say the LLM must be retrained ([[#10.4 Image understanding with token IDs|§10.4]]).

**Problem 16 (why raster-scan AR is slow).** Generating the 256-token image of Problem 15, how many sequential forward passes does raster-scan AR need, and how many does a masked-token generator with 8 refinement steps need?

AR emits one token per pass: 256 passes. The masked generator predicts every token in each pass: 8 passes.

$$\boxed{256\ \text{vs}\ 8:\ 32\times\ \text{fewer sequential steps}}$$

Each masked-generator pass processes all 256 positions at once, so it does more work than one AR step with a [[ATML_Lecture_7 - LLM Foundations#3.5 The KV cache|KV cache]]. The saving is in latency, the number of steps that must wait for each other ([[#10.7 Other families beyond raster-scan autoregression|§10.7]]).

**Problem 17 (the adversarial loss).** The discriminator outputs the probability that its input is real. Taking $L_{\text{adversarial}} = -\log \text{Disc}(\hat x)$ for the tokenizer, evaluate it when $\text{Disc}(\hat x) = 0.1$, $0.5$ and $0.9$.

$-\ln 0.1 = 2.303$, $-\ln 0.5 = 0.693$ and $-\ln 0.9 = 0.105$.

$$\boxed{2.303,\quad 0.693,\quad 0.105}$$

At 0.5 the discriminator is "maximally confused" ([[#10.3 Training the tokenizer, five losses|§10.3]]): it can't tell $\hat x$ from $x$. The loss keeps falling past 0.5 because the tokenizer is rewarded for fooling a discriminator that is still learning. *(The professor writes $\text{Disc}(x, \hat x)$, a discriminator that sees both images; the single-input form here is the standard GAN generator loss of [[ATML_Lecture_4 - GANs, and Inductive Biases#1.7 Saturating vs. Non-Saturating GANs|Lecture 4 §1.7]].)*

---

## Formula Summary Sheet

| Concept | Formula | Where |
|---|---|---|
| Similarity | $s_{ij} = v_i^\top t_j/\tau$, with $\lVert v_i\rVert = \lVert t_j\rVert = 1$ | [[#2.2 The similarity matrix\|§2.2]] |
| Image → text loss | $L_{i2t} = -\frac1N\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}$ | [[#2.3 The symmetric InfoNCE loss\|§2.3]] |
| Text → image loss | $L_{t2i} = -\frac1N\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}$ | [[#2.3 The symmetric InfoNCE loss\|§2.3]] |
| Symmetric InfoNCE ([[#^eq-1\|eq. (1)]]) | $L_c = \tfrac12(L_{i2t} + L_{t2i})$ | [[#2.3 The symmetric InfoNCE loss\|§2.3]], [[#9.5 Representational vs interface alignment\|§9.5]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.2 The Contrastive Pipeline and the InfoNCE Loss\|L3 §3.2]] |
| Zero-shot / retrieval | $\hat j = \arg\max_j v^\top t_j$ | [[#2.4 Goal and inference\|§2.4]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.6 Zero-Shot Inference with CLIP\|L3 §3.6]] |
| Alignment–uniformity split | $L_c = -\mathbb E[s_{ii}] + \mathbb E[\log\sum_j e^{s_{ij}}]$ | [[#3.3 Why the loss promotes the gap, alignment vs uniformity\|§3.3]] |
| Unit-sphere distance | $\lVert v - t\rVert^2 = 2 - 2\cos\varphi$ | [[#3.1 Property 1, a cosine-similarity space on the unit sphere\|§3.1]] |
| SigLIP labels | $y_{ij} = 1$ if $i = j$, else 0 | [[#4.1 SigLIP, matching as binary classification\|§4.1]] |
| SigLIP / ITM loss ([[#^eq-2\|eq. (2)]]) | $-\frac1{N^2}\sum_{i,j}[y_{ij}\log\sigma(s_{ij}) + (1-y_{ij})\log(1-\sigma(s_{ij}))]$ | [[#4.1 SigLIP, matching as binary classification\|§4.1]] |
| SigLIP logit (paper) | $s_{ij} = t'\,v_i^\top t_j + b$, init $t' = 10$, $b = -10$ | [[#4.1 SigLIP, matching as binary classification\|§4.1]] |
| Positive fraction | $N/N^2 = 1/N$ | [[#4.2 When SigLIP is good, and its costs\|§4.2]] |
| InfoNCE MI bound | $I \ge \log N - L$ | [[#4.2 When SigLIP is good, and its costs\|§4.2]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#3.3 InfoNCE as a Lower Bound on Mutual Information\|L3 §3.3]] |
| LLIP | $z_i^{V\mid T} = \text{CrossAttn}(Q = z_j^T,\ K, V = z_i^V)$ | [[#4.3 LLIP, caption-conditioned image embeddings\|§4.3]] |
| FLAVA ([[#^eq-3\|eq. (3)]]) | $\lambda_1L_{\text{cont}} + \lambda_2L_{\text{MIM}} + \lambda_3L_{\text{MLM}} + \lambda_4L_{\text{ITM}} + \lambda_5L_{\text{MMM}}$ | [[#5.3 The five-term FLAVA loss\|§5.3]] |
| CoCa | $\lambda_{\text{con}}L_{\text{cont}} + \lambda_{\text{cap}}L_{\text{cap}}$ | [[#6.1 CoCa\|§6.1]] |
| Vision tokens (no readout) | $z^V\in\mathbb R^{(N_p+1)\times d_v}$, $N_p = (H/P)(W/P)$ | [[#7.1 The pipeline\|§7.1]], [[#9.4 Why a VLM keeps every patch token\|§9.4]] |
| MLP projector | $\tilde z = W_2\,\sigma(W_1z)$ per token | [[#7.2 Bridge 1, linear or MLP projector\|§7.2]] |
| Gated cross-attention | $y = h + \tanh(\alpha)\,\text{xattn}(h, Z^V)$, $\alpha_0 = 0$ | [[#7.4 Bridge 3, gated cross-attention through an adapter\|§7.4]] |
| Patches | $N_p = HW/P^2$, $x_i\in\mathbb R^{P^2C}$ | [[#9.2 The ViT vision encoder\|§9.2]] |
| Patch embedding | $e_i = Ex_i + b$, $E\in\mathbb R^{d\times P^2C}$ (≡ conv, kernel $P$, stride $P$, $d$ channels) | [[#9.2 The ViT vision encoder\|§9.2]], [[ATML_Lecture_3 - Discriminative, Contrastive, and Generative Models#2.6 ViT — Patchify, Patch Embedding, CLS and Positional Embedding\|L3 §2.6]] |
| ViT input | $Z_0 = [e_{\text{CLS}}, e_i] + E_{\text{pos}}\in\mathbb R^{(N_p+1)\times d}$ | [[#9.2 The ViT vision encoder\|§9.2]] |
| ViT layer (pre-LN, no causal mask) | $z'_\ell = z_{\ell-1} + \text{MHA}(\text{LN}(z_{\ell-1}))$, $z_\ell = z'_\ell + \text{MLP}(\text{LN}(z'_\ell))$ | [[#9.2 The ViT vision encoder\|§9.2]] |
| Classification read-out | $\hat y = W_h\,\text{LN}(z_L^{(0)}) + b_h$, loss $-\log p(y\mid x^V)$ | [[#9.3 Read-out for classification\|§9.3]] |
| Mean pooling | $\bar z = \frac1{N_p}\sum_{i=1}^{N_p} z_L^{(i)}$ | [[#9.3 Read-out for classification\|§9.3]] |
| VLM connector | $V = \text{Connector}(Z_L)$ | [[#9.4 Why a VLM keeps every patch token\|§9.4]] |
| Image–text-grounded generation ([[#^eq-4\|eq. (4)]]) | $L_{\text{ITT}} = -\sum_t\log P_\theta(y_t\mid y_{<t}, x^T, V)$ | [[#9.5 Representational vs interface alignment\|§9.5]] |
| Image-grounded generation ([[#^eq-5\|eq. (5)]]) | $L_{\text{IT}} = -\sum_t\log P_\theta(y_t\mid y_{<t}, V)$ | [[#9.5 Representational vs interface alignment\|§9.5]] |
| Quantization ([[#^eq-6\|eq. (6)]]) | $q_{ij} = \arg\min_k\lVert z_{ij} - e_k\rVert_2^2$, $z^q_{ij} = e_{q_{ij}}$ | [[#10.2 The image tokenizer\|§10.2]] |
| Reconstruction loss | $\lVert x - \hat x\rVert_2^2$ | [[#10.3 Training the tokenizer, five losses\|§10.3]] |
| Codebook loss (moves codes) | $\lVert\text{sg}(z) - z^q\rVert_2^2$ | [[#10.3 Training the tokenizer, five losses\|§10.3]] |
| Commitment loss (moves encoder) | $\beta\lVert z - \text{sg}(z^q)\rVert_2^2$ | [[#10.3 Training the tokenizer, five losses\|§10.3]] |
| Perceptual loss | $\sum_\ell\lVert\phi_\ell(x) - \phi_\ell(\hat x)\rVert_1$ (VGG layer $\ell$) | [[#10.3 Training the tokenizer, five losses\|§10.3]], [[ATML_Lecture_2 - Information Theory and Divergences#4.1 The Two Families of Losses\|L2 §4.1]] |
| Adversarial loss | $-\log\text{Disc}(x, \hat x)$ | [[#10.3 Training the tokenizer, five losses\|§10.3]], [[ATML_Lecture_4 - GANs, and Inductive Biases#1.3 The Min-Max Game\|L4 §1.3]] |
| AR image generation | $P_\theta(z_i\mid z_{<i}, x^T)$ or $P_\theta(z_i\mid z_{<i}, V_{1:T'})$ | [[#10.5 Image generation, from token IDs back to pixels\|§10.5]], [[#10.6 Token-level autoregressive image generation\|§10.6]] |
| Tokens per image | $hw = (H/f)(W/f)$ for downsampling $f$; $\log_2K$ bits each | [[#10.4 Image understanding with token IDs\|§10.4]] |

---

## Summary

- **What a VLM is ([[#1. What a VLM Is|§1]])**
  - A parameterized system that learns a joint statistical interface between vision and text, to compare them, generate text, or generate/edit images.
  - Three families: representational (compare), fusion (interact token by token), generative (produce text or images).

- **Representational VLMs ([[#2. Representational VLMs|§2]])**
  - Two encoders, projection and normalization to the unit sphere, then $s_{ij} = v_i^\top t_j/\tau$ over the $N\times N$ batch.
  - Rows of $S$ are images and columns texts; $L_{i2t}$ normalizes along a row, $L_{t2i}$ down a column, and $L_c$ averages them.
  - CLIP asks "which caption is best?"; inference takes the argmax over candidate captions.

- **Geometry of the shared space ([[#3. Geometry of the Shared Space|§3]])**
  - Angles, not Euclidean distances, measure similarity on the unit sphere.
  - The modality gap persists: information mismatch, different inductive biases, and a loss that promotes it.
  - $L_c$ splits into alignment and uniformity; uniformity has no within-modality term, so separate clouds can satisfy the loss.
  - The gap should be neither too large (bad alignment) nor too small (maybe a collapsed modality): shared semantics, modality-specific details kept.
  - The cone effect is there at initialization: each encoder fills a narrow, off-centre cone.

- **SigLIP and LLIP ([[#4. Beyond Softmax — SigLIP and LLIP|§4]])**
  - SigLIP replaces the softmax with a per-pair sigmoid, asking "does this caption match?".
  - It is good for small $N$, but has class imbalance and weaker mutual-information maximization; both come from replacing the shared uniformity term with per-pair pushes.
  - LLIP makes the image embedding depend on the caption through cross-attention, keeping fine details and handling ambiguous text; the comparison is still one dot product per pair.

- **Fusion VLMs ([[#5. Fusion VLMs|§5]])**
  - Process image and text tokens jointly, so a word can be grounded in a region.
  - FLAVA's three models train on five losses: contrastive, MIM, MLM, ITM and MMM, using image-only, text-only and paired batches.
  - Inference uses the unmasked branches; fine-tuning puts a task head on the fused latent.

- **Generative VLMs ([[#6. Generative VLMs|§6]])**
  - CoCa adds a captioning decoder to a contrastive model; Chameleon and CM3leon put image and text tokens in one autoregressive LM.
  - Stable Diffusion and Imagen are text-to-image diffusion models; backbones can be retrained or frozen.

- **Adaptive modularity ([[#7. Adaptive Modularity|§7]])**
  - A vision encoder, a trainable bridge and a pre-trained LLM; the encoder passes all $N_p+1$ tokens.
  - Bridges: an MLP projector passes every token, a Q-Former compresses to a few learned-query tokens, gated cross-attention inserts vision inside the LLM layers.
  - The $\tanh(\alpha)$ gate starts at 0, like LoRA's $B = 0$, so training starts from the pre-trained LLM.
  - Three stages: connector only, then SFT with the LLM, then visual instruction tuning; the vision encoder is never trained.

- **Failure modes ([[#8. VLM Failure Modes|§8]])**
  - Perception, grounding, structure/composition, reasoning and trustworthiness (hallucination, non-robustness).
  - Perception fails in the vision encoder, alignment in the connector, grounding and composition in the LLM.

- **Inside the modular VLM ([[#9. Inside the Modular VLM — Vision Encoding and Alignment|§9]])**
  - Text enters as discrete IDs, images as continuous embeddings through a connector.
  - The ViT: patchify, patch embedding (a strided convolution), [CLS], positional embedding, pre-LN layers with no causal mask.
  - A classifier reads out one vector ([CLS], mean, max or attention pooling); a VLM keeps every row, since one vector loses spatial detail.
  - Representational alignment makes embeddings close ($L_{\text{cont}}$, $L_{\text{ITM}}$); interface alignment makes $V$ usable by the LLM ($L_{\text{ITT}}$, $L_{\text{IT}}$).
  - Limitations: the LLM is text-trained, there is no guideline for where fusion should happen, and the frozen encoder puts all the burden on the connector.

- **Modality unification ([[#10. Modality Unification — Images as Discrete Tokens|§10]])**
  - An image tokenizer (VQ-VAE style) maps each latent cell to its nearest codebook vector, and the code index is the image token.
  - Five tokenizer losses: reconstruction, codebook, commitment, perceptual and adversarial; stop-gradient decides whether the codes or the encoder move.
  - Understanding uses the encoder and codebook, with no connector, but the LLM must be retrained on the larger vocabulary; generation uses the codebook and decoder.
  - Raster-scan autoregressive generation is slow; masked-token, discrete-diffusion and progressive generators need fewer sequential steps.
  - How to reason on images is left open; one route is generating intermediate steps, in text or as image tokens.

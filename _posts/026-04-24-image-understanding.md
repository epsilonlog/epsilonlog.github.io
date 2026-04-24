---
title: "From Classification to Vision-Language Alignment"
date: 2026-04-24 12:00:00 +0000
categories: [AI Engineering, Vision, LLM]
tags: [AI, vision, llm]
description: A comprehensive, deep-dive guide into the understanding images with llm
math: true
---

## Introduction: The Foundation of Understanding

Understanding modern vision-language models requires building intuition from first principles. We'll start with the humble classifier and trace a mathematical path through contrastive learning, arriving at CLIP and SigLIP. Along the way, we'll see how each innovation solves a specific problem, transforming simple cross-entropy loss into powerful multimodal alignment.Then we go through BLIP2, LLAVA and Qwen 2.5 models. 

## Part 1: The K-Class Classifier and the Birth of Contrastive Thinking

### 1.1 Classical Classification: Softmax and Cross-Entropy

Consider a standard classification problem with $K$ classes. Given an input $\mathbf{x} \in \mathbb{R}^d$, we compute logits using a weight matrix $\mathbf{W} \in \mathbb{R}^{K \times d}$:

$$\mathbf{z} = \mathbf{W}\mathbf{x} \in \mathbb{R}^K$$

Each logit $z_i = \mathbf{w}_i^T \mathbf{x}$ represents the unnormalized score for class $i$, where $\mathbf{w}_i$ is the $i$-th row of $\mathbf{W}$.

The softmax function converts these logits into probabilities:

$$p_i = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

**Why softmax?** It pushes probability mass toward the highest logit while suppressing others. For the correct class $c$, we want $z_c \gg z_j$ for all $j \neq c$. The cross-entropy loss for a single sample is:

$$\mathcal{L}_{\text{CE}} = -\log p_c = -z_c + \log \sum_{j=1}^{K} \exp(z_j)$$

This loss has two competing terms:
1. $-z_c$: Maximize the correct class logit
2. $\log \sum_{j=1}^{K} \exp(z_j)$: Minimize all logits (regularization)

The balance creates a **contrastive effect**: push the correct class up, push wrong classes down.

### 1.2 The Lower Bound of Multi-Class Loss

An important property: the cross-entropy loss has a lower bound determined by the number of classes. When all logits are equal ($z_i = z$ for all $i$), we get:

$$\mathcal{L}_{\text{CE}} = -z + \log(K \cdot \exp(z)) = \log K$$

**Equality condition**: This minimum is achieved when the model is maximally uncertain, assigning equal probability $\frac{1}{K}$ to all classes. For a well-trained model with confidence, $\mathcal{L}_{\text{CE}} < \log K$, approaching 0 as $p_c \to 1$.

**Key insight**: As $K$ increases, the baseline difficulty increases logarithmically. This will become crucial when we move to dynamic, large-scale classification.

### 1.3 Softmax Invariance: A Hidden Superpower

Softmax has a remarkable property: it's invariant to constant shifts. For any scalar $\alpha$:

$$\frac{\exp(z_i + \alpha)}{\sum_j \exp(z_j + \alpha)} = \frac{\exp(z_i) \cdot \exp(\alpha)}{\sum_j \exp(z_j) \cdot \exp(\alpha)} = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

**Why does this matter?** 
1. **Numerical stability**: We can subtract $\max_j z_j$ to prevent overflow
2. **Temperature scaling**: We can divide logits by temperature $\tau$ to control sharpness
3. **Implicit normalization**: The model learns relative differences, not absolute scales

This invariance will reappear when we introduce temperature parameters in contrastive learning.

### 1.4 Matrix Form: Revealing the Contrastive Structure

Let's write the logits in matrix form. For a batch of $N$ samples $\mathbf{X} \in \mathbb{R}^{N \times d}$ (each row is a sample):

$$\mathbf{Z} = \mathbf{X}\mathbf{W}^T \in \mathbb{R}^{N \times K}$$

Each element $z_{ij} = \mathbf{x}_i^T \mathbf{w}_j$ is the similarity between sample $i$ and class prototype $j$.

**The contrastive interpretation**: For sample $i$ with true class $c_i$, the loss is:

$$\mathcal{L}_i = -\mathbf{x}_i^T \mathbf{w}_{c_i} + \log \sum_{j=1}^{K} \exp(\mathbf{x}_i^T \mathbf{w}_j)$$

This explicitly shows: maximize similarity to the correct class, minimize similarity to all classes (including the correct one, which creates the contrastive tension).

## Part 2: Dynamic Classes - When Data Becomes the Classifier

### 2.1 The Radical Idea: $\mathbf{W} = \mathbf{X}$

Now imagine a paradigm shift: **what if the classes themselves are the data?** 

Set $\mathbf{W} = \mathbf{X}^T$, so we have $N$ samples, each defining its own class. The logit matrix becomes:

$$\mathbf{Z} = \mathbf{X}\mathbf{X}^T \in \mathbb{R}^{N \times N}$$

Each element $z_{ij} = \mathbf{x}_i^T \mathbf{x}_j$ is the **similarity** between samples $i$ and $j$.

**The new classification task**: For each sample $i$, the "correct class" is itself ($j = i$), and all other samples are "wrong classes" ($j \neq i$).

The loss for sample $i$ becomes:

$$\mathcal{L}_i = -\mathbf{x}_i^T \mathbf{x}_i + \log \sum_{j=1}^{N} \exp(\mathbf{x}_i^T \mathbf{x}_j)$$

**Why is this powerful?**
1. **Dynamic classes**: The "classes" change with every batch
2. **Self-supervised**: No external labels needed
3. **Contrastive by design**: Each sample contrasts against all others

This is the essence of **contrastive learning**.

### 2.2 InfoNCE Loss: Formalizing the Contrastive Objective

The InfoNCE (Noise Contrastive Estimation) loss formalizes this idea. For a batch of $N$ samples, where sample $i$ is the "anchor" and sample $i$ itself is the "positive":

$$\mathcal{L}_{\text{InfoNCE}}^{(i)} = -\log \frac{\exp(\mathbf{x}_i^T \mathbf{x}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{x}_i^T \mathbf{x}_j / \tau)}$$

Here $\tau$ is a **temperature parameter** that controls the concentration of the distribution:
- Small $\tau$: Sharp distribution, focuses on the most similar samples
- Large $\tau$: Smooth distribution, treats all samples more equally

**The contrastive principle**: 
- **Numerator**: Pull the positive pair together ($\mathbf{x}_i^T \mathbf{x}_i$ is maximized)
- **Denominator**: Push away from all samples (including itself), creating contrast

The average loss over the batch is:

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{InfoNCE}}^{(i)}$$

### 2.3 The Problem: Trivial Solutions

But wait—there's a problem! If $\mathbf{x}_i^T \mathbf{x}_i$ is always the positive, the model could collapse to a trivial solution: make all $\mathbf{x}_i$ identical. Then $\mathbf{x}_i^T \mathbf{x}_j = \mathbf{x}_i^T \mathbf{x}_i$ for all $j$, and the loss becomes $-\log \frac{1}{N} = \log N$ (the lower bound we saw earlier).

**How do we prevent collapse?** We need **true positive pairs** that are semantically similar but not identical, and **true negative pairs** that are semantically different.

## Part 3: CLIP - Bridging Two Modalities

### 3.1 The Vision-Language Setup

CLIP (Contrastive Language-Image Pre-training) solves the collapse problem elegantly: use **two different modalities** as positive pairs.

Given:
- $N$ images: $\mathbf{X} \in \mathbb{R}^{N \times d_I}$ (image embeddings)
- $N$ texts: $\mathbf{Y} \in \mathbb{R}^{N \times d_T}$ (text embeddings)
- Paired data: image $i$ matches text $i$

After projection to a shared embedding space $\mathbb{R}^d$:
- Image embeddings: $\mathbf{I} = f_I(\mathbf{X}) \in \mathbb{R}^{N \times d}$
- Text embeddings: $\mathbf{T} = f_T(\mathbf{Y}) \in \mathbb{R}^{N \times d}$

Typically, embeddings are L2-normalized: $\|\mathbf{i}_k\|_2 = \|\mathbf{t}_k\|_2 = 1$.

### 3.2 The Dual Similarity Matrix

Compute two similarity matrices:

$$\mathbf{S}_{I \to T} = \mathbf{I}\mathbf{T}^T \in \mathbb{R}^{N \times N}$$
$$\mathbf{S}_{T \to I} = \mathbf{T}\mathbf{I}^T \in \mathbb{R}^{N \times N}$$

Each element:
- $s_{ij}^{I \to T} = \mathbf{i}_i^T \mathbf{t}_j$: similarity between image $i$ and text $j$
- $s_{ij}^{T \to I} = \mathbf{t}_i^T \mathbf{i}_j$: similarity between text $i$ and image $j$

Note: $\mathbf{S}_{I \to T} = (\mathbf{S}_{T \to I})^T$ (they're transposes).

### 3.3 Bidirectional Contrastive Loss

**Image-to-Text direction**: Treat each image as a query, and texts as classes. For image $i$, text $i$ is the correct "class":

$$\mathcal{L}_{I \to T}^{(i)} = -\log \frac{\exp(s_{ii}^{I \to T} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij}^{I \to T} / \tau)}$$

This is cross-entropy where:
- **Positive**: $s_{ii}^{I \to T} = \mathbf{i}_i^T \mathbf{t}_i$ (matched pair)
- **Negatives**: $s_{ij}^{I \to T} = \mathbf{i}_i^T \mathbf{t}_j$ for $j \neq i$ (mismatched pairs)

**Text-to-Image direction**: Symmetrically, treat each text as a query:

$$\mathcal{L}_{T \to I}^{(i)} = -\log \frac{\exp(s_{ii}^{T \to I} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij}^{T \to I} / \tau)}$$

**The CLIP loss** is the average of both directions:

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^{N} \left( \mathcal{L}_{I \to T}^{(i)} + \mathcal{L}_{T \to I}^{(i)} \right)$$

Or more compactly:

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathbb{E}[\mathcal{L}_{\text{CE}}(\mathbf{I}\mathbf{T}^T)] + \mathbb{E}[\mathcal{L}_{\text{CE}}(\mathbf{T}\mathbf{I}^T)] \right)$$

### 3.4 Why Bidirectional?

**Symmetry and balance**: Each modality should be separable when the other is treated as classes.

- **Image-to-Text**: Images learn to distinguish between different text descriptions
- **Text-to-Image**: Texts learn to distinguish between different images

Without both directions, the model could develop asymmetric representations where one modality dominates.

**Intuition**: Imagine learning a language. You need to:
1. Understand what others say (text → image)
2. Express yourself clearly (image → text)

Both skills reinforce each other.

### 3.5 The Contrastive Structure in CLIP

For a batch of size $N$:
- **Positive pairs**: $N$ (diagonal elements $s_{ii}$)
- **Negative pairs**: $N^2 - N$ (off-diagonal elements)

Each sample contrasts against $N-1$ negatives. As batch size increases:
- More negatives → harder task → better representations
- But also: more computation ($O(N^2)$ similarities)

**The temperature parameter $\tau$**: 
- Controls the "hardness" of negatives
- Learned during training (typically initialized around 0.07)
- Smaller $\tau$ → sharper distribution → focuses on hard negatives

### 3.6 From Cross-Entropy to InfoNCE

Notice that CLIP's loss is exactly InfoNCE applied bidirectionally:

$$\mathcal{L}_{\text{InfoNCE}}^{I \to T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathbf{i}_i^T \mathbf{t}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{i}_i^T \mathbf{t}_j / \tau)}$$

This is **categorical cross-entropy** where:
- Each image is a "sample"
- Each text is a "class"
- The correct class is the matched text

**The key insight**: We've transformed a vision-language alignment problem into a classification problem with $N$ dynamic classes per batch.

### 3.7 Why CLIP Doesn't Collapse

Unlike the single-modality case ($\mathbf{W} = \mathbf{X}$), CLIP avoids collapse because:

1. **Different modalities**: Images and texts have different information content
2. **Semantic pairing**: Matched pairs share meaning, not form
3. **Hard negatives**: In-batch negatives are often semantically similar (e.g., "a dog" vs "a cat"), forcing fine-grained distinctions

**Example**: In a batch with images of [dog, cat, car, tree] and corresponding texts:
- Image "dog" must distinguish text "a dog" from "a cat" (hard negative)
- This forces the model to learn semantic features, not just superficial patterns

### 3.8 The Role of Batch Size

Larger batches provide more negatives, improving representation quality:

**Small batch ($N=32$)**:
- 31 negatives per sample
- May miss hard negatives
- Faster training, but potentially weaker representations

**Large batch ($N=32768$)** (CLIP's actual batch size):
- 32,767 negatives per sample
- Rich diversity of hard negatives
- Requires distributed training, but yields superior representations

**The hard negative problem**: Not all negatives are equally useful. A negative that's very different (e.g., "dog" vs "car") provides little learning signal. A hard negative (e.g., "golden retriever" vs "labrador retriever") forces the model to learn fine-grained features.

Large batches naturally include more hard negatives, improving learning efficiency.

## Part 4: SigLIP - Refining the Contrastive Objective

### 4.1 The Problem with Softmax-Based Losses

CLIP's softmax-based loss has limitations:

1. **Imbalanced negatives**: $N^2 - N$ negatives vs $N$ positives → ratio approaches $N$ as batch grows
2. **Computational cost**: Softmax denominator requires summing over all $N$ samples
3. **Gradient flow**: Negatives dominate the gradient, potentially overwhelming positive signals

**The imbalance problem**: For $N=1024$, we have 1,023 negatives per positive. The loss is:

$$\mathcal{L}_i = -s_{ii} + \log \sum_{j=1}^{N} \exp(s_{ij})$$

The gradient with respect to $s_{ii}$ (positive) is:

$$\frac{\partial \mathcal{L}_i}{\partial s_{ii}} = -1 + p_{ii}$$

where $p_{ii} = \frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}$ is the softmax probability.

The gradient with respect to $s_{ik}$ (negative, $k \neq i$) is:

$$\frac{\partial \mathcal{L}_i}{\partial s_{ik}} = p_{ik}$$

**The issue**: The sum of negative gradients is $\sum_{k \neq i} p_{ik} = 1 - p_{ii}$, which can be much larger than the positive gradient magnitude when $p_{ii}$ is small (early training).

### 4.2 SigLIP: Sigmoid Loss for Language-Image Pre-training

SigLIP replaces softmax with **sigmoid loss**, treating each pair independently:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma(y_{ij} \cdot (\alpha \cdot s_{ij} + \beta))$$

where:
- $y_{ij} = \begin{cases} +1 & \text{if } i = j \text{ (positive pair)} \\ -1 & \text{if } i \neq j \text{ (negative pair)} \end{cases}$
- $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the sigmoid function
- $\alpha > 0$ is a **temperature-like scaling parameter**
- $\beta$ is a **bias term** that shifts the decision boundary

### 4.3 Understanding the Parameters: $\alpha$ and $\beta$

**The scaling parameter $\alpha$**: 

For a positive pair ($y_{ij} = +1$):
$$\mathcal{L}_{ij}^{+} = -\log \sigma(\alpha \cdot s_{ij} + \beta) = \log(1 + \exp(-\alpha \cdot s_{ij} - \beta))$$

For a negative pair ($y_{ij} = -1$):
$$\mathcal{L}_{ij}^{-} = -\log \sigma(-\alpha \cdot s_{ij} - \beta) = \log(1 + \exp(\alpha \cdot s_{ij} + \beta))$$

**Role of $\alpha$** (temperature inverse):
- Large $\alpha$: Steep sigmoid, sharp decision boundary, focuses on hard examples
- Small $\alpha$: Gentle sigmoid, soft decision boundary, treats all examples more equally

This is analogous to $1/\tau$ in CLIP's softmax formulation.

**Role of $\beta$** (bias shift):
- $\beta > 0$: Shifts decision boundary, making it easier to classify positives
- $\beta < 0$: Shifts decision boundary, making it harder to classify positives
- $\beta = 0$: Symmetric decision boundary at $s_{ij} = 0$

### 4.4 Why $\beta$ Matters: Gradient Analysis

Let's compute gradients to understand $\beta$'s effect.

**For $\beta = 0$** (symmetric case):

Positive pair gradient:
$$\frac{\partial \mathcal{L}_{ij}^{+}}{\partial s_{ij}} = -\alpha \cdot \sigma(-\alpha \cdot s_{ij}) = -\alpha \cdot \frac{1}{1 + \exp(\alpha \cdot s_{ij})}$$

Negative pair gradient:
$$\frac{\partial \mathcal{L}_{ij}^{-}}{\partial s_{ij}} = \alpha \cdot \sigma(\alpha \cdot s_{ij}) = \alpha \cdot \frac{1}{1 + \exp(-\alpha \cdot s_{ij})}$$

**Expected gradient over random initialization**: Assume $s_{ij} \sim \mathcal{N}(0, \sigma^2)$ initially.

For positives: $\mathbb{E}[\frac{\partial \mathcal{L}^{+}}{\partial s}] = -\alpha \cdot \mathbb{E}[\sigma(-\alpha s)]$

For negatives: $\mathbb{E}[\frac{\partial \mathcal{L}^{-}}{\partial s}] = \alpha \cdot \mathbb{E}[\sigma(\alpha s)]$

By symmetry of the normal distribution: $\mathbb{E}[\sigma(-\alpha s)] = \mathbb{E}[\sigma(\alpha s)] = 0.5$

So:
- Expected positive gradient: $-0.5\alpha$
- Expected negative gradient: $+0.5\alpha$

**The imbalance**: With $N$ positives and $N^2 - N \approx N^2$ negatives, the total expected gradient is:

$$\mathbb{E}[\nabla \mathcal{L}] \approx N \cdot (-0.5\alpha) + N^2 \cdot (0.5\alpha) = 0.5\alpha N(N - 1) \approx 0.5\alpha N^2$$

**The gradient is dominated by negatives!** This causes slow learning of positive pairs.

**For $\beta \neq 0$** (asymmetric case):

With $\beta > 0$:

Positive pair gradient:
$$\frac{\partial \mathcal{L}_{ij}^{+}}{\partial s_{ij}} = -\alpha \cdot \sigma(-\alpha \cdot s_{ij} - \beta)$$

At initialization ($s_{ij} \approx 0$):
$$\frac{\partial \mathcal{L}_{ij}^{+}}{\partial s_{ij}} \approx -\alpha \cdot \sigma(-\beta) = -\alpha \cdot \frac{1}{1 + \exp(\beta)}$$

For $\beta > 0$, $\sigma(-\beta) < 0.5$, so the positive gradient magnitude is **smaller**.

Negative pair gradient:
$$\frac{\partial \mathcal{L}_{ij}^{-}}{\partial s_{ij}} = \alpha \cdot \sigma(\alpha \cdot s_{ij} + \beta)$$

At initialization:
$$\frac{\partial \mathcal{L}_{ij}^{-}}{\partial s_{ij}} \approx \alpha \cdot \sigma(\beta) = \alpha \cdot \frac{1}{1 + \exp(-\beta)}$$

For $\beta > 0$, $\sigma(\beta) > 0.5$, so the negative gradient magnitude is **larger**.

**The balance**: By choosing $\beta$ appropriately, we can balance the total gradient:

$$N \cdot \alpha \cdot \sigma(-\beta) \approx N^2 \cdot \alpha \cdot \sigma(\beta)$$

Solving: $\sigma(-\beta) \approx N \cdot \sigma(\beta)$

For large $N$, this requires $\beta \approx \log N$ (approximately).

**Intuition**: $\beta$ shifts the decision boundary to account for the class imbalance, ensuring positives and negatives contribute equally to learning.

### 4.5 Faster Convergence with $\beta$

Let's verify this with a simplified analysis. Define the expected loss:

$$\mathbb{E}[\mathcal{L}] = N \cdot \mathbb{E}[\mathcal{L}^{+}] + N^2 \cdot \mathbb{E}[\mathcal{L}^{-}]$$

**With $\beta = 0$**: At initialization ($s \approx 0$):
$$\mathcal{L}^{+} \approx \log(1 + \exp(0)) = \log 2$$
$$\mathcal{L}^{-} \approx \log(1 + \exp(0)) = \log 2$$

Total loss: $\mathbb{E}[\mathcal{L}] \approx N \log 2 + N^2 \log 2 = (N + N^2) \log 2$

**With $\beta = \log N$**: At initialization:
$$\mathcal{L}^{+} \approx \log(1 + \exp(-\log N)) = \log(1 + 1/N) \approx 1/N$$
$$\mathcal{L}^{-} \approx \log(1 + \exp(\log N)) = \log(1 + N) \approx \log N$$

Total loss: $\mathbb{E}[\mathcal{L}] \approx N \cdot (1/N) + N^2 \cdot \log N = 1 + N^2 \log N$

**The difference**: With $\beta = \log N$, the positive loss is much smaller initially, allowing the model to focus on learning positive pairs first. This leads to faster convergence.

**Empirical results** (from SigLIP paper): With $\beta = \log N$ and $\alpha = 1$, SigLIP converges 2-3× faster than CLIP while achieving similar or better performance.

### 4.6 SigLIP vs CLIP: Key Differences

| Aspect | CLIP | SigLIP |
|--------|------|--------|
| Loss function | Softmax (cross-entropy) | Sigmoid (binary) |
| Pair treatment | Coupled (via softmax denominator) | Independent |
| Positive/negative ratio | Implicit (via softmax) | Explicit (via $\beta$) |
| Temperature | $\tau$ (learned) | $\alpha$ (fixed or learned) |
| Bias | None | $\beta$ (balances classes) |
| Computational cost | $O(N^2)$ per sample | $O(N^2)$ total |
| Gradient flow | Negatives dominate | Balanced via $\beta$ |
| Convergence speed | Baseline | 2-3× faster |

**Why sigmoid is better**:
1. **Decoupling**: Each pair is treated independently, simplifying optimization
2. **Balance**: $\beta$ explicitly addresses class imbalance
3. **Efficiency**: No need to compute softmax denominator per sample
4. **Flexibility**: $\alpha$ and $\beta$ can be tuned independently

## Part 5: Noisy Labels and Label Smoothing

### 5.1 The Real-World Problem: Mismatched Pairs

In practice, training data for vision-language models is noisy. Image-text pairs from the web may be:
- **Mismatched**: Image shows a dog, text says "cat"
- **Ambiguous**: Image shows multiple objects, text describes only one
- **Incomplete**: Text is a partial description of the image

**The problem**: Treating all pairs as perfect matches leads to:
1. **Overfitting to noise**: Model learns incorrect associations
2. **Brittle representations**: Model fails on out-of-distribution examples
3. **Reduced generalization**: Model doesn't learn robust features

### 5.2 Label Smoothing in Classification

In standard classification, **label smoothing** addresses overconfidence. Instead of hard targets:

$$\mathbf{y} = [0, 0, \ldots, 1, \ldots, 0]$$

we use soft targets:

$$\mathbf{y}_{\text{smooth}} = (1 - \epsilon) \mathbf{y} + \frac{\epsilon}{K} \mathbf{1}$$

where $\epsilon \in [0, 1]$ is the smoothing parameter and $\mathbf{1}$ is the all-ones vector.

**Effect**: The model is encouraged to assign probability $(1 - \epsilon)$ to the correct class and $\frac{\epsilon}{K}$ to each wrong class.

**Why it helps**:
1. **Prevents overconfidence**: Model doesn't push logits to infinity
2. **Regularization**: Encourages smoother decision boundaries
3. **Robustness**: Model is less sensitive to label noise

### 5.3 Label Smoothing for CLIP

For CLIP, we can apply label smoothing to the softmax targets. Instead of:

$$p_i^{\text{target}} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

we use:

$$p_i^{\text{smooth}} = \begin{cases} 1 - \epsilon & \text{if } i = j \\ \frac{\epsilon}{N-1} & \text{if } i \neq j \end{cases}$$

The loss becomes:

$$\mathcal{L}_{\text{CLIP}}^{\text{smooth}} = -\sum_{i=1}^{N} p_i^{\text{smooth}} \log p_i^{\text{pred}}$$

**Interpretation**: We're saying "this image matches this text with probability $1 - \epsilon$, and matches any other text with probability $\frac{\epsilon}{N-1}$."

**Effect on learning**:
- **Positive pairs**: Still encouraged, but not to the exclusion of all else
- **Negative pairs**: Allowed some probability mass, reducing brittleness
- **Robustness**: Model learns to handle ambiguous or noisy pairs

### 5.4 Label Smoothing for SigLIP

For SigLIP, label smoothing is more natural. Instead of hard labels $y_{ij} \in \{-1, +1\}$, we use soft labels:

$$y_{ij}^{\text{smooth}} = \begin{cases} 1 - \epsilon & \text{if } i = j \\ -(1 - \epsilon) & \text{if } i \neq j \end{cases}$$

Or equivalently, we can modify the loss to include a smoothing term:

$$\mathcal{L}_{\text{SigLIP}}^{\text{smooth}} = -\frac{1}{N^2} \sum_{i,j} \left[ y_{ij}^{\text{smooth}} \log \sigma(y_{ij} \cdot (\alpha s_{ij} + \beta)) \right]$$

**Practical implementation**: Often, $\epsilon$ is set based on estimated noise rate in the data (e.g., $\epsilon = 0.1$ if 10% of pairs are mismatched).

### 5.5 The Multi-Class Lower Bound Revisited

Recall that for $K$ classes, the cross-entropy loss has a lower bound $\log K$. With label smoothing:

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon) \log p_c - \frac{\epsilon}{K} \sum_{j=1}^{K} \log p_j$$

At the uniform distribution ($p_j = 1/K$ for all $j$):

$$\mathcal{L}_{\text{smooth}} = -(1-\epsilon) \log(1/K) - \epsilon \log(1/K) = \log K$$

**The lower bound is unchanged**, but the optimal distribution is different:
Without smoothing: $p_c = 1, p_j = 0$ for $j \neq c$ (infinitely confident)

With smoothing: The optimal distribution balances the two terms. Taking the gradient with respect to $p_c$ and setting to zero:

$$\frac{\partial \mathcal{L}_{\text{smooth}}}{\partial p_c} = -\frac{1-\epsilon}{p_c} - \frac{\epsilon}{K p_c} = 0$$

This is satisfied when the model outputs match the smoothed targets:

$$p_c^* = 1 - \epsilon + \frac{\epsilon}{K}, \quad p_j^* = \frac{\epsilon}{K} \text{ for } j \neq c$$

**The minimum loss** with these optimal probabilities:

$$\mathcal{L}_{\text{smooth}}^{\min} = -(1-\epsilon) \log\left(1 - \epsilon + \frac{\epsilon}{K}\right) - \epsilon \log\left(\frac{\epsilon}{K}\right)$$

For small $\epsilon$ and large $K$:

$$\mathcal{L}_{\text{smooth}}^{\min} \approx -(1-\epsilon) \log(1-\epsilon) - \epsilon \log\left(\frac{\epsilon}{K}\right)$$

Using $\log(1-\epsilon) \approx -\epsilon$:

$$\mathcal{L}_{\text{smooth}}^{\min} \approx \epsilon(1-\epsilon) + \epsilon \log K - \epsilon \log \epsilon$$

**Key insight**: The minimum achievable loss is now **higher** than zero, but this is intentional—it prevents overconfitting to potentially noisy labels.


---

## BLIP-2 - Bridging Frozen Models with Learnable Queries

### 6.1 The Motivation: Leveraging Pre-trained Giants

CLIP taught us to align vision and language through contrastive learning. But what if we want to go beyond simple similarity matching? What if we want:
- **Generative capabilities**: Describe images in natural language
- **Question answering**: Answer questions about image content
- **Reasoning**: Perform complex visual reasoning tasks

**The challenge**: Training large vision-language models from scratch is expensive. Can we leverage existing pre-trained models?

**BLIP-2's answer**: Use a **frozen image encoder** and a **frozen language model**, connected by a lightweight **learnable Q-Former** module.

### 6.2 The Architecture: Three Components

**Component 1: Frozen Image Encoder** $f_I$

The image encoder (e.g., ViT - Vision Transformer) converts an image into patch embeddings:

$$\mathbf{V} = f_I(\text{Image}) \in \mathbb{R}^{N_v \times d_v}$$

where:
- $N_v$ is the number of visual tokens (patches)
- $d_v$ is the dimension of visual features

**How patch encoding works**:
1. **Patch extraction**: Divide image of size $H \times W$ into patches of size $P \times P$
   - Number of patches: $N_v = \frac{H}{P} \times \frac{W}{P}$
   - Example: $224 \times 224$ image with $16 \times 16$ patches → $14 \times 14 = 196$ patches

2. **Linear projection**: Each patch $\mathbf{p}_i \in \mathbb{R}^{P^2 \cdot 3}$ (flattened RGB) is projected:
   $$\mathbf{v}_i = \mathbf{W}_{\text{patch}} \mathbf{p}_i + \mathbf{b}_{\text{patch}} \in \mathbb{R}^{d_v}$$

3. **Positional encoding**: Add position embeddings to preserve spatial information:
   $$\mathbf{v}_i \leftarrow \mathbf{v}_i + \mathbf{pos}_i$$

4. **Transformer encoding**: Pass through transformer layers to get contextualized features:
   $$\mathbf{V} = \text{Transformer}([\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_{N_v}])$$

**Component 2: Frozen Large Language Model** $f_{LLM}$

A pre-trained autoregressive language model (e.g., OPT, FlanT5):

$$P(\text{text}) = \prod_{t=1}^{T} P(w_t | w_{<t}, \text{context})$$

**Component 3: Learnable Q-Former** $G_Q$

The Q-Former is a lightweight transformer that bridges vision and language:

$$\mathbf{Z}_Q = G_Q(\mathbf{Q}, \mathbf{V}) \in \mathbb{R}^{L \times d_q}$$

where:
- $\mathbf{Q} \in \mathbb{R}^{L \times d_q}$ are **learnable query embeddings** (typically $L = 32$)
- $\mathbf{V} \in \mathbb{R}^{N_v \times d_v}$ are frozen visual features
- $\mathbf{Z}_Q$ are the output query representations

**The information bottleneck**: $L = 32$ queries must extract relevant information from $N_v \approx 196$ visual tokens. This bottleneck forces the Q-Former to learn a compressed, task-relevant representation.

### 6.3 Q-Former Architecture: Self-Attention and Cross-Attention

The Q-Former consists of $M$ layers (typically $M = 12$). Each layer $\ell$ has three components:

**Input to layer $\ell$**:
- Queries from previous layer: $\mathbf{Q}^{(\ell-1)} \in \mathbb{R}^{L \times d_q}$
- Text tokens (if present): $\mathbf{T}^{(\ell-1)} \in \mathbb{R}^{N_t \times d_q}$
- Visual features (frozen): $\mathbf{V} \in \mathbb{R}^{N_v \times d_v}$

**Step 1: Self-Attention on Queries and Text**

Concatenate queries and text:
$$\mathbf{X}^{(\ell-1)} = [\mathbf{Q}^{(\ell-1)}; \mathbf{T}^{(\ell-1)}] \in \mathbb{R}^{(L + N_t) \times d_q}$$

Apply self-attention:
$$\mathbf{X}_{\text{self}}^{(\ell)} = \text{SelfAttn}(\mathbf{X}^{(\ell-1)}) + \mathbf{X}^{(\ell-1)}$$

**Self-attention mechanism**:
$$\text{SelfAttn}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}_s \mathbf{K}_s^T}{\sqrt{d_k}}\right) \mathbf{V}_s$$

where:
$$\mathbf{Q}_s = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{(L+N_t) \times d_k}$$
$$\mathbf{K}_s = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{(L+N_t) \times d_k}$$
$$\mathbf{V}_s = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{(L+N_t) \times d_v}$$

**What self-attention does**: Queries and text tokens attend to each other, allowing:
- Queries to incorporate textual context
- Text tokens to be informed by query representations

**Step 2: Cross-Attention from Queries to Visual Features**

Extract queries from self-attention output:
$$\mathbf{Q}_{\text{self}}^{(\ell)} = \mathbf{X}_{\text{self}}^{(\ell)}[1:L, :] \in \mathbb{R}^{L \times d_q}$$

Apply cross-attention where **queries attend to visual features**:
$$\mathbf{Q}_{\text{cross}}^{(\ell)} = \text{CrossAttn}(\mathbf{Q}_{\text{self}}^{(\ell)}, \mathbf{V}) + \mathbf{Q}_{\text{self}}^{(\ell)}$$

**Cross-attention mechanism**:
$$\text{CrossAttn}(\mathbf{Q}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}_c \mathbf{K}_c^T}{\sqrt{d_k}}\right) \mathbf{V}_c$$

where:
$$\mathbf{Q}_c = \mathbf{Q}_{\text{self}}^{(\ell)} \mathbf{W}_Q^{\text{cross}} \in \mathbb{R}^{L \times d_k}$$
$$\mathbf{K}_c = \mathbf{V} \mathbf{W}_K^{\text{cross}} \in \mathbb{R}^{N_v \times d_k}$$
$$\mathbf{V}_c = \mathbf{V} \mathbf{W}_V^{\text{cross}} \in \mathbb{R}^{N_v \times d_v}$$

**The asymmetry of cross-attention**: 
- **Query source**: Comes from queries $\mathbf{Q}_{\text{self}}^{(\ell)}$
- **Key and Value source**: Come from visual features $\mathbf{V}$

This is fundamentally different from self-attention where Q, K, V all come from the same source.

**Why this matters**: 
- Queries "ask questions" about the image
- Visual features "provide answers"
- Information flows from $\mathbf{V}$ to $\mathbf{Q}$, not the reverse

**The information bottleneck in action**:
- Attention weights: $\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}_c \mathbf{K}_c^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times N_v}$
- Each of $L = 32$ queries attends to all $N_v = 196$ visual tokens
- Output: $\mathbf{Q}_{\text{cross}}^{(\ell)} = \mathbf{A} \mathbf{V}_c \in \mathbb{R}^{L \times d_v}$

**Interpretation**: Each query learns to aggregate information from relevant image regions. The bottleneck forces queries to extract high-level semantic information rather than memorizing pixel details.

**Step 3: Feed-Forward Network**

Apply FFN to queries:
$$\mathbf{Q}^{(\ell)} = \text{FFN}(\mathbf{Q}_{\text{cross}}^{(\ell)}) + \mathbf{Q}_{\text{cross}}^{(\ell)}$$

Apply FFN to text (if present):
$$\mathbf{T}^{(\ell)} = \text{FFN}(\mathbf{T}_{\text{self}}^{(\ell)}) + \mathbf{T}_{\text{self}}^{(\ell)}$$

where:
$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

**Final output**: After $M$ layers, we get:
$$\mathbf{Z}_Q = \mathbf{Q}^{(M)} \in \mathbb{R}^{L \times d_q}$$

### 6.4 Training Objective 1: Image-Text Contrastive Loss (ITC)

**Goal**: Align query representations $\mathbf{Z}_Q$ with text representations $\mathbf{T}_{\text{enc}}$.

**Setup**: For a batch of $N$ image-text pairs:
- Image queries: $\mathbf{Z}_Q^{(i)} \in \mathbb{R}^{L \times d_q}$ for $i = 1, \ldots, N$
- Text embeddings: $\mathbf{t}^{(i)} \in \mathbb{R}^{d_q}$ for $i = 1, \ldots, N$

**Pooling queries**: Aggregate $L$ queries into a single representation:
$$\mathbf{z}_Q^{(i)} = \text{Pool}(\mathbf{Z}_Q^{(i)}) \in \mathbb{R}^{d_q}$$

Common pooling strategies:
- **Mean pooling**: $\mathbf{z}_Q^{(i)} = \frac{1}{L} \sum_{\ell=1}^{L} \mathbf{Z}_Q^{(i)}[\ell, :]$
- **Attention pooling**: $\mathbf{z}_Q^{(i)} = \sum_{\ell=1}^{L} \alpha_\ell \mathbf{Z}_Q^{(i)}[\ell, :]$ where $\alpha_\ell$ are learned weights

**Similarity matrix**: Compute cosine similarities (after L2 normalization):
$$\mathbf{S}_{Q \to T} = \frac{\mathbf{Z}_Q \mathbf{T}^T}{\|\mathbf{Z}_Q\| \|\mathbf{T}\|} \in \mathbb{R}^{N \times N}$$

where $\mathbf{Z}_Q = [\mathbf{z}_Q^{(1)}, \ldots, \mathbf{z}_Q^{(N)}]^T$ and $\mathbf{T} = [\mathbf{t}^{(1)}, \ldots, \mathbf{t}^{(N)}]^T$.

**Contrastive loss** (exactly like CLIP):
$$\mathcal{L}_{\text{ITC}} = \frac{1}{2N} \sum_{i=1}^{N} \left( \mathcal{L}_{Q \to T}^{(i)} + \mathcal{L}_{T \to Q}^{(i)} \right)$$

where:
$$\mathcal{L}_{Q \to T}^{(i)} = -\log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)}$$

$$\mathcal{L}_{T \to Q}^{(i)} = -\log \frac{\exp(s_{ii} / \tau)}{\sum_{j=1}^{N} \exp(s_{ij} / \tau)}$$

**What this loss achieves**:
- Queries learn to extract image features that are **similar to text embeddings**
- Creates a shared semantic space between vision and language
- Enables zero-shot image-text retrieval

**Attention masking for ITC**: During this stage, queries and text can **fully attend to each other** in self-attention. No masking is applied.

### 6.5 Training Objective 2: Image-Text Matching Loss (ITM)

**Goal**: Train a binary classifier to determine if an image-text pair is matched or mismatched.

**Why this is needed**: ITC learns relative similarity, but ITM learns **absolute matching**. A pair might be relatively similar but still incorrect (e.g., "a dog" vs "a cat" are similar, but not a match).

**Setup**: For each image-text pair, we have:
- Query representations: $\mathbf{Z}_Q \in \mathbb{R}^{L \times d_q}$
- Text representations: $\mathbf{T} \in \mathbb{R}^{N_t \times d_q}$

**Multimodal fusion**: Concatenate queries and text, then pass through a classification head:
$$\mathbf{h} = \text{Pool}([\mathbf{Z}_Q; \mathbf{T}]) \in \mathbb{R}^{d_q}$$

**Binary classification**: Compute matching score:
$$s_{\text{match}} = \mathbf{w}^T \mathbf{h} + b \in \mathbb{R}$$

**The problem**: Like SigLIP, we have class imbalance. In a batch of $N$ pairs:
- **Positive pairs**: $N$ (matched)
- **Negative pairs**: $N \times K$ (mismatched, where $K$ is the number of hard negatives per sample)

**Binary cross-entropy loss** with bias correction:
$$\mathcal{L}_{\text{ITM}} = -\frac{1}{N(1+K)} \sum_{i=1}^{N} \left[ \log \sigma(s_i^+) + \sum_{k=1}^{K} \log \sigma(-s_{ik}^-) \right]$$

where:
- $s_i^+$ is the matching score for positive pair $i$
- $s_{ik}^-$ is the matching score for negative pair $(i, k)$

**Introducing bias like SigLIP**: To handle class imbalance, we modify the matching score:
$$s = \alpha \langle \mathbf{z}_Q, \mathbf{t} \rangle + \beta$$

where:
- $\alpha$ is a scaling parameter (temperature inverse)
- $\beta$ is a bias term

**Setting $\beta = -\log K$**: Following the same logic as SigLIP, we set $\beta = -\log K$ to balance positive and negative gradients.

**Why $\beta = -\log K$?**

At initialization, assume $\langle \mathbf{z}_Q, \mathbf{t} \rangle \approx 0$. Then:
- Positive score: $s^+ = \alpha \cdot 0 + \beta = \beta = -\log K$
- Negative score: $s^- = \alpha \cdot 0 + \beta = \beta = -\log K$

Expected gradients:
- Positive: $\mathbb{E}[\nabla_s \mathcal{L}^+] = -\sigma(-\beta) = -\sigma(\log K) = -\frac{K}{1+K}$
- Negative: $\mathbb{E}[\nabla_s \mathcal{L}^-] = \sigma(\beta) = \sigma(-\log K) = \frac{1}{1+K}$

Total gradient balance:
$$N \cdot \left(-\frac{K}{1+K}\right) + NK \cdot \frac{1}{1+K} = \frac{N(-K + K)}{1+K} = 0$$

**Perfect balance!** The bias ensures that positive and negative pairs contribute equally to learning initially.

**The loss with bias**:
$$\mathcal{L}_{\text{ITM}} = -\frac{1}{N(1+K)} \sum_{i=1}^{N} \left[ \log \sigma(\alpha s_i^+ - \log K) + \sum_{k=1}^{K} \log \sigma(-\alpha s_{ik}^- + \log K) \right]$$

**Hard negative mining**: To make ITM more effective, BLIP-2 uses **hard negatives**:
- For each image, select $K$ texts that have high ITC similarity but are incorrect
- These are "confusing" examples that force the model to learn fine-grained distinctions

**Attention masking for ITM**: Queries and text are **masked from each other** during cross-attention. 

**Why mask?** We don't want information leakage between modalities during matching. The classifier should learn to match based on independent representations, not on attention patterns that could "cheat" by directly comparing tokens.

**How masking works**:
- In self-attention, the attention mask $\mathbf{M} \in \{0, -\infty\}^{(L+N_t) \times (L+N_t)}$ is applied:
  $$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } i, j \text{ both in queries or both in text} \\ -\infty & \text{if } i \text{ in queries, } j \text{ in text, or vice versa} \end{cases}$$

- Modified attention:
  $$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

- When $\mathbf{M}_{ij} = -\infty$, the softmax output is 0, preventing attention between queries and text.

**Notation summary**:
$$\mathbf{M}_{\text{ITM}} = \begin{bmatrix} \mathbf{0}_{L \times L} & -\infty \cdot \mathbf{1}_{L \times N_t} \\ -\infty \cdot \mathbf{1}_{N_t \times L} & \mathbf{0}_{N_t \times N_t} \end{bmatrix}$$

This ensures queries and text develop independent representations before fusion in the classification head.

### 6.6 Training Objective 3: Language Modeling Loss (LM)

**Goal**: Train the Q-Former to generate text conditioned on image queries, preparing it to interface with the frozen LLM.

**Setup**: The input to the language modeling objective is:
$$\mathbf{X}_{\text{LM}} = [\mathbf{Q}; \mathbf{Z}_Q; \mathbf{T}] \in \mathbb{R}^{(L + L + N_t) \times d_q}$$

where:
- $\mathbf{Q}$: Initial learnable queries
- $\mathbf{Z}_Q$: Output queries from Q-Former (after cross-attention with image)
- $\mathbf{T}$: Text tokens

**Autoregressive objective**: Predict each text token given previous tokens and image context:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{N_t} \sum_{t=1}^{N_t} \log P(w_t | w_{<t}, \mathbf{Z}_Q)$$

**Detailed formulation**: For text token at position $t$:
$$P(w_t | w_{<t}, \mathbf{Z}_Q) = \text{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{h}_t)$$

where $\mathbf{h}_t$ is the hidden state at position $t$ after self-attention and cross-attention.

**Causal masking for LM**: To enforce autoregressive generation, we use a **causal mask** on text tokens:

$$\mathbf{M}_{\text{LM}} = \begin{bmatrix} \mathbf{0}_{L \times L} & \mathbf{0}_{L \times L} & \mathbf{0}_{L \times N_t} \\ \mathbf{0}_{L \times L} & \mathbf{0}_{L \times L} & \mathbf{0}_{L \times N_t} \\ \mathbf{0}_{N_t \times L} & \mathbf{0}_{N_t \times L} & \mathbf{M}_{\text{causal}} \end{bmatrix}$$

where $\mathbf{M}_{\text{causal}} \in \{0, -\infty\}^{N_t \times N_t}$ is the standard causal mask:
$$\mathbf{M}_{\text{causal}}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

**Why this masking?**
- Queries can attend to all queries (no restriction)
- Text tokens can attend to all queries (image context is always visible)
- Text tokens can only attend to **previous** text tokens (autoregressive constraint)

**Notation**:
- Position $t$ in text can attend to: $\mathbf{Q}, \mathbf{Z}_Q, w_1, w_2, \ldots, w_{t-1}$
- Position $t$ cannot attend to: $w_t, w_{t+1}, \ldots, w_{N_t}$ (future tokens)

**What this loss achieves**:
- Queries learn to extract image information useful for text generation
- Q-Former learns to produce representations compatible with language modeling
- Prepares the model for interfacing with the frozen LLM

### 6.7 Combined Training Objective

The total loss is a weighted combination:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ITC}} \mathcal{L}_{\text{ITC}} + \lambda_{\text{ITM}} \mathcal{L}_{\text{ITM}} + \lambda_{\text{LM}} \mathcal{L}_{\text{LM}}$$

Typical weights: $\lambda_{\text{ITC}} = 1.0$, $\lambda_{\text{ITM}} = 1.0$, $\lambda_{\text{LM}} = 1.0$

**Training stages**:
1. **Stage 1**: Train Q-Former with all three losses, keeping image encoder frozen
2. **Stage 2**: Connect Q-Former to frozen LLM, train only the projection layer and Q-Former with LM loss

### 6.8 Connecting to the Frozen LLM

After Q-Former training, we connect to the LLM:

**Projection**: Map query outputs to LLM input space:
$$\mathbf{Z}_{\text{LLM}} = \mathbf{W}_{\text{proj}} \mathbf{Z}_Q \in \mathbb{R}^{L \times d_{\text{LLM}}}$$

where $d_{\text{LLM}}$ is the LLM's embedding dimension.

**LLM input**: Concatenate visual tokens with text tokens:
$$\mathbf{X}_{\text{LLM}} = [\mathbf{Z}_{\text{LLM}}; \mathbf{T}_{\text{LLM}}] \in \mathbb{R}^{(L + N_t) \times d_{\text{LLM}}}$$

**Generation**: The LLM generates text autoregressively:
$$P(\text{output} | \text{image}, \text{prompt}) = \prod_{t=1}^{T} P(w_t | w_{<t}, \mathbf{Z}_{\text{LLM}}, \text{prompt})$$

### 6.9 BLIP-2 vs CLIP: Understanding vs Matching

**CLIP's limitation: Bag-of-Words**

CLIP learns to match images and text based on **global similarity**. It treats text as a bag of words:
- "A dog playing with a ball" ≈ "A ball playing with a dog" (CLIP can't distinguish)
- "A red car next to a blue house" ≈ "A blue car next to a red house" (word order doesn't matter)

**Why?** CLIP's contrastive loss only cares about overall similarity, not compositional understanding.

**BLIP-2's advantage: Compositional Understanding**

BLIP-2's three-stage training enables:

1. **ITC**: Learn global alignment (like CLIP)
2. **ITM**: Learn fine-grained matching (distinguish similar but incorrect pairs)
3. **LM**: Learn compositional structure (word order, relationships, reasoning)

**Example improvements**:
- **Spatial relationships**: "A cat on a table" vs "A table on a cat" (BLIP-2 understands, CLIP doesn't)
- **Counting**: "Two dogs" vs "Three dogs" (BLIP-2 can count, CLIP struggles)
- **Attributes**: "A red apple" vs "A green apple" (BLIP-2 distinguishes colors better)
- **Actions**: "A person throwing a ball" vs "A person catching a ball" (BLIP-2 understands actions)

**The role of the LLM**: By connecting to a frozen LLM, BLIP-2 inherits:
- **World knowledge**: Facts and common sense from LLM pre-training
- **Reasoning**: Multi-step inference capabilities
- **Language fluency**: Natural, coherent text generation

### 6.10 BLIP-2 Instruct: From Pre-training to Instruction Following

**BLIP-2 Instruct** extends the base model to follow instructions:

**Training data**: Instruction-following datasets:
- Visual question answering: "What color is the car?" → "Red"
- Image captioning: "Describe this image" → "A red car parked on a street"
- Visual reasoning: "Why is the person holding an umbrella?" → "Because it's raining"

**Training procedure**:
1. Start with pre-trained BLIP-2 (Q-Former + frozen models)
2. Fine-tune on instruction datasets with LM loss:
   $$\mathcal{L}_{\text{instruct}} = -\sum_{t=1}^{T} \log P(w_t | w_{<t}, \mathbf{Z}_Q, \text{instruction})$$

**Key insight**: The Q-Former learns to extract task-relevant information based on the instruction. Different instructions cause different attention patterns in cross-attention.

**Example**: For instruction "What color is the car?":
- Queries attend to car regions in the image
- Queries extract color information
- LLM generates "Red" based on query representations

## LLaVA - Removing the Bottleneck

### 7.1 The Architectural Shift: From Q-Former to Simple Projection

**BLIP-2's complexity**: Three training objectives (ITC, ITM, LM), learnable queries, cross-attention layers, and careful masking strategies.

**LLaVA's insight**: What if we simplify everything?

**LLaVA (Large Language and Vision Assistant)** takes a radically simpler approach:
1. **Frozen vision encoder** $f_I$ (CLIP ViT)
2. **Simple projection layer** $W_{\text{proj}}$ (linear or MLP)
3. **Trainable LLM** $f_{\text{LLM}}$ (Vicuna, LLaMA)

**The key difference**: Instead of a complex Q-Former with multiple objectives, LLaVA uses a simple feedforward projection and trains the LLM itself.

### 7.2 LLaVA Architecture: Mathematical Formulation

**Step 1: Visual Feature Extraction**

Given an image, the frozen vision encoder produces patch embeddings:
$$\mathbf{V} = f_I(\text{Image}) \in \mathbb{R}^{N_v \times d_v}$$

For a ViT with image size $224 \times 224$ and patch size $14 \times 14$:
- Number of patches: $N_v = \frac{224}{14} \times \frac{224}{14} = 16 \times 16 = 256$
- Feature dimension: $d_v = 1024$ (for ViT-L/14)

**Step 2: Visual Projection**

Map visual features to LLM embedding space:
$$\mathbf{Z}_v = W_{\text{proj}}(\mathbf{V}) \in \mathbb{R}^{N_v \times d_{\text{LLM}}}$$

**Two projection options**:

**Option 1: Linear projection**
$$\mathbf{Z}_v = \mathbf{V} W_{\text{linear}} + \mathbf{b} \in \mathbb{R}^{N_v \times d_{\text{LLM}}}$$

where $W_{\text{linear}} \in \mathbb{R}^{d_v \times d_{\text{LLM}}}$ and $\mathbf{b} \in \mathbb{R}^{d_{\text{LLM}}}$.

**Option 2: MLP projection (2-layer)**
$$\mathbf{H} = \text{GELU}(\mathbf{V} W_1 + \mathbf{b}_1) \in \mathbb{R}^{N_v \times d_h}$$
$$\mathbf{Z}_v = \mathbf{H} W_2 + \mathbf{b}_2 \in \mathbb{R}^{N_v \times d_{\text{LLM}}}$$

where:
- $W_1 \in \mathbb{R}^{d_v \times d_h}$, typically $d_h = 4096$
- $W_2 \in \mathbb{R}^{d_h \times d_{\text{LLM}}}$
- $d_{\text{LLM}} = 4096$ for LLaMA-7B

**Step 3: Multimodal Input Construction**

Tokenize the text instruction:
$$\mathbf{T} = \text{Tokenize}(\text{instruction}) \in \mathbb{R}^{N_t \times d_{\text{LLM}}}$$

Concatenate visual and text tokens:
$$\mathbf{X}_{\text{input}} = [\mathbf{Z}_v; \mathbf{T}] \in \mathbb{R}^{(N_v + N_t) \times d_{\text{LLM}}}$$

**Example input sequence**:
[<image_token_1>, <image_token_2>, ..., <image_token_256>, 
 "Describe", "this", "image", "in", "detail", "."]


**Step 4: Autoregressive Generation**

The LLM generates response tokens autoregressively:
$$P(\text{response} | \text{image}, \text{instruction}) = \prod_{t=1}^{T} P(w_t | w_{<t}, \mathbf{Z}_v, \mathbf{T}_{\text{inst}})$$

**Detailed formulation**: At each step $t$, the LLM computes:

1. **Self-attention** over all previous tokens (visual + text):
   $$\mathbf{h}_t = \text{LLM}([\mathbf{Z}_v; \mathbf{T}_{\text{inst}}; w_1, \ldots, w_{t-1}])$$

2. **Output distribution** over vocabulary:
   $$P(w_t | \text{context}) = \text{softmax}(\mathbf{W}_{\text{vocab}} \mathbf{h}_t) \in \mathbb{R}^{|\mathcal{V}|}$$
where $|\mathcal{V}|$ is the vocabulary size (typically 32,000 for LLaMA).

### 7.3 Training Objective: Pure Autoregressive Loss

**LLaVA uses only one loss**: Language modeling loss on the response tokens.

**Formal definition**: Given a training example $(\text{image}, \text{instruction}, \text{response})$:

$$\mathcal{L}_{\text{LLaVA}} = -\sum_{t=1}^{T_{\text{response}}} \log P(w_t | w_{<t}, \mathbf{Z}_v, \mathbf{T}_{\text{inst}})$$

**Key insight**: We only compute loss on **response tokens**, not instruction tokens.

**Why?** The instruction is given; we want the model to learn to generate the correct response conditioned on the image and instruction.

**Masking the loss**: Define a loss mask $\mathbf{m} \in \{0, 1\}^{T_{\text{total}}}$:

$$\mathbf{m}_t = \begin{cases} 
0 & \text{if } t \text{ corresponds to image or instruction tokens} \\
1 & \text{if } t \text{ corresponds to response tokens}
\end{cases}$$

**Masked loss**:
$$\mathcal{L}_{\text{LLaVA}} = -\frac{1}{\sum_{t=1}^{T_{\text{total}}} \mathbf{m}_t} \sum_{t=1}^{T_{\text{total}}} \mathbf{m}_t \log P(w_t | w_{<t}, \mathbf{X}_{\text{input}})$$

**Example**:
Input:  [<img_1>, ..., <img_256>, "What", "color", "is", "the", "car", "?"]
Target: [<img_1>, ..., <img_256>, "What", "color", "is", "the", "car", "?", "The", "car", "is", "red", "."]
Mask:   [   0,    ...,    0,       0,      0,     0,    0,    0,   0,   1,     1,    1,    1,   1]


Loss is computed only on ["The", "car", "is", "red", "."].

### 7.4 Information Bottleneck Analysis: BLIP-2 vs LLaVA

**Information bottleneck theory**: A good representation should:
1. **Compress** the input (remove irrelevant details)
2. **Preserve** task-relevant information

Formally, we want to maximize:
$$\mathcal{I}(Z; Y) - \beta \mathcal{I}(Z; X)$$

where:
- $\mathcal{I}(Z; Y)$ is mutual information between representation $Z$ and target $Y$
- $\mathcal{I}(Z; X)$ is mutual information between representation $Z$ and input $X$
- $\beta$ controls the compression-preservation tradeoff

**BLIP-2's bottleneck**: The Q-Former creates a **hard bottleneck**:

$$\text{Information flow: } \underbrace{\mathbf{V}}_{N_v = 256 \text{ tokens}} \xrightarrow{\text{Cross-Attn}} \underbrace{\mathbf{Z}_Q}_{L = 32 \text{ queries}} \rightarrow \text{LLM}$$

**Compression ratio**: $\frac{32}{256} = 0.125$ (87.5% compression)

**Mathematical analysis**:

The cross-attention mechanism computes:
$$\mathbf{Z}_Q = \text{softmax}\left(\frac{\mathbf{Q}_c \mathbf{K}_c^T}{\sqrt{d_k}}\right) \mathbf{V}_c$$

Each query $\mathbf{z}_q^{(i)}$ is a weighted sum of visual features:
$$\mathbf{z}_q^{(i)} = \sum_{j=1}^{N_v} \alpha_{ij} \mathbf{v}_j$$

where $\alpha_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}}\right)$.

**Information capacity**: Using Shannon's theorem, the maximum information that can flow through $L$ queries is bounded by:
$$\mathcal{I}(Z_Q; V) \leq L \cdot \log_2(d_q) \text{ bits}$$

For $L = 32$ and $d_q = 768$:
$$\mathcal{I}(Z_Q; V) \leq 32 \cdot \log_2(768) \approx 32 \cdot 9.58 = 306.6 \text{ bits}$$

Compare to the original visual information:
$$\mathcal{I}(V; \text{Image}) \leq 256 \cdot \log_2(1024) \approx 256 \cdot 10 = 2560 \text{ bits}$$

**Compression**: $\frac{306.6}{2560} \approx 0.12$ (88% information loss)

**LLaVA's approach**: **No bottleneck** - all visual tokens flow to the LLM:

$$\text{Information flow: } \underbrace{\mathbf{V}}_{N_v = 256 \text{ tokens}} \xrightarrow{W_{\text{proj}}} \underbrace{\mathbf{Z}_v}_{N_v = 256 \text{ tokens}} \rightarrow \text{LLM}$$

**Compression ratio**: $\frac{256}{256} = 1.0$ (0% compression)

**Information capacity**: The projection is a linear/MLP transformation that preserves dimensionality:
$$\mathcal{I}(Z_v; V) \approx \mathcal{I}(V; \text{Image})$$

Assuming the projection is not too lossy, nearly all visual information is available to the LLM.

### 7.5 When to Use Bottlenecks: Task-Dependent Analysis

**Bottlenecks are useful when**:

**1. Computational efficiency is critical**
- Fewer tokens → faster LLM inference
- BLIP-2: 32 visual tokens vs LLaVA: 256 visual tokens
- Inference speedup: $\frac{256}{32} = 8\times$ faster attention computation

**2. Task requires high-level semantic understanding**
- Image classification: "Is this a cat or dog?"
- Image-text retrieval: "Find images matching this caption"
- The bottleneck forces extraction of semantic features, discarding pixel-level details

**3. Training data is limited**
- Fewer parameters in Q-Former (32 queries) → less overfitting
- Bottleneck acts as regularization

**No bottleneck is better when**:

**1. Task requires fine-grained spatial information**
- Object detection: "Where is the cat?" (need precise locations)
- Counting: "How many apples?" (need to attend to all instances)
- OCR: "What text is in the image?" (need pixel-level details)

**2. Task requires reasoning about relationships**
- Spatial: "What is to the left of the car?"
- Compositional: "Is the red ball bigger than the blue ball?"
- More visual tokens → more capacity to represent complex relationships

**3. Computational resources are available**
- Modern GPUs can handle 256 tokens efficiently
- The LLM can learn to selectively attend to relevant tokens

### 7.6 Mathematical Tools for Bottleneck Analysis

**Tool 1: Mutual Information Estimation**

Estimate $\mathcal{I}(Z; Y)$ using the MINE (Mutual Information Neural Estimation) objective:

$$\mathcal{I}(Z; Y) \geq \mathbb{E}_{p(z,y)}[T_\theta(z, y)] - \log \mathbb{E}_{p(z)p(y)}[e^{T_\theta(z, y)}]$$

where $T_\theta$ is a learned critic network.

**Application**: Train a critic to estimate how much information about the task $Y$ is preserved in the representation $Z$.

**Tool 2: Effective Rank**

Measure the "effective dimensionality" of representations using the effective rank:

$$\text{erank}(\mathbf{Z}) = \exp\left(-\sum_{i=1}^{d} p_i \log p_i\right)$$

where $p_i = \frac{\sigma_i}{\sum_j \sigma_j}$ and $\sigma_i$ are singular values of $\mathbf{Z}$.

**Interpretation**:
- $\text{erank}(\mathbf{Z}) = d$: All dimensions are equally used (no compression)
- $\text{erank}(\mathbf{Z}) \ll d$: Only a few dimensions are used (high compression)

**Example**: For BLIP-2 queries $\mathbf{Z}_Q \in \mathbb{R}^{32 \times 768}$:
- If $\text{erank}(\mathbf{Z}_Q) \approx 32$: All queries are informative
- If $\text{erank}(\mathbf{Z}_Q) \approx 10$: Only 10 effective dimensions, suggesting redundancy

**Tool 3: Attention Entropy**

Measure how "focused" attention is using entropy:

$$H(\alpha_i) = -\sum_{j=1}^{N_v} \alpha_{ij} \log \alpha_{ij}$$

where $\alpha_{ij}$ are attention weights from query $i$ to visual token $j$.

**Interpretation**:
- $H(\alpha_i) = 0$: Query attends to a single token (highly focused)
- $H(\alpha_i) = \log N_v$: Query attends uniformly (no focus)

**Application**: Analyze which queries are extracting specific information vs. averaging over the image.

**Tool 4: Probing Classifiers**

Train linear probes on representations to measure task-relevant information:

$$\text{Probe accuracy} = \max_{\mathbf{w}} \mathbb{E}[\mathbb{1}(\mathbf{w}^T \mathbf{z} = y)]$$

**Example**: Train a linear classifier on $\mathbf{Z}_Q$ to predict:
- Object categories
- Spatial locations
- Attributes (color, size)

Higher probe accuracy → more task-relevant information is preserved.

### 7.7 LLaVA Training Data: From GPT-4 to Multimodal Instructions

**The data challenge**: How do we create high-quality instruction-following data for vision-language tasks?

**LLaVA's solution**: Use GPT-4 to generate diverse, high-quality instructions and responses.

**Step 1: Seed Data Collection**

Start with existing image-caption datasets:
- COCO Captions: 120K images with 5 captions each
- Visual Genome: 100K images with dense annotations

**Step 2: GPT-4 Instruction Generation**

For each image with captions and annotations, prompt GPT-4:

Given an image with the following captions:
1. "A red car parked on a street"
2. "A vehicle near a building"
3. "An automobile in an urban setting"

And the following object annotations:
- car (red, sedan, location: center)
- building (gray, tall, location: background)
- street (asphalt, location: bottom)

Generate diverse instruction-response pairs for this image.
Include questions about:
- Objects and their attributes
- Spatial relationships
- Reasoning and inference
- Detailed descriptions


**GPT-4 output example**:
```json
[
  {
    "instruction": "What color is the car in the image?",
    "response": "The car in the image is red."
  },
  {
    "instruction": "Describe the location of the car.",
    "response": "The car is parked on a street in the center of the image, with a tall gray building visible in the background."
  },
  {
    "instruction": "What type of vehicle is shown?",
    "response": "The vehicle is a red sedan."
  },
  {
    "instruction": "What can you infer about the setting?",
    "response": "The setting appears to be an urban environment, as indicated by the presence of a building and an asphalt street."
  }
]
```

**Step 3: Data Diversity and Complexity**

GPT-4 generates three types of instructions:

**Type 1: Conversation** (58K examples)
- Multi-turn dialogues about the image
- Follow-up questions and clarifications
- Example:
    User: What's in the image?
  Assistant: There's a red car parked on a street.
  User: What's behind the car?
  Assistant: There's a tall gray building in the background.
  ```

**Type 2: Detailed Description** (23K examples)
- Comprehensive image descriptions
- Include objects, attributes, spatial relationships, and context
- Example:
  ```
  Instruction: Provide a detailed description of this image.
  Response: The image shows a red sedan parked on an asphalt street in an urban setting. The car is positioned in the center of the frame. Behind the car, there is a tall gray building that dominates the background. The street appears to be in a city environment, with the building suggesting a commercial or residential area. The lighting indicates it might be daytime, and the overall scene conveys a typical urban parking scenario.
  ```

**Type 3: Complex Reasoning** (77K examples)
- Questions requiring inference, counting, comparison
- Example:
  ```
  Instruction: If someone wanted to find this car, what landmarks could they use?
  Response: They could use the tall gray building in the background as a landmark. The car is parked on the street directly in front of this building, making it easy to locate.
  ```

### 7.8 Location Understanding: Synthetic Data Generation

**The problem**: Standard image-caption datasets lack precise spatial information.

**LLaVA's solution**: Generate synthetic location-based data using GPT-4 and bounding box annotations.

**Step 1: Bounding Box Extraction**

Use object detection models (e.g., Faster R-CNN) to extract bounding boxes:
$$\text{bbox}_i = (x_{\min}, y_{\min}, x_{\max}, y_{\max})$$

**Step 2: Spatial Relationship Computation**

Define spatial relationships between objects:

**Relative position**:
$$\text{rel\_pos}(i, j) = \begin{cases}
\text{"left of"} & \text{if } x_{\max}^{(i)} < x_{\min}^{(j)} \\
\text{"right of"} & \text{if } x_{\min}^{(i)} > x_{\max}^{(j)} \\
\text{"above"} & \text{if } y_{\max}^{(i)} < y_{\min}^{(j)} \\
\text{"below"} & \text{if } y_{\min}^{(i)} > y_{\max}^{(j)} \\
\text{"overlapping"} & \text{otherwise}
\end{cases}$$

**Distance**:
$$d(i, j) = \sqrt{(c_x^{(i)} - c_x^{(j)})^2 + (c_y^{(i)} - c_y^{(j)})^2}$$

where $(c_x, c_y)$ is the center of the bounding box.

**Step 3: GPT-4 Prompt for Location Questions**


Given an image with the following objects and their locations:
- Object 1: "car" at bbox (100, 200, 300, 400), center (200, 300)
- Object 2: "building" at bbox (250, 50, 450, 250), center (350, 150)
- Object 3: "tree" at bbox (50, 100, 150, 350), center (100, 225)

Spatial relationships:
- car is below building
- car is right of tree
- tree is left of building

Generate questions about object locations and spatial relationships.

**GPT-4 output**:
json
[
  {
    "instruction": "Where is the car located in the image?",
    "response": "The car is located in the lower-center portion of the image, below the building and to the right of the tree."
  },
  {
    "instruction": "What is to the left of the car?",
    "response": "A tree is to the left of the car."
  },
  {
    "instruction": "Describe the relative positions of the car, building, and tree.",
    "response": "The building is at the top-center of the image. Below it is the car, positioned in the center-bottom area. To the left of the car is a tree, which is also to the left of the building."
  },
  {
    "instruction": "If I wanted to go from the tree to the building, which direction would I move?",
    "response": "You would move to the right and upward, as the building is to the right of and above the tree."
  }
]

**Step 4: Coordinate-Based Questions**

Generate questions that require understanding absolute positions:

json
[
  {
    "instruction": "What object is in the center of the image?",
    "response": "The car is in the center of the image, specifically in the lower-center area."
  },
  {
    "instruction": "What is in the top-right corner?",
    "response": "The top-right corner contains part of the building."
  }
]

**Training format**: The model learns to map visual tokens to spatial concepts:

$$P(\text{"The car is below the building"} | \mathbf{Z}_v, \text{"Where is the car?"})$$

The LLM must learn to:
1. Identify relevant visual tokens (car and building patches)
2. Extract spatial information from their positions
3. Generate appropriate spatial language

### 7.9 Comparative Analysis: BLIP-2 vs LLaVA

Let's analyze how these models handle different types of understanding:

#### 7.9.1 Counting Objects

**Task**: "How many apples are in the image?"

**BLIP-2 approach**:
- 32 queries must capture information about all apples
- Each query might attend to multiple apples
- Bottleneck makes it hard to maintain separate representations for each apple

**Mathematical analysis**: If there are $N_{\text{apples}}$ apples and $L = 32$ queries:
- Best case: Each query represents one apple → can count up to 32 apples
- Worst case: Queries average over apples → counting fails

**Attention pattern**: Query $i$ computes:
$$\mathbf{z}_q^{(i)} = \sum_{j \in \text{apple patches}} \alpha_{ij} \mathbf{v}_j + \sum_{j \notin \text{apple patches}} \alpha_{ij} \mathbf{v}_j$$

If $\alpha_{ij}$ is distributed across multiple apples, the query cannot distinguish individual instances.

**LLaVA approach**:
- All 256 visual tokens are available
- Each apple occupies multiple patches (e.g., 4-9 patches per apple)
- LLM can attend to each apple separately

**Attention pattern**: The LLM can learn to attend to each apple region independently:
$$\text{Attention}(\text{"How many"}) \rightarrow \{\text{apple}_1 \text{ patches}, \text{apple}_2 \text{ patches}, \ldots\}$$

**Expected performance**:
- **BLIP-2**: Struggles with $N_{\text{apples}} > 10$ (bottleneck limitation)
- **LLaVA**: Can count up to $N_{\text{apples}} \approx 20$ (limited by patch resolution)

**Empirical results** (hypothetical):

Image: 5 apples
BLIP-2: "There are several apples" (vague)
LLaVA: "There are 5 apples" (precise)

Image: 15 apples
BLIP-2: "There are many apples" (gives up counting)
LLaVA: "There are approximately 14-16 apples" (close estimate)

#### 7.9.2 Understanding Latent Attributes

**Task**: "What emotion is the person expressing?"

**Challenge**: Emotion is a **latent attribute** - not directly visible, must be inferred from facial features, body language, context.

**BLIP-2 approach**:
- Queries must extract high-level semantic features
- The bottleneck forces abstraction, which is beneficial here
- Queries can learn to represent "happiness", "sadness", etc.

**Mathematical intuition**: The cross-attention learns to extract emotion-relevant features:
$$\mathbf{z}_q^{(\text{emotion})} = \sum_{j \in \text{face patches}} \alpha_j \mathbf{v}_j$$

where $\alpha_j$ is high for patches containing emotion cues (eyes, mouth, eyebrows).

**LLaVA approach**:
- All visual tokens available, but LLM must learn to aggregate them
- More information, but also more noise
- Requires the LLM to learn which patches are relevant

**Expected performance**:
- **BLIP-2**: Good at high-level emotion recognition (happy, sad, angry)
- **LLaVA**: Can capture subtle emotions and context (e.g., "slightly amused", "nervously smiling")

**Example**:

Image: Person with slight smile and raised eyebrows

BLIP-2: "The person appears happy."
LLaVA: "The person has a slight smile and raised eyebrows, suggesting they are pleasantly surprised or mildly amused."

**Why LLaVA is better here**: More visual tokens allow the LLM to capture subtle facial features and context (e.g., body language, surrounding environment) that contribute to emotion understanding.

#### 7.9.3 Spatial Relationships and Relativity

**Task**: "Is the red ball bigger than the blue ball?"

**Challenge**: Requires:
1. Identifying both balls
2. Extracting size information
3. Comparing sizes
4. Understanding relative concepts ("bigger than")

**BLIP-2 approach**:
- Queries must represent both balls and their sizes
- Bottleneck makes it hard to maintain separate, detailed representations
- Comparison might be lossy

**Mathematical analysis**: Suppose query $i$ represents the red ball and query $j$ represents the blue ball:
$$\mathbf{z}_q^{(i)} = \sum_{k \in \text{red ball}} \alpha_{ik} \mathbf{v}_k$$
$$\mathbf{z}_q^{(j)} = \sum_{k \in \text{blue ball}} \alpha_{jk} \mathbf{v}_k$$

The LLM must compare these representations:
$$P(\text{"bigger"} | \mathbf{z}_q^{(i)}, \mathbf{z}_q^{(j)})$$

**Problem**: If size information is not well-preserved in the queries, comparison fails.

**LLaVA approach**:
- All patches for both balls are available
- LLM can directly compare patch counts or spatial extent

**Size estimation**: The LLM can learn to estimate size from the number of patches:
$$\text{size}(\text{red ball}) \approx \sum_{k \in \text{red ball}} 1 = N_{\text{red}}$$
$$\text{size}(\text{blue ball}) \approx \sum_{k \in \text{blue ball}} 1 = N_{\text{blue}}$$

**Comparison**:
$$\text{"bigger"} \Leftrightarrow N_{\text{red}} > N_{\text{blue}}$$

**Expected performance**:
- **BLIP-2**: Struggles with fine-grained size comparisons (e.g., "slightly bigger")
- **LLaVA**: Can make precise comparisons and quantify differences

**Example**:

Image: Red ball (50 pixels diameter), Blue ball (45 pixels diameter)

BLIP-2: "The red ball and blue ball are similar in size."
LLaVA: "The red ball is slightly bigger than the blue ball."

Image: Red ball (80 pixels), Blue ball (40 pixels)

BLIP-2: "The red ball is bigger than the blue ball."
LLaVA: "The red ball is approximately twice the size of the blue ball."

#### 7.9.4 Compositional Understanding

**Task**: "Describe the relationship between the cat, the ball, and the table."

**Challenge**: Requires understanding multiple objects and their pairwise relationships.

**BLIP-2 approach**:
- Queries must capture all three objects and their relationships
- With $L = 32$ queries, can represent multiple objects
- But relationships might be implicit in query representations

**LLaVA approach**:
- All visual tokens for all objects are available
- LLM can explicitly reason about relationships
- Can generate detailed compositional descriptions

**Example**:

Image: Cat on table, ball under table

BLIP-2: "A cat is on a table, and there is a ball nearby."
LLaVA: "A cat is sitting on top of a table. Below the table, on the floor, there is a ball. The cat appears to be looking down at the ball, suggesting it might be interested in playing with it."

**Why LLaVA is better**: More visual tokens + trainable LLM allows learning complex compositional reasoning patterns.

## Qwen-VL 2.5 - Adaptive Bottlenecks and Enhanced Capabilities

## 8.1 Qwen-VL 2.5 Architecture Overview

Qwen-VL 2.5 introduces a novel **adaptive fusion layer** that dynamically adjusts the information bottleneck based on the input. Unlike BLIP-2's fixed 32 queries or LLaVA's no-bottleneck approach, Qwen-VL 2.5 can adaptively choose how many visual tokens to use.

**Key innovation**: The bottleneck size $L_{\text{adapt}}$ is **input-dependent** and **task-dependent**.

### 8.2 Qwen-VL 2.5 Fusion Layer: Mathematical Formulation

**Input**: A sequence containing both visual and textual tokens:
$$\mathbf{X}_{\text{input}} = [\mathbf{V}'; \mathbf{T}] \in \mathbb{R}^{(N_v + N_t) \times d_{\text{model}}}$$

where:
- $\mathbf{V}' \in \mathbb{R}^{N_v \times d_{\text{model}}}$ are projected visual tokens from the vision encoder
- $\mathbf{T} \in \mathbb{R}^{N_t \times d_{\text{model}}}$ are text tokens
- $d_{\text{model}} = 4096$ (typical for Qwen-VL 2.5)

**The fusion layer consists of 4 sequential steps**:

---

#### **Step 1: Joint Self-Attention**

All tokens (visual + text) attend to each other:

$$\mathbf{H}_{\text{self}} = \text{SelfAttn}(\mathbf{X}_{\text{input}}) + \mathbf{X}_{\text{input}} \in \mathbb{R}^{(N_v + N_t) \times d_{\text{model}}}$$

**Detailed computation**:

**Query, Key, Value projections**:
$$\mathbf{Q}_{\text{self}} = \mathbf{X}_{\text{input}} \mathbf{W}_Q^{\text{self}} \in \mathbb{R}^{(N_v + N_t) \times d_k}$$
$$\mathbf{K}_{\text{self}} = \mathbf{X}_{\text{input}} \mathbf{W}_K^{\text{self}} \in \mathbb{R}^{(N_v + N_t) \times d_k}$$
$$\mathbf{V}_{\text{self}} = \mathbf{X}_{\text{input}} \mathbf{W}_V^{\text{self}} \in \mathbb{R}^{(N_v + N_t) \times d_v}$$

where $\mathbf{W}_Q^{\text{self}}, \mathbf{W}_K^{\text{self}} \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $\mathbf{W}_V^{\text{self}} \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

**Attention computation**:
$$\mathbf{A}_{\text{self}} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{self}} \mathbf{K}_{\text{self}}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{(N_v + N_t) \times (N_v + N_t)}$$

**Attention output**:
$$\text{SelfAttn}(\mathbf{X}_{\text{input}}) = \mathbf{A}_{\text{self}} \mathbf{V}_{\text{self}} \in \mathbb{R}^{(N_v + N_t) \times d_v}$$

**Residual connection**:
$$\mathbf{H}_{\text{self}} = \text{SelfAttn}(\mathbf{X}_{\text{input}}) + \mathbf{X}_{\text{input}}$$

**Purpose**: This allows visual and textual tokens to exchange information early, creating context-aware representations.

---

#### **Step 2: Modality-Specific Feed-Forward Networks**

After self-attention, visual and text tokens are processed separately through modality-specific FFNs.

**Split the output**:
$$\mathbf{H}_{\text{self}}^{\text{vis}} = \mathbf{H}_{\text{self}}[1:N_v, :] \in \mathbb{R}^{N_v \times d_{\text{model}}}$$
$$\mathbf{H}_{\text{self}}^{\text{txt}} = \mathbf{H}_{\text{self}}[N_v+1:N_v+N_t, :] \in \mathbb{R}^{N_t \times d_{\text{model}}}$$

**Visual FFN**:
$$\mathbf{H}_{\text{vis}} = \text{FFN}_{\text{vis}}(\mathbf{H}_{\text{self}}^{\text{vis}}) + \mathbf{H}_{\text{self}}^{\text{vis}} \in \mathbb{R}^{N_v \times d_{\text{model}}}$$

**Detailed FFN computation**:
$$\text{FFN}_{\text{vis}}(\mathbf{x}) = \mathbf{W}_2^{\text{vis}} \cdot \text{GELU}(\mathbf{W}_1^{\text{vis}} \mathbf{x} + \mathbf{b}_1^{\text{vis}}) + \mathbf{b}_2^{\text{vis}}$$

where:
- $\mathbf{W}_1^{\text{vis}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, typically $d_{\text{ff}} = 4 \times d_{\text{model}} = 16384$
- $\mathbf{W}_2^{\text{vis}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $\mathbf{b}_1^{\text{vis}} \in \mathbb{R}^{d_{\text{ff}}}$, $\mathbf{b}_2^{\text{vis}} \in \mathbb{R}^{d_{\text{model}}}$

**Text FFN**:
$$\mathbf{H}_{\text{txt}} = \text{FFN}_{\text{txt}}(\mathbf{H}_{\text{self}}^{\text{txt}}) + \mathbf{H}_{\text{self}}^{\text{txt}} \in \mathbb{R}^{N_t \times d_{\text{model}}}$$

$$\text{FFN}_{\text{txt}}(\mathbf{x}) = \mathbf{W}_2^{\text{txt}} \cdot \text{GELU}(\mathbf{W}_1^{\text{txt}} \mathbf{x} + \mathbf{b}_1^{\text{txt}}) + \mathbf{b}_2^{\text{txt}}$$

**Purpose**: Modality-specific processing allows the model to apply different transformations to visual vs. textual information.

---

#### **Step 3: Adaptive Cross-Attention (The Adaptive Bottleneck)**

This is the **key innovation** of Qwen-VL 2.5. Instead of using all $N_v$ visual tokens or a fixed number of queries, the model adaptively selects $L_{\text{adapt}}$ visual tokens.

**Adaptive token selection**: Define a selection mechanism that chooses which visual tokens to use:

$$\mathbf{S} = \text{SelectTopK}(\mathbf{H}_{\text{vis}}, L_{\text{adapt}}) \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$$

**Selection strategies**:

**Strategy 1: Attention-based selection**
Compute importance scores for each visual token:
$$s_i = \mathbf{w}_{\text{select}}^T \cdot \text{tanh}(\mathbf{W}_{\text{select}} \mathbf{h}_{\text{vis}}^{(i)}) \in \mathbb{R}$$

where $\mathbf{w}_{\text{select}} \in \mathbb{R}^{d_h}$ and $\mathbf{W}_{\text{select}} \in \mathbb{R}^{d_h \times d_{\text{model}}}$.

Select top-$L_{\text{adapt}}$ tokens:
$$\mathbf{S} = \mathbf{H}_{\text{vis}}[\text{argsort}(s_1, \ldots, s_{N_v})[-L_{\text{adapt}}:], :]$$

**Strategy 2: Learnable queries with dynamic count**
Use learnable query embeddings, but vary their count:
$$\mathbf{Q}_{\text{learnable}} \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$$

where $L_{\text{adapt}}$ is determined by a gating mechanism:
$$L_{\text{adapt}} = \text{round}\left(\sigma\left(\mathbf{w}_{\text{gate}}^T \cdot \text{mean}(\mathbf{H}_{\text{vis}})\right) \cdot L_{\max}\right)$$

where:
- $\sigma$ is the sigmoid function
- $L_{\max}$ is the maximum number of tokens (e.g., 256)
- $\mathbf{w}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}}}$ is a learnable gating vector

**Strategy 3: Hybrid approach (used in Qwen-VL 2.5)**
Combine selected visual tokens with learnable queries:
$$\mathbf{S} = [\mathbf{Q}_{\text{learnable}}; \mathbf{H}_{\text{vis}}[\text{selected indices}]] \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$$

**Cross-attention computation**:

The selected tokens $\mathbf{S}$ serve as **queries**, and text tokens $\mathbf{H}_{\text{txt}}$ serve as **keys and values**:

$$\mathbf{H}_{\text{cross}} = \text{CrossAttn}(\mathbf{S}, \mathbf{H}_{\text{txt}}) + \mathbf{S} \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$$

**Detailed cross-attention**:

**Query projection** (from selected visual tokens):
$$\mathbf{Q}_{\text{cross}} = \mathbf{S} \mathbf{W}_Q^{\text{cross}} \in \mathbb{R}^{L_{\text{adapt}} \times d_k}$$

**Key projection** (from text tokens):
$$\mathbf{K}_{\text{cross}} = \mathbf{H}_{\text{txt}} \mathbf{W}_K^{\text{cross}} \in \mathbb{R}^{N_t \times d_k}$$

**Value projection** (from text tokens):
$$\mathbf{V}_{\text{cross}} = \mathbf{H}_{\text{txt}} \mathbf{W}_V^{\text{cross}} \in \mathbb{R}^{N_t \times d_v}$$

**Attention weights**:
$$\mathbf{A}_{\text{cross}} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{cross}} \mathbf{K}_{\text{cross}}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{L_{\text{adapt}} \times N_t}$$

**Cross-attention output**:
$$\text{CrossAttn}(\mathbf{S}, \mathbf{H}_{\text{txt}}) = \mathbf{A}_{\text{cross}} \mathbf{V}_{\text{cross}} \in \mathbb{R}^{L_{\text{adapt}} \times d_v}$$

**Residual connection**:
$$\mathbf{H}_{\text{cross}} = \text{CrossAttn}(\mathbf{S}, \mathbf{H}_{\text{txt}}) + \mathbf{S}$$

**Purpose**: Visual tokens attend to text tokens, allowing text to guide which visual information is important.

---

#### **Step 4: Final Feed-Forward Network**

The cross-attention output is processed through a final FFN:

$$\mathbf{H}_{\text{output}} = \text{FFN}_{\text{final}}(\mathbf{H}_{\text{cross}}) + \mathbf{H}_{\text{cross}} \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$$

**FFN computation**:
$$\text{FFN}_{\text{final}}(\mathbf{x}) = \mathbf{W}_2^{\text{final}} \cdot \text{GELU}(\mathbf{W}_1^{\text{final}} \mathbf{x} + \mathbf{b}_1^{\text{final}}) + \mathbf{b}_2^{\text{final}}$$

where:
- $\mathbf{W}_1^{\text{final}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $\mathbf{W}_2^{\text{final}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$

**Output**: The final output $\mathbf{H}_{\text{output}}$ is passed to the next layer or used for generation.

---

### 8.3 Complete Fusion Layer Summary

**Input**: $\mathbf{X}_{\text{input}} = [\mathbf{V}'; \mathbf{T}] \in \mathbb{R}^{(N_v + N_t) \times d_{\text{model}}}$

**Step 1**: Joint self-attention
$$\mathbf{H}_{\text{self}} = \text{SelfAttn}(\mathbf{X}_{\text{input}}) + \mathbf{X}_{\text{input}}$$

**Step 2**: Modality-specific FFNs
$$\mathbf{H}_{\text{vis}} = \text{FFN}_{\text{vis}}(\mathbf{H}_{\text{self}}^{\text{vis}}) + \mathbf{H}_{\text{self}}^{\text{vis}}$$
$$\mathbf{H}_{\text{txt}} = \text{FFN}_{\text{txt}}(\mathbf{H}_{\text{self}}^{\text{txt}}) + \mathbf{H}_{\text{self}}^{\text{txt}}$$

**Step 3**: Adaptive cross-attention
$$\mathbf{S} = \text{SelectTopK}(\mathbf{H}_{\text{vis}}, L_{\text{adapt}})$$
$$\mathbf{H}_{\text{cross}} = \text{CrossAttn}(\mathbf{S}, \mathbf{H}_{\text{txt}}) + \mathbf{S}$$

**Step 4**: Final FFN
$$\mathbf{H}_{\text{output}} = \text{FFN}_{\text{final}}(\mathbf{H}_{\text{cross}}) + \mathbf{H}_{\text{cross}}$$

**Output**: $\mathbf{H}_{\text{output}} \in \mathbb{R}^{L_{\text{adapt}} \times d_{\text{model}}}$

---

### 8.4 Adaptive Bottleneck: How $L_{\text{adapt}}$ Changes

The key innovation is that $L_{\text{adapt}}$ is **not fixed** - it adapts based on:

1. **Input complexity**
2. **Task requirements**
3. **Learned gating mechanism**

**Mathematical formulation of adaptive gating**:

$$L_{\text{adapt}} = f_{\text{gate}}(\mathbf{H}_{\text{vis}}, \mathbf{H}_{\text{txt}})$$

**Option 1: Content-based gating**
$$L_{\text{adapt}} = \text{round}\left(\sigma\left(\mathbf{w}_{\text{gate}}^T \cdot [\text{mean}(\mathbf{H}_{\text{vis}}); \text{mean}(\mathbf{H}_{\text{txt}})] + b_{\text{gate}}\right) \cdot L_{\max}\right)$$

where:
- $\mathbf{w}_{\text{gate}} \in \mathbb{R}^{2 \times d_{\text{model}}}$ is learnable
- $L_{\max} = 256$ (maximum tokens)
- $\sigma$ is sigmoid function

**Option 2: Task-based gating**
Use the text instruction to determine $L_{\text{adapt}}$:

$$L_{\text{adapt}} = \begin{cases}
256 & \text{if task requires fine details (OCR, counting)} \\
128 & \text{if task requires moderate detail (spatial reasoning)} \\
64 & \text{if task requires high-level semantics (classification)} \\
32 & \text{if task is simple (yes/no questions)}
\end{cases}$$

**Task detection**: Use a classifier on text embeddings:
$$\text{task\_type} = \arg\max_k \left(\text{softmax}(\mathbf{W}_{\text{task}} \cdot \text{mean}(\mathbf{H}_{\text{txt}}))\right)_k$$

**Option 3: Learned dynamic gating (Qwen-VL 2.5 approach)**

Train a small MLP to predict $L_{\text{adapt}}$:

$$L_{\text{adapt}} = \text{MLP}_{\text{gate}}([\mathbf{H}_{\text{vis}}; \mathbf{H}_{\text{txt}}])$$

**MLP architecture**:
$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \cdot [\text{mean}(\mathbf{H}_{\text{vis}}); \text{mean}(\mathbf{H}_{\text{txt}})] + \mathbf{b}_1)$$
$$L_{\text{adapt}} = \text{round}\left(\sigma(\mathbf{w}_2^T \mathbf{h}_1 + b_2) \cdot L_{\max}\right)$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{512 \times 2d_{\text{model}}}$
- $\mathbf{w}_2 \in \mathbb{R}^{512}$

**Training**: The gating mechanism is trained end-to-end with the main objective, learning to allocate more tokens when needed.

---

### 8.5 Information Flow Analysis

**Comparison of information bottlenecks**:

| **Model** | **Bottleneck Size** | **Compression Ratio** | **Adaptivity** |
|-----------|---------------------|----------------------|----------------|
| **BLIP-2** | $L = 32$ (fixed) | $\frac{32}{256} = 0.125$ | None |
| **LLaVA** | $L = 256$ (no bottleneck) | $\frac{256}{256} = 1.0$ | None |
| **Qwen-VL 2.5** | $L_{\text{adapt}} \in [32, 256]$ | $\frac{L_{\text{adapt}}}{256} \in [0.125, 1.0]$ | **Adaptive** |

**Information capacity**:

**BLIP-2**: Fixed capacity
$$\mathcal{I}(Z; V) \leq 32 \cdot \log_2(d_{\text{model}}) \approx 32 \cdot 12 = 384 \text{ bits}$$

**LLaVA**: Maximum capacity
$$\mathcal{I}(Z; V) \leq 256 \cdot \log_2(d_{\text{model}}) \approx 256 \cdot 12 = 3072 \text{ bits}$$

**Qwen-VL 2.5**: Adaptive capacity
$$\mathcal{I}(Z; V) \leq L_{\text{adapt}} \cdot \log_2(d_{\text{model}}) \in [384, 3072] \text{ bits}$$

**Key advantage**: Qwen-VL 2.5 can dynamically adjust its information capacity based on the task, achieving:
- **Efficiency** when $L_{\text{adapt}}$ is small (like BLIP-2)
- **Accuracy** when $L_{\text{adapt}}$ is large (like LLaVA)

---

## 8.6 How Qwen-VL 2.5 Handles Different Tasks

### 8.6.1 Optical Character Recognition (OCR)

**Task**: "What text is written in this image?"

**Challenge**: OCR requires:
- Precise character recognition
- Correct character ordering
- Handling of various fonts, sizes, and orientations

**Qwen-VL 2.5's adaptive strategy**:

**Step 1: Task detection**
The model detects that the instruction involves text reading:
$$\text{Instruction: "What text is written..."}$$
$$\Rightarrow \text{Task type: OCR}$$

**Step 2: Adaptive gating**
For OCR, the gating mechanism sets $L_{\text{adapt}}$ to a **high value**:
$$L_{\text{adapt}} = 256 \text{ (maximum)}$$

**Why?** OCR requires fine-grained visual details. Each character might occupy only 2-4 patches, so we need all visual tokens.

**Step 3: Token selection**
Since $L_{\text{adapt}} = 256 = N_v$, all visual tokens are selected:
$$\mathbf{S} = \mathbf{H}_{\text{vis}} \in \mathbb{R}^{256 \times d_{\text{model}}}$$

**Step 4: Cross-attention**
All visual tokens attend to the text instruction:
$$\mathbf{H}_{\text{cross}} = \text{CrossAttn}(\mathbf{H}_{\text{vis}}, \mathbf{H}_{\text{txt}})$$

The attention pattern focuses on text regions:
$$\mathbf{A}_{\text{cross}}[i, :] \approx \begin{cases}
\text{high weights} & \text{if token } i \text{ is on text} \\
\text{low weights} & \text{otherwise}
\end{cases}$$

**Step 5: Character-by-character generation**
The LLM generates text autoregressively:
$$P(\text{"H"} | \mathbf{H}_{\text{output}}, \text{instruction})$$
$$P(\text{"e"} | \text{"H"}, \mathbf{H}_{\text{output}}, \text{instruction})$$
$$P(\text{"l"} | \text{"He"}, \mathbf{H}_{\text{output}}, \text{instruction})$$
$$\vdots$$

**Example**:
Image: Contains text "Hello World" in Arial font

Qwen-VL 2.5 processing:
- Detects OCR task → L_adapt = 256
- All visual tokens available
- Attends to character regions sequentially
- Generates: "The text in the image says 'Hello World'."

Comparison:
BLIP-2 (L=32): "The image contains text" (misses exact characters)
LLaVA (L=256): "The text says 'Hello World'" (correct)
Qwen-VL 2.5 (L_adapt=256): "The text says 'Hello World'" (correct, same as LLaVA)


**Hallucination mitigation**: By using all 256 tokens, Qwen-VL 2.5 has direct access to every character, making it nearly impossible to hallucinate non-existent text.

**Mathematical insight**: The probability of generating a character is directly conditioned on its visual representation:
$$P(c_t | \text{context}) = \text{softmax}(\mathbf{W}_{\text{vocab}} \cdot f(\mathbf{h}_{\text{char}}))$$

where $\mathbf{h}_{\text{char}}$ is the visual token corresponding to the character. If the character isn't in the image, $\mathbf{h}_{\text{char}}$ won't have the right features, and the model won't generate it.

---

### 8.6.2 Counting Objects

**Task**: "How many apples are in this image?"

**Challenge**: Counting requires:
- Identifying all instances of the object
- Distinguishing between instances
- Avoiding double-counting

**Qwen-VL 2.5's adaptive strategy**:

**Step 1: Task detection**
$$\text{Instruction: "How many apples..."}$$
$$\Rightarrow \text{Task type: Counting}$$

**Step 2: Adaptive gating based on object density**

The model estimates the number of objects from visual features:
$$N_{\text{estimated}} = \text{EstimateObjectCount}(\mathbf{H}_{\text{vis}})$$

**Estimation method**: Use a small CNN or attention-based counter:
$$N_{\text{estimated}} = \text{round}\left(\mathbf{w}_{\text{count}}^T \cdot \max_i(\mathbf{H}_{\text{vis}}^{(i)})\right)$$

**Adaptive gating rule**:
$$L_{\text{adapt}} = \min(256, \max(64, 4 \times N_{\text{estimated}}))$$

**Intuition**: Allocate roughly 4 tokens per object, but cap at 256.

**Examples**:
- 5 apples → $L_{\text{adapt}} = 4 \times 5 = 20$ → use 64 (minimum)
- 20 apples → $L_{\text{adapt}} = 4 \times 20 = 80$
- 50 apples → $L_{\text{adapt}} = 4 \times 50 = 200$
- 100 apples → $L_{\text{adapt}} = 4 \times 100 = 400$ → use 256 (maximum)

**Step 3: Token selection**
Select the top $L_{\text{adapt}}$ tokens based on object-relevance scores:

$$s_i = \mathbf{w}_{\text{object}}^T \cdot \text{tanh}(\mathbf{W}_{\text{object}} \mathbf{h}_{\text{vis}}^{(i)})$$

where $\mathbf{w}_{\text{object}}$ is trained to identify object-containing patches.

$$\mathbf{S} = \mathbf{H}_{\text{vis}}[\text{top-}L_{\text{adapt}} \text{ indices}]$$

**Step 4: Cross-attention and counting**
The selected tokens attend to the instruction:
$$\mathbf{H}_{\text{cross}} = \text{CrossAttn}(\mathbf{S}, \mathbf{H}_{\text{txt}})$$

**Step 5: Count aggregation**
The LLM learns to count by aggregating information from the selected tokens:

$$\text{count} = \sum_{i=1}^{L_{\text{adapt}}} \mathbb{1}(\text{token } i \text{ represents a distinct object})$$

**Example**:
Image: 12 apples scattered across the image

Qwen-VL 2.5 processing:
- Detects counting task
- Estimates ~12 objects → L_adapt = 4 × 12 = 48
- Selects 48 most object-relevant tokens
- Each apple occupies ~4 tokens
- Generates: "There are 12 apples in the image."

Comparison:
BLIP-2 (L=32): "There are several apples" (can't count precisely)
LLaVA (L=256): "There are 12 apples" (correct, but uses all tokens)
Qwen-VL 2.5 (L_adapt=48): "There are 12 apples" (correct, more efficient)


**Efficiency gain**: Qwen-VL 2.5 uses only 48 tokens instead of 256, achieving:
- **5.3× faster** inference (fewer tokens to process)
- **Same accuracy** as LLaVA
- **Better than BLIP-2** (which can't count accurately)

**Mathematical analysis**: The counting accuracy depends on having enough tokens to represent each object:

$$\text{Accuracy} \propto \min\left(1, \frac{L_{\text{adapt}}}{N_{\text{objects}} \times k}\right)$$

where $k \approx 3-4$ is the minimum tokens needed per object.

For 12 apples:
- BLIP-2: $\frac{32}{12 \times 4} = 0.67$ (insufficient)
- LLaVA: $\frac{256}{12 \times 4} = 5.33$ (more than enough)
- Qwen-VL 2.5: $\frac{48}{12 \times 4} = 1.0$ (just right)




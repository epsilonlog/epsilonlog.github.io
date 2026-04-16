---
title: "Sample Post: Testing Chirpy Theme Features"
date: 2026-04-10 14:30:00 +0330
categories: [Blogging, Tutorial]
tags: [jekyll, chirpy, markdown, math]
description: A comprehensive sample post to test Chirpy theme features including math rendering, code blocks, and typography.
math: true
---

This is a sample post to test the Chirpy theme setup. Let's explore various features.

## Text Formatting

This paragraph demonstrates **bold text**, *italic text*, and ***bold italic text***.
You can also use ~~strikethrough~~ and `inline code`.

> This is a blockquote. Useful for highlighting important notes.

## Lists

### Unordered List

- First item
- Second item
  - Nested item
  - Another nested item
- Third item

### Ordered List

1. First step
2. Second step
3. Third step

## Code Blocks

Here is a Python example:

~~~python
def attention_score(query, key, d_k):
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    return torch.softmax(scores, dim=-1)

Q = torch.randn(1, 8, 64)
K = torch.randn(1, 8, 64)
attention = attention_score(Q, K, 64)
~~~

Here is some JavaScript:

~~~javascript
function calculateAttention(query, key) {
  const scores = query.map((q, i) =>
    key.reduce((sum, k) => sum + q * k, 0)
  );
  return softmax(scores);
}
~~~

## Mathematics

### Inline Math

The attention mechanism computes
$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
where $d_k$ is the dimension of the key vectors.

### Block Math

The softmax function is defined as:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

For the self-attention mechanism, we compute:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

The KL divergence between two distributions $P$ and $Q$ is:

$$
D_{KL}(P \mid\mid Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

## Tables

| Model | Parameters | Accuracy |
|-------|------------|----------|
| GPT-2 | 1.5B       | 89.2%    |
| BERT  | 340M       | 91.5%    |
| T5    | 11B        | 93.8%    |

## Task Lists

- [x] Set up Jekyll with Chirpy theme
- [x] Configure GitHub Actions
- [x] Write sample post
- [ ] Add more content
- [ ] Customize theme colors

## Footnotes

Here is a sentence with a footnote.[^1]
Here is another one.[^2]

[^1]: This is the first footnote.
[^2]: This is the second footnote with more detail.

## Conclusion

If you can see this post with a proper sidebar, correct math rendering,
and highlighted code blocks — your Chirpy setup is working perfectly. 🚀

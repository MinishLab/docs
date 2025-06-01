---
layout: post
title: "New improvements to model2vec distillation"
categories: [model2vec]
---

We've made a lot of improvements to [model2vec](https://github.com/MinishLab/model2vec) since it came out, many of which target the baseline performance of our distillation process.

This post details how the distillation process has changed over time, and how this has impacted baseline performance of model2vec models. Spoiler alert: if you've distilled a model a couple of months ago, it can really pay off to update model2vec and re-run the distillation process.

# Improvements

Here are the improvements, in order of their appearance. In the last section, we'll contrast all of them, and show their impact MTEB performance.

For all experiments, we distill [`baai/bge-base-en-v1.5`](BAAI/bge-base-en-v1.5) using the default parameters.

## Basic

As a reference, the basic operations we apply when distilling are:

* *Token selection*: we propagate all individual tokens through the model together with an EOS and BOS token, and then _select_ the middle token as the representation.
* *PCA*: apply PCA with a specific number of dimensions (256 for all models).
* *Zipf*: we weight all individual tokens by estimating their frequency using [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law). The short of it is that we assume all tokens in the vocabulary are in _rank order_, and that they follow a power law distribution.

We tried many variations on this theme, including:
* Replacing PCA: we tried [ICA](https://en.wikipedia.org/wiki/Independent_component_analysis), [umap](https://umap-learn.readthedocs.io/en/latest/basic_usage.html), and [T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding). All worked a lot worse.
* Using different propagation strategies: we tried not including BOS/EOS, either only BOS or only EOS, and pooling over the BOS token (i.e., `[CLS]` pooling).
* Using different weighting strategies, including TF-IDF.

None of these really had the desired effect, but feel free to let us know if you come up with something else!

The basic performance of our model with these strategies is on MTEB is **45.34**. We released this model as [m2v_base_output](https://huggingface.co/minishlab/M2V_base_output).

## 1. Pooling

As a first change: we switched from selecting the token to mean pooling, that is, the representation of a token is the mean of the `EOS token BOS` we pass forward through the network. 

In code:

```python
# Before:
embedding = model(["EOS token BOS"])[:, 1]
# Now:
embedding = model(["EOS token BOS"]).mean(1)
```

We also tried a variety of other pooling strategies, including selecting specific tokens, adding queries, and adding prompts.

This raises the average score from **45.34** to **45.91**, but has a larger effect on models that don't perform well to begin with, such as modernBERT-based models.

## 2. SIF weighting

Following this, we replaced the Zipf weighting with a strategy based on the well-known [SIF algorithm](https://openreview.net/pdf?id=SyK00v5xx). In short, this algorithm creates a probability distribution over all tokens in the vocabulary, and downweights very frequent tokens, while upweighting very infrequent tokens. For weighting, it uses the following formula:

```python
sif = alpha / (alpha * proba)
```

Where `proba` is a vector of token probabilities. As before, we use Zipf's law to actually estimate the token probabilities, because we don't actually have access to them. Applying this on top of the mean pooling raises the score from **45.91** to **47.40**.

## 3. Normalization

Normalization has been a part of model2vec from the very first version. This is a boolean flag that, when set to `True`, unit normalizes all output vectors. This is set to `False` by default, but this turns out to be a bad choice. Setting it to `True` has a significant positive effect, especially on retrieval and clustering, and raises the average score from **47.40** to a whopping **47.79**.

# Taking stock

If you want more details, you can find the full table below. As you can see, the improvements we found are general, in the sense that they improve performance for all tasks except PEARL. Anecdotally, this also seems to hold for other models we tried.

|                    |   m2v_base_output |   +mean pooling |   +sif |   +norm |
|:-------------------|------------------:|----------:|----------------------:|-------------------:|
| Average (All)      |             46.79 |     47.32 |                 48.42 |              48.59 |
| Average (MTEB)     |             45.34 |     45.91 |                 47.4  |              47.79 |
| Classification     |             61.25 |     61.43 |                 63.76 |              63.22 |
| Clustering         |             25.58 |     26.13 |                 27.19 |              29.71 |
| PairClassification |             74.9  |     75.23 |                 74.9  |              75.22 |
| Reranking          |             47.63 |     47.73 |                 48.29 |              48.29 |
| Retrieval          |             26.14 |     27.17 |                 28.93 |              28.93 |
| STS                |             68.58 |     69.31 |                 70.89 |              70.89 |
| Summarization      |             29.2  |     29.45 |                 29.32 |              29.35 |
| PEARL              |             54.02 |     54.22 |                 53.88 |              52.73 |
| WordSim            |             49.18 |     49.7  |                 49.63 |              49.63 |

As you can see, adding the improvements increases the scores for distillations across all tasks, with PEARL being the notable exception.

# Where to go from here?

One active area of improvement is to make it a lot easier to tune your model on a specific dataset, so that the model gains knowledge about the specific problem or language you're trying to tackle. This will come up in a next release.

As always, if you have questions, don't hesitate to reach out!

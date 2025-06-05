---
layout: post
title: "ModernBERT support and why it doesn't work"
categories: [model2vec]
---

Our newest shiny release is here! 0.3.8! This is a small release in line for a big one we'll be releasing next week. See [here](https://github.com/MinishLab/model2vec/releases/tag/v0.3.8) for the release notes.

The biggest feature in this release is support for [ModernBERT](https://huggingface.co/blog/modernbert)! As the name implies, ModernBERT is a refresh of the venerable BERT model, trained on more data, with lots of nice tricks; harder, better, faster, stronger. Since its release at the end of last year, many embedders based on ModernBERT have appeared, including:

* [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base)
* [alibaba-nlp/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

And probably many more. 

We didn't support ModernBert out of the box because of a ~~bug~~ design decision, which we fixed in this release. Frustratingly, however, distilling a very good ModernBERT model does not lead to a good model2vec model. This blog post details why we think that is the case: we give a bunch of numbers and some explanations. 

# Distilling ModernBERT

As you probably know, a model2vec model is created by:

1. Downloading an existing sentence transformer
2. Embedding all tokens in the vocabulary (without context)
3. Reducing the resulting embeddings in size using PCA
4. Reweighting them using Zipf weighting

As ModernBERT-based models have about 50k tokens in their tokenizer, this is also how many embeddings our model2vec model will have. 

So, we created a model2vec distill of both ModernBERT-based embedders above. We fully expected this to work well, because in previous experiments, we saw that BERT-based encoder models worked best for model2vec distillation.

Here's the scores on a subset of [MTEB](https://huggingface.co/spaces/mteb/leaderboard) tasks, compared with a straight [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) distill. Note that both `gte-modern-bert-base` and `nomic-ai/modernbert-embed-base` outperform `bge-base-en-v1.5` on the MTEB leaderboard, so we expected a distilled model to also perform better.

|                  | STS  |  WordSim  | Classification |
|------------------|-----:|----------:|---------------:|
| bge-base-en-v1.5 | 69.3  |   49.7     |   62.4          |
| gte-modernbert-base | 66.5 (-2.8) | 25.6 (-24.1) | 60.4 (-2.0) |
| modernbert-embed-base  | 65.1 (-4.2)| 26.1 (-23.6) | 59.4 (-3.0) |

As you can see, that's not the case at all. `bge-base-en-v1.5` outperforms both ModernBERT-based distills on all tasks, and with a huuuuuge margin on `WordSim`. Luckily for us, the `WordSim` task provides us with a good reason for why this is the case. 

## WordSim

First, let's talk about WordSim! Wordsim is a very simple Semantic Textual Similarity task, comprised of 7 datasets, in which the cosine similarity two embeddings of single words are correlated with real-world judgments of similarity.

For example, if `apple` and `pear` are judged to be similar by humans, your model must give them a high cosine similarity in order to score high on this task.

This task is interesting to us because it provides us with an estimate of how good a model2vec model is at modeling lexical similarity without having access to any context. We also see  that being, for model2vec models, performing well on `WordSim` also correlates with performance on other tasks.

What is interesting about the performance of ModernBERT on `WordSim` is that it is atrociously low, lower than any model we've seen before, and that it does not seem to correlate at all with performance on other tasks, on which it scores lower, but not atrociously low. 

But why could this be the case, and why would it hold for both models? Because it seems to hurt both models equally, it looks like something in the base model is to blame. 

In our view, the answer is likely to be the tokenizer used in ModernBERT. ModernBERT's tokenizer, unlike the traditional `BERT` tokenizer, which is used in a lot of embedders, is a byte-pair encoding (BPE) tokenizer. To see what this means, let's take a look at five random BPE tokens from ModernBERT's tokenizer:

```
Ġnickel
ercul
tar
^),
encephal
```

As you can probably see, these tokens are not very likely to be informative by themselves: we can't just embed `ercul` and expect something useful. In contrast, here's five tokens from the `WordPiece`-based BERT tokenizer:

```
lastly
##ect
electro
defendants
ventured
```

As you can see, the `WordPiece` tokenizer has tokens that are more easily interpreted as words. Because BPE tokens are less likely to be words or naturally occurring suffixes, the model likely has to perform more operations to contextualize words, making it a bad fit for uncontextualized embeddings, such as model2vec models.
 
In addition: the tokenizer used in ModernBERT is a _cased_ tokenizer, which means that it contains both upper- and lowercase tokens. But, again, without any contextual cues, there is very little difference between upper- and lowercase tokens.

We think that both of these factors combined, but especially the BPE tokens, lead to low performance of the distilled model. The fact that both of the ModernBERT based models suffer from the same issue shows that the issue is likely caused by the base model, and not the specific fine-tuning strategy used.

## Fixes we tried

Of course, we realize you might be skeptical after reading this, so here's some things we tried:

* Using CLS pooling
* Using mean pooling
* Pooling by selecting the wordpiece
* Reversing the order of the BOS/EOS tokens
* Not applying PCA
* Not applying Zipf

And all combinations of the above. 

## Future work

Support for mutating BPE tokenizers in model2vec is lacking: we don't allow vocabulary changes for BPE tokenizers, but we do allow it for WordPiece tokenizers. 

If token removal was allowed, we could test whether the casing affects performance. If adding tokens to the tokenizer was allowed, we could see whether adding the words in `WordSim` would improve performance.

So one thing on our roadmap, but a very low priority one, is to add support for token addition and/or removal to model2vec. If you have an idea on how to do it, please let us know!

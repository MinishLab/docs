---
layout: post
title: "Tokenlearn 0.2.0"
categories: [Tokenlearn]
---

We've released a new version of tokenlearn! It contains usability improvements, fixes some bugs, and has a new learning algorithm under the hood that improves performance. Read on to see what it does and how you can use it.

## Why use tokenlearn?

Tokenlearn is a way to improve a distilled Model2Vec model by performing an additional knowledge distillation step using the base model (the sentence transformer you distilled) and a distilled Model2Vec model. The Model2Vec model is trained to directly mimic the vectors produced by the base model, which leads to massive improvements. Notably, this does not require any labeled data.

As an example: our new tokenlearn version was used to train our multilingual flagship model, [potion-multilingual-128M](https://huggingface.co/minishlab/potion-multilingual-128M). This model performs at about the same level as [static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) (which we will call MRL). The main difference between the two is how they were trained: MRL has been trained on 8.5 million cross-lingually aligned sentence pairs, while potion-multilingual has only been trained on _2 million random C4 passages_. This shows the power of tokenlearn! You can adapt any Model2Vec model to a specific domain with a small number of short documents, no annotations needed.

## How does it work?

Before starting on what's new, let's first go into how you can use tokenlearn. First, you need to select a base model, i.e., a sentence transformer you like using, and a dataset from which you will sample passages. For this, you need to use the `featurize` function.

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from tokenlearn.featurize import featurize

my_corpus = load_dataset("allenai/c4", "en", split="train", streaming=True)
model = SentenceTransformer("baai/bge-base-en-v1.5")
output_dir = "my_corpus_featurized"

featurize(
    dataset=my_corpus,
    model=model,
    output_dir=output_dir,
    max_means=2_000_000,  
    batch_size=32,
    text_key="text"
)

```

Leave this running for a while, and you will get a set of documents and means in `output_dir`. Note that this script can be resumed, if the arguments are the same, the embedding computation will pick up where you left it. Now that you have the documents in this directory, you can fit a model on them.

```python
from tokenlearn.train import train_model
from tokenlearn.utils import 

model_name = "baai/bge-base-en-v1.5"
data_dir = "my_corpus_featurized"
vocab_size = 250_000

# Collect paths for training data
paths = sorted(Path(data_dir).glob("*.json"))
train_txt, train_vec = collect_means_and_texts(paths)

model = train_model(
    model_name, 
    train_txt, 
    train_vec, 
    device=None, 
    vocab_size=vocab_size, 
    pca_dims=512
)

```

Running this command will get you a trained potion-like model, specifically fit for your domain. Two relevant options to keep in mind are `vocab_size` and `pca_dims`. These control the number of rows and columns in your embedding matrix, respectively. 

In general, setting `pca_dims` to 256 or 512 should be good enough for most problems, and depends on the explained variance of your target vectors. 

Setting the `vocab_size` parameter is more complicated. If `vocab_size` is > 0, we tokenize all texts before training, and select `vocab_size` words to add to the vocabulary of the distilled model based on their frequency. Whether this is useful really depends on the size of your training corpus, and how well it matches with your downstream task. If there's a lot of lexical overlap between the two, you can see a large improvement in performance, although at significant memory costs, as each added vocabulary item adds a whole row to your embedding matrix. Even setting `vocab_size` to 0 will improve performance over a raw distill, however.

## What does it do?

In short, `tokenlearn` training:

1. distills a Model2Vec model for you from a base model
2. adds vocabulary (if any) to the vocabulary of your model
3. perform PCA on the target embeddings we made using the base model
4. perform knowledge distillation from the Model2Vec model to the target embeddings

The knowledge distillation step is extremely simple: we simply reduce the Mean Squared Error (MSE) between the output vectors of the Model2Vec model and the output vectors of the base model, using a held-out set to perform early stopping. We separately optimize the embeddings and the norms of the static model, because we want to decouple the semantic of the token embeddings and the weight they have in a mean, and also want to encourage the model to pay attention to the weight each individual token has.

Applying PCA to both the base model and output embeddings turns out to be extremely important. If this is not done, the knowledge distillation step does not work at all.

## Differences between the new and old tokenlearn

In the old tokenlearn, we also applied a post-processing step, wherein we applied PCA over the learned weights, and then applied a [SIF-like](https://openreview.net/pdf?id=SyK00v5xx) transform to the embeddings. These steps are now no longer necessary.

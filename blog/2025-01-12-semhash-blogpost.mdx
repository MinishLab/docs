---
layout: post
title: "semhash: deduplication and dataset multitool"
categories: [Semhash]
---

We're super excited to announce the release of [semhash](https://github.com/MinishLab/semhash), our semantic deduplication and dataset multitool (other features coming soon).

# Introduction

One area of recent interest, especially around training Large Language Models (LLMs), is that having a lot of data is great, but having a little bit less _high quality_ data is even better. A good example of this can be found in the [fineweb blogpost](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), where the authors start from a really big set of common crawl dumps, on which they perform many quality checks, including dedupication and a suite of quality checks.

At Minish, we're interested in unlocking new possibilities by making very fast models. As you may know, we created the best smallest fast model in the world, [potion-base-8m](https://huggingface.co/minishlab/potion-base-8M). One of the areas we are interested in is `approximate deduplication`: we want to remove documents that are semantically very similar from a corpus. Previous text deduplication algorithms, like minhash or simhash, operate on character or word ngrams, and therefore only find similarity between sequences that are orthographically similar, and ignore semantic similarity.

While deduplication sounds like something that can only benefit LLM training, it can also be really beneficial to check small datasets for overlap: having even approximate overlap between train and test leads to performance overestimation, and having approximate duplicates in train leads to wasted compute, overestimation of feature importance, and a potential host of other issues. 

Additionally, deduplication techniques can also be used to give you a bird's eye view of larger datasets: checking approximate duplicates using `semhash` takes (milli)seconds, and allows you to see which items from your dataset look alike. If these make sense: great! If there are no duplicates... also great! Everything is better than training on incorrect data.

# How can I use deduplication?

Here's some cool use-cases to give you an idea on when deduplication makes sense:

## Classification

As mentioned above, it is important that there is no overlap in information between your train and test splits. Having overlap generally means that you overestimate performance, because the model no longer needs to generalize to perform well. Removing duplicates from within the train set, however, can also be very useful. Having a large number of duplicates of the same record in the training set makes the model overestimate the importance of the features of that record, and, in any case, leads to wasted compute and an overestimation of model fit.

## RAG systems

Duplicates in RAG systems sounds like something rare, until you consider that most RAG systems are built using chunks: while having completely duplicated documents will probably be rare, having duplicate chunks across documents or within documents is a lot more common. Having duplicate chunks in your knowledge base increases storage costs, increases the risk of retrieving irrelevant chunks, and forces you to implement diversification strategies much sooner than necessary.

## Explain your corpus

By running `semhash` with a low threshold, you can quickly get an overview of which documents are similar to others, and which aren't. This gives you a good idea of what to focus on, what kind of things are missing from your data, and how your documents relate to one another.

# How does it work?

At its core, `semhash` takes as input a collection of strings or dictionaries. You first initialize a model using set of reference documents, and then use this set of documents to deduplicate an incoming set. Any incoming document that is similar to a document from the reference set is removed, and stored separately with its approximate duplicates from the reference set.

```python
from datasets import load_dataset

from semhash import SemHash

dataset = load_dataset("ag_news")
train = dataset["train"]
test = dataset["test"]

# This creates an index over your train set. All records are stored in their entirety.
semhash = SemHash.from_records(records=train, columns=["text"])
# This deduplicates your texts with reference to `train`. Any items occurring in train are
# removed from test.
result = semhash.deduplicate(test, threshold=0.9)

# Set without duplicates
result.deduplicated

# Duplicates
result.duplicates
```

During fitting, all document are first encoded by an encoder. The default encoder is [potion-base-8m](https://huggingface.co/minishlab/potion-base-8M), a [model2vec](https://github.com/MinishLab/model2vec) model. The documents are then stored in a [vicinity](https://github.com/MinishLab/vicinity) vector store, backed by [usearch](https://github.com/unum-cloud/usearch). Then, for an incoming set of documents, we first encode them using the specified encoder, and then retrieve the nearest neighbors from the vector store. Every incoming document that has a nearest neighbor with a similarity above the threshold gets removed.

Because all of these components are very fast, deduplicating even really large datasets only takes minutes. For example, deduplicating the entire [Squad-2.0 dataset](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset, which has 130000 samples, only takes 7 seconds. This includes vectorization, fitting the index, and the actual deduplication. Smaller datasets only take a fraction of this time, while even datasets containing millions of documents only take minutes. For a comprehensive benchmark, see [our benchmarks](https://github.com/MinishLab/semhash?tab=readme-ov-file#benchmarks).

## Explainability

`semhash` can also be used to investigate your dataset. By using `self_deduplicate`, you can deduplicate the training set itself, which we will use as a jumping off point:

```python
from datasets import load_dataset

from semhash import SemHash

dataset = load_dataset("ag_news")
train = dataset["train"]
test = dataset["test"]

# This creates an index over your train set. All records are stored in their entirety.
semhash = SemHash.from_records(records=train, columns=["text"])
result = semhash.self_deduplicate(threshold=0.9)
```

Let's dive into what you can do with the `result`. First off, you can just get all deduplicated records:

```python
result.deduplicated
```

These records are exactly the records you put in, allowing you to use `semhash` within other ML pipelines. `semhash` doesn't change your data, it just reduces it in size.

You can easily see the proportion of records that were duplicates:

```python
result.duplicate_ratio
```

or exact duplicates:

```python
result.exact_duplicate_ratio
```

You can also see what got marked as a duplicate, and _why_. Each duplicated document gets stored together with the examples from the index that caused it to be marked as such. Exact duplicates get marked as such. The following code example demonstrates basic usage. 

```python
for duplicated_record in results.duplicates:
    print(duplicated_record.record)
    if duplicated_record.exact:
        print("Exact match")
        continue
    for index_duplicate in duplicated_record.duplicates:
        print(index_duplicate)
    print("-" * 25)
```

For ease of use, we also provide a helper function that shows you the _least_ similar deduplication record in your set of duplicates:

```python
result.get_least_similar_from_duplicates(1)
```

If this record still makes a lot of sense to be called a duplicate with reference to the record it duplicated, your duplication strategy makes sense! If it doesn't you can choose to re-threshold your result set. By doing this, you create a new threshold, thereby removing duplicates. As follows:

```python
print(result.duplicate_ratio)
result.rethreshold(0.95)
print(result.duplicate_ratio)
```

So, a general strategy could be to start with a relatively low threshold, unilt the results returned by `result.get_least_similar_from_duplicates` start making sense. In our experiments, however, a threshold if 0.9, which is the default, works fine, but be sure to check for your individual use-cases.

# Multi-column data

`semhash` also supports multi-column datasets, allowing you to deduplicate datasets that have text in multiple columns. For example, in QA datasets, you don't just want to deduplicate similar questions or similar contexts, but you want to only count items in which both fields are sufficiently similar as duplicated.

This is a difficult problem to tackle, but `semhash` can also handle this.

The following snippet demonstrates how this works:

```python
from datasets import load_dataset

from semhash import SemHash

dataset = load_dataset("rajpurkar/squad_v2")
train = dataset["train"]

# This creates an index over your train set. All records are stored in their entirety.
semhash = SemHash.from_records(records=train, columns=["context", "question"])
result = semhash.self_deduplicate(threshold=0.9)
```

This computes the similarity and only returns records for which both fields are similar. 

# Conclusion

Semhash is great! [Get semhash here](https://github.com/MinishLab/semhash)!
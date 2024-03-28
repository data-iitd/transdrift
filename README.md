# TransDrift

This repository is the official implementation of [TransDrift](https://arxiv.org/abs/2206.08081).

## TransDrift: Modeling Word-Embedding Drift using Transformer
Authors: [Nishtha Madaan](https://nishthaa.github.io/), Prateek Chaudhury, Nishant Kumar, and [Srikanta Bedathur](https://www.cse.iitd.ac.in/~srikanta/)

In modern NLP applications, word embeddings are a crucial backbone that can be readily shared across a number of tasks. However as the text distributions change and word semantics evolve over time, the downstream applications using the embeddings can suffer if the word representations do not conform to the data drift. Thus, maintaining word embeddings to be consistent with the underlying data distribution is a key problem. In this work, we tackle this problem and propose TransDrift, a transformer-based prediction model for word embeddings. Leveraging the flexibility of transformer, our model accurately learns the dynamics of the embedding drift and predicts the future embedding. In experiments, we compare with existing methods and show that our model makes significantly more accurate predictions of the word embedding than the baselines. Crucially, by applying the predicted embeddings as a backbone for downstream classification tasks, we show that our embeddings lead to superior performance compared to the previous methods.

## Citation
```
@unpublished{
TransDrift,
title={TransDrift: Modeling Word-Embedding Drift using Transformer},
author={Nishtha Madaan and Prateek Chaudhury and Nishant Kumar and Srikanta Bedathur},
url={https://arxiv.org/abs/2206.08081}
}
```

##  Training TransDrift model

To train the TransDrift model mentioned in the paper update script/arch.py, run this command:

```
python3 script/transDrift.py
```

## Performing downstream task

To perform downstream task mentioned in the paper for amazon dataset update downstream/amazon/arch.py, run this command:

```
python3 downstream/amazon/downstream.py
```

To perform downstream task mentioned in the paper for yelp dataset update downstream/yelp/arch.py, run this command:

```
python3 downstream/yelp/downstream.py
```



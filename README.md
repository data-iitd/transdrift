# TransDrift

This repository is the official implementation of [TransDrift](https://arxiv.org/abs/2206.08081).

## TransDrift: Modeling Word-Embedding Drift using Transformer
Authors: [Nishtha Madaan](https://nishthaa.github.io/), [Prateek Chaudhury](https://www.linkedin.com/in/prateek-chaudhury/), [Nishant Kumar](https://www.linkedin.com/in/nishant-kumar-91b6901a5/), and [Srikanta Bedathur](https://www.cse.iitd.ac.in/~srikanta/)

In modern NLP applications, word embeddings are a crucial backbone that can be readily shared across a number of tasks. However as the text distributions change and word semantics evolve over time, the downstream applications using the embeddings can suffer if the word representations do not conform to the data drift. Thus, maintaining word embeddings to be consistent with the underlying data distribution is a key problem. In this work, we tackle this problem and propose TransDrift, a transformer-based prediction model for word embeddings. Leveraging the flexibility of transformer, our model accurately learns the dynamics of the embedding drift and predicts the future embedding. In experiments, we compare with existing methods and show that our model makes significantly more accurate predictions of the word embedding than the baselines. Crucially, by applying the predicted embeddings as a backbone for downstream classification tasks, we show that our embeddings lead to superior performance compared to the previous methods.

## Citation
```
@inproceedings{10.1145/3589335.3651894,
author = {Madaan, Nishtha and Chaudhury, Prateek and Kumar, Nishant and Bedathur, Srikanta},
title = {TransDrift: Modeling Word-Embedding Drift using Transformer},
year = {2024},
isbn = {9798400701726},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589335.3651894},
doi = {10.1145/3589335.3651894},
abstract = {In modern NLP applications, word embeddings are a crucial backbone that can be readily shared across a number of tasks. However, as the text distributions change and word semantics evolve over time, the downstream applications using the embeddings can suffer if the word representations do not conform to the data drift. Thus, maintaining word embeddings to be consistent with the underlying data distribution is a key problem. In this work, we tackle this problem and propose TransDriftfootnoteCodebase: https://github.com/data-iitd/transdrift , a transformer-based prediction model for word embeddings. Leveraging the flexibility of transformer, our model accurately learns the dynamics of the embedding drift and predicts the future embedding. In experiments, we compare with existing methods and show that our model makes significantly more accurate predictions of the word embedding than the baselines. Crucially, by applying the predicted embeddings as a backbone for downstream classification tasks, we show that our embeddings lead to superior performance compared to the previous methods.},
booktitle = {Companion Proceedings of the ACM on Web Conference 2024},
pages = {1388–1393},
numpages = {6},
keywords = {drift, transformers, word2vec},
location = {<conf-loc>, <city>Singapore</city>, <country>Singapore</country>, </conf-loc>},
series = {WWW '24}
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



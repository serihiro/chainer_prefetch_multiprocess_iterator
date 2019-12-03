# WARNING!!
- All souce codes stored in this repository is just a part of all for now (3/12/2019).
- So now you can not execute the benchmark or example in this repository.
- I will update this repository so that everyone can execute all sample scripts.
- Please wait for my returning to Japan after BDCAT '19 :bowing_man:

# Chainer PrefetchMultiprocessIterator
- This is a reference implementation of [this paper](https://dl.acm.org/citation.cfm?id=3368768) [1].

# Overview
- This is an iterator implementation for [Chainer](https://chainer.org/)
- This iterator executes prefetching from slow storage (such like network connected parallel file systems, e.g., Lustre) into fast storage (such like local SSD), and generating mini-batches in same time.

## Design
- This implementation is designed as a pipeline which consists of three stages.
![](https://raw.githubusercontent.com/serihiro/chainer_prefetch_multiprocess_iterator/master/prefetch_multiprocess_iterator_spec.png)

## Timeline in executing
![](https://raw.githubusercontent.com/serihiro/chainer_prefetch_multiprocess_iterator/master/proposed_index_list_flow.png)

# Requirements
- Python >= 3.6

# Dependencies
- Chainer >= 6.4

# References

[1] Kazuhiro Serizawa and Osamu Tatebe. 2019. Accelerating Machine Learning I/O by Overlapping Data Staging and Mini-batch Generations. In Proceedings of the 6th IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT '19). ACM, New York, NY, USA, 31-34. DOI: https://doi.org/10.1145/3365109.3368768

# Chainer PrefetchMultiprocessIterator
- This is a reference implementation of [this study](http://id.nii.ac.jp/1001/00198056/) [1].

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

[1] Kazuhiro Serizawa and Osamu Tatebe. Automatic optimization for accelerating deep neural network training. In IPSJ SIG Technical Report, Vol. 2019-HPC-168, pp. 1–10, 2019 (In Japanese).

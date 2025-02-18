---
layout: post
title: Published two new preprints on flow/diffusion models. 
date: 2025-02-18
inline: false
related_posts: false
---

Published two new preprints on flow/diffusion models. 

* [Greed is Good: Guided Generation from a Greedy Perspective](https://arxiv.org/abs/2502.08006). We present a unifying theory of *training-free* guided generation from the perspective of a greedy strategy.
In particular, we view posterior sampling as a greedy strategy of end-to-end optimization techniques establishing a connection between the different families of training-free guided generation algorithms.

* [A Reversible Solver for Diffusion SDEs](https://arxiv.org/abs/2502.08834).
We propose an *algebraically reversible* solver for diffusion SDEs which can invert real samples back into the noise distribution.
Such an approach has no inversion error when using a single realization of the Wiener process both forwards and backwards in time.
We empirically illustrate that our approach has better stability over other inversion techniques making it more useful for image editing tasks.

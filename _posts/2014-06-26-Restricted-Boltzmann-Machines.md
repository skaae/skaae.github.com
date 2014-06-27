---
comments : True
layout: post
title: Training Restricted Boltzmann Machines
---
This post tries to reproduce a figure 4 in the paper ["Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf). 

The figure shows samples drawn from two different RBM's trained on the MNIST data set. The first RBM is trained with persistent contrastive divergence (PCD) and the second RBM is trained with contrastive divergence (CD). 

A RBM is a two layer bipartite graph. The bottom layer is usually called the visible layer and the top layer is the hidden layer. The 

Here is an example MathJax inline rendering \\( 1/x^{2} \\), and here is a block rendering: 
\\[ \frac{1}{n^{2}} \\]

![RBM]({{ site.url }}/downloads/rbm.png)
## Contrastive divergence
<iframe src="//www.youtube.com/embed/tD3kQmqNHw0" width="500" ></iframe>


## Persistent Contrastive Divergence 
<iframe src="//www.youtube.com/embed/c0xdBV70fgE" width="500" ></iframe>

% hinton DBN
The network was trained in two stages – pretraining and fine-tuning. The layer-by-
layer pretraining was the same as in the previous section, except that when training the top layer of 2000 feature detectors, each “data” vector had 510 components. The first 500 were the activation probabilities of the 500 feature detectors in the penultimate layer and the last 10 were the label values. The value of the correct label was set to 1 and the remainder were set to 0. So the top layer of feature detectors learns to model the joint distribution of the 500 penultimate features and the 10 labels.



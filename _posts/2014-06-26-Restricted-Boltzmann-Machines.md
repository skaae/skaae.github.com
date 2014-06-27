---
comments : True
layout: post
title: Training Restricted Boltzmann Machines
---
This post tries to reproduce a figure 4 in the paper ["Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf). 

The figure shows samples drawn from two different RBM's trained on the MNIST data set. The first RBM is trained with persistent contrastive divergence (PCD) and the second RBM is trained with contrastive divergence (CD). 

A RBM is a two layer bipartite graph which tries to model some distribution over data. The bottom layer is usually called the visible layer and the top layer is the hidden layer. In our model the units in the visible and hidden layer will be binary stochastic units. The RBM is a energy based model where the energy is given by:

$$ 
E(\mathbf{v},\mathbf{h}) = -\sum_{i\in \text{visible}} b_i v_i -\sum_{i\in \text{hidden}} c_j h_j - \sum^{}_{i,j} v_i h_j w_{ij} 
$$

Here  $$ w_{ij} $$ is the weight betweeen visible unit i and hidden unit j. $$ b_{i} $$ is the bias for visible unit i and $$ c_j $$ is the bias for hidden unit j. We define the joint probability of a visible and a hidden vector to be:

$$
p(\mathbf{v},\mathbf{h}) = \frac{1}{Z}e^{-E(\mathbf{v},\mathbf{h})}
$$

$$
Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
$$
Here $$Z$$ is the partition function given by summing over all possible pairs of visible and hidden vectors.
The probability that the network assigns to a visible vector, v, is given by summing over all possible hidden vectors:

$$
p(\mathbf{v}) = \frac{1}{Z} \sum_{\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
$$

We want the RBM to assign high probability, i.e low energy, to images in our dataset. To do this we adjust the weights and biases such that images in the data set gets lower energy and other images get higher energy, see [A practical guide to training restricted Boltzmann machines](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) for a better explanation. 
The update rule for stochastic gradient *ascent* can be shown to be:

$$
\Delta w_{ij} = \epsilon (<v_i h_j>_\text{data} - <v_i h_j>_\text{model})
$$ 

Where $$<>$$ denote expectation under the distribution specified by the subscript that follows. 
Getting a sample from $$<v_i h_j>_\text{data}$$ is easy, by noting that the probability of a hidden unit given a visible vector is given by:

$$
p(h_j = 1 | \mathbf{v}) = \sigma(c_j + \sum_i v_i w_{ij})   \qquad \text(RBMUP)
$$

and for visible unit i the probability is given by:

$$
p(v_i = 1 | \mathbf{h}) = \sigma(b_i + \sum_j h_j w_{ij})  \qquad \text(RBMDOWN)
$$

Here $$\sigma$$ is the sigmoid function. Getting a sample from $$<v_i h_j>_\text{model}$$ is can be done by gibbs sampling. We start at a random visible vector and perform alternating gibbs sampling for a long time, in gibbs sampling we use the above two equation to repeatedly update  the hidden units given the visible units and then the visible units given the hidden units. 

This procedure is impractical because gibbs ampling is slow. An alternative to get a sample from $$<v_i h_j>_\text{model}$$ is to "clamp" a data vector to the visible units, update the hidden units and then reconstruct the visible units. The update procedure then becomes: 

$$
\Delta w_{ij} = \epsilon (<v_i h_j>_\text{data} - <v_i h_j>_\text{recon})
$$ 
The above training procedure is called contrastive divergence (CD).  $$<v_i h_j>_\text{data}$$ is called positive phase and $$<v_i h_j>_\text{recon}$$ negative phase. In MATLAB the following code first collect the statistics and then calculates the gradients. In CD it is important to sample the value of h0 and v1, $$@sigmrnd$$ applies the sigmoid function and samples the units based on the caluculated probabilities. In the last update of h1 we do not sample because it introduces ampling noise in the calculations, i.e we use $$@sigm$$. 

```
% collect statistics
v0 = data;
h0 = rbmup(rbm,v0,@sigmrnd);
v1 = rbmdown(rbm,v0,@sigmrnd);
h1 = rbmup(rbm,v0,@sigm);

% calculate positive and negative phase
positive_phase = h0' * v0;
negative_phase = h1' * v1;

%calculate gradients
dw = positive_phase - negative_phase;
db =  sum(v0 - v1)';
dc =  sum(h0 - h1)';

```


![RBM]({{ site.url }}/downloads/rbm.png)

## Contrastive divergence

<iframe src="//www.youtube.com/embed/tD3kQmqNHw0" width="500" height="500" ></iframe>



## Persistent Contrastive Divergence 
<iframe src="//www.youtube.com/embed/c0xdBV70fgE" width="500" height="500" ></iframe>

# hinton DBN
The network was trained in two stages – pretraining and fine-tuning. The layer-by-
layer pretraining was the same as in the previous section, except that when training the top layer of 2000 feature detectors, each “data” vector had 510 components. The first 500 were the activation probabilities of the 500 feature detectors in the penultimate layer and the last 10 were the label values. The value of the correct label was set to 1 and the remainder were set to 0. So the top layer of feature detectors learns to model the joint distribution of the 500 penultimate features and the 10 labels.



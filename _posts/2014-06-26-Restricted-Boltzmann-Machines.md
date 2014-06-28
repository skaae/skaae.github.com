---
comments : True
layout: post
title: Training Restricted Boltzmann Machines
---
This post tries to reproduce a figure 4 in the paper ["Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf). The paper present "persistent contrastive divergence" (PCD) a new training algorithm restricted Boltzmann machines. PCD is compared to ["contrastive divergence"](http://learning.cs.toronto.edu/~hinton/csc2535/readings/nccd.pdf) (CD).

![Figure 4]({{ site.url }}/downloads/figure4.png)

A RBM is a two layer bipartite graph which tries to model some distribution over data. In a RBM there is no connections between hidden units and no connections between visible units. The central equations for RBM's are shown below, [A practical guide to training restricted Boltzmann machines](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf) is a more in debt description.

![RBM]({{ site.url }}/downloads/rbm.png)

In a RBM the bottom layer is usually called the visible layer and the top layer is the hidden layer. In our model the units in the visible and hidden layer will be binary stochastic units. The RBM is a energy based model where the energy is given by:

$$ 
E(\mathbf{v},\mathbf{h}) = -\sum_{i\in \text{visible}} b_i v_i -\sum_{i\in \text{hidden}} c_j h_j - \sum^{}_{i,j} v_i h_j w_{ij} 
$$

Here  $$ w_{ij} $$ is the weight between visible unit $$i$$ and hidden unit $$j$$. $$ b_{i} $$ is the bias for visible unit $$i$$ and $$ c_j $$ is the bias for hidden unit $$j$$. We define the joint probability of a visible and a hidden vector to be:

$$
p(\mathbf{v},\mathbf{h}) = \frac{1}{Z}e^{-E(\mathbf{v},\mathbf{h})}
$$

$$
Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
$$

Here $$Z$$ is the partition function given by summing over all possible pairs of visible and hidden vectors.
The probability that the network assigns to a visible vector, $$\mathbf{v}$$, is given by summing over all possible hidden vectors:

$$
p(\mathbf{v}) = \frac{1}{Z} \sum_{\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
$$

We want the RBM to assign high probability, i.e low energy, to images in our dataset. To do this we adjust the weights and biases such that images in the data set gets lower energy and other images get higher energy.
The update rule for stochastic gradient *ascent* can be shown to be:

$$
\Delta w_{ij} = \epsilon (<v_i h_j>_\text{data} - <v_i h_j>_\text{model})
$$ 

Where $$<>$$ denote expectation under the distribution specified by the subscript that follows. 
Getting a sample from $$<v_i h_j>_\text{data}$$ is easy, by noting that the probability of a hidden unit given by a visible vector is given by:

$$
p(h_j = 1 | \mathbf{v}) = \sigma(c_j + \sum_i v_i w_{ij})   \qquad \text(RBMUP)
$$

and for visible unit i the probability is given by:

$$
p(v_i = 1 | \mathbf{h}) = \sigma(b_i + \sum_j h_j w_{ij})  \qquad \text(RBMDOWN)
$$

Here $$\sigma$$ is the sigmoid function. The conditional probabilities can be calculated in the RBM because hidden units are independent given the visible units and visible units are independent given the hidden units. Getting a sample from $$<v_i h_j>_\text{model}$$ is can be done by Gibbs sampling. We start at a random visible vector and perform alternating Gibbs sampling for a long time, in Gibbs sampling we use the above two equation to repeatedly update  the hidden units given the visible units and then the visible units given the hidden units. This procedure is impractical because Gibbs sampling is slow. 

## Contrastive divergence 

An alternative to get a sample from $$<v_i h_j>_\text{model}$$ is to "clamp" a data vector to the visible units, update the hidden units and then reconstruct the visible units. The update procedure then becomes: 

$$
\Delta w_{ij} = \epsilon (<v_i h_j>_\text{data} - <v_i h_j>_\text{recon})
$$ 

The above training procedure is called contrastive divergence (CD).  $$<v_i h_j>_\text{data}$$ is called positive phase and $$<v_i h_j>_\text{recon}$$ negative phase. In MATLAB the following code first collect the statistics and then calculates the gradients. 

Below we is a basic implementation of a single weight update for a RBM using contrastive divergence:

```Matlab
% helper functions 
function x = rbmup(rbm, x,act)
    x = act(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
end

function x = rbmdown(rbm, x,act)
    x = act(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
end

%% collect statistics
% positive phase
v0 = data;  %clamp data to visible units
h0 = rbmup(rbm,v0,@sigmrnd);

% negative phase
v1 = rbmdown(rbm,v0,@sigmrnd);
h1 = rbmup(rbm,v0,@sigm);

% calculate positive and negative phase
positive_phase = h0' * v0;
negative_phase = h1' * v1;

%calculate gradients
dw = positive_phase - negative_phase;
db =  sum(v0 - v1)';
dc =  sum(h0 - h1)';

% update weights - epsilon is the learning rate
rbm.W = rbm.W + epsilon * dw / minibatch_size;
rbm.b = rbm.b + epsilon * db / minibatch_size;
rbm.c = rbm.c + epsilon * dc / minibatch_size;
```

`@sigmrnd` and `@sigm` determines whether the probabilities from the logistic function should used (`@sigm`) or if the states should be sampled randomly based on the probabilities (`@sigmrnd`).

["Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient"](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf) presents CD which differ slightly from CD. In CD we do

1. clamp data to visible vector
2. RBMUP  (´@sigmrnd´)
3. RBMDOWN (´@sigmrnd´)
4. RBMUP  (´@sigm´)

I.e we start at the data and do a single Gibbs step and collect the statistics. In PCD we collect the positive statistics the same way as in CD. In the negative phase we use a "persistent" number of Markov chains which are not resent between weight updates, i.e each time we collect the negative statistics we start the Markov chain where it ended last time. The number of Markov chains is usually equal to the size of the mini batches. In the MATLAB code the collection of negative statics then becomes
 
```    
hid = rbmup(rbm,markovchains,@sigmrnd);
v1 = rbmdown(rbm,hid,@sigmrnd);
h1 = rbmup(rbm,vk,@sigm);
``
the state of the Markov chains are stored in the ´markovchains´ variable. Before training we initialize the markov chains to some random data points. 

## Training RBM's with CD and PCD

To reproduce figure 4 a RBM was trained on the MNISt dataset. The RBM has  784 visible units and 500 hidden units are trained first with CD and then with PCD. 
We then draw samples from the RBM. The following function was used to draw samples from the RBM:

```Matlab
function [vis_sampled] = rbmsample(rbm,n,k)
%RBMSAMPLE generate n samples from RBM using gibbs sampling with k steps
%   INPUTS:
%       rbm               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps 
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix

% create n random binary starting vectors based on bias
bx = repmat(rbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));

for i = 1:k
    hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(rbm,hid_sampled,@sigmrnd);

end
    hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(rbm,hid_sampled,@sigm);
end
```

Note that we start at a random vector sampled from the probabilities given by the bias to the visible vectors. 
Initializing the visible vectors at random will not produce any digits. 

The videos below show how the RBM's converge. I did 1000 Gibbs steps and recorded the reconstruction for every 10th Gibbs step.

Samples drawn from RBM trained with CD

<iframe src="//www.youtube.com/embed/tD3kQmqNHw0" width="500" height="500" ></iframe>

Samples drawn from RBM trained with PCD

<iframe src="//www.youtube.com/embed/c0xdBV70fgE" width="500" height="500" ></iframe>

To reproduce the above figures download https://github.com/skaae/DeepLearnToolbox_noGPU/ and 
run

```Matlab
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
rand('state',0)
dbn.sizes = [500];
opts.traintype = 'PCD';  % PCD | CD 
opts.numepochs =   100; % probably way to high?
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence
T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.05;    % initial learning rate
f = 0.95;     % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum); 
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
opts.L2 = 0.00;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts,test_x);
rbmsampledbnmovie(dbn,50,1000,'vid.avi',10,@visualize);
```
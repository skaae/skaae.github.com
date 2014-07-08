---
comments : True
layout: post
title: Deriving the gradient for a generative RBM
---
In this post I will discuss the training objective of an RBM. As mentioned in the 
[previous]({{ site.baseurl }}/2014/06/26/Restricted-Boltzmann-Machines/) post a RBM is a energy based model. Energy based models assigns a energy to each configuration of the variables in interest in the model. In the learning an energy based models we modify the energy function such that it has the desired properties, e.g. in the generative model the energy function should assign low energy to likely values of the variables of interest. This post will derive the update rules for a RBM trained with a generative training 


The following notation will be used:

 * $$W$$ visible to hidden weights
 * $$b$$ bias of visible units
 * $$c$$ bias of hidden units
 * $$\mathbf{v}$$ state of all visible units. $$v_i$$ is the state of visible unit $$i$$
 * $$\mathbf{x}$$ is a single training example. In a RBM we set the state of the visible units, "clamp", to the state of the the training example.
   $$\mathbf{x}_t$$ is training example $$t$$. $$N$$ is the number of training examples.
 * $$\mathbf{h}$$ state of all visible units. $$h_j$$ is the state of hidden unit $$j$$
 * $$\theta = \{\mathbf{W,b,c}\}$$.
 * $$N$$ number of samples in training set


 The energy function is given by:

$$ 
E(\mathbf{v},\mathbf{h}) =  \mathbf{h}^T \mathbf{W} \mathbf{v} - \mathbf{b}^T \mathbf{v} - \mathbf{c}^T\mathbf{h}
$$

We define the probability of a particular configuration of the hidden and visible states to be:

$$
p(\mathbf{v},\mathbf{h}) = \frac{1}{Z}e^{-E(\mathbf{v},\mathbf{h})}
$$

$$
Z = \sum_{\mathbf{v},\mathbf{h}} e^{-E(\mathbf{v},\mathbf{h})}
$$

From the definition of probability above it can be seen that states with high energy will have low probability. If we train the RBM as a generative model we want the RBM to assign high probability to training examples and lower probability to other vectors. We get the probability of a visible vector by [marginalizing](http://en.wikipedia.org/wiki/Marginal_distribution) over $$\mathbf{h}$$ in equation $$p(\mathbf{v},\mathbf{h})$$, i.e

$$
p(\mathbf{v}) = \frac{1}{Z} \sum_\mathbf{h} e^{-E(\mathbf{v},\mathbf{h})} 
$$

To optimize the parameters we define loss function for the generative RBM to be the negative log probability of $$p(\mathbf{x}_t)$$


$$
L_\text{generative} = -\sum_{t=1}^N log(P(\mathbf{x}_t))
$$

We can minimize the log probability by find the gradient w.r.t. to the weights in the RBM. First we use the fact that $$f(g(a))' = f'(g)\cdot g'$$ 

$$\begin{align}
	\frac{\partial}{\partial \theta} -\log(P(\mathbf{x})) &= \frac{\partial}{\partial \theta} -log(\sum_h \exp(-E(\mathbf{v},\mathbf{h}))\cdot Z^{-1})   \\
	f(a) &= log(x) \Rightarrow f'(a) = \frac{1}{a} \\
	g    &= \sum_h \exp(-E(\mathbf{v},\mathbf{h}))\cdot Z^{-1} 
\end{align}
$$

To derive $$g'$$ we use that $$g' = k\cdot l'+k' \cdot l$$

$$\begin{align}
    k &= \sum_h \exp(-E(\mathbf{v},\mathbf{h})) \Rightarrow k' = \frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h})) \\
    l &= Z^{-1}  \Rightarrow l' = -\frac{\partial Z}{\partial \theta} Z^{-2} \\
    g'&= k\cdot  l'+k'\cdot l = -\sum_h \exp(-E(\mathbf{v},\mathbf{h}))\cdot\frac{\partial Z}{\partial \theta} Z^{-2} + Z^{-1}\cdot  \frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h}))
\end{align}
$$

Using $$f$$, $$f'$$, $$g$$ and $$g'$$ we can write: 

$$\begin{align}
	\frac{\partial}{\partial \theta} -\log(P(\mathbf{x})) &= 
	-\frac{Z}{\sum_h \exp(-E(\mathbf{v},\mathbf{h}))} \cdot 
	\left( 
		-\sum_h \exp(-E(\mathbf{v},\mathbf{h}))\cdot\frac{\partial Z}{\partial \theta} Z^{-2} + Z^{-1}\cdot  \frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h})) 
	\right) \\
	&=-\frac{Z}{\sum_h \exp(-E(\mathbf{v},\mathbf{h}))} \cdot		\left( 
		 Z^{-1}\cdot  \frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h})) 
	\right) 
	+ \frac{\partial Z}{\partial \theta} Z^{-1} \\
	&=-\frac{\frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h}))}{\sum_{h^*} \exp(-E(\mathbf{v},\mathbf{h}^*))} +\frac{\partial Z}{\partial \theta} Z^{-1} 
\end{align}
$$

We will first look at the first term and then the second term. For the first term note that $$p(\mathbf{h}\mid \mathbf{v}) = \frac{p(\mathbf{h},\mathbf{v})}{p(\mathbf{v})} = \frac{\exp{-E(\mathbf{v},\mathbf{h})}}{\sum_\mathbf{h} \exp{-E(\mathbf{v},\mathbf{h})}}$$. 

$$
-\frac{\frac{\partial}{\partial \theta} \sum_h \exp(-E(\mathbf{v},\mathbf{h}))}{\sum_{h^*} \exp(-E(\mathbf{v},\mathbf{h}^*))} = 
\sum_h \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta}\frac{\exp(-E(\mathbf{v},\mathbf{h}))}{\sum_{h^*} \exp(-E(\mathbf{v},\mathbf{h}^*))} = 
\sum_h \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta} \cdot p(\mathbf{h}\mid\mathbf{v})
$$

Which is the expected value of $$\frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta}$$, given that the data vectors are clamped to the visible units.  This term is usually called the *positive phase*. 
Turning to the second first note that $$ p(\mathbf{v},\mathbf{h}) = \frac{1}{Z}e^{-E(\mathbf{v},\mathbf{h})} $$, then 

$$\begin{align}
\frac{\partial Z}{\partial \theta} Z^{-1} &= \frac{\partial \sum_{\mathbf{v},\mathbf{h}} \exp\left(-E(\mathbf{v},\mathbf{h})\right)}{\partial \theta} \cdot \frac{1}{Z} = 
- \sum_{\mathbf{v},\mathbf{h}}\frac{\partial  E(\mathbf{v},\mathbf{h})}{\partial \theta} \exp\left(-E(\mathbf{v},\mathbf{h})\right)
\cdot
\frac{1}{Z}\\
&=  - \sum_{\mathbf{v},\mathbf{h}}\frac{\partial  E(\mathbf{v},\mathbf{h})}{\partial \theta} \cdot
		p(\mathbf{v},\mathbf{h})
\end{align}
$$

This is the expected value of $$\frac{\partial  E(\mathbf{v},\mathbf{h})}{\partial \theta}$$ under the model distribution, ususally called the *negative phase*. The postive phase lowers the energy of the training data, the negative phase assigns higher energy to other data the the model generate. 

Using $$E_\text{data}$$ to denote the expectation with the data vector clamped (positive phase) and $$E_\text{model}$$ to denote the expectation under the model distribution (negative phase) we get the following udate rule for the parameters:

$$ 
\frac{- \partial \log(P(\mathbf{x}))}{\partial \theta} =E_\text{data}\left[ \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta} \right] - E_\text{model}\left[ \frac{\partial E(\mathbf{v},\mathbf{h})}{\partial \theta} \right]
$$

We can caluculate the partial derivaties of $$\mathbf{W,b,c}$$ to get the update rule for the weights and biases, i.e

$$\begin{align}
\frac{- \partial \log(P(\mathbf{x}))}{\partial \mathbf{W}} &=E_\text{data}\left[ -\mathbf{h}^T\mathbf{v} \right] - 
															E_\text{model}\left[ -\mathbf{h}^T\mathbf{v} \right] \\
\frac{- \partial \log(P(\mathbf{x}))}{\partial \mathbf{b}} &=E_\text{data}\left[ -\mathbf{v} \right] - 
															E_\text{model}\left[ -\mathbf{v} \right] \\
\frac{- \partial \log(P(\mathbf{x}))}{\partial \mathbf{c}} &=E_\text{data}\left[ -\mathbf{h} \right] - 
															E_\text{model}\left[ -\mathbf{h} \right] \\
\end{align}
$$

To use the update rule we need to get samples from $$p(\mathbf{h}\mid \mathbf{v})$$ is easy, clamp a data sample to the visible units and sample the hidden units. To get a sample from the model distribution $$p(\mathbf{h}, \mathbf{v})$$ is hard, one option is to start a a data vector and use gibbs sampling to sample the hidden and visible state many times.

Currently Contrastive divergence (CD) is used to approximate  $$E_\text{model}$$. In CD we we only run the gibbs chain a few steps, typically one, before we sample $$E_\text{model}$$. To use contrastive divergence do:

 1. Clamp data sample to visible units (positive phase)
 2. sample the hidden states  (positive phase)
 3. sample visible states (negative phase)
 4. sample hidden sates (negative phase) 

---
comments : True
layout: post
title: Discriminative training of RBM's
---

Discriminative RBM update equations 

$$\begin{align*}
\frac{- \partial \log(P(y_t\vert \mathbf{x}_t))}{\partial \mathbf{\Theta}} = 
&-\sum_j sigm(o_{y_t,j}(\mathbf{x}_t)) \frac{\partial c_j + \sum_k W_{j,k}x_k+U_{j,y_t}}{\partial \mathbf{\Theta}}  \\
&+\sum_{j,y^*} sigm(o_{y^*,j}(\mathbf{x}_t)) p(y^* \vert \mathbf{x_t})
\frac{\partial (c_j + \sum_k W_{j,k}x_k+U_{j,y^*})}{\partial \mathbf{\Theta}}  \\
%%
o_{y,j}(\mathbf{x}_t) =& c_j + \sum_k W_{j,k} x_k + U_{j,y}
\end{align*}
$$

Here $$\Theta$ is  $$\{\mathbf{c,U,W},\}$$ and the size of the matrices are:

 * $$\mathbf{c}$$  [#hidden X 1]
 * $$\mathbf{U}$$ [#hidden X #classes] 
 * $$\mathbf{W}$$ [#hidden X #visible]
 * $$y_t$$ [#classes X 1]
 * $$\mathbf{x_t}$$ [#visible X 1]



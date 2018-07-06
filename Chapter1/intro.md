# Intro

### Paradigms

* Supervised Learning

  $ Given\ D = \{X_i, Y_i\},\  learn\  f(\cdot ): Y_i = f(X_i),\  s.t. D^{new} = \{X_j\} => \{Y_j\} $

* Unsupervised Learning

  $ Given\ D = \{X_i\},\  learn\  f(\cdot ): Y_i = f(X_i),\  s.t. D^{new} = \{X_j\} => \{Y_j\} $



### Example 

Polynomial curve fitting

Fit the data using a polynomial function of the form:

$ y(x, \bf{w}) = w_0 + w_1x + ... +w_Mx^M = \sum\limits_{j=0}^Mw_jx^j$

$ \bf w $ is the parameters we need to adapt according to dataset $ \{(x_n, y_n)\}_N $

Minimize "loss function" to find the w:

$ \bf w = argmin_w\{E(w)\} , E(w) = \frac{1}{2}\sum\limits_{n=1}^N\{y(x_n,\bf{w}) -t_n\}^2 $



#### Overfitting

![poly_overfitting1](./images/poly_overfitting1.png)

​	For M = 9, the training set error goes to zero, while test set error become very large due to overfitting. The reason is that we have 10 coefficients($ w_0\ to\ w_9 $) thus containing **10 degrees of freedom, and so they can be tuned exactly to the 10 data points in the training set**.



#### Avoid overfitting(1)

**More data**

### ![poly_overfitting2](./images/poly_overfitting2.png)

#### Avoid overfitting(2)

Loss function with **panalty item(or regularization) ** on ||w||

$ E(w) = \frac{1}{2}\sum\limits_{n=1}^N\{y(x_n,\bf{w}) -t_n\}^2 + \frac{\lambda}{2}||\bf w||^2 $

![poly_overfitting3](./images/poly_overfitting3.png)



### Probability Theory

#### Rules of Probability

* **sum rule:**  $ p(Y) = \sum\limits_{Y}{p(X, Y)} $ 
* **product rule:**  $ p(Y, X) = p(Y|X)P(X) $
* **Bayes' theorem: ** $p(Y|X)=\frac{p(X|Y)p(Y)}{P(X)},\  P(X)=\sum\limits_Yp(X|Y)p(Y)$

![prob_rules](./images/prob_rules.png)

#### Probability densities

$ p(x\in(a,b))=\int_a^bp(x)dx,\ p(x)\ge0:\ density\ function $ 

* Note: Under a **nonlinear change of variable**, a probability density transforms differently from a simple function, due to the **Jacobian factor**.  
  $$
  given\ x = g(y)\\
  \because p_x(x)dx \simeq p_y(y)dy\\
  \begin{aligned}
  \therefore p_y(y) &= p_x(x)|\frac{dx}{dy}|\\
  &=p_x(g(y))
  \end{aligned}
  $$
  One consequence of this property is that the concept of the maximum of a probability density is **dependent on the choice of variable**. 

#### Expectations and covariances

$ E[f]=\sum\limits_xp(x)f(x),\ E[f]=\int p(x)f(x)dx $

$var[f] = E[(f(x)-E[f])^2] = E[f^2]-E[f(x)]^2$

$ cov[x, y] = E_{x,y}[(x-E[x])(y-E[y])] = E_{x,y}[xy] - E[x]E[y] $

$ cov[\bf x, \bf y] = E_{x,y}[{\bf x - E[\bf x]}{\bf y^T-E[\bf y^T]}] = E_{x,y}[\bf {xy^T}]-E[\bf x]E[\bf y^T] $

#### Bayes' View

Bayes’ theorem was used to **convert a prior probability into a posterior probability by incorporating the evidence provided by the observed data**. 

Prior probability can be regarded as **knowledge gained before or "common sense"**.

From frequests' view, the **w** learned from dataset is fixed(by maximize likelihood function), while From Bayes' view, it's an uncertain variable represented by a probability distribution $ p(\bf w) $.

Common path of Bayes' learning:

**Loop** 

1. prior: $ p(\bf w) $
2. Observed dataset: $ D={t_1,...,t_N} $
3. Posterior: $ p(\bf{w}|D)=\frac{p(D|\bf w)p(\bf w)}{p(D)} $ and regard it as new prior(updated by observations).

$ posterior \propto likelihood \times prior$ 

$p(D)=\bf \int p(D|w)p(w)dw$



#### Gaussian distribution

$ \mathcal{N}(x|\mathcal{u}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}exp\{-\frac{1}{2\sigma^2}(x-\mathcal{u})^2\} $

$ \mathcal{N}(x|u, \sum) = \frac{1}{\sqrt{(2\pi)^D|\sum|}}exp\{-\frac{1}{2}(x-u)^T\sum(x-u)\} $


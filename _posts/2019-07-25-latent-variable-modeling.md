---
layout: post
title: "Latent Variable Modeling"
date: 2019-07-19
category: blog
mathjax: true
synopsis: "Introduction to the EM Algorithm"
---

Latent variables exist in almost any application where we observe some data. It's rarely ever a case in the real world where we have a situation where we observe all the variables that are part of the estimation problem. In this blog, we'd like to start with latent variable models, and how we may interpret them and then begin to learn how to solve problems involving latent variables. Ideally we will try to stay task agnostic and only reason about a task when we talk about applications of our model, like in gaussian mixture models.

---

## Introduction

We start with the definition of KL-divergence:

$$ \Large KL (p \vert \vert q) = \int q(x) log (\frac{q(x)}{p(x)})$$

One of the key assumptions in this definition is that the support of $q(x)$ and $p(x)$ must be the same, if they are not, then this integral is not properly defined. We may also rewrite the equation as $\mathbb{E}_{q(x)}[log (\frac{q(x)}{p(x)})]$.

As a preliminary exercise, let's look at a simple multimodal distribution. ![mixture_gaussians.png](https://github.com/pranavsubramani/pranavsubramani.github.io/raw/master/images/blog_images/mixture_gaussians.png).
*True Distribution $p(x)$*

Let $p(x)$ be the distribution represented in the plot above (a mixture of univariate gaussians). Assume that $q(x)$ is a single component gaussian.

![mixture_gaussians.png](https://github.com/pranavsubramani/pranavsubramani.github.io/raw/master/images/blog_images/approx_univ_gaussian.png)
*Approximate Distribution $q(x)$*


Try and understand what optimizing $KL(p \vert \vert q)$ and what optimizing $KL (q \vert \vert p)$ will result in.

---

$\textbf{Lemma 1:} \quad KL(q(x) \vert \vert p(x)) \ge 0$

$\textbf{Proof:}$ Consider $-KL(q(x) \vert \vert p(x)) = -\int q(x) \log (\frac{q(x)}{p(x)}) dx$
$ = \int -q(x) \log q(x) + q(x) \log p(x) dx = \int q(x) \log p(x) - q(x) \log q(x) dx $
$ = \int q(x) \log (\frac{p(x)}{q(x)}) dx \le \log \int q(x) \frac{p(x)}{q(x)}$
$  \log \int p(x) dx = \log 1 = 0$.
This completes the proof.

---

## Latent Variables in the Wild

A lot of times, if we have familiarized ourselves with machine learning, we may have heard the word _latent variables_ used constantly. From Autoencoders, Mixture Models, Flows, we have all probably come across these terms. If you have not, then this will be one way of thinking about them.

First and foremost, latent variables are unobserved. This is to say that, if we have some varibales that are observed, $X_1, \dots, X_n$. Then these variables have some $Z$ that affects our observation of them. A more realistic example is, if we observe the traffic (say, the number of cars on the road at a given point in time) at fixed time intervals, say $t_1, t_2, \dots, t_n$. This observation is clearly affected by the accidents on that day, road closures and a variety of other _latent_ variables that are not part of our training data. While this is not a rigorous definition, in this situation, it may be easier to work with an intuitive notion of latent as opposed to a rigorous one.

Let's try and build a simple example where we have a strong notion of what the latent variables are:

Consider $X \sim \pi_1 \mathcal{N}(\mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(\mu_2, \sigma_2^2) + \pi_3 \mathcal{N}(\mu_3, \sigma_3^2)$. This is standard notation to say that $X$ is sampled from a mixture model with mixture weights $\pi_1, \pi_2, \pi_3$ such that $\sum_{i} \pi_i = 1$. Intuitively we may think of this as meaning that $X$ has $\pi_1$ probability of being sampled from $\mathcal{N}(\mu_1, \sigma_1^2)$, $\pi_2$ probability of being sampled from $\mathcal{N}(\mu_2, \sigma_2^2)$ and $\pi_3$ probability of being samped from $\mathcal{N}(\mu_3, \sigma_3^2)$. Let's say we sample $X, k$ times. Given the parameters of the true distributions, we can exactly figure out which latent variable affected our observation of $X_i$.

Let's try out a simple example problem to explain this. Consider the earlier example, where we have a mixture model whose mixture components we are aware of. Our goal is to estimate the parameters of our model (which we will soon define).

$$\Large p(X, Z | \theta) = \prod_{i = 1}^{n} p(x_i, z_i \vert \theta) = \prod_{i = 1}^{n} p(x_i \vert z_i, \theta) p(z_i \vert \theta)$$

And we have that $p(z_i \vert \theta) = \pi_i$ (is the common notation used in mixture models to define the mixture components weights). Furthermore, we make the assumption that $ p(x_i \vert z_i, \theta) \sim \mathcal{N}(\mu_{z_i} \sigma_{z_i}^2)$. The set of parameters that we want to estimate, $\Theta = (\mu_j, \sigma_j, \pi_j)_{i = 1}^{K}$ (the notation is that we have a $K$-component mixture model). Now,

$$ \Large \Theta_{MLE} = \text{argmax}_{\Theta} p(X, Z \vert \Theta) = \text{argmax}_{\Theta} \log p(X, Z \vert \Theta)$$

The reason the log is usually preferred is a matter of numerical stability in the solutions. In theory, we need not work in the log space, computationally it makes a world of difference.

 However, in practice, we do not have it so easy. If we just make a small change to the problem and hid the paramter values, it no longer becomes as easy as the original problem. If our goal was to maximize the likelihood, can we still solve this. Yes!

---

## Incomplete Likelihoods

We now work through what happens if we don't know $Z$ at all.

$$ \Large \log p(X \vert \theta) = \int q(z) \log p(X \vert \theta) dz = \int q(z) \log (\frac{p(X, z \vert \theta)}{p(Z \vert \theta)}) dz  $$

We obtain the above by marginalizing over the latent variable, then we decompose the joint distribution into the joint ove the marginal.

$$ \Large \int q(z) \log (\frac{p(X, z \vert \theta)}{p(Z \vert \theta)}) dz = \int q(z) \log (\frac{q(z) \cdot p(X, z \vert \theta)}{q(z) \cdot p(z \vert X, \theta)}) dz$$

We simply multiply the numerator and denominator by $q(z)$, which, of course, does not invalidate the original equation. However, it leads to a very nice decomposition.

$$ \Large \int q(z) \log (\frac{q(z) \cdot p(X, z \vert \theta)}{q(z) \cdot p(z \vert X, \theta)}) dz = \int q(z) \log (\frac{p(X, z \vert \theta)}{q(z)}) + \int q(z) \log (\frac{q(z)}{p(z \vert X, \theta)}) dz$$

If we observe the second term on the right hand side, it looks familiar. Indeed, it is $KL(q(z) \vert \vert p(Z \vert X, \theta))$. From Lemma $1$, we know that $KL(q \vert \vert p) \ge 0$. So, this implies that

$$  \Large \log p(X \vert \theta) \ge \int q(z) \log (\frac{p(X, z \vert \theta)}{q(z)})$$

The right hand term is called the Variational Lower Bound or ELBO. Furthermore, the simple observation that $\mathcal{L} (q, \theta) + KL(q \vert \vert p) \ge \mathcal{L}(q, \theta)$ will serve immensely useful when discussing variational inference. So, instead of optimizing $\log p(X \vert \theta)$, we can instead optimize $\mathcal{L}(q, \theta)$ and we will be optimizing the lower bound of the likelihood.

This is actually very useful, because there is a clear way to optimize $\mathcal{L}(q, \theta)$. We can perform block coordinate descent on $q(Z)$ and $\theta$ and alternatively minimize them.

---

## Abstract Variational Lower Bound


$\small \textbf{Definition:}$ A Function $g(\zeta, x)$ is called a variational lower bound of $f(x)$ if and only if:

1. $\Large \forall \zeta, \forall x, f(x) \ge g(\zeta, x)$
2. $\Large \forall x_0, \exists \zeta(x_0)$ $\Large \text{such that}$ $\Large f(x_0) = g(\zeta(x_0))$

At first glance, this definition appears restrictive, however, it is not so. In fact, in practice, the variational lower bounds can even be concave functions which are a very nice (are easier to optimize over) to work with.

An example of an abstract variational lower bound is a tangent place for a convex function.

E.g: The definition is called variational because there is a parameter in the definition $\zeta$ that we vary and still maintain the lower bound.

---

## Expectation Maximization

Now, we have all the machinery we require to define the most arbitrary form of the EM algorithm. We wish to solve $\mathcal{L}(q, \theta) = \int q(z) \log (\frac{p(X, z \vert \theta)}{q(z)}) dz$.

We start from some initial $q_0$ (which is some arbitrary variational lower bound) and repeat the following two steps:

E-step: $$q(z) = \text{argmax}_{q} \mathcal{L}(q, \theta) \equiv \text{argmin}_{q} KL(q \vert \vert p) = p(Z \vert X, \theta_0)$$. This step is updating the distribution $q$

M-Step: $$\Theta_* = \text{argmax}_{\Theta} \mathcal{L}(q, \theta) = \text{argmax}_{\Theta} \mathbb{E}_{z} \log p(X, Z \vert \Theta)$$

And that's the entire algorithm, while most of the posts on EM tend to be catered towards mixtures of gaussians and indeed, they are fantastic for estimating the parameters in mixtures of gaussians, the idea is fairly arbitrary and may be extended naturally to any probabilistic model.

It should be noted that EM converges monotonically to a stationary point of $\log p(X \vert \theta)$. The proof of this may be added to this blog post later, if you are interested in proving it for yourself, the key observation is that $l(\theta_t) \le l(\theta_{t + 1})$ where $l(\theta_t)$ is the likelihood at time step $t$. If you are able to show this, then you will have that EM increases the likelihood at each step, which is equivalent to what we originally intended to solve.

---

## Conclusion

Hopefully this post sheds some light into latent variable modeling and how natural the expectation maximization algorithm really is. Finally, we state that EM can be extended to continuous latent variables and categorical latent variables as well with some caveats. In subsequent posts, I'll try to get a little deeper into Variational Inference and Mean-Field approximations.

## References

1. https://davidrosenberg.github.io/mlcourse/Archive/2017/Lectures/14a.EM-algorithm.pdf
2. http://cs229.stanford.edu/notes/cs229-notes8.pdf

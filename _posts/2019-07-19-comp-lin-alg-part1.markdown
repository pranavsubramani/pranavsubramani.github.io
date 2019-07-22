---
layout: post
title: "Introduction To Computational Linear Algebra - Partial Pivoting"
date: 2019-07-19
category: blog
mathjax: true
synopsis: "An Introduction To Computational Linear Algebra - Part 1"
---

As part of my first set of blog posts, I'll be trying to cover some interesting findings in the world of computational linear algebra. To begin with, Computational Linear Algebra is used in almost every little application. Whether it be solving systems of linear equations to fast fourier transformations to computational differential equations.

## Introduction


This blog post will deal solely with solving systems of linear equations. Unlike a first year algebra course, we must understand that in practice, we solve systems of linear equations of the order of a million variables in large scale optimization problems. This is an extremely important task, especially in solving problems that may arise in the industry (like airline planning/vehicle routing/Generalized Assignment Problems/Plant Location Problems).

Solving systems of linear equations on the computer has a plethora of benefits, but isn't devoid of problems. While, in theory, we have arbitrary precision to express our problems, in practice we are often restricted by the floating point precision that our computers will allow us to express. As a result, we must be careful about how we implement algorithms.

## Elementary Notation

To begin with, we must first understand that we have to explain how to deal with this notion of precision in a computer. If the reader has taken a course in Numerical Methods, they may choose to omit this section as this material is likely repetitive.

Most floating point systems can be characterized with the following values: $\beta$ which represents the base (e.g. Decimal, Binary), $\alpha$ which represents the exponent, usually the range will be from $[-\alpha, \alpha]$, and finally $t$ which represents the precision, or, how many significant figures we are willing to express numbers to. It should also be noted that for some $e \in [-\alpha, \alpha], e \in \mathbb{Z}$.

An example of a number written in this format is, the number $10000$, with $\beta = 10, t = 3, \alpha = 16$, is expressed as $1.0 \times 10^5$, alternatively it could be $10.0 \times 10^4$ (until we have a precision of $3$).

## Reason For Pivoting

Here, we will distinguish our results from theory, because theoretically, there is absolutely no reason we should be careful in the choice of the pivot element.

Consider the example where:

$$\Large \begin{align*}
  0.0001 x_1 + x_2  &= \,1 \\
  x_1 + x_2 &= \, 2 \\
\end{align*} $$

Now, certainly we can solve this system of equations if we stare at it for long enough. Indeed, we know that $x_1 = \frac{1}{1 - 0.0001}$, and $x_2 = 2 - \frac{1}{1 - 0.0001}$. We may pull out our phones and immediately solve these equations and be on our way. Even more concerning is that, we may implement a method to solve general system of linear equations without considering the rammifications of our approach.

Let us step through this system of equations line by line. Let,

$$ \Large
A =
\begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
  \end{bmatrix}
  =
  \begin{bmatrix}
    0.0001 & 1 \\
    1 & 1
  \end{bmatrix}
$$

We choose $a_{11}$ as our pivot element and thus we perform the following row operations, $R_2 = R_2 - 10^4 R_1$.

$$\Large
\left(\begin{array}{@{}cc|c@{}}
     0.0001 &  1 & 1 \\
    0 &  -9999 & -9998 \\
\end{array}\right)
$$

Now, solving the second system of equations, we obtain $-0.1 \times 10^{5} x_2 = -0.1 \times 10^{5} \implies x_2 = 1$. If $x_2 = 1$, then we must have that $x_1 = 0$. Note that $-9999$ cannot be accurately represented in our floating point system, so we must round it, which is how we arrive at $0.1 \times 10^5$. However, this is clearly false, since this answer does not agree with our theoretical result. Moreover, if we plug in our values into equation $2$, we know that $1 \neq 2$ (no matter who shows you otherwise!). Thus, we've encountered a possible bug in the implementation of gaussian elimination, simply because we picked the wrong pivot!

How do we fix this? In this case, it is an obvious fix, because we only have two choices for a pivot, and one of them did not work. However, what is a possible general strategy that abstracts to arbitrary systems of linear equations?

Let's first think about the _brute force_ way of solving the problem. What if we performed gaussian elimination with every element as the pivot and eliminated the ones in which there was some underflow (which, for the sake of argument, let's assume we can check for -- in particular, given an operation, we have an oracle which tells us whether the operation will underflow or overflow). First, this solution scales abyssmally with the size of the matrix, so we are clearly in trouble here, however, it _does_ solve our problem.

The above solution is highly inefficient, but certainly no solver uses this as we can compute solutions to systems of equations blazingly fast.

The second possible approach is to randomly pick an element to pivot on, this operation would be $O(1)$ for every column, so it certainly meets the criteria for being fast. However, in an industrial setting, the worst case scenario should _never_ happen. So, while this solution may have an average case analysis of working better, it does not solve the problem. Furthermore, if we have the scales of some of the coefficients of variables to be vastly different, then we run into the same numerical instability as seen before.

Back to the original problem, we observe that the issue here might have been because the pivot we picked was **too small**. If the pivot we choose is too small, then the multiplier that we will have will be extremely large, which may lead to instability. So, if we assume that we have our system of equations represented to sufficient accuracy, then we really care about ensuring that the multiplier is small. To ensure that the multiplier is as small as can be, we can just pivot on the absolute value of the maximum element of a column.

This strategy is known as **partial pivoting**. Commercial solvers use this when solving systems of linear equations. This is certainly one of the simplest fixes to one of the most fundamental algorithms in history. In fact, some consider it to be one of the top $10$ algorithms in the recent decade.

## References

1. [https://web.mit.edu/10.001/Web/Course_Notes/GaussElimPivoting.html](https://web.mit.edu/10.001/Web/Course_Notes/GaussElimPivoting.html)
2. [http://www.math.iitb.ac.in/~neela/partialpivot.pdf](http://www.math.iitb.ac.in/~neela/partialpivot.pdf)

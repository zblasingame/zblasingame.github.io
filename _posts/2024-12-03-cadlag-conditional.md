---
layout: distill
title: Gradients for Time Scheduled Conditional Variables in Neural Differential Equations
description: A short derivation of the continuous adjoint equation for time scheduled conditional variables.
tags: diffusion adjoint neuralODEs guided-generation
giscus_comments: false
date: 2024-12-03
featured: false
citation: true

authors:
  - name: Zander W. Blasingame
    url: https://zblasingame.github.io/
    affiliations:
      name: Clarkson University
      url: https://camel.clarkson.edu/

bibliography: 2024-11-20-adjointdeis.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
#toc:
#    - name: Introduction

---

<div style="display:none">
<!--Math macros-->
$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\B}{\mathcal{B}}
\newcommand{\T}{\mathbb{T}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathcal{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\pr}{\mathbb{P}}
\newcommand{\bfx}{\mathbf{x}}
\newcommand{\bfy}{\mathbf{y}}
\newcommand{\bfz}{\mathbf{z}}
\newcommand{\bfa}{\mathbf{a}}
\newcommand{\bfw}{\mathbf{w}}
\newcommand{\bfA}{\mathbf{A}}
\newcommand{\bfV}{\mathbf{V}}
\newcommand{\bsf}{\boldsymbol{f}}
\newcommand{\bsg}{\boldsymbol{g}}
\newcommand{\bseps}{\boldsymbol{\epsilon}}
\newcommand{\rmd}{\mathrm{d}}
\DeclareMathOperator{\var}{Var}
\DeclareMathOperator{\ex}{\mathbb{E}}
\DeclareMathOperator{\argmax}{arg\,max}
\DeclareMathOperator{\argmin}{arg\,min}
\newtheorem{proposition}{Proposition}
$$
</div>

## Introduction

The advent of large-scale diffusion models conditioned on text embeddings<d-cite key="ldm,2022arXiv220406125R,NEURIPS2022_ec795aea"></d-cite> has allowed for creative control over the generative process.
A recent and powerful technique is that of *prompt scheduling*, *i.e.,* instead of passing a fixed prompt to the diffusion model, the prompt can changed depending on the timestep.
This concept was initially proposed by Doggettx [in this reddit post](https://www.reddit.com/r/StableDiffusion/comments/xas2os/simple_prompt2prompt_implementation_with_prompt/) and the code changes to the stable diffusion repository can be seen [here](https://github.com/CompVis/stable-diffusion/commit/ccb17b55f2e7acbd1a112b55fb8f8415b4862521).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/scheduled_conditionals/prompt_scheduling_example.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Examples of the prompt scheduling technique proposed by Doggettx.
</div>

More generally, we can view this as have the conditional information (in this case text embeddings) scheduled w.r.t. time.
Formally, assume we have a U-Net trained on the noise-prediction task $$\bseps_\theta(\bfx_t, \bfz, t)$$ conditioned on a time scheduled text embedding $$\bfz(t)$$.
The sampling procedure amounts to solving the *probability flow ODE* from time $$T$$ to time $$0$$.
$$\begin{equation}
    \frac{\rmd \bfx_t}{\rmd t} = f(t)\bfx_t + \frac{g^2(t)}{2\sigma_t}\bseps_\theta(\bfx_t, \bfz(t), t),
\end{equation}$$
where $$f, g$$ define the drift and diffusion coefficients of a Variance Preserving (VP) type SDE <d-cite key="song2021denoising"></d-cite>.

### Training-free guidance
A closely related area of active research has been the development of techniques which search of the optimal generation parameters.

More specifically, they attempt to solve the following optimization problem:
$$\begin{equation}
    \label{eq:problem_stmt_ode}
    \argmin_{\bfx_T, \bfz, \theta}\quad \mathcal{L}\bigg(\bfx_T + \int_T^0 f(t)\bfx_t + \frac{g^2(t)}{2\sigma_t}\bseps_\theta(\bfx_t, \bfz, t)\;\rmd t\bigg),
\end{equation}$$
where $$\mathcal L$$ is a real-valued loss function on the output $$\bfx_0$$.

Several recent works this year <d-cite key="pan2024adjointdpm,adjointdeis,marion2024implicit"></d-cite> explore solving the continuous adjoint equations <d-cite key="kidger_thesis"></d-cite> to find the gradients:
$$\begin{equation}
    \frac{\partial \mathcal L}{\partial \bfx_t}, \qquad
    \frac{\partial \mathcal L}{\partial \bfz}, \qquad
    \frac{\partial \mathcal L}{\partial \theta}.
\end{equation}$$
These gradients can the be used in combination with gradient descent algorithms to solve the optimization problem.
However, what if $$\bfz$$ is scheduled and not constant w.r.t to time?

**Problem statement.** Given
$$\begin{equation}
    \bfx_0 = \bfx_T + \int_T^0 f(t)\bfx_t + \frac{g^2(t)}{2\sigma_t}\bseps_\theta(\bfx_t, \bfz(t), t)\;\rmd t,
\end{equation}$$
and $$\mathcal L (\bfx_0)$$, find:
$$\begin{equation}
    \frac {\partial \mathcal L}{\partial \bfz(t)}, \qquad t \in [0,T].
\end{equation}$$

In an earlier [blog post](https://zblasingame.github.io/blog/2024/adjointdeis/) we showed how to find $$\partial L / \partial \bfz$$ by solving the continuous adjoint equations.
How do the continuous adjoint equations change with replacing $$\bfz$$ with time scheduled $$\bfz(t)$$ in the sampling equation?
What we will now show is that

> We can just **simply** replace $$\bfz$$ with $$\bfz(t)$$ in the continuous adjoint equations.

This result will intuitive, does require some technical details to show.

## Gradients of time-scheduled conditional variables
It is well known that diffusion models are just a special type of neural differential equation, either a neural ODE or SDE.
As such we will show this result holds more generally for neural ODEs.

**Theorem** (Continuous adjoint equations for time scheduled conditional variables)**.**
<i>
Suppose there exists a function $$\bfz: [0,T] \to \R^z$$ which can be defined as a ca&#768;dla&#768;g<d-footnote>French: <i>continue a&#768; droite, limite a&#768; gauche.</i></d-footnote> piecewise function where $$\bfz$$ is continuous on each partition of $$[0, T]$$ given by $$\Pi = \{0 = t_0 < t_1 < \cdots < t_n = T\}$$ and whose right derivatives exists for all $$t \in [0,T]$$.
Let $$\bsf_\theta: \R^d \times \R^z \times [0,T] \to \R^d$$ be continuous in $$t$$, uniformly Lipschitz in $$\bfy$$, and continuously differentiable in $$\bfy$$. Let $$\bfy: [0, T] \to \R^d$$ be the unique solution for the ODE
$$\begin{equation}
    \frac{\rmd \bfy}{\rmd t}(t) = \bsf_\theta(\bfy(t), \bfz(t), t),
\end{equation}$$
with initial condition $$\bfy(0) = \bfy_0$$. Then $$\partial \mathcal L / \partial \bfz(t) := \bfa_\bfz(t)$$ and there exists a unique solution $$\bfa_\bfz: [0, T] \to \R^z$$ to the following initial value problem:
$$\begin{equation}
    \bfa_\bfz(T) = \mathbf 0, \qquad \frac{\rmd \bfa_\bfz}{\rmd t}(t) = - \bfa_\bfy(t)^\top \frac{\partial \bsf_\theta(\bfy(t), \bfz(t), t)}{\partial \bfz(t)}.
\end{equation}$$
</i>

{% details Why ca&#768;dla&#768;g? %}
In practice $$\bfz(t)$$ is often a discrete set $$\{\bfz_k\}_{k=1}^n$$ where $$n$$ corresponds to the number of discretization steps the numerical ODE solver takes.
While the proof is easier for a continuously differentiable function $$\bfz(t)$$ we opt for this construction for the sake of generality.
We choose a ca&#768;dla&#768;g piecewise function, a relatively mild assumption, to ensure that the we can define the augmented state on each continuous interval of the piecewise function in terms of the right derivative.
{% enddetails %}

In the remainder of this blog post will provide the proof of this result.
Our proof technique is an extension of the one used by Patrick Kidger (Appendix C.3.1) <d-cite key="kidger_thesis"></d-cite> used to prove the existence to the solution to the continuous adjoint equations for neural ODEs.

*Proof.* Recall that $$\bfz(t)$$ is a piecewise function of time with partition of the time domain $$\Pi$$. Without loss of generality we consider some time interval $$\pi = [t_{m-1}, t_m]$$ for some $$1 \leq m \leq n$$.
Consider the augmented state defined on the interval $$\pi$$:
$$\begin{equation}
    \frac{\rmd}{\rmd t} \begin{bmatrix}
        \bfy\\
        \bfz
    \end{bmatrix}(t) = \bsf_{\text{aug}} = \begin{bmatrix}
        \bsf_\theta(\bfy_t, \bfz_t, t)\\
        \overrightarrow\partial\bfz(t)
    \end{bmatrix},
\end{equation}$$
where $$\overrightarrow\partial\bfz(t): [0,T] \to \R^z$$ denotes the right derivative of $$\bfz$$ at time $$t$$.
Let $$\bfa_\text{aug}$$ denote the augmented state as
$$\begin{equation}
    \label{eq:app:adjoint_aug}
    \bfa_\text{aug}(t) := \begin{bmatrix}
        \bfa_\bfy\\\bfa_\bfz
    \end{bmatrix}(t).
\end{equation}$$
Then the Jacobian of $$\bsf_\text{aug}$$ is defined as
$$\begin{equation}
    \label{eq:app:jacobian_aug}
    \frac{\partial \bsf_\text{aug}}{\partial [\bfy, \bfz]} = \begin{bmatrix}
        \frac{\partial \bsf_\theta(\bfy, \bfz, t)}{\partial \bfy} & \frac{\partial \bsf_\theta(\bfy, \bfz, t)}{\partial \bfz}\\
        \mathbf 0 & \mathbf 0\\
    \end{bmatrix}. 
\end{equation}$$
As the state $$\bfz(t)$$ evolves with $$\overrightarrow\partial\bfz(t)$$ on the interval $$[t_{m-1}, t_m]$$ in the forward direction the derivative of this augmented vector field w.r.t. $$\bfz$$ is clearly $$\mathbf 0$$ as it only depends on time.
Remark, as the bottom row of the Jacobian of $$\bsf_\text{aug}$$ is all $$\mathbf 0$$ and $$\bsf_\theta$$ is continuous in $$t$$ we can consider the evolution of $$\bfa_\text{aug}$$ over the whole interval $$[0,T]$$ rather than just a partition of it.
The evolution of the augmented adjoint state on $$[0,T]$$ is then given as
$$\begin{equation}
    \frac{\rmd \bfa_\text{aug}}{\rmd t}(t) = -\begin{bmatrix}
            \bfa_\bfy & \bfa_\bfz
        \end{bmatrix}(t)  \frac{\partial \bsf_\text{aug}}{\partial [\bfy, \bfz]}(t).
\end{equation}$$
Therefore, $\bfa_\bfz(t)$ is a solution to the initial value problem:
$$\begin{equation}
    \bfa_\bfz(T) = 0, \qquad \frac{\rmd \bfa_\bfz}{\rmd t}(t) = -\bfa_\bfy(t)^\top \frac{\partial \bsf_\theta(\bfy(t), \bfz(t), t)}{\partial \bfz(t)}.
\end{equation}$$

Next we show that there exist a unique solution to the initial value problem.
Now as $$\bfy$$ is continuous and $$\bsf_\theta$$ is continuously differentiable in $$\bfy$$ it follows that $$t \mapsto \frac{\partial \bsf_\theta}{\partial \bfy}(\bfy(t), \bfz(t), t)$$ is a continuous function on the compact set $$[t_{m-1}, t_m]$$.
As such it is bounded by some $$L > 0$$.
Likewise, for $$\bfa_\bfy \in \R^d$$ the map $$(t, \bfa_\bfy) \mapsto -\bfa_\bfy \frac{\partial \bsf_\theta}{\partial [\bfy, \bfz]}(\bfy(t), \bfz(t), t)$$ is Lipschitz in $$\bfa_\bfy$$ with Lipschitz constant $$L$$ and this constant is independent of $$t$$.
Therefore, by the [Picard-Lindelo&#776;f theorem](https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem) the solution $$\bfa_\text{aug}(t)$$ exists and is unique.
<div style="text-align: right">&#x25A1;</div>

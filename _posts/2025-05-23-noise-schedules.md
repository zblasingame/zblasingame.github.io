---
layout: distill
title: Deriving the Drift and Diffusion Coefficients for Diffusion Models
description: A short derivation of the drift and diffusion coefficients for diffusion models.
tags: diffusion SDEs
giscus_comments: false
date: 2025-05-25
featured: true

authors:
  - name: Zander W. Blasingame
    url: https://zblasingame.github.io/
    affiliations:
      name: Clarkson University
      url: https://camel.clarkson.edu/

bibliography: 2025-05-23-noise-schedules.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
    - name: Introduction
    - name: Preliminaries
    - name: Deriving the drift and diffusion coefficients
      subsections:
        - name: Finding the drift coefficient
        - name: Finding the diffusion coefficient
        - name: General transition kernel
    - name: Concluding remarks

_styles: >
    .theorem {
        background: var(--global-code-bg-color);
        border-left: 10px solid var(--global-theme-color);
    }
    .theorem h3 {
        margin-top: 5px;
        margin-bottom: 0px;
        margin-left: 5px;
        color: var(--global-theme-color);
    }
    .theorem p {
        margin-left: 10px;
        font-style: italic;
    }

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
\newcommand{\bfx}{\boldsymbol{x}}
\newcommand{\bfX}{\boldsymbol{X}}
\newcommand{\bfy}{\boldsymbol{y}}
\newcommand{\bfz}{\boldsymbol{z}}
\newcommand{\bfa}{\boldsymbol{a}}
\newcommand{\bfw}{\boldsymbol{w}}
\newcommand{\bfW}{\boldsymbol{W}}
\newcommand{\bfA}{\boldsymbol{A}}
\newcommand{\bfV}{\boldsymbol{V}}
\newcommand{\bsf}{\boldsymbol{f}}
\newcommand{\bsg}{\boldsymbol{g}}
\newcommand{\bseps}{\boldsymbol{\epsilon}}
\newcommand{\rmd}{\mathrm{d}}
\DeclareMathOperator{\var}{Var}
\DeclareMathOperator{\ex}{\mathbb{E}}
\DeclareMathOperator{\argmax}{arg\,max}
\DeclareMathOperator{\argmin}{arg\,min}
\newcommand{\innerprod}[2]{\left\langle #1, #2 \right\rangle}
$$
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diffusion/diffusion_process.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Illustration of a diffusion process governed by the SDE in \eqref{eq:ideal_sde}. The plot shows the evolution of the marginal distribution $p(t, \bfX_t)$ with time.
    We initially start with a mixture distribution at $-1$ and $1$ and diffuse the process into a Gaussian distribution. We use the schedule $\alpha_t = 1 - t^2$ and $\sigma_t^2 = 1 - \alpha_t^2$ on the time interval $[0, 1]$.
</div>


## Introduction

$$\begin{equation}
    \label{eq:ideal_sde}
    \rmd \bfX_t = \underbrace{\vphantom{\sqrt{\frac{\rmd \sigma_t^2}{\rmd t}}}\frac{\rmd \log \alpha_t}{\rmd t}}_{f(t)} \bfX_t \; \rmd t + \underbrace{\sqrt{\frac{\rmd \sigma_t^2}{\rmd t} - 2\sigma_t \frac{\rmd \log \alpha_t}{\rmd t}}}_{g(t)}\; \rmd \bfW_t
\end{equation}$$
<div class="caption">
    In this blog post we derive these choices of drift and diffusion coefficients.
</div>


As I was writing up my Ph.D. thesis <d-cite key="blasingame2025thesis"></d-cite> I was looking to explain *how* we chose the popular form of \eqref{eq:ideal_sde} for diffusion models.
It is well known <d-cite key="lu2022dpmsolver,kingma2021variational"></d-cite>, that this formulation yields the following transition kernel 
$$
\begin{equation}
    \label{eq:tran_kernel}
    q(t, \bfx_t | 0, \bfx_0) = \mathcal N(\bfx_t; \alpha_t\bfx_0, \sigma_t^2 \boldsymbol I),
\end{equation}
$$
where $$\bfx_0 \in \R^d$$ is the initial clean data sample, and with abuse of notation $\mathcal N(\cdot ; \boldsymbol \mu, \boldsymbol \Sigma)$ denotes the *density* function of a multivariate Gaussian distribution with mean vector $\boldsymbol \mu \in \R^d$ and covariance matrix $\boldsymbol \Sigma \in \R^{d\times d}$.
For notational shorthand we will write $q_{t|s}(\bfx\|\bfy) \mapsto q(t, \bfx \| s, \bfy)$ .

This is a very *convenient* property for diffusion models as it implies sampling in forward time reduces to the nice form of
$$\begin{equation}
    \bfx_t = \alpha_t \bfx_0 + \sigma_t \boldsymbol \epsilon, \qquad \boldsymbol \epsilon \sim \mathcal{N}(\boldsymbol 0, \boldsymbol I),
\end{equation}$$
which enables the flexible use of *simulation-free* training techniques <d-cite key="song2021scorebased"></d-cite>.
However, as I looked at prior works which popularized these choices of noise schedules <d-cite key="lu2022dpmsolver,kingma2021variational"></d-cite> I noticed that the choice of drift and diffusion coefficients for the *stochastic differential equation* [(SDE)](https://en.wikipedia.org/wiki/Stochastic_differential_equation) in \eqref{eq:ideal_sde} were simply stated as correct (which they are), but the derivations were elided.

**Goal.** In this blog post I walk through a derivation of these coefficients, starting with the ideal transition kernel, and then deriving the corresponding SDE which produces this transition kernel.
My hope is that this is helpful to other researchers diving into the maths behind diffusion models.

## Preliminaries
Before diving straight into the derivation, we will cover some useful maths on the transition kernel.
Consider a general $d$-dimensional [Ito&#x302; SDE](https://en.wikipedia.org/wiki/It%C3%B4_calculus) driven by the standard $d'$-dimensional [Brownian motion](https://en.wikipedia.org/wiki/Brownian_motion#Mathematics) $\{\bfW_t : 0 \leq t \leq T\}$,
$$
\begin{equation}
    \label{eq:sde}
    \rmd \bfX_t = \bsf(t, \bfX_t)\; \rmd t + \bsg(t, \bfX_t)\; \rmd \bfW_t.
\end{equation}
$$


{% details What is an SDE? %}
For the reader unfamiliar with SDEs one can think of the first term in \eqref{eq:sde}, $\bsf(t, \bfX_t)\; \rmd t$, as a standard *ordinary differential equation* (ODE).
The diffusion coefficient $\bsg: [0,T] \times \R^d \to \R^{d\times d'}$ and infinitesimal $\rmd \bfW_t$ can be thought of another differential equation that is *controlled* by the noisy signal $\bfW_t$.
There is a lot technical details required for Ito&#x302; integration to be well-defined which we elide in this post. For further details we recommend Peter Holderrieth's excellent [blog post](https://www.peterholderrieth.com/blog/2023/Diffusion-Models-with-Stochastic-Differential-Equations-A-Introduction-with-Self-Contained-Mathematical-Proofs/).
{% enddetails %}


To understand the transition kernel, $q_{t|0}(\bfx_t \| \bfx_0)$, of this SDE it would be helpful to understand the dynamics of $\ex[\bfX_t]$ and $\var[\bfX_t]$.
Or in other words we would like to find some functions $\boldsymbol \mu: [0,T] \times \R^d \to \R^d$ and $\boldsymbol \Sigma: [0,T] \times \R^d \to \R^{d\times d}$ such that
$$\begin{align}
    \rmd \ex[\bfX_t] &= \boldsymbol\mu(t, \bfX_t) \; \rmd t,\\
    \rmd \var[\bfX_t] &= \boldsymbol\Sigma(t, \bfX_t) \; \rmd t.
\end{align}$$
Written in this form it seems natural to apply the chain rule from calculus since we have an expression for $\bfX_t$ in \eqref{eq:sde}.

Unlike in traditional calculus, the chain rule for Ito&#x302; calculus is given by the famous Ito&#x302;'s lemma (or Ito&#x302;'s formula), and has a second-order correction term which can be thought of as accounting for the complexities of integration against rough stochastic signals.

<div id="itolemma" class="theorem">
<h3>Theorem 1 (Ito&#x302;'s lemma).</h3>
<p>Consider the Ito&#x302; SDE in \eqref{eq:sde}. Then, for a sufficiently smooth function $\phi: [0,T] \times \R^d \to \R^{d''}$ we can write
$\begin{equation}
\label{eq:itolemma}
\begin{aligned}
    \rmd \phi(t, \bfX_t) &= \bigg(\frac{\partial}{\partial t}\phi(t, \bfX_t) + \innerprod{\nabla_\bfx \phi(t, \bfX_t)}{\bsf(t, \bfX_t)}\\
                        &\qquad + \frac 12 \innerprod{\nabla_\bfx^2 \phi(t, \bfX_t)}{\bsg(t, \bfX_t)\bsg(t, \bfX_t)^\top}_F\bigg)\;\rmd t\\
                        & + \innerprod{\nabla_\bfx \phi(t, \bfX_t)}{\bsg(t, \bfX_t)\;\rmd \bfW_t},
\end{aligned}
\end{equation}$
where $\innerprod{\cdot}{\cdot}_F$ is the Frobenius inner product.
</p>
</div>

Thus for some sufficiently smooth $\phi$ we can take the expectation of both sides in \eqref{eq:itolemma} and formally divide both sides by $\rmd t$ to find
$$\begin{equation}
\begin{aligned}
    \frac{\rmd \ex[\phi(t, \bfX_t)]}{\rmd t} &= \ex\left[\frac{\partial \phi}{\partial t}\right] + \ex\left[\innerprod{\nabla_\bfx \phi(t, \bfX_t)}{\bsf(t, \bfX_t)}\right]\\
    &+ \frac 12\ex\left[\innerprod{\nabla_\bfx^2\phi(t,\bfX_t)}{\bsg(t, \bfX_t)\bsg(t, \bfX_t)^\top}_F\right].
\end{aligned}
\end{equation}$$
Now let $\phi$ denote the identity function $(t, \bfX_t) \mapsto \bfX_t$.
Then we arrive at the rather elegant ODE
$$\begin{equation}
    \label{eq:mean}
    \frac{\rmd \ex[\bfX_t]}{\rmd t} = \ex[\bsf(t, \bfX_t)].
\end{equation}$$
Recall that the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) can be defined as
$$\begin{equation}
    \var[\bfX_t] = \ex[(\bfX_t - \ex[\bfX_t])(\bfX_t - \ex[\bfX_t])^\top].
\end{equation}$$
Thus, with a little algebra we find
$$\begin{equation}
    \label{eq:var}
    \begin{aligned}
        \frac{\rmd \var [\bfX_t]}{\rmd t} &= \ex[\bsf(t, \bfX_t)(\bfX_t - \ex[\bfX_t])^\top]\\
        &+ \ex[(\bfX_t - \ex[\bfX_t])\bsf(t, \bfX_t)^\top]\\
        &+ \ex[\bsg(t, \bfX_t)\bsg(t, \bfX_t)^\top].
    \end{aligned}
\end{equation}$$
For more details on deriving these equations for the mean and covariance of Ito&#x302; processes<d-footnote>These equations cannot be used <i>in general</i> as the expectations should be taken w.r.t. the actual distribution of the state, which is described via the Fokker-Planck-Kolmogorov equation.</d-footnote> we refer the reader to Section 5.5 of S&auml;rkk&auml; and Solin's excellent [book](https://users.aalto.fi/~ssarkka/pub/sde_book.pdf) <d-cite key="sarkka2019applied"></d-cite>.
## Deriving the drift and diffusion coefficients
Now in the context of diffusion models we often operate within the much simpler framework of affine coefficients, *i.e.*,
$$\begin{equation}
    \label{eq:linear_ito}
    \rmd \bfX_t = f(t)\bfX_t\; \rmd t + g(t)\; \rmd \bfW_t.
\end{equation}$$

Given this SDE we will derive the drift and diffusion coefficients that yield the desired transition kernel in \eqref{eq:tran_kernel}, *i.e.*,
we will spend the rest of this blog proving the following proposition.

<div id="noisesched_prop" class="theorem">
<h3>Proposition 2 (Coefficients of Gaussian processes with fixed perturbation kernel).</h3>
<p>Given the linear Ito&#x302; SDE in \eqref{eq:linear_ito}, a strictly monotonically decreasing smooth function $\alpha_t \in \mathcal C^\infty([0,T];\R_{\geq 0})$, a strictly monotonically increasing smooth function $\sigma_t \in \mathcal C^\infty([0,T]; \R_{\geq 0})$, with boundary conditions $\alpha_0 = 1$ and $\sigma_0 = 0$; and a desired transition kernel of the form
$\begin{equation}
    q_{t|0}(\bfx_t|\bfx_0) = \mathcal N(\bfx_t; \alpha_t\bfx_0, \sigma_t^2 \boldsymbol I),
\end{equation}$
the drift and the diffusion coefficients for the linear SDE are:
$\begin{align}
    f(t) &= \frac{\rmd \log \alpha_t}{\rmd t},\\
    g(t) &= \frac{\rmd \sigma_t^2}{\rmd t} - 2\sigma_t^2 \frac{\rmd \log \alpha_t}{\rmd t}.
\end{align}$
</p>
</div>

**Remark.** This particular SDE in \eqref{eq:linear_ito} describes a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) and thus the transition kernel is entirely described by the mean vector and covariance matrix in \eqref{eq:mean} and \eqref{eq:var}.<d-footnote>This characterization clearly doesn't hold for <i>any</i> arbitrary SDE.</d-footnote>

### Finding the drift coefficient
We will start by deriving the drift coefficient.
Let $\boldsymbol \mu(t) = \ex[\bfX_t]$, then by \eqref{eq:mean} we have the following ODE
$$\begin{equation}
    \frac{\rmd \boldsymbol\mu}{\rmd t}(t) = f(t)\boldsymbol \mu(t),
\end{equation}$$
with initial condition $\boldsymbol \mu(0) = \bfx_0$.
We can solve this ODE by using the [integrating factor](https://en.wikipedia.org/wiki/Integrating_factor) $\exp \int_0^t f(\tau)\;\rmd \tau$ to find the solution for the first mean vector:
$$\begin{equation}
    \boldsymbol \mu(t) = \bfx_0 e^{\int_0^t f(\tau)\;\rmd \tau}.
\end{equation}$$
From our definition of the transition kernel we know that $\boldsymbol \mu(t) = \alpha_t \bfx_0$, and thus we can derive $f(t)$ in terms of the schedule $\alpha_t$:
$$\begin{align}
    \alpha_t \bfx_0 &= \bfx_0 e^{\int_0^t f(\tau)\;\rmd \tau},\nonumber\\
    \alpha_t &= e^{\int_0^t f(\tau)\;\rmd \tau},\nonumber\\
    \log \alpha_t &= \int_0^t f(\tau)\;\rmd \tau,\nonumber\\
    \frac{\rmd \log \alpha_t}{\rmd t} &= f(t).
\end{align}$$

### Finding the diffusion coefficient
Next, we turn towards finding an expression for $g(t)$.
For convenience let $\boldsymbol \Sigma(t) = \var[\bfX_t]$.
Next we perform the following simplification
$$\begin{align}
    \ex[f(t)\bfX_t(\bfX_t - \boldsymbol \mu(t))^\top] &= f(t)\ex[\bfX_t(\bfX_t - \boldsymbol \mu(t))^\top],\nonumber\\
    &= f(t)\boldsymbol \Sigma(t),
\end{align}$$
and the same for $\ex[(\bfX_t - \boldsymbol \mu(t))f(t)\bfX_t^\top]$ *mutatis mutandis*; likewise,
$$\begin{equation}
    \ex[g(t)\boldsymbol I g(t) \boldsymbol I] = g^2(t) \boldsymbol I.
\end{equation}$$
Then, from \eqref{eq:var} the dynamics of the covariance matrix is described by
$$\begin{equation}
    \frac{\rmd \boldsymbol \Sigma}{\rmd t}(t) = 2f(t)\boldsymbol \Sigma(t) + g^2(t) \boldsymbol I.
\end{equation}$$
From the boundary conditions we have $\boldsymbol \Sigma(0) = \boldsymbol 0$, thus using the method of integrating factors again, we find a closed form expression for $\boldsymbol \Sigma(t)$:
$$\begin{equation}
    \boldsymbol \Sigma(t) = e^{2\int_0^t f(\tau)\;\rmd \tau} \int_0^t e^{-2\int_0^\tau f(u)\;\rmd u} g^2(\tau)\boldsymbol I\; \rmd \tau.
\end{equation}$$
Next, by definition of the desired transition kernel we assert that $\boldsymbol \Sigma(t) = \sigma_t^2 \boldsymbol I$.
Substituting this into the previous equation yields
$$\begin{align}
    \sigma_t^2 \boldsymbol I &= \frac{\alpha_t^2}{\alpha_0^2} \int_0^t \frac{\alpha_0^2}{\alpha_\tau^2}g^2(\tau)\boldsymbol I\; \rmd \tau,\nonumber\\
    \frac{\sigma_t^2}{\alpha_t^2} \boldsymbol I &= \int_0^t \frac{g^2(\tau)}{\alpha_\tau^2}\boldsymbol I\; \rmd \tau.
\end{align}$$
Then with a little algebra and using Newton's notation<d-footnote><i>I.e.</i>, $\dot\alpha_t = \frac{\rmd}{\rmd t}\alpha_t$.</d-footnote> we find:
$$\begin{align}
    \int_0^t \frac{g^2(\tau)}{\alpha_\tau^2}\; \rmd \tau &= \frac{\sigma_t^2}{\alpha_t^2},\nonumber\\
    \frac{g^2(t)}{\alpha_t^2} &= \frac{\rmd}{\rmd t}\left(\frac{\sigma_t^2}{\alpha_t^2}\right),\nonumber\\
     &\stackrel{(i)}= \frac{2\sigma_t\dot\sigma_t\alpha_t^2 - \sigma_t^22\alpha_t\dot\alpha_t}{\alpha_t^4},\nonumber\\
     g^2(t) &= \frac{2\sigma_t\dot\sigma_t\alpha_t^2 - \sigma_t^22\alpha_t\dot\alpha_t}{\alpha_t^2},\nonumber\\
     &= 2\sigma_t\dot\sigma_t - 2\sigma_t^2 \frac{\dot\alpha_t}{\alpha_t},\nonumber\\
     &\stackrel{(ii)}= \frac{\rmd \sigma_t^2}{\rmd t} - 2\sigma_t^2 \frac{\rmd \log \alpha_t}{\rmd t},
\end{align}$$
where (i) holds by the quotient rule and where (ii) holds by applications of the chain rule.

### General transition kernel
With a little more work one can easily show the result of Kingma *et al.* (<d-cite key="kingma2021variational"></d-cite>, Appendix A.1) for constructing the general form of the transition kernel.
We restate their result below as a corollary of <a href="#noisesched_prop">Proposition 2</a>.

<div class="theorem">
<h3>Corollary 2.1 (Transition kernel for Gaussian processes).</h3>
<p>
The general transition kernel $q_{t|s}(\bfx_t\|\bfx_s)$ for $s < t$ of the Ito&#x302; described in <a href="#noisesched_prop">Proposition 2</a> is
$$\begin{equation}
    q_{t|s}(\bfx_t|\bfx_s) = \mathcal N\left(\bfx_t; \frac{\alpha_t}{\alpha_s}\bfx_s, \left(\sigma_t^2 - \frac{\alpha_t}{\alpha_s}\sigma_s^2\right) \boldsymbol I\right).
\end{equation}$$
</p>
</div>

We leave the proof as an exercise for the reader as it follows straight forwardly from our derivations for <a href="#noisesched_prop">Proposition 2</a> with a simple change in the initial conditions.


## Concluding remarks
In this blog post we presented a brief derivation for the commonly used drift and diffusion coefficients for diffusion models, starting with our desired transition kernel and then working backwards to find the resulting SDE.


<!--One of the key results in stochastic calculus is that the transition kernel $$p(t, \bfx | s, \bfy)$$ for this SDE is described the [Fokker-Planck](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) or forward Kolmogorov equation.-->
<!--*I.e.*, we can describe the evolution of the transition kernel via a *partial differential equation* (PDE).-->
<!--\begin{equation}-->
<!--\frac{\partial}{\partial t} p(t, \bfx | s, \bfy) = -\innerprod{\nabla}{\bsf(t, \bfx)p(t, \bfx | s, \bfy)} + \frac 12 \innerprod{\nabla^2}{\boldsymbol D(t, \bfx)p(t, \bfx|s, \bfy)}_F,-->
<!--\end{equation}-->
<!--$$-->
<!--with initial condition $$p(s, \bfx | s, \bfy) = \delta(\bfx-\bfy)$$; and where $$\boldsymbol D(t, \bfx) = \bsg(t, \bfx)\bsg(t, \bfx)^\top$$ is the diffusion tensor and $$\innerprod{\cdot}{\cdot}_F$$ is the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product).-->

If you would like to cite this post in an academic context, you can use this BibTex snippet:
{% highlight bibtex %}
@misc{blasingame2025noiseschedules,
    author = {Blasingame, Zander W},
    year = {2025},
    title = {Deriving the Drift and Diffusion Coefficients for Diffusion Models},
    url = {https://zblasingame.github.io/blog/2025/noise-schedules/}
}
{% endhighlight %}

<!--ca&#768;dla&#768;g<d-footnote>French: <i>continue a&#768; droite, limite a&#768; gauche.</i></d-footnote>-->

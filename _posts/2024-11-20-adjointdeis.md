---
layout: distill
title: The Continuous Adjoint Equations for Diffusion Models
description: This blog introduces the topic of the continuous adjoint equations for diffusion models, an efficient way to calculate gradients for diffusion models. We show how to design bespoke ODE/SDE solvers of the continuous adjoint equations and show that adjoint diffusion SDEs actually simplify to the adjoint diffusion ODE.
tags: diffusion adjoint AdjointDEIS ODEs SDEs
giscus_comments: false
date: 2024-11-20
featured: true

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
toc:
    - name: Introduction
    - name: Diffusion models
      subsections:
            - name: Reversing the diffusion SDE
            - name: Probability Flow ODE
    - name: Guided generation for diffusion models
    - name: Continuous adjoint equations for diffusion models
      subsections:
        - name: Simplified formulation
        - name: Numerical solvers
        - name: Implementation
        - name: Adjoint diffusion SDEs are actually ODEs
    - name: Concluding Remarks

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
Guided generation is an important problem problem within machine learning.
Solutions to this problem enable us to steer the output of the generative process to some desired output.
This is especially important for allowing us to inject creative control into generative models.
While there are several forms of this problem, we focus on problems which optimize the output of generative model towards some goal defined by a guidance (or loss) function defined on the output.
These particular approaches excel in steering the generative process to perform [adversarial ML](https://en.wikipedia.org/wiki/Adversarial_machine_learningi) attacks, *e.g.*, bypassing security features, attacking Face Recognition (FR) systems, *&c.*

More formally, suppose we have some $$\R^d$$ generative model, $$\bsg_\theta: \R^z \times \R^c \to \R^d$$ parameterized by $$\theta \in \R^m$$ which takes an initial latent $$\bfz \in \R^z$$ and conditional information $$\mathbf{c} \in \R^c$$.
Furthermore, assume we have a scalar-valued guidance function $$\mathcal{L}: \R^d \to \R$$.
Then the guided generation problem can be expressed as an optimization problem:
\begin{equation}
    \label{eq:opt_init}
    \argmin_{\bfz, \mathbf{c}, \theta} \quad \mathcal{L}(\bsg_\theta(\bfz, \mathbf{c})).
\end{equation}
*I.e.*, we wish to find the optimal $$\bfz$$, $$\mathbf{c}$$, and $$\theta$$ which minimizes our guidance function.
A very natural solution to this kind of problem is to perform [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) by using [reverse-mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) to find the gradients.

In this blog post, we focus on a technique for finding the gradients for a very popular class of generative models known as *diffusion models* <d-cite key="song2021denoising,ddpm"></d-cite> by solving the *continuous adjoint equations* <d-cite key="kidger_thesis"></d-cite>.

## Diffusion models
First we give a brief introduction on diffusion models and score-based generative modeling.
More comprehensive coverage can be found at [Yang Song's blog post](https://yang-song.net/blog/2021/score/) and [Lilian Weng's blog post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) on this topic.

Diffusion models start with a diffusion process which perturbs the original data distribution $$p_{\textrm{data}}(\bfx)$$ on $$\R^d$$ into isotropic Gaussian noise $$\mathcal{N}(\mathbf 0, \mathbf I)$$.
This process can be modeled with an Ito&#x302; [Stochastic Differential Equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation) (SDE) of the form
\begin{equation}
\label{eq:ito_diffusion}
\mathrm{d}\bfx_t = \underbrace{f(t)\bfx_t\; \mathrm dt}_{\textrm{Deterministic term $\approx$ an ODE}} + \underbrace{g(t)\; \mathrm d\mathbf{w}_t,}\_{\textrm{Stochastic term}}
\end{equation}
where $$f, g$$ are real-valued functions, $$\{\bfw_t\}_{t \in [0, T]}$$ is the standard [Wiener process](https://en.wikipedia.org/wiki/Wiener_process) on time $$[0, T]$$, and $$\mathrm d\bfw_t$$ can be thought of as infinitesimal white noise.
The drift coefficient $$f(t)\bfx_t$$ is the deterministic part of the SDE and $$f(t)\bfx_t\;\mathrm dt$$ can be thought of as the [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) term of the SDE.
Conversely, the diffusion coefficient $$g(t)$$ is the stochastic part of the SDE which controls how much noise is injected into the system.
<!--We can think of $$g(t)\;\mathrm d\bfw_t$$ as the *control* term of the SDE.-->

The solution to this SDE is a continuous collection of random variables $$\{\bfx_t\}_{t \in [0, T]}$$ over the real interval $$[0, T]$$, these random variables trace stochastic trajectories over the time interval.
Let $$p_t(\bfx_t)$$ denote the marginal [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) of $$\bfx_t$$.
Then $$p_0(\bfx_0) = p_{\textrm{data}}(\bfx)$$ is the data distribution, likewise, for some sufficiently large $$T \in \R$$ the terminal distribution $$p_T(\bfx_T)$$ is *close* to some tractable noise distribution $$\pi(\bfx)$$, called the **prior distribution**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/diffusion_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of diffusion SDE. Original clean image (left) is slowly perturbed by additions of white noise until there is only noise (right). To sample a clean image from a noisy image we only need to solve the SDE in reverse-time, see Equation \eqref{eq:rev_sde}.
</div>



### Reversing the diffusion SDE

So far we have only covered how to destroy data by perturbing it with white noise, however, for sampling we need to be able reverse this process to *create* data from noise.
Remarkably, Anderson <d-cite key="reverse_time_sdes"></d-cite> showed that the Ito&#x302; SDE in Equation \eqref{eq:ito_diffusion} has a corresponding reverse SDE given in closed form by
\begin{equation}
\label{eq:rev_sde}
\mathrm d\bfx_t = [f(t)\bfx_t - g^2(t)\underbrace{\nabla_\bfx\log p_t(\bfx_t)}_{\textrm{Score function}}]\;\mathrm dt + g(t)\; \mathrm d\bar\bfw_t,
\end{equation}
where $$\rmd t$$ denotes a *negative* infinitesmial timestep and $$\nabla_\bfx\log p_t(\bfx_t)$$ denotes the **score function** of $$p_t(\bfx_t)$$.
Note, the stochastic term is now driven by a *different* Wiener process defined on the backwards flow of time, *i.e.*, $$\bar\bfw_T = \mathbf 0$$ a.s.<d-footnote>For technical reasons we use the abbreviation a.s. which denotes almost surely, <i>i.e.</i>, an event happens <i>almost surely</i> if it happens with probability 1.
</d-footnote>
<!--Note, that the *control* term is now driven by the *backwards* Wiener process defined as $$\tilde\bfw_t := \bfw_t - \bfw_T$$. -->
For a modern derivation of Anderson's result we recommend checking out the excellent blog post by [Ludwig Winkler on this topic](https://ludwigwinkler.github.io/blog/ReverseTimeAnderson/).

To train a diffusion model, then, we just need to learn the score function via score-matching <d-cite key="song2021scorebased"></d-cite> or some closely related quantity like the added noise or $$\bfx_0$$-prediction <d-cite key="ddpm,progressive_distillation"></d-cite>.
Many diffusion models use the following choice of drift and diffusion coefficients:
\begin{equation}
    f(t) = \frac{\mathrm d \log \alpha_t}{\mathrm dt},\qquad g^2(t)= \frac{\mathrm d \sigma_t^2}{\mathrm dt} - 2 \frac{\mathrm d \log \alpha_t}{\mathrm dt} \sigma_t^2.
\end{equation}
Where $$\alpha_t,\sigma_t$$ form a noise schedule such that $$\alpha_t^2 + \sigma_t^2 = 1$$ and
\begin{equation}
    \bfx_t = \alpha_t\bfx_0 + \sigma_t\boldsymbol\epsilon_t \qquad \boldsymbol\epsilon_t \sim \mathcal{N}(\mathbf 0, \mathbf I).
\end{equation}
Diffusion models which use noise prediction train a neural network $$\boldsymbol\epsilon_\theta(\bfx_t, t)$$ parameterized by $$\theta$$ to predict $$\boldsymbol\epsilon_t$$ given $$\bfx_t$$ which is equivalent to learning $$\boldsymbol\epsilon_\theta(\bfx_t, t) \approx -\sigma_t\nabla_\bfx \log p_t(\bfx_t)$$.
This choice of drift and coefficients form the Variance Preserving type SDE (VP type SDE) <d-cite key="song2021scorebased"></d-cite>.


### Probability Flow ODE
Song *et al.* <d-cite key="song2021scorebased"></d-cite> showed the existence of an ODE, dubbed the *Probability Flow* ODE, whose trajectories have the same marginals as Equation \eqref{eq:rev_sde} of the form
\begin{equation}
\label{eq:pf_ode}
\frac{\mathrm d\bfx_t}{\mathrm dt} = f(t)\bfx_t - \frac 12 g^2(t) \nabla_\bfx \log p_t(\bfx_t).
\end{equation}
*N.B.*, this form can be found by following the derivation used by Anderson <d-cite key="reverse_time_sdes"></d-cite> and manipulating [Kolmogorov equations](https://en.wikipedia.org/wiki/Kolmogorov_equations) to write a reverse-time SDE with $$\mathbf 0$$ for the diffusion coefficient, *i.e.*, an ODE.

One of key benefits of expressing diffusion models in ODE form is that ODEs are easily reversible, by simply integrating forwards and backwards in time we can encode images from $$p_0(\bfx_0)$$ into $$p_T(\bfx_T)$$ and back again.
With a neural network, often a U-Net <d-cite key="unet"></d-cite>, $$\boldsymbol\epsilon_\theta(\bfx_t, t)$$ trained on noise prediction the *empirical Probability Flow* ODE is now
\begin{equation}
\label{eq:empirical_pf_ode}
\frac{\mathrm d\bfx_t}{\mathrm dt} = f(t)\bfx_t  + \frac{g^2(t)}{2\sigma_t} \boldsymbol\epsilon_\theta(\bfx_t, t).
\end{equation}


## Guided generation for diffusion models
Researchers have proposed many ways to perform guided generation with diffusion models.
Outside of directly conditioning the noise-prediction network on additional latent information Dhariwal and Nichol proposed classifier guidance <d-cite key="diff_beat_gan"></d-cite> which uses an external classifier $$p(\bfz|\bfx)$$ is used to augment the score function $$\nabla_\bfx \log p_t(\bfx_t|\bfz)$$.
Later work, by Ho and Salimans <d-cite key="ho2021classifierfree"></d-cite> showed the classifier could be omitted by incorporating the conditional information in training with the following parameterization of the noise-prediction model
\begin{equation}
    \tilde{\boldsymbol\epsilon}\_\theta(\bfx_t, \bfz, t) := \gamma \boldsymbol\epsilon_\theta(\bfx_t, \bfz, t) + (1 - \gamma) \boldsymbol\epsilon_\theta(\bfx_t, \mathbf 0, t),
\end{equation}
where $\gamma \geq 0$ is the guidance scale.

Outside of methods which require the additional to the diffusion model, or some external network, there are **training-free methods** which we broadly categorize into the following two categories:
1. Techniques which directly optimize the solution trajectory during sampling <d-cite key="yu2023freedom,greedy_dim,liu2023flowgrad"></d-cite>.
2. Techniques which search for the optimal generation parameters, *e.g.*, $$(\bfx_T, \bfz, \theta)$$, (this can include optimizing the solution trajectory as well) <d-cite key="doodl,pan2024adjointdpm,adjointdeis,marion2024implicit"></d-cite>.

The second solution of techniques is related to our initial problem statement in the introduction from Equation \eqref{eq:opt_init}.
We reframe this problem for the specific case of diffusion ODEs.

**Problem statement.** Given the diffusion ODE in Equation \eqref{eq:empirical_pf_ode}, we wish to solve the following optimization problem:
\begin{equation}
    \label{eq:problem_stmt_ode}
    \argmin_{\bfx_T, \bfz, \theta}\quad \mathcal{L}\bigg(\bfx_T + \int_T^0 f(t)\bfx_t + \frac{g^2(t)}{2\sigma_t}\bseps_\theta(\bfx_t, \bfz, t)\;\rmd t\bigg).
\end{equation}
*N.B.*, without loss of generality we let $$\bseps_\theta(\bfx_t, \bfz, t)$$ denote a noise-prediction network conditioned either directly on $\bfz$ or as the classifier-free guidance model $$\tilde \bseps_\theta(\bfx_t, \bfz, t)$$.

From this formulation it is readily apparent the difficulty introduced by diffusion models, over say other methods like [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) or [VAEs](https://en.wikipedia.org/wiki/Variational_autoencoder), is that we need to perform backpropagation through an ODE solve.
Luckily, diffusion models are a type of Neural ODE <d-cite key="neural_ode"></d-cite> which means we can solve the *continuous adjoint equations* to calculate these gradients.

## Continuous adjoint equations for diffusion models
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/adjointdeis/overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of solving the continuous adjoint equations with diffusion models. The sampling schedule consists of $\{t_n\}_{n=0}^N$ timesteps for the diffusion model and $\{\tilde t_n\}_{n=0}^M$ timesteps for AdjointDEIS. The gradients $\bfa_\bfx(T)$ can be used to optimize $\bfx_T$ to find some optimal $\bfx_T^*$. In this example we use the solver known as AdjointDEIS <d-cite key="adjointdeis"></d-cite>.
</div>

The technique of solving an *adjoint* backwards-in-time ODE to calculate the gradients of an ODE is widely used and widespread technique initially proposed by Pontryagin *et al.* <d-cite key="adjoint_sensitivity_method"></d-cite>.
The technique was recently popularized in the ML community by Chen *et al.* <d-cite key="neural_ode"></d-cite> in their seminal work on Neural ODEs with extensions to other models such as SDEs <d-cite key="adjointsde,kidger_thesis"></d-cite>.

We can write the diffusion ODE as a Neural ODE of the form:
$$\begin{equation}
    \frac{\rmd \bfx_t}{\rmd t} = \bsf_\theta(\bfx_t, \bfz, t) := f(t)\bfx_t + \frac{g^2(t)}{2\sigma_t}\bseps(\bfx_t, \bfz, t).
\end{equation}$$
Then $$\bsf_\theta(\bfx_t, \bfz, t)$$ and assuming $$\bsf_\theta$$ is continuous in $t$ and uniformly Lipschitz in $$\bfx$$,<d-footnote>These conditions are to ensure that the Picard-Lindelo&#776;f theorem holds guaranteeing a unique solution to the IVP.
</d-footnote>
then $$\bsf_\theta(\bfx_t, \bfz, t)$$ describes a neural ODE which has a unique solution with the initial condition $$\bfx_T$$ and admits an adjoint state $$\bfa_\bfx(t) := \partial\mathcal{L} / \partial \bfx_t$$ (and likewise for $$\bfa_\bfz(t)$$ and $$\bfa_\theta(t)$$), which solve the continuous adjoint equations, see Theorem 5.2 in <d-cite key="kidger_thesis"></d-cite>, in the form of the [Initial Value Problem](https://en.wikipedia.org/wiki/Initial_value_problem) (IVP):
$$\begin{align}
    \bfa_{\bfx}(0) &= \frac{\partial \mathcal{L}}{\partial \bfx_0}, \qquad && \frac{\rmd \bfa_{\bfx}}{\rmd t}(t) = -\bfa_{\bfx}(t)^\top \frac{\partial \bsf_\theta(\bfx_t, \bfz, t)}{\partial \bfx_t},\nonumber \\
    \bfa_{\bfz}(0) &= \mathbf 0, \qquad && \frac{\rmd \bfa_{\bfz}}{\rmd t}(t) = -\bfa_{\bfx}(t)^\top \frac{\partial \bsf_\theta(\bfx_t, \bfz, t)}{\partial \bfz},\nonumber \\ 
    \bfa_{\theta}(0) &= \mathbf 0, \qquad && \frac{\rmd \bfa_{\theta}}{\rmd t}(t) = -\bfa_{\bfx}(t)^\top \frac{\partial \bsf_\theta(\bfx_t, \bfz, t)}{\partial \theta}.
    \label{eq:adjoint_ode}
\end{align}$$
We call this augmented ODE system an adjoint diffusion ODE.
The adjoint state models the following gradients:
* $$\bfa_\bfx(t) := \partial \mathcal L / \partial \bfx_t$$. The gradient of the guidance function w.r.t. solution trajectory at any time $$t$$.
* $$\bfa_\bfz(T) := \partial \mathcal L / \partial \bfz$$. The gradient of the guidance function  w.r.t. the model conditional information.
* $$\bfa_\theta(T) := \partial \mathcal L / \partial \theta$$. The gradient of the guidance function w.r.t. the model parameters.

While this formulation can calculate the desired gradients to solve the optimization problem it, however, fails to account of the unique construction of diffusion models in particular the special formulation of $$f$$ and $$g$$.
Recent work <d-cite key="pan2024adjointdpm,adjointdeis"></d-cite> has shown that by taking this construction into consideration the adjoint diffusion ODE can be considerably simplified enabling the creation of efficient solvers for the continuous adjoint equations.

{% details Note on the flow of time %}
In the literature of diffusion models the sampling process is often done in reverse-time, *i.e.*, the initial noise is $$\bfx_T$$ and the final sample is $$\bfx_0$$.
Due to this convention solving the adjoint diffusion ODE *backwards* actually means integrating *forwards* in time.
Thus while diffusion models learn to compute $$\bfx_t$$ from $$\bfx_s$$ with $$s > t$$, the adjoint diffusion ODE seeks to compute $$\bfa_\bfx(s)$$ from $$\bfa_\bfx(t)$$.
{% enddetails %}

### Simplified formulation
Recent work on efficient ODE solvers for diffusion models <d-cite key="dpm_solver,deis_georgiatech"></d-cite> have shown that by using *exponential integrators* <d-cite key="exponential_integrators"></d-cite> diffusion ODEs can be simplified and the error in the linear term removed entirely.
Likewise, <d-cite key="pan2024adjointdpm,adjointdeis"></d-cite> showed that this same property follows for adjoint diffusion ODEs.

The continuous adjoint equation for $$\bfa_\bfx(t)$$ in Equation \eqref{eq:adjoint_ode} can be rewritten as
$$\begin{equation}
\label{eq:empirical_adjoint_ode}
    \frac{\mathrm d\bfa_\bfx}{\mathrm dt}(t) = -f(t)\bfa_\bfx(t) - \frac{g^2(t)}{2\sigma_t}\bfa_\bfx(t)^\top \frac{\partial \bseps_\theta(\bfx_t, \bfz, t)}{\partial \bfx_t}.
\end{equation}$$

Due to the gradient of the drift term in Equation \eqref{eq:empirical_adjoint_ode}, further manipulations are required to put the empirical adjoint probability flow ODE into a sufficiently ``nice'' form.
We can transform this [stiff ODE](https://en.wikipedia.org/wiki/Stiff_equation) into a non-stiff form by applying the integrating factor $$\exp\big({\int_0^t f(\tau)\;\mathrm d\tau}\big)$$ to Equation \eqref{eq:empirical_adjoint_ode}, which is expressed as:
$$\begin{equation}
    \label{eq:empirical_adjoint_ode_IF}
    \frac{\mathrm d}{\mathrm dt}\bigg[e^{\int_0^t f(\tau)\;\mathrm d\tau} \bfa_\bfx(t)\bigg] = -e^{\int_0^t f(\tau)\;\mathrm d\tau} \frac{g^2(t)}{2\sigma_t}\bfa_\bfx(t)^\top \frac{\partial \bseps_\theta(\bfx_t, \bfz, t)}{\partial \bfx_t}.
\end{equation}$$
Then, the exact solution at time $$s$$ given time $$t < s$$ is found to be
$$\begin{align}
   \bfa_\bfx(s) = \underbrace{\vphantom{\int_t^s}e^{\int_s^t f(\tau)\;\mathrm d\tau} \bfa_\bfx(t)}_{\textrm{linear}} - \underbrace{\int_t^s e^{\int_s^u f(\tau)\;\mathrm d\tau} \frac{g^2(u)}{2\sigma_u} \bfa_\bfx(u)^\top \frac{\bseps_\theta(\bfx_u, \bfz, u)}{\partial \bfx_u}\;\rmd u}_{\textrm{non-linear}}.
    \label{eq:empirical_adjoint_ode_x}
\end{align}$$
With this transformation we can compute the linear in closed form, thereby **eliminating** the discretization error in the linear term.
However, we still need to approximate the non-linear term which consists of a difficult integral about the complex noise-prediction model.
This is where the insight of Lu *et al.* <d-cite key="dpm_solver"></d-cite> to integrate in the log-SNR domain becomes invaluable.
Let $$\lambda_t := \log(\alpha_t/\sigma_t)$$ be one half of the log-SNR.
Then, with using this new variable and computing the drift and diffusion coefficients in closed form, we can rewrite Equation \eqref{eq:empirical_adjoint_ode_x} as
$$\begin{equation}
    \label{eq:empirical_adjoint_ode_x2}
    \bfa_\bfx(s) = \frac{\alpha_t}{\alpha_s} \bfa_\bfx(t) + \frac{1}{\alpha_s}\int_t^s \alpha_u\sigma_u \frac{\rmd \lambda_u}{\rmd u} \bfa_\bfx(u)^\top \frac{\bseps_\theta(\bfx_u, \bfz, u)}{\partial \bfx_u}\;\rmd u.
\end{equation}$$
As $$\lambda_t$$ is a strictly decreasing function w.r.t. $$t$$ it therefore has an inverse function $$t_{\lambda}$$ which satisfies $$t_{\lambda}(\lambda_t) = t$$, and, with abuse of notation, we let $$\bfx_{\lambda} := \bfx_{t_\lambda(\lambda)}$$, $$\bfa_\bfx(\lambda) := \bfa_\bfx(t_{\lambda}(\lambda))$$, *&c.* and let the reader infer from context if the function is mapping the log-SNR back into the time domain or already in the time domain.
Then by rewriting Equation \eqref{eq:empirical_adjoint_ode_x2} as an exponentially weighted integral and performing an analogous derivation for $$\bfa_\bfz(t)$$ and $$\bfa_\theta(t)$$, we arrive at:

**Proposition** *(Exact solution of adjoint diffusion ODEs)***.**
    Given initial values $$[\bfa_\bfx(t), \bfa_\bfz(t), \bfa_\theta(t)]$$ at time $$t \in (0,T)$$, the solution $$[\bfa_\bfx(s), \bfa_\bfz(s), \bfa_\theta(s)]$$ at time $$s \in (t, T]$$ of adjoint diffusion ODEs in Equation \eqref{eq:adjoint_ode} is
    $$\begin{align}
        \label{eq:exact_sol_empirical_adjoint_ode_x}
        \bfa_\bfx(s) &= \frac{\alpha_t}{\alpha_s} \bfa_\bfx(t) + \frac{1}{\alpha_s}\int_{\lambda_t}^{\lambda_s} \alpha_\lambda^2 e^{-\lambda} \bfa_\bfx(\lambda)^\top \frac{\partial \bseps_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \bfx_\lambda}\;\rmd \lambda,\\
        \label{eq:exact_sol_empirical_adjoint_ode_z}
        \bfa_\bfz(s) &= \bfa_\bfz(t) + \int_{\lambda_t}^{\lambda_s}\alpha_\lambda e^{-\lambda} \bfa_\bfx(\lambda)^\top \frac{\partial \boldsymbol\epsilon_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \bfz}\;\rmd\lambda,\\
        \label{eq:exact_sol_empirical_adjoint_ode_theta}
        \bfa_\theta(s) &= \bfa_\theta(t) + \int_{\lambda_t}^{\lambda_s}\alpha_\lambda e^{-\lambda} \bfa_\bfx(\lambda)^\top \frac{\partial \boldsymbol\epsilon_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \theta}\;\rmd\lambda.
    \end{align}$$

### Numerical solvers
Now that we have a simplified formulation of the continuous adjoint equations we can construct bespoke numerical solvers.
To do this we take approximate the integral term via a Taylor expansion which we illustrate for Equation \eqref{eq:exact_sol_empirical_adjoint_ode_x}.For $$k \geq 1$$ a $$(k-1)$$-th Taylor expansion of the scaled vector Jacobian about $$\lambda_t$$ is equal to
<div class="l-body-outset">
$$\begin{equation}
    \alpha_\lambda^2\bfa_\bfx(\lambda)^\top \frac{\partial \boldsymbol\epsilon_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \bfx_\lambda} = \sum_{n=0}^{k-1} \frac{(\lambda - \lambda_t)^n}{n!} \frac{\mathrm d^n}{\mathrm d\lambda^n}\bigg[\alpha_\lambda^2\bfa_\bfx(\lambda)^\top \frac{\partial \boldsymbol\epsilon_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \bfx_\lambda}\bigg]_{\lambda = \lambda_t} + \mathcal{O}((\lambda  - \lambda_t)^k).
\end{equation}$$
</div>
For notational convenience we denote the $n$-th order derivative of scaled vector-Jacobian products at $$\lambda_t$$ as
$$\begin{equation}
    \label{eq:app:vjp_def_x}
    \bfV^{(n)}(\bfx; \lambda_t) = \frac{\rmd^n}{\rmd \lambda^n}\bigg[\alpha_\lambda^2\bfa_\bfx(\lambda)^\top \frac{\partial \bseps_\theta(\bfx_\lambda, \bfz, \lambda)}{\partial \bfx_\lambda}\bigg]_{\lambda = \lambda_t}.
\end{equation}$$
Then substituting our Taylor expansion into Equation \eqref{eq:exact_sol_empirical_adjoint_ode_x} and letting $$h = \lambda_s - \lambda_t$$ denote the step size we have a $$k$$-th order solver for the continuous adjoint equation for $$\bfa_\bfx(t)$$:
<div class="l-body-outset">
$$\begin{equation}
    \bfa_\bfx(s) = 
    \underbrace{
        \vphantom{\int_{\lambda_t}^{\lambda_s}}
        \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)
    }_{\substack{\textrm{Linear term}\\\textbf{Exactly computed}}}
    +\frac{1}{\alpha_s} \sum_{n=0}^{k-1}
    \underbrace{
        \vphantom{\int_{\lambda_t}^{\lambda_s}}
        \bfV^{(n)}(\bfx; \lambda_t)
    }_{\substack{\textrm{Derivatives}\\\textbf{Approximated}}}\;
    \underbrace{
        \int_{\lambda_t}^{\lambda_s}  \frac{(\lambda - \lambda_t)^n}{n!} e^{-\lambda}\;\mathrm d\lambda
    }_{\substack{\textrm{Coefficients}\\\textbf{Analytically computed}}}
    +
    \underbrace{
        \vphantom{\int_{\lambda_t}^{\lambda_s}}
        \mathcal{O}(h^{k+1}).
    }_{\substack{\textrm{Higher-order errors}\\\textbf{Omitted}}}
\end{equation}$$
</div>

Let's break this down term by term.

1. **Linear term.** The linear term of the adjoint diffusion ODE can be calculated exactly using ratio of the signal schedule $$\alpha_t / \alpha_s$$. As $$\alpha_t \geq \alpha_s$$ for $$t \leq s$$ this implies $$\alpha_t / \alpha_s \geq 1$$.
    $$\begin{equation*}
        \bfa_\bfx(s) = 
        {\color{orange}\underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)
        }_{\substack{\textrm{Linear term}\\\textbf{Exactly computed}}}}
        +\frac{1}{\alpha_s} \sum_{n=0}^{k-1}
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \bfV^{(n)}(\bfx; \lambda_t)
        }_{\substack{\textrm{Derivatives}\\\textbf{Approximated}}}\;
        \underbrace{
            \int_{\lambda_t}^{\lambda_s}  \frac{(\lambda - \lambda_t)^n}{n!} e^{-\lambda}\;\mathrm d\lambda
        }_{\substack{\textrm{Coefficients}\\\textbf{Analytically computed}}}
        +
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \mathcal{O}(h^{k+1}).
        }_{\substack{\textrm{Higher-order errors}\\\textbf{Omitted}}}
    \end{equation*}$$

2. **Derivatives.** The $$n$$-th order derivatives of scaled vector-Jacobian product can be efficiently estimated using multi-step methods <d-cite key="atkinson2011numerical"></d-cite>.
    $$\begin{equation*}
        \bfa_\bfx(s) = 
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)
        }_{\substack{\textrm{Linear term}\\\textbf{Exactly computed}}}
        +\frac{1}{\alpha_s} \sum_{n=0}^{k-1}
        {\color{orange}\underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \bfV^{(n)}(\bfx; \lambda_t)
        }_{\substack{\textrm{Derivatives}\\\textbf{Approximated}}}}\;
        \underbrace{
            \int_{\lambda_t}^{\lambda_s}  \frac{(\lambda - \lambda_t)^n}{n!} e^{-\lambda}\;\mathrm d\lambda
        }_{\substack{\textrm{Coefficients}\\\textbf{Analytically computed}}}
        +
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \mathcal{O}(h^{k+1}).
        }_{\substack{\textrm{Higher-order errors}\\\textbf{Omitted}}}
    \end{equation*}$$

3. **Coefficients.** The exponentially weighted integral can be analytically computed in closed form.
    $$\begin{equation*}
        \bfa_\bfx(s) = 
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)
        }_{\substack{\textrm{Linear term}\\\textbf{Exactly computed}}}
        +\frac{1}{\alpha_s} \sum_{n=0}^{k-1}
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \bfV^{(n)}(\bfx; \lambda_t)
        }_{\substack{\textrm{Derivatives}\\\textbf{Approximated}}}\;
        {\color{orange}\underbrace{
            \int_{\lambda_t}^{\lambda_s}  \frac{(\lambda - \lambda_t)^n}{n!} e^{-\lambda}\;\mathrm d\lambda
        }_{\substack{\textrm{Coefficients}\\\textbf{Analytically computed}}}}
        +
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \mathcal{O}(h^{k+1}).
        }_{\substack{\textrm{Higher-order errors}\\\textbf{Omitted}}}
    \end{equation*}$$

4. **Higher-order errors.** The remaining higher-order error terms are discarded. If $$h^{k+1}$$ is sufficiently small than these errors are negligible.
    $$\begin{equation*}
        \bfa_\bfx(s) = 
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)
        }_{\substack{\textrm{Linear term}\\\textbf{Exactly computed}}}
        +\frac{1}{\alpha_s} \sum_{n=0}^{k-1}
        \underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \bfV^{(n)}(\bfx; \lambda_t)
        }_{\substack{\textrm{Derivatives}\\\textbf{Approximated}}}\;
        \underbrace{
            \int_{\lambda_t}^{\lambda_s}  \frac{(\lambda - \lambda_t)^n}{n!} e^{-\lambda}\;\mathrm d\lambda
        }_{\substack{\textrm{Coefficients}\\\textbf{Analytically computed}}}
        +
        {\color{orange}\underbrace{
            \vphantom{\int_{\lambda_t}^{\lambda_s}}
            \mathcal{O}(h^{k+1}).
        }_{\substack{\textrm{Higher-order errors}\\\textbf{Omitted}}}}
    \end{equation*}$$

{% details Computing the exponentially weighted integral %}
The exponentially weighted integral can be solved **analytically** by applying $$n$$ times integration by parts <d-cite key="dpm_solver,exponential_integrators"></d-cite> such that
$$\begin{equation}
    \label{eq:exponential_integral}
    \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \frac{(\lambda - \lambda_t)^n}{n!}\;\mathrm d\lambda = \frac{\sigma_s}{\alpha_s} h^{n+1}\varphi_{n+1}(h),
\end{equation}$$
with special $$\varphi$$-functions <d-cite key="exponential_integrators"></d-cite>.
These functions are defined as
$$\begin{equation}
    \varphi_{n+1}(h) := \int_0^1 e^{(1-u)h} \frac{u^n}{n!}\;\mathrm du,\qquad\varphi_0(h) = e^h,
\end{equation}$$
which satisfy the recurrence relation $$\varphi_{k+1}(h) = (\varphi_{k}(h) - \varphi_k(0)) / h$$ and have closed forms for $k = 1, 2$:
$$
\begin{align}
    \varphi_1(h) &= \frac{e^h - 1}{h},\\
    \varphi_2(h) &= \frac{e^h - h - 1}{h^2}.
\end{align}
$$
{% enddetails %}

From this construction there are only two-sources of error. The error in approximating the $$n$$-th order derivative of the vector-Jacobian and the higher-order errors.
Therefore, as we long as we pick a sufficiently small step size, $$h$$, and appropriate order, $$k$$, we can achieve accurate (enough) estimates of the gradients.
The derivations for the solvers of $$\bfa_\bfz(t)$$ and $$\bfa_\theta(t)$$ are omitted for brevity but follow an analogous derivation.
The $$k$$-th order solvers resulting from this method are called **AdjointDEIS-$$k$$**.
In <d-cite key="adjointdeis"></d-cite> we prove that that AdjointDEIS-$$k$$ are $$k$$-th order solvers for $$k = 1, 2$$.

### Implementation
Consider the case when $$k=1$$ then we have the following first-order solver.

**AdjointDEIS-1.** Given an initial augmented adjoint state $$[\bfa_\bfx(t), \bfa_\bfz(t), \bfa_\theta(t)]$$ at time $$t \in (0, T)$$, the solution $$[\bfa_\bfx(s), \bfa_\bfz(s), \bfa_\theta(s)]$$ at time $$s \in (t, T]$$ is approximated by
$$\begin{align}
     \bfa_\bfx(s) &= \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t) + \sigma_s (e^h - 1) \frac{\alpha_t^2}{\alpha_s^2}\bfa_\bfx(t)^\top \frac{\partial \bseps(\bfx_t, \bfz, t)}{\partial \bfx_t},\nonumber\\
     \bfa_\bfz(s) &= \bfa_\bfz(t) + \sigma_s (e^h - 1) \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)^\top \frac{\partial \bseps(\bfx_t, \bfz, t)}{\partial \bfz},\nonumber\\
     \bfa_\theta(s) &= \bfa_\theta(t) + \sigma_s (e^h - 1) \frac{\alpha_t}{\alpha_s}\bfa_\bfx(t)^\top \frac{\partial \bseps(\bfx_t, \bfz, t)}{\partial \theta}.
     \label{eq:adjoint_deis_1_at}
\end{align}$$

The vector-Jacobian product can be easily calculated using reverse-mode automatic differentiation provided by most modern ML frameworks.
We illustrate an implementation of this first-order solver using PyTorch. For simplicity we omit the code for calculating $$\bfa_\theta$$ as it requires more boilerplate code.

{% highlight python %}
def adjointdeis_1(model, scheduler, x0, z, guidance_function, timesteps):
    """
    Args:
        model (torch.nn.Module): Noise prediction model takes `(x, z, t)` as inputs.
        scheduler: Object which manages the noise schedule and sampling solver for the diffusion model.
        x0 (torch.Tensor): Generated image `x0`.
        z (torch.Tensor): Conditional information.
        guidance_function: A scalar-valued guidance function which takes `x0` as input.
        timesteps (torch.Tensor): A sequence of strictly monotonically increasing timesteps.
    """
    x0.requires_grad_(True)
    adjoint_x = torch.autograd.grad(guidance_function(x0).mean(), x0)[0]
    adjoint_z = torch.zeros_like(z)
    xt = x0

    for i in range(timesteps):
        if i == 0:
            t, s = 0, timesteps[i]
        else:
            t, s = timesteps[i], timesteps[i + 1]

        model_out = model(xt, z, t)

        # Compute vector Jacobians
        vec_J_xt = torch.autograd.grad(model_out, xt, adjoint_x, retain_graph=True)[0]
        vec_J_z = torch.autograd.grad(model_out, z, adjoint_z)[0]
        
        # Compute noise schedule parameters
        lambda_t, lambda_s = scheduler.lambda_t([t, s])
        alpha_t, alpha_s = scheduler.alpha_t([t, s])
        sigma_t, sigma_s = scheduler.sigma_t([t, s])

        h = lambda_s - lambda_t

        # Solve AdjointDEIS-1
        adjoint_x = (alpha_t / alpha_s) * adjoint_x + sigma_s * torch.expm1(h) * (alpha_t**2 / alpha_s**2) * vec_J_xt
        adjoint_z = adjoint_z + sigma_s * torch.expm1(h) * (alpha_t / alpha_s) * vec_J_xt

        # Use some ODE solver to find next xt
        xt = scheduler.step(xt, t, s)

    return xt, adjoint_x, adjoint_z
{% endhighlight %}


### Adjoint diffusion SDEs are actually ODEs
What about diffusion SDEs, the problem statement in Equation \eqref{eq:problem_stmt_ode} would become
$$\begin{equation}
    \label{eq:problem_stmt_sde}
    \argmin_{\bfx_T, \bfz, \theta}\quad \mathcal{L}\bigg(\bfx_T + \int_T^0 f(t)\bfx_t + \frac{g^2(t)}{\sigma_t}\bseps_\theta(\bfx_t, \bfz, t)\;\rmd t + \int_T^0 g(t) \; \rmd \bar\bfw_t\bigg).
\end{equation}$$
The technical details of working with SDEs are beyond the scope of this post; however, we will highlight one of the key insights from our work <d-cite key="adjointdeis"></d-cite>.

Suppose we have an SDE in the [Stratonovich](https://en.wikipedia.org/wiki/Stratonovich_integral) sense of the form
$$\begin{equation}
    \label{eq:stratonovich_sde}
    \rmd \bfx_t = \bsf(\bfx_t, t)\;\rmd t + \bsg(t) \circ \rmd \bfw_t
\end{equation}$$
where $$\circ \rmd \bfw_t$$ denotes integration in the Stratonovich sense and $$\bsf \in \mathcal{C}_b^{\infty, 1}(\R^d)$$, *i.e.*, $$\bsf$$ is continuous function to $$\R^d$$ and has infinitely many bounded derivatives w.r.t. the state and bounded first derivatives w.r.t. to time. Likewise, let $$\bsg \in \mathcal{C}_b^1(\R^{d \times w})$$ be a continuous function with bounded first derivatives. Lastly, let $$\bfw_t: [0,T] \to \R^w$$ be a $$w$$-dimensional Wiener process.
Then Equation \eqref{eq:stratonovich_sde} has unique strong solution given by $$\bfx_t: [0, T] \to \R^d$$.

We show in <d-cite key="adjointdeis"></d-cite> that the continuous adjoint equations of such an SDE reduce to a backwards-in-time SDE of the form
$$\begin{equation}
    \label{eq:sde_is_ode}
    \rmd \bfa_\bfx(t) = -\bfa_\bfa(t)^\top \frac{\partial \bsf}{\partial \bfx_t}(\bfx_t, t)\;\rmd t
\end{equation}$$
with a $$\mathbf 0$$ coefficient for the diffusion term and that there exists a unique strong solution to this SDE of the form $$\bfa_\bfx: [0,T] \to \R^d$$.
As the diffusion coefficient for this SDE is $$\mathbf 0$$ then it is essentially an ODE.
While glossing over some technical details this result should be straightforwardly apparent as the diffusion coefficient $$\bsg(t)$$ relies only on time and not the state, nor other parameters of interest.

**Remark.** While the adjoint state evolves with an ODE the underlying state $$\bfx_t$$ still evolves with a backwards-in-time SDE!
This was the reason for our choice of Stratonovich over Ito&#770; as the Stratonovich integral is symmetric.

Now our diffusion SDE can be easily converted into Stratonovich form due to the diffusion coefficient depending only on time.
Moreover, due to the shared derivation using the Kolmogorov equations in constructing diffusion SDEs and diffusion ODEs, the two forms differ only by a factor of 2 within the drift term.
$$
\begin{equation}
        {\color{orange}\underbrace{\rmd \bfx_t = f(t)\bfx_t + {\color{black}2} \frac{g^2(t)}{2\sigma_t} \bseps_\theta(\bfx_t, \bfz, t)\;\rmd t}_{\textrm{Diffusion ODE}}} + g(t)\circ\rmd\bar\bfw_t.
    \end{equation}
$$
Furthermore, notice that SDE has form
$$\begin{equation}
    \rmd \bfx_t = {\color{orange}\underbrace{f(t)\bfx_t + \frac{g^2(t)}{\sigma_t} \bseps_\theta(\bfx_t, \bfz, t)}_{= \bsf_\theta(\bfx_t,\bfz, t)}}\;\rmd t + g(t)\;\rmd\bar\bfw_t.
\end{equation}$$
and then by our result from Equation \eqref{eq:sde_is_ode} the adjoint diffusion SDE evolves with the following ODE
$$
\begin{equation}
    \frac{\rmd \bfa_\bfx}{\rmd t}(t) = -\bfa_\bfx(t)^\top \frac{\partial \bsf_\theta(\bfx_t, \bfz, t)}{\partial \bfx_t}.
\end{equation}$$

As the only difference between $$\bsf_\theta$$ for diffusion SDEs and ODEs are a factor of 2 we realize that:
> We can use the **same** ODE solvers for adjoint diffusion SDEs! 

With the only caveat being the factor of 2.
Therefore, we can modify the update equations from our code from above to now solve adjoint diffusion SDEs.
{% highlight python %}
adjoint_x = (alpha_t / alpha_s) * adjoint_x + 2 * sigma_s * torch.expm1(h) * (alpha_t**2 / alpha_s**2) * vec_J_xt
adjoint_z = adjoint_z + 2 * sigma_s * torch.expm1(h) * (alpha_t / alpha_s) * vec_J_xt
{% endhighlight %}


## Concluding remarks
This blog post gives a detailed introduction to the continuous adjoint equations.
We discuss the theory behind them and why it is an appropriate tool for solving guided generation problems for diffusion models.
This post serves as a summary for our recent NeurIPS paper:

* [Zander W. Blasingame and Chen Liu. *AdjointDEIS: Efficient Gradients for Diffusion Models*. NeurIPS 2024](https://openreview.net/forum?id=fAlcxvrOEX)

For examples of this technique used in practice check out our full paper and concurrent work from our colleagues <d-cite key="marion2024implicit,pan2024adjointdpm"></d-cite> which explore different experiements and focus on different aspects of implementing the continuous adjoint equations.

---
layout: distill
title: Face Morphing with Diffusion Models
description: This blog introduces a new family of face morphing attacks known as <b>Di</b>fusion <b>M</b>orphs (<b>DiM</b>). DiMs are a novel method for constructing morphed faces which exploit the iterative nature of diffusion models to construct face morphs which are more effective and more realisitc in appearance. DiMs achieve state-of-the-art morphing performance and visual fidelity, far surpassing previous methods. In this blog post I will detail the intution, basic concepts, and applications of DiMs.
tags: diffusion DiM face-morphing numerical-methods ODEs SDEs greedy-algorithms
giscus_comments: false
date: 2024-08-27
featured: true

authors:
  - name: Zander W. Blasingame
    url: https://zblasingame.github.io/
    affiliations:
      name: Clarkson University

bibliography: 2024-08-27-dim.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Diffusion
    subsections:
        - name: Reversing the diffusion SDE
        - name: Probability Flow ODE
  - name: DiM
    subsections:
        - name: High-order ODE solvers
  - name: Greedy Guided Generation
    subsections:
        - name: The search space of Greedy-DiM is well-posed
  - name: Concluding Remarks

---

<!--Math macros-->
$$
\newcommand{\R}{\mathbb{R}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\B}{\mathcal{B}}
\newcommand{\T}{\mathcal{T}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathcal{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\pr}{\mathbb{P}}
\newcommand{\bfx}{\mathbf{x}}
\newcommand{\bfy}{\mathbf{y}}
\newcommand{\bfz}{\mathbf{z}}
\newcommand{\bfa}{\mathbf{a}}
\newcommand{\bfw}{\mathbf{w}}
\DeclareMathOperator{\var}{Var}
\DeclareMathOperator{\ex}{\mathbb{E}}
\DeclareMathOperator{\argmax}{arg\,max}
\DeclareMathOperator{\argmin}{arg\,min}
$$

## Introduction

Face morphing is a kind of attack wherein the attacker attempts to combine the faces of two real identities into *one* face.
This attack exploits the structure of the embedding space of the Face Recognition (FR) system <d-cite key="Ferrara2016,morphed_first"></d-cite>.
FR systems generally work by embedding the image which is hard to compare into some kind of vector space which makes it easier to compare, this space is referred to as the feature space of the FR system.
Once a user is enrolled into the FR system an acceptance region<d-footnote>If the measure of distance is a proper metric, like $\ell^2$, and not just a semi-metric this forms a ball around the enrolled user.</d-footnote> is defined around the enrolled user in the feature space.
If there exists an overlap between the acceptance regions of two identities in the feature space then a face morphing attack can be performed.
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/morph_diagram.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of the face morphing attack. The goal is to create a morphed face that lies within the acceptance regions of <i>both</i> identities.
    Bona fide images from the FRLL dataset <d-cite key="frll"></d-cite>.
</div>

Face morphing attacks fall broadly into two categories.

* **Landmark-based attacks** create morphed images by warping and aligning facial landmarks of the two face images before performing a pixel-wise average to obtain the morph <d-cite key="can_gan_beat_landmark,multe-scale-block-fusion"></d-cite>.

* **Representation-based attacks** create morphs by first embedding the original images into a representational space, once embedded into this space an interpolation between the two embeddings is constructed to a morphed representation. This morph representation is then projected back into the image space to created the morphed image <d-cite key="morgan"></d-cite>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/001_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>a</i></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/opencv/001_013.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">OpenCV <d-cite key="syn-mad22"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/mipgan2/001_013.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">MIPGAN-II <d-cite key="mipgan"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/013_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>b</i></div>
    </div>
</div>
<div class="caption">
    Comparison between a landmark-based morph (OpenCV) and representation-based morph (MIPGAN-II).
</div>

Historically, representation-based morphs have been created using Generative Adversarial Networks (GANs) <d-cite key="gan"></d-cite> a type of generative model which learns a mapping from the representation space to image space in an adversarial manner.
One challenge to figure out with GAN-based morphing is learning how to embed the original images into the representation space.
Researchers have proposed several techniques for this ranging from training an encoder network jointly with the GAN <d-cite key="bigan,aae,alae"></d-cite>, training an additional encoding network <d-cite key="e4e"></d-cite>, to inverting an image into the representation space via optimization <d-cite key="gan_opt_invert,Abdal_2019_ICCV"></d-cite>.
Once the original faces are mapped into the representation space, the representations can be morphed using linear interpolation <d-cite key="morgan"></d-cite> and additionally optimized w.r.t. some identity loss <d-cite key="mipgan"></d-cite>.

Landmark-based attacks are simple and surprisingly effective <d-cite key="sebastien_gan_threaten"></d-cite> when compared to representation-based attacks.
However, they struggle with prominent visual artefacts---especially outside the central face region.
While representation-based attacks do not suffer from the glaring visual artefacts which plague the landmark-based attacks, their ability to fool an FR system is severely lacking <d-cite key="sebastien_gan_threaten"></d-cite>.


In this blog post, I introduce a novel family of representation-based attacks collectively known as **Di**fusion **M**orphs (**DiM**) which addresses both the issues of prominent visual artefacts and inability to adequately fool an FR system.
The key idea is to use a type of iterative generative model known as *diffusion models* or *score-based models* <d-cite key="song2021scorebased"></d-cite>.
DiMs have achieved state-of-the-art performance on the face morphing tasks <d-cite key="blasingame_dim,fast_dim,greedy_dim"></d-cite> and have even yielded insight on related tasks, *e.g.*, guided generation of diffusion models <d-cite key="greedy_dim"></d-cite>.


## Diffusion
Diffusion models start with a diffusion process which perturbs the original data distribution $$p(\bfx)$$ on same subset of Euclidean space $$\X \subseteq \R^n$$ into isotropic Gaussian noise $$\mathcal{N}(\mathbf 0, \mathbf I)$$.
This process can be modeled with an Ito&#x302; [Stochastic Differential Equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation) (SDE) of the form
\begin{equation}
\mathrm{d}\bfx_t = f(t)\bfx_t\; \mathrm dt + g(t)\; \mathrm d\mathbf{w}_t
\end{equation}
where $$f, g$$ are real-valued functions, $$\{\bfw_t\}_{t \in [0, T]}$$ is the standard [Wiener process](https://en.wikipedia.org/wiki/Wiener_process) on time $$[0, T]$$, and $$\mathrm d\bfw_t$$ can be thought of as infinitesimal white noise.
The drift coefficient $$f(t)\bfx_t$$ is the deterministic part of the SDE and $$f(t)\bfx_t\;\mathrm dt$$ can be thought of as the [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) term of the SDE.
Conversely, the diffusion coefficient $$g(t)$$ is the stochastic part of the SDE which controls how much noise is injected into the system.
We can think of $$g(t)\;\mathrm d\bfw_t$$ as the *control* term of the SDE.

The solution to this SDE is a continuous collection of random variables $$\{\bfx_t\}_{t \in [0, T]}$$ over the real interval $$[0, T]$$, these random variables trace stochastic trajectories over the time interval.
Let $$p_t(\bfx_t)$$ denote the marginal [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) of $$\bfx_t$$.
Then $$p_0(\bfx_0) = p(\bfx)$$ is the data distribution, likewise, for some sufficiently large $$T \in \R$$ the terminal distribution $$p_T(\bfx_T)$$ is *close* to some tractable noise distribution $$\pi(\bfx)$$, called the **prior distribution**.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/diffusion_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of diffusion SDE. Original clean image (left) is slowly perturbed by additions of white noise until there is only noise (right).
</div>



### Reversing the diffusion SDE

So far we have only covered how to destroy data by perturbing it with white noise, however, for sampling we need to be able reverse this process to *create* data from noise.
Remarkably, Anderson <d-cite key="anderson_diffusion"></d-cite> showed that any Ito&#x302; SDE has a corresponding reverse SDE given in closed form by
\begin{equation}
\label{eq:rev_sde}
\mathrm d\bfx_t = [f(t)\bfx_t - g^2(t)\nabla_\bfx\log p_t(\bfx_t)]\;\mathrm dt + g(t)\; \mathrm d\tilde\bfw_t
\end{equation}
where $$\nabla_\bfx\log p_t(\bfx_t)$$ denotes the **score function** of $$p_t(\bfx_t)$$.
Note, that the *control* term is now driven by the *backwards* Wiener process defined as $$\tilde\bfw_t := \bfw_t - \bfw_T$$. 
To train a diffusion model, then, we just need to learn the score function <d-cite key="song2021scorebased"></d-cite> or some closely related quantity like the added noise or $$\bfx_0$$-prediction <d-cite key="ddpm,progressive_distillation"></d-cite>.
Many diffusion models used the following choice of drift and diffusion coefficients
\begin{equation}
    f(t) = \frac{\mathrm d \log \alpha_t}{\mathrm dt}\qquad g^2(t)= \frac{\mathrm d \sigma_t^2}{\mathrm dt} - 2 \frac{\mathrm d \log \alpha_t}{\mathrm dt} \sigma_t^2
\end{equation}
where $$\alpha_t,\sigma_t$$ form a noise schedule such that $$\alpha_t^2 + \sigma_t^2 = 1$$ and
\begin{equation}
    \bfx_t = \alpha_t\bfx_0 + \sigma_t\boldsymbol\epsilon_t \qquad \boldsymbol\epsilon_t \sim \mathcal{N}(\mathbf 0, \mathbf I)
\end{equation}
Diffusion models which use noise prediction are train a neural network $$\boldsymbol\epsilon_\theta(\bfx_t, t)$$ parameterized by $$\theta$$ to predict $$\boldsymbol\epsilon_t$$ given $$\bfx_t$$ which is equivalent to learning $$\boldsymbol\epsilon_\theta(\bfx_t, t) = -\sigma_t\nabla_\bfx \log p_t(\bfx_t)$$.
This choice of drift and coefficients forms the Variance Preserving SDE (VP SDE) type of diffusion SDE <d-cite key="song2021scorebased"></d-cite>.


### *Probability Flow* ODE
Song *et al.* <d-cite key="song2021scorebased"></d-cite> showed the existence of an ODE, dubbed the *Probability Flow* ODE, whose trajectories have the same marginals as Equation \eqref{eq:rev_sde} of the form
\begin{equation}
\label{eq:pf_ode}
\frac{\mathrm d\bfx_t}{\mathrm dt} = f(t)\bfx_t - \frac 12 g^2(t) \nabla_\bfx \log p_t(\bfx_t)
\end{equation}
One of key benefits of expressing diffusion models in ODE form is that ODEs are easily reversible, by simply integrating forwards and backwards in time we can encode images from $$p_0(\bfx_0)$$ into $$p_T(\bfx_T)$$ and back again.
With a neural network, often a U-Net <d-cite key="unet"></d-cite>, $$\boldsymbol\epsilon_\theta(\bfx_t, t)$$ trained on noise prediction the *empirical Probability Flow* ODE is now
\begin{equation}
\label{eq:empirical_pf_ode}
\frac{\mathrm d\bfx_t}{\mathrm dt} = f(t)\bfx_t  + \frac{g^2(t)}{2\sigma_t} \boldsymbol\epsilon_\theta(\bfx_t, t)
\end{equation}


## DiM
**Di**fusion **M**orphs (**DiM**) are a novel kind of face morphing algorithm which solve the *Probability Flow* ODE *both* forwards and backwards in time to achieve state-of-the-art visual fidelity and morphing performance far surpassing previous representation-based morphing attacks. 
The DiM framework could use many different diffusion backbones like DDPM <d-cite key="ddpm"></d-cite>, LDM <d-cite key="ldm"></d-cite>, DiT <d-cite key="Peebles2022DiT"></d-cite>, &amp;c.
However, in our work we opted to use the Diffusion Autoencoder <d-cite key="diffae"></d-cite> trained on the FFHQ dataset <d-cite key="stylegan"></d-cite> which conditions the noise prediction network $$\boldsymbol\epsilon_\theta(\bfx_t, \bfz, t)$$ on a latent representation, $$\bfz$$, of the target image.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/dim_overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    DiM Face morphing pipeline.
</div>

Before discussing the DiM pipeline let's establish some nomenclature.
Let $$\bfx_0^{(a)}$$ and $$\bfx_0^{(b)}$$ denote two bona fide images of identities $$a$$ and $$b$$.
Let $$\Z$$ denote the latent vector space and let $$\bfz_a = E(\bfx_0^{(a)})$$ denote the latent representation of $$a$$ and likewise for $$\bfz_b$$ where $$E: \X \to \Z$$ is an encoding network <d-cite key="diffae"></d-cite>.
Let $$\Phi(\bfx_0, \bfz, \mathbf{h}_\theta, \{t_n\}_{n=1}^N) \mapsto \bfx_T$$ denote a numerical ODE solver which takes an initial image $$\bfx_0$$, latent representation of $$\bfx_0$$ denoted by $$\bfz$$, the ODE in Equation \eqref{eq:empirical_pf_ode} denoted by $$\mathrm d\bfx_t / \mathrm dt = \mathbf{h}_\theta(\bfx_t, \bfz, t)$$, and timesteps $$\{t_n\}_{n=1}^N \subseteq [0, T]$$.
Let $$\mathrm{lerp}(\cdot, \cdot; \gamma)$$ denote linear interpolation by a weight of $$\gamma$$ on the first argument and likewise $$\mathrm{slerp}(\cdot, \cdot; \gamma)$$ for [spherical linear interpolation](https://en.wikipedia.org/wiki/Slerp).

The DiM pipeline works by first solving the *Probability Flow* ODE as time flows with $$N_F \in \N$$ timesteps forwards for both identities, *i.e.*,
\begin{equation}
    \bfx\_T^{(\\{a, b\\})} = \Phi(\bfx_0^{(\\{a, b\\})}, \bfz\_{\\{a, b\\}}, \mathbf{h}\_\theta, \\{t_n\\}\_{n=1}^{N_F})
\end{equation}
to find the noisy representations of the original images.
We then morph both the noisy and latent representations by a factor of $$\gamma = 0.5$$ to find
\begin{equation}
    \bfx\_T^{(ab)} = \mathrm{slerp}(\bfx\_T^{(a)}, \bfx\_T^{(b)}; \gamma) \qquad \bfz\_{ab} = \mathrm{lerp}(\bfz\_a, \bfz\_b; \gamma)
\end{equation}
Then we solve the *Probability Flow* ODE as time runs *backwards* with $$N \in \N$$ timesteps $$\{\tilde t_n\}_{n=1}^N$$ where $$\tilde t_1 = T$$ such that
\begin{equation}
    \bfx\_0^{(ab)} = \Phi(\bfx\_T^{(ab)}, \bfz\_{ab}, \mathbf{h}\_\theta, \\{\tilde t_n\\}\_{n=1}^{N})
\end{equation}

All these equations for the DiM can be summarized with the following PyTorch pseudo code.

```python
def dim(model, encoder, diff_eq, ode_solver, x_a, x_b, eps=0.002, T=80., n_encode=250, n_sample=100):
    """
    DiM algorithm.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        encoder (nn.Module): Encoder network.
        diff_eq (function): RHS of Probabilty Flow ODE.
        ode_solver (function): Numerical ODE solver, e.g., RK-4, Adams-Bashforth.
        x_a (torch.Tensor): Image of identity a.
        x_b (torch.Tensor): Image of identity b.
        eps (float): Starting timstep. Defaults to 0.002. For numeric stability.
        T (float): Terminal timestep. Defaults to 80.
        n_encode (int): Number of encoding steps. Defaults to 250.
        n_sample (int): Number of sampling steps. Defaults to 100.

    Returns:
        x_morph (torch.Tensor): The morphed image.
    """

    # Create latents
    z_a = encoder(x_a)
    z_b = encoder(x_b)
    
    # Encode images into noise
    timesteps = torch.linspace(eps, T, n_encode)
    xs = ode_solver(torch.cat((x_a, x_b), dim=1), torch.cat((z_a, z_b), dim=1), diff_eq, timesteps)
    x_a, x_b = xs.chunk(2, dim=1)

    # Morph representations
    z_ab = torch.lerp(z_a, z_b, 0.5)
    x_ab = slerp(x_a, x_b, 0.5)  # assumes slerp is defined somewhere

    # Generate morph
    timesteps = torch.linspace(T, eps, n_sample) 
    x_morph = ode_solver(x_ab, z_ab, diff_eq, timesteps)
    
    return x_morph
```

A few important observations.

* The number of encoding steps $$N_F$$ does not need to equal to the number of sampling steps $$N$$, *i.e.*, these process are decoupled.
* We can use separate numerical ODE solvers for encoding and sampling.
* Samples are iteratively generated, meaning there are more ways to guide the morph generation process

By using the powerful and flexible framework of diffusion models we can achieve **state-of-the-art** visual fidelity and morphing performance (measured in MMPMR<d-footnote>The Mated Morph Presentation Match Rate (MMPMR) <d-cite key="mmpmr"></d-cite> metric is a measure of how vulnerable a FR system is to a morphing attack and is defined as
$$M(\delta) = \frac{1}{M} \sum_{n=1}^M \bigg\{\bigg[\min_{n \in \{1,\ldots,N_m\}} S_m^n\bigg] > \delta\bigg\}$$
where $\delta$ is the verification threshold, $S_m^n$ is the similarity score of the $n$-th subject of morph $m$, $N_m$ is the total number of contributing subjects to morph $m$, and $M$ is the total number of morphed images.
</d-footnote>).
Visually, the morphs produced by DiM look more realistic than the landmark-based morphs with their prominent artefacts and even better than the GAN-based morphs.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/004_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>a</i></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/opencv/004_012.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">OpenCV <d-cite key="syn-mad22"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/dim_a/004_012.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">DiM <d-cite key="blasingame_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/mipgan2/004_012.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">MIPGAN-II <d-cite key="mipgan"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/012_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>b</i></div>
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/009_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>a</i></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/opencv/009_112.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">OpenCV <d-cite key="syn-mad22"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/dim_a/009_112.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">DiM <d-cite key="blasingame_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/mipgan2/009_112.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">MIPGAN-II <d-cite key="mipgan"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/112_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>b</i></div>
    </div>
</div>
<div class="caption">
    Comparison between a morphed image generated via OpenCV, DiM, and MIPGAN-II<d-footnote>Due to the computational demands needed to train a diffusion model on the image space the U-Net we use was trained at a $256 \times 256$ resolution, whereas less computationally demanding morphs are at a higher resolution. This difference is trivial, however, as most images are cropped and down-sampled, <i>e.g.</i>, $112 \times 112$ before being passed to the FR system.</d-footnote>.
</div>


| Morphing Attack | AdaFace | ArcFace | ElasticFace |
| :-------------- | -------:|--------:|------------:|
| FaceMorpher <d-cite key="syn-mad22"></d-cite>  | 89.78 | 87.73 | 89.57|
| OpenCV <d-cite key="syn-mad22"></d-cite>  | 94.48 | 92.43 | 94.27 |
| MIPGAN-I <d-cite key="mipgan"></d-cite>  | 72.19 | 77.51 | 66.46 |
| MIPGAN-II <d-cite key="mipgan"></d-cite>  | 70.55 | 72.19 | 65.24 |
| DiM <d-cite key="blasingame_dim"></d-cite>  | 92.23 | 90.18 | 93.05 |

<div class="caption">
    Vulnerability of different of FR systems across different morphing attacks on the SYN-MAD 20222 dataset. False Match Rate (FMR) is set at 0.1% for all FR systems. Higher is better.
</div>

### High-order ODE solvers

The *Probability Flow* ODE in Equation \eqref{eq:empirical_pf_ode} can be transformed from a [stiff ODE](https://en.wikipedia.org/wiki/Stiff_equation) into a semi-linear ODE by using *[exponential integrators](https://en.wikipedia.org/wiki/Exponential_integrator)*.
Lu *et al.* <d-cite key="dpm_solver"></d-cite> showed that the exact solution at time $$t$$ given $$\bfx_s$$ starting from time $$s$$ is given by
\begin{equation}
    \bfx_t = \underbrace{\vphantom{\int_{\lambda_s}}\frac{\alpha_t}{\alpha_s}\bfx_s}\_{\textrm{analytically computed}} - \alpha_t \underbrace{\int_{\lambda_s}^{\lambda_t} e^{-\lambda} \boldsymbol\epsilon_\theta(\bfx_\lambda, \lambda)\;\mathrm d\lambda}\_{\textrm{approximated}}
\end{equation}
where $$\lambda_t := \log(\alpha_t / \sigma_t)$$ is one half the log-[SNR](https://en.wikipedia.org/wiki/Signal-to-noise_ratio).
which removes the error for the linear term.
The remaining integral term approximated by exploiting the wealth of literature on exponential integrators.
E.g., take a $$k$$-th order Taylor expansion around $$\lambda_s$$ and apply multi-step methods to approximate the $$n$$-th order derivatives.

Previously, DiM used *DDIM* a first-order ODE solver of the form
\begin{equation}
    \bfx_{t_n} = \frac{\alpha_{t_n}}{\alpha_{t_{n-1}}}\bfx_{t_{n-1}} - \sigma_{t_n}(e^{h_n} - 1)\boldsymbol\epsilon_\theta(\bfx_{t_{n-1}}, \bfz, t_{n-1})
\end{equation}
where $$h_n = \lambda_{t_n} - \lambda_{t_{n-1}}$$ is the step size of the ODE solver.
This, however, means that we have a [global truncation error](https://en.wikipedia.org/wiki/Truncation_error_(numerical_integration)) of $$\mathcal{O}(h)$$ (Theorem 3.2 in <d-cite key="dpm_solver"></d-cite>).
This means that we need a very small step size to keep the error small or, in other words, we need a **large** number of sampling steps to accurate encode **and** sample the *Probability Flow* ODE.
The initial DiM implementation <d-cite key="blasingame_dim"></d-cite> used $$N_F = 250$$ encoding steps and $$N = 100$$ sampling steps.
An initial study we conducted found these to be optimal with the first-order solver.

By using a second-order multistep ODE solver *DPM++ 2M* <d-cite key="lu2023dpmsolver"></d-cite> we can significantly reduce the number of Network Function Evalutions (NFE), *i.e.*, the number of times we run the neural nework---a very expenesive operation, without meaningfully reducing the performance of the morphing attack.
Likewise, the visual impact on the appearance of the morphed face seems to be quite small.
Therefore, by switching to a high-order ODE solver we can **significantly** reduce the NFE while retaining comparable performance to vanilla DiM.
We call this approach *Fast-DiM* due to its reduced computational demand.

| ODE Solver | NFE (&darr;) | AdaFace (MMPMR &uarr;) | ArcFace (MMPMR &uarr;)| ElasticFace (MMPMR &uarr;) |
| :-------------- | ------: | -------:|--------:|------------:|
| DDIM | 100 | 92.23 | 90.18 | 93.05 |
| DPM++ 2M | 50 | 92.02 | 90.18 | 93.05 |
| DPM++ 2M | 20 | 91.26 | 89.98 | 93.25 |

<div class="caption">
    Impact of the ODE solver in the sampling process of DiM.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/fast_dim_ddim_vs_dpm_pp_2m.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    From left to right: face <i>a</i>, morph generated with DDIM <i>(N = 100)</i>, morph generated with DPM++ 2M <i>(N = 20)</i>, face <i>b</i>.
</div>


While we have shown that by solving the *Probability Flow* ODE as time runs backwards with high-order solves we can significantly reduce the NFE with no downside, we have yet to explore the impact on the encoding process.
Interestingly, we discovered the success of DiM morphs is highly sensitive to the accuracy of the encoding process.
Merely, dropping the number of encoding steps from 250 to 100 showed a marked decline in performance; conversely, increasing sampling steps over 100 showed no benefit.
We posit that this might be due to the odd nature of the encoded images.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/ddim_encode.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    From left to right: Original image, encoded image, true white noise. The encoded noise has strange artefacts in what was the background of the image and the outline of the head is still visible even when fully encoded into noise.
</div>



| ODE Solver | NFE (&darr;) | AdaFace (MMPMR &uarr;) | ArcFace (MMPMR &uarr;)| ElasticFace (MMPMR &uarr;) |
| :-------------- | ------: | -------:|--------:|------------:|
| DDIM | 100 | 91.82 | 88.75 | 91.21 |
| DPM++ 2M | 100 | 90.59 | 87.15 | 90.8 |
| DDIM | 50 | 89.78 | 86.3 | 89.37 |
| DPM++ 2M | 50 | 90.18 | 86.5 | 88.96 |

<div class="caption">
    Impact of the ODE solver in the encoding process of DiM. Sampling was all done with DPM++ 2M <i>(N = 50)</i>
</div>



## Greedy Guided Generation
Identity-guided generation massively improved the effectiveness of GAN-based morphs <d-cite key="mipgan"></d-cite>; however, these techniques relied on being able to solve
\begin{equation}
    \bfz^* = \argmin_{\bfz \in \Z} \quad \mathcal{L}(g(\bfz), \bfx\_0^{(a)}, \bfx\_0^{(b)})
\end{equation}
efficiently w.r.t some loss function $$\mathcal{L}$$ with generator network $$g: \Z \to \X$$ and requires the gradient $$\partial \mathcal{L} / \partial \bfz$$ to perform optimziation in the latent space.
However, obtaining an analogous gradient in DiM is not a trivial task due to the iterative generative nature of diffusion models.
Naïvely, trying to estimate the gradient by backproping through all the neural network evaluations is horrendously inefficient and will consume the memory of all but the largest of rigs.
While there is active research on *efficiently* estimating the gradients for diffusion models $$\partial \mathcal{L} / \partial \bfx_T$$ and $$\partial \mathcal{L} / \partial \bfz$$, see Marion *et al.* <d-cite key="implicit_diffusion"></d-cite> and our own work <d-cite key="adjoint_deis"></d-cite>, we instead propose an *elegant* solution tailored for the face morphing problem.

> We don't actually care about the intermediate latents $$\bfx_T^{(ab)}, \bfz_{ab}$$ just the output $$\bfx_0^{(ab)}$$

With this insight we instead propose a greedy strategy that optimizes each step of the ODE solver $$\Phi$$ w.r.t. to the identity loss.
We propose a new family of DiM algorithms called *Greedy-DiM* which leverages this greedy strategy to generate optimal morphs.


| | DiM| Fast-DiM  | Morph-PIPE  | Greedy-DiM |
| :---- | :---- | :---- | :----- | :----- |
| ODE Solver | DDIM | DPM++ 2M | DDIM | DDIM |
| Number of sampling steps | 100 | 50 | 2100 | 20 |
| Heuristic function | ✘ | ✘ | $$\mathcal{L}_{ID}^*$$ | $$\mathcal{L}_{ID}^*$$ |
| Search strategy | ✘ | ✘ | Brute-force search | Greedy optimization |
| Search space | ✘ | ✘ | Set of 21 morphs | $$\X$$ |
| Probability the search space contains the optimal solution | ✘ | ✘ | 0 | 1 |

<div class="caption">
    High level overview of all current DiM algorithms.
</div>



To further motivate Greedy-DiM we highlight recent work by Zhang *et al.* <d-cite key="morph_pipe"></d-cite> who also explored the problem of identity-guided generation with DiMs.
They proposed Morph-PIPE a simple extension upon our DiM algorithm by incorporating identity information and succeeds upon improving upon vanilla DiM.
The approach can be summarized as
1. Find latent representations $$\bfx_T^{(a)},\bfx_T^{(b)},\bfz_a,\bfz_b$$.
2. Create a set of $$N = 21$$ blend parameters $$\{\gamma_n\}_{n=1}^N \subseteq [0, 1]$$.
3. Generate $$N$$ morphs with latents $$\mathrm{slerp}(\bfx_T^{(a)},\bfx_T^{(b)};\gamma_n)$$ and $$\mathrm{lerp}(\bfz_a, \bfz_b; \gamma_n)$$.
4. Select the best morph w.r.t. the identity loss $$\mathcal{L}_{ID}^*$$<d-footnote>
The identity loss was initially proposed by Zhang <i>et al.</i> <d-cite key="mipgan"></d-cite> and was slightly modified by <d-cite key="morph_pipe"></d-cite> defined as the sum of two sub-losses:
$$\mathcal{L}_{ID} = d(v_{ab}, v_a) + d(v_{ab}, v_b)\qquad \mathcal{L}_{diff} = \big|d(v_{ab}, v_a) - d(v_{ab}, v_b))\big |$$
with
$$\mathcal{L}_{ID}^* = \mathcal{L}_{ID} + \mathcal{L}_{diff}\label{eq:loss_id}$$
where $v_a = F(\mathbf{x}_0^{(a)}), v_b = F(\mathbf{x}_0^{(b)}), v_{ab} = F(\mathbf{x}_0^{(ab)})$, and $F: \mathcal{X} \to V$ is an FR system which embeds images into a vector space $V$ which is equipped with a measure of distance, $d$, <i>e.g.</i> cosine distance.</d-footnote>.

While Morph-PIPE does outperform DiM it, however, does have a few notable drawbacks.

1. The approach is very computationally expensive requiring the user to fully generate a set of $$N = 21$$ morphs.
2. The probability that Morph-PIPE actually finds the optimal morph is 0, even as $$N \to \infty$$. More on this later.

In contrast our Greedy-DiM algorithm addresses both of these issues.
First, by greedily optimizing a neighborhood around the predicted noise at each step of the ODE solver we reduce **significantly** reduce the number of calls to the U-Net while simultaneously expanding the size of the search space.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/greedy_dim_star.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Overview of a single step of the Greedy-DiM algorithm. Additions compared to vanilla DiM highlighted in green.
</div>

Remember that for VP type SDEs a sample at time $$t$$ can be expressed as $$\bfx_t = \alpha_t\bfx_0 + \sigma_t\boldsymbol\epsilon_t$$, this means we can rearrange the equation to solve for the denoised image $$\bfx_0$$ to which we find
\begin{equation}
    \label{eq:x-pred}
    \bfx_0 = \frac{\bfx_t - \sigma_t\boldsymbol\epsilon_t}{\alpha_t}
\end{equation}
Therefore, we recast our noise prediction from $$\boldsymbol\epsilon_\theta(\bfx_t, \bfz, t)$$ into a prediction of $$\bfx_0$$.
It is this approximation which we then pass to the identity loss $$\mathcal{L}_{ID}^*$$.
We can then easily calculate $$\nabla_{\boldsymbol\epsilon} \mathcal{L}_{ID}^*$$ using an automatic differentation library and preform gradient descent on $$\boldsymbol\epsilon$$ to find the optimal $$\boldsymbol\epsilon$$ w.r.t. the identity loss.
We outline the Greedy-DiM algorithm with the following PyTorch pseudo code.

```python
def dim(model, encoder, diff_eq, ode_solver, x_a, x_b, loss_func, eps=0.002, T=80., n_encode=250, n_sample=20, n_opt_steps=50, opt_kwargs={}):
    """
    DiM algorithm.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        encoder (nn.Module): Encoder network.
        diff_eq (function): RHS of Probabilty Flow ODE.
        ode_solver (function): Numerical ODE solver, must be first-order!
        x_a (torch.Tensor): Image of identity a.
        x_b (torch.Tensor): Image of identity b.
        loss_fn (func): Loss function to guide generation.
        eps (float): Starting timstep. Defaults to 0.002. For numeric stability.
        T (float): Terminal timestep. Defaults to 80.
        n_encode (int): Number of encoding steps. Defaults to 250.
        n_sample (int): Number of sampling steps. Defaults to 100.
        n_opt_steps (int): Number of optimization steps per timestep of the ODE solver. Defaults to 50.
        opt_kwargs (dict): Dictionary of optimizer arguments. Defaults to {}.

    Returns:
        x_morph (torch.Tensor): The morphed image.
    """

    # Create latents
    z_a = encoder(x_a)
    z_b = encoder(x_b)
    
    # Encode images into noise
    timesteps = torch.linspace(eps, T, n_encode)
    xs = ode_solver(torch.cat((x_a, x_b), dim=1), torch.cat((z_a, z_b), dim=1), diff_eq, timesteps)
    x_a, x_b = xs.chunk(2, dim=1)

    # Morph representations
    z = torch.lerp(z_a, z_b, 0.5)
    xt = slerp(x_a, x_b, 0.5)  # assumes slerp is defined somewhere

    # Generate morph
    timesteps = torch.linspace(T, eps, n_sample) 

    for i, t in enumerate(timesteps[:-1]):
        with torch.no_grad():
            eps = model(xt, z, t)

        eps = eps.detach().clone().requires_grad(True)
        opt = torch.optim.RAdam([eps], **opt_kwargs)

        x0_pred = convert_eps_to_x0(eps, t, xt.detach())  # assumes Eq. (13) is implemented somewhere
        best_loss = loss_fn(x0_pred)
        best_eps = eps

        for _ in range(n_opt_steps):
            opt.zero_grad()

            x0_pred = convert_eps_to_x0(eps, t, xt.detach())
            loss = loss_fn(x0_pred)

            do_update = (loss < best_loss).float()  # handles batches of morphs
            best_loss = do_update * loss + (1. - do_update) * best_loss
            best_eps = do_update[:, None, None, None] * eps + (1. - do_update)[:, None, None, None] * best_eps

            loss.mean().backward()
            opt.step()

        eps = best_eps

        xt = ode_solver(xt, z, diff_eq, [t, timesteps[i+1]])
            
    return xt
```

This simple greedy strategy results in **massive** improvements over DiM in both visual fidelity and morphing performance.
The performance of Greedy-DiM is **unreasonably effective** fooling the studied FR systems a **100%** of the time.

| Morphing Attack | NFE (&darr;) | AdaFace (&uarr;)| ArcFace (&uarr;) | ElasticFace (&uarr;) |
| :-------------- | ----: | -------:|--------:|------------:|
| FaceMorpher <d-cite key="syn-mad22"></d-cite>  | - | 89.78 | 87.73 | 89.57|
| OpenCV <d-cite key="syn-mad22"></d-cite>  | - |94.48 | 92.43 | 94.27 |
| MIPGAN-I <d-cite key="mipgan"></d-cite>  | - |72.19 | 77.51 | 66.46 |
| MIPGAN-II <d-cite key="mipgan"></d-cite>  | - |70.55 | 72.19 | 65.24 |
| DiM <d-cite key="blasingame_dim"></d-cite>  | 350 | 92.23 | 90.18 | 93.05 |
| Fast-DiM <d-cite key="fast_dim"></d-cite>  | 300 | 92.02 | 90.18 | 93.05 |
| Morph-PIPE <d-cite key="morph_pipe"></d-cite>  | 2350 | 95.91 | 92.84 | 95.3 |
| Greedy-DiM <d-cite key="greedy_dim"></d-cite>  | 270 | **100** | **100** | **100** |

<div class="caption">
    Vulnerability of different of FR systems across different morphing attacks on the SYN-MAD 20222 dataset. False Match Rate (FMR) is set at 0.1% for all FR systems. Higher is better.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/066_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>a</i></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/dim_a/066_087.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">DiM <d-cite key="blasingame_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/greedy_dim/066_087.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Greedy-DiM <d-cite key="greedy_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/morph_pipe/066_087.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Morph-PIPE <d-cite key="morph_pipe"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/087_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>b</i></div>
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/096_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>a</i></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/dim_a/096_137.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">DiM <d-cite key="blasingame_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/greedy_dim/096_137.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Greedy-DiM <d-cite key="greedy_dim"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/morph_pipe/096_137.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Morph-PIPE <d-cite key="morph_pipe"></d-cite></div>
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/frll/bona_fide/137_03.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <div class="caption">Face <i>b</i></div>
    </div>
</div>
<div class="caption">
    Visual comparison of DiM methods. Notice the prominent red and blue saturation artefacts present in DiM and Morph-PIPE. These are absent in Greedy-DiM.
</div>

Given the unreasonably strong empirical performance a natural question is to ask why Greedy-DiM performs so well.
First, this greedy strategy is globally optimal (Theorem 3.1 <d-cite key="greedy_dim"></d-cite>).
This means that at every timestep the locally optimal choice is globally optimal.
Because of Equation \eqref{eq:x-pred} it follows that for any two points along the *same* solution trajectory $$\bfx_s, \bfx_t$$ with $$s, t \in [0, T]$$ and using a first-order solver for the *Probability Flow* ODE it follows that $$\boldsymbol\epsilon_s = \boldsymbol\epsilon_t$$, i.e., the have the same realization of noise<d-footnote>This is only true for diffusion models using the <i>Probability Flow</i> ODE formulation. It is more complicated to ensure this for diffusion SDEs, the math of which is beyond the scope of this post.</d-footnote>
Hence, at any time $$t$$, the $$\boldsymbol\epsilon_t$$ which minimizes the identity loss is globally optimal.
The second justification for the *unreasonable* performance of Greedy-DiM lies in the structure of its search space.


### The search space of Greedy-DiM is well-posed
The search space of Greedy-DiM is well-posed, meaning the probability the search space contains the optimal solution is 1.
Formally, let $$\pr$$ be probability measure on compact subset $$\X$$ which denotes the distribution of the optimal face morph.
For technical reasons we also assume $$\pr$$ is absolutely continuous<d-footnote>
A measure $\mu$ on measurable space $(\mathcal{X}, \Sigma)$ is said to be absolutely continuous w.r.t. $\nu$ iff
$$(\forall A \in \Sigma) \quad \nu(A) = 0 \implies \mu(A) = 0$$
</d-footnote> w.r.t. the $$n$$-dimensional Lebesgue measure $$\lambda^n$$ on $$\X$$.
The probability the optimal face morph lies on some set $$A \subseteq \X$$ is denoted as $$\pr(A)$$.
Then let $$\mathcal{S}_P$$ denote the search space of Morph-PIPE and let $$\mathcal{S}^*$$ of Greedy-DiM.
With some measure theory it is simple to prove that $$\pr(\mathcal{S}_P) = 0$$ and $$\pr(\mathcal{S}^*) = 1$$ (Theorem 3.2 <d-cite key="greedy_dim"></d-cite>).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dim/search_space.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Illustration of the search space of different DiM algorithms on $\mathbb{R}^2$. The purple denotes Morph-PIPE while green denotes the search space of Greedy-DiM.
</div>

Intuitively, the $$N = 21$$ countably finite number of morphs that Morph-PIPE explores is infinitesimally small compared to the whole search space which is compact subset of Euclidean space.

> The search space of Greedy-DiM allows it to find the optimal morph, while Morph-PIPE's forbids this

Due to this ability of Greedy-DiM to better explore the space of morphs we can find the optimal morph.
This combined with globally optimal substructure of the identity-guided DiM problem gives a reasonable explanation for why Greedy-DiM fooled the studied FR systems **100%** of the time.

Note, the reason the numerical ODE solver $$\Phi$$ must be first-order is that our greedy strategy breaks the guarantees needed for to accurately estimate the $$n$$-th order derivatives used in high-order solvers.


## Concluding Remarks

This blog post gives a detailed introduction to DiM.
I develop the motivation for DiM, and it's successors Fast-DiM and Greedy-DiM, through a principled analysis of the face morphing problem.
I was then able to demonstrate that this new family of morphing attacks pose a serious threat to FR systems and represent a significant advance in face morphing attacks.
This post is a compilation of several papers we have published in the last few years. Please visit them if you are interested in more details.

* [Zander W. Blasingame and Chen Liu. *Leveraging Diffusion for Strong and High Quality Face Morphing Attacks*. IEEE TBIOM 2024.](https://arxiv.org/abs/2301.04218)
* [Zander W. Blasingame and Chen Liu. *Fast-DiM: Towards Fast Diffusion Morphs*. IEEE Security & Privacy 2024.](https://arxiv.org/abs/2310.09484)
* [Richard E. Neddo, Zander W. Blasingame, and Chen Liu. *The Impact of Print-and-Scan in Heterogeneous Morph Evaluation Scenarios*. IJCB 2024](https://arxiv.org/abs/2404.06559)
* [Zander W. Blasingame and Chen Liu. *Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs*. IJCB 2024](https://arxiv.org/abs/2404.06025)

For more reading on more *efficiently* estimating the gradients of diffusion models please check out our recent paper.

* [Zander W. Blasingame and Chen Liu. *AdjointDEIS: Efficient Gradients for Diffusion Models*. Pre-print.](https://arxiv.org/abs/2405.15020)

Interestingly, our work on Greedy-DiM bares some similarity to recent work done by Yu *et al.* <d-cite key="yu2023freedom"></d-cite> as they also end up doing a one-shot gradient estimation via $$\bfx_0$$-prediction; however, they develop their approach from the prospective of energy guidance.

It is my belief that the insights gained here are not only relevant to face morphing, but also to other downstream task with diffusion models. *E.g.,* template inversion <d-cite key="template_inversion_sebastien"></d-cite>, guided generation <d-cite key="yu2023freedom,doodl"></d-cite>, adversarial attacks <d-cite key="chen2023diffusion"></d-cite>, and even other modalities like audio.

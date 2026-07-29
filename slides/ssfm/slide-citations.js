/* ───────────────────────────────────────────────────────────────────
   slide-citations.js — numeric [N] citation resolver for the Rex
   reveal.js deck.  Numbers are assigned in document order (first
   appearance), tooltips carry full bibliographic info, and an
   auto-built References slide is rendered into #references-list-host.

   Markup:
     HTML:  <span class="cite" data-key="key"></span>
     SVG:   <text class="tl-cite" data-key="key" x=".." y=".."></text>

   Catalog C holds only the papers directly cited in the deck.
   ─────────────────────────────────────────────────────────────────── */
(function(){
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';

  // ─── Bibliographic catalog (papers directly mentioned in the deck) ───
  const C = {
    'song2021denoising':  {authors:'Song, J., Meng, C., Ermon, S.', title:'Denoising Diffusion Implicit Models', venue:'ICLR', year:2021, url:'https://arxiv.org/abs/2010.02502'},
    'zhuang2021mali':      {authors:'Zhuang, J., Dvornek, N., Tatikonda, S., Duncan, J.', title:'MALI: A memory-efficient and reverse-accurate integrator for neural ODEs', venue:'ICLR', year:2021, url:'https://openreview.net/forum?id=blfSjHeFM_e'},
    'kidger2021efficient': {authors:'Kidger, P., Foster, J., Li, X. C., Lyons, T.', title:'Efficient and Accurate Gradients for Neural SDEs', venue:'NeurIPS', year:2021, url:'https://arxiv.org/abs/2105.13493'},
    'kidger_thesis':       {authors:'Kidger, P.', title:'On Neural Differential Equations', venue:'PhD Thesis, University of Oxford', year:2021, url:'https://arxiv.org/abs/2202.02435'},
    'wallace2023edict':    {authors:'Wallace, B., Gokul, A., Naik, N.', title:'EDICT: Exact Diffusion Inversion via Coupled Transformations', venue:'CVPR', year:2023, url:'https://arxiv.org/abs/2211.12446'},
    'zhang2024bdia':       {authors:'Zhang, G., Lewis, J. P., Kleijn, W. B.', title:'Exact Diffusion Inversion via Bidirectional Integration Approximation', venue:'ECCV', year:2024, doi:'10.1007/978-3-031-72998-0_2'},
    'wang2024belm':        {authors:'Wang, F., Yin, H., Dong, Y.-J., Zhu, H., Zhang, C., Zhao, H., Qian, H., Li, C.', title:'BELM: Bidirectional Explicit Linear Multi-step Sampler for Exact Inversion in Diffusion Models', venue:'NeurIPS', year:2024, url:'https://openreview.net/forum?id=ccQ4fmwLDb'},
    'mccallum2024foster':  {authors:'McCallum, S., Foster, J.', title:'Efficient, Accurate and Stable Gradients for Neural ODEs', venue:'arXiv:2410.11648', year:2024, url:'https://arxiv.org/abs/2410.11648'},
    'lu2022dpmsolverpp':   {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models', venue:'arXiv:2211.01095', year:2022, url:'https://arxiv.org/abs/2211.01095'},
    'lu2022dpmsolver':     {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps', venue:'NeurIPS', year:2022, url:'https://arxiv.org/abs/2206.00927'},
    'gonzalez2024seeds':   {authors:'Gonzalez, M., Fernandez Pinto, N., Tran, T., Hajri, H., Masmoudi, N. et al.', title:'SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models', venue:'NeurIPS', year:2024, url:'https://arxiv.org/abs/2305.14267'},
    'zhang2023gddim':      {authors:'Zhang, Q., Tao, M., Chen, Y.', title:'gDDIM: Generalized Denoising Diffusion Implicit Models', venue:'ICLR', year:2023, url:'https://openreview.net/forum?id=1hKE9qjvz-'},
    'foster2024high':      {authors:'Foster, J. M., dos Reis, G., Strange, C.', title:'High order splitting methods for SDEs satisfying a commutativity condition', venue:'SIAM J. Numer. Anal.', year:2024, url:'https://doi.org/10.1137/22M1535304'},
    'foster2020optimal':   {authors:'Foster, J., Lyons, T., Oberhauser, H.', title:'An optimal polynomial approximation of Brownian motion', venue:'SIAM J. Numer. Anal.', year:2020, url:'https://doi.org/10.1137/19M124447X'},
    'rossler2010runge':    {authors:'Rößler, A.', title:'Runge–Kutta methods for the strong approximation of solutions of stochastic differential equations', venue:'SIAM J. Numer. Anal.', year:2010, url:'https://doi.org/10.1137/09076636X'},
    'dormand1980family':   {authors:'Dormand, J. R., Prince, P. J.', title:'A family of embedded Runge-Kutta formulae', venue:'J. Comp. Appl. Math., 6(1):19&ndash;26', year:1980, url:'https://doi.org/10.1016/0377-0427(80)90013-3'},
    // ─── Boltzmann sampling baselines (slide 11) ───
    'tan2025scalable':     {authors:'Tan, C. B., Bose, J., Lin, C., Klein, L., Bronstein, M. M., Tong, A.', title:'Scalable Equilibrium Sampling with Sequential Boltzmann Generators', venue:'ICML', year:2025, url:'https://openreview.net/forum?id=U7eMoRDIGi'},
    'rehman2026efficient': {authors:'Rehman, D., Davis, O., Lu, J., Tang, J., Bronstein, M., Bengio, Y., Tong, A., Bose, A. J.', title:'Efficient Regression-based Training of Normalizing Flows for Boltzmann Generators', venue:'ICLR', year:2026, url:'https://openreview.net/forum?id=ctdnzPxDI3'},
    'peebles2023scalable': {authors:'Peebles, W., Xie, S.', title:'Scalable Diffusion Models with Transformers', venue:'ICCV', year:2023, url:'https://arxiv.org/abs/2212.09748'},
    // ─── future work: effectively symmetric schemes ───
    'shmelev2025explicit':       {authors:'Shmelev, D., Salvi, C.', title:'Explicit and Effectively Symmetric Schemes for Neural SDEs', venue:'arXiv:2509.20599', year:2025, url:'https://arxiv.org/abs/2509.20599'},
    'shmelev2025explicit_older': {authors:'Shmelev, D., Ebrahimi-Fard, K., Tapia, N., Salvi, C.', title:'Explicit and Effectively Symmetric Runge-Kutta Methods', venue:'arXiv:2507.21006', year:2025, url:'https://arxiv.org/abs/2507.21006'},
  };

  /* ─── Bibliographic entry formatter (pure) ─────────────────────── */
  function fmtBib(key){
    const c = C[key];
    if(!c) return '[missing: ' + key + ']';
    let s = '';
    if(c.authors) s += c.authors;
    if(c.year)    s += ' (' + c.year + ').';
    if(c.title){
      s += ' ' + c.title;
      if(!/[.!?]$/.test(c.title)) s += '.';
    }
    if(c.venue) s += ' ' + c.venue + '.';
    if(c.url)       s += ' <a href="' + c.url + '" target="_blank" rel="noopener">link</a>';
    else if(c.doi)  s += ' <a href="https://doi.org/' + c.doi + '" target="_blank" rel="noopener">doi</a>';
    return s;
  }

  /* ─── Plain-text bib for SVG <title> tooltips (pure) ───────────── */
  function fmtBibPlain(key){
    const c = C[key];
    if(!c) return '[missing: ' + key + ']';
    let s = '';
    if(c.authors) s += c.authors;
    if(c.year)    s += ' (' + c.year + ').';
    if(c.title)   s += ' ' + c.title + (/[.!?]$/.test(c.title) ? '' : '.');
    if(c.venue)   s += ' ' + c.venue + '.';
    return s;
  }

  function isSVGNode(el){
    return el.namespaceURI === SVGNS || el.tagName.toLowerCase() === 'text';
  }

  /* ─── Build the inline HTML citation (pure-ish: returns DOM node) ── */
  function makeHtmlCite(key, num){
    const a = document.createElement('a');
    a.className = 'cite-link';
    a.setAttribute('data-ref', num);
    a.setAttribute('role', 'link');
    a.setAttribute('tabindex', '0');
    a.innerHTML =
      '<span class="cite-num">[' + num + ']</span>' +
      '<span class="cite-tip">' + fmtBib(key) + '</span>';
    return a;
  }

  /* ─── Fill an SVG <text> cite in place ─────────────────────────── */
  function fillSvgCite(el, key, num){
    el.textContent = '[' + num + ']';
    el.setAttribute('data-bib', fmtBibPlain(key));
  }

  /* ─── Floating CSS tooltip for SVG timeline cites ──────────────── */
  function wireSvgTooltips(){
    const tip = document.getElementById('citeTooltip');
    if(!tip) return;

    function showTip(el, e){
      tip.innerHTML = fmtBib(el.getAttribute('data-key'));
      tip.classList.add('is-visible');
      moveTip(e);
    }
    function moveTip(e){
      let x = e.clientX + 14, y = e.clientY + 14;
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      if(x + tw > window.innerWidth - 8)  x = e.clientX - tw - 14;
      if(y + th > window.innerHeight - 8) y = e.clientY - th - 14;
      tip.style.left = x + 'px';
      tip.style.top  = y + 'px';
    }
    function hideTip(){ tip.classList.remove('is-visible'); }

    document.querySelectorAll('.tl-cite').forEach(el => {
      el.style.cursor = 'help';
      el.addEventListener('mouseenter', (e) => showTip(el, e));
      el.addEventListener('mousemove',  moveTip);
      el.addEventListener('mouseleave', hideTip);
    });
  }

  /* ─── Main pass: number by document order, render, collect refs ── */
  function process(){
    const seen = Object.create(null);   // key -> number
    const refs = [];                     // [{key, num}] in order
    let counter = 0;

    const nodes = document.querySelectorAll('.cite, .tl-cite');

    nodes.forEach(el => {
      const key = (el.getAttribute('data-key') || '').trim();
      if(!key) return;
      if(!C[key]){ el.classList.add('cite-missing'); return; }

      let num = seen[key];
      if(num === undefined){
        counter += 1;
        seen[key] = counter;
        refs.push({key, num: counter});
        num = counter;
      }

      if(isSVGNode(el)){
        fillSvgCite(el, key, num);
      } else {
        el.replaceWith(makeHtmlCite(key, num));
      }
    });

    buildRefList(refs);
    wireCiteNavigation();
    wireSvgTooltips();
  }

  /* ─── References slide builder ────────────────────────────────── */
  function buildRefList(refs){
    const host = document.getElementById('references-list-host');
    if(!host) return;
    const ol = document.createElement('ol');
    ol.className = 'ref-list';

    refs.forEach(r => {
      const li = document.createElement('li');
      li.id = 'ref-' + r.num;
      li.className = 'ref-entry';
      li.innerHTML =
        '<span class="ref-num">[' + r.num + ']</span> ' +
        '<span class="ref-body">' + fmtBib(r.key) + '</span>';
      ol.appendChild(li);
    });

    while(host.firstChild) host.removeChild(host.firstChild);
    host.appendChild(ol);
  }

  /* ─── Click an inline cite → jump to the References slide ──────── */
  function wireCiteNavigation(){
    const host = document.getElementById('references-list-host');
    if(!host) return;
    const refSection = host.closest('section');

    document.querySelectorAll('.reveal .cite-link').forEach(link => {
      link.addEventListener('click', (ev) => {
        ev.preventDefault();
        if(window.Reveal && refSection && typeof Reveal.getIndices === 'function'){
          const idx = Reveal.getIndices(refSection);
          Reveal.slide(idx.h, idx.v);
        }
      });
    });
  }

  /* ─── Run after reveal.js is ready, then sync slide count ─────── */
  function run(){
    try {
      process();
      if(window.Reveal && typeof Reveal.sync === 'function'){
        Reveal.sync();
      }
    } catch(e) {
      console.error('[slide-citations]', e);
    }
  }

  if(window.Reveal && Reveal.on){
    Reveal.on('ready', run);
  } else if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();

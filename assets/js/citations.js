/* ───────────────────────────────────────────────────────────────────
   citations.js — resolves <span class="cite" data-key="…"></span>
   into numbered Tufte-style sidenotes.

   Usage in a post:
     ... has been studied <span class="cite" data-key="kidger_thesis"></span>.
     ... combined approaches <span class="cite" data-key="adjointdeis,marion2024implicit"></span>.

   The script walks the post body, replaces each cite span with a
   numbered superscript + adjacent (hidden checkbox) + sidenote element.
   Footnotes use <span class="footnote">...</span> the same way.

   Citation data is keyed by bib key. Each entry: { authors, title, year,
   venue?, url?, doi? }. Only the keys used across the 4 posts are
   included.
   ─────────────────────────────────────────────────────────────────── */
(function(){
  const C = {
    // ─── diffusion / generative modeling ───
    'ddpm':              {authors:'Ho, J., Jain, A., Abbeel, P.', title:'Denoising Diffusion Probabilistic Models', venue:'NeurIPS', year:2020, url:'https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html'},
    'song2021scorebased':{authors:'Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B.', title:'Score-Based Generative Modeling through Stochastic Differential Equations', venue:'ICLR', year:2021, url:'https://openreview.net/forum?id=PxTIG12RRHS'},
    'song2021denoising': {authors:'Song, J., Meng, C., Ermon, S.', title:'Denoising Diffusion Implicit Models', venue:'ICLR', year:2021, url:'https://openreview.net/forum?id=St1giarCHLP'},
    'progressive_distillation':{authors:'Salimans, T., Ho, J.', title:'Progressive Distillation for Fast Sampling of Diffusion Models', venue:'ICLR', year:2022, url:'https://openreview.net/forum?id=TIdIXIpzhoI'},
    'kingma2021variational':{authors:'Kingma, D., Salimans, T., Poole, B., Ho, J.', title:'Variational Diffusion Models', venue:'NeurIPS', year:2021},
    'ldm':               {authors:'Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.', title:'High-Resolution Image Synthesis with Latent Diffusion Models', venue:'CVPR', year:2022},
    '2022arXiv220406125R':{authors:'Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.', title:'Hierarchical Text-Conditional Image Generation with CLIP Latents', venue:'arXiv:2204.06125', year:2022, url:'https://arxiv.org/abs/2204.06125'},
    'NEURIPS2022_ec795aea':{authors:'Saharia, C., Chan, W., Saxena, S., et al.', title:'Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)', venue:'NeurIPS', year:2022},
    'Peebles2022DiT':    {authors:'Peebles, W., Xie, S.', title:'Scalable Diffusion Models with Transformers', venue:'ICCV', year:2023, url:'https://arxiv.org/abs/2212.09748'},
    'diffae':            {authors:'Preechakul, K., Chatthee, N., Wizadwongsa, S., Suwajanakorn, S.', title:'Diffusion Autoencoders: Toward a Meaningful and Decodable Representation', venue:'CVPR', year:2022},
    'diff_beat_gan':     {authors:'Dhariwal, P., Nichol, A.', title:'Diffusion Models Beat GANs on Image Synthesis', venue:'NeurIPS', year:2021},
    'ho2021classifierfree':{authors:'Ho, J., Salimans, T.', title:'Classifier-Free Diffusion Guidance', venue:'NeurIPS Workshop on Deep Generative Models', year:2021, url:'https://openreview.net/forum?id=qw8AKxfYbI'},
    'Karras2022edm':     {authors:'Karras, T., Aittala, M., Aila, T., Laine, S.', title:'Elucidating the Design Space of Diffusion-Based Generative Models', venue:'NeurIPS', year:2022},

    // ─── solvers / numerics ───
    'lu2022dpmsolver':   {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps', venue:'NeurIPS', year:2022, url:'https://openreview.net/forum?id=2uAaGwlP_V'},
    'dpm_solver':        {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps', venue:'NeurIPS', year:2022},
    'lu2023dpmsolver':   {authors:'Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.', title:'DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models', venue:'arXiv:2211.01095', year:2023, url:'https://arxiv.org/abs/2211.01095'},
    'deis_georgiatech':  {authors:'Zhang, Q., Chen, Y.', title:'Fast Sampling of Diffusion Models with Exponential Integrator', venue:'ICLR', year:2023},
    'exponential_integrators':{authors:'Hochbruck, M., Ostermann, A.', title:'Exponential Integrators', venue:'Acta Numerica 19', year:2010},
    'exponential_integrators_sde':{authors:'Gonzalez, M., Fernandez Pinto, N., Tran, T., Gherbi, E., Hajri, H., Masmoudi, N.', title:'SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models', venue:'NeurIPS', year:2023},
    'atkinson2011numerical':{authors:'Atkinson, K., Han, W., Stewart, D. E.', title:'Numerical Solution of Ordinary Differential Equations', venue:'Wiley', year:2011},
    'butcher2016numerical':{authors:'Butcher, J. C.', title:'Numerical Methods for Ordinary Differential Equations', venue:'Wiley', year:2016},

    // ─── neural ODE / SDE / adjoint ───
    'neural_ode':        {authors:'Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D.', title:'Neural Ordinary Differential Equations', venue:'NeurIPS', year:2018},
    'kidger_thesis':     {authors:'Kidger, P.', title:'On Neural Differential Equations', venue:'Ph.D. thesis, Oxford University', year:2022},
    'adjoint_sensitivity_method':{authors:'Pontryagin, L. S., Boltyanskii, V. G., Gamkrelidze, R. V., Mishechenko, E. F.', title:'The Mathematical Theory of Optimal Processes', venue:'ZAMM 43(10-11)', year:1963},
    'adjointsde':        {authors:'Li, X., Wong, T.-K. L., Chen, R. T. Q., Duvenaud, D.', title:'Scalable Gradients for Stochastic Differential Equations', venue:'AISTATS', year:2020},
    'reverse_time_sdes': {authors:'Anderson, B. D. O.', title:'Reverse-Time Diffusion Equation Models', venue:'Stochastic Processes and their Applications 12(3)', year:1982},
    'anderson_diffusion':{authors:'Anderson, B. D. O.', title:'Reverse-Time Diffusion Equation Models', venue:'Stochastic Processes and their Applications 12(3)', year:1982},

    // ─── guidance / optimization ───
    'adjointdeis':       {authors:'Blasingame, Z. W., Liu, C.', title:'AdjointDEIS: Efficient Gradients for Diffusion Models', venue:'NeurIPS', year:2024, url:'https://openreview.net/forum?id=fAlcxvrOEX'},
    'adjoint_deis':      {authors:'Blasingame, Z. W., Liu, C.', title:'AdjointDEIS: Efficient Gradients for Diffusion Models', venue:'arXiv:2405.15020', year:2024, url:'https://arxiv.org/abs/2405.15020'},
    'pan2024adjointdpm': {authors:'Pan, J., Liew, J. H., Tan, V., Feng, J., Yan, H.', title:'AdjointDPM: Adjoint Sensitivity Method for Gradient Backpropagation of Diffusion Probabilistic Models', venue:'ICLR', year:2024, url:'https://openreview.net/forum?id=y33lDRBgWI'},
    'marion2024implicit':{authors:'Marion, P., Korba, A., Bartlett, P., et al.', title:'Implicit Diffusion: Efficient Optimization through Stochastic Sampling', venue:'arXiv:2402.05468', year:2024, url:'https://arxiv.org/abs/2402.05468'},
    'doodl':             {authors:'Wallace, B., Gokul, A., Ermon, S., Naik, N.', title:'End-to-End Diffusion Latent Optimization Improves Classifier Guidance', venue:'arXiv:2303.13703', year:2023, url:'https://arxiv.org/abs/2303.13703'},
    'yu2023freedom':     {authors:'Yu, J., Wang, Y., Zhao, C., Ghanem, B., Zhang, J.', title:'FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model', venue:'ICCV', year:2023},
    'liu2023flowgrad':   {authors:'Liu, X., Wu, L., Zhang, S., Gong, C., Ping, W., Liu, Q.', title:'FlowGrad: Controlling the Output of Generative ODEs with Gradients', venue:'CVPR', year:2023},
    'universal_guidance':{authors:'Bansal, A., Chu, H.-M., Schwarzschild, A., et al.', title:'Universal Guidance for Diffusion Models', venue:'arXiv:2302.07121', year:2023, url:'https://arxiv.org/abs/2302.07121'},
    'wallace2023edict':  {authors:'Wallace, B., Gokul, A., Naik, N.', title:'EDICT: Exact Diffusion Inversion via Coupled Transformations', venue:'CVPR', year:2023},

    // ─── face morphing / FR ───
    'Ferrara2016':       {authors:'Ferrara, M., Franco, A., Maltoni, D.', title:'On the Effects of Image Alterations on Face Recognition Accuracy', venue:'Face Recognition Across the Imaging Spectrum, Springer', year:2016},
    'morphed_first':     {authors:'Raghavendra, R., Raja, K. B., Busch, C.', title:'Detecting Morphed Face Images', venue:'IEEE BTAS', year:2016},
    'frll':              {authors:'DeBruine, L., Jones, B.', title:'Face Research Lab London Set', venue:'figshare 5047666', year:2017, url:'https://figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666'},
    'morgan':            {authors:'Damer, N., Saladié, A. M., Braun, A., Kuijper, A.', title:'MorGAN: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Generative Adversarial Network', venue:'IEEE BTAS', year:2018},
    'mipgan':            {authors:'Zhang, H., Venkatesh, S., Ramachandra, R., Raja, K., Damer, N., Busch, C.', title:'MIPGAN — Generating Strong and High Quality Morphing Attacks Using Identity Prior Driven GAN', venue:'IEEE T-BIOM 3(3)', year:2021},
    'can_gan_beat_landmark':{authors:'Venkatesh, S., Zhang, H., Ramachandra, R., Raja, K., Damer, N., Busch, C.', title:'Can GAN Generated Morphs Threaten Face Recognition Systems Equally as Landmark Based Morphs? — Vulnerability and Detection', venue:'IWBF', year:2020},
    'multe-scale-block-fusion':{authors:'Scherhag, U., Kunze, J., Rathgeb, C., Busch, C.', title:'Face Morph Detection for Unknown Morphing Algorithms and Image Sources: a Multi-Scale Block Local Binary Pattern Fusion Approach', venue:'IET Biometrics 9', year:2020},
    'sebastien_gan_threaten':{authors:'Sarkar, E., Korshunov, P., Colbois, L., Marcel, S.', title:'Are GAN-based morphs threatening face recognition?', venue:'ICASSP', year:2022},
    'mmpmr':             {authors:'Scherhag, U., Nautsch, A., Rathgeb, C., et al.', title:'Biometric Systems under Morphing Attacks: Assessment of Morphing Techniques and Vulnerability Reporting', venue:'BIOSIG', year:2017},
    'morph_pipe':        {authors:'Zhang, H., Ramachandra, R., Raja, K., Busch, C.', title:'Morph-PIPE: Plugging in Identity Prior to Enhance Face Morphing Attack Based on Diffusion Model', venue:'NISK', year:2023},
    'syn-mad22':         {authors:'Huber, M., Boutros, F., Luu, A. T., et al.', title:'SYN-MAD 2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data', venue:'IJCB', year:2022},
    'blasingame_dim':    {authors:'Blasingame, Z. W., Liu, C.', title:'Leveraging Diffusion for Strong and High Quality Face Morphing Attacks', venue:'IEEE T-BIOM 6(1)', year:2024, doi:'10.1109/TBIOM.2024.3349857'},
    'fast_dim':          {authors:'Blasingame, Z. W., Liu, C.', title:'Fast-DiM: Towards Fast Diffusion Morphs', venue:'IEEE Security &amp; Privacy 22(4)', year:2024, doi:'10.1109/MSEC.2024.3410112'},
    'greedy_dim':        {authors:'Blasingame, Z. W., Liu, C.', title:'Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs', venue:'IJCB', year:2024},
    'template_inversion_sebastien':{authors:'Otroshi Shahreza, H., Marcel, S.', title:'Face Reconstruction from Facial Templates by Learning Latent Space of a Generator Network', venue:'NeurIPS', year:2023},

    // ─── GAN / inversion / encoders ───
    'gan':               {authors:'Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al.', title:'Generative Adversarial Nets', venue:'NeurIPS', year:2014},
    'bigan':             {authors:'Donahue, J., Krähenbühl, P., Darrell, T.', title:'Adversarial Feature Learning', venue:'arXiv:1605.09782', year:2016, url:'https://arxiv.org/abs/1605.09782'},
    'aae':               {authors:'Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., Frey, B.', title:'Adversarial Autoencoders', venue:'ICLR Workshop', year:2016, url:'https://arxiv.org/abs/1511.05644'},
    'alae':              {authors:'Pidhorskyi, S., Adjeroh, D., Doretto, G.', title:'Adversarial Latent Autoencoders', venue:'arXiv:2004.04467', year:2020, url:'https://arxiv.org/abs/2004.04467'},
    'e4e':               {authors:'Tov, O., Alaluf, Y., Nitzan, Y., Patashnik, O., Cohen-Or, D.', title:'Designing an Encoder for StyleGAN Image Manipulation', venue:'ACM TOG 40(4)', year:2021},
    'gan_opt_invert':    {authors:'Creswell, A., Bharath, A. A.', title:'Inverting the Generator of a Generative Adversarial Network', venue:'IEEE TNNLS 30(7)', year:2019},
    'Abdal_2019_ICCV':   {authors:'Abdal, R., Qin, Y., Wonka, P.', title:'Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?', venue:'ICCV', year:2019},
    'stylegan':          {authors:'Karras, T., Laine, S., Aila, T.', title:'A Style-Based Generator Architecture for Generative Adversarial Networks', venue:'CVPR', year:2019},
    'unet':              {authors:'Ronneberger, O., Fischer, P., Brox, T.', title:'U-Net: Convolutional Networks for Biomedical Image Segmentation', venue:'MICCAI', year:2015},

    // ─── theory / measure ───
    'sarkka2019applied': {authors:'Särkkä, S., Solin, A.', title:'Applied Stochastic Differential Equations', venue:'Cambridge University Press', year:2019, url:'https://users.aalto.fi/~ssarkka/pub/sde_book.pdf'},
    'blasingame2025thesis':{authors:'Blasingame, Z. W.', title:'On Guided and Reversible Solvers for Neural Differential Equations', venue:'Ph.D. thesis, Clarkson University', year:2025},

    // ─── face morphing & adversarial ───
    'chen2023diffusion': {authors:'Chen, J., Chen, H., Chen, K., Zhang, Y., Zou, Z., Shi, Z.', title:'Diffusion Models for Imperceptible and Transferable Adversarial Attack', venue:'arXiv:2305.08192', year:2023, url:'https://arxiv.org/abs/2305.08192'},
  };

  /* ─── Sidenote (footnote) markup ─────────────────────────────── */
  let snCounter = 0;
  function makeNote(html){
    snCounter += 1;
    const id = 'sn-' + snCounter;
    const sup  = '<label class="margin-toggle-label" for="' + id + '"><sup class="sidenote-number"></sup></label>';
    const cb   = '<input type="checkbox" id="' + id + '" class="margin-toggle">';
    const note = '<span class="sidenote">' + html + '</span>';
    return sup + cb + note;
  }

  /* ─── Author-year formatting helpers ─────────────────────────── */
  // Extract surnames from an "authors" field like
  // "Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D."
  // Heuristic: split on commas; surnames are tokens NOT matching
  // an "all initials" pattern (e.g. "R. T. Q.").
  function parseSurnames(s){
    if(!s) return [];
    const parts = s.split(',').map(t => t.trim()).filter(Boolean);
    const out = [];
    for(const p of parts){
      // "et al." marker
      if(/^et\s+al\.?$/i.test(p)) { out.push('et al.'); continue; }
      // initials block: only capital letters, dots, spaces, hyphens
      if(/^([A-Z]\.?\s*\-?\s*)+$/.test(p)) continue;
      out.push(p);
    }
    return out;
  }

  function shortAuthors(s){
    const sur = parseSurnames(s);
    if(!sur.length) return '';
    // "et al." literal present in source
    if(sur.length === 1) return sur[0];
    if(sur[1] === 'et al.') return sur[0] + ' et al.';
    if(sur.length === 2)   return sur[0] + ' and ' + sur[1];
    return sur[0] + ' et al.';
  }

  /* ─── Bibliography entry formatting ──────────────────────────── */
  function fmtBibEntry(key){
    const c = C[key];
    if(!c) return '<em>[missing: ' + key + ']</em>';
    let s = '';
    if(c.authors) s += c.authors;
    if(c.year)    s += ' (' + c.year + ').';
    if(c.title){
      s += ' <em>' + c.title + '</em>';
      if(!/[.!?]$/.test(c.title)) s += '.';
    }
    if(c.venue) s += ' ' + c.venue + '.';
    if(c.url)       s += ' <a href="' + c.url + '">link</a>';
    else if(c.doi)  s += ' <a href="https://doi.org/' + c.doi + '">doi</a>';
    return s;
  }

  /* ─── Main pass ──────────────────────────────────────────────── */
  function process(){
    // Track each key → list of in-text anchor ids (for back-refs)
    const cited = Object.create(null);   // key -> [{id, idx}]
    let siteCounter = 0;

    // Walk every <span class="cite"> in document order
    document.querySelectorAll('.cite').forEach(el => {
      const keys = (el.getAttribute('data-key') || '').split(',').map(s => s.trim()).filter(Boolean);
      if(!keys.length){ el.remove(); return; }
      const style = (el.getAttribute('data-style') || 'parenthetical').toLowerCase();

      siteCounter += 1;
      const siteId = 'cite-' + siteCounter;

      // Render the inline citation
      const parts = keys.map(k => {
        const c = C[k];
        if(!c){
          return '<span class="cite-missing">[missing: ' + k + ']</span>';
        }
        // Record this site for the bibliography back-reference
        (cited[k] = cited[k] || []).push({id: siteId, idx: siteCounter});
        const auth = shortAuthors(c.authors);
        const year = c.year || 'n.d.';
        const href = '#bib-' + cssEscape(k);
        if(style === 'year' && keys.length === 1){
          return '<a href="' + href + '" class="cite-link">(' + year + ')</a>';
        }
        if(style === 'narrative' && keys.length === 1){
          return '<a href="' + href + '" class="cite-link">' + auth + '</a>\u202F(' + year + ')';
        }
        return '<a href="' + href + '" class="cite-link">' + auth + ', ' + year + '</a>';
      });

      let inner;
      if((style === 'narrative' || style === 'year') && keys.length === 1){
        inner = parts[0];
      } else {
        inner = '(' + parts.join('; ') + ')';
      }
      const html = '<span class="cite-inline" id="' + siteId + '">' + inner + '</span>';

      const tmp = document.createElement('span');
      tmp.innerHTML = html;
      const node = tmp.firstChild;

      // Insert a leading space if the citation hugs a preceding word.
      // (Source markup commonly attaches <span class="cite"> directly
      // to the word it qualifies; in author-year style we want a gap.)
      const prev = el.previousSibling;
      const needSpace = prev && prev.nodeType === Node.TEXT_NODE && !/\s$/.test(prev.nodeValue)
                      || prev && prev.nodeType === Node.ELEMENT_NODE; // tail of <em>, etc.
      el.replaceWith(node);
      if(needSpace){
        node.parentNode.insertBefore(document.createTextNode(' '), node);
      }
    });

    // Footnotes (free-form sidenotes — unchanged behavior)
    document.querySelectorAll('.footnote').forEach(el => {
      const html = el.innerHTML;
      const wrap = document.createElement('span');
      wrap.innerHTML = makeNote(html);
      el.replaceWith(wrap);
    });

    // Margin notes (no number)
    document.querySelectorAll('.margin').forEach(el => {
      el.classList.remove('margin');
      el.classList.add('marginnote');
    });

    // Build References section
    buildBibliography(cited);
  }

  function buildBibliography(cited){
    const keys = Object.keys(cited);
    if(!keys.length) return;

    // Sort alphabetically by first-author surname, then year
    keys.sort((a, b) => {
      const A = (parseSurnames((C[a]||{}).authors)[0] || a).toLowerCase();
      const B = (parseSurnames((C[b]||{}).authors)[0] || b).toLowerCase();
      if(A < B) return -1;
      if(A > B) return  1;
      const yA = (C[a]||{}).year || 0;
      const yB = (C[b]||{}).year || 0;
      return yA - yB;
    });

    const section = document.createElement('section');
    section.id = 'references';
    section.className = 'references';
    section.setAttribute('aria-label', 'References');

    const h2 = document.createElement('h2');
    h2.textContent = 'References';
    section.appendChild(h2);

    const ol = document.createElement('ol');
    ol.className = 'bib-list';

    keys.forEach(k => {
      const li = document.createElement('li');
      li.id = 'bib-' + cssEscape(k);
      li.className = 'bib-entry';

      const body = document.createElement('span');
      body.className = 'bib-body';
      body.innerHTML = fmtBibEntry(k);
      li.appendChild(body);

      // Back-references: [§1, §3, §5]
      const sites = cited[k];
      if(sites && sites.length){
        const back = document.createElement('span');
        back.className = 'bib-backrefs';
        back.appendChild(document.createTextNode(' ['));
        sites.forEach((s, i) => {
          if(i) back.appendChild(document.createTextNode(', '));
          const a = document.createElement('a');
          a.href = '#' + s.id;
          a.className = 'bib-backref';
          a.textContent = '§' + s.idx;
          back.appendChild(a);
        });
        back.appendChild(document.createTextNode(']'));
        li.appendChild(back);
      }

      ol.appendChild(li);
    });
    section.appendChild(ol);

    // Insert after .post-body's closing (i.e., after the article body, before <footer>)
    const article = document.querySelector('article');
    if(article){
      article.appendChild(section);
    } else {
      document.body.appendChild(section);
    }
  }

  /* tiny CSS.escape polyfill for old browsers */
  function cssEscape(s){
    if(window.CSS && CSS.escape) return CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_\-]/g, '_');
  }

  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', process);
  } else {
    process();
  }
})();

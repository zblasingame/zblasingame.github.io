/* bib.js — copy-to-clipboard for publication BibTeX panels.
   Wires up every .bib-copy button: copies the sibling <pre> text and
   flips the button into a "copied" state for ~1.4s. */
(function () {
  function flash(btn) {
    btn.classList.add('copied');
    btn.setAttribute('aria-label', 'Copied!');
    setTimeout(function () {
      btn.classList.remove('copied');
      btn.setAttribute('aria-label', 'Copy BibTeX');
    }, 1400);
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    return ok;
  }

  function onClick(e) {
    var btn = e.currentTarget;
    var panel = btn.closest('.bib-panel');
    if (!panel) return;
    var pre = panel.querySelector('pre');
    if (!pre) return;
    var text = pre.textContent;

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { flash(btn); },
        function () { if (fallbackCopy(text)) flash(btn); }
      );
    } else {
      if (fallbackCopy(text)) flash(btn);
    }
  }

  function init() {
    var btns = document.querySelectorAll('.bib-panel .bib-copy');
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener('click', onClick);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

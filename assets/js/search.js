/* ───────────────────────────────────────────────────────────────────
   Search palette — Ctrl/Cmd+K to open, ↑/↓ to navigate, Enter to go.
   Builds a fixed overlay, fetches assets/js/search-index.json once,
   and ranks matches across title + snippet fields.
   ─────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  // ─── locate the JSON index from any page depth ────────────────────
  // pages live at: /, /subpages/, /writings/<slug>/
  function indexUrl() {
    var p = location.pathname;
    // depth = number of slashes after the leading one, minus the trailing filename
    // simplest: count "/" segments excluding empty + final filename
    var parts = p.split("/").filter(Boolean);
    // if last part has a dot it's the filename; drop it
    if (parts.length && parts[parts.length - 1].indexOf(".") !== -1)
      parts.pop();
    var depth = parts.length;
    // also handle GitHub Pages subdir hosting (e.g. /repo/), where the user
    // site root may be one segment deep. We're zblasingame.github.io (root site),
    // so depth here is the literal subfolder count from /.
    var prefix = depth === 0 ? "" : "../".repeat(depth);
    return prefix + "assets/js/search-index.json";
  }

  // ─── tokenize + score ─────────────────────────────────────────────
  function tokenize(s) {
    return s
      .toLowerCase()
      .split(/[^a-z0-9]+/i)
      .filter(Boolean);
  }

  function score(entry, qTokens, qRaw) {
    var hay =
      " " +
      (entry.title || "").toLowerCase() +
      "  " +
      (entry.snippet || "").toLowerCase() +
      " ";
    var titleHay = " " + (entry.title || "").toLowerCase() + " ";
    var s = 0;
    var allMatch = true;
    for (var i = 0; i < qTokens.length; i++) {
      var t = qTokens[i];
      var inTitle = titleHay.indexOf(t) !== -1;
      var inAny = hay.indexOf(t) !== -1;
      if (!inAny) {
        allMatch = false;
        break;
      }
      if (inTitle) s += 10;
      s += 3;
      // exact word boundary bonus
      if (new RegExp("\\b" + escapeRegExp(t) + "\\b").test(hay)) s += 2;
    }
    if (!allMatch) return 0;
    // bonus: full raw substring in title
    if (qRaw && titleHay.indexOf(qRaw.toLowerCase()) !== -1) s += 15;
    return s;
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function highlight(text, qTokens) {
    if (!text) return "";
    // escape HTML first
    var safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    if (!qTokens.length) return safe;
    var pattern = qTokens
      .map(escapeRegExp)
      .filter(Boolean)
      .sort(function (a, b) {
        return b.length - a.length;
      })
      .join("|");
    if (!pattern) return safe;
    return safe.replace(
      new RegExp("(" + pattern + ")", "ig"),
      "<mark>$1</mark>",
    );
  }

  // ─── build overlay DOM ────────────────────────────────────────────
  var overlay,
    input,
    results,
    list,
    hint,
    indexData = null,
    indexPromise = null;
  var selected = 0;

  function buildOverlay() {
    overlay = document.createElement("div");
    overlay.className = "search-overlay";
    overlay.setAttribute("hidden", "");
    overlay.innerHTML =
      '<div class="search-backdrop" data-close></div>' +
      '<div class="search-modal" role="dialog" aria-label="Site search">' +
      '  <div class="search-bar">' +
      '    <span class="search-icon" aria-hidden="true">⌕</span>' +
      '    <input type="search" class="search-input" placeholder="Search news, publications, writings…" autocomplete="off" spellcheck="false" />' +
      '    <kbd class="search-esc">esc</kbd>' +
      "  </div>" +
      '  <div class="search-results" role="listbox"></div>' +
      '  <div class="search-hint">' +
      "    <span><kbd>↑</kbd><kbd>↓</kbd> navigate</span>" +
      "    <span><kbd>↵</kbd> open</span>" +
      "    <span><kbd>esc</kbd> close</span>" +
      "  </div>" +
      "</div>";
    document.body.appendChild(overlay);
    input = overlay.querySelector(".search-input");
    results = overlay.querySelector(".search-results");
    hint = overlay.querySelector(".search-hint");

    overlay.addEventListener("click", function (e) {
      if (e.target.dataset.close !== undefined) close();
    });
    input.addEventListener("input", onInput);
    input.addEventListener("keydown", onInputKey);
    results.addEventListener("click", onResultClick);
  }

  function loadIndex() {
    if (indexData) return Promise.resolve(indexData);
    if (indexPromise) return indexPromise;
    // preferred path: inline JS shim sets window.__searchIndex (works on file://)
    if (Array.isArray(window.__searchIndex)) {
      indexData = window.__searchIndex;
      return Promise.resolve(indexData);
    }
    // fallback: fetch the JSON (works on http(s)://)
    indexPromise = fetch(indexUrl())
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        indexData = data;
        return data;
      })
      .catch(function (err) {
        results.innerHTML =
          '<div class="search-empty">Could not load search index: ' +
          err.message +
          "</div>";
        return [];
      });
    return indexPromise;
  }

  function open() {
    if (!overlay) buildOverlay();
    overlay.removeAttribute("hidden");
    document.documentElement.classList.add("search-open");
    setTimeout(function () {
      input.focus();
      input.select();
    }, 0);
    loadIndex().then(function () {
      // if there's already a query (re-open), refresh results
      if (input.value) onInput();
    });
  }

  function close() {
    if (!overlay) return;
    overlay.setAttribute("hidden", "");
    document.documentElement.classList.remove("search-open");
  }

  function isOpen() {
    return overlay && !overlay.hasAttribute("hidden");
  }

  function onInput() {
    var q = input.value.trim();
    if (!q) {
      results.innerHTML =
        '<div class="search-empty">Type to search the site…</div>';
      selected = 0;
      return;
    }
    var tokens = tokenize(q);
    var scored = [];
    for (var i = 0; i < indexData.length; i++) {
      var s = score(indexData[i], tokens, q);
      if (s > 0) scored.push({ s: s, e: indexData[i] });
    }
    scored.sort(function (a, b) {
      return b.s - a.s;
    });
    var top = scored.slice(0, 40);
    if (!top.length) {
      results.innerHTML =
        '<div class="search-empty">No matches for &ldquo;' +
        q.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") +
        "&rdquo;.</div>";
      selected = 0;
      return;
    }
    var html = "";
    for (var j = 0; j < top.length; j++) {
      var e = top[j].e;
      html +=
        '<a class="search-item' +
        (j === 0 ? " is-selected" : "") +
        '" href="' +
        resolveHref(e.url) +
        '" data-idx="' +
        j +
        '">' +
        '<div class="search-item-row">' +
        '<span class="search-cat search-cat-' +
        e.category +
        '">' +
        e.category +
        "</span>" +
        '<span class="search-title">' +
        highlight(e.title, tokens) +
        "</span>" +
        (e.date ? '<span class="search-date">' + e.date + "</span>" : "") +
        "</div>" +
        (e.snippet
          ? '<div class="search-snippet">' +
            highlight(truncate(e.snippet, 180), tokens) +
            "</div>"
          : "") +
        "</a>";
    }
    results.innerHTML = html;
    selected = 0;
  }

  function truncate(s, n) {
    if (!s) return "";
    if (s.length <= n) return s;
    return s.slice(0, n - 1).trimEnd() + "…";
  }

  // resolve a stored relative href against current page depth
  function resolveHref(href) {
    if (/^(https?:|mailto:|#)/i.test(href)) return href;
    var p = location.pathname;
    var parts = p.split("/").filter(Boolean);
    if (parts.length && parts[parts.length - 1].indexOf(".") !== -1)
      parts.pop();
    var depth = parts.length;
    var prefix = depth === 0 ? "" : "../".repeat(depth);
    return prefix + href;
  }

  function onInputKey(e) {
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      moveSelection(1);
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      moveSelection(-1);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      var sel = results.querySelector(".search-item.is-selected");
      if (sel) window.location.href = sel.href;
      return;
    }
  }

  function moveSelection(delta) {
    var items = results.querySelectorAll(".search-item");
    if (!items.length) return;
    selected = (selected + delta + items.length) % items.length;
    items.forEach(function (el, i) {
      el.classList.toggle("is-selected", i === selected);
    });
    var sel = items[selected];
    if (sel && sel.scrollIntoView) {
      sel.scrollIntoView({ block: "nearest" });
    }
  }

  function onResultClick(e) {
    // let the anchor navigate naturally; just close the overlay first
    close();
  }

  // ─── global hotkeys ───────────────────────────────────────────────
  document.addEventListener("keydown", function (e) {
    // Ctrl+K / Cmd+K to open from anywhere
    if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
      e.preventDefault();
      isOpen() ? close() : open();
      return;
    }
    // "/" opens search unless typing in an input/textarea/contenteditable
    if (e.key === "/" && !isOpen()) {
      var t = e.target;
      var tag = t && t.tagName ? t.tagName.toLowerCase() : "";
      if (tag === "input" || tag === "textarea" || (t && t.isContentEditable))
        return;
      e.preventDefault();
      open();
    }
  });

  // delegated click handler for any element with [data-search-open]
  document.addEventListener("click", function (e) {
    var t = e.target;
    while (t && t !== document) {
      if (
        t.nodeType === 1 &&
        t.hasAttribute &&
        t.hasAttribute("data-search-open")
      ) {
        e.preventDefault();
        open();
        return;
      }
      t = t.parentNode;
    }
  });

  // expose for any custom triggers (e.g. a nav button)
  window.__siteSearch = { open: open, close: close };
})();

window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
      displayMath: [
        ["$$", "$$"],
        ["\\[", "\\]"]
      ]
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          mjx-container {
            display: inline-flex;
            color: inherit;
            overflow-x: auto;
            overflow-y: hidden;
            max-width: 100%;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};

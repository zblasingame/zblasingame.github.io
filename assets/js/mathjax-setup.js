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
          mjx-math[width="full"] {
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

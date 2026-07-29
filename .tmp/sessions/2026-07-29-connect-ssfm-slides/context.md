# Task Context: Connect SSFM slides to personal site

Session ID: 2026-07-29-connect-ssfm-slides
Created: 2026-07-29
Status: in_progress

## Current Request
Copy the ssfm slides (from /Users/zblasingame/socials/slidedecks/ssfms/) into the
personal site and connect it. User confirmed: separate assets folder for these
html slidedecks.

## Context Files (Standards to Follow)
None — no .opencode/context/ standards exist for this repo. Follow existing
site conventions: plain HTML, relative links, the nav + publications patterns
in index.html.

## Reference Files (Source Material to Look At)
- index.html (publications section lines 229-376; nav lines 48-56)
- /Users/zblasingame/socials/slidedecks/ssfms/slides.html (the deck)
- /Users/zblasingame/socials/slidedecks/ssfms/slide-citations.js
- /Users/zblasingame/socials/slidedecks/ssfms/assets/{logos,figures}

## Components
1. Copy deck into slides/ssfm/ (self-contained, own assets/ folder) — DONE
2. Add "slides" link to SSFM publication entry in index.html .extras,
   matching the Rex and Greed paper patterns.

## Constraints
- Deck uses relative assets/ paths — must stay in a folder with its own assets/.
- Zero path edits needed since copied verbatim into slides/ssfm/.

## Exit Criteria
- [x] Deck + slide-citations.js + assets copied to slides/ssfm/
- [ ] slides link added to SSFM publication in index.html
- [ ] links resolve (relative path correct)
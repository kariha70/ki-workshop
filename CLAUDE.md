# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A German-language beginner workshop ("KI-Workshop: Vom Eisverkäufer zum Sprachmodell") on neural networks, LLMs and semantic search/RAG, published as a static site on GitHub Pages from `docs/`. All content, identifiers, comments and commit messages are in German; keep it that way. Everything is explained through one running example: an ice-cream seller (Eisverkäufer/Eisdiele).

## Commands

All Python files are uv inline scripts (PEP 723); uv installs their dependencies on first run. There are no tests or linters.

```bash
uv run werkzeuge/seite_bauen.py                                  # rebuild all generated HTML pages in docs/
uv run docs/beispiele/01_eisverkaeufer.py                        # chapter 1: numbers + docs/bilder/*.png (~30 s)
uv run docs/beispiele/02_tokenizer.py                            # chapter 2: numbers, docs/bilder/kapitel-2/, docs/app/token-beispiele.js
uv run docs/beispiele/02_tokenizer.py "Eigener Text"             # just tokenize the given text
uv run docs/beispiele/03_semantische_suche.py                    # chapter 3: numbers, docs/bilder/kapitel-3/, docs/app/such-daten.js
uv run docs/beispiele/03_semantische_suche.py "Eigene Frage?"    # just search the handbook
uv run docs/beispiele/04_agenten.py                              # chapter 4: agent runs, docs/bilder/kapitel-4/, docs/app/agenten-daten.js
uv run docs/beispiele/05_nachdenken.py                           # chapter 5: simulations, docs/bilder/kapitel-5/, docs/app/denk-daten.js
uv run docs/beispiele/06_prompts.py                              # chapter 6: prompt token counts, docs/bilder/kapitel-6/, docs/app/prompt-daten.js
npm --prefix werkzeuge/tiktoken install && npm --prefix werkzeuge/tiktoken run bauen   # rebuild docs/app/vendor/tiktoken.js (only when bumping js-tiktoken)
```

Scripts 2, 4, 5 and 6 download tiktoken vocabularies and script 3 downloads a ~470 MB ONNX embedding model on first run (cached afterwards).

## Architecture

There is no build step in CI: `.github/workflows/pages.yml` uploads `docs/` as-is on every push to `main`. **Generated files are committed**, so after editing any source, regenerate locally and commit the outputs too. A quick sanity check: running `seite_bauen.py` on a clean tree must leave `git status` clean.

Three layers, each generated from the one before:

1. **Python scripts** (`docs/beispiele/`) are the single source of truth for every number in the text, every image in `docs/bilder/`, and the lab data files `docs/app/token-beispiele.js` / `docs/app/such-daten.js` (headed "Erzeugt von …, bitte nicht von Hand ändern"); same for `agenten-daten.js`, `denk-daten.js` and `prompt-daten.js`. Scripts 4 and 6 import script 1 (sales data, regression), and script 4 also reads `docs/app/such-daten.js`, so rerun them after changing either. Script 4's language model is a fixed "Drehbuch"; the loop, tools and token counts are real. The sample answers in script 6 are invented but marked as such. When a number in the Markdown changes, it must come from a script run, not be edited by hand.
2. **Markdown** (`docs/0N-*.md`, `glossar.md`, `leitfaden.md`) → **`werkzeuge/seite_bauen.py`** → `docs/kapitel-N.html`, `glossar.html`, `leitfaden.html`, `index.html`. Generated HTML carries a "bitte nicht von Hand ändern" comment; never edit it directly.
   - Page chrome (nav bar `leiste()`, footer, `<head>`) lives in `seite()`. The start page, including its inline SVG illustration, is built in `startseite()`.
   - The `KAPITEL` list holds per-chapter metadata: teaser, topic chips, and the `labore` map {section number → lab hint box}. These hint texts sometimes quote numbers from the scripts (e.g. "629,2"), so keep them in sync.
   - Section numbers come from `## N. Titel` headings; headings "Das Wichtigste in Kürze" and "Quellen" get special CSS classes. Paragraphs before the first `##` become the page intro. Reading time counts only text outside `<details class="vertiefung">`.
   - Links between chapter `.md` files are rewritten to the `.html` targets automatically.
3. **Labs** (`docs/app/labor-N.html`) are hand-written standalone pages with inline CSS/JS. They are *not* generated, so they duplicate the nav bar, theme bootstrap `<script>` and footer from `seite()`. Changes to navigation or page chrome must be mirrored by hand in all labs. Labs also link to chapter section anchors (slugs of the `##` headings, e.g. `kapitel-1.html#5-lernen-heißt-schritt-für-schritt-bergab`), so renaming a heading breaks those links. Lab 2/3 use the bundled `vendor/tiktoken.js`; Lab 3 loads Transformers.js from jsDelivr.

Shared assets: `assets/leiste.css` (nav bar, also used by the labs), `assets/site.css` (generated pages), `assets/thema.js` (light/dark toggle, stored under localStorage key `ki-workshop-thema`), `assets/site.js` (reading progress, active TOC entry, image zoom, open all `<details>` on print).

## Markdown conventions in the chapters

Markdown is processed with `md_in_html`, so HTML blocks need `markdown="1"`:

- `<div class="frage" markdown="1">` with `**Kurz überlegen:** …`: reflection question that opens each larger idea, before the technical term is introduced.
- `<details class="vertiefung" markdown="1">` with summary "Für Neugierige": formulas, code and background. The main text must stay free of formulas.
- `> **Merke:** …`: one key takeaway at the end of every section.
- Each chapter ends with `## Das Wichtigste in Kürze`, then `## Teste dich` with `<details class="quiz" markdown="1">` items; chapters 2 to 6 also have `## Quellen`.

## Content rules (from the README)

- All example data is invented; every number in the text comes from the scripts. Illustrative values (e.g. in schematic graphics) are explicitly marked as made up.
- New technical terms also go into `docs/glossar.md`.
- Every explained concept gets its English original term at first mention, in the format of chapters 1 and 2: `(englisch Gradient Descent)`, no italics, English nouns capitalized. Glossary entries carry it too.
- The most recent commit replaced stock phrases and filler wording throughout; write plainly and avoid filler.
- If chapters, labs or screenshots change substantially, update `README.md` (and `docs/bilder/startseite.png`, `labor-N.png` screenshots) accordingly.

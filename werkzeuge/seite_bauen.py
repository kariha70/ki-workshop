# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown"]
# ///
"""Baut die Webseite des Workshops aus den Kapiteln in docs/.

Start: uv run werkzeuge/seite_bauen.py

Erzeugt docs/index.html (Startseite), docs/kapitel-1.html bis docs/kapitel-3.html sowie
das Glossar und den Leitfaden für Moderierende aus den Markdown-Dateien. Nach jeder
Änderung an einem Kapitel neu ausführen.
Die Labore in docs/app/ sind eigenständige Seiten und werden hier nur verlinkt.
"""

import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

DOCS = Path(__file__).resolve().parent.parent / "docs"

KAPITEL = [
    {
        "nr": 1,
        "quelle": "01-Neuronale-Netze-Grundlagen.md",
        "ziel": "kapitel-1.html",
        "labor": "app/labor-1.html",
        "bild": "bilder/11-zwei-knicke.png",
        "teaser": "Wie lernt eine Maschine aus Beispielen? Ein Eisverkäufer, eine Gerade und ein kleines Netz zeigen das Grundprinzip hinter jeder heutigen KI.",
        "themen": ["Aus Beispielen lernen", "Fehler messen", "Schritt für Schritt bergab", "Auswendiglernen erkennen", "Neuron", "Neuronales Netz"],
        # Nach diesen Abschnitten erscheint ein Hinweis auf das passende Labor
        "labore": {
            4: ("app/labor-1.html#gerade", "Gerade und Fehler", "Such mit zwei Reglern die beste Gerade. Wie nah kommst du an einen Fehler von 629,2?"),
            5: ("app/labor-1.html#bergab", "Fehler minimieren", "Geh Schritt für Schritt bergab und probier aus, was bei einer zu großen Lernrate passiert."),
            9: ("app/labor-1.html#neuron", "Neuron mit Knick", "Stell −10 °C ein und schalte den Knick ein und aus. Was passiert an den Hochsommertagen?"),
            10: ("app/labor-1.html#netz", "Kleines Netz", "Lass das Netz mehrmals mit Zufallswerten lernen. Findet es die Kurve jedes Mal?"),
        },
    },
    {
        "nr": 2,
        "quelle": "02-Tokenizer-und-Sprachmodelle.md",
        "ziel": "kapitel-2.html",
        "labor": "app/labor-2.html",
        "bild": "bilder/kapitel-2/08-transformer.png",
        "teaser": "Wie schreibt ein Computer Texte? Was in ChatGPT und Co. steckt, warum es so flüssig formuliert und warum man trotzdem nachprüfen sollte.",
        "themen": ["Das nächste Wort", "Tokens", "Bedeutung als Zahlen", "Attention", "Transformer", "Temperatur", "Halluzinationen"],
        "labore": {
            3: ("app/labor-2.html#tokens", "Tokens", "Tipp deinen Namen oder einen eigenen Satz ein und sieh zu, wie der Tokenizer ihn zerlegt."),
            8: ("app/labor-2.html#temperatur", "Temperatur", "Verstell die Temperatur und lass 20-mal auslosen. Wann landet die Schokolade im Regen?"),
        },
    },
    {
        "nr": 3,
        "quelle": "03-Semantische-Suche-und-RAG.md",
        "ziel": "kapitel-3.html",
        "labor": "app/labor-3.html",
        "bild": "bilder/kapitel-3/04-suche.png",
        "teaser": "Wie beantwortet eine KI Fragen zu eigenen Unterlagen? Suchen nach Bedeutung statt nach Wörtern, und Nachschlagen statt Raten.",
        "themen": ["Suchen nach Bedeutung", "Ähnlichkeit messen", "Embedding-Modell", "RAG", "Grenzen", "Praxistipps"],
        "labore": {
            3: ("app/labor-3.html#kosinus", "Kosinus von Hand", "Zieh zwei Pfeile und beobachte, wie sich Kosinus und Abstand verändern."),
            4: ("app/labor-3.html#vergleich", "Sätze vergleichen", "Vergleich einen Satz mit seiner Übersetzung, oder „Eis mit Nüssen“ mit „Eis ohne Nüsse“."),
            5: ("app/labor-3.html#suche", "Suche im Handbuch", "Stell eigene Fragen und vergleich semantische Suche und Stichwortsuche nebeneinander."),
            6: ("app/labor-3.html#rag", "Prompt für RAG", "Stell den Prompt zusammen und gib ihn einem Chat-Assistenten. Nennt er die Quelle?"),
        },
    },
]

# Seiten neben den Kapiteln, ohne Labor und ohne Blättern
ZUSATZ = [
    {"quelle": "glossar.md", "ziel": "glossar.html", "augenbraue": "Zum Nachschlagen",
     "beschreibung": "Alle Fachbegriffe aus dem KI-Workshop in Alltagssprache erklärt, mit Verweis auf das passende Kapitel."},
    {"quelle": "leitfaden.md", "ziel": "leitfaden.html", "augenbraue": "Für Moderierende",
     "beschreibung": "Ablauf, Methoden, Fragen an die Gruppe und Technik-Checkliste für den KI-Workshop vor Ort."},
]

LABORE = [
    ("app/labor-1.html", "Labor 1: Wie eine Maschine lernt", "Eine Gerade mit Reglern verschieben, Schritt für Schritt bergab gehen, den Knick einschalten und ein kleines Netz lernen lassen.", "bilder/labor-1.png"),
    ("app/labor-2.html", "Labor 2: Tokens und Temperatur", "Eigenen Text in Tokens zerlegen, Deutsch und Englisch vergleichen und mit der Temperatur das nächste Wort auslosen.", "bilder/labor-2.png"),
    ("app/labor-3.html", "Labor 3: Suchen nach Bedeutung", "Ähnlichkeit von Hand ausprobieren, eigene Sätze vergleichen, im Handbuch der Eisdiele suchen und einen Prompt für einen Chat-Assistenten bauen.", "bilder/labor-3.png"),
]

LOGO = (DOCS / "assets" / "logo.svg").read_text().strip()

SYMBOL_LABOR = '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h10M4 17h6M18 7h2M14 17h6"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="17" r="2"/></svg>'


def leiste(aktiv):
    eintraege = [("kapitel-1.html", "Kapitel 1"), ("kapitel-2.html", "Kapitel 2"), ("kapitel-3.html", "Kapitel 3"), None,
                 ("app/labor-1.html", "Labor 1"), ("app/labor-2.html", "Labor 2"), ("app/labor-3.html", "Labor 3"), None,
                 ("glossar.html", "Glossar")]
    links = ""
    for eintrag in eintraege:
        if eintrag is None:
            links += '<span class="trenner"></span>'
        else:
            ziel, name = eintrag
            markierung = ' aria-current="page"' if ziel == aktiv else ""
            links += f'<a href="{ziel}"{markierung}>{name}</a>'

    return f"""<header class="leiste"><div class="leiste-innen">
  <a class="marke" href="index.html" aria-label="KI-Workshop, zur Startseite">{LOGO}<span>KI-Workshop</span></a>
  <nav aria-label="Workshop">{links}</nav>
  <button class="thema-knopf" type="button" aria-label="Design wechseln"><svg class="mond" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg><svg class="sonne" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
</div></header>"""


def seite(titel, beschreibung, aktiv, inhalt):
    return f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>try{{var t=localStorage.getItem("ki-workshop-thema");if(t==="dark"||t==="light")document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<title>{html.escape(titel)}</title>
<meta name="description" content="{html.escape(beschreibung)}">
<link rel="icon" href="assets/logo.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/leiste.css">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<!-- Erzeugt von werkzeuge/seite_bauen.py, bitte nicht von Hand ändern. -->
<div class="fortschritt" aria-hidden="true"><div></div></div>
{leiste(aktiv)}
<main>
{inhalt}
</main>
<footer class="fuss"><div class="huelle">
  <span>KI-Workshop: vom Eisverkäufer zum Sprachmodell. Alle Daten im Beispiel sind erfunden.<br>Texte und Grafiken: <a href="https://creativecommons.org/licenses/by/4.0/deed.de">CC BY 4.0</a> · Code: <a href="https://github.com/kariha70/ki-workshop/blob/main/LICENSE">MIT</a> · <a href="https://github.com/kariha70/ki-workshop">Quellcode auf GitHub</a></span>
  <span><a href="index.html">Startseite</a> · <a href="kapitel-1.html">Kapitel 1</a> · <a href="kapitel-2.html">Kapitel 2</a> · <a href="kapitel-3.html">Kapitel 3</a> · <a href="app/labor-1.html">Labor 1</a> · <a href="app/labor-2.html">Labor 2</a> · <a href="app/labor-3.html">Labor 3</a> · <a href="glossar.html">Glossar</a> · <a href="leitfaden.html">Leitfaden</a></span>
</div></footer>
<script src="assets/thema.js"></script>
<script src="assets/site.js"></script>
</body>
</html>
"""


def kapitel_lesen(k):
    quelle = (DOCS / k["quelle"]).read_text(encoding="utf-8")
    titel, rumpf = quelle.split("\n", 1)
    # Die Lesezeit gilt für den Hauptweg, die aufklappbaren Vertiefungen zählen nicht mit
    hauptweg = re.sub(r'<details class="vertiefung".*?</details>', "", quelle, flags=re.S)
    worte = len(re.findall(r"\w+", hauptweg))
    return titel.removeprefix("# ").strip(), rumpf, round(worte / 200 / 5) * 5 or 5


def umwandeln(k, rumpf):
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html"],
        extension_configs={"toc": {"slugify": slugify_unicode, "toc_depth": 2}},
    )
    inhalt = md.convert(rumpf)

    # Verweise zwischen den Kapiteln zeigen auf die erzeugten Seiten
    for andere in KAPITEL:
        inhalt = inhalt.replace(f'href="{andere["quelle"]}', f'href="{andere["ziel"]}')
    # Kein Zeilenumbruch zwischen Zahl und Einheit
    inhalt = re.sub(r"(\d) (%|°C)", "\\1\u00a0\\2", inhalt)
    # Tabellen scrollbar machen, Bilder als Abbildungen
    inhalt = inhalt.replace("<table>", '<div class="tabelle"><table>').replace("</table>", "</table></div>")
    inhalt = re.sub(r"<p>(<img [^>]*?)\s*/?></p>", r'<figure>\1 loading="lazy"></figure>', inhalt)

    # Einleitung: die Absätze vor dem ersten Abschnitt kommen in den Kopf der Seite
    vorne, _, rest = inhalt.partition("<h2")
    rest = "<h2" + rest
    absaetze = re.match(r"(\s*<p>.*?</p>)+", vorne, re.S)
    einleitung = absaetze.group(0) if absaetze else ""
    vorne = vorne[len(einleitung):]

    abschnitte = []
    for teil in re.split(r"(?=<h2 )", rest):
        if not teil.strip():
            continue
        kopf = re.match(r'<h2 id="([^"]+)">(.*?)</h2>', teil, re.S)
        anker, ueberschrift = kopf.group(1), kopf.group(2)
        teil = teil.replace(f' id="{anker}"', "", 1)
        nummer = re.match(r"(\d+)\.", ueberschrift)
        if nummer and int(nummer.group(1)) in k.get("labore", {}):
            ziel, name, text = k["labore"][int(nummer.group(1))]
            teil += f"""<aside class="labor-hinweis">
  <span class="symbol">{SYMBOL_LABOR}</span>
  <div><strong>Ausprobieren: {name}</strong><span>{text}</span></div>
  <a class="knopf primaer" href="{ziel}">Zum Labor</a>
</aside>"""
        klasse = {"Das Wichtigste in Kürze": "kurz", "Quellen": "quellen"}.get(ueberschrift, "")
        klassen = f' class="{klasse}"' if klasse else ""
        abschnitte.append((anker, ueberschrift, f'<section id="{anker}"{klassen}>{teil}</section>'))
    return einleitung, vorne, abschnitte


def kapitelseite(k):
    titel, rumpf, minuten = kapitel_lesen(k)
    einleitung, vorne, abschnitte = umwandeln(k, rumpf)
    verzeichnis = "".join(f'<li><a href="#{anker}">{ueberschrift}</a></li>' for anker, ueberschrift, _ in abschnitte)
    index = KAPITEL.index(k)
    blaettern = ""
    if index > 0:
        vorher = KAPITEL[index - 1]
        blaettern += f'<a href="{vorher["ziel"]}"><small>← Kapitel {vorher["nr"]}</small>{kapitel_lesen(vorher)[0]}</a>'
    if index < len(KAPITEL) - 1:
        weiter = KAPITEL[index + 1]
        blaettern += f'<a class="weiter" href="{weiter["ziel"]}"><small>Kapitel {weiter["nr"]} →</small>{kapitel_lesen(weiter)[0]}</a>'

    inhalt = f"""<div class="kapitel-kopf"><div class="huelle">
  <p class="augenbraue">Kapitel {k["nr"]} · etwa {minuten} Minuten Lesezeit</p>
  <h1>{titel}</h1>
  <div class="einleitung">{einleitung}</div>
  <div class="aktionen">
    <a class="knopf primaer" href="#{abschnitte[0][0]}">Lesen</a>
    <a class="knopf" href="{k["labor"]}">Im Labor ausprobieren</a>
  </div>
</div></div>
<div class="huelle raster">
  <aside class="verzeichnis" aria-label="Inhalt">
    <details open><summary>Inhalt</summary><ol>{verzeichnis}</ol></details>
    <a class="zum-labor" href="{k["labor"]}">Labor {k["nr"]}<small>Alles aus diesem Kapitel zum Ausprobieren</small></a>
  </aside>
  <article class="text">
    {vorne}
    {"".join(a for _, _, a in abschnitte)}
    <nav class="blaettern" aria-label="Weitere Kapitel">{blaettern}</nav>
  </article>
</div>"""
    (DOCS / k["ziel"]).write_text(seite(f"Kapitel {k['nr']}: {titel}", k["teaser"], k["ziel"], inhalt), encoding="utf-8")
    return titel, minuten


def zusatzseite(z):
    titel, rumpf, _ = kapitel_lesen(z)
    einleitung, vorne, abschnitte = umwandeln(z, rumpf)
    verzeichnis = "".join(f'<li><a href="#{anker}">{ueberschrift}</a></li>' for anker, ueberschrift, _ in abschnitte)
    inhalt = f"""<div class="kapitel-kopf"><div class="huelle">
  <p class="augenbraue">{z["augenbraue"]}</p>
  <h1>{titel}</h1>
  <div class="einleitung">{einleitung}</div>
</div></div>
<div class="huelle raster">
  <aside class="verzeichnis" aria-label="Inhalt">
    <details open><summary>Inhalt</summary><ol>{verzeichnis}</ol></details>
  </aside>
  <article class="text">
    {vorne}
    {"".join(a for _, _, a in abschnitte)}
  </article>
</div>"""
    (DOCS / z["ziel"]).write_text(seite(titel, z["beschreibung"], z["ziel"], inhalt), encoding="utf-8")
    return titel


ILLUSTRATION = """<svg viewBox="0 0 320 360" role="img" aria-label="Eine Eiswaffel, deren Kugeln als kleines neuronales Netz verbunden sind">
  <defs><linearGradient id="waffel" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#f2b27a"/><stop offset="1" stop-color="#e08a4a"/></linearGradient></defs>
  <path d="M92 196 H228 L160 346 Z" fill="url(#waffel)"/>
  <path d="M110 196 L178 306 M146 196 L196 266 M182 196 L212 230 M210 196 L142 306 M174 196 L124 266 M138 196 L108 230" stroke="#c8733a" stroke-width="2" opacity="0.6"/>
  <g style="stroke:var(--text2)" stroke-width="2" opacity="0.55">
    <line x1="24" y1="96" x2="112" y2="158"/><line x1="24" y1="96" x2="208" y2="158"/>
    <line x1="24" y1="214" x2="112" y2="158"/><line x1="24" y1="214" x2="208" y2="158"/>
    <line x1="112" y1="158" x2="160" y2="78"/><line x1="208" y1="158" x2="160" y2="78"/>
    <line x1="160" y1="78" x2="296" y2="78"/>
  </g>
  <rect x="6" y="78" width="36" height="36" rx="8" style="fill:var(--surface);stroke:var(--axis)" stroke-width="2"/>
  <rect x="6" y="196" width="36" height="36" rx="8" style="fill:var(--surface);stroke:var(--axis)" stroke-width="2"/>
  <circle cx="112" cy="158" r="50" fill="#2a78d6"/>
  <circle cx="208" cy="158" r="50" fill="#5598e7"/>
  <circle cx="160" cy="78" r="52" fill="#eb6834"/>
  <polyline points="94,170 112,170 130,148" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="190,170 208,170 226,148" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="160" y="86" text-anchor="middle" fill="#fff" font-size="26" font-weight="700" font-family="system-ui, sans-serif">350</text>
  <circle cx="296" cy="78" r="10" style="fill:var(--surface);stroke:var(--axis)" stroke-width="2"/>
</svg>"""

SYMBOLE = {
    "faden": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--blue)"><path d="M7 10h10L12 22z"/><circle cx="12" cy="7" r="5"/></svg>',
    "frage": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--blue)"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17.5h.01"/></svg>',
    "labor": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--blue)"><path d="M4 7h10M4 17h6M18 7h2M14 17h6"/><circle cx="16" cy="7" r="2"/><circle cx="12" cy="17" r="2"/></svg>',
    "kasten": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--blue)"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 12h8M12 8v8"/></svg>',
}


def startseite(infos):
    karten = ""
    for k, (titel, minuten) in zip(KAPITEL, infos):
        themen = "".join(f"<li>{t}</li>" for t in k["themen"])
        karten += f"""<article class="karte">
  <div class="karte-bild"><img src="{k["bild"]}" alt="" loading="lazy"></div>
  <div class="karte-text">
    <p class="augenbraue">Kapitel {k["nr"]}</p>
    <h3>{titel}</h3>
    <p>{k["teaser"]}</p>
    <ul class="chips">{themen}</ul>
    <p class="meta">etwa {minuten} Minuten Lesezeit</p>
    <div class="aktionen"><a class="knopf primaer" href="{k["ziel"]}">Kapitel lesen</a><a class="knopf" href="{k["labor"]}">Labor {k["nr"]}</a></div>
  </div>
</article>"""
    labore = "".join(
        f'<a class="laborkarte" href="{ziel}"><img src="{bild}" alt="" loading="lazy"><div><h3>{name}</h3><p>{text}</p></div></a>'
        for ziel, name, text, bild in LABORE
    )
    inhalt = f"""<section class="held"><div class="huelle">
  <div>
    <p class="augenbraue">KI-Workshop</p>
    <h1>Vom Eisverkäufer zum Sprachmodell</h1>
    <p class="lead">Wie lernt eine Maschine? Was steckt hinter ChatGPT und Co.? Und wie schlägt eine KI in eigenen Unterlagen nach? Dieser Workshop erklärt es an einem durchgehenden Beispiel, ohne Vorkenntnisse und ohne Programmieren.</p>
    <div class="aktionen"><a class="knopf primaer" href="kapitel-1.html">Mit Kapitel 1 beginnen</a><a class="knopf" href="#labore">Direkt ausprobieren</a></div>
  </div>
  {ILLUSTRATION}
</div></section>

<div class="huelle">
  <section class="bereich">
    <h2>Die Kapitel</h2>
    <p class="bereich-sub">Jedes Kapitel beantwortet eine Leitfrage. Die Kapitel bauen aufeinander auf, Vorwissen ist nicht nötig: Grundrechenarten genügen.</p>
    <div class="kapitelkarten">{karten}</div>
  </section>

  <section class="bereich">
    <h2>So lernst du in diesem Workshop</h2>
    <p class="bereich-sub">Erst selbst nachdenken und ausprobieren, dann den Fachbegriff kennenlernen. Jedes Kapitel folgt demselben Muster.</p>
    <div class="merkmale">
      <div class="merkmal"><div class="symbol">{SYMBOLE["faden"]}</div><h3>Ein Beispiel, das trägt</h3><p>Ein Eisverkäufer will wissen, wie viel Eis er braucht. An seinen Daten entstehen Gerade, Neuron und Netz, und auch Sprachmodell und Suche knüpfen daran an.</p></div>
      <div class="merkmal"><div class="symbol">{SYMBOLE["frage"]}</div><h3>Mitdenken statt nur zuhören</h3><p>Lernziele am Anfang, Fragen zum Kurz-Überlegen, ein Merksatz nach jedem Abschnitt und ein kleiner Selbsttest am Ende jedes Kapitels.</p></div>
      <div class="merkmal"><div class="symbol">{SYMBOLE["labor"]}</div><h3>Ausprobieren im Labor</h3><p>Regler verschieben, Texte zerlegen, nach Bedeutung suchen. Die Labore laufen direkt im Browser, ohne Installation.</p></div>
      <div class="merkmal"><div class="symbol">{SYMBOLE["kasten"]}</div><h3>Mathematik nur für Neugierige</h3><p>Formeln und Rechenwege stehen in aufklappbaren Kästen. Wer sie überspringt, verliert nicht den Faden.</p></div>
    </div>
  </section>

  <section class="bereich" id="labore">
    <h2>Die Labore</h2>
    <p class="bereich-sub">Interaktive Seiten zu den Kapiteln, gedacht zum gemeinsamen Ausprobieren im Workshop, am besten zu zweit an einem Gerät.</p>
    <div class="laborkarten">{labore}</div>
  </section>

  <section class="bereich">
    <h2>Nachschlagen und vorbereiten</h2>
    <div class="zusatzkarten">
      <a class="zusatzkarte" href="glossar.html"><h3>Glossar</h3><p>Alle Fachbegriffe in Alltagssprache, von „Attention“ bis „Weight“, jeweils mit Verweis auf das Kapitel, in dem sie erklärt werden.</p></a>
      <a class="zusatzkarte" href="leitfaden.html"><h3>Leitfaden für Moderierende</h3><p>Ablauf für zwei Sitzungen mit 10 bis 15 Teilnehmenden, Fragen an die Gruppe, typische Rückfragen aus dem Publikum und eine Checkliste für die Technik.</p></a>
    </div>
  </section>

  <details class="start-anleitung">
    <summary>Für technisch Interessierte: alles selbst nachrechnen</summary>
    <p>Hinter jeder Zahl und jeder Grafik steht ein Python-Skript. Für den Workshop ist es nicht nötig. Die Skripte brauchen nur <a href="https://docs.astral.sh/uv/">uv</a>, die Abhängigkeiten installiert uv selbst:</p>
    <div class="text"><pre><code>uv run docs/beispiele/01_eisverkaeufer.py
uv run docs/beispiele/02_tokenizer.py "Drei Kugeln Vanille, bitte."
uv run docs/beispiele/03_semantische_suche.py "Habt ihr Eis für Hunde?"</code></pre></div>
  </details>
</div>"""
    (DOCS / "index.html").write_text(seite("KI-Workshop: Vom Eisverkäufer zum Sprachmodell", "Wie neuronale Netze, Sprachmodelle und die Suche nach Bedeutung funktionieren, an einem durchgehenden Beispiel erklärt, ohne Vorkenntnisse und mit Laboren zum Ausprobieren.", "index.html", inhalt), encoding="utf-8")


def main():
    infos = [kapitelseite(k) for k in KAPITEL]
    zusatz = [zusatzseite(z) for z in ZUSATZ]
    startseite(infos)
    # Ohne diese Datei würde GitHub Pages die Markdown-Dateien mit Jekyll umwandeln
    (DOCS / ".nojekyll").write_text("")
    for k, (titel, minuten) in zip(KAPITEL, infos):
        print(f"docs/{k['ziel']}: {titel} (etwa {minuten} Minuten)")
    for z, titel in zip(ZUSATZ, zusatz):
        print(f"docs/{z['ziel']}: {titel}")
    print("docs/index.html")


if __name__ == "__main__":
    main()

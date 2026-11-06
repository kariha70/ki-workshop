# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime", "tokenizers", "huggingface_hub", "numpy", "matplotlib"]
# ///
"""Semantische Suche und RAG: die Beispiele aus Kapitel 3.

Start:        uv run docs/beispiele/03_semantische_suche.py
Eigene Frage: uv run docs/beispiele/03_semantische_suche.py "Habt ihr Eis für Hunde?"

Beim ersten Start lädt das Skript das Embedding-Modell paraphrase-multilingual-MiniLM-L12-v2
von Hugging Face (etwa 470 MB) und legt es im Cache ab. Es rechnet die Beispiele aus
03-Semantische-Suche-und-RAG.md nach, erzeugt die Grafiken in docs/bilder/kapitel-3/ und
die Daten für docs/app/labor-3.html.

Das Modell läuft hier in voller Genauigkeit. Labor 3 lädt für eigene Eingaben eine
verkleinerte Fassung (etwa 118 MB), deren Werte um ein bis zwei Hundertstel abweichen können.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from matplotlib.patches import Arc, FancyBboxPatch
from tokenizers import Tokenizer

DOCS = Path(__file__).resolve().parent.parent
BILDER = DOCS / "bilder" / "kapitel-3"
APP_DATEN = DOCS / "app" / "such-daten.js"

# Dasselbe Modell, das Labor 3 mit transformers.js im Browser lädt
MODELL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"

# Das erfundene Handbuch der Eisdiele: (Thema, Überschrift, Text)
HANDBUCH = [
    ("Sorten", "Unsere Sorten", "Jeden Tag gibt es zwölf Sorten, darunter Vanille, Schokolade, Erdbeere, Stracciatella und Pistazie."),
    ("Sorten", "Sorte des Monats", "Im Juli ist Mango-Maracuja die Sorte des Monats, im August Zitrone-Basilikum."),
    ("Sorten", "Herstellung", "Das Eis wird jeden Morgen frisch in der eigenen Küche gerührt, ohne Fertigmischungen."),
    ("Sorten", "Zutaten", "Für das Milcheis verwenden wir Vollmilch und Sahne von einem Bauernhof aus der Region."),
    ("Gesundheit", "Vegane Sorten", "Alle Fruchtsorbets sind vegan und enthalten keine Laktose. Sie werden nur aus Obst, Wasser und Zucker gemacht."),
    ("Gesundheit", "Nüsse", "Pistazie und Haselnuss enthalten Nüsse. Spuren von Nüssen können in allen Sorten vorkommen."),
    ("Gesundheit", "Glutenfrei", "Die Waffeln enthalten Weizenmehl. Wer kein Gluten verträgt, nimmt das Eis besser im Becher."),
    ("Gesundheit", "Zucker", "Eine Kugel Milcheis enthält etwa 15 Gramm Zucker, ein Sorbet etwas mehr."),
    ("Gesundheit", "Hirnfrost", "Wer sehr kaltes Eis zu schnell isst, bekommt manchmal kurz stechende Kopfschmerzen. Langsam essen hilft."),
    ("Öffnungszeiten", "Öffnungszeiten", "Geöffnet ist von März bis Oktober, täglich von 11 bis 20 Uhr."),
    ("Öffnungszeiten", "Wetter", "Bei Gewitter oder Dauerregen schließt der Stand früher. Unter 10 °C bleibt er ganz zu."),
    ("Öffnungszeiten", "Winterpause", "Von November bis Februar ist Winterpause. Dann wird die Eismaschine gewartet."),
    ("Preise", "Bezahlen", "Bezahlt werden kann bar, mit Girokarte oder kontaktlos mit dem Smartphone."),
    ("Preise", "Preise", "Eine Kugel kostet 1,80 Euro, jede weitere 1,60 Euro. Sahne kostet 50 Cent extra."),
    ("Preise", "Stempelkarte", "Für jede gekaufte Kugel gibt es einen Stempel. Nach zehn Stempeln ist die nächste Kugel gratis."),
    ("Preise", "Kinder", "Für Kinder unter sechs Jahren gibt es eine kleine Kinderkugel für 1,20 Euro."),
    ("Am Stand", "Lagerung", "Das Eis wird bei minus 18 °C gelagert und in der Theke bei minus 12 °C serviert, damit es cremig bleibt."),
    ("Am Stand", "Mitnehmen", "Für den Weg nach Hause gibt es Isolierboxen. Darin hält das Eis etwa eine Stunde."),
    ("Am Stand", "Auftauen", "Angetautes Eis wird nicht wieder eingefroren, sondern weggeworfen."),
    ("Am Stand", "Eisbedarf", "Bei 25 °C verkauft der Stand an einem Werktag etwa 270 Kugeln, am Wochenende etwa 320."),
    ("Am Stand", "Personal", "An heißen Wochenenden arbeiten drei Leute am Stand, sonst zwei."),
    ("Am Stand", "Lieferung", "Milch und Sahne werden jeden Montag und Donnerstag geliefert."),
    ("Am Stand", "Hunde", "Hunde sind willkommen. Für sie gibt es ein spezielles Hundeeis ohne Zucker."),
    ("Am Stand", "Feiern", "Für Geburtstage und Firmenfeiern liefern wir Eis ab 50 Kugeln."),
]

# Fragen für Text, Grafiken und Labor: (Frage, Hinweis)
FRAGEN = [
    ("Mir tut der Kopf weh, wenn ich Eis esse.", "Außer „Eis“ gibt es kein gemeinsames Wort, trotzdem landet „Hirnfrost“ ganz oben."),
    ("Kann ich mit dem Handy bezahlen?", "Im Handbuch steht „Smartphone“, nicht „Handy“. Die Bedeutung reicht."),
    ("Darf mein Hund mit?", "Die Stichwortsuche findet nichts, denn im Handbuch steht „Hunde“, nicht „Hund“."),
    ("Bekomme ich eine Kugel umsonst?", "„umsonst“ und „gratis“ sind verschiedene Wörter mit derselben Bedeutung."),
    ("Wie lange hält das Eis im Auto?", "Vom Auto steht nichts im Handbuch, die Isolierbox passt trotzdem."),
    ("Do you have vegan ice cream?", "Eine englische Frage findet den deutschen Text, denn das Modell kennt über 50 Sprachen."),
    ("Habt ihr etwas ohne Milch?", "Grenze: Das „ohne“ geht unter. Oben landet der Text, in dem es um Milch geht."),
    ("Gibt es Eis für Diabetiker?", "Grenze: Gesucht wäre „Zucker“. Es landet erst auf Platz 7, ganz oben steht „Glutenfrei“."),
    ("Liefert ihr auch?", "Grenze: Ähnlich ist nicht gleich passend. „Lieferung“ handelt von Milch für den Stand, gemeint ist „Feiern“."),
    ("1,20 Euro", "Grenze: Die semantische Suche bevorzugt „Preise“. Die Stichwortsuche findet „Kinder“, wo genau 1,20 Euro steht."),
    ("Wie wird das Wetter morgen?", "Grenze: Eine Wettervorhersage steht nicht im Handbuch. Die Suche liefert trotzdem Treffer, und „Wetter“ klingt passend."),
]

# Satzpaare zum Vergleichen (Abschnitte 3, 4 und 7): (Satz A, Satz B, Hinweis)
VERGLEICHE = [
    ("Vanille", "Schokolade", "Zwei Eissorten: deutlich ähnlich."),
    ("Vanille", "Steuererklärung", "Zwei Wörter ohne Zusammenhang: fast 0."),
    ("Mir tut der Kopf weh.", "Ich habe Kopfschmerzen.", "Verschiedene Wörter, dieselbe Bedeutung."),
    ("Zwei Kugeln Schokolade im Becher, bitte.", "Two scoops of chocolate in a cup, please.", "Derselbe Satz auf Deutsch und Englisch, die Bestellung aus Kapitel 2."),
    ("Hund beißt Mann.", "Mann beißt Hund.", "Andere Reihenfolge, andere Bedeutung, und doch fast gleiche Zahlenlisten."),
    ("Eis mit Nüssen", "Eis ohne Nüsse", "Gegenteilige Bedeutung, trotzdem sehr ähnlich: Verneinungen gehen leicht unter."),
]

# Fragen und Antworten aus dem Handbuch für die Trainingsgrafik (Abschnitt 4)
PAARE = [
    ("Was kostet eine Kugel?", "Preise"),
    ("Wann habt ihr geöffnet?", "Öffnungszeiten"),
    ("Darf mein Hund mit?", "Hunde"),
    ("Ist das Sorbet vegan?", "Vegane Sorten"),
]

# Ausgedachte Pfeile mit zwei Zahlen zum Nachrechnen (Abschnitte 2 und 3)
SPIELZEUG = [
    ("F", "Frage: „Was kostet eine Kugel?“", (3, 1)),
    ("P", "„Preise: Eine Kugel kostet …“", (6, 2)),
    ("H", "„Hirnfrost: Wer sehr kaltes Eis …“", (1, 3)),
]

# Anweisung am Anfang des Prompts für RAG (Abschnitt 6)
ANWEISUNG = (
    "Beantworte die Frage nur mit Hilfe der Auszüge aus dem Handbuch der Eisdiele.\n"
    "Wenn die Antwort nicht in den Auszügen steht, sag das. Nenne die Nummer des Auszugs, auf den du dich stützt."
)

# Häufige Wörter, die die Stichwortsuche nicht mitzählt
STOPPWOERTER = sorted(set(
    "der die das den dem des ein eine einen einem einer und oder aber ist sind wird werden bei mit für von vom zu zum zur im in an am auf aus "
    "es er sie ich du wir ihr mein meine dein habt hat haben gibt kann darf wenn was wie wer wo auch nicht noch nur sehr etwas "
    "the a an of in to do you have is are and for".split()
))

AQUA = "#1baf7a"
GELB = "#eda100"
LILA = "#8a63d2"

BLAU = "#2a78d6"
ORANGE = "#eb6834"
FLAECHE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LEISE = "#52514e"
GRAU = "#898781"
GITTER = "#e1e0d9"
ACHSE = "#c3c2b7"

# Beschriftungen auf der Karte, die nicht rechts über ihrem Punkt stehen: Versatz in Punkten und Ausrichtung
KARTE_LAGE = {"Hunde": (5, -13, "left"), "Öffnungszeiten": (5, -13, "left"), "Lagerung": (-6, -4, "right")}

THEMEN = {"Sorten": BLAU, "Gesundheit": AQUA, "Öffnungszeiten": GELB, "Preise": ORANGE, "Am Stand": LILA}

plt.rcParams.update(
    {
        # Mehrere Schriften: Fehlt ein Zeichen wie „→“, nimmt matplotlib es aus der nächsten
        "font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 12,
        "figure.facecolor": FLAECHE,
        "axes.facecolor": FLAECHE,
        "axes.edgecolor": ACHSE,
        "axes.labelcolor": TEXT_LEISE,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": GRAU,
        "ytick.color": GRAU,
        "xtick.labelcolor": TEXT_LEISE,
        "ytick.labelcolor": TEXT_LEISE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def modell_laden():
    """Lädt Tokenizer und Modell und gibt eine Funktion zurück, die Text in ein Embedding verwandelt."""
    tokenizer = Tokenizer.from_file(hf_hub_download(MODELL, "tokenizer.json"))
    sitzung = onnxruntime.InferenceSession(hf_hub_download(MODELL, "onnx/model.onnx"))

    def einbetten(text):
        kodiert = tokenizer.encode(text)
        ids = np.array([kodiert.ids])
        # Der Transformer liefert für jedes Token eine Zahlenliste mit 384 Einträgen ...
        pro_token = sitzung.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)})[0][0]
        # ... der Durchschnitt über alle Tokens ist das Embedding des ganzen Textes, auf Länge 1 gebracht
        mittel = pro_token.mean(axis=0)
        return mittel / np.linalg.norm(mittel)

    return einbetten, tokenizer


def kosinus(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stueck(eintrag):
    """Jedes Stück bekommt seine Überschrift mit, damit es auch allein verständlich bleibt."""
    _, titel, text = eintrag
    return f"{titel}: {text}"


def woerter(text):
    """Wörter für die Stichwortsuche: klein geschrieben, ohne häufige Füllwörter."""
    return {w for w in re.findall(r"[^\W_]+", text.lower()) if w not in STOPPWOERTER}


def stichwortsuche(frage):
    """Zählt für jedes Stück, wie viele Wörter der Frage darin vorkommen."""
    gesucht = woerter(frage)
    return [len(gesucht & woerter(stueck(e))) for e in HANDBUCH]


def rangliste(werte):
    """Nummern der Stücke, das beste zuerst. Bei Gleichstand gewinnt das frühere Stück."""
    return sorted(range(len(werte)), key=lambda i: (-werte[i], i))


def komma(zahl, stellen=2):
    return f"{zahl:.{stellen}f}".replace(".", ",").replace("-", "−")


def pca(vektoren, dimensionen=2):
    """Die zwei Richtungen, in denen sich die Vektoren am stärksten unterscheiden."""
    mitte = vektoren.mean(axis=0)
    _, _, achsen = np.linalg.svd(vektoren - mitte, full_matrices=False)
    achsen = achsen[:dimensionen]
    # Vorzeichen festlegen, damit die Karte bei jedem Lauf gleich herum liegt
    achsen *= np.sign(achsen[np.arange(dimensionen), np.abs(achsen).argmax(axis=1)])[:, None]
    return mitte, achsen


def grafik_pfeile():
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    farben = {"F": BLAU, "P": ORANGE, "H": AQUA}
    punkte = {name: np.array(xy, float) for name, _, xy in SPIELZEUG}
    f, p, h = punkte["F"], punkte["P"], punkte["H"]
    # Abstand von F zu H als gestrichelte Linie
    ax.plot([f[0], h[0]], [f[1], h[1]], linestyle=(0, (3, 3)), color=GRAU, linewidth=1.3, zorder=1)
    ax.text((f[0] + h[0]) / 2 + 0.15, (f[1] + h[1]) / 2 + 0.1, f"Abstand {komma(np.linalg.norm(f - h))}", color=TEXT_LEISE, fontsize=10.5)
    # Abstand von F zu P liegt auf dem Pfeil P, deshalb leicht versetzt darüber
    versatz = np.array([-p[1], p[0]]) / np.linalg.norm(p) * 0.3
    ax.annotate("", p + versatz, f + versatz, arrowprops={"arrowstyle": "<->", "color": GRAU, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0})
    mitte = (f + p) / 2 + versatz * 1.6
    ax.text(mitte[0], mitte[1], f"Abstand {komma(np.linalg.norm(f - p))}", color=TEXT_LEISE, fontsize=10.5, ha="center", va="bottom", rotation=np.degrees(np.arctan2(p[1], p[0])), rotation_mode="anchor")
    # Winkel zwischen F und H
    winkel_f = np.degrees(np.arctan2(f[1], f[0]))
    winkel_h = np.degrees(np.arctan2(h[1], h[0]))
    ax.add_patch(Arc((0, 0), 2.4, 2.4, theta1=winkel_f, theta2=winkel_h, color=TEXT_LEISE, linewidth=1.2))
    ax.text(1.05, 0.95, f"{winkel_h - winkel_f:.0f}°", color=TEXT_LEISE, fontsize=10.5)
    # P zuerst, damit F oben liegt und sichtbar bleibt
    lagen = {"P": ((0.15, 0.1), (0.15, -0.25)), "F": ((0.1, -0.45), (0.1, -0.78)), "H": ((-0.05, 0.62), (-0.05, 0.3))}
    for name, beschreibung, (x, y) in sorted(SPIELZEUG, key=lambda s: s[0] != "P"):
        ax.annotate("", (x, y), (0, 0), arrowprops={"arrowstyle": "-|>", "color": farben[name], "linewidth": 3.2 if name == "F" else 2.4,
                                                    "mutation_scale": 16, "shrinkA": 0, "shrinkB": 0}, zorder=3 if name == "F" else 2)
        (nx, ny), (bx, by) = lagen[name]
        ax.text(x + nx, y + ny, f"{name} ({x}, {y})", color=farben[name], fontsize=12, fontweight="bold")
        ax.text(x + bx, y + by, beschreibung, color=TEXT_LEISE, fontsize=10)
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 4.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 10))
    ax.set_yticks(range(0, 5))
    ax.grid(color=GITTER, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel("Zahl 1: Wie sehr geht es um Geld? (ausgedacht)")
    ax.set_ylabel("Zahl 2: Wie sehr um Gesundheit?")
    fig.savefig(BILDER / "01-pfeile.png")
    plt.close(fig)


def grafik_zahlenlisten(einbetten):
    texte = [
        ("Was kostet eine Kugel?", "Frage"),
        (stueck(HANDBUCH[13]), "Preise"),
        (stueck(HANDBUCH[8]), "Hirnfrost"),
    ]
    vektoren = np.array([einbetten(t) for t, _ in texte])
    grenze = np.abs(vektoren).max()
    fig, achsen = plt.subplots(len(texte), 1, figsize=(10, 2.9), sharex=True)
    farbskala = matplotlib.colors.LinearSegmentedColormap.from_list("blau_orange", [BLAU, FLAECHE, ORANGE])
    for ax, (text, name), v in zip(achsen, texte, vektoren):
        ax.imshow(v[None, :], aspect="auto", cmap=farbskala, vmin=-grenze, vmax=grenze, interpolation="nearest")
        ax.set_yticks([])
        for seite in ("left", "bottom"):
            ax.spines[seite].set_visible(False)
        aehnlich = "" if name == "Frage" else f"   Kosinus zur Frage: {komma(kosinus(vektoren[0], v))}"
        kurz = text if len(text) < 60 else text[:57] + " …"
        ax.set_title(f"{name}: „{kurz}“{aehnlich}", fontsize=11, fontweight="normal", color=TEXT, pad=4)
    achsen[-1].set_xticks([0, 95, 191, 287, 383], ["1", "96", "192", "288", "384"])
    achsen[-1].tick_params(axis="x", length=0)
    achsen[-1].set_xlabel("Nummer der Zahl in der Liste. Blau: negativ, Orange: positiv, hell: nahe 0")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(BILDER / "02-zahlenlisten.png")
    plt.close(fig)
    return vektoren


def grafik_training(einbetten, vektoren_handbuch):
    titel = [e[1] for e in HANDBUCH]
    fragen = np.array([einbetten(f) for f, _ in PAARE])
    antworten = np.array([vektoren_handbuch[titel.index(a)] for _, a in PAARE])
    matrix = fragen @ antworten.T
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    farbskala = matplotlib.colors.LinearSegmentedColormap.from_list("hell_blau", [FLAECHE, BLAU])
    ax.imshow(matrix, cmap=farbskala, vmin=0, vmax=1)
    for i in range(len(PAARE)):
        for j in range(len(PAARE)):
            ax.text(j, i, komma(matrix[i, j]), ha="center", va="center", fontsize=12, color="#ffffff" if matrix[i, j] > 0.5 else TEXT,
                    fontweight="bold" if i == j else "normal")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=2.5))
    ax.set_xticks(range(len(PAARE)), [a.replace("Öffnungszeiten", "Öffnungs-\nzeiten").replace(" ", "\n") for _, a in PAARE])
    ax.set_yticks(range(len(PAARE)), [f for f, _ in PAARE])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for seite in ax.spines.values():
        seite.set_visible(False)
    ax.set_xlabel("Orange umrandet: die passende Antwort. Sie soll in jeder Zeile gewinnen.", color=TEXT_LEISE)
    fig.savefig(BILDER / "03-training.png")
    plt.close(fig)
    return matrix


def balken_rangliste(ax, werte, reihenfolge, anzahl, format_wert, markiert=(), titel=""):
    zeilen = reihenfolge[:anzahl]
    y = np.arange(len(zeilen))[::-1]
    farben = [ORANGE if i in markiert else BLAU for i in zeilen]
    ax.barh(y, [werte[i] for i in zeilen], height=0.6, color=farben)
    ax.set_yticks(y, [HANDBUCH[i][1] for i in zeilen])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    groesst = max(max(werte), 1e-9)
    for yi, i in zip(y, zeilen):
        ax.text(werte[i] + groesst * 0.02, yi, format_wert(werte[i]), va="center", color=TEXT, fontsize=11)
    ax.set_xlim(0, groesst * 1.22)
    ax.set_title(titel, fontsize=12)


def grafik_suche(frage, aehnlichkeiten, treffer):
    titel = [e[1] for e in HANDBUCH]
    ziel = {titel.index("Hirnfrost")}
    fig, achsen = plt.subplots(1, 2, figsize=(10.5, 3.9))
    balken_rangliste(achsen[0], aehnlichkeiten, rangliste(aehnlichkeiten), 6, komma, ziel, "Semantische Suche: Kosinus")
    balken_rangliste(achsen[1], treffer, rangliste(treffer), 6, lambda w: f"{w} Wort" if w == 1 else f"{w} Wörter", ziel, "Stichwortsuche: gemeinsame Wörter")
    fig.suptitle(f"Frage: „{frage}“", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout(w_pad=4)
    fig.savefig(BILDER / "04-suche.png")
    plt.close(fig)


def grafik_karte(vektoren, mitte, achsen, frage, frage_vektor):
    punkte = (vektoren - mitte) @ achsen.T
    frage_punkt = (frage_vektor - mitte) @ achsen.T
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for thema, farbe in THEMEN.items():
        auswahl = [i for i, e in enumerate(HANDBUCH) if e[0] == thema]
        ax.scatter(punkte[auswahl, 0], punkte[auswahl, 1], s=70, color=farbe, edgecolors=FLAECHE, linewidths=1.5, zorder=3, label=thema)
        for i in auswahl:
            # Bei dicht beieinander liegenden Punkten die Beschriftung verschieben
            dx, dy, ausrichtung = KARTE_LAGE.get(HANDBUCH[i][1], (5, 4, "left"))
            ax.annotate(HANDBUCH[i][1], punkte[i], xytext=(dx, dy), textcoords="offset points", ha=ausrichtung, color=TEXT, fontsize=10)
    ax.scatter([frage_punkt[0]], [frage_punkt[1]], s=260, marker="*", color=TEXT, edgecolors=FLAECHE, linewidths=1, zorder=4)
    ax.annotate("Frage:\n„" + frage.replace(", ", ",\n") + "“", frage_punkt, xytext=(10, -22), textcoords="offset points", va="top", color=TEXT, fontsize=10.5, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": TEXT_LEISE, "linewidth": 1, "shrinkA": 2, "shrinkB": 8})
    ax.set_xticks([])
    ax.set_yticks([])
    for seite in ("left", "bottom"):
        ax.spines[seite].set_visible(False)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT_LEISE, handletextpad=0.3)
    fig.savefig(BILDER / "05-karte.png")
    plt.close(fig)
    return punkte


def kasten(ax, x, y, breite, hoehe, titel, unterzeile="", farbe=FLAECHE, rand=ACHSE, textfarbe=TEXT, schrift=12):
    ax.add_patch(FancyBboxPatch((x, y), breite, hoehe, boxstyle="round,pad=0,rounding_size=1.2", facecolor=farbe, edgecolor=rand, linewidth=1.4, zorder=2))
    mitte = x + breite / 2
    if unterzeile:
        ax.text(mitte, y + hoehe * 0.66, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)
        ax.text(mitte, y + hoehe * 0.3, unterzeile, ha="center", va="center", color=TEXT_LEISE if textfarbe == TEXT else textfarbe, fontsize=9.5, zorder=3, linespacing=1.3)
    else:
        ax.text(mitte, y + hoehe / 2, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)


def pfeil(ax, start, ende, beschriftung=""):
    ax.annotate("", ende, start, arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0})
    if beschriftung:
        ax.text((start[0] + ende[0]) / 2 + 1, (start[1] + ende[1]) / 2, beschriftung, color=TEXT_LEISE, fontsize=9.5, va="center")


def grafik_rag():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 46)
    ax.axis("off")
    # Vorab, einmal: das Handbuch wird zerlegt und eingebettet
    ax.add_patch(FancyBboxPatch((0, 30), 123.5, 15, boxstyle="round,pad=0,rounding_size=1.4", facecolor=GITTER, alpha=0.35, edgecolor="none", zorder=1))
    ax.text(1.5, 42.8, "Vorab, einmal", color=TEXT_LEISE, fontsize=10.5, fontweight="bold", va="center")
    kasten(ax, 21.5, 31.5, 22, 11, "Handbuch", "24 kurze Stücke\nmit Überschrift")
    kasten(ax, 48.5, 31.5, 22, 11, "Embedding-Modell", "eine Zahlenliste\npro Stück")
    kasten(ax, 75.5, 31.5, 22, 11, "Gespeichert", "24 Listen mit\nje 384 Zahlen")
    pfeil(ax, (43.9, 37), (48.1, 37))
    pfeil(ax, (70.9, 37), (75.1, 37))
    # Bei jeder Frage
    ax.text(1.5, 25.5, "Bei jeder Frage", color=TEXT_LEISE, fontsize=10.5, fontweight="bold", va="center")
    kasten(ax, 0, 5, 17.5, 16, "Frage", "„Mir tut der\nKopf weh, wenn\nich Eis esse.“", farbe=FLAECHE)
    kasten(ax, 21.5, 5, 18.5, 16, "Embedding-\nModell", "", farbe=FLAECHE, schrift=11.5)
    kasten(ax, 44, 5, 23, 16, "Suchen", "Kosinus mit allen\ngespeicherten Listen,\ndie besten 3 nehmen", farbe=BLAU, rand=BLAU, textfarbe="#ffffff")
    kasten(ax, 71, 5, 25, 16, "Prompt", "Anweisung\n+ 3 Stücke aus dem\nHandbuch + Frage", farbe=FLAECHE)
    kasten(ax, 100, 5, 23.5, 16, "Sprachmodell", "antwortet mit Hilfe\nder Stücke und\nnennt die Quelle", farbe=ORANGE, rand=ORANGE, textfarbe="#ffffff")
    for x0, x1 in ((17.5, 21.5), (40, 44), (67, 71), (96, 100)):
        pfeil(ax, (x0 + 0.3, 13), (x1 - 0.3, 13))
    pfeil(ax, (86.5, 31.2), (55.5, 21.3))
    ax.text(76, 26.8, "vergleichen", color=TEXT_LEISE, fontsize=9.5, va="center")
    fig.savefig(BILDER / "06-rag.png")
    plt.close(fig)


def grafik_grenze(frage, aehnlichkeiten):
    titel = [e[1] for e in HANDBUCH]
    gesucht = titel.index("Vegane Sorten")
    reihenfolge = rangliste(aehnlichkeiten)
    platz = reihenfolge.index(gesucht) + 1
    fig, ax = plt.subplots(figsize=(8, 3.4))
    balken_rangliste(ax, aehnlichkeiten, reihenfolge, max(6, platz), komma, {gesucht})
    ax.set_title(f"„{frage}“: Die passende Antwort landet erst auf Platz {platz}", fontsize=12.5)
    fig.savefig(BILDER / "07-grenze.png")
    plt.close(fig)
    return platz


def runden(vektor, stellen=5):
    return [round(float(x), stellen) for x in vektor]


def app_daten(vektoren, mitte, achsen, punkte, frage_vektoren, vergleich_werte):
    daten = {
        "modell": {"name": MODELL, "dimensionen": int(vektoren.shape[1])},
        "themen": THEMEN,
        "handbuch": [{"thema": thema, "titel": titel, "text": text} for thema, titel, text in HANDBUCH],
        "vektoren": [runden(v) for v in vektoren],
        "karte": {"mitte": runden(mitte), "achsen": [runden(a) for a in achsen], "punkte": [runden(p, 4) for p in punkte], "lage": KARTE_LAGE},
        "fragen": [{"text": frage, "hinweis": hinweis, "vektor": runden(v)} for (frage, hinweis), v in zip(FRAGEN, frage_vektoren)],
        "vergleiche": [{"a": a, "b": b, "hinweis": hinweis, "wert": round(w, 4)} for (a, b, hinweis), w in zip(VERGLEICHE, vergleich_werte)],
        "spielzeug": [{"name": name, "text": text, "xy": list(xy)} for name, text, xy in SPIELZEUG],
        "stoppwoerter": STOPPWOERTER,
        "anweisung": ANWEISUNG,
    }
    APP_DATEN.write_text(
        "// Erzeugt von docs/beispiele/03_semantische_suche.py, bitte nicht von Hand ändern.\n"
        "window.SUCH_DATEN = " + json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def prompt(frage, vektoren, frage_vektor, anzahl=3):
    """Der Prompt für RAG: Anweisung, die besten Stücke aus dem Handbuch, Frage."""
    auszuege = "\n".join(f"[{platz}] {stueck(HANDBUCH[i])}" for platz, i in enumerate(rangliste(list(vektoren @ frage_vektor))[:anzahl], start=1))
    return f"{ANWEISUNG}\n\n{auszuege}\n\nFrage: {frage}"


def zeige_rangliste(frage, aehnlichkeiten, treffer, anzahl=5):
    print(f"\n   Frage: {frage}")
    for platz, i in enumerate(rangliste(aehnlichkeiten)[:anzahl], start=1):
        print(f"   {platz}. {aehnlichkeiten[i]:5.2f}  {HANDBUCH[i][1]}")
    bester = rangliste(treffer)[0]
    anzahl_woerter = "1 gemeinsames Wort" if treffer[bester] == 1 else f"{treffer[bester]} gemeinsame Wörter"
    stichwort = f"{HANDBUCH[bester][1]} ({anzahl_woerter})" if treffer[bester] else "nichts"
    gleichauf = sum(t == treffer[bester] for t in treffer)
    print(f"   Stichwortsuche: {stichwort}" + (f", gleichauf mit {gleichauf - 1} weiteren Stück" + ("en" if gleichauf > 2 else "") if treffer[bester] and gleichauf > 1 else ""))


def main():
    einbetten, tokenizer = modell_laden()
    vektoren = np.array([einbetten(stueck(e)) for e in HANDBUCH])

    if len(sys.argv) > 1:
        frage = " ".join(sys.argv[1:])
        zeige_rangliste(frage, list(vektoren @ einbetten(frage)), stichwortsuche(frage), anzahl=len(HANDBUCH))
        return

    BILDER.mkdir(parents=True, exist_ok=True)

    print("Abschnitt 2 und 3: Ausgedachte Pfeile mit zwei Zahlen")
    punkte = {name: np.array(xy, float) for name, _, xy in SPIELZEUG}
    for name in ("P", "H"):
        a, b = punkte["F"], punkte[name]
        print(f"   F · {name} = {a[0]:.0f}·{b[0]:.0f} + {a[1]:.0f}·{b[1]:.0f} = {a @ b:.0f}, |F| = {np.linalg.norm(a):.2f}, |{name}| = {np.linalg.norm(b):.2f}, "
              f"Kosinus = {kosinus(a, b):.2f}, Winkel = {np.degrees(np.arccos(min(1, kosinus(a, b)))):.0f}°, Abstand = {np.linalg.norm(a - b):.2f}")

    print("\nAbschnitt 2: Ein echtes Embedding")
    frage = "Was kostet eine Kugel?"
    teile = tokenizer.encode(frage).tokens
    v = einbetten(frage)
    print(f"   „{frage}“ → {len(teile)} Tokens {teile}")
    print(f"   {len(v)} Zahlen, die ersten sechs: " + ", ".join(f"{x:.3f}" for x in v[:6]) + f", Länge {np.linalg.norm(v):.3f}")

    print("\nAbschnitt 3 und 4: Satzpaare")
    vergleich_werte = []
    for a, b, _ in VERGLEICHE:
        wert = kosinus(einbetten(a), einbetten(b))
        vergleich_werte.append(wert)
        print(f"   {wert:5.2f}  {a}  |  {b}")

    print("\nAbschnitt 4: Das Modell")
    print(f"   {MODELL}: {len(v)} Zahlen pro Text, Vokabular {tokenizer.get_vocab_size()} Tokens")
    matrix = grafik_training(einbetten, vektoren)
    print("   Fragen gegen Antworten (Kosinus):")
    for (f, _), zeile in zip(PAARE, matrix):
        print(f"   {f:28s} " + "  ".join(f"{x:5.2f}" for x in zeile))

    print("\nAbschnitt 5 und 7: Suchen im Handbuch")
    frage_vektoren = [einbetten(f) for f, _ in FRAGEN]
    for (f, _), fv in zip(FRAGEN, frage_vektoren):
        zeige_rangliste(f, list(vektoren @ fv), stichwortsuche(f), anzahl=3)
    print(f"\n   Tokens pro Stück im Handbuch: {min(len(tokenizer.encode(stueck(e)).ids) for e in HANDBUCH)} bis {max(len(tokenizer.encode(stueck(e)).ids) for e in HANDBUCH)}")

    haupt = FRAGEN[0][0]
    haupt_vektor = frage_vektoren[0]
    print("\nAbschnitt 6: Der Prompt für RAG\n")
    print("   " + prompt(haupt, vektoren, haupt_vektor).replace("\n", "\n   "))
    grafik_pfeile()
    grafik_zahlenlisten(einbetten)
    grafik_suche(haupt, list(vektoren @ haupt_vektor), stichwortsuche(haupt))
    mitte, achsen = pca(vektoren)
    karte = grafik_karte(vektoren, mitte, achsen, haupt, haupt_vektor)
    grafik_rag()
    milch = FRAGEN[6][0]
    platz = grafik_grenze(milch, list(vektoren @ frage_vektoren[6]))
    print(f"\n   „{milch}“: Vegane Sorten auf Platz {platz}")
    app_daten(vektoren, mitte, achsen, karte, frage_vektoren, vergleich_werte)
    print(f"\nGrafiken gespeichert in {BILDER}")
    print(f"Daten für die interaktive Seite gespeichert in {APP_DATEN}")


if __name__ == "__main__":
    main()

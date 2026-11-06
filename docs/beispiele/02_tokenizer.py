# /// script
# requires-python = ">=3.10"
# dependencies = ["tiktoken", "numpy", "matplotlib"]
# ///
"""Tokenizer und Sprachmodelle: die Beispiele aus Kapitel 2.

Start:        uv run docs/beispiele/02_tokenizer.py
Eigener Text: uv run docs/beispiele/02_tokenizer.py "Drei Kugeln Vanille, bitte."

Beim ersten Start lädt tiktoken die Vokabulare aus dem Internet und legt sie im Cache ab.
Das Skript rechnet die Beispiele aus 02-Tokenizer-und-Sprachmodelle.md nach, erzeugt die
Grafiken in docs/bilder/kapitel-2/ und die Beispieldaten für docs/app/labor-2.html.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent.parent
BILDER = DOCS / "bilder" / "kapitel-2"
APP_DATEN = DOCS / "app" / "token-beispiele.js"

# Die beiden Tokenizer und die Modelle, für die tiktoken sie verwendet
TOKENIZER = {"o200k_base": "gpt-4o", "cl100k_base": "gpt-4"}

# Beispieltexte für Text, Grafik und interaktive Seite: (Titel, Text, Hinweis)
BEISPIELE = [
    ("Bestellung auf Deutsch", "Zwei Kugeln Schokolade im Becher, bitte.", "Häufige Wörter wie „bitte“ sind ein einziges Token, seltenere wie „Kugeln“ zerfallen in Teile."),
    ("Bestellung auf Englisch", "Two scoops of chocolate in a cup, please.", "Derselbe Inhalt auf Englisch: Fast jedes Wort ist ein eigenes Token."),
    ("Satz aus Kapitel 1", "Der Eisverkäufer verkauft bei 25 Grad 290 Kugeln.", "Der ältere Tokenizer zerlegt das „ä“ in „Eisverkäufer“ in noch mehr Teile."),
    ("Satz aus Kapitel 1 auf Englisch", "The ice cream vendor sells 290 scoops at 25 degrees.", "Beide Tokenizer brauchen hier gleich viele Tokens."),
    ("Leerzeichen und Großschreibung", "Kugeln, Kugeln, kugeln, KUGELN", "Mit Leerzeichen davor, klein oder groß geschrieben: Jedes Mal entstehen andere Tokens."),
    ("Zahlen", "12345678 und 2025 und 3,14", "Zahlen werden in Blöcke von höchstens drei Ziffern zerlegt."),
    ("Buchstaben zählen", "Wie viele r stecken in Erdbeere?", "Das Modell sieht „Erd“, „be“ und „ere“, aber keine einzelnen Buchstaben."),
    ("Ein sehr langes Wort", "Donaudampfschifffahrtsgesellschaftskapitän", "Seltene Wörter zerfallen in viele Stücke, die nicht immer Silben sind."),
    ("Ein Rückbezug", "Ich nehme Erdbeere. Für meinen Sohn bitte dasselbe.", "„dasselbe“ bezieht sich auf „Erdbeere“. Dafür braucht das Modell Attention (Abschnitt 5)."),
    ("Ein Emoji", "Ein Eis 🍦 bitte!", "Das Emoji besteht aus vier Bytes. Sie landen in verschiedenen Tokens, die einzeln kein Zeichen ergeben."),
    ("Programmcode", "def neuron(inputs, weights, bias):", "Auch Code ist für das Modell nur eine Folge von Tokens."),
]

# Ausgedachte Punktzahlen für das nächste Token nach „Zwei Kugeln Schokolade im“ (Abschnitt 8)
PUNKTZAHLEN = {" Becher": 3.0, " Hörnchen": 2.2, " Glas": 1.0, " Kühlschrank": -0.5, " Regen": -1.5}

# Kleine Textsammlung, an der ein Mini-Tokenizer sein Vokabular lernt (Abschnitt 2)
KORPUS = [
    "zwei kugeln schoko im becher",
    "eine kugel vanille in der waffel",
    "drei kugeln schoko und eine kugel vanille",
    "eine kugel erdbeere im becher",
    "zwei kugeln vanille in der waffel",
    "vier kugeln schoko im becher",
]

# Modellgrößen aus den Quellen in Kapitel 2 (Abschnitt 7)
MODELLE = [
    ("Netz des Eisverkäufers", 9),
    ("Llama 3, kleine Version", 8e9),
    ("Chinchilla", 70e9),
    ("Llama 3, große Version", 70e9),
    ("GPT-3", 175e9),
    ("Gopher", 280e9),
]

AQUA = "#1baf7a"
GELB = "#eda100"

BLAU = "#2a78d6"
ORANGE = "#eb6834"
FLAECHE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LEISE = "#52514e"
GRAU = "#898781"
GITTER = "#e1e0d9"
ACHSE = "#c3c2b7"

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


def tokens(enc, text):
    """Zerlegt einen Text und gibt jedes Token als (ID, Text) zurück.

    Tokens, die für sich kein vollständiges Zeichen sind, erscheinen als Bytes, etwa <F0 9F 8D>.
    """
    ergebnis = []
    for token_id in enc.encode(text, disallowed_special=()):
        roh = enc.decode_single_token_bytes(token_id)
        try:
            anzeige = roh.decode("utf-8")
        except UnicodeDecodeError as fehler:
            # Gültige Zeichen am Anfang zeigen, den angebrochenen Rest als Bytes
            anzeige = roh[: fehler.start].decode("utf-8") + "<" + "\u00a0".join(f"{b:02X}" for b in roh[fehler.start :]) + ">"
        ergebnis.append((token_id, anzeige))
    return ergebnis


def sichtbar(token):
    """Leerzeichen als ␣ zeigen, damit man sieht, dass sie zum Token gehören."""
    return token.replace(" ", "␣")


def softmax(punkte, temperatur=1.0):
    werte = np.array(list(punkte.values())) / temperatur
    werte = np.exp(werte - werte.max())
    return werte / werte.sum()


def zeige(enc, text):
    teile = tokens(enc, text)
    print(f"   {enc.name:12s} {len(teile):3d} Tokens: " + " | ".join(sichtbar(t) for _, t in teile))
    print(f"   {'':12s}     IDs:    {[i for i, _ in teile]}")


def leere_achse(breite, hoehe, x=(0, 100), y=(0, 10)):
    """Achse ohne Rahmen für Schaubilder aus Kästchen und Text."""
    fig, ax = plt.subplots(figsize=(breite, hoehe))
    ax.set_xlim(*x)
    ax.set_ylim(*y)
    ax.axis("off")
    fig.canvas.draw()
    return fig, ax


def textbreite(ax, text):
    """Breite eines Textobjekts in Datenkoordinaten."""
    box = text.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, _), (x1, _) = ax.transData.inverted().transform([(box.x0, box.y0), (box.x1, box.y1)])
    return x1 - x0


def chips(ax, x, y, stuecke, farben, schrift=13, unten=None, hoehe=0.56, abstand=1.1):
    """Zeichnet Textstücke als farbige Kästchen nebeneinander.

    farben: pro Stück ein Paar (Farbe, Deckkraft). unten: optionaler kleiner Text unter jedem Stück.
    Gibt die Mitte jedes Kästchens und das rechte Ende zurück.
    """
    mitten = []
    for i, stueck in enumerate(stuecke):
        beschriftung = ax.text(x, y, stueck, color=TEXT, fontsize=schrift, va="center", ha="left", zorder=3)
        breite = kasten = textbreite(ax, beschriftung)
        if unten is not None:
            zusatz = ax.text(x, y - hoehe * 0.9, str(unten[i]), color=TEXT_LEISE, fontsize=8.5, va="center", ha="center", zorder=3)
            kasten = max(breite, textbreite(ax, zusatz))
            zusatz.set_x(x + kasten / 2)
        beschriftung.set_x(x + (kasten - breite) / 2)
        farbe, deckkraft = farben[i]
        ax.add_patch(FancyBboxPatch((x - 0.35, y - hoehe / 2), kasten + 0.7, hoehe, boxstyle="round,pad=0,rounding_size=0.12", facecolor=farbe, alpha=deckkraft, edgecolor="none", zorder=1))
        mitten.append(x + kasten / 2)
        x += kasten + abstand
    return mitten, x


def abwechselnd(anzahl):
    return [(BLAU if i % 2 == 0 else ORANGE, 0.16) for i in range(anzahl)]


def grafik_trainingsaufgaben(woerter):
    zeilen = len(woerter) - 1
    fig, ax = leere_achse(10, 0.62 * zeilen + 0.6, y=(-zeilen - 0.2, 0.9))
    for i in range(1, len(woerter)):
        y = -i + 1
        ax.text(0, y, f"Aufgabe {i}", color=TEXT_LEISE, fontsize=11, va="center")
        _, x = chips(ax, 13, y, woerter[:i], [(BLAU, 0.16)] * i)
        ax.text(x + 0.4, y, "→", color=TEXT_LEISE, fontsize=14, va="center")
        chips(ax, x + 4, y, [woerter[i]], [(ORANGE, 0.35)])
    ax.text(0, -zeilen + 0.05, "Blau: was das Modell bekommt. Orange: was es vorhersagen soll.", color=TEXT_LEISE, fontsize=10.5, va="center")
    fig.savefig(BILDER / "01-trainingsaufgaben.png")
    plt.close(fig)


def grafik_kreuzentropie():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    p = np.linspace(0.004, 1, 400)
    ax.plot(p, -np.log(p), color=BLAU, linewidth=2.5)
    for wert, versatz in ((1.0, (0, 0)), (0.5, (-0.02, -0.55)), (0.1, (0.04, 0.2)), (0.01, (0.04, 0))):
        fehler = abs(math.log(wert))
        ax.scatter([wert], [fehler], s=70, color=ORANGE, edgecolors=FLAECHE, linewidths=1.5, zorder=3)
        fehlertext = "0" if fehler == 0 else f"{fehler:.2f}".replace(".", ",")
        text = f"{wert * 100:.0f} % für das richtige Token: Fehler {fehlertext}"
        if wert == 1.0:
            # Rechts unten ist wenig Platz: Beschriftung höher setzen und mit einem Strich verbinden
            ax.annotate(text, (wert, fehler), xytext=(0.99, 0.8), ha="right", va="bottom", color=TEXT, fontsize=11,
                        arrowprops={"arrowstyle": "-", "color": TEXT_LEISE, "linewidth": 1, "shrinkA": 2, "shrinkB": 5})
        else:
            ax.text(wert + versatz[0], fehler + versatz[1], text, color=TEXT, fontsize=11, ha="right" if versatz[0] < 0 else "left", va="bottom")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 5.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("Wahrscheinlichkeit, die das Modell dem richtigen Token gegeben hat")
    ax.set_ylabel("Fehler (Kreuzentropie)")
    ax.grid(axis="y", color=GITTER, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_title("Je sicherer das Modell falsch liegt, desto teurer wird es")
    fig.savefig(BILDER / "02-kreuzentropie.png")
    plt.close(fig)


def bpe_lernen(saetze, schritte):
    """Ein Mini-Tokenizer: fasst Schritt für Schritt das häufigste Paar zusammen."""
    woerter = Counter(wort for satz in saetze for wort in satz.split())
    zerlegung = {wort: list(wort) for wort in woerter}
    verlauf = [(None, 0, {w: list(z) for w, z in zerlegung.items()})]
    for _ in range(schritte):
        paare = Counter()
        for wort, anzahl in woerter.items():
            teile = zerlegung[wort]
            for a, b in zip(teile, teile[1:]):
                paare[(a, b)] += anzahl
        (a, b), anzahl = paare.most_common(1)[0]
        for wort, teile in zerlegung.items():
            neu, i = [], 0
            while i < len(teile):
                if i + 1 < len(teile) and (teile[i], teile[i + 1]) == (a, b):
                    neu.append(a + b)
                    i += 2
                else:
                    neu.append(teile[i])
                    i += 1
            zerlegung[wort] = neu
        verlauf.append(((a, b), anzahl, {w: list(z) for w, z in zerlegung.items()}))
    return verlauf


def grafik_bpe(verlauf, woerter):
    zeilen = len(verlauf)
    spalten = [42 + 16 * i for i in range(len(woerter))]
    fig, ax = leere_achse(10, 0.6 * zeilen + 0.9, x=(0, 106), y=(-zeilen + 0.3, 1.1))
    ax.text(0, 0.8, "Lernschritt", color=TEXT_LEISE, fontsize=12, fontweight="bold", va="center")
    for x, wort in zip(spalten, woerter):
        ax.text(x, 0.8, wort, color=TEXT_LEISE, fontsize=12, fontweight="bold", va="center")
    for zeile, (paar, anzahl, zerlegung) in enumerate(verlauf):
        y = -zeile
        if paar is None:
            ax.text(0, y, "Start: einzelne Buchstaben", color=TEXT, fontsize=12.5, va="center")
        else:
            ax.text(0, y, f"Schritt {zeile}: {paar[0]} + {paar[1]} → {paar[0] + paar[1]}  ({anzahl}-mal)", color=TEXT, fontsize=12.5, va="center")
        for x, wort in zip(spalten, woerter):
            teile = zerlegung[wort]
            chips(ax, x, y, teile, abwechselnd(len(teile)), schrift=13.5, hoehe=0.66, abstand=0.95)
    fig.savefig(BILDER / "03-bpe.png")
    plt.close(fig)


def grafik_tokens(enc, saetze):
    fig, ax = leere_achse(10, 0.4 + 1.45 * len(saetze), y=(-1.45 * len(saetze) + 0.2, 0.9))
    for zeile, (titel, text) in enumerate(saetze):
        y = -1.45 * zeile
        teile = tokens(enc, text)
        ax.text(0, y + 0.55, f"{titel}: {len(teile)} Tokens", color=TEXT, fontsize=12.5, fontweight="bold", va="center")
        chips(ax, 0, y, [sichtbar(t) for _, t in teile], abwechselnd(len(teile)), unten=[i for i, _ in teile])
    ax.text(0, -1.45 * len(saetze) + 0.45, "␣ steht für ein Leerzeichen. Darunter die Nummer des Tokens im Vokabular.", color=TEXT_LEISE, fontsize=10.5, va="center")
    fig.savefig(BILDER / "04-tokens.png")
    plt.close(fig)


def grafik_deutsch_englisch(kodierungen):
    paare = [("Bestellung", BEISPIELE[0][1], BEISPIELE[1][1]), ("Satz aus Kapitel 1", BEISPIELE[2][1], BEISPIELE[3][1])]
    fig, achsen = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (name, enc) in zip(achsen, kodierungen.items()):
        x = np.arange(len(paare))
        for versatz, spalte, farbe, sprache in ((-0.18, 1, BLAU, "Deutsch"), (0.18, 2, ORANGE, "Englisch")):
            werte = [len(tokens(enc, paar[spalte])) for paar in paare]
            ax.bar(x + versatz, werte, width=0.32, color=farbe, label=sprache)
            for xi, wert in zip(x + versatz, werte):
                ax.text(xi, wert + 0.4, str(wert), ha="center", va="bottom", color=TEXT, fontsize=11.5)
        ax.set_xticks(x, [paar[0] for paar in paare])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 23)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{name} ({TOKENIZER[name].replace('gpt-4o', 'GPT-4o').replace('gpt-4', 'GPT-4')})", fontsize=12.5)
    achsen[0].legend(loc="upper left", frameon=False, labelcolor=TEXT_LEISE)
    fig.tight_layout(w_pad=3)
    fig.savefig(BILDER / "05-deutsch-englisch.png")
    plt.close(fig)


def grafik_embeddings():
    # Schematisch: Die Positionen sind ausgedacht, echte Embeddings haben tausende Dimensionen
    gruppen = [
        ("Eissorten", BLAU, [("Vanille", 2.3, 7.5), ("Schokolade", 0.4, 8.3), ("Erdbeere", 2.4, 9.1), ("Stracciatella", 0.8, 6.7)]),
        ("Gefäße", ORANGE, [("Becher", 7.0, 7.9), ("Waffel", 8.3, 8.7), ("Hörnchen", 8.2, 7.0)]),
        ("Wetter", AQUA, [("Sonne", 7.3, 2.6), ("Hitze", 8.5, 3.3), ("Regen", 6.6, 1.5), ("Gewitter", 8.4, 1.6)]),
        ("Wochentage", GELB, [("Montag", 1.6, 2.8), ("Samstag", 2.9, 1.8), ("Sonntag", 1.2, 1.4)]),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, farbe, woerter in gruppen:
        xs, ys = np.array([w[1] for w in woerter]), np.array([w[2] for w in woerter])
        ax.add_patch(plt.Circle((xs.mean(), ys.mean()), 1.75, color=farbe, alpha=0.09, linewidth=0))
        ax.text(xs.mean(), ys.mean() + 1.95, name, color=TEXT, fontsize=12, fontweight="bold", ha="center")
        ax.scatter(xs, ys, s=70, color=farbe, edgecolors=FLAECHE, linewidths=1.5, zorder=3)
        for wort, x, y in woerter:
            ax.text(x + 0.18, y - 0.05, wort, color=TEXT, fontsize=11, va="center")
    ax.annotate("", (7.2, 5.4), (2.4, 5.4), arrowprops={"arrowstyle": "<->", "color": TEXT_LEISE, "linewidth": 1.2})
    ax.text(4.8, 5.15, "großer Abstand: wenig verwandt", ha="center", va="top", color=TEXT_LEISE, fontsize=10.5)
    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(-0.4, 11.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for seite in ("left", "bottom"):
        ax.spines[seite].set_visible(False)
    ax.set_aspect("equal")
    fig.savefig(BILDER / "06-embeddings.png")
    plt.close(fig)


def grafik_attention(enc, text, ziel, gewichte):
    teile = [sichtbar(t) for _, t in tokens(enc, text)]
    fig, ax = leere_achse(11, 3.4, y=(-1.3, 4.6))
    farben = [(ORANGE, 0.4) if i == ziel else (BLAU, 0.10 + 0.6 * gewichte.get(i, 0)) for i in range(len(teile))]
    mitten, _ = chips(ax, 0, 0, teile, farben, schrift=13, hoehe=0.7)
    for quelle, gewicht in gewichte.items():
        ax.add_patch(
            FancyArrowPatch((mitten[ziel], 0.45), (mitten[quelle], 0.45), connectionstyle="arc3,rad=0.45", arrowstyle="-|>", mutation_scale=10,
                            color=BLAU, alpha=0.85, linewidth=0.6 + 12 * gewicht, zorder=2)
        )
    ax.text(0, -1.0, f"Wohin schaut „{teile[ziel]}“? Je dicker der Bogen, desto mehr Gewicht. Die Gewichte sind ausgedacht.", color=TEXT_LEISE, fontsize=10.5, va="center")
    fig.savefig(BILDER / "07-attention.png")
    plt.close(fig)


def kasten(ax, x, y, breite, hoehe, titel, unterzeile="", farbe=FLAECHE, rand=ACHSE, textfarbe=TEXT, schrift=12):
    ax.add_patch(FancyBboxPatch((x, y), breite, hoehe, boxstyle="round,pad=0,rounding_size=1.2", facecolor=farbe, edgecolor=rand, linewidth=1.4, zorder=2))
    mitte = x + breite / 2
    if unterzeile:
        ax.text(mitte, y + hoehe * 0.64, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)
        ax.text(mitte, y + hoehe * 0.3, unterzeile, ha="center", va="center", color=TEXT_LEISE if textfarbe == TEXT else textfarbe, fontsize=9.5, zorder=3, linespacing=1.3)
    else:
        ax.text(mitte, y + hoehe / 2, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)


def pfeil_rechts(ax, x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0})


def grafik_transformer():
    fig, ax = leere_achse(12, 3.6, x=(0, 124), y=(0, 36))
    stationen = [
        (0, "Text", "„Zwei Kugeln\nSchokolade im“"),
        (17, "Tokenizer", "Tokens und\nihre Nummern"),
        (34, "Embeddings", "eine Zahlenliste\npro Token"),
    ]
    for x, titel, unter in stationen:
        kasten(ax, x, 11, 14, 14, titel, unter)
    for x in (14, 31, 48):
        pfeil_rechts(ax, x + 0.4, x + 2.6, 18)
    # Der Block, der viele Male wiederholt wird
    ax.add_patch(FancyBboxPatch((51, 2), 30, 32, boxstyle="round,pad=0,rounding_size=1.6", facecolor=BLAU, alpha=0.08, edgecolor=BLAU, linewidth=1.4, zorder=1))
    ax.text(66, 31, "Block, viele Male hintereinander", ha="center", va="center", color=TEXT, fontsize=11, fontweight="bold")
    kasten(ax, 54, 17, 24, 11, "Attention", "Tokens tauschen\nInformationen aus", farbe=BLAU, rand=BLAU, textfarbe="#ffffff")
    kasten(ax, 54, 4, 24, 11, "Neuronen mit Knick", "jedes Token für sich,\nwie in Kapitel 1", farbe=BLAU, rand=BLAU, textfarbe="#ffffff")
    ax.annotate("", (66, 15.2), (66, 16.8), arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0})
    pfeil_rechts(ax, 81.4, 83.6, 18)
    kasten(ax, 84, 11, 17, 14, "Ausgabe", "Punktzahl je Token,\ndann Softmax")
    pfeil_rechts(ax, 101.4, 103.6, 18)
    kasten(ax, 104, 11, 19.5, 14, "Nächstes Token", "zum Beispiel\n␣Be (für Becher)", farbe=ORANGE, rand=ORANGE, textfarbe="#ffffff", schrift=11.5)
    fig.savefig(BILDER / "08-transformer.png")
    plt.close(fig)


def grafik_skalierung():
    namen = [name for name, _ in MODELLE]
    werte = np.array([wert for _, wert in MODELLE])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(MODELLE))[::-1]
    farben = [ORANGE if wert < 1000 else BLAU for wert in werte]
    ax.barh(y, werte, height=0.55, color=farben)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e13)
    for yi, wert in zip(y, werte):
        text = f"{wert:.0f}" if wert < 1000 else f"{wert / 1e9:.0f} Milliarden"
        ax.text(wert * 1.6, yi, text, va="center", color=TEXT, fontsize=11)
    ax.set_yticks(y, namen)
    ax.tick_params(axis="y", length=0)
    stufen = [1, 1e3, 1e6, 1e9, 1e12]
    ax.set_xticks(stufen, ["1", "1000", "1 Million", "1 Milliarde", "1 Billion"])
    ax.minorticks_off()
    ax.grid(axis="x", color=GITTER, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel("Zahl der Parameter (jeder Teilstrich: 1000-mal so viele)")
    fig.savefig(BILDER / "09-skalierung.png")
    plt.close(fig)


def grafik_erzeugen(enc, text, anfang):
    teile = [sichtbar(t) for _, t in tokens(enc, text)] + ["‹Ende›"]
    schritte = len(teile) - anfang
    fig, ax = leere_achse(11, 0.62 * schritte + 0.7, y=(-schritte - 0.1, 0.9))
    for schritt in range(1, schritte + 1):
        y = -schritt + 1
        anzahl = anfang + schritt
        farben = [(BLAU, 0.16)] * (anzahl - 1) + [(GRAU, 0.35) if teile[anzahl - 1] == "‹Ende›" else (ORANGE, 0.35)]
        ax.text(0, y, f"Durchlauf {schritt}", color=TEXT_LEISE, fontsize=11, va="center")
        chips(ax, 14, y, teile[:anzahl], farben, schrift=12.5)
    ax.text(0, -schritte + 0.1, "Blau: Text bis hierher. Orange: das gewählte nächste Token, es wird angehängt.", color=TEXT_LEISE, fontsize=10.5, va="center")
    fig.savefig(BILDER / "10-erzeugen.png")
    plt.close(fig)


def grafik_temperatur(punkte):
    temperaturen = (0.5, 1.0, 2.0)
    fig, achsen = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    namen = [sichtbar(t) for t in punkte]
    y = np.arange(len(namen))[::-1]
    for ax, temperatur in zip(achsen, temperaturen):
        anteile = softmax(punkte, temperatur)
        ax.barh(y, anteile, height=0.55, color=BLAU)
        for yi, anteil in zip(y, anteile):
            ax.text(anteil + 0.02, yi, f"{anteil:.0%}".replace(".", ","), va="center", color=TEXT, fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Temperatur {temperatur:g}".replace(".", ","), fontsize=12.5)
    achsen[0].set_yticks(y, namen)
    fig.tight_layout(w_pad=2)
    fig.savefig(BILDER / "11-temperatur.png")
    plt.close(fig)


def grafik_assistent():
    fig, ax = leere_achse(12, 4.6, x=(0, 120), y=(0, 46))
    stufen = [
        (0, "1. Vortraining", "Billionen Tokens Text", "„Wie viel Eis brauche ich bei 30 °C?\nUnd bei 35 °C?“", "setzt den Text einfach fort"),
        (41, "2. Feinabstimmung", "Beispielgespräche", "„Bei 25 °C etwa 290 Kugeln.“", "beantwortet die Frage"),
        (82, "3. Menschliches Feedback", "Bewertungen von Antworten", "„An einem Werktag etwa 270 Kugeln,\nam Wochenende etwa 320.“", "hilfreicher und genauer"),
    ]
    ax.text(0, 43.5, "Frage an das Modell: „Wie viel Eis brauche ich bei 25 °C?“", color=TEXT, fontsize=12, fontweight="bold", va="center")
    for x, titel, daten, antwort, wirkung in stufen:
        kasten(ax, x, 26, 37, 13, titel, daten, farbe=BLAU, rand=BLAU, textfarbe="#ffffff", schrift=12.5)
        ax.add_patch(FancyBboxPatch((x, 2), 37, 20, boxstyle="round,pad=0,rounding_size=1.2", facecolor=FLAECHE, edgecolor=ACHSE, linewidth=1.2, zorder=2))
        ax.text(x + 18.5, 14.5, antwort, ha="center", va="center", color=TEXT, fontsize=11, zorder=3, linespacing=1.35)
        ax.text(x + 18.5, 6, wirkung, ha="center", va="center", color=TEXT_LEISE, fontsize=10.5, style="italic", zorder=3)
        ax.annotate("", (x + 18.5, 22.3), (x + 18.5, 25.7), arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0})
    for x in (37, 78):
        pfeil_rechts(ax, x + 0.5, x + 3.5, 32.5)
    fig.savefig(BILDER / "12-assistent.png")
    plt.close(fig)


def app_daten(kodierungen):
    daten = {
        "tokenizer": {name: {"modell": TOKENIZER[name], "vokabular": enc.n_vocab} for name, enc in kodierungen.items()},
        "beispiele": [
            {"titel": titel, "text": text, "hinweis": hinweis, **{name: tokens(enc, text) for name, enc in kodierungen.items()}}
            for titel, text, hinweis in BEISPIELE
        ],
        "punktzahlen": PUNKTZAHLEN,
    }
    APP_DATEN.write_text(
        "// Erzeugt von docs/beispiele/02_tokenizer.py, bitte nicht von Hand ändern.\n"
        "window.TOKEN_BEISPIELE = " + json.dumps(daten, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )


def main():
    kodierungen = {name: tiktoken.get_encoding(name) for name in TOKENIZER}

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Text: {text} ({len(text)} Zeichen)")
        for enc in kodierungen.values():
            zeige(enc, text)
        return

    BILDER.mkdir(parents=True, exist_ok=True)

    print("Abschnitt 1: Der Fehler beim nächsten Token (Kreuzentropie)")
    for p in (1.0, 0.5, 0.25, 0.1, 0.01):
        print(f"   Wahrscheinlichkeit für das richtige Token {p:5.0%}: Fehler {abs(math.log(p)):.2f}")

    print("\nAbschnitt 2: Ein Mini-Tokenizer lernt an Eisbestellungen")
    verlauf = bpe_lernen(KORPUS, 8)
    for schritt, (paar, anzahl, zerlegung) in enumerate(verlauf):
        name = "Start" if paar is None else f"Schritt {schritt}: {paar[0]} + {paar[1]} -> {paar[0] + paar[1]} ({anzahl}-mal)"
        print(f"   {name:36s} " + "  ".join("·".join(zerlegung[w]) for w in ("kugeln", "kugel", "schoko", "vanille", "becher")))

    print("\nAbschnitt 3: Welcher Tokenizer gehört zu welchem Modell")
    for modell in ("gpt-4o", "gpt-4", "gpt-3.5-turbo"):
        print(f"   {modell}: {tiktoken.encoding_for_model(modell).name}")
    for name, enc in kodierungen.items():
        print(f"   {name}: {enc.n_vocab} Tokens im Vokabular")

    print("\n   Beispiele")
    for titel, text, _ in BEISPIELE:
        print(f"\n   {titel}: {text}")
        for enc in kodierungen.values():
            zeige(enc, text)

    print("\n   Einzelne Wörter (o200k_base)")
    o200k = kodierungen["o200k_base"]
    for wort in ("Eis", " Eis", "Kugeln", " Kugeln", " Schokolade", " Vanille", " verkauft", " bitte", "Erdbeere"):
        print(f"   {sichtbar(wort):13s} " + " | ".join(sichtbar(t) for _, t in tokens(o200k, wort)))

    print("\n   Deutsch gegen Englisch")
    for de, en in ((BEISPIELE[0][1], BEISPIELE[1][1]), (BEISPIELE[2][1], BEISPIELE[3][1])):
        for name, enc in kodierungen.items():
            print(f"   {name}: Deutsch {len(tokens(enc, de))} ({len(de)} Zeichen), Englisch {len(tokens(enc, en))} ({len(en)} Zeichen)")

    print("\nAbschnitt 8: Temperatur (ausgedachte Punktzahlen nach „Zwei Kugeln Schokolade im“)")
    print("   Token          Punkte    " + "  ".join(f"T = {t:<4g}" for t in (0.5, 1, 2)))
    tabelle = np.column_stack([softmax(PUNKTZAHLEN, t) for t in (0.5, 1, 2)])
    for (token, punkte), zeile in zip(PUNKTZAHLEN.items(), tabelle):
        print(f"   {sichtbar(token):13s} {punkte:6.1f}    " + "  ".join(f"{p:7.1%}" for p in zeile))

    grafik_trainingsaufgaben(["Zwei", "Kugeln", "Schokolade", "im", "Becher", ",", "bitte", "."])
    grafik_kreuzentropie()
    grafik_bpe(verlauf, ["kugeln", "schoko", "becher", "vanille"])
    grafik_tokens(o200k, [("Deutsch", BEISPIELE[0][1]), ("Englisch", BEISPIELE[1][1])])
    grafik_deutsch_englisch(kodierungen)
    grafik_embeddings()
    # „elbe“ in „dasselbe“ schaut vor allem auf „Erdbeere“ zurück (ausgedachte Gewichte, Summe 1)
    grafik_attention(o200k, BEISPIELE[8][1], 11, {0: 0.02, 1: 0.10, 2: 0.26, 3: 0.18, 4: 0.22, 5: 0.02, 6: 0.03, 7: 0.03, 8: 0.06, 9: 0.03, 10: 0.05})
    grafik_transformer()
    grafik_skalierung()
    grafik_erzeugen(o200k, BEISPIELE[0][1], 7)
    grafik_temperatur(PUNKTZAHLEN)
    grafik_assistent()
    app_daten(kodierungen)
    print(f"\nGrafiken gespeichert in {BILDER}")
    print(f"Daten für die interaktive Seite gespeichert in {APP_DATEN}")


if __name__ == "__main__":
    main()

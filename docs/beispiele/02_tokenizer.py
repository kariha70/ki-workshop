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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib.patches import FancyBboxPatch

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
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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
    for token_id in enc.encode(text):
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


def grafik_tokens(enc, saetze):
    fig, ax = plt.subplots(figsize=(10, 0.4 + 1.45 * len(saetze)))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.45 * len(saetze) + 0.2, 0.9)
    ax.axis("off")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    in_daten = ax.transData.inverted()

    for zeile, (titel, text) in enumerate(saetze):
        y = -1.45 * zeile
        teile = tokens(enc, text)
        ax.text(0, y + 0.55, f"{titel}: {len(teile)} Tokens", color=TEXT, fontsize=12.5, fontweight="bold", va="center")
        x = 0.0
        for nummer, (token_id, token) in enumerate(teile):
            beschriftung = ax.text(x, y, sichtbar(token), color=TEXT, fontsize=13, va="center", ha="left", zorder=3)
            box = beschriftung.get_window_extent(renderer)
            (x0, _), (x1, _) = in_daten.transform([(box.x0, box.y0), (box.x1, box.y1)])
            breite = x1 - x0
            id_text = ax.text(x + breite / 2, y - 0.5, str(token_id), color=TEXT_LEISE, fontsize=8.5, va="center", ha="center")
            id_box = id_text.get_window_extent(renderer)
            (i0, _), (i1, _) = in_daten.transform([(id_box.x0, id_box.y0), (id_box.x1, id_box.y1)])
            kasten = max(breite, i1 - i0)
            beschriftung.set_x(x + (kasten - breite) / 2)
            id_text.set_x(x + kasten / 2)
            farbe = BLAU if nummer % 2 == 0 else ORANGE
            ax.add_patch(
                FancyBboxPatch((x - 0.35, y - 0.28), kasten + 0.7, 0.56, boxstyle="round,pad=0,rounding_size=0.12", facecolor=farbe, alpha=0.16, edgecolor="none", zorder=1)
            )
            x += kasten + 1.1

    ax.text(0, -1.45 * len(saetze) + 0.45, "␣ steht für ein Leerzeichen. Darunter die Nummer des Tokens im Vokabular.", color=TEXT_LEISE, fontsize=10.5, va="center")
    fig.savefig(BILDER / "01-tokens.png")
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
    fig.savefig(BILDER / "02-temperatur.png")
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
        print(f"   Wahrscheinlichkeit für das richtige Token {p:5.0%}: Fehler {-math.log(p):.2f}")

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
            print(f"   {name}: Deutsch {len(enc.encode(de))} ({len(de)} Zeichen), Englisch {len(enc.encode(en))} ({len(en)} Zeichen)")

    print("\nAbschnitt 8: Temperatur (ausgedachte Punktzahlen nach „Zwei Kugeln Schokolade im“)")
    print("   Token          Punkte    " + "  ".join(f"T = {t:<4g}" for t in (0.5, 1, 2)))
    tabelle = np.column_stack([softmax(PUNKTZAHLEN, t) for t in (0.5, 1, 2)])
    for (token, punkte), zeile in zip(PUNKTZAHLEN.items(), tabelle):
        print(f"   {sichtbar(token):13s} {punkte:6.1f}    " + "  ".join(f"{p:7.1%}" for p in zeile))

    grafik_tokens(o200k, [("Deutsch", BEISPIELE[0][1]), ("Englisch", BEISPIELE[1][1])])
    grafik_temperatur(PUNKTZAHLEN)
    app_daten(kodierungen)
    print(f"\nGrafiken gespeichert in {BILDER}")
    print(f"Daten für die interaktive Seite gespeichert in {APP_DATEN}")


if __name__ == "__main__":
    main()

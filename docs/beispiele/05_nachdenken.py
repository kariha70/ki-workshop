# /// script
# requires-python = ">=3.10"
# dependencies = ["tiktoken", "numpy", "matplotlib"]
# ///
"""Vom Sprachmodell zum Reasoning-Modell: die Beispiele aus Kapitel 5.

Start: uv run docs/beispiele/05_nachdenken.py

Rechnet die Aufgabe des Eisverkäufers aus 05-Reasoning-Modelle.md nach, zählt die Tokens
der Antworten mit und ohne Rechenweg und spielt drei kleine Simulationen durch: ein
Sprachmodell, das bei jedem Rechenschritt mit einer bestimmten Wahrscheinlichkeit einen
Fehler macht, eine Abstimmung über mehrere Antworten und ein Spielzeug-Training mit
Belohnung für richtige Ergebnisse. Erzeugt die Grafiken in docs/bilder/kapitel-5/ und die
Daten für docs/app/labor-5.html.

Die Fehlerwahrscheinlichkeiten in den Simulationen sind ausgedacht. Sie zeigen das
Prinzip, nicht das Verhalten eines bestimmten Modells. Beim ersten Start lädt tiktoken
das Vokabular o200k_base aus dem Internet und legt es im Cache ab.
"""

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib.patches import FancyBboxPatch

DOCS = Path(__file__).resolve().parent.parent
BILDER = DOCS / "bilder" / "kapitel-5"
APP_DATEN = DOCS / "app" / "denk-daten.js"

# ---------- Die Aufgabe ----------

# Der Wetterbericht für das lange Wochenende: (Tag, Temperatur in °C, Wochenende ja = 1 / nein = 0)
WOCHENENDE = [("Freitag", 24, 0), ("Samstag", 27, 1), ("Sonntag", 25, 1)]
KUGELN_PRO_BEHAELTER = 70
VORRAT = 4

FRAGE = (
    "Für das lange Wochenende sind angesagt: Freitag 24 °C, Samstag 27 °C, Sonntag 25 °C. "
    "In einen Behälter passen 70 Kugeln, im Gefrierschrank stehen noch 4 volle Behälter. "
    "Wie viele Behälter muss der Eisverkäufer nachbestellen?"
)


def prognose(temperatur, wochenende):
    """Das Modell aus Kapitel 1, Abschnitt 6: Kugeln = 10 · Temperatur + 50 · Wochenende + 20."""
    return 10 * temperatur + 50 * wochenende + 20


# Die sechs Rechenschritte und der typische Fehler, der in jedem Schritt passieren kann
SCHRITTE = [
    ("freitag", "Bedarf Freitag", "vergisst die 20 Kugeln Grundwert"),
    ("samstag", "Bedarf Samstag", "vergisst, dass Samstag Wochenende ist"),
    ("sonntag", "Bedarf Sonntag", "vergisst, dass Sonntag Wochenende ist"),
    ("summe", "Summe", "verrechnet sich beim Addieren um 100"),
    ("behaelter", "Behälter", "rundet ab statt auf"),
    ("vorrat", "Vorrat abziehen", "vergisst den Vorrat im Gefrierschrank"),
]


def rechnen(fehler=frozenset()):
    """Rechnet die Aufgabe Schritt für Schritt. fehler: Namen der Schritte, in denen etwas schiefgeht.

    Gibt die Zwischenergebnisse in der Reihenfolge von SCHRITTE zurück, das letzte ist die Antwort.
    """
    werte = []
    for tag, temperatur, wochenende in WOCHENENDE:
        name = tag.lower()
        if name in fehler:
            kugeln = prognose(temperatur, 0) if wochenende else prognose(temperatur, wochenende) - 20
        else:
            kugeln = prognose(temperatur, wochenende)
        werte.append(kugeln)
    summe = sum(werte) - (100 if "summe" in fehler else 0)
    behaelter = summe // KUGELN_PRO_BEHAELTER if "behaelter" in fehler else math.ceil(summe / KUGELN_PRO_BEHAELTER)
    bestellen = behaelter if "vorrat" in fehler else behaelter - VORRAT
    return werte + [summe, behaelter, bestellen]


RICHTIG = rechnen()[-1]

# ---------- Drei Antworten auf dieselbe Frage ----------

KURZANTWORT = "Er muss 10 Behälter nachbestellen."

# Der Rechenweg, Zeile für Zeile, in der Reihenfolge von SCHRITTE
RECHENWEG = [
    "Freitag ist ein Werktag: 10 · 24 + 20 = 260 Kugeln.",
    "Samstag ist Wochenende: 10 · 27 + 50 + 20 = 340 Kugeln.",
    "Sonntag ist auch Wochenende: 10 · 25 + 50 + 20 = 320 Kugeln.",
    "Zusammen: 260 + 340 + 320 = 920 Kugeln.",
    "920 : 70 = 13,1, also werden 14 Behälter gebraucht.",
    "4 Behälter sind noch da: 14 − 4 = 10.",
    "Er muss 10 Behälter nachbestellen.",
]

# So ähnlich sieht das Denkprotokoll eines Reasoning-Modells aus (ausgedachter Text):
# Es plant, verrechnet sich, bemerkt den Fehler und macht zum Schluss eine Probe.
NACHDENKEN = [
    "Gefragt sind Behälter, nicht Kugeln. Ich rechne erst den Bedarf für jeden Tag aus.",
    "Freitag, 24 °C, Werktag: 10 · 24 + 20 = 260 Kugeln.",
    "Samstag, 27 °C, Wochenende: 10 · 27 + 50 + 20 = 340 Kugeln.",
    "Sonntag, 25 °C: 10 · 25 + 20 = 270 Kugeln.",
    "Moment, der Sonntag gehört auch zum Wochenende. Also 10 · 25 + 50 + 20 = 320 Kugeln.",
    "Zusammen: 260 + 340 + 320 = 920 Kugeln. Probe: 260 + 340 = 600, dazu 320 ergibt 920.",
    "920 : 70 = 13,14. 13 Behälter reichen nur für 910 Kugeln, das sind 10 zu wenig. Also 14 Behälter.",
    "Im Gefrierschrank stehen 4: 14 − 4 = 10.",
    "Gegenprobe: 10 neue und 4 alte Behälter fassen 14 · 70 = 980 Kugeln, das reicht für 920.",
    "Antwort: 10 Behälter.",
]

# Was das Reasoning-Modell insgesamt erzeugt: erst das Denkprotokoll, dann die eigentliche Antwort
MIT_DENKPROTOKOLL = "\n".join(NACHDENKEN) + "\n\n" + KURZANTWORT

# ---------- Simulation: ein Sprachmodell, das sich verrechnen kann ----------

# Ausgedachte Wahrscheinlichkeit für einen Fehler pro Rechenschritt
FEHLER_KURZ = 0.25  # alles „im Kopf“, die Antwort kommt sofort
FEHLER_RECHENWEG = 0.08  # jedes Zwischenergebnis steht im Text

# Für das Spielzeug-Training (Abschnitt 6): Wer einen Schritt nachprüft, findet einen Fehler darin mit dieser Wahrscheinlichkeit
FINDEN = 0.8
TRAINING_RUNDEN = 150
TRAINING_VERSUCHE = 32  # Antworten pro Runde
TRAINING_LERNRATE = 1.0
TRAINING_START = -2.5  # Startwert der Stellschraube, entspricht einer Prüfquote von etwa 8 %

STIMMEN = list(range(1, 26, 2))
WIEDERHOLUNGEN = 20000
SEED = 5

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
        "font.family": ["Helvetica Neue", "Arial Unicode MS", "Nimbus Sans", "DejaVu Sans"],
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


def prozent(p, stellen=0):
    return f"{p * 100:.{stellen}f} %".replace(".", ",")


def alle_fehlerkombinationen():
    """Alle 64 Möglichkeiten, welche der sechs Schritte schiefgehen, mit der Antwort, die dabei herauskommt."""
    namen = [name for name, _, _ in SCHRITTE]
    for maske in itertools.product((0, 1), repeat=len(namen)):
        fehler = frozenset(n for n, f in zip(namen, maske) if f)
        yield maske, rechnen(fehler)


def verteilung(p_fehler):
    """Genaue Wahrscheinlichkeit jeder Antwort, wenn jeder Schritt unabhängig mit p_fehler schiefgeht."""
    ergebnis = Counter()
    for maske, werte in alle_fehlerkombinationen():
        k = sum(maske)
        ergebnis[werte[-1]] += p_fehler**k * (1 - p_fehler) ** (len(maske) - k)
    return dict(sorted(ergebnis.items()))


def abstimmung(p_fehler, stimmen, rng, wiederholungen=WIEDERHOLUNGEN):
    """Anteil der Abstimmungen, bei denen die richtige Antwort die meisten Stimmen bekommt.

    Bei Gleichstand wird ausgelost, wie bei der Temperatur in Kapitel 2.
    """
    antworten, anteile = zip(*verteilung(p_fehler).items())
    antworten = np.array(antworten)
    ziehungen = rng.choice(len(antworten), size=(wiederholungen, stimmen), p=np.array(anteile))
    zaehler = np.zeros((wiederholungen, len(antworten)))
    for spalte in range(stimmen):
        zaehler[np.arange(wiederholungen), ziehungen[:, spalte]] += 1
    zaehler += rng.random(zaehler.shape) * 0.1  # Gleichstand zufällig auflösen
    gewinner = antworten[zaehler.argmax(axis=1)]
    return float(np.mean(gewinner == RICHTIG))


def tokens(enc, text):
    ergebnis = []
    for token_id in enc.encode(text):
        try:
            stueck = enc.decode_single_token_bytes(token_id).decode("utf-8")
        except UnicodeDecodeError:
            stueck = enc.decode_single_token_bytes(token_id).decode("utf-8", errors="replace")
        ergebnis.append([token_id, stueck])
    return ergebnis


def training(rng, schritt_tokens, pruef_tokens):
    """Spielzeug-Training mit Belohnung für richtige Ergebnisse.

    Das „Modell“ hat eine einzige Stellschraube: wie oft es einen Rechenschritt nachprüft.
    Nachprüfen kostet Tokens, findet aber Fehler. Belohnt wird nur das richtige Endergebnis.
    Niemand sagt dem Modell, dass Nachprüfen hilft. Es merkt es an den Belohnungen.
    """
    theta = TRAINING_START
    verlauf = []
    for _ in range(TRAINING_RUNDEN + 1):
        q = 1 / (1 + math.exp(-theta))
        pruefen = rng.random((TRAINING_VERSUCHE, len(SCHRITTE))) < q
        fehler = rng.random((TRAINING_VERSUCHE, len(SCHRITTE))) < FEHLER_RECHENWEG
        gefunden = pruefen & (rng.random((TRAINING_VERSUCHE, len(SCHRITTE))) < FINDEN)
        bleibt = fehler & ~gefunden
        belohnung = np.array([rechnen(frozenset(SCHRITTE[i][0] for i in np.flatnonzero(zeile)))[-1] == RICHTIG for zeile in bleibt], dtype=float)
        laenge = (len(SCHRITTE) * schritt_tokens + pruefen.sum(axis=1) * pruef_tokens).mean()
        # Genau erwartete Werte für die Grafik, damit die Kurve nicht vom Zufall der 32 Versuche abhängt
        erfolg_schritt = 1 - FEHLER_RECHENWEG * (1 - q * FINDEN)
        verlauf.append({"pruefquote": q, "treffer": erfolg_schritt ** len(SCHRITTE), "laenge": len(SCHRITTE) * (schritt_tokens + q * pruef_tokens), "laenge_stichprobe": laenge})
        # Verstärkungslernen (REINFORCE): Was in überdurchschnittlich belohnten Antworten vorkam, wird wahrscheinlicher
        vorteil = belohnung - belohnung.mean()
        gradient = ((pruefen - q).sum(axis=1) * vorteil).mean()
        theta += TRAINING_LERNRATE * gradient
    return verlauf


def stelle_der_zahl(enc):
    """An welcher Stelle der Kurzantwort das Token mit der Zahl steht (ab 1 gezählt)."""
    stuecke = [t for _, t in tokens(enc, KURZANTWORT)]
    return next(i + 1 for i, t in enumerate(stuecke) if str(RICHTIG) in t)


# ---------- Grafiken ----------


def leere_achse(breite, hoehe, x=(0, 100), y=(0, 10)):
    fig, ax = plt.subplots(figsize=(breite, hoehe))
    ax.set_xlim(*x)
    ax.set_ylim(*y)
    ax.axis("off")
    fig.canvas.draw()
    return fig, ax


def kasten(ax, x, y, breite, hoehe, titel, unterzeile="", farbe=FLAECHE, rand=ACHSE, textfarbe=TEXT, schrift=12):
    ax.add_patch(FancyBboxPatch((x, y), breite, hoehe, boxstyle="round,pad=0,rounding_size=1.2", facecolor=farbe, edgecolor=rand, linewidth=1.4, zorder=2))
    mitte = x + breite / 2
    if unterzeile:
        ax.text(mitte, y + hoehe * 0.68, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)
        ax.text(mitte, y + hoehe * 0.32, unterzeile, ha="center", va="center", color=TEXT_LEISE if textfarbe == TEXT else textfarbe, fontsize=9.5, zorder=3, linespacing=1.3)
    else:
        ax.text(mitte, y + hoehe / 2, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)


def pfeil_rechts(ax, x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.4, "shrinkA": 0, "shrinkB": 0})


def grafik_zeitleiste():
    # (Zeitpunkt, Position der Beschriftung, Höhe der Beschriftung, Datum, Name, Beschreibung)
    ereignisse = [
        (2017.45, 2017.45, 1.0, "Juni 2017", "Transformer", "die Bauweise heutiger\nSprachmodelle"),
        (2019.12, 2019.12, -1.0, "Feb. 2019", "GPT-2", "setzt Texte fort"),
        (2020.4, 2020.4, 1.0, "Mai 2020", "GPT-3", "lernt aus Beispielen\nim Prompt"),
        (2022.07, 2021.75, -1.0, "2022", "Chain of Thought", "„Schritt für Schritt“\nim Prompt"),
        (2022.92, 2022.92, 1.0, "Nov. 2022", "ChatGPT", "Assistent durch\nmenschliches Feedback"),
        (2024.7, 2024.4, 1.0, "Sept. 2024", "OpenAI o1", "erstes\nReasoning-Modell"),
        (2025.05, 2025.05, -1.0, "Jan. 2025", "DeepSeek-R1", "Trainingsweg offen\nbeschrieben"),
        (2025.15, 2025.9, 1.0, "Feb. 2025", "Claude 3.7 Sonnet", "Denkzeit\neinstellbar"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(2016.6, 2026.7)
    ax.set_ylim(-3.0, 2.9)
    ax.axis("off")
    ax.plot([2016.8, 2025.9], [0, 0], color=ACHSE, linewidth=2, zorder=1)
    for jahr in range(2017, 2026):
        ax.text(jahr, -0.18, str(jahr), ha="center", va="top", color=GRAU, fontsize=9.5)
        ax.plot([jahr, jahr], [-0.08, 0.08], color=ACHSE, linewidth=1.2)
    for x, x_text, h, datum, name, text in ereignisse:
        farbe = ORANGE if x >= 2024 else BLAU
        oben = h > 0
        ax.plot([x, x_text], [0, h - (0.05 if oben else -0.05)], color=farbe, linewidth=1.2, zorder=1)
        ax.scatter([x], [0], s=60, color=farbe, zorder=3)
        if oben:
            beschreibung = ax.text(x_text, h, f"{datum}: {text}", ha="center", va="bottom", color=TEXT_LEISE, fontsize=9.2, linespacing=1.25)
            ax.text(x_text, h + 0.03 + 0.33 * (text.count(chr(10)) + 1), name, ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
        else:
            ax.text(x_text, h, name, ha="center", va="top", color=TEXT, fontsize=11, fontweight="bold")
            ax.text(x_text, h - 0.4, f"{datum}: {text}", ha="center", va="top", color=TEXT_LEISE, fontsize=9.2, linespacing=1.25)
    ax.text(2016.8, -2.85, "Blau: Sprachmodelle und Assistenten. Orange: Reasoning-Modelle, die vor der Antwort nachdenken.", color=TEXT_LEISE, fontsize=10)
    fig.savefig(BILDER / "01-zeitleiste.png")
    plt.close(fig)


def grafik_rechenweg(enc):
    fig, ax = leere_achse(12, 4.5, x=(0, 124), y=(0, 46))
    ax.text(0, 43.5, "Ohne Rechenweg: Die Antwort muss sofort kommen", color=TEXT, fontsize=12, fontweight="bold", va="center")
    kasten(ax, 0, 31, 22, 9, "Frage", "Wetter, Behälter,\nVorrat")
    pfeil_rechts(ax, 22.5, 25.5, 35.5)
    n = len(enc.encode(KURZANTWORT))
    kasten(ax, 26, 31, 40, 9, "„Er muss 10 Behälter nachbestellen.“", f"{n} Tokens, die Zahl {RICHTIG} ist das {stelle_der_zahl(enc)}. davon", farbe=ORANGE, rand=ORANGE, textfarbe="#ffffff", schrift=11)

    ax.text(0, 25, "Mit Rechenweg: Jedes Zwischenergebnis steht im Text und kann wieder gelesen werden", color=TEXT, fontsize=12, fontweight="bold", va="center")
    kasten(ax, 0, 10, 14, 11, "Frage", "")
    stuecke = [("Fr", "260"), ("Sa", "340"), ("So", "320"), ("Summe", "920"), ("Behälter", "14"), ("minus Vorrat", "10")]
    x = 16
    breite = 15.4
    mitten = []
    for (name, wert), zeile in zip(stuecke, RECHENWEG):
        pfeil_rechts(ax, x - 1.7, x - 0.3, 15.5)
        letzte = name == "minus Vorrat"
        kasten(ax, x, 10, breite, 11, wert, f"{name}\n{len(enc.encode(zeile))} Tokens", farbe=ORANGE if letzte else BLAU, rand=ORANGE if letzte else BLAU, textfarbe="#ffffff", schrift=13)
        mitten.append(x + breite / 2)
        x += breite + 2
    # Bögen unter den Kästen: die Summe liest die drei Tagesbedarfe wieder
    for m in mitten[:3]:
        ax.annotate("", (mitten[3] - 1.5, 9.8), (m, 9.8), arrowprops={"arrowstyle": "-|>", "color": GRAU, "linewidth": 1, "connectionstyle": "arc3,rad=0.25", "shrinkA": 2, "shrinkB": 2})
    ax.text(0, 1.2, "Graue Bögen: Für die Summe schaut das Modell auf die drei Zahlen zurück, die schon im Text stehen.", color=TEXT_LEISE, fontsize=10, va="center", ha="left")
    fig.savefig(BILDER / "02-rechenweg.png")
    plt.close(fig)


def grafik_antworten(kurz, lang):
    antworten = sorted(set(kurz) | set(lang))
    fig, achsen = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, daten, titel in ((achsen[0], kurz, f"Ohne Rechenweg (Fehler pro Schritt {prozent(FEHLER_KURZ)})"), (achsen[1], lang, f"Mit Rechenweg (Fehler pro Schritt {prozent(FEHLER_RECHENWEG)})")):
        werte = [daten.get(a, 0) for a in antworten]
        farben = [ORANGE if a == RICHTIG else BLAU for a in antworten]
        x = np.arange(len(antworten))
        ax.bar(x, werte, color=farben, width=0.65)
        for xi, w in zip(x, werte):
            if w >= 0.005:
                ax.text(xi, w + 0.015, prozent(w), ha="center", va="bottom", fontsize=9.5, color=TEXT)
        ax.set_xticks(x, [str(a) for a in antworten])
        ax.set_title(titel, fontsize=12)
        ax.set_xlabel("Antwort: so viele Behälter nachbestellen")
        ax.set_ylim(0, 0.75)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.text(0.01, -0.04, f"Orange: die richtige Antwort {RICHTIG}. Die Fehlerwahrscheinlichkeiten sind ausgedacht.", color=TEXT_LEISE, fontsize=10)
    fig.tight_layout(w_pad=2)
    fig.savefig(BILDER / "03-antworten.png")
    plt.close(fig)


def grafik_abstimmen(kurven):
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for (name, werte), farbe in zip(kurven.items(), (BLAU, ORANGE)):
        ax.plot(STIMMEN, werte, color=farbe, linewidth=2.2, marker="o", markersize=4.5)
        ax.text(STIMMEN[-1] + 0.6, werte[-1], f"{name}\n{prozent(werte[-1])}", color=farbe, fontsize=10.5, va="center")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 5, 9, 13, 17, 21, 25])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="y", color=GITTER, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlabel("Zahl der Antworten, über die abgestimmt wird")
    ax.set_ylabel("Richtige Antwort gewinnt")
    fig.savefig(BILDER / "04-abstimmen.png")
    plt.close(fig)


def grafik_stufen():
    fig, ax = leere_achse(12, 3.9, x=(0, 124), y=(4, 44))
    stufen = [
        (0, "1. Vortraining", "Text fortsetzen", "War das nächste\nToken richtig?", BLAU),
        (31.5, "2. Feinabstimmung", "Beispielgespräche", "Passt die Antwort\nzum Vorbild?", BLAU),
        (63, "3. Feedback", "Antworten vergleichen", "Welche Antwort finden\nMenschen besser?", BLAU),
        (94.5, "4. Belohnung", "Aufgaben mit prüfbarer Lösung", "Stimmt das Ergebnis?\nLaufen die Programmtests?", ORANGE),
    ]
    for x, titel, daten, bewertung, farbe in stufen:
        kasten(ax, x, 27, 29, 12, titel, daten, farbe=farbe, rand=farbe, textfarbe="#ffffff", schrift=12)
        ax.add_patch(FancyBboxPatch((x, 9), 29, 14, boxstyle="round,pad=0,rounding_size=1.2", facecolor=FLAECHE, edgecolor=ACHSE, linewidth=1.2, zorder=2))
        ax.text(x + 14.5, 19.8, "Bewertet wird:", ha="center", va="center", color=TEXT_LEISE, fontsize=9.5, zorder=3)
        ax.text(x + 14.5, 14.5, bewertung, ha="center", va="center", color=TEXT, fontsize=10.5, zorder=3, linespacing=1.35)
        ax.annotate("", (x + 14.5, 23.3), (x + 14.5, 26.7), arrowprops={"arrowstyle": "-|>", "color": TEXT_LEISE, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0})
    for x in (29, 60.5, 92):
        pfeil_rechts(ax, x + 0.3, x + 2.2, 33)
    ax.text(0, 42, "Stufen 1 bis 3 kennst du aus Kapitel 2. Stufe 4 macht aus einem Assistenten ein Reasoning-Modell.", color=TEXT, fontsize=11.5, va="center")
    ax.text(94.5, 6, "Wie lange das Modell vorher nachdenkt, ist frei.", color=TEXT_LEISE, fontsize=9.5, va="center", style="italic")
    fig.savefig(BILDER / "05-stufen.png")
    plt.close(fig)


def grafik_training(verlauf):
    runden = np.arange(len(verlauf))
    treffer = [v["treffer"] for v in verlauf]
    laenge = [v["laenge"] for v in verlauf]
    fig, achsen = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = achsen[0]
    ax.plot(runden, treffer, color=ORANGE, linewidth=2.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_title("Anteil richtiger Antworten", fontsize=12)
    ax2 = achsen[1]
    ax2.plot(runden, laenge, color=BLAU, linewidth=2.2)
    ax2.set_ylim(0, max(laenge) * 1.2)
    ax2.set_title("Länge der Antwort in Tokens", fontsize=12)
    for a, werte, fmt in ((ax, treffer, prozent), (ax2, laenge, lambda w: f"{w:.0f}")):
        a.grid(axis="y", color=GITTER, linewidth=0.75)
        a.set_axisbelow(True)
        a.set_xlabel("Trainingsrunde")
        a.set_xlim(0, len(verlauf) - 1)
        a.annotate(fmt(werte[0]), (0, werte[0]), (6, -14), textcoords="offset points", color=TEXT_LEISE, fontsize=10)
        a.annotate(fmt(werte[-1]), (len(werte) - 1, werte[-1]), (-30, 8), textcoords="offset points", color=TEXT, fontsize=10.5, fontweight="bold")
    fig.text(0.01, -0.04, "Spielzeug-Simulation mit ausgedachten Werten. Belohnt wird nur das richtige Endergebnis.", color=TEXT_LEISE, fontsize=10)
    fig.tight_layout(w_pad=3)
    fig.savefig(BILDER / "06-training.png")
    plt.close(fig)


def grafik_denkaufwand(anzahlen):
    namen = list(anzahlen)
    werte = [anzahlen[n] for n in namen]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    y = np.arange(len(namen))[::-1]
    ax.barh(y, werte, height=0.55, color=[BLAU, BLAU, ORANGE])
    for yi, w in zip(y, werte):
        ax.text(w + max(werte) * 0.015, yi, f"{w} Tokens", va="center", color=TEXT, fontsize=11)
    ax.set_yticks(y, namen)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(werte) * 1.18)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    fig.savefig(BILDER / "07-denkaufwand.png")
    plt.close(fig)


def app_daten(enc, kurz, lang, kombinationen, schritt_tokens, pruef_tokens):
    daten = {
        "frage": FRAGE,
        "richtig": RICHTIG,
        "schritte": [{"name": n, "titel": t, "fehler": f} for n, t, f in SCHRITTE],
        "zeilen": RECHENWEG[:-1],
        # Für jede der 64 Fehlerkombinationen (Bitmuster wie in SCHRITTE) die Zwischenergebnisse
        "kombinationen": {"".join(map(str, m)): w for m, w in kombinationen},
        "fehlerKurz": FEHLER_KURZ,
        "fehlerRechenweg": FEHLER_RECHENWEG,
        "verteilungKurz": kurz,
        "verteilungRechenweg": lang,
        "antworten": {
            "kurz": {"titel": "Kurzantwort", "text": KURZANTWORT, "tokens": tokens(enc, KURZANTWORT)},
            "rechenweg": {"titel": "Mit Rechenweg", "text": "\n".join(RECHENWEG), "tokens": tokens(enc, "\n".join(RECHENWEG))},
            "nachdenken": {"titel": "Denkprotokoll und Antwort", "text": MIT_DENKPROTOKOLL, "tokens": tokens(enc, MIT_DENKPROTOKOLL)},
        },
        "training": {
            "finden": FINDEN, "runden": TRAINING_RUNDEN, "versuche": TRAINING_VERSUCHE, "lernrate": TRAINING_LERNRATE,
            "start": TRAINING_START, "schrittTokens": schritt_tokens, "pruefTokens": pruef_tokens,
        },
    }
    APP_DATEN.write_text(
        "// Erzeugt von docs/beispiele/05_nachdenken.py, bitte nicht von Hand ändern.\n"
        "window.DENK_DATEN = " + json.dumps(daten, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )


def main():
    BILDER.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    enc = tiktoken.get_encoding("o200k_base")

    print("Abschnitt 1: Die Aufgabe")
    print(f"   {FRAGE}")
    werte = rechnen()
    for (tag, temperatur, wochenende), kugeln in zip(WOCHENENDE, werte):
        print(f"   {tag}: 10 · {temperatur} + 50 · {wochenende} + 20 = {kugeln} Kugeln")
    print(f"   Zusammen {werte[3]} Kugeln, {werte[3]} : {KUGELN_PRO_BEHAELTER} = {werte[3] / KUGELN_PRO_BEHAELTER:.2f} -> {werte[4]} Behälter, minus {VORRAT} Vorrat = {werte[5]}")
    print("   Antworten, wenn genau ein Schritt schiefgeht:")
    for name, titel, fehler in SCHRITTE:
        print(f"   {titel:16s} ({fehler}): {rechnen(frozenset({name}))[-1]}")

    print("\nAbschnitt 3 und 7: Tokens der Antworten (o200k_base)")
    n_kurz = len(enc.encode(KURZANTWORT))
    n_rechenweg = len(enc.encode("\n".join(RECHENWEG)))
    n_nachdenken = len(enc.encode("\n".join(NACHDENKEN)))
    n_antwort = len(enc.encode(MIT_DENKPROTOKOLL))
    print(f"   Kurzantwort:  {n_kurz} Tokens, die 10 ist Token Nr. {stelle_der_zahl(enc)}")
    print(f"   Stücke:       {' | '.join(t for _, t in tokens(enc, KURZANTWORT))}")
    for zeile in RECHENWEG:
        print(f"   {len(enc.encode(zeile)):3d} Tokens: {zeile}")
    print(f"   Rechenweg:    {n_rechenweg} Tokens ({n_rechenweg / n_kurz:.1f}-mal so viele)")
    print(f"   Nachdenken:   {n_nachdenken} Tokens Denkprotokoll, mit Antwort {n_antwort} ({n_antwort / n_kurz:.1f}-mal so viele)")

    print("\nAbschnitt 4: Ein Modell, das sich verrechnen kann (ausgedachte Fehlerwahrscheinlichkeiten)")
    kurz = verteilung(FEHLER_KURZ)
    lang = verteilung(FEHLER_RECHENWEG)
    for name, p, vert in (("Ohne Rechenweg", FEHLER_KURZ, kurz), ("Mit Rechenweg", FEHLER_RECHENWEG, lang)):
        print(f"   {name} (Fehler pro Schritt {prozent(p)}): richtig in {prozent(vert[RICHTIG], 1)} der Fälle")
        print("      " + ", ".join(f"{a}: {prozent(w, 1)}" for a, w in sorted(vert.items(), key=lambda x: -x[1])[:6]))
        print(f"      ohne jeden Fehler: {prozent((1 - p) ** len(SCHRITTE), 1)}, richtig trotz Fehlern, die sich aufheben: {prozent(vert[RICHTIG] - (1 - p) ** len(SCHRITTE), 1)}")

    print(f"   Zum Vergleich, 20 Schritte mit Rechenweg: richtig in {prozent((1 - FEHLER_RECHENWEG) ** 20)} der Fälle")

    print(f"\nAbschnitt 5: Abstimmen ({WIEDERHOLUNGEN} Wiederholungen je Punkt)")
    kurven = {"Mit Rechenweg": [], "Ohne Rechenweg": []}
    for stimmen in STIMMEN:
        kurven["Mit Rechenweg"].append(abstimmung(FEHLER_RECHENWEG, stimmen, rng))
        kurven["Ohne Rechenweg"].append(abstimmung(FEHLER_KURZ, stimmen, rng))
        if stimmen == 1:
            # Bei einer einzigen Antwort gibt es nichts abzustimmen: genauen Wert aus Abschnitt 4 nehmen
            kurven["Mit Rechenweg"][-1] = lang[RICHTIG]
            kurven["Ohne Rechenweg"][-1] = kurz[RICHTIG]
        print(f"   {stimmen:2d} Antworten: mit Rechenweg {prozent(kurven['Mit Rechenweg'][-1])}, ohne Rechenweg {prozent(kurven['Ohne Rechenweg'][-1])}")

    print("\nAbschnitt 6: Spielzeug-Training mit Belohnung für richtige Ergebnisse")
    schritt_tokens = round(sum(len(enc.encode(z)) for z in RECHENWEG[:-1]) / len(SCHRITTE))
    pruef_tokens = round((n_nachdenken - n_rechenweg) / len(SCHRITTE))
    print(f"   Ein Rechenschritt: etwa {schritt_tokens} Tokens, eine Prüfung: etwa {pruef_tokens} Tokens zusätzlich")
    verlauf = training(rng, schritt_tokens, pruef_tokens)
    for runde in (0, 25, 50, 100, TRAINING_RUNDEN):
        v = verlauf[runde]
        print(f"   Runde {runde:3d}: prüft {prozent(v['pruefquote']):>5s} der Schritte, richtig {prozent(v['treffer'])}, Länge {v['laenge']:.0f} Tokens")

    kombinationen = list(alle_fehlerkombinationen())
    grafik_zeitleiste()
    grafik_rechenweg(enc)
    grafik_antworten(kurz, lang)
    grafik_abstimmen(kurven)
    grafik_stufen()
    grafik_training(verlauf)
    grafik_denkaufwand({"Kurzantwort": n_kurz, "Mit Rechenweg": n_rechenweg, "Nachdenken mit Proben": n_antwort})
    app_daten(enc, kurz, lang, kombinationen, schritt_tokens, pruef_tokens)
    print(f"\nGrafiken gespeichert in {BILDER}")
    print(f"Daten für die interaktive Seite gespeichert in {APP_DATEN}")


if __name__ == "__main__":
    main()

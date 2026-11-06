# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Der Eisverkäufer: von der linearen Regression zum Neuron.

Start: uv run docs/beispiele/01_eisverkaeufer.py

Rechnet das Beispiel aus 01-Neuronale-Netze-Grundlagen.md nach
und erzeugt die Grafiken in docs/bilder/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

BILDER = Path(__file__).resolve().parent.parent / "bilder"

# Erfundene Beobachtungen: (Temperatur in °C, Wochenende ja = 1 / nein = 0, verkaufte Kugeln)
TAGE = [
    (13, 0, 146),
    (15, 1, 226),
    (16, 0, 177),
    (19, 1, 257),
    (20, 0, 228),
    (22, 0, 245),
    (23, 1, 293),
    (26, 0, 272),
    (27, 1, 344),
    (29, 0, 312),
]

# Weitere erfundene Beobachtungen aus dem Hochsommer (ab Abschnitt 9)
HOCHSOMMER = [
    (31, 0, 333),
    (32, 1, 347),
    (33, 0, 345),
    (34, 1, 350),
    (35, 0, 348),
    (36, 1, 346),
    (37, 0, 350),
    (38, 1, 349),
]

temperatur, wochenende, kugeln = np.array(TAGE, dtype=float).T
alle_temperatur, alle_wochenende, alle_kugeln = np.array(TAGE + HOCHSOMMER, dtype=float).T

# Farben und Linien für alle Grafiken
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
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GITTER,
        "grid.linewidth": 0.75,
        "xtick.color": GRAU,
        "ytick.color": GRAU,
        "xtick.labelcolor": TEXT_LEISE,
        "ytick.labelcolor": TEXT_LEISE,
        "lines.solid_capstyle": "round",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def regression(*inputs):
    """Sucht Weights und Bias, die den quadratischen Fehler minimieren."""
    X = np.column_stack([*inputs, np.ones_like(kugeln)])
    *weights, bias = np.linalg.lstsq(X, kugeln, rcond=None)[0]
    return [float(w) for w in weights], float(bias)


def mittlere_abweichung(vorhersage):
    return float(np.mean(np.abs(kugeln - vorhersage)))


def quadratischer_fehler(vorhersage):
    """Mittlerer quadratischer Fehler: Fehler quadrieren, dann den Durchschnitt bilden."""
    return float(np.mean((kugeln - vorhersage) ** 2))


def steigungen(inputs, weights, bias):
    """Steigung der Fehlerkurve für jedes Weight und für den Bias."""
    fehler = kugeln - (sum(w * x for w, x in zip(weights, inputs)) + bias)
    return [float(-2 * np.mean(fehler * x)) for x in inputs], float(-2 * np.mean(fehler))


def trainiere(inputs, weights, bias, lernrate, schritte):
    """Gradientenabstieg: viele kleine Schritte gegen die Steigung, also bergab."""
    weights = list(weights)
    for _ in range(schritte):
        steigung_w, steigung_b = steigungen(inputs, weights, bias)
        weights = [w - lernrate * s for w, s in zip(weights, steigung_w)]
        bias -= lernrate * steigung_b
    return weights, bias


def neuron(inputs, weights, bias):
    """Ein Neuron: jeden Input gewichten, alles addieren, Bias dazu."""
    return sum(x * w for x, w in zip(inputs, weights)) + bias


def aktivierung(wert):
    """Der Knick (ReLU): Negative Werte werden zu 0, alles andere bleibt."""
    return np.maximum(0, wert)


def neuron_mit_knick(inputs, weights, bias):
    return aktivierung(neuron(inputs, weights, bias))


def zwei_neuronen(t, we, a=(10, 50, 20), b=(10, 50, -330), knick=True):
    """Winziges Netz: Neuron A minus Neuron B, wahlweise ohne Aktivierungsfunktion."""
    neuron_a = a[0] * t + a[1] * we + a[2]
    neuron_b = b[0] * t + b[1] * we + b[2]
    if knick:
        neuron_a, neuron_b = aktivierung(neuron_a), aktivierung(neuron_b)
    return neuron_a, neuron_b, neuron_a - neuron_b


def trainiere_mit_knick(inputs, weights, bias, lernrate, schritte):
    """Wie trainiere(), aber abgeschnittene Vorhersagen liefern keine Steigung."""
    weights = np.array(weights, dtype=float)
    for _ in range(schritte):
        summe = sum(w * x for w, x in zip(weights, inputs)) + bias
        aktiv = summe > 0
        fehler = kugeln - aktivierung(summe)
        weights += lernrate * 2 * np.array([np.mean(fehler * x * aktiv) for x in inputs])
        bias += lernrate * 2 * np.mean(fehler * aktiv)
    return weights, float(bias)


def schritte_bis_ziel(x, lernrate, ziel=629.21):
    """Zählt die Schritte, bis der Fehler den besten Wert fast erreicht hat."""
    w, b = 0.0, 0.0
    for schritt in range(1, 300_000):
        fehler = kugeln - (w * x + b)
        w += lernrate * 2 * np.mean(fehler * x)
        b += lernrate * 2 * np.mean(fehler)
        if quadratischer_fehler(w * x + b) < ziel:
            return schritt


def trainingsfehler(*inputs):
    weights, bias = regression(*inputs)
    return quadratischer_fehler(sum(w * x for w, x in zip(weights, inputs)) + bias)


def fehler_auf_neuen_tagen(*inputs):
    """Lässt jeden Tag einmal weg, lernt mit dem Rest und sagt den Tag dann vorher."""
    X = np.column_stack([*inputs, np.ones_like(kugeln)])
    fehler = []
    for tag in range(len(kugeln)):
        rest = np.arange(len(kugeln)) != tag
        parameter = np.linalg.lstsq(X[rest], kugeln[rest], rcond=None)[0]
        fehler.append(kugeln[tag] - X[tag] @ parameter)
    return float(np.mean(np.square(fehler)))


def netz_vorhersage(x, parameter):
    w, b, v, c = parameter
    return aktivierung(x @ w.T + b) @ v + c


def trainiere_netz(x, ziel, seed, lernrate=0.05, schritte=30_000):
    """Backpropagation von Hand: zwei Neuronen mit Knick plus eine Ausgabe, Zufallsstart."""
    zufall = np.random.default_rng(seed)
    w, b, v = zufall.normal(size=(2, x.shape[1])), zufall.normal(size=2), zufall.normal(size=2)
    c = 0.0
    for _ in range(schritte):
        summe = x @ w.T + b
        aktiv = summe > 0
        versteckt = aktivierung(summe)
        fehler = (ziel - (versteckt @ v + c))[:, None]
        # Ausgabe wie gewohnt, innere Neuronen mit dem Ausgabe-Weight als zusätzlichem Faktor
        steigung_w = -2 * ((fehler * aktiv).T @ x) / len(x) * v[:, None]
        steigung_b = -2 * np.mean(fehler * aktiv, axis=0) * v
        steigung_v = -2 * np.mean(fehler * versteckt, axis=0)
        steigung_c = -2 * np.mean(fehler)
        w, b, v, c = w - lernrate * steigung_w, b - lernrate * steigung_b, v - lernrate * steigung_v, c - lernrate * steigung_c
    return w, b, v, c


def minus(zahl):
    return f"{zahl:.0f}".replace("-", "−")


def knoten(ax, x, y, r, oben, unten=None, farbe=FLAECHE, rand=ACHSE, textfarbe=TEXT):
    ax.add_patch(Circle((x, y), r, facecolor=farbe, edgecolor=rand, linewidth=1.5, zorder=2))
    if unten is None:
        ax.text(x, y, oben, ha="center", va="center", color=textfarbe, fontsize=12, zorder=3)
    else:
        ax.text(x, y + 0.14, oben, ha="center", va="center", color=textfarbe, fontsize=11, zorder=3)
        ax.text(x, y - 0.17, unten, ha="center", va="center", color=textfarbe, fontsize=13, fontweight="bold", zorder=3)


def pfeil(ax, start, r_start, ziel, r_ziel):
    start, ziel = np.array(start), np.array(ziel)
    richtung = (ziel - start) / np.linalg.norm(ziel - start)
    ax.add_patch(
        FancyArrowPatch(
            start + richtung * r_start,
            ziel - richtung * r_ziel,
            arrowstyle="-|>",
            mutation_scale=14,
            color=TEXT_LEISE,
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def achsen_temperatur(ax):
    ax.set_xlabel("Temperatur (°C)")
    ax.set_ylabel("Verkaufte Kugeln")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 420)


def punkte(ax, x, y, farbe, **kwargs):
    ax.scatter(x, y, s=70, color=farbe, edgecolors=FLAECHE, linewidths=1.5, zorder=3, **kwargs)


def grafik_gerade(w, b):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array([0, 35])
    ax.plot(x, w * x + b, color=BLAU, linewidth=2, alpha=0.35, zorder=2)
    x_daten = np.array([temperatur.min(), temperatur.max()])
    ax.plot(x_daten, w * x_daten + b, color=BLAU, linewidth=2.5, zorder=2)
    punkte(ax, temperatur, kugeln, BLAU)

    # Bias: hier schneidet die Gerade die y-Achse
    ax.scatter([0], [b], s=90, color=BLAU, edgecolors=FLAECHE, linewidths=1.5, zorder=4, clip_on=False)
    ax.text(0.9, 12, f"Bias = {b:.0f} (Verkauf bei 0 °C)", color=TEXT, fontsize=12)

    # Weight: Steigungsdreieck
    x0, x1 = 5, 10
    y0, y1 = w * x0 + b, w * x1 + b
    ax.plot([x0, x1, x1], [y0, y0, y1], color=TEXT_LEISE, linewidth=1.25, zorder=2)
    ax.text((x0 + x1) / 2, y0 - 8, "+5 °C", ha="center", va="top", color=TEXT_LEISE, fontsize=11)
    ax.text(x1 + 0.4, (y0 + y1) / 2, f"+{w * (x1 - x0):.0f} Kugeln", va="center", color=TEXT_LEISE, fontsize=11)
    ax.text(1.2, 165, f"Weight = {w:.0f}\n(Kugeln pro Grad)", color=TEXT, fontsize=12)

    achsen_temperatur(ax)
    ax.set_title(f"Kugeln = {w:.0f} · Temperatur + {b:.0f}")
    fig.savefig(BILDER / "01-gerade.png")
    plt.close(fig)


def grafik_weight_bias(w, b):
    fig, (links, rechts) = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    x = np.array([0, 35])

    for ax, varianten, titel in (
        (links, [(w - 5, b), (w + 5, b)], "Weight ändern: steiler oder flacher"),
        (rechts, [(w, b - 40), (w, b + 60)], "Bias ändern: nach oben oder unten"),
    ):
        for w_neu, b_neu in varianten:
            ax.plot(x, w_neu * x + b_neu, color=GRAU, linewidth=1.5, zorder=2)
        ax.plot(x, w * x + b, color=BLAU, linewidth=2.5, zorder=3)
        achsen_temperatur(ax)
        ax.set_ylim(0, 620)
        ax.set_title(titel, fontsize=12.5)

    for ax, varianten in (
        (links, [(w + 5, b), (w, b), (w - 5, b)]),
        (rechts, [(w, b + 60), (w, b), (w, b - 40)]),
    ):
        for w_neu, b_neu in varianten:
            name = f"Weight {round(w_neu)}" if ax is links else f"Bias {round(b_neu)}"
            ist_original = (w_neu, b_neu) == (w, b)
            ax.text(35.6, w_neu * 35 + b_neu, name, va="center", color=TEXT if ist_original else TEXT_LEISE, fontsize=11)
    rechts.set_ylabel("")

    fig.tight_layout(w_pad=4)
    fig.savefig(BILDER / "02-weight-und-bias.png")
    plt.close(fig)


def grafik_fehlerquadrate(w, b):
    fig, achsen = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    durchschnitt = float(kugeln.mean())
    x = np.array([10, 36])

    for ax in achsen:
        ax.set_xlim(*x)
        ax.set_ylim(100, 400)
        ax.set_xlabel("Temperatur (°C)")
    achsen[0].set_ylabel("Verkaufte Kugeln")
    fig.tight_layout(w_pad=3)

    for ax, w_linie, b_linie, titel in (
        (achsen[0], w, b, f"Kugeln = {w:.0f} · Temperatur + {b:.0f}"),
        (achsen[1], 0.0, durchschnitt, f"Waagerechte Linie: Kugeln = {durchschnitt:.0f}"),
    ):
        vorhersage = w_linie * temperatur + b_linie
        fehler = kugeln - vorhersage

        # Gleiche Seitenlänge in Pixeln, damit die Quadrate auch wie Quadrate aussehen
        box = ax.get_window_extent()
        breite_pro_kugel = (box.height / 300) / (box.width / 26)
        for t, unten, f in zip(temperatur, np.minimum(kugeln, vorhersage), np.abs(fehler)):
            ax.add_patch(Rectangle((t, unten), f * breite_pro_kugel, f, facecolor=ORANGE, alpha=0.18, edgecolor="none", zorder=1))
        ax.vlines(temperatur, kugeln, vorhersage, color=ORANGE, linewidth=1.5, zorder=2)

        ax.plot(x, w_linie * x + b_linie, color=BLAU, linewidth=2.5, zorder=2)
        punkte(ax, temperatur, kugeln, BLAU)
        ax.set_title(titel, fontsize=12.5)
        ax.text(
            10.8,
            392,
            f"Summe der Fehler: {abs(fehler.sum()):.0f}\nMittlerer quadratischer Fehler: {np.mean(fehler ** 2):.0f}",
            va="top",
            color=TEXT,
            fontsize=11,
            linespacing=1.4,
        )

    fig.savefig(BILDER / "03-fehlerquadrate.png")
    plt.close(fig)


def grafik_fehlerkurve(w, b):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    weights = np.linspace(w - 6, w + 6, 200)
    fehler = [quadratischer_fehler(wert * temperatur + b) for wert in weights]
    ax.plot(weights, fehler, color=BLAU, linewidth=2.5, zorder=2)

    links, rechts = w - 3, w + 3
    ax.scatter([links, rechts], [quadratischer_fehler(wert * temperatur + b) for wert in (links, rechts)], s=70, color=BLAU, edgecolors=FLAECHE, linewidths=1.5, zorder=3)
    minimum = quadratischer_fehler(w * temperatur + b)
    ax.scatter([w], [minimum], s=110, color=ORANGE, edgecolors=FLAECHE, linewidths=1.5, zorder=3)

    ax.text(links + 0.9, 700, "Weight zu klein:\nvergrößern senkt den Fehler", ha="right", color=TEXT, fontsize=11.5)
    ax.text(rechts - 0.9, 700, "Weight zu groß:\nverkleinern senkt den Fehler", ha="left", color=TEXT, fontsize=11.5)
    ax.text(w, minimum + 2000, f"Tiefster Punkt\nWeight {w:.0f}, Fehler {minimum:.0f}", ha="center", color=TEXT, fontsize=11.5)

    ax.set_xlabel(f"Weight (Bias fest bei {b:.0f})")
    ax.set_ylabel("Mittlerer quadratischer Fehler")
    ax.set_xlim(w - 6, w + 6)
    ax.set_ylim(0, None)
    ax.set_title("Die Fehlerkurve hat genau einen tiefsten Punkt")
    fig.savefig(BILDER / "04-fehlerkurve.png")
    plt.close(fig)


def grafik_fehler_minimieren(b):
    fig, achsen = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    weights = np.linspace(-6, 26, 200)
    kurve = [quadratischer_fehler(w * temperatur + b) for w in weights]

    for ax, lernrate, schritte, titel in (
        (achsen[0], 0.0001, 6, "Lernrate 0,0001: zu klein"),
        (achsen[1], 0.0005, 6, "Lernrate 0,0005: passend"),
        (achsen[2], 0.0022, 6, "Lernrate 0,0022: zu groß"),
    ):
        ax.plot(weights, kurve, color=GRAU, linewidth=1.5, zorder=1)

        pfad = [0.0]
        for _ in range(schritte):
            pfad.append(trainiere([temperatur], [pfad[-1]], b, lernrate, 1)[0][0])
        fehler = [quadratischer_fehler(w * temperatur + b) for w in pfad]

        ax.plot(pfad, fehler, color=ORANGE, linewidth=1.5, zorder=2)
        punkte(ax, pfad, fehler, ORANGE)
        ax.scatter(pfad[:1], fehler[:1], s=90, color=BLAU, edgecolors=FLAECHE, linewidths=1.5, zorder=4)
        ax.set_xlim(-6, 26)
        ax.set_ylim(0, 90000)
        ax.set_xlabel("Weight")
        ax.set_title(titel, fontsize=12.5)

    achsen[0].set_ylabel("Mittlerer quadratischer Fehler")
    fig.tight_layout(w_pad=2)
    fig.savefig(BILDER / "05-fehler-minimieren.png")
    plt.close(fig)


def grafik_wochenende(w_temperatur, w_wochenende, b):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array([10, 32])
    werktag = wochenende == 0

    ax.plot(x, w_temperatur * x + b, color=BLAU, linewidth=2.5, zorder=2)
    ax.plot(x, w_temperatur * x + w_wochenende + b, color=ORANGE, linewidth=2.5, zorder=2)
    punkte(ax, temperatur[werktag], kugeln[werktag], BLAU, marker="o", label="Werktag")
    punkte(ax, temperatur[~werktag], kugeln[~werktag], ORANGE, marker="s", label="Wochenende")

    # Abstand der beiden Geraden: das zweite Weight
    x_abstand = 11.5
    y_unten = w_temperatur * x_abstand + b
    ax.annotate(
        "",
        (x_abstand, y_unten + w_wochenende),
        (x_abstand, y_unten),
        arrowprops={"arrowstyle": "<->", "color": TEXT_LEISE, "linewidth": 1.25, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(x_abstand - 0.4, y_unten + w_wochenende / 2, f"+{w_wochenende:.0f} Kugeln", ha="right", va="center", color=TEXT, fontsize=12)

    ax.set_xlabel("Temperatur (°C)")
    ax.set_ylabel("Verkaufte Kugeln")
    ax.set_xlim(6, 32)
    ax.set_ylim(100, 400)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_LEISE)
    ax.set_title("Zwei parallele Geraden: gleiches Temperatur-Weight, Abstand 50")
    fig.savefig(BILDER / "06-wochenende.png")
    plt.close(fig)


def grafik_knick(w, w_wochenende, b):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.linspace(-15, 35, 501)

    for wert, farbe, name in ((0, BLAU, "Werktag"), (1, ORANGE, "Wochenende")):
        gerade = w * x + w_wochenende * wert + b
        ax.plot(x, gerade, color=farbe, linewidth=2, alpha=0.3, zorder=2)
        ax.plot(x, aktivierung(gerade), color=farbe, linewidth=2.5, zorder=3, label=name)
        knick_x = -(w_wochenende * wert + b) / w
        ax.scatter([knick_x], [0], s=90, color=farbe, edgecolors=FLAECHE, linewidths=1.5, zorder=4)
        ax.annotate(
            f"Knick bei {minus(knick_x)} °C",
            (knick_x, 0),
            xytext=(knick_x + (-5.5 if wert else 1.2), 45 if wert else -42),
            color=TEXT,
            fontsize=11.5,
        )

    ax.axhline(0, color=ACHSE, linewidth=1, zorder=1)
    ax.set_xlabel("Temperatur (°C)")
    ax.set_ylabel("Vorhergesagte Kugeln")
    ax.set_xlim(-15, 35)
    ax.set_ylim(-140, 440)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_LEISE)
    ax.set_title("Die Aktivierungsfunktion knickt beide Linien bei 0 ab")
    fig.savefig(BILDER / "08-knick.png")
    plt.close(fig)


def tage_punkte(ax, alle=True):
    t, we, k = (alle_temperatur, alle_wochenende, alle_kugeln) if alle else (temperatur, wochenende, kugeln)
    punkte(ax, t[we == 0], k[we == 0], BLAU, marker="o")
    punkte(ax, t[we == 1], k[we == 1], ORANGE, marker="s")


def grafik_hochsommer(w_temperatur, w_wochenende, b):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array([10, 40])
    ax.plot(x, w_temperatur * x + b, color=BLAU, linewidth=2.5, zorder=2, label="Werktag")
    ax.plot(x, w_temperatur * x + w_wochenende + b, color=ORANGE, linewidth=2.5, zorder=2, label="Wochenende")
    tage_punkte(ax)

    ax.axvspan(30, 40, color=ACHSE, alpha=0.18, zorder=0, linewidth=0)
    ax.text(30.4, 118, "Hochsommer", color=TEXT_LEISE, fontsize=11.5)
    ax.annotate(
        "Vorhersage 450,\nverkauft 349",
        (38, 349),
        xytext=(31.4, 228),
        color=TEXT,
        fontsize=11.5,
        arrowprops={"arrowstyle": "-", "color": TEXT_LEISE, "linewidth": 1},
    )

    ax.set_xlabel("Temperatur (°C)")
    ax.set_ylabel("Verkaufte Kugeln")
    ax.set_xlim(10, 40)
    ax.set_ylim(100, 480)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_LEISE)
    ax.set_title("Im Hochsommer sagt das Neuron zu viel voraus")
    fig.savefig(BILDER / "10-hochsommer.png")
    plt.close(fig)


def grafik_zwei_knicke():
    fig, achsen = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    x = np.linspace(-15, 40, 551)

    for ax, knick, titel in (
        (achsen[0], False, "Ohne Knick: immer nur 350"),
        (achsen[1], True, "Mit Knick: Die Kurven treffen die Tage"),
    ):
        neuron_a, neuron_b, werktag = zwei_neuronen(x, 0, knick=knick)
        wochenende_kurve = zwei_neuronen(x, 1, knick=knick)[2]
        ax.plot(x, neuron_a, color=GRAU, linewidth=1.5, zorder=2)
        ax.plot(x, neuron_b, color=GRAU, linewidth=1.5, zorder=2)
        ax.plot(x, wochenende_kurve, color=ORANGE, linewidth=2.5, zorder=3, label="A − B am Wochenende")
        ax.plot(x, werktag, color=BLAU, linewidth=2.5, zorder=4, label="A − B am Werktag")
        tage_punkte(ax)

        ax.text(40.8, neuron_a[-1], "Neuron A", va="center", color=TEXT_LEISE, fontsize=11)
        ax.text(40.8, neuron_b[-1], "Neuron B", va="center", color=TEXT_LEISE, fontsize=11)
        ax.set_xlim(-15, 40)
        ax.set_ylim(-150, 480)
        ax.set_xlabel("Temperatur (°C)")
        ax.set_title(titel, fontsize=12.5)

    achsen[0].set_ylabel("Kugeln")
    handles, labels = achsen[1].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=2, frameon=False, labelcolor=TEXT_LEISE, title="Neuron A und B jeweils für den Werktag, Punkte: alle 18 Tage", title_fontsize=10.5)
    fig.tight_layout(w_pad=4, rect=(0, 0.12, 1, 1))
    fig.savefig(BILDER / "11-zwei-knicke.png")
    plt.close(fig)


def grafik_neuron(w_temperatur, w_wochenende, b, temperatur_wert=25, wochenende_wert=1, mit_knick=False):
    breite = 11.8 if mit_knick else 10
    fig, ax = plt.subplots(figsize=(0.9 * breite, 4.2))
    ax.set_xlim(0, breite)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    summe = neuron([temperatur_wert, wochenende_wert], [w_temperatur, w_wochenende], b)
    vorhersage = float(aktivierung(summe)) if mit_knick else summe

    temperatur_pos, wochenende_pos, neuron_pos = (1.0, 3.6), (1.0, 1.1), (5.2, 2.35)
    vorhersage_pos = (10.8, 2.35) if mit_knick else (9.0, 2.35)
    knoten(ax, *temperatur_pos, 0.85, "Temperatur", f"{minus(temperatur_wert)} °C")
    knoten(ax, *wochenende_pos, 0.85, "Wochenende", minus(wochenende_wert))
    knoten(ax, *neuron_pos, 0.95, "addieren", "Σ", farbe=BLAU, rand=BLAU, textfarbe="#ffffff")
    knoten(ax, *vorhersage_pos, 0.85, "Vorhersage", minus(vorhersage))

    pfeil(ax, temperatur_pos, 0.85, neuron_pos, 0.95)
    pfeil(ax, wochenende_pos, 0.85, neuron_pos, 0.95)
    pfeil(ax, (5.2, 4.5), 0, neuron_pos, 0.95)

    if mit_knick:
        knick_pos = (8.0, 2.35)
        ax.add_patch(Circle(knick_pos, 0.85, facecolor=ORANGE, edgecolor=ORANGE, linewidth=1.5, zorder=2))
        ax.text(knick_pos[0], knick_pos[1] + 0.34, "Knick", ha="center", va="center", color="#ffffff", fontsize=11, zorder=3)
        ax.plot(
            [knick_pos[0] - 0.5, knick_pos[0] - 0.02, knick_pos[0] + 0.48],
            [knick_pos[1] - 0.42, knick_pos[1] - 0.42, knick_pos[1] + 0.08],
            color="#ffffff",
            linewidth=2.5,
            zorder=3,
        )
        pfeil(ax, neuron_pos, 0.95, knick_pos, 0.85)
        pfeil(ax, knick_pos, 0.85, vorhersage_pos, 0.85)
        ax.text(6.65, 2.5, minus(summe), ha="center", color=TEXT, fontsize=12, fontweight="bold")
        rechnung = (
            f"{minus(temperatur_wert)} · {w_temperatur:.0f}  +  {minus(wochenende_wert)} · {w_wochenende:.0f}  +  {b:.0f}"
            f"  =  {minus(summe)},   Knick: aus {minus(summe)} wird {minus(vorhersage)}"
        )
        ax.text(5.9, 0.3, rechnung, ha="center", color=TEXT_LEISE, fontsize=12)
    else:
        pfeil(ax, neuron_pos, 0.95, vorhersage_pos, 0.85)
        ax.text(5.2, 0.3, "25 · 10  +  1 · 50  +  20  =  320", ha="center", color=TEXT_LEISE, fontsize=12)

    ax.text(3.15, 3.42, f"Weight {w_temperatur:.0f}", ha="center", color=TEXT, fontsize=12)
    ax.text(3.15, 1.12, f"Weight {w_wochenende:.0f}", ha="center", color=TEXT, fontsize=12)
    ax.text(5.42, 4.25, f"Bias {b:.0f}", ha="left", va="center", color=TEXT, fontsize=12)

    fig.savefig(BILDER / ("09-neuron-mit-knick.png" if mit_knick else "07-neuron.png"))
    plt.close(fig)


def grafik_netz(temperatur_wert=35.0, wochenende_wert=1.0):
    neuron_a, neuron_b, ausgabe = (float(wert) for wert in zwei_neuronen(temperatur_wert, wochenende_wert))
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")

    temperatur_pos, wochenende_pos, oben, unten, ende = (1.2, 4.1), (1.2, 1.4), (5.9, 4.1), (5.9, 1.4), (10.3, 2.75)

    # Die Eingaben sind keine Neuronen, sondern nur Werte, und bekommen deshalb ein Rechteck
    for (x, y), name, wert in ((temperatur_pos, "Temperatur", f"{minus(temperatur_wert)} °C"), (wochenende_pos, "Wochenende", minus(wochenende_wert))):
        ax.add_patch(Rectangle((x - 0.95, y - 0.6), 1.9, 1.2, facecolor=FLAECHE, edgecolor=ACHSE, linewidth=1.5, zorder=2))
        ax.text(x, y + 0.2, name, ha="center", va="center", color=TEXT, fontsize=11, zorder=3)
        ax.text(x, y - 0.17, wert, ha="center", va="center", color=TEXT, fontsize=13, fontweight="bold", zorder=3)

    for position, name, wert in ((oben, "Neuron A", neuron_a), (unten, "Neuron B", neuron_b)):
        x, y = position
        ax.add_patch(Circle(position, 0.95, facecolor=BLAU, edgecolor=BLAU, linewidth=1.5, zorder=2))
        ax.text(x, y + 0.48, name, ha="center", va="center", color="#ffffff", fontsize=11, zorder=3)
        ax.text(x, y + 0.08, minus(wert), ha="center", va="center", color="#ffffff", fontsize=14, fontweight="bold", zorder=3)
        ax.plot([x - 0.4, x, x + 0.4], [y - 0.62, y - 0.62, y - 0.22], color="#ffffff", linewidth=2.5, zorder=3)

    knoten(ax, *ende, 0.95, "Ausgabe", minus(ausgabe), farbe=ORANGE, rand=ORANGE, textfarbe="#ffffff")

    # Jede Eingabe geht an jedes Neuron, die Weights stehen direkt auf den Pfeilen
    for start, ziel, anteil, name in (
        (temperatur_pos, oben, 0.42, "Weight 10"),
        (temperatur_pos, unten, 0.27, "Weight 10"),
        (wochenende_pos, oben, 0.27, "Weight 50"),
        (wochenende_pos, unten, 0.42, "Weight 50"),
    ):
        pfeil(ax, (start[0] + 0.95, start[1]), 0, ziel, 0.95)
        x = start[0] + 0.95 + anteil * (ziel[0] - start[0] - 0.95)
        y = start[1] + anteil * (ziel[1] - start[1])
        ax.text(x, y, name, ha="center", va="center", color=TEXT, fontsize=11.5, zorder=4, bbox={"facecolor": FLAECHE, "edgecolor": "none", "pad": 2.5})

    pfeil(ax, oben, 0.95, ende, 0.95)
    pfeil(ax, unten, 0.95, ende, 0.95)
    ax.text(8.1, 3.95, "Weight +1", ha="center", color=TEXT, fontsize=12)
    ax.text(8.1, 1.35, "Weight −1", ha="center", color=TEXT, fontsize=12)

    # Der Bias kommt bei jedem Neuron als eigener Pfeil dazu
    pfeil(ax, (oben[0], 5.75), 0, oben, 0.95)
    pfeil(ax, (unten[0], -0.25), 0, unten, 0.95)
    pfeil(ax, (ende[0], 1.1), 0, ende, 0.95)
    ax.text(oben[0] + 0.2, 5.45, "Bias 20", va="center", color=TEXT, fontsize=12)
    ax.text(unten[0] + 0.2, 0.05, "Bias −330", va="center", color=TEXT, fontsize=12)
    ax.text(ende[0] + 0.2, 1.4, "Bias 0", va="center", color=TEXT, fontsize=12)

    eingaben = f"10 · {minus(temperatur_wert)} + 50 · {minus(wochenende_wert)}"
    ax.text(5.9, 5.9, f"{eingaben} + 20 = {minus(neuron_a)}", ha="center", color=TEXT, fontsize=12)
    ax.text(5.9, -0.7, f"{eingaben} − 330 = {minus(neuron_b)}", ha="center", color=TEXT, fontsize=12)
    ax.text(10.3, 3.95, f"+1 · {minus(neuron_a)} − 1 · {minus(neuron_b)} + 0\n= {minus(ausgabe)}", ha="center", color=TEXT, fontsize=12, linespacing=1.3)

    for x, titel, zusatz in (
        (1.2, "1. Eingabe", "nur Werte, keine Neuronen"),
        (5.9, "2. Verborgene Schicht", "zwei Neuronen mit Knick"),
        (10.3, "3. Ausgabe-Schicht", "ein Neuron ohne Knick"),
    ):
        ax.text(x, 6.95, titel, ha="center", color=TEXT, fontsize=12.5, fontweight="bold")
        ax.text(x, 6.55, zusatz, ha="center", color=TEXT_LEISE, fontsize=11)

    fig.savefig(BILDER / "12-netz.png")
    plt.close(fig)


def main():
    BILDER.mkdir(exist_ok=True)

    print("Abschnitte 2 bis 5: Regression mit einem Input (Temperatur)")
    [w1], b1 = regression(temperatur)
    print(f"   Kugeln = {w1:.2f} · Temperatur + {b1:.2f}")
    print(f"   Mittlere Abweichung: {mittlere_abweichung(w1 * temperatur + b1):.2f} Kugeln")
    print(f"   Vorhersage für 25 °C: {neuron([25], [w1], b1):.0f} Kugeln")

    print("\n   Fehler pro Tag (Messwert - Vorhersage)")
    print("   Temp.  Kugeln  Vorhersage  Fehler  Quadrat")
    vorhersage = w1 * temperatur + b1
    for t, k, v in zip(temperatur, kugeln, vorhersage):
        print(f"   {t:5.0f}  {k:6.0f}  {v:10.0f}  {k - v:+6.0f}  {(k - v) ** 2:7.0f}")
    print(f"   Summe der Fehler: {abs((kugeln - vorhersage).sum()):.0f}")
    print(f"   Summe der Quadrate: {((kugeln - vorhersage) ** 2).sum():.0f}")
    print(f"   Mittlerer quadratischer Fehler: {quadratischer_fehler(vorhersage):.1f}")

    print("\n   Andere Linien im Vergleich")
    for w_test, b_test in ((8, b1), (w1, b1), (12, b1), (w1, 20), (w1, 60), (0, float(kugeln.mean()))):
        v = w_test * temperatur + b_test
        print(
            f"   Weight {w_test:4.0f}, Bias {b_test:4.0f}: Summe der Fehler {abs((kugeln - v).sum()):6.0f}, "
            f"mittlerer quadratischer Fehler {quadratischer_fehler(v):7.1f}"
        )

    print("\n   Fehler minimieren: nur das Weight lernen, Bias fest, Lernrate 0,0005")
    print("   Schritt  Weight   Steigung  Änderung    Fehler")
    w = 0.0
    for schritt in range(9):
        [steigung], _ = steigungen([temperatur], [w], b1)
        print(f"   {schritt:7d}  {w:6.2f}  {steigung:9.1f}  {-0.0005 * steigung:+8.2f}  {quadratischer_fehler(w * temperatur + b1):8.1f}")
        w -= 0.0005 * steigung

    print("\n   Steigung durch Ausprobieren (Weight 0 gegen Weight 0,01)")
    f0, f1 = (quadratischer_fehler(w * temperatur + b1) for w in (0, 0.01))
    print(f"   Fehler {f0:.1f} gegen {f1:.1f}, Steigung etwa {(f1 - f0) / 0.01:.0f}")

    print("\n   Schritte bis zum besten Fehler")
    print(f"   Temperatur unverändert, Lernrate 0,001: {schritte_bis_ziel(temperatur, 0.001)}")
    print(f"   Temperatur skaliert, Lernrate 0,1: {schritte_bis_ziel((temperatur - temperatur.mean()) / temperatur.std(), 0.1)}")

    print("\n   Fehler minimieren: Weight und Bias zusammen, Start bei 0, Lernrate 0,001")
    print("   Schritte  Weight    Bias    Fehler")
    for schritte in (0, 1, 10, 100, 1000, 10000, 50000, 100000):
        [w], b = trainiere([temperatur], [0.0], 0.0, 0.001, schritte)
        print(f"   {schritte:8d}  {w:6.2f}  {b:6.2f}  {quadratischer_fehler(w * temperatur + b):8.1f}")

    print("\nAbschnitt 6: Regression mit zwei Inputs (Temperatur, Wochenende)")
    (w_t, w_we), b2 = regression(temperatur, wochenende)
    print(f"   Kugeln = {w_t:.2f} · Temperatur + {w_we:.2f} · Wochenende + {b2:.2f}")
    vorhersage = w_t * temperatur + w_we * wochenende + b2
    print(f"   Mittlere Abweichung: {mittlere_abweichung(vorhersage):.2f} Kugeln")
    print(f"   Mittlerer quadratischer Fehler: {quadratischer_fehler(vorhersage):.1f}")

    print("\n   Dasselbe durch Training (Start bei 0, Lernrate 0,001, 100000 Schritte)")
    (w_t_neu, w_we_neu), b_neu = trainiere([temperatur, wochenende], [0.0, 0.0], 0.0, 0.001, 100000)
    print(f"   Kugeln = {w_t_neu:.2f} · Temperatur + {w_we_neu:.2f} · Wochenende + {b_neu:.2f}")

    print("\nAbschnitt 7: Fehler auf neuen Tagen (jeden Tag einmal auslassen, mittlerer quadratischer Fehler)")
    print(f"   Nur Temperatur: Training {trainingsfehler(temperatur):.0f}, neue Tage {fehler_auf_neuen_tagen(temperatur):.0f}")
    print(f"   Temperatur + Wochenende: Training {trainingsfehler(temperatur, wochenende):.0f}, neue Tage {fehler_auf_neuen_tagen(temperatur, wochenende):.0f}")
    zufall = np.random.default_rng(1)
    for anzahl in (1, 3, 6):
        laeufe = [(temperatur, wochenende, *zufall.normal(size=(anzahl, len(kugeln)))) for _ in range(200)]
        training = np.median([trainingsfehler(*inputs) for inputs in laeufe])
        neu = np.median([fehler_auf_neuen_tagen(*inputs) for inputs in laeufe])
        print(f"   Plus {anzahl} Zufalls-Inputs (Median aus 200 Läufen): Training {training:.0f}, neue Tage {neu:.0f}")
    inputs = (temperatur, wochenende, *zufall.normal(size=(7, len(kugeln))))
    print(f"   Plus 7 Zufalls-Inputs (10 Parameter für 10 Tage): Training {trainingsfehler(*inputs):.1f}")

    print("\nAbschnitt 8: Dasselbe als Neuron")
    for t, we, name in ((25, 0, "Werktag"), (25, 1, "Samstag")):
        print(f"   {t} °C, {name}: {neuron([t, we], [w_t, w_we], b2):.0f} Kugeln")

    print("\nAbschnitt 9: Der Knick (Aktivierungsfunktion)")
    for t, we in ((-10, 0), (-10, 1), (0, 0), (25, 1)):
        summe = neuron([t, we], [w_t, w_we], b2)
        print(f"   {t:4d} °C, Wochenende {we}: Summe {summe:6.0f}, mit Knick {neuron_mit_knick([t, we], [w_t, w_we], b2):6.0f}")
    print(f"   Vorhersagen der Beobachtungstage mit Knick unverändert: {bool((w_t * temperatur + w_we * wochenende + b2 > 0).all())}")

    print("\n   Training mit Knick (Lernrate 0,001, 100000 Schritte)")
    for start in (0.0, -1.0, 1.0):
        (w_neu_t, w_neu_we), b_neu = trainiere_mit_knick([temperatur, wochenende], [start, max(start, 0.0)], max(start, 0.0), 0.001, 100000)
        v = aktivierung(w_neu_t * temperatur + w_neu_we * wochenende + b_neu)
        print(f"   Start {start:4.0f}, {max(start, 0.0):.0f}, {max(start, 0.0):.0f}: Weights {w_neu_t:.2f}, {w_neu_we:.2f}, Bias {b_neu:.2f}, Fehler {quadratischer_fehler(v):.1f}")

    print("\n   Hochsommertage: bisheriges Neuron gegen tatsächliche Verkäufe")
    heiss = alle_temperatur >= 31
    bisher = w_t * alle_temperatur + w_we * alle_wochenende + b2
    for t, we, k, v in zip(alle_temperatur[heiss], alle_wochenende[heiss], alle_kugeln[heiss], bisher[heiss]):
        print(f"   {t:3.0f} °C, Wochenende {we:.0f}: Vorhersage {v:4.0f}, verkauft {k:3.0f}, Fehler {k - v:+5.0f}")
    print(f"   Mittlere Abweichung Hochsommer: {np.mean(np.abs(alle_kugeln[heiss] - bisher[heiss])):.1f} Kugeln")

    def vergleich(name, vorhersage):
        fehler = alle_kugeln - vorhersage
        print(f"   {name}: im Mittel daneben {np.mean(np.abs(fehler)):.1f}, Hochsommer {np.mean(np.abs(fehler[heiss])):.1f}, quadratischer Fehler {np.mean(fehler ** 2):.1f}")

    print("\n   Vergleich auf allen 18 Tagen")
    X = np.column_stack([alle_temperatur, alle_wochenende, np.ones_like(alle_kugeln)])
    parameter_neuron = np.linalg.lstsq(X, alle_kugeln, rcond=None)[0]
    print(f"   Neu trainiertes Neuron: {parameter_neuron[0]:.2f} · Temperatur + {parameter_neuron[1]:.2f} · Wochenende + {parameter_neuron[2]:.2f}")
    vergleich("Bisheriges Neuron", bisher)
    vergleich("Neu trainiertes Neuron", X @ parameter_neuron)
    vergleich("Zwei Neuronen von Hand", zwei_neuronen(alle_temperatur, alle_wochenende)[2])

    print("\n   Zwei Neuronen: A = 10 · Temperatur + 50 · Wochenende + 20, B = 10 · Temperatur + 50 · Wochenende - 330")
    print("   Temp.  WE   ohne Knick: A, B, A - B     mit Knick: A, B, A - B")
    for t, we in ((-10.0, 0.0), (0.0, 0.0), (20.0, 0.0), (33.0, 0.0), (35.0, 0.0), (28.0, 1.0), (35.0, 1.0)):
        ohne, mit = zwei_neuronen(t, we, knick=False), zwei_neuronen(t, we)
        print(f"   {t:5.0f}  {we:2.0f}   {ohne[0]:6.0f} {ohne[1]:6.0f} {ohne[2]:6.0f}      {mit[0]:6.0f} {mit[1]:6.0f} {mit[2]:6.0f}")

    print("\nAbschnitt 10: Backpropagation an einem Samstag (35 °C, Ziel 350, Temperatur-Weight von B 9 statt 10)")

    def fehler_quadrat(w_a, w_b):
        return (350 - (aktivierung(w_a * 35 + 50 + 20) - aktivierung(w_b * 35 + 50 - 330))) ** 2

    a, b = 10 * 35 + 50 + 20, 9 * 35 + 50 - 330
    print(f"   A = {a}, B = {b}, Ausgabe = {a - b}, Fehler = {350 - (a - b)}")
    print(f"   Steigung Weight A: Formel {-2 * (350 - (a - b)) * 1 * 35}, ausprobiert {(fehler_quadrat(10.0001, 9) - fehler_quadrat(10, 9)) / 0.0001:.0f}")
    print(f"   Steigung Weight B: Formel {-2 * (350 - (a - b)) * -1 * 35}, ausprobiert {(fehler_quadrat(10, 9.0001) - fehler_quadrat(10, 9)) / 0.0001:.0f}")

    print("\n   Das Netz lernt aus den 18 Tagen: 20 Zufallsstarts")
    eingaben = np.column_stack([(alle_temperatur - 25) / 12, alle_wochenende])

    def netz_fehler(parameter, tage, x=eingaben):
        return float(np.mean((alle_kugeln[tage] - netz_vorhersage(x[tage], parameter) * 350) ** 2))

    alle_tage = np.ones_like(alle_kugeln, dtype=bool)
    netze = [trainiere_netz(eingaben, alle_kugeln / 350, seed) for seed in range(20)]
    fehler = sorted(netz_fehler(parameter, alle_tage) for parameter in netze)
    print(f"   Fehler je Start: {[round(f) for f in fehler]}")
    print(f"   Einzelnes Neuron auf denselben Tagen: {np.mean((alle_kugeln - X @ parameter_neuron) ** 2):.1f}")
    print(f"   Waagerechte Linie: {float(np.var(alle_kugeln)):.0f}")

    print("\n   Test an neuen Tagen: jeder dritte Tag bleibt beim Training weg")
    reihenfolge = np.argsort(alle_temperatur, kind="stable")
    test = np.zeros_like(alle_tage)
    test[reihenfolge[2::3]] = True
    print(f"   Testtage: {[(int(t), int(we)) for t, we in zip(alle_temperatur[test], alle_wochenende[test])]}")
    neuron_test = np.linalg.lstsq(X[~test], alle_kugeln[~test], rcond=None)[0]
    print(f"   Einzelnes Neuron: Training {np.mean((alle_kugeln[~test] - X[~test] @ neuron_test) ** 2):.1f}, neue Tage {np.mean((alle_kugeln[test] - X[test] @ neuron_test) ** 2):.1f}")
    kandidaten = [trainiere_netz(eingaben[~test], alle_kugeln[~test] / 350, seed) for seed in range(20)]
    bestes = min(kandidaten, key=lambda parameter: netz_fehler(parameter, ~test))
    print(f"   Bestes von 20 Netzen: Training {netz_fehler(bestes, ~test):.1f}, neue Tage {netz_fehler(bestes, test):.1f}")

    grafik_gerade(w1, b1)
    grafik_weight_bias(w1, b1)
    grafik_fehlerquadrate(w1, b1)
    grafik_fehlerkurve(w1, b1)
    grafik_fehler_minimieren(b1)
    grafik_wochenende(w_t, w_we, b2)
    grafik_neuron(w_t, w_we, b2)
    grafik_knick(w_t, w_we, b2)
    grafik_neuron(w_t, w_we, b2, temperatur_wert=-10, wochenende_wert=0, mit_knick=True)
    grafik_hochsommer(w_t, w_we, b2)
    grafik_zwei_knicke()
    grafik_netz()
    print(f"\nGrafiken gespeichert in {BILDER}")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = ["tiktoken", "numpy", "matplotlib"]
# ///
"""Gut fragen: die Beispiele aus Kapitel 6.

Start: uv run docs/beispiele/06_prompts.py

Setzt die Prompts aus 06-Gut-fragen.md aus ihren Bausteinen zusammen, zählt ihre Tokens,
misst, wie viele Tokens auf 100 Wörter im Text des Workshops kommen, und erzeugt die
Grafiken in docs/bilder/kapitel-6/ und die Daten für docs/app/labor-6.html.

Die Beispielantworten sind ausgedacht. Ein Sprachmodell würde bei jedem Versuch etwas
anders antworten, sie zeigen aber den typischen Unterschied. Die Verkaufszahlen für die
Auswertung lädt das Skript aus dem Skript zu Kapitel 1. Beim ersten Start lädt tiktoken
das Vokabular o200k_base aus dem Internet und legt es im Cache ab.
"""

import importlib.util
import itertools
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DOCS = Path(__file__).resolve().parent.parent
BILDER = DOCS / "bilder" / "kapitel-6"
APP_DATEN = DOCS / "app" / "prompt-daten.js"
ENC = tiktoken.get_encoding("o200k_base")


def kapitel_1_laden():
    """Das Skript zu Kapitel 1 als Modul laden, um seine Verkaufszahlen zu nutzen."""
    spec = importlib.util.spec_from_file_location("eisverkaeufer", DOCS / "beispiele" / "01_eisverkaeufer.py")
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


KAPITEL_1 = kapitel_1_laden()

# Die sechs Bausteine eines Prompts: (Schlüssel, Name, englischer Begriff, Frage dahinter)
BAUSTEINE = [
    ("aufgabe", "Aufgabe", "Task", "Was soll herauskommen?"),
    ("hintergrund", "Hintergrund", "Context", "Wer, wofür, für wen?"),
    ("material", "Material", "Input Data", "Womit soll das Modell arbeiten?"),
    ("form", "Form", "Output Format", "Wie lang, wie gegliedert, in welchem Ton?"),
    ("beispiel", "Beispiel", "Example", "Wie sieht ein gutes Ergebnis aus?"),
    ("regeln", "Regeln", "Constraints", "Was ist erlaubt, was nicht?"),
]


def verkaufstabelle():
    zeilen = ["Tag | Temperatur | Wochenende | Kugeln"]
    for nr, (t, we, k) in enumerate(KAPITEL_1.TAGE, start=1):
        zeilen.append(f"{nr} | {t} °C | {'ja' if we else 'nein'} | {k}")
    return "\n".join(zeilen)


# Drei Aufgaben aus dem Alltag der Eisdiele, jeweils mit allen Bausteinen.
# Ein Baustein, den es für eine Aufgabe nicht braucht, ist None.
AUFGABEN = [
    {
        "id": "beschwerde",
        "titel": "Antwort auf eine Beschwerde",
        "bausteine": {
            "aufgabe": "Schreib eine Antwort auf die E-Mail einer Kundin.",
            "hintergrund": "Ich betreibe eine kleine Eisdiele am Stadtpark. Die Kundin kommt seit Jahren zu uns. "
                           "Am Samstag war es sehr voll, und wir waren nur zu zweit am Stand.",
            "material": "<email>\nHallo, ich habe am Samstag 25 Minuten auf zwei Kugeln gewartet. "
                        "Und dann war die Kugel Pistazie auch noch winzig. So macht das keinen Spaß.\n</email>",
            "form": "Höchstens 80 Wörter, in der Sie-Form, freundlich und ohne Floskeln. "
                    "Unterschreib mit „Ihr Eisverkäufer vom Stadtpark“.",
            "beispiel": "So klingt eine frühere Antwort von mir:\n<beispiel>\nDanke, dass Sie uns Bescheid geben. "
                        "Die Waffeln waren an dem Tag zu weich, das ärgert mich selbst. "
                        "Beim nächsten Besuch geht eine Kugel auf mich.\n</beispiel>",
            "regeln": "Biete als Entschuldigung eine Kugel gratis beim nächsten Besuch an, aber keinen Rabatt in Euro. "
                      "Wenn dir eine Angabe fehlt, frag nach, statt etwas zu erfinden.",
        },
        "typisch": ["aufgabe", "material"],
        # Ausgedachte, aber typische Antworten auf den typischen und den vollständigen Prompt
        "antwort_typisch": "Sehr geehrte Kundin,\n\nvielen Dank für Ihre Nachricht. Wir bedauern sehr, dass Ihr Besuch "
                           "nicht Ihren Erwartungen entsprochen hat. Ihre Zufriedenheit liegt uns am Herzen, und wir "
                           "nehmen Ihr Feedback zum Anlass, unsere Abläufe zu überprüfen. Wir hoffen, Sie bald wieder "
                           "bei uns begrüßen zu dürfen.\n\nMit freundlichen Grüßen\n[Ihr Name]",
        "antwort_vollstaendig": "Guten Tag,\n\n25 Minuten für zwei Kugeln sind zu lang, und eine winzige Kugel Pistazie "
                                "ärgert mich genauso wie Sie. Am Samstag waren wir nur zu zweit am Stand, das reicht an "
                                "so einem Tag nicht. Beim nächsten Besuch geht eine Kugel auf mich, gern wieder Pistazie, "
                                "dann in voller Größe.\n\nIhr Eisverkäufer vom Stadtpark",
    },
    {
        "id": "aushang",
        "titel": "Aushang für das Sommerfest",
        "bausteine": {
            "aufgabe": "Schreib einen Aushang für unser Sommerfest.",
            "hintergrund": "Die Eisdiele am Stadtpark wird zehn Jahre alt. Der Aushang hängt im Schaufenster, "
                           "die meisten Leute gehen nur kurz daran vorbei.",
            "material": "Termin: Samstag, 12. Juli, 14 bis 18 Uhr. Programm: Kinderschminken, Musik, "
                        "die neue Sorte Zitrone-Basilikum zum Probieren. Jede zweite Kugel kostet die Hälfte.",
            "form": "Eine Überschrift mit höchstens sechs Wörtern, darunter höchstens drei kurze Zeilen. "
                    "Fröhlich, aber nicht kitschig.",
            "beispiel": "Ein Aushang, der gut ankam:\n<beispiel>\nNeu: Mango-Maracuja\nNur im Juli. "
                        "Probieren kostet nichts.\n</beispiel>",
            "regeln": "Keine Emojis und höchstens ein Ausrufezeichen. Alle Angaben müssen aus den Stichpunkten stammen.",
        },
        "typisch": ["aufgabe", "material"],
        "antwort_typisch": "🍦☀️ GROSSES SOMMERFEST! ☀️🍦\n\nLiebe Eisfreunde! Feiert mit uns einen unvergesslichen "
                           "Sommertag voller Spaß, Musik und köstlicher Überraschungen! Am Samstag, den 12. Juli, von "
                           "14 bis 18 Uhr erwartet euch Kinderschminken, tolle Musik und unsere brandneue Sorte "
                           "Zitrone-Basilikum! Und das Beste: Jede zweite Kugel kostet nur die Hälfte!!! Wir freuen "
                           "uns riesig auf euch!",
        "antwort_vollstaendig": "Zehn Jahre Eis am Stadtpark\nSamstag, 12. Juli, 14 bis 18 Uhr\n"
                                "Musik, Kinderschminken, Zitrone-Basilikum zum Probieren\n"
                                "Jede zweite Kugel zum halben Preis!",
    },
    {
        "id": "auswertung",
        "titel": "Verkaufszahlen auswerten",
        "bausteine": {
            "aufgabe": "Fasse die Verkaufszahlen der letzten zehn Tage zusammen.",
            "hintergrund": "Ich will entscheiden, ob sich am Wochenende eine dritte Aushilfe lohnt.",
            "material": "<tabelle>\n" + verkaufstabelle() + "\n</tabelle>",
            "form": "Antworte in höchstens drei Sätzen und nenne die Zahlen, auf die du dich stützt.",
            "beispiel": None,
            "regeln": "Rechne mit einem Rechenwerkzeug, nicht im Kopf. Wenn dir für die Entscheidung eine Angabe fehlt, sag, welche.",
        },
        "typisch": ["aufgabe", "material"],
        # Beide Antworten werden aus den Verkaufszahlen berechnet, siehe auswertung_antwort()
        "antwort_typisch": None,
        "antwort_vollstaendig": None,
    },
]

# Für das Labor: schwache Prompts, bei denen man die fehlenden Bausteine findet
# (Prompt, fehlende Bausteine, Erklärung, verbesserter Prompt)
SCHWACHE_PROMPTS = [
    ("Schreib was über unser neues Eis.", ["aufgabe", "hintergrund", "material", "form"],
     "Was für ein Text: Aushang, Post oder Speisekarte? Welches Eis, für wen, wie lang? Das Modell muss alles raten.",
     "Schreib einen kurzen Text für unsere Speisekarte über die neue Sorte Zitrone-Basilikum. Sie ist ein Sorbet "
     "nur aus Obst, Kräutern, Wasser und Zucker, also vegan, und ziemlich sauer. Zwei Sätze, ohne Werbesprache."),
    ("Übersetz die Karte ins Englische.", ["material", "hintergrund", "regeln"],
     "Die Karte selbst fehlt. Außerdem weiß das Modell nicht, dass die Leser Touristen sind, und nicht, dass Preise "
     "und Sortennamen gleich bleiben sollen.",
     "Übersetz unsere Eiskarte unten ins Englische, für Touristen. Preise bleiben in Euro mit Komma, wie sie sind. "
     "Eigennamen wie Stracciatella nicht übersetzen.\n<karte>\n…\n</karte>"),
    ("Mach den Text besser:\n<text>…</text>", ["aufgabe", "hintergrund", "form", "regeln"],
     "„Besser“ kann kürzer, freundlicher, förmlicher oder fehlerfrei heißen. Ohne Ziel und ohne Leser verändert das "
     "Modell irgendetwas und oft auch Dinge, die bleiben sollten.",
     "Kürz den Text unten auf die Hälfte. Er geht an unsere Aushilfen, also gern locker. Die Uhrzeiten und Namen "
     "müssen genau so bleiben.\n<text>…</text>"),
    ("Wie viel Eis brauche ich am Wochenende?", ["material", "regeln"],
     "Wettervorhersage, Verkaufszahlen und Vorrat kennt das Modell nicht. Ohne sie kann es nur raten, und eine "
     "geratene Zahl klingt genauso überzeugend wie eine richtige.",
     "Für Samstag sind 27 °C angesagt, für Sonntag 25 °C. Mein Modell lautet Kugeln = 10 · Temperatur + 50 · "
     "Wochenende + 20. Wie viele Kugeln brauche ich zusammen? Schreib jeden Rechenschritt auf."),
    ("Fass den Vertrag zusammen.\n<vertrag>… 12 Seiten …</vertrag>", ["hintergrund", "form", "regeln"],
     "Ein Vertrag lässt sich auf viele Arten zusammenfassen. Das Modell weiß nicht, welche Punkte für dich zählen, wie lang "
     "die Zusammenfassung sein soll und dass es Fundstellen nennen soll, damit du jede Aussage nachprüfen kannst.",
     "Ich prüfe einen neuen Liefervertrag für Milch und Sahne. Liste Laufzeit, Kündigungsfrist, Preise und "
     "Strafen bei Lieferausfall auf, je ein Stichpunkt mit der Nummer des Paragrafen. Steht etwas nicht im Vertrag, "
     "schreib „nicht geregelt“.\n<vertrag>…</vertrag>"),
]

# Für das Labor: Suggestivfragen und neutrale Fragen mit ausgedachten, aber typischen Antworten
# (Titel, suggestive Frage, Antwort, neutrale Frage, Antwort, Erklärung)
FRAGEPAARE = [
    ("Eine Idee bewerten",
     "Meine Idee, im Winter Glühwein-Eis zu verkaufen, ist doch super, oder?",
     "Das ist eine tolle Idee! Glühwein-Eis verbindet zwei beliebte Genüsse und macht Ihre Eisdiele auch im Winter "
     "zum Anziehungspunkt. Viele Kunden werden die ausgefallene Sorte lieben.",
     "Ich überlege, im Winter Glühwein-Eis zu verkaufen. Was spricht dafür, was dagegen?",
     "Dafür: Die Sorte fällt auf und passt zur Jahreszeit. Dagegen: Im Winter ist weniger los, und Ihr Stand ist "
     "von November bis Februar geschlossen. Eis mit Alkohol eignet sich außerdem nicht für Kinder, die einen großen Teil "
     "Ihrer Kundschaft ausmachen. "
     "Ein Test auf dem Weihnachtsmarkt wäre ein günstiger erster Schritt.",
     "Die erste Frage gibt die Antwort schon vor. Sprachmodelle neigen dazu, ihrem Gegenüber zuzustimmen "
     "(Gefälligkeit, englisch Sycophancy). Die neutrale Frage lässt Platz für Einwände, etwa die Winterpause, wenn die "
     "Öffnungszeiten in den eigenen Anweisungen stehen."),
    ("Eine Rechnung prüfen",
     "Ich habe 920 : 70 = 13 Behälter gerechnet, das stimmt doch?",
     "Ja, das stimmt: 920 geteilt durch 70 ergibt rund 13. Sie brauchen also 13 Behälter.",
     "Wie viele Behälter zu je 70 Kugeln brauche ich für 920 Kugeln? Schreib den Rechenweg auf.",
     "920 : 70 = 13,14. Mit 13 Behältern hätten Sie nur 910 Kugeln, also 10 zu wenig. Sie brauchen 14 Behälter.",
     "Die erste Frage lädt zum Zustimmen ein, und „rund 13“ klingt plausibel. Mit einer offenen Frage und einem "
     "Rechenweg fällt auf, dass 13 Behälter nicht reichen. Die Aufgabe stammt aus Kapitel 5."),
    ("Bei einer Antwort nachhaken",
     "Bist du sicher? Ich glaube, es sind 13 Behälter.",
     "Sie haben recht, entschuldigen Sie. 13 Behälter reichen aus.",
     "Rechne bitte noch einmal Schritt für Schritt nach: Reichen 13 Behälter zu je 70 Kugeln für 920 Kugeln?",
     "13 · 70 = 910. Das sind 10 Kugeln weniger als 920. 13 Behälter reichen nicht, es bleibt bei 14.",
     "Auf Widerspruch hin geben Modelle eine richtige Antwort manchmal auf. Bitte statt „Bist du sicher?“ um "
     "eine nachprüfbare Rechnung und rechne sie selbst nach."),
]


# ---------- Rechnen ----------

def tokens(text):
    return len(ENC.encode(text, disallowed_special=()))


def woerter(text):
    return len(re.findall(r"[\wÄÖÜäöüß]+(?:[-'][\wÄÖÜäöüß]+)*", text))


def zusammensetzen(aufgabe, auswahl):
    """Die gewählten Bausteine in fester Reihenfolge, getrennt durch Leerzeilen."""
    return "\n\n".join(aufgabe["bausteine"][k] for k, *_ in BAUSTEINE if k in auswahl and aufgabe["bausteine"][k])


def auswertung_antwort():
    """Beide Antworten zur Auswertung, aus den Verkaufszahlen aus Kapitel 1 berechnet.

    Die typische Antwort beschreibt nur das Offensichtliche, die vollständige beantwortet die Frage hinter der Aufgabe.
    """
    tage = KAPITEL_1.TAGE
    we = [(t, k) for t, w, k in tage if w]
    wt = [(t, k) for t, w, k in tage if not w]
    m_we, m_wt = sum(k for _, k in we) / len(we), sum(k for _, k in wt) / len(wt)
    t_we, t_wt = sum(t for t, _ in we) / len(we), sum(t for t, _ in wt) / len(wt)
    kalt, warm = min(tage, key=lambda x: x[2]), max(tage, key=lambda x: x[2])

    def k(x):
        return f"{x:.0f}" if x == round(x) else f"{x:.1f}".replace(".", ",")
    typisch = ("Die Verkaufszahlen der letzten zehn Tage zeigen einen klaren Zusammenhang zwischen Temperatur und Absatz: "
               f"Je wärmer es war, desto mehr Kugeln wurden verkauft. Der niedrigste Wert lag bei {kalt[2]} Kugeln an einem "
               f"kühlen Tag mit {kalt[0]} °C, der höchste bei {warm[2]} Kugeln bei {warm[0]} °C. Insgesamt ist die "
               "Entwicklung erfreulich, und an warmen Tagen ist die Nachfrage besonders hoch.")
    if abs(t_we - t_wt) < 0.05:
        wetter = f"Die Temperaturen lagen an beiden Arten von Tagen im Schnitt bei {k(t_we)} °C"
    else:
        wetter = f"Die Wochenendtage waren im Schnitt {k(t_we)} °C warm, die Werktage {k(t_wt)} °C"
    voll = (f"Am Wochenende haben Sie im Schnitt {k(m_we)} Kugeln verkauft, an Werktagen {k(m_wt)}, also {k(m_we - m_wt)} mehr. "
            f"{wetter}, der Unterschied kommt also vom Wochenende selbst. Ob {k(m_we - m_wt)} Kugeln mehr eine dritte "
            "Aushilfe rechtfertigen, kann ich ohne eine Angabe nicht sagen: wie viele Kugeln zwei Leute pro Stunde schaffen.")
    return typisch, voll, (m_we, m_wt, t_we, t_wt)


def workshop_text():
    """Den Text der Kapitel 1 bis 5 ohne Code, Bilder und HTML, für das Verhältnis von Tokens zu Wörtern."""
    teile = []
    for datei in sorted(DOCS.glob("0[1-5]-*.md")):
        text = datei.read_text(encoding="utf-8")
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\]\([^)]*\)", "]", text)
        teile.append(text)
    return "\n".join(teile)


# ---------- Grafiken ----------

BLAU = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GELB = "#eda100"
LILA = "#8a63d2"
ROSA = "#d6488f"
FLAECHE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_LEISE = "#52514e"
GRAU = "#898781"
GITTER = "#e1e0d9"
ACHSE = "#c3c2b7"

FARBEN = {"aufgabe": BLAU, "hintergrund": AQUA, "material": GELB, "form": LILA, "beispiel": ROSA, "regeln": ORANGE}

plt.rcParams.update(
    {
        # Mehrere Schriften: Fehlt ein Zeichen wie „→“, nimmt matplotlib es aus der nächsten
        "font.family": ["Helvetica Neue", "Arial Unicode MS", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "figure.facecolor": FLAECHE,
        "axes.facecolor": FLAECHE,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }
)


def kasten(ax, x, y, breite, hoehe, titel, unterzeile="", farbe=FLAECHE, rand=ACHSE, textfarbe=TEXT, schrift=12):
    ax.add_patch(FancyBboxPatch((x, y), breite, hoehe, boxstyle="round,pad=0,rounding_size=1.2", facecolor=farbe, edgecolor=rand, linewidth=1.4, zorder=2))
    mitte = x + breite / 2
    if unterzeile:
        ax.text(mitte, y + hoehe - 3.2, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)
        ax.text(mitte, y + (hoehe - 5.5) / 2, unterzeile, ha="center", va="center", color=TEXT_LEISE if textfarbe == TEXT else textfarbe, fontsize=9.5, zorder=3, linespacing=1.35)
    else:
        ax.text(mitte, y + hoehe / 2, titel, ha="center", va="center", color=textfarbe, fontsize=schrift, fontweight="bold", zorder=3)


def pfeil(ax, start, ende, bogen=0.0):
    ax.add_patch(FancyArrowPatch(start, ende, arrowstyle="-|>", mutation_scale=14, color=TEXT_LEISE, linewidth=1.4,
                                 connectionstyle=f"arc3,rad={bogen}", shrinkA=0, shrinkB=0, zorder=1))


def leere_flaeche(breite, hoehe, x, y):
    fig, ax = plt.subplots(figsize=(breite, hoehe))
    ax.set_xlim(-1, x + 1)
    ax.set_ylim(0, y)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def umbrechen(text, breite):
    """Text auf Zeilen von höchstens breite Zeichen verteilen, Zeilenumbrüche im Text bleiben erhalten."""
    zeilen = []
    for absatz in text.split("\n"):
        zeile = ""
        for wort in absatz.split(" "):
            if zeile and len(zeile) + 1 + len(wort) > breite:
                zeilen.append(zeile)
                zeile = wort
            else:
                zeile = f"{zeile} {wort}" if zeile else wort
        zeilen.append(zeile)
    return zeilen


def grafik_bausteine(aufgabe, anzahl):
    """Abschnitt 2: Der vollständige Prompt, nach Bausteinen eingefärbt, mit Tokens pro Baustein."""
    teile = [(k, name, englisch, aufgabe["bausteine"][k]) for k, name, englisch, _ in BAUSTEINE if aufgabe["bausteine"][k]]
    zeilen = [(k, name, englisch, umbrechen(text, 66)) for k, name, englisch, text in teile]
    hoehe = sum(max(len(z), 2) * 2.4 + 2.6 for *_, z in zeilen) + 2
    fig, ax = leere_flaeche(12, hoehe / 10, 120, hoehe)
    y = hoehe - 1
    for k, name, englisch, z in zeilen:
        h = max(len(z), 2) * 2.4 + 1.6
        y -= h
        ax.add_patch(FancyBboxPatch((0, y), 22, h, boxstyle="round,pad=0,rounding_size=1", facecolor=FARBEN[k], edgecolor="none"))
        ax.text(1.6, y + h - 1.9, name, color="#ffffff", fontsize=11.5, fontweight="bold", va="center")
        ax.text(1.6, y + h - 4.2, englisch, color="#ffffff", fontsize=9.5, va="center", alpha=0.9)
        ax.add_patch(FancyBboxPatch((24, y), 84, h, boxstyle="round,pad=0,rounding_size=1", facecolor=FLAECHE, edgecolor=FARBEN[k], linewidth=1.4))
        for i, zeile in enumerate(z):
            ax.text(25.6, y + h - 1.9 - 2.4 * i, zeile, color=TEXT, fontsize=9.5, va="center", family=["DejaVu Sans Mono"])
        ax.text(119.5, y + h / 2, f"{anzahl[k]} Tokens", color=TEXT_LEISE, fontsize=10, ha="right", va="center")
        y -= 1
    fig.savefig(BILDER / "01-bausteine.png")
    plt.close(fig)


def grafik_gespraech():
    """Abschnitt 6: Im Gespräch verbessern oder neu anfangen."""
    fig, ax = leere_flaeche(12, 4.6, 120, 46)
    kasten(ax, 0, 24, 22, 16, "Prompt", "Aufgabe, Hintergrund,\nMaterial, Form …")
    kasten(ax, 34, 24, 24, 16, "Entwurf", "das Modell\nantwortet", farbe=ORANGE, rand=ORANGE, textfarbe="#ffffff")
    kasten(ax, 70, 24, 24, 16, "Rückmeldung", "„Kürzer, und die\nUhrzeit fehlt.“")
    kasten(ax, 98, 24, 22, 16, "Fertig", "wenn das Ergebnis\npasst")
    kasten(ax, 34, 1, 60, 14, "Gespräch verfahren?", "Neues Gespräch: Prompt verbessern und nur das Wichtigste\naus dem alten Gespräch mitnehmen", farbe=GITTER, rand=GITTER)
    pfeil(ax, (22.3, 32), (33.7, 32))
    pfeil(ax, (58.3, 35), (69.7, 35))
    pfeil(ax, (69.7, 29), (58.3, 29))
    pfeil(ax, (94.3, 32), (97.7, 32))
    ax.text(64, 37, "lesen", ha="center", va="bottom", fontsize=9.5, color=TEXT_LEISE)
    ax.text(64, 27, "überarbeiten", ha="center", va="top", fontsize=9.5, color=TEXT_LEISE)
    pfeil(ax, (82, 23.7), (82, 15.3))
    pfeil(ax, (33.7, 8), (11, 23.7), bogen=-0.3)
    fig.savefig(BILDER / "02-gespraech.png")
    plt.close(fig)


def grafik_aufteilen():
    """Abschnitt 7: Eine große Aufgabe in Schritte zerlegen."""
    fig, ax = leere_flaeche(12, 3.2, 120, 32)
    schritte = [
        ("1. Sammeln", "Was soll in die\nSpeisekarte?"),
        ("2. Gliedern", "Reihenfolge und\nÜberschriften"),
        ("3. Schreiben", "Abschnitt für\nAbschnitt"),
        ("4. Prüfen", "Preise und Allergene\ngegen die Liste"),
    ]
    breite, abstand = 25, 6.6
    for i, (titel, text) in enumerate(schritte):
        x = i * (breite + abstand)
        kasten(ax, x, 6, breite, 20, titel, text, farbe=BLAU if i < 3 else ORANGE, rand=BLAU if i < 3 else ORANGE, textfarbe="#ffffff")
        if i:
            pfeil(ax, (x - abstand + 0.4, 16), (x - 0.4, 16))
    ax.text(0, 2, "Jeder Schritt ist ein eigener Prompt. Sein Ergebnis kommt als Material in den nächsten.",
            fontsize=10, color=TEXT_LEISE, va="center")
    fig.savefig(BILDER / "03-aufteilen.png")
    plt.close(fig)


# ---------- Ausgabe ----------

def app_daten(aufgaben_daten):
    daten = {
        "bausteine": [{"id": k, "name": name, "englisch": englisch, "frage": frage} for k, name, englisch, frage in BAUSTEINE],
        "aufgaben": aufgaben_daten,
        "schwach": [{"prompt": p, "fehlt": fehlt, "erklaerung": e, "besser": b} for p, fehlt, e, b in SCHWACHE_PROMPTS],
        "fragepaare": [{"titel": t, "suggestiv": s, "antwortSuggestiv": a_s, "neutral": n, "antwortNeutral": a_n, "erklaerung": e}
                       for t, s, a_s, n, a_n, e in FRAGEPAARE],
    }
    APP_DATEN.write_text(
        "// Erzeugt von docs/beispiele/06_prompts.py, bitte nicht von Hand ändern.\n"
        "window.PROMPT_DATEN = " + json.dumps(daten, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def main():
    BILDER.mkdir(parents=True, exist_ok=True)
    typisch, antwort, (m_we, m_wt, t_we, t_wt) = auswertung_antwort()
    AUFGABEN[2]["antwort_typisch"], AUFGABEN[2]["antwort_vollstaendig"] = typisch, antwort

    aufgaben_daten = []
    for aufgabe in AUFGABEN:
        vorhanden = [k for k, *_ in BAUSTEINE if aufgabe["bausteine"][k]]
        anzahl = {k: tokens(aufgabe["bausteine"][k]) for k in vorhanden}
        # Tokens für jede mögliche Auswahl, genau gezählt (Schlüssel: Bausteine in fester Reihenfolge, mit + verbunden)
        kombinationen = {}
        for n in range(1, len(vorhanden) + 1):
            for auswahl in itertools.combinations(vorhanden, n):
                kombinationen["+".join(auswahl)] = tokens(zusammensetzen(aufgabe, auswahl))
        typisch = zusammensetzen(aufgabe, aufgabe["typisch"])
        voll = zusammensetzen(aufgabe, vorhanden)
        print(f"\n{aufgabe['titel']}")
        for k in vorhanden:
            print(f"   {k:12s} {anzahl[k]:4d} Tokens, {woerter(aufgabe['bausteine'][k]):3d} Wörter")
        print(f"   Nur die Aufgabe: {tokens(aufgabe['bausteine']['aufgabe'])} Tokens")
        print(f"   Typischer Prompt (Aufgabe und Material): {tokens(typisch)} Tokens")
        print(f"   Vollständiger Prompt: {tokens(voll)} Tokens, {woerter(voll)} Wörter")
        for art in ("antwort_typisch", "antwort_vollstaendig"):
            print(f"   {art}: {woerter(aufgabe[art])} Wörter, {tokens(aufgabe[art])} Tokens")
        aufgaben_daten.append({
            "id": aufgabe["id"], "titel": aufgabe["titel"], "bausteine": aufgabe["bausteine"], "typisch": aufgabe["typisch"],
            "antwortTypisch": aufgabe["antwort_typisch"], "antwortVollstaendig": aufgabe["antwort_vollstaendig"],
            "tokens": kombinationen,
        })
        if aufgabe["id"] == "beschwerde":
            grafik_bausteine(aufgabe, anzahl)
            print("\n   Der vollständige Prompt:\n   " + voll.replace("\n", "\n   "))

    print("\nAuswertung der Verkaufszahlen aus Kapitel 1")
    print(f"   Wochenende: {m_we:.1f} Kugeln bei {t_we:.1f} °C, Werktage: {m_wt:.1f} Kugeln bei {t_wt:.1f} °C")
    print("   " + antwort)

    text = workshop_text()
    w, t = woerter(text), tokens(text)
    print(f"\nAbschnitt 5: Tokens pro Wort im Text des Workshops (Kapitel 1 bis 5)")
    print(f"   {w} Wörter, {t} Tokens, {100 * t / w:.0f} Tokens pro 100 Wörter")
    for beispiel in ("Höchstens 80 Wörter.", "Eisdielenstammkundschaft", "Two scoops of chocolate in a cup, please.",
                     "Zwei Kugeln Schokolade im Becher, bitte."):
        print(f"   {woerter(beispiel):2d} Wörter, {tokens(beispiel):2d} Tokens: {beispiel}")

    grafik_gespraech()
    grafik_aufteilen()
    app_daten(aufgaben_daten)
    print(f"\nGrafiken gespeichert in {BILDER}")
    print(f"Daten für die interaktive Seite gespeichert in {APP_DATEN}")


if __name__ == "__main__":
    main()

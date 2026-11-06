# Tokenizer und große Sprachmodelle

In Kapitel 1 hat ein Netz mit neun Parametern gelernt, wie viele Kugeln der Eisverkäufer verkauft. Ein großes Sprachmodell (englisch Large Language Model, kurz LLM) wie ChatGPT, Claude oder Llama ist nach demselben Rezept gebaut: Inputs gewichten, Fehler messen, bergab gehen. Neu sind drei Dinge: Der Input ist Text, die Vorhersage ist das nächste Stück Text, und alles ist um viele Größenordnungen größer.

Dieses Kapitel zeigt die Ideen dahinter, ohne Rechnungen von Hand. Wo man wirklich hineinschauen kann, beim Tokenizer, arbeiten wir mit echten Ausgaben.

| | Das Netz des Eisverkäufers | Ein großes Sprachmodell |
| --- | --- | --- |
| Input | Temperatur und Wochenende | der bisherige Text, zerlegt in Tokens |
| Vorhersage | eine Zahl: verkaufte Kugeln | eine Wahrscheinlichkeit für jedes mögliche nächste Token |
| Parameter | 9 | viele Milliarden |
| Trainingsdaten | 18 Tage | Billionen Tokens |
| Fehler | mittlerer quadratischer Fehler | Kreuzentropie (Abschnitt 1) |
| Lernverfahren | Gradientenabstieg und Backpropagation | dasselbe |

## 1. Eine neue Aufgabe: Was kommt als Nächstes?

Stellen wir uns vor, der Eisverkäufer will nicht mehr wissen, *wie viele* Kugeln er verkauft, sondern *welche Sorte* der nächste Kunde bestellt. Die Antwort ist dann keine Zahl, sondern eine Wahrscheinlichkeit für jede Sorte, etwa Schokolade 50 %, Vanille 30 %, Erdbeere 20 %.

Genau das macht ein Sprachmodell mit Text. Es bekommt den bisherigen Text und gibt für jedes mögliche nächste Stück eine Wahrscheinlichkeit an:

```text
Zwei Kugeln Schokolade im  →  Becher 62 %, Hörnchen 28 %, Glas 8 %, …
```

Die Prozentzahlen sind ausgedacht. Zur Auswahl stehen dabei nicht drei Sorten, sondern das ganze Vokabular des Modells, oft 100.000 Einträge und mehr. Die letzte Schicht liefert für jeden Eintrag eine Punktzahl, und die **Softmax-Funktion** macht daraus Wahrscheinlichkeiten, die zusammen 100 % ergeben.

**Trainingsdaten gibt es umsonst.** Der Eisverkäufer musste seine Verkäufe aufschreiben. Ein Text dagegen ist schon eine Sammlung fertiger Aufgaben, denn an jeder Stelle ist bekannt, wie es weitergeht:

![Der Satz „Zwei Kugeln Schokolade im Becher, bitte.“ ergibt sieben Aufgaben. In Aufgabe 1 bekommt das Modell „Zwei“ und soll „Kugeln“ vorhersagen, in Aufgabe 2 bekommt es „Zwei Kugeln“ und soll „Schokolade“ vorhersagen, und so weiter bis zum Punkt.](bilder/kapitel-2/01-trainingsaufgaben.png)

Niemand muss etwas beschriften, deshalb kann man mit riesigen Textmengen aus Büchern, Webseiten und Programmcode trainieren. Das nennt man **selbstüberwachtes Lernen**.

**Ein neuer Fehler.** Zwischen „Becher“ und „Glas“ gibt es keinen Abstand, der quadratische Fehler passt also nicht mehr. Stattdessen fragt man: Wie viel Wahrscheinlichkeit hat das Modell dem Token gegeben, das tatsächlich kam? Daraus berechnet sich die **Kreuzentropie**, der negative Logarithmus dieser Wahrscheinlichkeit. Man kann sie lesen als: Wie überrascht war das Modell?

![Der Fehler fällt von links oben nach rechts unten. Bei 1 Prozent für das richtige Token ist er 4,61, bei 10 Prozent 2,30, bei 50 Prozent 0,69 und bei 100 Prozent 0.](bilder/kapitel-2/02-kreuzentropie.png)

Wie beim Quadrieren in Kapitel 1 kosten grobe Fehlgriffe besonders viel. Und auch diese Kurve ist glatt, der Gradientenabstieg funktioniert also wie gehabt.

> **Merke:** Ein Sprachmodell sagt immer nur das nächste Token voraus. Jeder Text liefert dafür seine Übungsaufgaben selbst.

## 2. Text in Zahlen: der Tokenizer

Netze rechnen nur mit Zahlen. In Kapitel 1 wurde deshalb aus dem Wochenende eine 0 oder 1. Für Text erledigt das der **Tokenizer**: Er zerlegt Text in Stücke, die **Tokens**, und gibt jedem Stück eine Nummer.

| Zerlegen in      | Vorteil                            | Nachteil                                                               |
| ---------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| einzelne Zeichen | kleines Vokabular, jeder Text geht | sehr lange Folgen                                                      |
| ganze Wörter     | kurze Folgen                       | riesiges Vokabular, unbekannte Wörter gehen gar nicht                  |
| Teilwörter       | der Kompromiss aus beidem          | die Stücke wirken manchmal willkürlich                                 |

Heutige Sprachmodelle verwenden Teilwörter. Das verbreitete Verfahren heißt **Byte-Pair-Encoding** und lernt sein Vokabular selbst: Es beginnt mit einzelnen Zeichen und fasst Schritt für Schritt das Paar zusammen, das am häufigsten nebeneinander vorkommt.

So sieht das aus, wenn man es an sechs Eisbestellungen wie „zwei kugeln schoko im becher“ lernen lässt. Das Skript zu diesem Kapitel rechnet es nach:

![Acht Lernschritte eines Mini-Tokenizers. Zuerst wird „e“ und „l“ zu „el“ zusammengefasst, dann „e“ und „r“ zu „er“. Nach fünf Schritten ist „kugel“ ein eigenes Token, nach acht Schritten auch „kugeln“. „vanille“ zerfällt dagegen noch in einzelne Buchstaben, weil es seltener vorkommt.](bilder/kapitel-2/03-bpe.png)

Häufige Wörter wie „kugeln“ werden schnell zu einem einzigen Token, seltene wie „vanille“ bleiben in Stücken. Echte Tokenizer machen dasselbe mit riesigen Textmengen, beginnen bei den 256 möglichen Bytes und hören erst bei 100.000 oder 200.000 Tokens auf. Weil sie bei Bytes beginnen, können sie jeden Text zerlegen, auch Tippfehler und Emojis.

> **Merke:** Der Tokenizer wird einmal vorab gelernt und bleibt danach fest. Er ist die Brille, durch die das Modell jeden Text sieht.

## 3. Ein echter Tokenizer: tiktoken

Wie das in der Praxis aussieht, zeigt **tiktoken**, die frei verfügbare Tokenizer-Bibliothek von OpenAI, mit zwei ihrer Vokabulare:

| Vokabular     | Tokens im Vokabular | Verwendet unter anderem für |
| ------------- | ------------------- | --------------------------- |
| `o200k_base`  | 200.019             | GPT-4o                      |
| `cl100k_base` | 100.277             | GPT-4 und GPT-3.5           |

Andere Modellfamilien wie Claude, Gemini oder Llama haben eigene Tokenizer, das Prinzip ist dasselbe. So zerlegt `o200k_base` eine Bestellung:

![Der deutsche Satz „Zwei Kugeln Schokolade im Becher, bitte.“ wird in 12 Tokens zerlegt, zum Beispiel ␣Kug, eln, ␣Sch und okolade. Der englische Satz „Two scoops of chocolate in a cup, please.“ braucht 11 Tokens. Unter jedem Token steht seine Nummer im Vokabular.](bilder/kapitel-2/04-tokens.png)

Das Zeichen ␣ steht für ein Leerzeichen. An solchen Zerlegungen sieht man fünf Dinge:

| Beobachtung | Beispiel (`o200k_base`) | Folge |
| --- | --- | --- |
| Das Leerzeichen gehört zum Wort | `␣Kugeln` → `␣Kug` `eln`, aber `Kugeln` am Satzanfang → `K` `ug` `eln` | Dasselbe Wort kann ganz verschieden aussehen |
| Häufig heißt ganz | `␣bitte` und `␣verkauft` sind je ein Token, `Erdbeere` → `E` `rd` `be` `ere` | Seltene Wörter kosten mehr Tokens |
| Zahlen kommen in Blöcken | `12345678` → `123` `456` `78` | Rechnen mit langen Zahlen ist schwer |
| Buchstaben sind unsichtbar | „Erdbeere“ kommt als Nummern 104051, 1464, 512 an | „Wie viele r stecken in Erdbeere?“ ist überraschend schwer |
| Die Nummern bedeuten nichts | `␣Kug` ist 110582, `␣bitte` ist 49309 | Wie bei den Wochentag-Nummern in Kapitel 1 sagt die Größe nichts aus |

**Deutsch ist teurer als Englisch.** Für denselben Inhalt braucht Deutsch mehr Tokens, und ältere Tokenizer mit kleinerem Vokabular trifft es stärker:

![Zwei Balkendiagramme. Mit o200k_base braucht die Bestellung auf Deutsch 12 Tokens, auf Englisch 11, der Satz aus Kapitel 1 auf Deutsch 15, auf Englisch 14. Mit cl100k_base sind es 15 gegen 11 und 20 gegen 14.](bilder/kapitel-2/05-deutsch-englisch.png)

Das hat praktische Folgen, denn bei Sprachmodellen wird fast alles in Tokens gemessen: der Preis einer Anfrage, die Geschwindigkeit der Antwort und wie viel Text das Modell auf einmal überblicken kann (Abschnitt 9).

> **Merke:** Tokens sind die Währung von Sprachmodellen. Wer weiß, wie ein Text zerlegt wird, versteht viele seltsame Schwächen.

## 4. Bedeutung als Zahlenliste: Embeddings

Eine Token-Nummer allein sagt nichts über die Bedeutung. Deshalb bekommt jedes Token eine eigene Liste von Zahlen, ein **Embedding**. Bei großen Modellen hat sie einige tausend Einträge.

Diese Zahlen sind ganz gewöhnliche Parameter: Am Anfang sind sie zufällig, beim Training werden sie mit dem Gradientenabstieg angepasst, wie das Wochenend-Weight in Kapitel 1. Weil das Ziel nur ist, das nächste Token gut vorherzusagen, landen Tokens, die in ähnlichen Zusammenhängen vorkommen, bei ähnlichen Zahlenlisten:

![Schematische Karte mit vier Gruppen von Wörtern. Eissorten wie Vanille, Schokolade, Erdbeere und Stracciatella liegen nah beieinander, ebenso Gefäße wie Becher, Waffel und Hörnchen, Wetterwörter und Wochentage. Zwischen den Gruppen ist viel Abstand.](bilder/kapitel-2/06-embeddings.png)

Die Karte ist schematisch: Echte Embeddings haben tausende Dimensionen, nicht zwei, und die Positionen hier sind ausgedacht. Das Prinzip stimmt aber, und niemand hat es dem Modell gesagt, es ergibt sich aus den Daten.

Allein diese Tabelle kann sehr groß werden. Bei 200.000 Tokens mit je 4.000 Zahlen wären es schon 800 Millionen Parameter. Zusätzlich bekommt das Modell mit, an welcher Stelle ein Token steht. Sonst wäre „Hund beißt Mann“ dasselbe wie „Mann beißt Hund“.

> **Merke:** Embeddings übersetzen Token-Nummern in Bedeutung: Ähnliches liegt nah beieinander.

## 5. Ein Wort reicht nicht: Attention

Zum Vorhersagen genügt das letzte Token selten. Ein Kunde sagt: „Ich nehme Erdbeere. Für meinen Sohn bitte dasselbe.“ Um „dasselbe“ zu verstehen, muss das Modell weit zurückschauen. In einem anderen Satz steht das wichtige Wort an einer ganz anderen Stelle, feste Weights pro Position helfen also nicht.

Die Lösung heißt **Attention** (Aufmerksamkeit). Jedes Token berechnet, wie wichtig ihm jedes frühere Token gerade ist, und holt sich deren Informationen entsprechend gewichtet:

![Die Tokens des Satzes stehen in einer Reihe. Vom Token „elbe“ aus „dasselbe“ führen Bögen zurück zu allen früheren Tokens. Die dicksten Bögen gehen zu „␣Erd“, „be“ und „ere“, also zur Erdbeere, dünne zu „␣Für“ oder „Ich“.](bilder/kapitel-2/07-attention.png)

Die Gewichte in der Grafik sind ausgedacht. Wichtig ist der Unterschied zu Kapitel 1: Diese Gewichtung ist kein fester Parameter, sie wird für jeden Text neu berechnet. Gelernt wird nur, *wie* sie berechnet wird. Man kann sich das wie ein Neuron vorstellen, dessen Weights vom Input abhängen.

Zwei Einzelheiten noch:

- Jedes Token schaut nur auf frühere Tokens, sonst könnte das Modell beim Training die Lösung abschreiben.
- Ein Satz enthält viele Beziehungen gleichzeitig: wer bestellt, was bestellt wird, worauf sich ein Wort bezieht. Deshalb rechnet das Modell mehrere Gewichtungen parallel, die **Attention-Köpfe**, und jeder lernt, auf andere Zusammenhänge zu achten.

> **Merke:** Attention lässt jedes Token auf die früheren schauen, und zwar für jeden Text mit neuen Gewichten.

## 6. Der Transformer

Die Bauweise heutiger Sprachmodelle heißt **Transformer** und wurde 2017 im Artikel „Attention Is All You Need“ vorgestellt. Der Weg vom Text zum nächsten Token sieht so aus:

![Ein Ablauf von links nach rechts: Text, Tokenizer, Embeddings, dann ein Block aus Attention und Neuronen mit Knick, der viele Male hintereinander kommt, dann die Ausgabe mit einer Punktzahl je Token und Softmax, am Ende das nächste Token, zum Beispiel ␣Be für Becher.](bilder/kapitel-2/08-transformer.png)

Das Herzstück ist der Block, und große Modelle stapeln Dutzende davon. Jeder Block hat zwei Teile:

1. **Attention:** Die Tokens tauschen Informationen untereinander aus (Abschnitt 5).
2. **Neuronen mit Knick:** Jedes Token wird für sich weiterverarbeitet. Das ist die verborgene Schicht aus Kapitel 1, nur mit tausenden Neuronen statt zwei. Heutige Modelle verwenden meist einen leicht abgerundeten Knick, die Aufgabe ist dieselbe.

Dazu kommen technische Hilfsmittel, die das Training vieler Schichten stabil halten. Die meisten Parameter stecken in den Neuronen der Blöcke.

Gelernt wird wie in Kapitel 1: Vorhersage berechnen, Fehler messen, mit Backpropagation die Steigungen bestimmen, einen Schritt bergab gehen. Nur geschieht das nicht 30000-mal mit 18 Tagen, sondern für Billionen von Tokens, verteilt auf tausende Spezialchips über Wochen oder Monate.

> **Merke:** Ein Transformer ist ein Stapel aus Attention und den bekannten Neuronen mit Knick, trainiert mit dem bekannten Gradientenabstieg.

## 7. Skalierung: Warum größer besser wird

Ein Sprachmodell lässt sich an drei Stellschrauben vergrößern: **Parameter**, **Daten** (Trainingstokens) und **Rechenaufwand**, der mit den beiden anderen wächst. Wie groß die Unterschiede sind, zeigt eine Achse, auf der jeder Teilstrich tausendmal so viel bedeutet wie der vorige:

![Waagerechte Balken auf einer logarithmischen Achse. Das Netz des Eisverkäufers hat 9 Parameter, Llama 3 in der kleinen Version 8 Milliarden, Chinchilla und Llama 3 in der großen Version je 70 Milliarden, GPT-3 175 Milliarden und Gopher 280 Milliarden.](bilder/kapitel-2/09-skalierung.png)

Drei Erkenntnisse aus der Forschung:

- **Der Fehler sinkt vorhersagbar.** 2020 zeigten Forschende von OpenAI: Mit mehr Parametern, Daten oder Rechenaufwand sinkt der Fehler sehr regelmäßig, nach einem einfachen Potenzgesetz und über mehr als sieben Größenordnungen. Deshalb kann man an kleinen Modellen ausprobieren und recht zuverlässig vorhersagen, wie gut ein großes wird.
- **Daten und Parameter gehören zusammen.** 2022 zeigte DeepMind, dass Modellgröße und Datenmenge gleich stark wachsen sollten: Doppelt so viele Parameter brauchen doppelt so viele Trainingstokens. Das Modell Chinchilla hatte nur ein Viertel der Größe von Gopher (70 statt 280 Milliarden Parameter), bekam bei gleichem Rechenaufwand aber viermal so viele Daten und schnitt durchweg besser ab, auch als GPT-3.
- **Fähigkeiten kommen nebenbei.** Um Text wirklich gut fortzusetzen, muss ein Modell Grammatik, Fakten, Übersetzungen und Programmcode lernen. Niemand bringt ihm das gezielt bei, es steckt in der Aufgabe „nächstes Token“.

**Und die Überanpassung aus Kapitel 1?** Dort lernte ein Modell mit 10 Parametern für 10 Tage nur auswendig. Das gilt weiterhin, entscheidend ist aber das Verhältnis: Ein großes Sprachmodell hat Milliarden Parameter, sieht aber noch viel mehr Tokens, und viele davon nur ein einziges Mal (Llama 3 zum Beispiel über 15 Billionen). Ganz verschwindet das Auswendiglernen trotzdem nicht: Sehr häufige Texte kann ein Modell manchmal wörtlich wiedergeben. Und geprüft wird auch hier an Texten, die es beim Training nicht gesehen hat.

> **Merke:** Größer wird besser, wenn Parameter und Daten gemeinsam wachsen, und zwar erstaunlich vorhersagbar.

## 8. Text erzeugen

Ein Sprachmodell sagt immer nur ein einziges nächstes Token voraus. Längere Antworten entstehen in einer Schleife: vorhersagen, ein Token auswählen, anhängen, wieder von vorn. Das geht so lange, bis das Modell ein spezielles Ende-Token wählt:

![Sechs Durchläufe untereinander. Im ersten wird an „Zwei Kugeln Schokolade im“ das Token „␣Be“ angehängt, im zweiten „cher“, dann ein Komma, „␣bitte“, ein Punkt und zuletzt das Ende-Token.](bilder/kapitel-2/10-erzeugen.png)

Hier sieht man auch: Das Wort „Becher“ braucht zwei Durchläufe, weil es aus den Tokens `␣Be` und `cher` besteht. Für jedes Token läuft das ganze Modell einmal durch. Das ist die Anwendung aus Kapitel 1: Die Weights bleiben fest, das Modell rechnet nur noch.

**Wie wird ausgewählt?** Nimmt das Modell immer das wahrscheinlichste Token, werden Texte eintönig und wiederholen sich. Deshalb wird meist zufällig gezogen, entsprechend den Wahrscheinlichkeiten. Darum gibt dasselbe Modell auf dieselbe Frage nicht immer dieselbe Antwort.

**Die Temperatur** ist eine Stellschraube beim Ziehen. Mit der Temperatur aus Kapitel 1 hat sie nichts zu tun, auch wenn es beim Eisverkäufer gut passt. Vor der Softmax-Funktion werden alle Punktzahlen durch sie geteilt. Der Einfachheit halber sind die Kandidaten in der Grafik ganze Wörter, die Punktzahlen sind ausgedacht:

![Drei Balkendiagramme mit den Wahrscheinlichkeiten für das Wort nach „Zwei Kugeln Schokolade im“. Bei Temperatur 0,5 bekommt Becher 82 Prozent, bei Temperatur 1 sind es 62 Prozent, bei Temperatur 2 nur noch 43 Prozent, und Regen steigt von 0 auf 5 Prozent.](bilder/kapitel-2/11-temperatur.png)

| Temperatur | Wirkung | Geeignet für |
| --- | --- | --- |
| niedrig, nahe 0 | fast immer das wahrscheinlichste Token, vorhersagbar und gleichförmig | Fakten, Code, Zusammenfassungen |
| hoch | Wahrscheinlichkeiten gleichen sich an, abwechslungsreich, aber fehleranfälliger | Ideen sammeln, kreatives Schreiben |

Bei hoher Temperatur hat plötzlich selbst „Zwei Kugeln Schokolade im Regen“ eine echte Chance.

> **Merke:** Text entsteht Token für Token. Die Temperatur bestimmt, wie mutig dabei gezogen wird.

## 9. Vom Textvervollständiger zum Assistenten

Nach dem **Vortraining** mit riesigen Textmengen kann ein Modell Text fortsetzen, ist aber noch kein hilfreicher Assistent. Erst zwei weitere Trainingsschritte machen einen daraus. Die Antworten in der Grafik sind ausgedacht, zeigen aber den typischen Unterschied:

![Drei Stufen nebeneinander: Vortraining mit Billionen Tokens Text, Feinabstimmung mit Beispielgesprächen und menschliches Feedback mit Bewertungen von Antworten. Auf die Frage „Wie viel Eis brauche ich bei 25 °C?“ setzt das vortrainierte Modell den Text mit weiteren Fragen fort, nach der Feinabstimmung antwortet es „Bei 25 °C etwa 290 Kugeln“, nach dem Feedback „An einem Werktag etwa 270 Kugeln, am Wochenende etwa 320“.](bilder/kapitel-2/12-assistent.png)

1. **Feinabstimmung:** Das Modell wird mit vielen Beispielgesprächen weitertrainiert, in denen auf eine Anfrage eine gute Antwort folgt. Das Verfahren bleibt gleich, nur die Daten ändern sich.
2. **Lernen aus menschlichem Feedback:** Menschen vergleichen Antworten des Modells und wählen die bessere, und das Modell wird so weitertrainiert, dass bevorzugte Antworten wahrscheinlicher werden. Wie stark das wirkt, zeigte OpenAI 2022: Die Antworten eines so trainierten Modells mit 1,3 Milliarden Parametern wurden denen von GPT-3 mit 175 Milliarden vorgezogen, obwohl es hundertmal kleiner war.

Auch im Gespräch bleiben die Weights fest, das Modell lernt nicht dazu. Es bekommt bei jeder Antwort das bisherige Gespräch als Text mitgeschickt. Dieser Text hat eine Obergrenze, das **Kontextfenster**, und auch sie wird in Tokens gemessen.

> **Merke:** Vortraining bringt das Wissen, Feinabstimmung und Feedback bringen das hilfreiche Verhalten.

## 10. Grenzen ehrlich benennen

Aus dem, was ein Sprachmodell lernt, folgen auch seine typischen Schwächen:

| Schwäche | Warum | Was hilft |
| --- | --- | --- |
| **Plausibel statt wahr** (Halluzination) | Das Modell lernt, welcher Text wahrscheinlich folgt, nicht welcher stimmt. Wo die Daten wenig hergeben, klingt es trotzdem überzeugend. | Wichtige Aussagen prüfen, Quellen verlangen |
| **Nur, was in den Daten steckt** | Es kennt die Welt nur bis zum Ende seiner Trainingsdaten und übernimmt deren Einseitigkeiten. | Aktuelle Informationen im Gespräch mitgeben |
| **Die Brille des Tokenizers** | Es sieht Tokens, keine Buchstaben oder Ziffern (Abschnitt 3). | Für Zählen und Rechnen ein Werkzeug nutzen |
| **Zufall beim Erzeugen** | Durch das Ziehen bekommt dieselbe Frage verschiedene Antworten (Abschnitt 8). | Niedrige Temperatur, Ergebnis prüfen |

Die Halluzination ist dabei dasselbe Problem wie in Kapitel 1, als die Gerade bei −10 °C selbstbewusst −60 Kugeln vorhersagte: Außerhalb seiner Daten rechnet ein Modell einfach weiter.

> **Merke:** Ein Sprachmodell erzeugt plausiblen Text, nicht zwingend wahren.

## Das Wichtigste in Kürze

- Ein Sprachmodell sagt immer nur das **nächste Token** voraus, als Wahrscheinlichkeit für jedes Token im Vokabular.
- Gemessen wird der Fehler mit der **Kreuzentropie**. Gelernt wird wie in Kapitel 1 mit Gradientenabstieg und Backpropagation.
- Jeder Text liefert seine Trainingsaufgaben selbst. Deshalb kann man mit riesigen Datenmengen trainieren.
- Der **Tokenizer** zerlegt Text in Teilwörter und gibt ihnen Nummern. Er wird vorab gelernt und bleibt danach fest.
- Tokens bestimmen Preis, Geschwindigkeit und Kontextlänge. Deutsch braucht mehr Tokens als Englisch, Zahlen und Buchstaben sieht das Modell nur in Stücken.
- **Embeddings** machen aus jeder Token-Nummer eine gelernte Zahlenliste. Ähnliche Tokens bekommen ähnliche Listen.
- **Attention** lässt jedes Token auf frühere Tokens schauen, mit einer Gewichtung, die für jeden Text neu berechnet wird.
- Ein **Transformer** stapelt viele Blöcke aus Attention und Neuronen mit Knick.
- **Skalierung:** Der Fehler sinkt vorhersagbar mit mehr Parametern, Daten und Rechenaufwand. Parameter und Daten sollten dabei gemeinsam wachsen.
- Beim **Erzeugen** wird Token für Token gezogen. Die **Temperatur** bestimmt, wie vorhersagbar oder abwechslungsreich das Ergebnis wird.
- Aus einem Textvervollständiger wird durch **Feinabstimmung** und **menschliches Feedback** ein Assistent.
- Ein Sprachmodell erzeugt plausiblen Text, nicht zwingend wahren.

## Selbst ausprobieren

In [Labor 2](app/labor-2.html) lässt sich beliebiger Text eintippen und sofort mit beiden Tokenizern zerlegen. Dort arbeitet js-tiktoken, eine Umsetzung von tiktoken für den Browser, die exakt dieselben Tokens liefert. Außerdem enthält das Labor alle Beispiele aus Abschnitt 3 und die Temperatur aus Abschnitt 8 zum freien Einstellen. Wie die Seite zu Kapitel 1 läuft es ohne Internet und ohne Server.

Auch das Skript zerlegt eigene Texte. Dafür wird wieder [uv](https://docs.astral.sh/uv/) benötigt. Beim ersten Start lädt tiktoken die Vokabulare aus dem Internet und speichert sie, danach geht es auch offline. Für einen Workshop sollte man es deshalb vorher einmal starten.

```bash
uv run docs/beispiele/02_tokenizer.py "Drei Kugeln Vanille, bitte."
```

Ohne eigenen Text rechnet das Skript alle Beispiele dieses Kapitels nach, auch den Mini-Tokenizer aus Abschnitt 2, und erzeugt alle Grafiken sowie die Daten für die interaktive Seite neu.

## Quellen

- Vaswani u. a. (2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Kaplan u. a. (2020): [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Brown u. a. (2020): [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- Hoffmann u. a. (2022): [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- Ouyang u. a. (2022): [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Meta (2024): [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- OpenAI: [tiktoken](https://github.com/openai/tiktoken)

# Tokenizer und große Sprachmodelle

In Kapitel 1 hat ein Netz aus drei Neuronen mit neun Parametern gelernt, wie viele Kugeln der Eisverkäufer verkauft. Ein großes Sprachmodell (englisch Large Language Model, kurz LLM) wie ChatGPT, Claude oder Llama ist nach demselben Rezept gebaut: Inputs gewichten, Fehler messen, mit dem Gradientenabstieg bergab gehen. Neu sind drei Dinge: Der Input ist Text, die Vorhersage ist das nächste Stück Text, und alles ist um viele Größenordnungen größer.

Dieses Kapitel bleibt deshalb abstrakter als das erste. Rechnungen von Hand wären bei Milliarden von Parametern sinnlos, wichtig sind die Ideen dahinter. Konkret wird es dort, wo man wirklich hineinschauen kann: beim Tokenizer, mit echten Ausgaben.

| | Das Netz des Eisverkäufers | Ein großes Sprachmodell |
| --- | --- | --- |
| Input | Temperatur und Wochenende | der bisherige Text, zerlegt in Tokens |
| Vorhersage | eine Zahl: verkaufte Kugeln | für jedes mögliche nächste Token eine Wahrscheinlichkeit |
| Parameter | 9 | viele Milliarden |
| Trainingsdaten | 18 Tage | Billionen Tokens |
| Fehler | mittlerer quadratischer Fehler | Kreuzentropie (Abschnitt 1) |
| Lernverfahren | Gradientenabstieg und Backpropagation | dasselbe |

## 1. Eine neue Aufgabe: Was kommt als Nächstes?

Stellen wir uns vor, der Eisverkäufer will nicht mehr wissen, *wie viele* Kugeln er verkauft, sondern *welche Sorte* der nächste Kunde bestellt. Darauf gibt es keine Zahl als Antwort, sondern eine Auswahl. Ein Modell sagt dann für jede Sorte eine Wahrscheinlichkeit voraus, etwa Schokolade 50 %, Vanille 30 %, Erdbeere 20 %. Zusammen ergeben sie immer 100 %.

Genau das macht ein Sprachmodell, nur mit Text. Es bekommt den bisherigen Text und sagt vorher, was als Nächstes kommt:

```text
Zwei Kugeln Schokolade im  →  Becher 62 %, Hörnchen 28 %, Glas 8 %, …
```

Die Prozentzahlen sind hier ausgedacht (mehr dazu in Abschnitt 8). Die Auswahl umfasst dabei nicht drei Sorten, sondern das gesamte Vokabular des Modells, oft 100.000 Einträge und mehr (Abschnitt 3).

**Wie aus dem Netz Wahrscheinlichkeiten werden.** Die letzte Schicht hat für jedes Token im Vokabular ein Ausgabe-Neuron, jedes liefert eine Punktzahl. Eine feste Formel, die **Softmax-Funktion**, macht daraus Wahrscheinlichkeiten: Höhere Punktzahlen bekommen mehr, alle Werte sind positiv, und zusammen ergeben sie 100 %.

**Ein neuer Fehler.** Der quadratische Fehler passt nicht mehr, denn es gibt keinen Abstand zwischen „Becher“ und „Glas“. Stattdessen fragt man: Wie viel Wahrscheinlichkeit hat das Modell dem Token gegeben, das tatsächlich kam? Der Fehler heißt **Kreuzentropie** und ist der negative Logarithmus dieser Wahrscheinlichkeit. Man kann ihn lesen als: Wie überrascht war das Modell?

| Wahrscheinlichkeit für das richtige Token | Fehler |
| ----------------------------------------- | ------ |
| 100 %                                     | 0      |
| 50 %                                      | 0,69   |
| 25 %                                      | 1,39   |
| 10 %                                      | 2,30   |
| 1 %                                       | 4,61   |

Wie beim Quadrieren in Kapitel 1 kosten grobe Fehlgriffe besonders viel: Wer sich sicher ist und falsch liegt, wird hart bestraft. Und auch diese Fehlerkurve ist glatt, der Gradientenabstieg funktioniert also wie gehabt.

**Trainingsdaten, die es umsonst gibt.** Der Eisverkäufer musste seine Verkäufe aufschreiben. Für ein Sprachmodell ist jeder Text schon eine Sammlung fertiger Aufgaben mit Lösung, denn an jeder Stelle ist bekannt, wie es weitergeht:

| Input                          | Richtige Vorhersage |
| ------------------------------ | ------------------- |
| Zwei                           | Kugeln              |
| Zwei Kugeln                    | Schokolade          |
| Zwei Kugeln Schokolade         | im                  |
| Zwei Kugeln Schokolade im      | Becher              |

Niemand muss etwas beschriften. Deshalb kann man mit riesigen Textmengen aus Büchern, Webseiten und Programmcode trainieren. Man nennt das **selbstüberwachtes Lernen**.

## 2. Text in Zahlen: der Tokenizer

Netze rechnen nur mit Zahlen. In Kapitel 1 haben wir deshalb aus dem Wochenende eine 0 oder 1 gemacht. Für Text braucht es dasselbe in groß: Der **Tokenizer** zerlegt Text in Stücke, die **Tokens**, und gibt jedem Stück eine Nummer.

Für die Größe der Stücke gibt es drei Möglichkeiten:

| Zerlegen in      | Vorteil                                | Nachteil                                                                 |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| einzelne Zeichen | kleines Vokabular, jeder Text geht     | sehr lange Folgen, das Modell muss jedes Wort erst zusammensetzen         |
| ganze Wörter     | kurze Folgen                           | riesiges Vokabular, unbekannte Wörter gehen gar nicht, „Kugel“ und „Kugeln“ haben nichts miteinander zu tun |
| Teilwörter       | der Kompromiss aus beidem              | die Stücke wirken manchmal willkürlich                                    |

Heutige Sprachmodelle verwenden Teilwörter. Das verbreitete Verfahren heißt **Byte-Pair-Encoding** und lernt sein Vokabular selbst aus einer großen Textsammlung:

1. Mit den 256 möglichen Bytes beginnen. Daraus lässt sich jeder Text zusammensetzen.
2. Das Paar benachbarter Tokens suchen, das in der Textsammlung am häufigsten vorkommt.
3. Dieses Paar zu einem neuen Token zusammenfassen.
4. Die Schritte 2 und 3 wiederholen, bis das Vokabular die gewünschte Größe hat.

Häufige Wörter werden so zu einem einzigen Token, seltene zerfallen in mehrere Stücke. Weil das Verfahren bei Bytes beginnt, gibt es keinen Text, den es nicht zerlegen kann, auch keine Tippfehler oder Emojis.

Der Tokenizer wird einmal vor dem eigentlichen Modell gelernt und danach nicht mehr verändert. Er ist gewissermaßen die feste Brille, durch die das Modell jeden Text sieht.

## 3. Ein echter Tokenizer: tiktoken

Wie das in der Praxis aussieht, zeigt **tiktoken**, die frei verfügbare Tokenizer-Bibliothek von OpenAI. Wir verwenden zwei ihrer Vokabulare:

| Vokabular     | Tokens im Vokabular | Verwendet unter anderem für |
| ------------- | ------------------- | --------------------------- |
| `o200k_base`  | 200.019             | GPT-4o                      |
| `cl100k_base` | 100.277             | GPT-4 und GPT-3.5           |

Andere Modellfamilien wie Claude, Gemini oder Llama haben eigene Tokenizer. Die genauen Zahlen unterscheiden sich, das Prinzip ist dasselbe.

So zerlegt `o200k_base` eine Bestellung auf Deutsch und auf Englisch:

![Der deutsche Satz „Zwei Kugeln Schokolade im Becher, bitte.“ wird in 12 Tokens zerlegt, zum Beispiel ␣Kug, eln, ␣Sch und okolade. Der englische Satz „Two scoops of chocolate in a cup, please.“ braucht 11 Tokens. Unter jedem Token steht seine Nummer im Vokabular.](bilder/kapitel-2/01-tokens.png)

Das Zeichen ␣ steht für ein Leerzeichen. An den Beispielen lässt sich einiges ablesen.

**Das Leerzeichen gehört zum Wort.** `␣Kugeln` mit Leerzeichen davor wird zu `␣Kug` und `eln`, am Satzanfang ohne Leerzeichen zu `K`, `ug` und `eln`. `␣Eis` ist ein einziges Token, `Eis` am Anfang sind zwei. Auch Großschreibung zählt: `␣KUGELN` wird zu `␣K`, `UG`, `EL` und `N`.

**Häufig heißt ganz.** Wörter, die in den Trainingstexten oft vorkommen, sind ein einziges Token, etwa `␣verkauft` oder `␣bitte`. Seltene Wörter zerfallen, und die Stücke sind nicht unbedingt Silben:

| Text                                         | Tokens (`o200k_base`)                                            |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `Erdbeere`                                   | `E` `rd` `be` `ere`                                              |
| `Eisverkäufer`                               | `E` `is` `ver` `kä` `ufer`                                       |
| `Donaudampfschifffahrtsgesellschaftskapitän` | 13 Tokens, von `D` `ona` `ud` bis `gesellschaft` `skap` `it` `än` |

**Zahlen kommen in Blöcken.** `12345678` wird zu `123`, `456` und `78`, die Jahreszahl `2025` zu `202` und `5`. Das Modell sieht also nicht die einzelnen Ziffern an ihrer Stelle. Das ist ein Grund, warum Sprachmodelle beim Rechnen mit langen Zahlen unsicher sind.

**Buchstaben sind unsichtbar.** Die Frage „Wie viele r stecken in Erdbeere?“ kommt beim Modell als `␣Erd`, `be` und `ere` an, also als die Nummern 104051, 1464 und 512. Einzelne Buchstaben sieht es nicht. Dass `␣Erd` ein r enthält, muss es aus Trainingstexten gelernt haben, in denen Wörter buchstabiert wurden. Deshalb scheitern Modelle an solchen scheinbar einfachen Aufgaben überraschend oft.

**Deutsch ist teurer als Englisch.** Für denselben Inhalt braucht Deutsch mehr Tokens, und ältere Tokenizer mit kleinerem Vokabular trifft es stärker:

| Satz                                                    | `o200k_base` | `cl100k_base` |
| ------------------------------------------------------- | ------------ | ------------- |
| Zwei Kugeln Schokolade im Becher, bitte.                | 12           | 15            |
| Two scoops of chocolate in a cup, please.               | 11           | 11            |
| Der Eisverkäufer verkauft bei 25 Grad 290 Kugeln.       | 15           | 20            |
| The ice cream vendor sells 290 scoops at 25 degrees.    | 14           | 14            |

Das hat praktische Folgen, denn bei Sprachmodellen wird fast alles in Tokens gemessen: der Preis einer Anfrage, die Geschwindigkeit der Antwort und wie viel Text das Modell auf einmal überblicken kann (Abschnitt 9).

**Die Nummern bedeuten nichts.** `␣Kug` hat die Nummer 110582, `␣bitte` die 49309. Diese Nummern sind nur Plätze in einer Liste, so wie die Wochentag-Nummern 1 bis 7 aus Kapitel 1. Dass 110582 größer ist als 49309, sagt nichts aus. Wie das Modell trotzdem mit Bedeutung rechnen kann, zeigt der nächste Abschnitt.

## 4. Bedeutung als Zahlenliste: Embeddings

Beim Wochenende war die Lösung eine 0 oder 1. Für 200.000 Tokens hieße das: eine Liste mit 200.000 Stellen, von denen genau eine 1 ist. Das funktioniert, sagt aber nichts darüber aus, welche Tokens einander ähnlich sind. „Vanille“ wäre von „Schokolade“ genauso weit entfernt wie von „Steuererklärung“.

Stattdessen bekommt jedes Token eine eigene Liste von Zahlen, ein **Embedding**. Die Liste hat bei großen Modellen einige tausend Einträge. Diese Zahlen sind ganz gewöhnliche Parameter: Am Anfang sind sie zufällig, und beim Training werden sie mit dem Gradientenabstieg angepasst wie das Wochenend-Weight in Kapitel 1. Allein diese Tabelle kann sehr groß sein. Bei 200.000 Tokens mit je 4.000 Zahlen wären es schon 800 Millionen Parameter.

Weil das Training nur das Ziel hat, das nächste Token gut vorherzusagen, landen Tokens, die in ähnlichen Zusammenhängen vorkommen, bei ähnlichen Zahlenlisten. Eissorten liegen nah beieinander, Gefäße wie Becher und Waffel ebenfalls. Niemand hat dem Modell das gesagt, es ergibt sich aus den Daten.

Zusätzlich bekommt das Modell mit, an welcher Stelle im Text ein Token steht. Sonst wäre „Hund beißt Mann“ dasselbe wie „Mann beißt Hund“.

## 5. Ein Wort reicht nicht: Attention

Zum Vorhersagen genügt das letzte Token selten. Ein Kunde sagt:

```text
Ich nehme Erdbeere. Für meinen Sohn bitte dasselbe.
```

Um zu verstehen, was „dasselbe“ heißt, muss das Modell mehrere Tokens zurückschauen, zu `␣Erd` `be` `ere`. In einem anderen Satz steht das wichtige Wort an einer ganz anderen Stelle. Feste Weights pro Position, wie wir sie aus Kapitel 1 kennen, können das nicht leisten.

Die Lösung heißt **Attention** (Aufmerksamkeit). Für jedes Token berechnet das Modell, wie wichtig ihm jedes frühere Token gerade ist, und mischt deren Informationen entsprechend gewichtet in sein eigenes Embedding. Bei `␣dass` `elbe` bekommt `␣Erd` `be` `ere` viel Gewicht, `␣Für` wenig.

Der entscheidende Unterschied zu Kapitel 1: Diese Gewichtung ist kein fester Parameter. Sie wird für jeden Text neu ausgerechnet, aus den Embeddings der beteiligten Tokens. Fest und gelernt ist nur, *wie* das Modell diese Gewichtung ausrechnet. Man kann sich das wie ein Neuron vorstellen, dessen Weights selbst vom Input abhängen.

Damit die Vorhersage ehrlich bleibt, darf jedes Token nur nach vorne schauen, also auf frühere Tokens. Und weil ein Satz viele Beziehungen gleichzeitig enthält (wer bestellt, was bestellt wird, worauf sich ein Wort bezieht), rechnet das Modell mehrere Attention-Gewichtungen parallel, die **Attention-Köpfe**. Jeder Kopf lernt im Training, auf andere Zusammenhänge zu achten.

## 6. Der Transformer

Die Bauweise heutiger Sprachmodelle heißt **Transformer**. Sie wurde 2017 im Artikel „Attention Is All You Need“ vorgestellt. Ein Transformer besteht aus vielen gleich aufgebauten Blöcken hintereinander, und jeder Block hat zwei Teile:

1. **Attention:** Die Tokens tauschen Informationen untereinander aus (Abschnitt 5).
2. **Ein Netz aus Neuronen mit Knick:** Jedes Token wird für sich weiterverarbeitet. Das ist genau die verborgene Schicht aus Kapitel 1, nur mit tausenden Neuronen statt zwei. Statt des harten Knicks verwenden heutige Modelle meist eine leicht abgerundete Variante, die Aufgabe ist dieselbe.

Dazu kommen technische Hilfsmittel, die das Training vieler Schichten stabil halten. Der ganze Weg vom Text zur Vorhersage sieht so aus:

```text
Text
  → Tokenizer:            Tokens und ihre Nummern
  → Embeddings:           eine Zahlenliste pro Token
  → Block 1 … Block N:    jeweils Attention und Neuronen mit Knick
  → Ausgabe-Schicht:      eine Punktzahl pro Token im Vokabular
  → Softmax:              Wahrscheinlichkeiten für das nächste Token
```

Große Modelle stapeln Dutzende solcher Blöcke. Die meisten Parameter stecken in den Neuronen der Blöcke, ein weiterer Teil in Attention und Embeddings.

Gelernt wird wie in Kapitel 1: Vorhersage berechnen, Fehler messen (hier die Kreuzentropie), mit Backpropagation die Steigung für jeden Parameter bestimmen und einen kleinen Schritt bergab gehen. Nur geschieht das nicht 30000-mal mit 18 Tagen, sondern für Billionen von Tokens, verteilt auf tausende Spezialchips über Wochen oder Monate.

## 7. Skalierung: Warum größer besser wird

Ein Sprachmodell lässt sich an drei Stellschrauben vergrößern:

- **Parameter:** wie viele Weights und Biases das Modell hat
- **Daten:** mit wie vielen Tokens es trainiert wird
- **Rechenaufwand:** wie viele Rechenoperationen das Training insgesamt braucht. Er wächst mit Parametern und Daten zugleich.

Einige Größenordnungen zum Vergleich:

| Modell                 | Jahr | Parameter       | Trainingsdaten              |
| ---------------------- | ---- | --------------- | --------------------------- |
| Netz des Eisverkäufers | –    | 9               | 18 Tage                     |
| GPT-3                  | 2020 | 175 Milliarden  | –                           |
| Chinchilla             | 2022 | 70 Milliarden   | viermal so viel wie Gopher  |
| Llama 3                | 2024 | 8 und 70 Milliarden | über 15 Billionen Tokens |

**Der Fehler sinkt vorhersagbar.** 2020 zeigten Forschende von OpenAI, dass der Fehler eines Sprachmodells sehr regelmäßig sinkt, wenn man Modellgröße, Datenmenge oder Rechenaufwand erhöht. Der Zusammenhang folgt einem einfachen Potenzgesetz und hält über mehr als sieben Größenordnungen. Diese **Skalierungsgesetze** sind der Grund, warum Firmen Milliarden in größere Modelle investieren: Man kann an kleinen Modellen ausprobieren und recht zuverlässig vorhersagen, wie gut ein großes werden wird.

**Daten und Parameter gehören zusammen.** 2022 untersuchte DeepMind, wie man ein festes Rechenbudget am besten aufteilt. Das Ergebnis: Modellgröße und Datenmenge sollten gleich stark wachsen. Verdoppelt man die Parameter, sollte man auch die Trainingstokens verdoppeln. Das Modell Chinchilla hatte mit 70 Milliarden Parametern nur ein Viertel der Größe des Vorgängers Gopher (280 Milliarden), bekam bei gleichem Rechenaufwand aber viermal so viele Daten. Es schnitt durchweg besser ab als Gopher und auch als das größere GPT-3. Viele frühere Modelle waren also zu groß für ihre Datenmenge.

**Und die Überanpassung?** In Kapitel 1 (Abschnitt 7) hat sich gezeigt: Mehr Parameter sind nicht automatisch besser, mit 10 Parametern für 10 Tage lernt das Modell nur auswendig. Das gilt weiterhin. Entscheidend ist aber nicht die Zahl der Parameter allein, sondern ihr Verhältnis zur Datenmenge. Ein großes Sprachmodell hat zwar Milliarden Parameter, bekommt aber noch viel mehr Tokens zu sehen, und viele davon nur ein einziges Mal. Ganz verschwindet das Auswendiglernen trotzdem nicht: Texte, die in den Daten sehr oft vorkommen, kann ein Modell manchmal wörtlich wiedergeben. Und geprüft wird auch hier an Texten, die das Modell beim Training nicht gesehen hat.

**Was mit der Größe kommt.** Um Text wirklich gut fortzusetzen, muss ein Modell vieles nebenbei lernen: Grammatik, Fakten, Zusammenhänge, Übersetzungen, Programmcode. Niemand hat ihm das gezielt beigebracht, es steckt in der Aufgabe „nächstes Token“. Mit wachsender Größe werden diese Fähigkeiten deutlich besser.

## 8. Text erzeugen

Ein Sprachmodell sagt immer nur ein einziges nächstes Token voraus. Längere Antworten entstehen durch eine Schleife:

1. Den bisherigen Text in Tokens zerlegen und durch das Modell schicken.
2. Aus den Wahrscheinlichkeiten ein Token auswählen.
3. Dieses Token an den Text anhängen.
4. Wieder bei Schritt 1 beginnen, bis das Modell ein spezielles Ende-Token wählt.

Für jedes einzelne Token läuft also das ganze Modell einmal durch. Das ist die Anwendung aus Kapitel 1: Die Weights bleiben fest, das Modell rechnet nur noch Vorhersagen aus.

**Wie wird ausgewählt?** Nimmt das Modell immer das wahrscheinlichste Token, werden die Texte eintönig und wiederholen sich gern. Deshalb wird meist zufällig gezogen, und zwar entsprechend den Wahrscheinlichkeiten: Ein Token mit 62 % kommt oft dran, eines mit 8 % gelegentlich. Deshalb gibt dasselbe Modell auf dieselbe Frage nicht immer dieselbe Antwort.

**Die Temperatur.** Mit der Temperatur aus Kapitel 1 hat diese Temperatur nichts zu tun, auch wenn es beim Eisverkäufer gut passt. Sie ist eine Stellschraube beim Ziehen: Vor der Softmax-Funktion werden alle Punktzahlen durch die Temperatur geteilt.

![Drei Balkendiagramme mit den Wahrscheinlichkeiten für das Wort nach „Zwei Kugeln Schokolade im“. Bei Temperatur 0,5 bekommt Becher 82 Prozent, bei Temperatur 1 sind es 62 Prozent, bei Temperatur 2 nur noch 43 Prozent, und Regen steigt von 0 auf 5 Prozent.](bilder/kapitel-2/02-temperatur.png)

Die Punktzahlen in der Grafik sind ausgedacht, die Wirkung ist echt:

- **Niedrige Temperatur:** Das wahrscheinlichste Token wird noch wahrscheinlicher. Die Antworten werden vorhersagbarer und gleichförmiger. Bei einer Temperatur nahe 0 wird fast immer dasselbe Token gewählt.
- **Hohe Temperatur:** Die Wahrscheinlichkeiten gleichen sich an. Die Antworten werden abwechslungsreicher, aber auch fehleranfälliger. Plötzlich hat selbst „Zwei Kugeln Schokolade im Regen“ eine echte Chance.

## 9. Vom Textvervollständiger zum Assistenten

Nach dem Training mit riesigen Textmengen, dem **Vortraining**, kann ein Modell Texte fortsetzen, ist aber noch kein hilfreicher Assistent. Auf eine Frage antwortet es vielleicht mit weiteren Fragen, weil Fragen in seinen Trainingstexten oft in Listen stehen. Zwei weitere Trainingsschritte ändern das:

1. **Feinabstimmung:** Das Modell wird mit vielen Beispielgesprächen weitertrainiert, in denen auf eine Anfrage eine gute Antwort folgt. Dabei ändert sich nicht das Verfahren, nur die Daten.
2. **Lernen aus menschlichem Feedback:** Menschen vergleichen verschiedene Antworten des Modells und wählen die bessere. Das Modell wird dann so weitertrainiert, dass bevorzugte Antworten wahrscheinlicher werden. Wie stark das wirkt, zeigte OpenAI 2022: Die Antworten eines so trainierten Modells mit 1,3 Milliarden Parametern wurden denen von GPT-3 mit 175 Milliarden Parametern vorgezogen, obwohl es hundertmal kleiner war.

Auch im Gespräch mit einem Assistenten gilt, was in Kapitel 1 über Training und Anwendung gesagt wurde: Die Weights bleiben fest. Das Modell lernt nicht aus dem Gespräch. Es bekommt bei jeder Antwort das bisherige Gespräch als Text mitgeschickt, und dieser Text hat eine Obergrenze, das **Kontextfenster**. Auch das wird in Tokens gemessen.

## 10. Grenzen ehrlich benennen

Aus dem, was ein Sprachmodell lernt, folgen auch seine typischen Schwächen.

- **Plausibel statt wahr.** Das Modell lernt, welcher Text wahrscheinlich folgt, nicht welcher Text stimmt. Meist fällt beides zusammen. Wo die Trainingsdaten aber wenig hergeben, erzeugt es trotzdem flüssigen, überzeugend klingenden Text, der falsch sein kann. Man nennt das **Halluzination**. Es ist dasselbe Problem wie in Kapitel 1, als die Gerade bei −10 °C selbstbewusst −60 Kugeln vorhersagte: Außerhalb seiner Daten rechnet ein Modell einfach weiter.
- **Nur, was in den Daten steckt.** Ein Modell kennt die Welt nur bis zum Ende seiner Trainingsdaten. Spätere Ereignisse kennt es nicht, es sei denn, man gibt sie ihm im Gespräch mit. Auch Einseitigkeiten und Vorurteile in den Daten übernimmt es.
- **Die Brille des Tokenizers.** Buchstaben zählen, Wörter rückwärts schreiben oder mit langen Zahlen rechnen fällt schwer, weil das Modell Tokens sieht und keine einzelnen Zeichen (Abschnitt 3).
- **Zufall beim Erzeugen.** Durch das Ziehen (Abschnitt 8) kann dieselbe Frage verschiedene Antworten bekommen. Für Aufgaben, bei denen es auf genau eine richtige Antwort ankommt, sollte man das Ergebnis prüfen.

## Das Wichtigste in Kürze

- Ein Sprachmodell sagt immer nur das **nächste Token** voraus, als Wahrscheinlichkeit für jedes Token im Vokabular.
- Gemessen wird der Fehler mit der **Kreuzentropie**: Wie wenig Wahrscheinlichkeit bekam das richtige Token? Gelernt wird wie in Kapitel 1 mit Gradientenabstieg und Backpropagation.
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

Ohne eigenen Text rechnet das Skript alle Beispiele dieses Kapitels nach und erzeugt die Grafiken sowie die Daten für die interaktive Seite neu.

## Quellen

- Vaswani u. a. (2017): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Kaplan u. a. (2020): [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Brown u. a. (2020): [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- Hoffmann u. a. (2022): [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- Ouyang u. a. (2022): [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- Meta (2024): [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- OpenAI: [tiktoken](https://github.com/openai/tiktoken)

# Semantische Suche und RAG

In Kapitel 2 waren Embeddings ein Nebenprodukt: Zahlenlisten, die ein Sprachmodell lernt, um das nächste Token vorherzusagen. Dieses Kapitel macht sie selbst zum Werkzeug. Wenn ähnliche Bedeutung zu ähnlichen Zahlen führt, kann man nach Bedeutung suchen statt nach Wörtern. Das heißt **semantische Suche**.

Damit lässt sich auch eine Schwäche aus Kapitel 2 angehen: Ein Sprachmodell weiß nur, was in seinen Trainingsdaten steckt. Sucht man passende Texte heraus und gibt sie ihm mit, kann es mit Wissen antworten, das es nie gelernt hat. Dieses Verfahren heißt **RAG**.

| | Kapitel 2: Sprachmodell | Kapitel 3: Embedding-Modell |
| --- | --- | --- |
| Input | der bisherige Text | ein kurzer Text, etwa eine Frage oder ein Absatz |
| Ausgabe | Wahrscheinlichkeiten für das nächste Token | eine Zahlenliste für den ganzen Text |
| Wozu | Text erzeugen | Texte vergleichen, suchen, sortieren |
| Im Beispiel | Llama 3, GPT-3 und andere | paraphrase-multilingual-MiniLM-L12-v2, 118 Millionen Parameter |

Alle Zahlen in diesem Kapitel sind echt. Das Skript zum Kapitel rechnet sie mit dem Modell aus der Tabelle nach, und Labor 3 lädt dasselbe Modell in den Browser.

## 1. Suchen nach Bedeutung

Der Eisverkäufer hat ein Handbuch für seinen Stand geschrieben: 24 kurze Abschnitte über Sorten, Preise, Öffnungszeiten und Allergien. Es ist wie alle Daten im Workshop erfunden. Ein Kunde fragt:

```text
Mir tut der Kopf weh, wenn ich Eis esse.
```

Im Handbuch steht die Antwort unter „Hirnfrost“:

```text
Hirnfrost: Wer sehr kaltes Eis zu schnell isst, bekommt manchmal kurz stechende Kopfschmerzen. Langsam essen hilft.
```

Eine **Stichwortsuche** zählt, wie viele Wörter der Frage in einem Abschnitt vorkommen. Füllwörter wie „der“ oder „ich“ lässt sie weg. Übrig bleiben „mir“, „tut“, „kopf“, „weh“, „eis“ und „esse“. Im Hirnfrost-Abschnitt kommt davon nur „eis“ vor. „Kopf“ ist nicht „Kopfschmerzen“, und „esse“ ist nicht „isst“. Das Wort „Eis“ steht aber in sieben Abschnitten, und alle sieben liegen gleichauf.

Die semantische Suche dagegen setzt „Hirnfrost“ mit großem Abstand an die Spitze:

![Zwei Balkendiagramme für die Frage „Mir tut der Kopf weh, wenn ich Eis esse.“ Links die semantische Suche: Hirnfrost liegt mit 0,81 weit vorn, danach folgen Auftauen mit 0,49 und Glutenfrei mit 0,43. Rechts die Stichwortsuche: sechs Abschnitte haben je ein gemeinsames Wort, Hirnfrost steht erst an dritter Stelle.](bilder/kapitel-3/04-suche.png)

So geht es mit vielen Fragen, die Menschen tatsächlich stellen:

| Frage | Im Handbuch steht | Stichwortsuche | Semantische Suche |
| --- | --- | --- | --- |
| Kann ich mit dem Handy bezahlen? | „… kontaktlos mit dem Smartphone“ | findet es über „bezahlen“ | findet es (0,80) |
| Darf mein Hund mit? | „Hunde sind willkommen.“ | findet nichts | findet es (0,61) |
| Bekomme ich eine Kugel umsonst? | „… ist die nächste Kugel gratis“ | wählt „Zucker“ | findet es (0,51) |
| Do you have vegan ice cream? | „Alle Fruchtsorbets sind vegan …“ | findet es über „vegan“ | findet es (0,51) |

Die Zahlen in Klammern messen die Ähnlichkeit. Wie sie entstehen, zeigen die nächsten beiden Abschnitte.

> **Merke:** Die Stichwortsuche vergleicht Buchstaben. Die semantische Suche vergleicht Bedeutungen.

## 2. Ein Text, ein Pfeil

Die Idee aus Kapitel 2 lässt sich auf ganze Texte übertragen: Jeder Text bekommt eine Liste von Zahlen, sein Embedding. Texte mit ähnlicher Bedeutung sollen ähnliche Listen bekommen.

Zum Nachrechnen genügen zwei Zahlen pro Text. Die erste soll messen, wie sehr es um Geld geht, die zweite, wie sehr um Gesundheit. Diese Werte sind ausgedacht:

| | Text | Zahl 1: Geld | Zahl 2: Gesundheit |
| --- | --- | --- | --- |
| F | Frage: „Was kostet eine Kugel?“ | 3 | 1 |
| P | Abschnitt „Preise: Eine Kugel kostet 1,80 Euro …“ | 6 | 2 |
| H | Abschnitt „Hirnfrost: Wer sehr kaltes Eis …“ | 1 | 3 |

Zwei Zahlen sind ein Punkt in einem Koordinatensystem. Zeichnet man vom Nullpunkt einen Pfeil zu diesem Punkt, erhält man einen **Vektor**:

![Drei Pfeile vom Nullpunkt aus. Der blaue Pfeil F zeigt auf (3, 1), der orange Pfeil P auf (6, 2) in genau dieselbe Richtung, nur doppelt so weit. Der grüne Pfeil H zeigt auf (1, 3), steil nach oben. Zwischen F und H liegt ein Winkel von 53 Grad. Der Abstand von F zu H beträgt 2,83, der von F zu P 3,16.](bilder/kapitel-3/01-pfeile.png)

Der Preise-Abschnitt zeigt in genau dieselbe Richtung wie die Frage, sein Pfeil ist nur länger, weil er mehr über Geld sagt. Der Hirnfrost-Abschnitt zeigt in eine ganz andere Richtung. Für die Suche zählt also die Richtung.

**Echte Embeddings** funktionieren genauso, nur mit viel mehr Zahlen. Das Modell aus diesem Kapitel macht aus „Was kostet eine Kugel?“ eine Liste mit 384 Zahlen. Sie beginnt so:

```text
0,053   0,088   −0,050   0,001   0,034   −0,033   …
```

Anders als im Spielzeugbeispiel bedeutet keine dieser Zahlen etwas für sich allein, so wie die Weights eines Neurons in Kapitel 1. Bedeutung hat nur das Muster aller 384 Zahlen zusammen. Sichtbar wird das, wenn man jede Zahl als Farbstreifen zeichnet:

![Drei Farbstreifen mit je 384 schmalen Strichen in Blau und Orange. Die Streifen für die Frage „Was kostet eine Kugel?“ und für den Abschnitt „Preise“ haben ein ähnliches Muster, ihr Kosinus ist 0,85. Der Streifen für „Hirnfrost“ sieht anders aus, sein Kosinus zur Frage ist −0,04.](bilder/kapitel-3/02-zahlenlisten.png)

Einen Pfeil in 384 Dimensionen kann sich niemand vorstellen. Rechnen kann man damit trotzdem genau wie mit zwei Zahlen.

> **Merke:** Ein Embedding macht aus einem ganzen Text einen Pfeil. Ähnliche Bedeutung heißt: ähnliche Richtung.

## 3. Ähnlichkeit messen: der Kosinus

Wie ähnlich sind zwei Richtungen? Das misst der **Kosinus** des Winkels zwischen zwei Pfeilen. Ausrechnen lässt er sich in drei Schritten, ganz ohne Winkelmesser:

1. **Skalarprodukt:** Die Zahlen an gleicher Stelle multiplizieren und alles addieren.
2. **Längen:** Für jeden Pfeil die Zahlen quadrieren, addieren und die Wurzel ziehen (der Satz des Pythagoras).
3. **Teilen:** Skalarprodukt geteilt durch das Produkt der beiden Längen.

Für die Frage F und den Preise-Abschnitt P:

```text
Skalarprodukt:  3 · 6 + 1 · 2           = 20
Längen:         |F| = √(3² + 1²) = √10  = 3,16
                |P| = √(6² + 2²) = √40  = 6,32
Kosinus:        20 / (3,16 · 6,32)      = 1,00
```

Und für die Frage F und den Hirnfrost-Abschnitt H:

```text
Skalarprodukt:  3 · 1 + 1 · 3           = 6
Längen:         |F| = 3,16,  |H| = √10  = 3,16
Kosinus:        6 / (3,16 · 3,16)       = 0,60
```

Der Preise-Abschnitt gewinnt mit 1,00, das ist der größtmögliche Wert. Was die Werte bedeuten:

| Kosinus | Winkel | Bedeutung |
| --- | --- | --- |
| 1 | 0° | gleiche Richtung |
| 0 | 90° | nichts gemeinsam |
| −1 | 180° | entgegengesetzte Richtung |

**Warum nicht einfach den Abstand messen?** Die Spitze von H liegt mit 2,83 näher an F als die Spitze von P mit 3,16. Nach dem Abstand wäre der Hirnfrost-Abschnitt die bessere Antwort auf die Preisfrage, und das ist falsch. Der Abstand wird von der Länge der Pfeile gestört, der Kosinus nicht.

Viele Embedding-Modelle bringen alle Pfeile ohnehin auf die Länge 1, auch das Modell in diesem Kapitel. Dann fällt Schritt 2 weg, der Kosinus ist einfach das Skalarprodukt. Für 384 Zahlen sind das 384 Multiplikationen und eine Summe, das schafft jeder Computer millionenfach pro Sekunde.

Mit echten Embeddings liegen die Werte selten nahe −1. Zwei Beispiele:

| Text A | Text B | Kosinus |
| --- | --- | --- |
| Vanille | Schokolade | 0,57 |
| Vanille | Steuererklärung | 0,21 |

Vanille und Schokolade lagen schon auf der Karte in Kapitel 2, Abschnitt 4 nah beieinander. Dort waren die Positionen ausgedacht, hier ist die Ähnlichkeit gemessen.

> **Merke:** Der Kosinus misst, wie gleich zwei Pfeile ausgerichtet sind: 1 heißt gleiche Richtung, 0 heißt nichts gemeinsam.

## 4. Woher die Zahlen kommen: Embedding-Modelle

Die Embeddings aus Kapitel 2 gehören zu einzelnen Tokens. Für die Suche braucht man ein Embedding für einen ganzen Text. Das liefert ein eigenes **Embedding-Modell**.

**Der Aufbau ist bekannt.** Ein Embedding-Modell ist ein Transformer wie in Kapitel 2, Abschnitt 6: Tokenizer, Embeddings, ein Stapel aus Blöcken mit Attention und Neuronen mit Knick. Nur die Ausgabe fehlt, es sagt kein nächstes Token voraus. Stattdessen nimmt man am Ende die Zahlenlisten aller Tokens und bildet den Durchschnitt, Stelle für Stelle. Weil jedes Token durch Attention die anderen gesehen hat, steckt in diesem Durchschnitt der ganze Satz. Anders als im Sprachmodell darf dabei jedes Token auch auf spätere Tokens schauen, denn es muss nichts vorhergesagt werden.

**Das Training ist neu.** Ein Embedding-Modell lernt an Paaren von Texten, die zusammengehören, etwa eine Frage und ihre Antwort oder zwei Sätze mit derselben Bedeutung. Beim Training bekommt es einen Stapel solcher Paare. Für jede Frage soll die passende Antwort den höchsten Kosinus haben, alle anderen Antworten im Stapel einen niedrigeren. So sieht das bei einem fertig trainierten Modell aus, mit vier Fragen und vier Abschnitten aus dem Handbuch:

![Eine Tabelle mit vier Fragen in den Zeilen und vier Abschnitten in den Spalten. Auf der Diagonale, orange umrandet, stehen die passenden Paare mit den höchsten Werten: „Was kostet eine Kugel?“ und Preise 0,85, „Wann habt ihr geöffnet?“ und Öffnungszeiten 0,37, „Darf mein Hund mit?“ und Hunde 0,61, „Ist das Sorbet vegan?“ und Vegane Sorten 0,62. Die übrigen Werte liegen zwischen −0,17 und 0,27.](bilder/kapitel-3/03-training.png)

Die passende Antwort gewinnt in jeder Zeile. Gemessen wird beim Training wieder mit der Kreuzentropie aus Kapitel 2: Die Kosinuswerte einer Zeile werden mit der Softmax-Funktion zu Wahrscheinlichkeiten, und die richtige Antwort soll eine möglichst hohe bekommen. Danach geht es wie immer mit Backpropagation und Gradientenabstieg bergab.

**Das Modell in diesem Kapitel** heißt paraphrase-multilingual-MiniLM-L12-v2 und ist frei verfügbar:

| | |
| --- | --- |
| Parameter | 118 Millionen, davon 96 Millionen in der Tabelle der Token-Embeddings |
| Vokabular | etwa 250.000 Tokens, für über 50 Sprachen |
| Blöcke | 12, mit je 12 Attention-Köpfen |
| Zahlen pro Text | 384 |
| Text pro Durchlauf | höchstens 128 Tokens, der Rest wird abgeschnitten |

Gelernt hat es in zwei Stufen. Zuerst wurde ein englisches Modell an Paaren von Sätzen mit gleicher Bedeutung trainiert. Danach hat das mehrsprachige Modell gelernt, für einen übersetzten Satz denselben Pfeil zu liefern wie das englische Modell für das Original. Deshalb funktioniert die Suche über Sprachgrenzen hinweg:

| Text A | Text B | Kosinus |
| --- | --- | --- |
| Mir tut der Kopf weh. | Ich habe Kopfschmerzen. | 0,92 |
| Zwei Kugeln Schokolade im Becher, bitte. | Two scoops of chocolate in a cup, please. | 0,88 |

Die Bestellung aus Kapitel 2 braucht auf Deutsch und Englisch verschiedene Tokens, ihre Pfeile zeigen trotzdem fast in dieselbe Richtung.

> **Merke:** Ein Embedding-Modell ist ein Transformer, der darauf trainiert wurde, zusammengehörige Texte nah beieinander abzulegen.

## 5. Suchen in einer Textsammlung

Mit Embeddings und Kosinus lässt sich nun eine Suchmaschine bauen. Sie arbeitet in zwei Phasen.

**Vorab, einmal:**

1. **Zerlegen:** Die Textsammlung wird in kurze Stücke geteilt. Das Handbuch besteht schon aus 24 Abschnitten mit je 16 bis 35 Tokens. Jedes Stück bekommt seine Überschrift mit, damit es auch allein verständlich bleibt.
2. **Einbetten:** Das Modell berechnet für jedes Stück ein Embedding.
3. **Speichern:** 24 Listen mit je 384 Zahlen, zusammen gut 9.000 Zahlen.

**Bei jeder Frage:**

1. Das Modell berechnet das Embedding der Frage.
2. Der Kosinus mit jedem gespeicherten Embedding wird berechnet.
3. Die Stücke mit den höchsten Werten sind die Treffer.

Das teure Modell läuft bei einer Frage also nur ein einziges Mal. Alles andere sind Multiplikationen und Summen.

Wie die Stücke zueinander liegen, zeigt eine Karte. Dafür werden die 384 Zahlen auf die zwei Richtungen verkürzt, in denen sich die Stücke am stärksten unterscheiden:

![Eine Karte mit 24 farbigen Punkten, einem pro Abschnitt des Handbuchs. Preise, Bezahlen, Kinder und Stempelkarte bilden unten eine Gruppe. Rechts oben liegen Zucker, Nüsse, Zutaten und Vegane Sorten beieinander. Links oben liegen Hirnfrost, Auftauen, Lagerung, Winterpause und Wetter. Die Frage „Mir tut der Kopf weh, wenn ich Eis esse.“ ist als Stern eingezeichnet und liegt in dieser Gruppe zwischen Wetter und Herstellung.](bilder/kapitel-3/05-karte.png)

Was zusammengehört, liegt meist auch zusammen, besonders deutlich bei den Preisen. Die Karte zeigt aber auch, was beim Verkürzen verloren geht: Auf ihr liegt die Frage näher an „Wetter“ als an „Hirnfrost“. In allen 384 Dimensionen ist „Hirnfrost“ mit 0,81 klar am nächsten, „Wetter“ kommt nur auf 0,19. Für die Suche wird deshalb immer mit allen Zahlen gerechnet. Die Karte dient nur zum Anschauen.

**Und bei Millionen von Texten?** Dann wird es zu langsam, jede Frage mit jedem Stück zu vergleichen. **Vektordatenbanken** ordnen die gespeicherten Pfeile deshalb vorab so, dass sie gute Kandidaten finden, ohne alles durchzugehen. Dafür nehmen sie in Kauf, dass ab und zu ein Treffer fehlt.

> **Merke:** Semantische Suche heißt: alles vorab einbetten, bei jeder Frage nur noch die Frage einbetten und die höchsten Kosinuswerte nehmen.

## 6. RAG: Nachschlagen statt Raten

Ein Sprachmodell kennt das Handbuch der Eisdiele nicht. Fragt man es nach Hirnfrost, antwortet es mit Allgemeinwissen. Fragt man nach dem Preis einer Kinderkugel, kann es nur raten, und das klingt nach Kapitel 2, Abschnitt 10 trotzdem überzeugend.

**Retrieval-Augmented Generation**, kurz **RAG**, heißt ungefähr: durch Nachschlagen ergänztes Erzeugen. Vor der Antwort wird gesucht, und die Treffer kommen mit in den Text, den das Sprachmodell bekommt:

![Ein Ablauf in zwei Zeilen. Oben, vorab und einmal: Das Handbuch mit 24 kurzen Stücken geht durch das Embedding-Modell, gespeichert werden 24 Listen mit je 384 Zahlen. Unten, bei jeder Frage: Die Frage geht durch das Embedding-Modell, dann wird gesucht, also der Kosinus mit allen gespeicherten Listen berechnet und die besten drei genommen. Aus einer Anweisung, den drei Stücken und der Frage entsteht der Prompt. Das Sprachmodell antwortet mit Hilfe der Stücke und nennt die Quelle.](bilder/kapitel-3/06-rag.png)

Der Text, den das Sprachmodell bekommt, heißt **Prompt**. Für die Kopfschmerz-Frage sieht er so aus:

```text
Beantworte die Frage nur mit Hilfe der Auszüge aus dem Handbuch der Eisdiele.
Wenn die Antwort nicht in den Auszügen steht, sag das. Nenne die Nummer des Auszugs, auf den du dich stützt.

[1] Hirnfrost: Wer sehr kaltes Eis zu schnell isst, bekommt manchmal kurz stechende Kopfschmerzen. Langsam essen hilft.
[2] Auftauen: Angetautes Eis wird nicht wieder eingefroren, sondern weggeworfen.
[3] Glutenfrei: Die Waffeln enthalten Weizenmehl. Wer kein Gluten verträgt, nimmt das Eis besser im Becher.

Frage: Mir tut der Kopf weh, wenn ich Eis esse.
```

Eine typische Antwort wäre dann: „Das klingt nach Hirnfrost [1]. Iss das Eis langsamer, dann vergehen die Kopfschmerzen meist schnell.“ Die Auszüge 2 und 3 passen nicht, das Sprachmodell kann sie aber selbst als unpassend erkennen. Die Suche muss also nicht perfekt sein, nur die richtige Antwort unter die ersten Treffer bringen.

Am Sprachmodell selbst ändert RAG nichts. Wie in Kapitel 2, Abschnitt 9 bleiben die Weights fest, das Modell lernt nicht dazu. Es liest nur einen längeren Prompt. Damit greift RAG zwei der Grenzen aus Kapitel 2 an:

| Grenze aus Kapitel 2 | Was RAG ändert |
| --- | --- |
| **Nur, was in den Daten steckt** | Das Wissen kommt aus dem Handbuch. Ändert sich ein Preis, wird ein einziges Stück neu eingebettet, ganz ohne neues Training. |
| **Plausibel statt wahr** | Das Modell stützt sich auf Auszüge und nennt sie. Ob die Antwort stimmt, lässt sich am Auszug nachprüfen. |

**Warum nicht gleich das ganze Handbuch mitschicken?** Bei 24 kurzen Abschnitten ginge das tatsächlich, sie passen locker ins Kontextfenster. Bei tausenden Seiten passen sie nicht mehr hinein, und wo sie passen, kostet jede Frage viele Tokens. RAG schickt nur das Nötige mit.

> **Merke:** RAG sucht passende Texte heraus und gibt sie dem Sprachmodell im Prompt mit. Das Modell schlägt nach, statt zu raten.

## 7. Grenzen ehrlich benennen

Die semantische Suche ist nicht klüger als ihr Training. Mit demselben Handbuch und demselben Modell gehen diese Fragen schief:

![Balkendiagramm für die Frage „Habt ihr etwas ohne Milch?“. Zutaten liegt mit 0,63 vorn, dann Lieferung mit 0,53 und Zucker mit 0,51. Der passende Abschnitt Vegane Sorten folgt erst auf Platz 4 mit 0,45.](bilder/kapitel-3/07-grenze.png)

| Schwäche | Beispiel | Was hilft |
| --- | --- | --- |
| **Verneinungen gehen unter** | Bei „Habt ihr etwas ohne Milch?“ gewinnt der Abschnitt, in dem es um Milch geht. „Eis mit Nüssen“ und „Eis ohne Nüsse“ haben einen Kosinus von 0,90. | Mehrere Treffer an das Sprachmodell geben. Es liest das „ohne“ genau. |
| **Die Reihenfolge zählt wenig** | „Hund beißt Mann.“ und „Mann beißt Hund.“ kommen auf 0,93. | Dasselbe: Das Sprachmodell liest den genauen Wortlaut. |
| **Ähnlich ist nicht gleich passend** | „Liefert ihr auch?“ findet „Lieferung“, dort geht es um Milch für den Stand. Gemeint war „Feiern“: „… liefern wir Eis ab 50 Kugeln“. | Die Stücke so schreiben, dass sie für sich verständlich sind. |
| **Genaue Zahlen und Namen** | Bei „1,20 Euro“ gewinnt „Preise“. Die Stichwortsuche findet „Kinder“, wo genau dieser Betrag steht. | **Hybride Suche:** Stichwortsuche und semantische Suche kombinieren. |
| **Es gibt immer einen Treffer** | Auf „Wie wird das Wetter morgen?“ liefert die Suche den Abschnitt „Wetter“, obwohl keine Vorhersage im Handbuch steht. | Im Prompt erlauben, dass die Antwort fehlt („Wenn die Antwort nicht in den Auszügen steht, sag das.“) |
| **Zu lange Stücke** | Das Modell liest höchstens 128 Tokens, der Rest wird abgeschnitten. Auch innerhalb dieser Grenze verwässert ein Durchschnitt über viele Themen. | Kurze Stücke mit einem Thema, jeweils mit Überschrift |

Auch mit RAG bleibt ein Sprachmodell ein Sprachmodell. Findet die Suche den falschen Abschnitt, antwortet es überzeugend aus dem falschen Abschnitt. Bei wichtigen Antworten lohnt deshalb der Blick in die genannte Quelle.

> **Merke:** Die semantische Suche findet Ähnliches, nicht zwingend Richtiges. Verneinungen, Zahlen und Namen sind ihre Schwachstellen.

## Das Wichtigste in Kürze

- Die **Stichwortsuche** vergleicht Wörter und scheitert an Umschreibungen, Synonymen und zusammengesetzten Wörtern.
- Ein **Embedding** macht aus einem ganzen Text eine Zahlenliste, einen Pfeil. Ähnliche Bedeutung heißt ähnliche Richtung.
- Der **Kosinus** misst, wie gleich zwei Pfeile ausgerichtet sind: Skalarprodukt geteilt durch das Produkt der Längen. Bei Pfeilen der Länge 1 ist er einfach das Skalarprodukt.
- Ein **Embedding-Modell** ist ein Transformer ohne Vorhersage des nächsten Tokens. Den Durchschnitt über alle Tokens nimmt man als Embedding des Textes.
- Trainiert wird es an **Paaren**, die zusammengehören: Die passende Antwort soll unter allen im Stapel den höchsten Kosinus bekommen. Gemessen wird wieder mit der Kreuzentropie.
- **Semantische Suche:** Alle Stücke einmal vorab einbetten, bei jeder Frage nur die Frage einbetten und die höchsten Kosinuswerte nehmen.
- **RAG** gibt die Treffer dem Sprachmodell im Prompt mit. Es antwortet mit Wissen, das nicht in seinen Weights steckt, und kann die Quelle nennen.
- Verneinungen, genaue Zahlen und Namen sind Schwachstellen. Dagegen helfen mehrere Treffer, eine **hybride Suche** und eine Anweisung, fehlende Antworten zuzugeben.

## Selbst ausprobieren

In [Labor 3](app/labor-3.html) lässt sich der Kosinus mit zwei Pfeilen von Hand nachrechnen. Außerdem kann man eigene Sätze vergleichen, im Handbuch der Eisdiele suchen und den Prompt für RAG zusammenstellen. Die Beispiele aus diesem Kapitel funktionieren sofort. Für eigene Eingaben lädt das Labor das Embedding-Modell einmalig in den Browser, etwa 118 MB. Danach rechnet es ohne Server direkt auf dem eigenen Rechner. Diese verkleinerte Fassung des Modells liefert Werte, die um ein bis zwei Hundertstel von denen im Text abweichen können.

Das Skript rechnet alle Zahlen dieses Kapitels nach und erzeugt die Grafiken und die Daten für das Labor neu. Beim ersten Start lädt es das Modell in voller Genauigkeit herunter, etwa 470 MB. Mit einer eigenen Frage durchsucht es das Handbuch:

```bash
uv run docs/beispiele/03_semantische_suche.py "Habt ihr Eis für Hunde?"
```

Das Handbuch steht am Anfang von [`beispiele/03_semantische_suche.py`](beispiele/03_semantische_suche.py) und kann dort erweitert werden.

## Quellen

- Reimers und Gurevych (2019): [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- Reimers und Gurevych (2020): [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813)
- Karpukhin u. a. (2020): [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- Lewis u. a. (2020): [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Hugging Face: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Hugging Face: [Transformers.js](https://huggingface.co/docs/transformers.js)

# Glossar: Fachbegriffe in Alltagssprache

Alle Fachbegriffe aus dem Workshop, kurz erklärt und alphabetisch sortiert. Viele stammen aus dem Englischen, weil die Forschung zu KI überwiegend auf Englisch stattfindet. Der Verweis führt jeweils zu dem Abschnitt, in dem der Begriff ausführlich erklärt wird.

## A

**Aktivierungsfunktion** (auch „der Knick“, häufigste Form: ReLU): Eine kleine Regel, die ein Neuron am Ende auf sein Ergebnis anwendet, etwa „Negative Werte werden zu 0“. Ohne sie ergeben auch viele Neuronen zusammen nur eine Gerade. → [Kapitel 1, Abschnitt 9](01-Neuronale-Netze-Grundlagen.md#9-der-knick-die-aktivierungsfunktion)

**Anwendung** (auch Inferenz): Die Phase nach dem Training, in der ein Modell nur noch Vorhersagen ausrechnet. Seine Parameter bleiben dabei fest. Wenn du mit einem Chat-Assistenten schreibst, ist das Anwendung. → [Kapitel 1, Abschnitt 6](01-Neuronale-Netze-Grundlagen.md#6-eine-zweite-beobachtung-das-wochenende)

**Attention** (Aufmerksamkeit): Der Baustein im Sprachmodell, mit dem jedes Wort auf die Wörter davor schaut und entscheidet, welche gerade wichtig sind. So erfasst das Modell Zusammenhänge über viele Wörter hinweg. → [Kapitel 2, Abschnitt 5](02-Tokenizer-und-Sprachmodelle.md#5-ein-wort-reicht-nicht-attention)

## B

**Backpropagation** (Rückwärtsverteilung des Fehlers): Das Verfahren, mit dem ein neuronales Netz herausfindet, welches Neuron wie viel zu einem Fehler beigetragen hat. Der Fehler wird vom Ende des Netzes rückwärts durchgereicht. → [Kapitel 1, Abschnitt 10](01-Neuronale-Netze-Grundlagen.md#10-vom-neuron-zum-netz)

**Bias**: Der Grundwert eines Neurons, zu dem alles andere dazukommt. Bei der Geraden ist es der Schnittpunkt mit der y-Achse. Nicht zu verwechseln mit „Bias“ im Sinne von Verzerrung oder Vorurteil, etwa wenn ein Modell Einseitigkeiten aus seinen Trainingsdaten übernimmt. → [Kapitel 1, Abschnitt 3](01-Neuronale-Netze-Grundlagen.md#3-zwei-stellschrauben-weight-und-bias)

**Byte-Pair-Encoding**: Das verbreitete Verfahren, mit dem ein Tokenizer sein Vokabular lernt: Häufig nebeneinander stehende Zeichen werden Schritt für Schritt zu größeren Stücken zusammengefasst. → [Kapitel 2, Abschnitt 2](02-Tokenizer-und-Sprachmodelle.md#2-text-in-zahlen-der-tokenizer)

## D

**Deep Learning**: Lernen mit neuronalen Netzen, die viele Schichten hintereinander haben. Die meisten heutigen KI-Anwendungen beruhen darauf. → [Kapitel 1, Abschnitt 10](01-Neuronale-Netze-Grundlagen.md#10-vom-neuron-zum-netz)

## E

**Embedding**: Eine Liste von Zahlen, die die Bedeutung eines Wortes oder Textes darstellt. Ähnliche Bedeutungen bekommen ähnliche Zahlenlisten. Man kann sich ein Embedding als Pfeil vorstellen. → [Kapitel 2, Abschnitt 4](02-Tokenizer-und-Sprachmodelle.md#4-bedeutung-als-zahlenliste-embeddings) und [Kapitel 3, Abschnitt 2](03-Semantische-Suche-und-RAG.md#2-ein-text-ein-pfeil)

**Embedding-Modell**: Ein Modell, das aus einem ganzen Text ein Embedding macht. Es ist ähnlich gebaut wie ein Sprachmodell, erzeugt aber keinen Text. Grundlage der semantischen Suche. → [Kapitel 3, Abschnitt 4](03-Semantische-Suche-und-RAG.md#4-woher-die-zahlen-kommen-embedding-modelle)

## F

**Feinabstimmung** (Fine-Tuning): Ein zusätzliches Training nach dem Vortraining, mit ausgewählten Beispielen. Bei Chat-Assistenten sind das Beispielgespräche mit guten Antworten. → [Kapitel 2, Abschnitt 9](02-Tokenizer-und-Sprachmodelle.md#9-vom-textvervollständiger-zum-assistenten)

## G

**Gradient**: Die Richtung, in der der Fehler am stärksten ansteigt, für alle Parameter zugleich. Der Gradientenabstieg geht in die Gegenrichtung. → [Kapitel 1, Abschnitt 5](01-Neuronale-Netze-Grundlagen.md#5-lernen-heißt-schritt-für-schritt-bergab)

**Gradientenabstieg** (Gradient Descent): Das Lernverfahren fast aller heutigen KI: Steigung des Fehlers messen, ein kleines Stück bergab gehen, wiederholen, ähnlich wie bei einem Abstieg im Nebel. → [Kapitel 1, Abschnitt 5](01-Neuronale-Netze-Grundlagen.md#5-lernen-heißt-schritt-für-schritt-bergab)

## H

**Halluzination**: Eine Antwort eines Sprachmodells, die plausibel klingt, aber nicht stimmt, etwa eine erfundene Quelle. Sie entsteht, weil das Modell wahrscheinlichen Text erzeugt, ohne ihn zu prüfen. → [Kapitel 2, Abschnitt 10](02-Tokenizer-und-Sprachmodelle.md#10-typische-schwächen)

**Hybride Suche**: Eine Kombination aus Stichwortsuche und semantischer Suche. Sie findet sowohl Umschreibungen als auch genaue Zahlen und Namen. → [Kapitel 3, Abschnitt 7](03-Semantische-Suche-und-RAG.md#7-typische-schwächen)

## I

**Inferenz**: siehe Anwendung.

## K

**Künstliche Intelligenz** (KI): Sammelbegriff für Computerprogramme, die Aufgaben lösen, für die man bisher menschliche Intelligenz brauchte, etwa Sprache verstehen oder Bilder erkennen. Fast alle heutigen KI-Anwendungen beruhen auf neuronalen Netzen, die aus Beispielen lernen. → [Kapitel 1](01-Neuronale-Netze-Grundlagen.md)

**Kontextfenster**: Die Obergrenze dafür, wie viel Text ein Sprachmodell auf einmal berücksichtigen kann, gemessen in Tokens. Dazu gehören das bisherige Gespräch und alle mitgeschickten Dokumente. → [Kapitel 2, Abschnitt 9](02-Tokenizer-und-Sprachmodelle.md#9-vom-textvervollständiger-zum-assistenten)

**Kosinus** (Kosinus-Ähnlichkeit): Ein Maß dafür, wie gleich zwei Pfeile ausgerichtet sind. 1 heißt gleiche Richtung, 0 heißt nichts gemeinsam. Damit misst die semantische Suche, wie ähnlich zwei Texte sind. → [Kapitel 3, Abschnitt 3](03-Semantische-Suche-und-RAG.md#3-ähnlichkeit-messen-der-kosinus)

**Kreuzentropie**: Das Fehlermaß beim Training von Sprachmodellen. Es misst, wie wenig Wahrscheinlichkeit das Modell dem Wort gegeben hat, das tatsächlich folgt. → [Kapitel 2, Abschnitt 1](02-Tokenizer-und-Sprachmodelle.md#1-was-kommt-als-nächstes)

## L

**Lernen aus menschlichem Feedback** (englisch Reinforcement Learning from Human Feedback, RLHF): Ein Trainingsschritt, bei dem Menschen Antworten des Modells vergleichen und das Modell so weitertrainiert wird, dass bevorzugte Antworten wahrscheinlicher werden. → [Kapitel 2, Abschnitt 9](02-Tokenizer-und-Sprachmodelle.md#9-vom-textvervollständiger-zum-assistenten)

**Lernrate**: Die Schrittlänge beim Gradientenabstieg. Ist sie zu klein, dauert das Lernen sehr lange. Ist sie zu groß, springt das Modell über das Ziel hinaus und lernt gar nicht. → [Kapitel 1, Abschnitt 5](01-Neuronale-Netze-Grundlagen.md#5-lernen-heißt-schritt-für-schritt-bergab)

**Lineare Regression**: Ein Verfahren, das eine Gerade so durch Messpunkte legt, dass sie insgesamt möglichst wenig danebenliegt. Der einfachste Fall eines lernenden Modells. → [Kapitel 1, Abschnitt 2](01-Neuronale-Netze-Grundlagen.md#2-eine-gerade-durch-die-punkte)

**LLM** (Large Language Model): siehe Sprachmodell.

**Loss** (Verlustfunktion): Allgemeiner Name für die Zahl, die misst, wie falsch ein Modell liegt. Lernen heißt, diese Zahl möglichst klein zu machen. → [Kapitel 1, Abschnitt 4](01-Neuronale-Netze-Grundlagen.md#4-wie-gut-passt-die-gerade-fehler-messen)

## M

**Maschinelles Lernen**: Aus Beispielen mit bekannter Antwort eine Regel gewinnen, die auch für neue Fälle eine Antwort liefert, statt die Regel von Hand zu programmieren. → [Kapitel 1, Abschnitt 1](01-Neuronale-Netze-Grundlagen.md#1-der-eisverkäufer)

**Mittlerer quadratischer Fehler** (Mean Squared Error, MSE): Ein Fehlermaß: alle Fehler quadrieren und den Durchschnitt bilden. So heben sich Fehler nach oben und unten nicht auf, und große Fehler zählen besonders stark. → [Kapitel 1, Abschnitt 4](01-Neuronale-Netze-Grundlagen.md#4-wie-gut-passt-die-gerade-fehler-messen)

**Modell**: Eine Rechenregel, die aus Eingaben eine Vorhersage macht und deren Zahlen (Parameter) aus Daten gelernt wurden. → [Kapitel 1, Abschnitt 2](01-Neuronale-Netze-Grundlagen.md#2-eine-gerade-durch-die-punkte)

## N

**Neuron** (künstliches Neuron): Der Grundbaustein neuronaler Netze. Es nimmt Eingaben, multipliziert jede mit ihrem Weight, addiert alles, zählt den Bias dazu und gibt das Ergebnis durch die Aktivierungsfunktion. Das Gehirn ist nur ein grobes Vorbild. → [Kapitel 1, Abschnitt 8](01-Neuronale-Netze-Grundlagen.md#8-vom-modell-zum-neuron)

**Neuronales Netz**: Viele Neuronen in Schichten, bei denen die Ergebnisse der einen Schicht die Eingaben der nächsten sind. → [Kapitel 1, Abschnitt 10](01-Neuronale-Netze-Grundlagen.md#10-vom-neuron-zum-netz)

## P

**Parameter**: Die Zahlen, die ein Modell beim Training einstellt, also alle Weights und Biases. Die Gerade des Eisverkäufers hat 2, sein kleines Netz 9, große Sprachmodelle viele Milliarden. → [Kapitel 1, Abschnitt 3](01-Neuronale-Netze-Grundlagen.md#3-zwei-stellschrauben-weight-und-bias)

**Prompt**: Der Text, den ein Sprachmodell als Eingabe bekommt: deine Frage, das bisherige Gespräch und gegebenenfalls Anweisungen und mitgeschickte Textauszüge. → [Kapitel 3, Abschnitt 6](03-Semantische-Suche-und-RAG.md#6-rag-erst-nachschlagen-dann-antworten)

## R

**RAG** (Retrieval-Augmented Generation, durch Nachschlagen ergänztes Erzeugen): Vor der Antwort werden passende Textstellen gesucht und dem Sprachmodell im Prompt mitgegeben. Das Modell stützt seine Antwort auf diese Textstellen und kann sie als Quelle nennen. → [Kapitel 3, Abschnitt 6](03-Semantische-Suche-und-RAG.md#6-rag-erst-nachschlagen-dann-antworten)

**ReLU**: siehe Aktivierungsfunktion.

## S

**Schicht**: Eine Gruppe von Neuronen, die gleichzeitig rechnen. Die Schichten zwischen Eingabe und Ausgabe heißen verborgene Schichten (Hidden Layers). → [Kapitel 1, Abschnitt 10](01-Neuronale-Netze-Grundlagen.md#10-vom-neuron-zum-netz)

**Selbstüberwachtes Lernen**: Lernen aus Daten, die ihre richtigen Antworten selbst mitbringen. Bei Text ist an jeder Stelle bekannt, wie es weitergeht, niemand muss etwas beschriften. → [Kapitel 2, Abschnitt 1](02-Tokenizer-und-Sprachmodelle.md#1-was-kommt-als-nächstes)

**Semantische Suche**: Eine Suche, die Texte nach ihrer Bedeutung vergleicht. Sie findet auch Texte, die die Frage mit ganz anderen Worten beantworten. → [Kapitel 3, Abschnitt 1](03-Semantische-Suche-und-RAG.md#1-suchen-nach-bedeutung)

**Skalierung**: Das Vergrößern eines Modells: mehr Parameter, mehr Daten, mehr Rechenaufwand. Der Fehler sinkt dabei so regelmäßig, dass man ihn vorab abschätzen kann. → [Kapitel 2, Abschnitt 7](02-Tokenizer-und-Sprachmodelle.md#7-skalierung-warum-größer-besser-wird)

**Softmax**: Eine Rechenvorschrift, die aus beliebigen Punktzahlen Wahrscheinlichkeiten macht, die zusammen 100 % ergeben. → [Kapitel 2, Abschnitt 1](02-Tokenizer-und-Sprachmodelle.md#1-was-kommt-als-nächstes)

**Sprachmodell** (großes Sprachmodell, englisch Large Language Model, LLM): Ein neuronales Netz, das darauf trainiert ist, das nächste Stück Text vorherzusagen. Chat-Assistenten wie ChatGPT, Claude oder Gemini beruhen darauf. → [Kapitel 2](02-Tokenizer-und-Sprachmodelle.md)

**Stichwortsuche**: Eine Suche, die zählt, wie viele Wörter der Frage in einem Text vorkommen. Sie scheitert an Umschreibungen und anderen Wortformen, findet aber genaue Zahlen und Namen zuverlässig. → [Kapitel 3, Abschnitt 1](03-Semantische-Suche-und-RAG.md#1-suchen-nach-bedeutung)

## T

**Temperatur**: Eine Stellschraube beim Erzeugen von Text. Bei niedriger Temperatur wählt das Modell fast immer das wahrscheinlichste Wort, bei hoher Temperatur wird es abwechslungsreicher, aber auch fehleranfälliger. → [Kapitel 2, Abschnitt 8](02-Tokenizer-und-Sprachmodelle.md#8-text-erzeugen)

**Testdaten**: Daten, die vor dem Training beiseitegelegt werden, um das fertige Modell zu prüfen. Nur an ihnen sieht man, ob es auch mit Neuem zurechtkommt. → [Kapitel 1, Abschnitt 7](01-Neuronale-Netze-Grundlagen.md#7-die-prüfung-funktioniert-es-auch-an-neuen-tagen)

**Token**: Ein Textstück, mit dem ein Sprachmodell arbeitet, meist ein Wort oder ein Wortteil. Preis, Geschwindigkeit und Textlänge werden bei Sprachmodellen in Tokens gemessen. → [Kapitel 2, Abschnitt 2](02-Tokenizer-und-Sprachmodelle.md#2-text-in-zahlen-der-tokenizer)

**Tokenizer**: Das Programm, das Text in Tokens zerlegt und jedem eine Nummer gibt. Es wird vorab gelernt und bleibt danach fest. → [Kapitel 2, Abschnitt 2](02-Tokenizer-und-Sprachmodelle.md#2-text-in-zahlen-der-tokenizer)

**Training**: Die Phase, in der ein Modell aus Daten lernt, also seine Parameter einstellt. → [Kapitel 1, Abschnitt 6](01-Neuronale-Netze-Grundlagen.md#6-eine-zweite-beobachtung-das-wochenende)

**Transformer**: Die Bauweise heutiger Sprachmodelle: ein Stapel aus Blöcken mit Attention und Neuronen mit Knick. Das „T“ in GPT steht dafür. → [Kapitel 2, Abschnitt 6](02-Tokenizer-und-Sprachmodelle.md#6-der-transformer)

## Ü

**Überanpassung** (Overfitting): Ein Modell lernt die Trainingsdaten auswendig, statt den Zusammenhang zu erkennen. Im Training sieht es gut aus, bei neuen Daten liegt es weit daneben. → [Kapitel 1, Abschnitt 7](01-Neuronale-Netze-Grundlagen.md#7-die-prüfung-funktioniert-es-auch-an-neuen-tagen)

## V

**Vektor**: Ein Pfeil, beschrieben durch eine Liste von Zahlen. Embeddings sind Vektoren. → [Kapitel 3, Abschnitt 2](03-Semantische-Suche-und-RAG.md#2-ein-text-ein-pfeil)

**Vektordatenbank**: Ein Speicher für sehr viele Embeddings, der ähnliche Einträge schnell findet, ohne jeden einzeln zu vergleichen. → [Kapitel 3, Abschnitt 5](03-Semantische-Suche-und-RAG.md#5-suchen-in-einer-textsammlung)

**Vortraining** (Pretraining): Die erste und aufwendigste Trainingsphase eines Sprachmodells, mit riesigen Textmengen. Danach kann es Text fortsetzen, ist aber noch kein Assistent. → [Kapitel 2, Abschnitt 9](02-Tokenizer-und-Sprachmodelle.md#9-vom-textvervollständiger-zum-assistenten)

## W

**Weight** (Gewicht): Die Zahl, mit der ein Neuron eine Eingabe malnimmt. Sie sagt, wie stark diese Eingabe die Vorhersage beeinflusst. Bei der Geraden ist es die Steigung. → [Kapitel 1, Abschnitt 3](01-Neuronale-Netze-Grundlagen.md#3-zwei-stellschrauben-weight-und-bias)

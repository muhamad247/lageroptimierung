# Lagerbestand optimieren

Wann nachbestellen und wie viel? Das Projekt rechnet Bestellmenge und Meldebestand mit den Lehrbuchformeln aus und prüft dann per Simulation, ob der angestrebte Servicegrad wirklich erreicht wird. Dazu kommen eine Kostenoptimierung und eine ABC/XYZ-Analyse für ein Sortiment.

## Das Ergebnis vorweg: Die Lehrbuchformel verfehlt das Ziel

Ziel: 95 % der Lieferzyklen ohne Fehlmenge (α-Servicegrad), Bestellmenge nach EOQ, 30 Jahre tagesgenau simuliert.

| Artikel | Lehrbuch | mit Undershoot-Korrektur | per Simulation kalibriert |
|---|---|---|---|
| Schnelldreher (20 Stück/Tag, 5 Tage Lieferzeit) | s = 117 → **75 %** | s = 128 → 95,2 % | s = 128 → 95,2 % |
| Lieferzeit schwankt (± 3 Tage) | s = 168 → 92 % | s = 179 → 96,1 % | s = 175 → 95,1 % |
| Ersatzteil, sporadische Nachfrage | s = 20 → 87 % | s = 24 → 92 % | s = 28 → 95,2 % |

Zwei Gründe:

1. **Undershoot.** Die Formel s = μ·L + z·σ_L geht davon aus, dass der Bestand laufend überwacht wird und genau bei s bestellt wird. Wird nur einmal am Tag geprüft, liegt die Bestandsposition beim Bestellen im Schnitt schon E[D²]/(2·E[D]) unter s, beim Schnelldreher gut 10 Stück. Genau das fehlt dann als Sicherheitsbestand.
2. **Schiefe Nachfrage.** Bei sporadischer Nachfrage (meist 0, ab und zu ein großer Auftrag) ist die Normalverteilung eine schlechte Näherung. Sie unterschätzt die großen Ausschläge, auch nach der Undershoot-Korrektur.

Die Kalibrierung sucht per Simulation den kleinsten Meldebestand, der das Ziel erreicht. Sie braucht keine Verteilungsannahme. Damit sie nicht nur auf ihr eigenes Zufallsszenario passt, wird sie zusätzlich auf frisch gezogenen Daten geprüft (Spalte `neue_daten` in `python -m lager`).

Ein zweiter Befund: Die **Fill Rate** (Anteil der Nachfrage, der sofort geliefert wird) liegt beim Schnelldreher selbst mit dem Lehrbuchwert bei 99,5 %, obwohl jeder vierte Lieferzyklus eine Fehlmenge hat. Ob man α oder β steuert, macht also einen großen Unterschied, und das ist eine Geschäftsentscheidung, keine statistische.

## Inhalt

- **Formeln** (`lager/model.py`): EOQ, Sicherheitsbestand mit schwankender Nachfrage und Lieferzeit, Undershoot-Korrektur, empirisches Quantil der Lieferzeitnachfrage, Fill-Rate-Näherung über die Verlustfunktion
- **Simulation** (`lager/simulate.py`): tagesgenau, mit Rückständen, für (s, Q) und (R, S). Alle Politiken laufen auf denselben Zufallszahlen (Common Random Numbers), damit Unterschiede von der Politik kommen und nicht vom Zufall
- **Kostenoptimierung**: Raster über s und Q, Jahreskosten aus Lager-, Bestell- und Fehlmengenkosten
- **ABC/XYZ** (`lager/portfolio.py`): Klassen nach Verbrauchswert und Schwankung, Ziel-Servicegrad je Feld

## Starten

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m lager                 # Methodenvergleich, Kostenoptimum, ABC/XYZ in der Konsole
streamlit run app.py            # eigene Artikel durchspielen
pytest
```

## Tests

Neben Formeln und Reproduzierbarkeit prüfen die Tests das Verhalten des Modells:
- Bestandserhaltung: Endbestand = Anfang + Wareneingänge − Nachfrage, exakt
- höherer Meldebestand → höherer Servicegrad, höherer Bestellbestand → höhere Fill Rate
- die Lehrbuchformel verfehlt das Ziel, die korrigierte und die kalibrierte erreichen es
- die Kostenkurve hat ihr Minimum im Inneren des Rasters

## Annahmen und Grenzen

- Nicht gelieferte Mengen werden nachgeliefert (Rückstand), nicht storniert.
- Fehlmengenkosten pro Stück sind in der Praxis schwer zu beziffern. Oft gibt man deshalb lieber einen Servicegrad vor, als Fehlmengen in Euro zu bewerten.
- Die Nachfrage ist stationär, ohne Saison und Trend. Für echte Daten käme eine Prognose davor, deren Fehler dann die σ-Werte bestimmt.

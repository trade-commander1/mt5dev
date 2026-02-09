# Spezifikation: MQL5 Experten-Dialog (GUI)

## 1. Obere Sektion: Trading-Interaktionen
Drei horizontal angeordnete Haupt-Buttons zur Steuerung der Positionen:

* **[Open Long]**: 
    * Blinkt **grün**, wenn ein Long-Signal vorliegt (CRV >= Schwellenwert).
    * Manuelle Eröffnung jederzeit möglich, sofern CRV-Bedingung erfüllt ist.
* **[Open Short]**: 
    * Blinkt **rot**, wenn ein Short-Signal vorliegt (CRV >= Schwellenwert).
    * Manuelle Eröffnung jederzeit möglich, sofern CRV-Bedingung erfüllt ist.
* **[Close Position]**: 
    * Schließt die aktuelle Position des Symbols sofort.

**Status-Reset:** Wenn keine Position offen ist und kein Signal anliegt, werden die Buttons auf "No Signal" zurückgesetzt (Standardfarben, kein Blinken).

---

## 2. Mittlere Sektion: Prognose & Risiko-Parameter
Anzeige der vom Neuronalen Netz (NN) gelieferten Echtzeit-Daten und Nutzereingaben:

* **Eingabefeld [Volumen]**: Festlegung der Lot-Größe für den nächsten Trade.
* **Prognose-Anzeige**:
    * Aktuelle Richtung (Trend-Voraussage).
    * **Max. Run-up (MFE)**: Erwarteter Gewinn in Kontowährung (kalkuliert aus Lot-Größe & NN-Prognose).
    * **Max. Drawdown (MAE)**: Erwartetes Risiko in Kontowährung (kalkuliert aus Lot-Größe & NN-Prognose).

---

## 3. Untere Sektion: Steuerung & Automatisierung
Zustandsabhängige Buttons zur Filterung und Ablaufsteuerung:

* **Richtungsfilter (Tri-State Button)**:
    * Zustände: `Long` | `Short` | `All`. 
    * Filtert, welche Signale vom System beachtet oder automatisch ausgeführt werden.
* **Execution Mode (Multi-State Button)**:
    * Zustände: `Manual` -> `Auto 1` -> `Auto 2` -> `Auto 3` -> zurück zu `Manual`.
    * `Auto N`: Führt die nächsten N Signale automatisch aus, ohne dass [Execute] gedrückt werden muss.
    * Zähler dekrementiert nach jeder automatischen Eröffnung (z. B. von Auto 2 auf Auto 1).

---

## 4. Footer: Account-Monitor & Deposit Load
Permanente Überwachung der Performance und des Risikos:

* **Profit-Anzeige**:
    * **Symbol Profit**: Aktueller Profit des geladenen Symbols.
    * **Gesamt Profit**: Gesamter Realized Profit des Kontos.
    * **Session Profit**: Profit seit 00:00 Uhr lokaler Zeit.
* **Deposit Load Balken**:
    * Grafische Anzeige der Kontobelastung (0% - 100%).
    * **Alarm-Funktion**: Der Balken blinkt, sobald der Wert den Experten-Parameter (Default 10%) übersteigt.

---

## 5. Hintergrund-Logik & Experten-Parameter (Non-GUI)
Diese Werte steuern das Verhalten der GUI-Elemente im Hintergrund:

* **CRV-Schwellenwert**: Verhältnis von MFE zu MAE (Default 2.0). Bestimmt, ab wann Signale als valide gelten und die Buttons blinken.
* **Stop-Loss Modus (Radio Buttons)**:
    * `Real`: Stop-Loss wird an den Broker gesendet.
    * `Virtual`: Stop-Loss wird nur lokal vom EA verwaltet (unsichtbar für Broker).
    * `Mixed`: Automatisches Umschalten zwischen Real und Virtuell.
* **Mixed-Mode Parameter**:
    * Zufällige Umschaltdauer zwischen 10 und 60 Sekunden.
    * Plausibilitätsprüfung: Untere Schwelle > 1s und < obere Schwelle.
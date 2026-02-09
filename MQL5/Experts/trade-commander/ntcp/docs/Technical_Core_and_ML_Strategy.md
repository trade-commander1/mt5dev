# Technical_Core_and_ML_Strategy.md

## 1. Projektziel & Hardware-Spezifikation
Entwicklung einer kommerziellen MLOps-Plattform zur Erstellung prädiktiver Multi-Target-Handelsmodelle für den MetaTrader 5, optimiert für lokale High-End-Hardware.

* **Hardware:** AMD Ryzen 9 7950X3D (Preprocessing), 64 GB RAM, NVIDIA RTX 5080 (GPU-beschleunigtes CUDA-Training).
* **Software-Stack:** Python 3.10+, PyTorch (Deep Learning), SQLite (Experiment-Tracking), PyQt6 (Interface).

## 2. Strategie-Logik: Multi-Target-Regression
Übergang von starren Regeln zu einer KI-basierten Wahrscheinlichkeits-Prognose für Risiko (MAE) und Ertrag (MFE/Momentum).

### A. Datengrundlage (Dataset)
* **Kontinuierliches Lernen:** Training auf allen verfügbaren Kerzen (ca. 100.000+ Bars), um Marktstrukturen jenseits isolierter Signale zu erfassen.
* **Multi-Timeframe (MTF):** Synchronisation von Basis-Timeframe (M5) und Anker-Timeframe (H1). Die H1-Daten werden um ein Intervall geshiftet (T-1), um Look-ahead Bias zu verhindern.

### B. Eingangsdaten (Feature-Vektor)
* **Sequenzlänge (N):** 20 bis 30 Zeitschritte (Lookback).
* **Fixiertes MA-Spektrum (11 Stufen):** `MA_SPECTRUM = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]`
* **Signal-Boundary Index:** `SIGNAL_BOUNDARY_INDEX = 3` (Bestimmt die MAs 5, 8, 13 und 21 als trigger-relevant).
* **Features pro Zeitschritt:**
    * **11x MAs / 11x StdDevs:** Vollständiges Spektrum als Input für das NN-Modell.
    * **Trend-Indikatoren:** RSI, ATR, ADX (Standard-Perioden).
    * **Marktstruktur:** Abstand zum Tages-Hoch/Tief, Squeeze-Bandbreite (Max_MA - Min_MA).
    * **Kerzen-Geometrie:** Körper, Docht, Lunte (relativ zur High-Low-Range).
    * **Energie:** Relatives Volumen (Ticks / SMA_20_Ticks).
    * **Anker-Daten (H1):** MA, Close und Vola des Ankers (relativ zum Basis-Close).
    * **Zeit-Features:** Stunde und Wochentag als zyklische Sinus/Cosinus-Werte.

## 3. Normalisierungs-Protokoll (Stationarität)
Alle Eingangs- und Ausgangswerte werden transformiert, um Preisunabhängigkeit zu garantieren:

### A. Feature-Normalisierung (Inputs)
| Feature-Gruppe | Formel / Methode |
| :--- | :--- |
| **Gleitende Durchschnitte (MA)** | `(MA_n / Close) - 1.0` |
| **Standardabweichungen (SD)** | `SD_n / Close` |
| **Oszillatoren (RSI, ADX)** | `(Wert / 100.0)` |
| **Volatilität (ATR)** | `ATR_14 / Close` |
| **Kerzen-Körper** | `(Close - Open) / (High - Low)` |
| **Docht / Lunte** | `Docht_Lunte / (High - Low)` |
| **Volumen (Ticks)** | `Ticks / SMA(Ticks, 20)` |
| **Zeit (Stunde/Tag)** | `sin/cos(2*π*t/T)` |

### B. Label-Normalisierung (Targets)
Prognose der relativen Preisänderung bezogen auf $Close_{entry}$ für die Horizonte **3, 5, 10 und 20 Bars**:
* **Momentum:** `(Price_T+n / Close_entry) - 1.0`
* **MFE (Max Run-up):** `(Highest_High_n / Close_entry) - 1.0`
* **MAE (Max Run-down):** `(Lowest_Low_n / Close_entry) - 1.0`
* **Clipping:** Ausreißer werden beim 1. und 99. Perzentil begrenzt (News-Spike Schutz).

## 4. Handels-Ausführung (MQL5 EA)
Strikte Trennung zwischen strukturellem Trigger und statistischem KI-Kontext.

* **Signal-Trigger (Blinken):** Nur MAs bis `SIGNAL_BOUNDARY_INDEX`.
    * *Bedingung:* Kurs-Lage + Steigung (Slope >= 0 für Long / <= 0 für Short) + CRV-Check des NN.
* **Einstiegs-Trigger (Statisch):** Kurs-Lage relativ zu den Trigger-MAs + CRV-Check.
    * *Option:* `Allow_Entry_Without_Slope` (Deaktivierbar via Parameter).
* **NN-Kontext:** Das NN nutzt das gesamte Spektrum (bis MA 500). Lange MAs fungieren als Filter (CRV-Unterdrückung bei Gegenwind).
* **Exit-Management:** Initial-Stop via MAE-Prognose; danach Trailing-Stop über den MA-Fächer ("Trend-Segeln").
* **Security:** Mixed-Mode Stop-Loss (Real/Virtuell-Switching, 10-60s Zufallsintervall) zur Broker-Anonymisierung.

## 5. Software-Architektur
* **DataManager:** Skalierbare Pipeline für CSV-Parsing, MTF-Merge und Normalisierung.
* **Trainer-Modul:** Multi-Target-Netzwerk (GRU/LSTM); Unterstützung für Full Retraining, Fine-Tuning und Sliding-Window.
* **DatabaseManager:** SQLite-Logging von Experimenten, Hyperparametern und Backtest-Telemetrie.
* **MQL5-Schnittstelle:** Export von Gewichten und Skalierungs-Faktoren (Single Source of Truth) via JSON/Header-Files.
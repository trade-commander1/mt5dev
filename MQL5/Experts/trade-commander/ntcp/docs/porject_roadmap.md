# 🚀 Projekt-Roadmap & Business-Logik: NTCP Commercial Edition

## 1. Projekt-Kontext & Vision
Die **Neural Trading Consultant Plattform (NTCP)** wird als kommerzielles Software-Produkt entwickelt. Ziel ist der Vertrieb über **Digistore24** als High-End-Lösung für algorithmischen Handel im MetaTrader 5.

* **Zielgruppe:** Ambitionierte Retail-Trader und institutionelle Kleininvestoren.
* **Alleinstellungsmerkmal (USP):** Kopplung von klassischer MA-Strukturlogik mit modernster Multi-Target-KI (MFE/MAE Prognose) auf lokaler Hardware-Power.

## 2. Entwicklungs-Infrastruktur & KI-Strategie
Die Entwicklung erfolgt hochgradig automatisiert unter Ausnutzung einer High-End-Workstation:
* **Hardware-Basis:** AMD Ryzen 9 7950X3D & NVIDIA RTX 5080 (CUDA).
* **Entwicklungswerkzeuge:** - **Cursor & Claude Code:** Primäre IDEs für die Code-Generierung und Architektur-Refactoring.
    - **Cross-Platform:** Windows 11 Pro für die MQL5-Anbindung und PyQt6-GUI; Ubuntu 22.04 VM für Linux-spezifische ML-Workflows.
* **KI-Anweisung:** Code muss modular, dokumentiert und "Self-Explanatory" sein, um die automatische Erstellung von Hilfe-Seiten zu unterstützen.

## 3. Kommerzialisierung (Digistore24 & Web)
Das Produkt muss "Out-of-the-box" professionell wirken und sicher sein.

### A. Lizenzierung & Sicherheit
- Implementierung eines Lizenzschlüsselsystems (Anbindung an Digistore24 API oder externer Validierungsserver).
- Schutz des geistigen Eigentums durch Code-Obfuskation im MQL5-Teil und ggf. Verschlüsselung der exportierten Modell-Gewichte.

### B. Produkt-Webseite & Integration
- Integration der Verkaufsseite in die bestehende Web-Infrastruktur.
- Bereitstellung von HTML/Markdown-Dokumentation für "Help Centers".

### C. Content-Erstellung (Marketing & Tutorials)
- **Demo-Videos:** Aufzeichnung von Live-Backtests und Echtzeit-Signalgenerierung (Blink-Logik).
- **Tutorials:** Schritt-für-Schritt-Anleitungen von der Daten-Export-Funktion im MT5 bis zum Training in der Python-App.
- **Hilfe-Seiten:** Dynamische FAQ-Listen, die technische Hürden (z.B. CUDA-Treiber Installation) adressieren.

## 4. Google-Tools Ecosystem (Management)
Zur effizienten Steuerung der kommerziellen Phase werden folgende Dienste genutzt:
- **Google Drive:** Zentrale Ablage für Marketing-Assets, Videomanuskripte und rechtliche Dokumente.
- **YouTube:** Hosten der Tutorial-Serie und Produkt-Demos (Unlisted/Public-Strategie).
- **Google Search Console:** Überwachung der Sichtbarkeit der Produktseite.
- **Google Keep/Tasks:** Schnelles Sammeln von Feature-Requests aus Kundenfeedback.

## 5. Meilensteine bis zum Launch
1. **Beta-Phase:** Funktionsfähiger MQL5-EA mit Anbindung an die lokale Python-API.
2. **UX-Optimierung:** Finalisierung der GUI (Blink-Logik, Profit-Tracker) für maximale Benutzerfreundlichkeit.
3. **Tutorial-Produktion:** Erstellung der Dokumentation parallel zum Code-Freeze.
4. **Digistore24 Setup:** Integration von Zahlungsabwicklung und automatisiertem Lizenzversand.
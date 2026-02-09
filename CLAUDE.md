# MetaTrader 5 Trading Repo

Modulares MQL5-Projekt mit Klassen (SMA/EMA), Expert Advisors, Indikatoren, Skripte und GitHub-Integration. Nur MT5-Syntax, keine MT4-Kompatibilität.

## Tech Stack
- **Sprache**: MQL5 (C++-ähnlich)
- **IDE**: MetaEditor (Kompilierung), Cursor/VSCode (Editing)
- **Testing**: MT5 Strategy Tester
- **Versionierung**: Git (gitignore .ex5, Logs)

## Projektstruktur


## MT5 Pfade (DEINE Installation!)
MetaEditor: D:\trading\mt5dev\metaeditor64.exe
StrategyTester: D:\trading\mt5dev\tester64.exe

## Coding Standards (MQL5 only!)
- **ENGLISH COMMENTS ONLY**: All comments, docstrings, and documentation MUST be in English
- Variable names: English (camelCase oder snake_case)
- Function names: English, descriptive (z.B. `CalculateMovingAverage()`)
- NO German comments/translations in code
- Example:
  ```mql5
  // GOOD - English comment
  // Calculate SMA for given period and shift
  double CalculateSMA(int period, int shift) { ... }
  
  // BAD - No German comments!
  // Schlechter - kein Deutsch!
  
  

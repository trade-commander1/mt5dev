//+------------------------------------------------------------------+
//|                                               ntcp_exporter.mq5  |
//|                              Copyright 2026, trade-commander.com |
//|                                  https://www.trade-commander.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, trade-commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property description "NTCP Data Exporter — Sync chart data to Python data folder"
#property description "Uses WinAPI to write directly outside the MT5 sandbox."
#property description "Requires: Tools > Options > Expert Advisors > Allow DLL imports"

//--- Input parameters
input int    InpBarsToExport = 0;  // Bars to export (0 = all available)
input bool   InpAutoSync     = false;   // Auto-export on every new bar

//+------------------------------------------------------------------+
//| WinAPI constants                                                  |
//+------------------------------------------------------------------+
#define GENERIC_WRITE          0x40000000
#define CREATE_ALWAYS          2
#define FILE_ATTRIBUTE_NORMAL  0x80
//#define INVALID_HANDLE         -1

//+------------------------------------------------------------------+
//| WinAPI imports — kernel32.dll                                     |
//+------------------------------------------------------------------+
#import "kernel32.dll"
long CreateFileW(string lpFileName, uint dwDesiredAccess, uint dwShareMode,
                 long lpSecurityAttributes, uint dwCreationDisposition,
                 uint dwFlagsAndAttributes, long hTemplateFile);
int  WriteFile(long hFile, uchar &lpBuffer[], uint nNumberOfBytesToWrite,
               uint &lpNumberOfBytesWritten[], long lpOverlapped);
int  CloseHandle(long hObject);
int  CreateDirectoryW(string lpPathName, long lpSecurityAttributes);
#import

//+------------------------------------------------------------------+
//| Constants                                                         |
//+------------------------------------------------------------------+
const string BTN_NAME   = "ntcp_sync_btn";
const int    CHUNK_SIZE  = 1000;  // CSV lines per WinAPI write call

//+------------------------------------------------------------------+
//| Globals                                                           |
//+------------------------------------------------------------------+
string   g_symbol;
string   g_timeframe;
string   g_filename;
string   g_outputDir;
string   g_fullPath;
datetime g_lastBarTime;

//+------------------------------------------------------------------+
//| Convert ENUM_TIMEFRAMES to short string  (PERIOD_M5 -> "M5")     |
//+------------------------------------------------------------------+
string PeriodToString(ENUM_TIMEFRAMES tf)
{
   string s = EnumToString(tf);
   StringReplace(s, "PERIOD_", "");
   return s;
}

//+------------------------------------------------------------------+
//| Strip characters that are invalid in filenames                    |
//+------------------------------------------------------------------+
string SanitizeSymbol(string sym)
{
   StringReplace(sym, "/",  "");
   StringReplace(sym, "#",  "");
   StringReplace(sym, "\\", "");
   StringReplace(sym, ":",  "");
   return sym;
}

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol      = SanitizeSymbol(Symbol());
   g_timeframe   = PeriodToString(_Period);
   g_filename    = "NTCP_DATA_" + g_symbol + "_" + g_timeframe + ".csv";
   g_lastBarTime = 0;

   // Absolute path built from the terminal installation directory
   g_outputDir = TerminalInfoString(TERMINAL_PATH)
               + "\\MQL5\\Experts\\trade-commander\\ntcp\\python\\data";
   g_fullPath  = g_outputDir + "\\" + g_filename;

   // Ensure the target directory exists (harmless if it already does)
   CreateDirectoryW(g_outputDir, 0);

   CreateSyncButton();

   Print("[NTCP] Initialized on ", Symbol(), " ", g_timeframe);
   Print("[NTCP] Output: ", g_fullPath);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectDelete(0, BTN_NAME);
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Create the dark-themed sync button                                |
//+------------------------------------------------------------------+
void CreateSyncButton()
{
   ObjectDelete(0, BTN_NAME);
   ObjectCreate(0, BTN_NAME, OBJ_BUTTON, 0, 0, 0);

   ObjectSetInteger(0, BTN_NAME, OBJPROP_CORNER,       CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_XDISTANCE,    220);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_YDISTANCE,    20);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_XSIZE,        200);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_YSIZE,        32);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_FONTSIZE,     10);
   ObjectSetString (0, BTN_NAME, OBJPROP_FONT,         "Segoe UI");
   ObjectSetString (0, BTN_NAME, OBJPROP_TEXT,
                    ShortToString(0x25BA) + " Sync " + g_symbol + " "
                    + g_timeframe + " to Python");
   // Dark theme
   ObjectSetInteger(0, BTN_NAME, OBJPROP_COLOR,        clrWhite);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_BGCOLOR,      C'30,30,40');
   ObjectSetInteger(0, BTN_NAME, OBJPROP_BORDER_COLOR, C'60,60,80');
   ObjectSetInteger(0, BTN_NAME, OBJPROP_STATE,        false);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_SELECTABLE,   false);

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Update button text and colours for visual feedback                |
//+------------------------------------------------------------------+
void SetButtonStyle(string text, color bgClr, color txtClr = clrWhite)
{
   ObjectSetString (0, BTN_NAME, OBJPROP_TEXT,    text);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_BGCOLOR, bgClr);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_COLOR,   txtClr);
   ObjectSetInteger(0, BTN_NAME, OBJPROP_STATE,   false);
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Write a UTF-8 string to an open WinAPI file handle                |
//+------------------------------------------------------------------+
bool WinWriteString(long hFile, string text)
{
   uchar buf[];
   int len = StringToCharArray(text, buf, 0, WHOLE_ARRAY, CP_UTF8);
   if(len <= 1)
      return true;   // empty string — nothing to write

   uint toWrite = (uint)(len - 1);   // exclude null terminator
   uint written[1] = {0};
   return (WriteFile(hFile, buf, toWrite, written, 0) != 0);
}

//+------------------------------------------------------------------+
//| Export chart data to CSV via WinAPI                                |
//+------------------------------------------------------------------+
int ExportData()
{
   //--- Determine bar count
   int available = Bars(Symbol(), _Period);
   int count     = (InpBarsToExport <= 0 || InpBarsToExport > available)
                   ? available : InpBarsToExport;
   if(count <= 0)
   {
      Print("[NTCP] ERROR: No bars available for ", Symbol(), " ", g_timeframe);
      return 0;
   }

   //--- Fetch rates (index 0 = oldest bar)
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(Symbol(), _Period, 0, count, rates);
   if(copied <= 0)
   {
      Print("[NTCP] ERROR: CopyRates failed — code ", GetLastError());
      return 0;
   }

   //--- Open file for writing
   long hFile = CreateFileW(g_fullPath, GENERIC_WRITE, 0, 0,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
   if(hFile == INVALID_HANDLE)
   {
      Print("[NTCP] ERROR: CreateFileW failed for: ", g_fullPath);
      Print("[NTCP] Verify the directory exists and DLL imports are allowed.");
      return 0;
   }

   //--- Write CSV header
   if(!WinWriteString(hFile, "Date,Open,High,Low,Close,TickVolume\r\n"))
   {
      Print("[NTCP] ERROR: Failed to write CSV header");
      CloseHandle(hFile);
      return 0;
   }

   //--- Write data rows in chunks for performance
   int    digits = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   string chunk  = "";

   for(int i = 0; i < copied; i++)
   {
      chunk += TimeToString(rates[i].time, TIME_DATE | TIME_SECONDS) + ","
             + DoubleToString(rates[i].open,  digits) + ","
             + DoubleToString(rates[i].high,  digits) + ","
             + DoubleToString(rates[i].low,   digits) + ","
             + DoubleToString(rates[i].close, digits) + ","
             + IntegerToString(rates[i].tick_volume) + "\r\n";

      if((i + 1) % CHUNK_SIZE == 0 || i == copied - 1)
      {
         if(!WinWriteString(hFile, chunk))
         {
            Print("[NTCP] ERROR: Write failed at bar ", i);
            CloseHandle(hFile);
            return 0;
         }
         chunk = "";
      }
   }

   CloseHandle(hFile);
   return copied;
}

//+------------------------------------------------------------------+
//| Chart event handler — sync button click                           |
//+------------------------------------------------------------------+
void OnChartEvent(const int    id,
                  const long   &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id != CHARTEVENT_OBJECT_CLICK || sparam != BTN_NAME)
      return;

   //--- Visual feedback: exporting
   SetButtonStyle("Exporting...", C'200,120,0');

   int exported = ExportData();

   if(exported > 0)
   {
      string ok = ShortToString(0x2714) + " " + g_filename
                + " " + IntegerToString(exported);
      SetButtonStyle(ok, C'0,120,60');
      Print("[NTCP] SUCCESS: ", exported, " bars -> ", g_fullPath);
      Alert("Export of ", exported, " bars completed successfully.");
   }
   else
   {
      SetButtonStyle(ShortToString(0x2718) + " Export Failed", C'180,30,30');
   }

   // Reset button after brief delay so the user sees the result
   Sleep(3000);
   CreateSyncButton();
}

//+------------------------------------------------------------------+
//| Tick handler — optional auto-sync on new bar                      |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!InpAutoSync)
      return;

   datetime barTime = iTime(Symbol(), _Period, 0);
   if(barTime == g_lastBarTime)
      return;
   g_lastBarTime = barTime;

   SetButtonStyle("Auto-syncing...", C'200,120,0');
   int exported = ExportData();

   if(exported > 0)
   {
      string ok = ShortToString(0x2714) + " Synced "
                + IntegerToString(exported);
      SetButtonStyle(ok, C'0,120,60');
      Print("[NTCP] Auto-sync: ", exported, " bars -> ", g_fullPath);
   }
   else
   {
      SetButtonStyle(ShortToString(0x2718) + " Sync Failed", C'180,30,30');
   }
}
//+------------------------------------------------------------------+

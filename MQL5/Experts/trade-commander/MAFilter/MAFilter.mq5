//+------------------------------------------------------------------+
//| MAFilter.mq5                                                      |
//| MA Filter Strategy — entry/exit verification EA                    |
//+------------------------------------------------------------------+
#property copyright "NTCP"
#property version   "1.00"
#property strict

//--- Input parameters
input double LotSize            = 0.1;    // Lot size
input int    MinLen              = 5;      // Shortest MA period
input int    MaxLen              = 21;     // Longest MA period
input int    NbrMa               = 4;      // Number of MAs (log-spaced)
input double MinSlopeFactor      = 1.0;    // Min slope vs average slope
input double MinLaminarLevel     = 0.8;    // Fslope ordering threshold
input double MaxBandwidthFactor  = 1.0;    // Max BW / SMA(BW) ratio
input int    NH                  = 1000;   // BW historical SMA window
input double StdDevFactor        = 1.0;    // StdDev multiplier for exit A
input int    ExitOption          = 0;      // 0=StdDev exit, 1=Slope exit

//--- Magic number for position identification
input int    MagicNumber         = 20260210;

//--- Global variables
int    g_maPeriods[];
int    g_maHandles[];
int    g_stdDevHandle = INVALID_HANDLE;
int    g_nbrMa = 0;

//+------------------------------------------------------------------+
//| Generate log-spaced MA periods                                     |
//+------------------------------------------------------------------+
void GenerateMAPeriods(int minLen, int maxLen, int count, int &periods[])
{
   ArrayResize(periods, count);

   if(count < 2)
   {
      periods[0] = minLen;
      return;
   }

   double logRatio = MathLog((double)maxLen / minLen);

   // Temporary array for deduplication
   int temp[];
   ArrayResize(temp, count);
   int uniqueCount = 0;

   for(int i = 0; i < count; i++)
   {
      double raw = minLen * MathExp(i / (double)(count - 1) * logRatio);
      int rounded = (int)MathMax(2, MathRound(raw));

      // Check for duplicate
      bool isDuplicate = false;
      for(int j = 0; j < uniqueCount; j++)
      {
         if(temp[j] == rounded)
         {
            isDuplicate = true;
            break;
         }
      }

      if(!isDuplicate)
      {
         temp[uniqueCount] = rounded;
         uniqueCount++;
      }
   }

   ArrayResize(periods, uniqueCount);
   for(int i = 0; i < uniqueCount; i++)
      periods[i] = temp[i];
}

//+------------------------------------------------------------------+
//| Compute LaminarLevel (pairwise slope concordance)                  |
//| Returns value in [-1, +1]                                          |
//| +1 = shorter MAs have steeper slopes (proper trend fan-out)        |
//| -1 = longer MAs have steeper slopes (reverse ordering)             |
//+------------------------------------------------------------------+
double ComputeLaminarLevel(const double &slopes[], int count)
{
   int totalPairs = count * (count - 1) / 2;
   if(totalPairs == 0)
      return 0.0;

   int concordant = 0;
   for(int a = 0; a < count; a++)
   {
      for(int b = a + 1; b < count; b++)
      {
         // Concordant if shorter-period MA (index a) has larger slope
         if(slopes[a] > slopes[b])
            concordant++;
         else if(slopes[a] < slopes[b])
            concordant--;
      }
   }

   return (double)concordant / totalPairs;
}

//+------------------------------------------------------------------+
//| Compute bandwidth: max(MAs) - min(MAs)                            |
//+------------------------------------------------------------------+
double ComputeBandwidth(const double &maValues[], int count)
{
   double maxVal = maValues[0];
   double minVal = maValues[0];

   for(int i = 1; i < count; i++)
   {
      if(maValues[i] > maxVal) maxVal = maValues[i];
      if(maValues[i] < minVal) minVal = maValues[i];
   }

   return maxVal - minVal;
}

//+------------------------------------------------------------------+
//| Check if we have a position with our magic number                  |
//| Returns: +1 = long, -1 = short, 0 = flat                          |
//+------------------------------------------------------------------+
double GetPosition()
{
   double positions=0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PositionSelectByTicket(PositionGetTicket(i)))
         continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)
      {
         positions += PositionGetDouble(POSITION_VOLUME);
      }
      else if(posType == POSITION_TYPE_SELL)
      {
         positions -= PositionGetDouble(POSITION_VOLUME);
      }

   }

   return positions;
}

//+------------------------------------------------------------------+
//| Get order filling type for symbol                                                    |
//+------------------------------------------------------------------+

ENUM_ORDER_TYPE_FILLING GetFillingBySymbol(const string symbol)
{
   uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);

   if ((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;
   else if ((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;
   else
      return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Open a position                                                    |
//+------------------------------------------------------------------+
bool OpenPosition(int direction)
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = _Symbol;
   request.volume       = LotSize;
   request.magic        = MagicNumber;
   request.type_filling = GetFillingBySymbol(_Symbol);//(ENUM_ORDER_TYPE_FILLING) ORDER_FILLING_IOC;//SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   
   
   request.deviation = 10;

   if(direction > 0)
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.comment = "MAFilter Long";
   }
   else
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      request.comment = "MAFilter Short";
   }

   if(!OrderSend(request, result))
   {
      Print("OrderSend failed: ", GetLastError(),
            " retcode=", result.retcode);
      return false;
   }

   Print(direction > 0 ? "LONG" : "SHORT",
         " opened at ", request.price);
   return true;
}

//+------------------------------------------------------------------+
//| Close all positions with our magic number                          |
//+------------------------------------------------------------------+
bool ClosePosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};

      request.action    = TRADE_ACTION_DEAL;
      request.symbol    = _Symbol;
      request.volume    = PositionGetDouble(POSITION_VOLUME);
      request.deviation = 10;
      request.position  = ticket;
      request.type_filling = GetFillingBySymbol(_Symbol);//(ENUM_ORDER_TYPE_FILLING) ORDER_FILLING_IOC;//SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

      ENUM_POSITION_TYPE posType =
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(posType == POSITION_TYPE_BUY)
      {
         request.type  = ORDER_TYPE_SELL;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      }
      else
      {
         request.type  = ORDER_TYPE_BUY;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      }

      if(!OrderSend(request, result))
      {
         Print("Close failed: ", GetLastError(),
               " retcode=", result.retcode);
         return false;
      }

      Print("Position closed at ", request.price);
   }

   return true;
}

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   // Generate MA periods
   GenerateMAPeriods(MinLen, MaxLen, NbrMa, g_maPeriods);
   g_nbrMa = ArraySize(g_maPeriods);

   if(g_nbrMa < 2)
   {
      Print("Error: Need at least 2 unique MA periods. Got ", g_nbrMa);
      return INIT_FAILED;
   }

   // Create MA handles
   ArrayResize(g_maHandles, g_nbrMa);

   for(int i = 0; i < g_nbrMa; i++)
   {
      g_maHandles[i] = iMA(_Symbol, PERIOD_CURRENT, g_maPeriods[i],
                            0, MODE_SMA, PRICE_CLOSE);
      if(g_maHandles[i] == INVALID_HANDLE)
      {
         Print("Failed to create MA(", g_maPeriods[i], ") handle");
         return INIT_FAILED;
      }
      else
         Print("Create MA(", g_maPeriods[i], ") handle");
   }

   // StdDev handle for exit option A (based on slowest MA period)
   g_stdDevHandle = iStdDev(_Symbol, PERIOD_CURRENT, g_maPeriods[g_nbrMa - 1],
                             0, MODE_SMA, PRICE_CLOSE);
   if(g_stdDevHandle == INVALID_HANDLE)
   {
      Print("Failed to create StdDev handle");
      return INIT_FAILED;
   }

   // Log configuration
   string periods = "";
   for(int i = 0; i < g_nbrMa; i++)
   {
      periods += IntegerToString(g_maPeriods[i]);
      if(i < g_nbrMa - 1) periods += ", ";
   }

   Print("MAFilter initialized | Periods: [", periods, "]",
         " | MinSlopeFactor=", MinSlopeFactor,
         " | MinLaminar=", MinLaminarLevel,
         " | MaxBWFactor=", MaxBandwidthFactor,
         " | NH=", NH,
         " | Exit=", ExitOption == 0 ? "StdDev" : "Slope");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int i = 0; i < g_nbrMa; i++)
   {
      if(g_maHandles[i] != INVALID_HANDLE)
         IndicatorRelease(g_maHandles[i]);
   }

   if(g_stdDevHandle != INVALID_HANDLE)
      IndicatorRelease(g_stdDevHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   // Bars needed: NH + 2 (NH for BW SMA, +1 for slope, +1 safety)
   int barsNeeded = NH + 2;
   int barsAvailable = Bars(_Symbol, PERIOD_CURRENT);
   if(barsAvailable < barsNeeded)
   {
      // Not enough history yet
      return;
   }

   //--- Copy MA buffers for all periods ---
   // maData[k][i]: k=period index, i=0(oldest)..barsNeeded-1(newest=bar 0)
   double maData[];
   ArrayResize(maData, g_nbrMa * barsNeeded);

   for(int k = 0; k < g_nbrMa; k++)
   {
      double buf[];
      ArrayResize(buf, barsNeeded);
      if(CopyBuffer(g_maHandles[k], 0, 0, barsNeeded, buf) < barsNeeded)
      {
         Print("CopyBuffer failed for MA(", g_maPeriods[k], ")");
         lastBarTime = 0;  // retry next tick
         return;
      }
      // buf[0]=oldest, buf[barsNeeded-1]=bar 0 (current)
      for(int i = 0; i < barsNeeded; i++)
         maData[k * barsNeeded + i] = buf[i];
   }

   //--- Copy StdDev buffer (only need current bar) ---
   double stdDevBuf[];
   ArrayResize(stdDevBuf, 2);
   if(CopyBuffer(g_stdDevHandle, 0, 0, 2, stdDevBuf) < 2)
   {
      lastBarTime = 0;
      return;
   }
   double stdDevCurrent = stdDevBuf[1]; // bar 0
   
   
   

   //--- Current close price (bar 1, last completed bar) ---
   // Actually we evaluate on bar 1 (completed) to avoid repainting
   // bar 0 = forming bar, bar 1 = last completed
   // In the maData array: bar 1 = index barsNeeded-2, bar 2 = index barsNeeded-3
   int bar1 = barsNeeded - 2;  // last completed bar
   int bar2 = barsNeeded - 3;  // bar before that

   double closePrice = iClose(_Symbol, PERIOD_CURRENT, 1);
   
   
   //--- Get MA values and slopes at bar 1 ---
   double maBar1[];
   ArrayResize(maBar1, g_nbrMa);
   double slopes[];
   ArrayResize(slopes, g_nbrMa);

   for(int k = 0; k < g_nbrMa; k++)
   {
      maBar1[k] = maData[k * barsNeeded + bar1];
      double maPrev = maData[k * barsNeeded + bar2];
      slopes[k] = maBar1[k] - maPrev;
   }

   //--- Condition 1: Close vs all MAs ---
   bool closeAboveAll = true;
   bool closeBelowAll = true;
   for(int k = 0; k < g_nbrMa; k++)
   {
      if(closePrice <= maBar1[k]) closeAboveAll = false;
      if(closePrice >= maBar1[k]) closeBelowAll = false;
   }

   //--- Condition 2: All slopes same sign ---
   bool allSlopesPos = true;
   bool allSlopesNeg = true;
   for(int k = 0; k < g_nbrMa; k++)
   {
      if(slopes[k] <= 0) allSlopesPos = false;
      if(slopes[k] >= 0) allSlopesNeg = false;
   }

   //--- Condition 3: Each slope vs its own NH-bar historical average ---
   // For each MA k, compute SMA(|slope[k]|, NH) and check current >= factor * avg
   bool slopesStrong = true;
   for(int k = 0; k < g_nbrMa; k++)
   {
      double histSum = 0;
      int histCount = 0;
      for(int shift = 0; shift < NH && shift < barsNeeded - 3; shift++)
      {
         int barCur = bar1 - shift;
         int barPrv = barCur - 1;
         if(barPrv < 0) break;
         double s = MathAbs(maData[k * barsNeeded + barCur]
                          - maData[k * barsNeeded + barPrv]);
         histSum += s;
         histCount++;
      }
      double avgAbsSlope = (histCount > 0) ? histSum / histCount : 0.0;
      double slopeThreshold = MinSlopeFactor * avgAbsSlope;
      if(MathAbs(slopes[k]) < slopeThreshold)
      {
         slopesStrong = false;
         break;
      }
   }
   
   

   //--- Condition 4: LaminarLevel ---
   double laminarLevel = ComputeLaminarLevel(slopes, g_nbrMa);
   bool laminarLong  = laminarLevel >= MinLaminarLevel;
   bool laminarShort = laminarLevel <= -MinLaminarLevel;
   
  // PrintFormat("´close=%.8f stdDevCurrent=%.8f avgAbsSlope=%.8f laminarLevel=%.8f",closePrice,stdDevCurrent,avgAbsSlope,laminarLevel);

   //--- Condition 5: Bandwidth factor ---
   double bwBar1 = ComputeBandwidth(maBar1, g_nbrMa);
   double bwSum = 0;
   int bwCount = 0;

   for(int shift = 0; shift < NH && shift < barsNeeded - 2; shift++)
   {
      int barIdx = bar1 - shift;
      if(barIdx < 0) break;

      double maAtShift[];
      ArrayResize(maAtShift, g_nbrMa);
      for(int k = 0; k < g_nbrMa; k++)
         maAtShift[k] = maData[k * barsNeeded + barIdx];

      bwSum += ComputeBandwidth(maAtShift, g_nbrMa);
      bwCount++;
   }

   double bwSma = (bwCount > 0) ? bwSum / bwCount : 0.0;
   double bwFactor = (bwSma > 1e-10) ? bwBar1 / bwSma : 1.0;
   bool bwOk = bwFactor <= MaxBandwidthFactor;
   
   
   //PrintFormat("´close=%.8f stdDevCurrent=%.8f avgAbsSlope=%.8f laminarLevel=%.8f bwFactor=%.8f",closePrice,stdDevCurrent,avgAbsSlope,laminarLevel,bwFactor);

   //--- Current position ---
   double pos = GetPosition();
   
//  PrintFormat("posDir=%d le=%d se=%d close=%.8f caa=%d cba=%d stdDevCurrent=%.8f slopesStrong=%d laminarLevel=%.8f bwFactor=%.8f"
//   ,posDir,longEntry,shortEntry,closePrice,closeAboveAll,closeBelowAll,stdDevCurrent,slopesStrong,laminarLevel,bwFactor);
   

   //--- Exit logic ---
   if(pos != 0)
   {
      bool exitSignal = false;

      if(ExitOption == 0)
      {
         // Exit A: Close breaches slowest MA +/- StdDevFactor * StdDev
         double slowestMA = maBar1[g_nbrMa - 1];
         double band = StdDevFactor * stdDevCurrent;

         if(pos > 0 && closePrice < slowestMA - band)
            exitSignal = true;
         if(pos < 0 && closePrice > slowestMA + band)
            exitSignal = true;
            
        if(exitSignal == true)
        PrintFormat("pos=%.2f close=%.8f slowestMA=%.8f stdDevCurrent=%.8f band=%.8f",pos,closePrice,slowestMA,stdDevCurrent,band);
            
      }
      else
      {
         // Exit B: Fastest MA slope reverses
         double fastestSlope = slopes[0];
         if(pos > 0 && fastestSlope < 0)
            exitSignal = true;
         if(pos < 0 && fastestSlope > 0)
            exitSignal = true;
      }

      if(exitSignal)
      {
         Print("EXIT signal | pos=", pos > 0 ? "LONG" : "SHORT",
               " | close=", closePrice,
               " | slowMA=", maBar1[g_nbrMa - 1],
               " | stddev=", stdDevCurrent,
               " | slope[0]=", slopes[0]);
         ClosePosition();
         return;  // do not open new trade on same bar as exit
      }

      return;  // already in a position, skip entry check
   }


   //--- Entry signals ---
   bool longEntry  = closeAboveAll && allSlopesPos && slopesStrong
                     && laminarLong && bwOk;
   bool shortEntry = closeBelowAll && allSlopesNeg && slopesStrong
                     && laminarShort && bwOk;


   if(longEntry)
   {
      Print("LONG entry | close=", closePrice,
            " | MAs=[", maBar1[0], ",", maBar1[g_nbrMa-1], "]",
            " | laminar=", DoubleToString(laminarLevel, 3),
            " | bwFactor=", DoubleToString(bwFactor, 3),
            " | slopes=[", slopes[0], ",", slopes[g_nbrMa-1], "]");
      OpenPosition(+1);
   }
   else if(shortEntry)
   {
      Print("SHORT entry | close=", closePrice,
            " | MAs=[", maBar1[0], ",", maBar1[g_nbrMa-1], "]",
            " | laminar=", DoubleToString(laminarLevel, 3),
            " | bwFactor=", DoubleToString(bwFactor, 3),
            " | slopes=[", slopes[0], ",", slopes[g_nbrMa-1], "]");
      OpenPosition(-1);
   }
}
//+------------------------------------------------------------------+

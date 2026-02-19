//+------------------------------------------------------------------+
//| LaminarLevel.mq5                                                  |
//| Laminar Level indicator using tcMABundle                          |
//| Displays laminar level (-1 to 1) in separate window               |
//+------------------------------------------------------------------+
#property copyright "Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_minimum -1
#property indicator_maximum 1
#property indicator_level1 0
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue

#include <trade-commander\moving_average.mqh>

input int  InpMinLen   = 5;   // Min MA Length
input int  InpMaxLen   = 21;  // Max MA Length
input int  InpNbrMA    = 4;   // Number of MAs
input bool InpStrict   = false;   // Strict Flag

double ExtLaminarBuffer[];

tcMABundle m_bundle;

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, ExtLaminarBuffer, INDICATOR_DATA);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpMaxLen);

   int minLen = MathMin(InpMinLen, InpMaxLen);
   int maxLen = MathMax(InpMinLen, InpMaxLen);
   int nbrMA = MathMax(1, InpNbrMA);

   m_bundle.setup(nbrMA, minLen, maxLen, 1000, TC_MA_SMA);

   IndicatorSetString(INDICATOR_SHORTNAME,
      "LaminarLevel(" + IntegerToString(minLen) + "-" + IntegerToString(maxLen) + "," + IntegerToString(nbrMA) + ")");
   IndicatorSetInteger(INDICATOR_DIGITS, 3);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total <= 0) return 0;

   int minLen = MathMin(InpMinLen, InpMaxLen);
   int maxLen = MathMax(InpMinLen, InpMaxLen);
   int nbrMA = MathMax(1, InpNbrMA);

   //m_bundle.setup(nbrMA, minLen, maxLen, 1000, TC_MA_SMA);
   if(prev_calculated == 0)
   {
      ArrayInitialize(ExtLaminarBuffer, EMPTY_VALUE);
   }
   
   bool isNewBar=(rates_total > prev_calculated ? true : false);

   for(int i = prev_calculated; i < rates_total && !IsStopped(); i++)
   {
      m_bundle.update(close[i], 1.0, isNewBar);
      ExtLaminarBuffer[i] = m_bundle.laminar_level(InpStrict, false);
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}

//+------------------------------------------------------------------+
//| TrendShot.mq5                                                    |
//| Trend strength indicator using tcMABundle                         |
//| LaminarLevel * SlopeLevel / SlopeFactor / (BandwidthLevel/MaxBW)  |
//|   * Pow(RelativePricePositionLevel, 4)                            |
//+------------------------------------------------------------------+
#property copyright "Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue

#include <trade-commander\moving_average.mqh>

input int    InpMinLen            = 5;    // Min MA Length
input int    InpMaxLen            = 21;   // Max MA Length
input int    InpNbrMA             = 4;    // Number of MAs
input int    InpHistLen           = 1000;  // History Length
input bool   InpVolumeWeighted    = false;
input bool   InpOHCL              = false; // OHCL mean as input
input double InpMaxBandwidthLevel = 1.0;
input double InpSlopeFactor       = 1.0;
input bool   InpStrict            = false; // Strict laminar order
input double InpSigThreshold      = 1.0;   // Signal threshold (>= 1)

double ExtTrendShotBuffer[];

tcMABundle m_bundle;

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, ExtTrendShotBuffer, INDICATOR_DATA);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpHistLen);

   int minLen = MathMin(InpMinLen, InpMaxLen);
   int maxLen = MathMax(InpMinLen, InpMaxLen);
   int nbrMA = MathMax(1, InpNbrMA);

   m_bundle.setup(nbrMA, minLen, maxLen, InpHistLen, TC_MA_SMA);

   double sigThresh = MathMax(1.0, InpSigThreshold);
   IndicatorSetInteger(INDICATOR_LEVELS, 3);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, sigThresh);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrGreen);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, 0);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 1, clrWhite);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 2, -sigThresh);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 2, clrRed);

   IndicatorSetString(INDICATOR_SHORTNAME,
      "TrendShot(" + IntegerToString(minLen) + "-" + IntegerToString(maxLen) + "," + IntegerToString(nbrMA) + ")");
   IndicatorSetInteger(INDICATOR_DIGITS, 4);

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

   //m_bundle.setup(nbrMA, minLen, maxLen, InpHistLen, TC_MA_SMA);

   if(prev_calculated == 0)
   {
      ArrayInitialize(ExtTrendShotBuffer, EMPTY_VALUE);
   }

   double slopeFactor = MathMax(1e-10, InpSlopeFactor);
   double maxBwLevel = MathMax(1e-10, InpMaxBandwidthLevel);

   bool isNewbar=(rates_total > prev_calculated ? true : false);
   for(int i = prev_calculated; i < rates_total && !IsStopped(); i++)
   {
      double value = InpOHCL
         ? (open[i] + high[i] + low[i] + 2.0 * close[i]) / 5.0
         : close[i];
      double weight = InpVolumeWeighted ? (double)tick_volume[i] : 1.0;

      m_bundle.update(value, weight, isNewbar);

      double laminar = MathAbs(m_bundle.laminar_level(InpStrict, false));
      
      double slopeLevel = MathPow(MathAbs(m_bundle.SlopeLevel()),0.25);
      double SlopeRatio = slopeLevel / slopeFactor;
      
      //tcMA* histBw = m_bundle.GetHistoricalBandwidth();
      double relPricePos = m_bundle.CloseMALevel(value);

      double bwRatio = MathMax(1e-10, m_bundle.bandwidth_level() / maxBwLevel);
      
      double mag = relPricePos;
      double result = relPricePos;//laminar  / bwRatio * mag;

      ExtTrendShotBuffer[i] = result;
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}

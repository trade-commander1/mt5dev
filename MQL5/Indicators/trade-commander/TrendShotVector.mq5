//+------------------------------------------------------------------+
//| TrendShotVector.mq5                                               |
//| Vector-based trend indicator — scalar product vs norm vector      |
//| Copyright 2026, Trade-Commander.com                               |
//+------------------------------------------------------------------+
#property copyright "Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue

#include <trade-commander\TrendShotVectorAlerts.mqh>

input int    InpMinLen            = 5;     // Min MA Length
input int    InpMaxLen            = 21;    // Max MA Length
input int    InpNbrMA             = 4;     // Number of MAs
input int    InpHistLen           = 1000;  // History Length
input int    Inpdiffpoints        = 0;     // Difference points for bandwidth normalization
input bool   InpVolumeWeighted    = false;
input bool   InpOHCL              = false; // OHCL mean as input
input bool   InpStrict            = false; // Strict laminar order
input double InpNormLaminar       = 1.0;   // Norm Vector: laminar_level
input double InpNormSlopeLevel    = 1.0;   // Norm Vector: SlopeLevel
input double InpNormBandwidthLevel= 1.0;   // Norm Vector: bandwidth_level
input double InpNormCloseMALevel  = 1.0;   // Norm Vector: CloseMALevel
input double InpIndicatorMin     = -3.0;  // Indicator minimum (scale only)
input double InpIndicatorMax     = 3.0;   // Indicator maximum (scale only)
input double InpIndicatorTriggerLong  = 1.0;   // Trigger Long (green level)
input double InpIndicatorTriggerShort = -1.0;  // Trigger Short (red level)
input double InpStdDevLevel           = 1.1;   // StdDev multiplier for exit
input bool   InpEnableAlert           = true;  // Enable alerts

double ExtBuffer[];
double ExtSlowestMABuffer[];
double ExtStdDevBuffer[];

TrendShotVectorAlerts m_alerts;

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, ExtBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ExtSlowestMABuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(2, ExtStdDevBuffer, INDICATOR_CALCULATIONS);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, InpHistLen);

   int minLen = MathMin(InpMinLen, InpMaxLen);
   int maxLen = MathMax(InpMinLen, InpMaxLen);
   int nbrMA = MathMax(1, InpNbrMA);

   m_alerts.SetMinLen(InpMinLen);
   m_alerts.SetMaxLen(InpMaxLen);
   m_alerts.SetNbrMA(InpNbrMA);
   m_alerts.SetHistLen(InpHistLen);
   m_alerts.SetDiffpoints(Inpdiffpoints);
   m_alerts.SetPointValue(SymbolInfoDouble(_Symbol, SYMBOL_POINT));
   m_alerts.SetVolumeWeighted(InpVolumeWeighted);
   m_alerts.SetOHCL(InpOHCL);
   m_alerts.SetStrict(InpStrict);
   m_alerts.SetNormLaminar(InpNormLaminar);
   m_alerts.SetNormSlopeLevel(InpNormSlopeLevel);
   m_alerts.SetNormBandwidthLevel(InpNormBandwidthLevel);
   m_alerts.SetNormCloseMALevel(InpNormCloseMALevel);
   m_alerts.SetTriggerLong(InpIndicatorTriggerLong);
   m_alerts.SetTriggerShort(InpIndicatorTriggerShort);
   m_alerts.SetStdDevLevel(InpStdDevLevel);
   m_alerts.SetEnableAlert(InpEnableAlert);

   if(!m_alerts.Setup())
      return INIT_FAILED;

   IndicatorSetDouble(INDICATOR_MINIMUM, InpIndicatorMin);
   IndicatorSetDouble(INDICATOR_MAXIMUM, InpIndicatorMax);
   IndicatorSetInteger(INDICATOR_LEVELS, 3);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, 0.0);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 0, clrWhite);
   IndicatorSetInteger(INDICATOR_LEVELSTYLE, 0, STYLE_DASH);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, InpIndicatorTriggerLong);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 1, clrGreen);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 2, InpIndicatorTriggerShort);
   IndicatorSetInteger(INDICATOR_LEVELCOLOR, 2, clrRed);

   IndicatorSetString(INDICATOR_SHORTNAME,
      "TrendShotVector(" + IntegerToString(minLen) + "-" + IntegerToString(maxLen) + "," + IntegerToString(nbrMA) + ")");
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

   if(prev_calculated == 0)
   {
      ArrayInitialize(ExtBuffer, EMPTY_VALUE);
      ArrayInitialize(ExtSlowestMABuffer, 0.0);
      ArrayInitialize(ExtStdDevBuffer, 0.0);
   }

   bool isNewbar = (rates_total > prev_calculated ? true : false);
   int startIdx = (isNewbar && rates_total >= 2) ? 0 : prev_calculated;

   for(int i = startIdx; i < rates_total && !IsStopped(); i++)
   {
      double value = InpOHCL
         ? (open[i] + high[i] + low[i] + 2.0 * close[i]) / 5.0
         : close[i];
      double weight = InpVolumeWeighted ? (double)tick_volume[i] : 1.0;

      m_alerts.UpdateBar(value, weight, isNewbar);
      if(InpEnableAlert && isNewbar && rates_total >= 2 && i == rates_total - 1)
         m_alerts.CheckAlerts();

      ExtBuffer[i] = m_alerts.GetLastIndicatorValue();
      ExtSlowestMABuffer[i] = m_alerts.GetLastSlowestMA();
      ExtStdDevBuffer[i] = m_alerts.GetLastStdDev();
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}
//+------------------------------------------------------------------+

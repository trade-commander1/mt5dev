//+------------------------------------------------------------------+
//| TrendShotVectorEA.mq5                                             |
//| Example EA using TrendShotVectorAlerts for Entry/Exit signals      |
//| Copyright 2026, Trade-Commander.com                               |
//+------------------------------------------------------------------+
#property copyright "Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict

#include <trade-commander\TrendShotVectorAlerts.mqh>

input int    InpMinLen            = 5;     // Min MA Length
input int    InpMaxLen            = 21;    // Max MA Length
input int    InpNbrMA             = 4;     // Number of MAs
input int    InpHistLen           = 1000;  // History Length
input int    Inpdiffpoints        = 0;     // Difference points for bandwidth
input bool   InpVolumeWeighted    = false;
input bool   InpOHCL              = false;
input bool   InpStrict            = false;
input double InpNormLaminar       = 1.0;
input double InpNormSlopeLevel    = 1.0;
input double InpNormBandwidthLevel= 2.0;
input double InpNormCloseMALevel  = 1.0;
input double InpIndicatorTriggerLong  = 3.0;
input double InpIndicatorTriggerShort = -3.0;
input double InpStdDevLevel           = 1.1;
input bool   InpEnableAlert           = true;

TrendShotVectorAlerts m_alerts;

//+------------------------------------------------------------------+
int OnInit()
{
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

   if(!m_alerts.SyncHistory(InpHistLen))
      Print("TrendShotVectorEA: SyncHistory failed, may need more bars");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime s_lastBar = 0;
   MqlRates rates[];
   if(CopyRates(_Symbol, Period(), 0, 2, rates) < 2) return;

   datetime currentBar = rates[1].time;
   bool isNewBar = (currentBar != s_lastBar);
   s_lastBar = currentBar;

   double value = InpOHCL
      ? (rates[1].open + rates[1].high + rates[1].low + 2.0 * rates[1].close) / 5.0
      : rates[1].close;
   double weight = InpVolumeWeighted ? (double)rates[1].tick_volume : 1.0;

   m_alerts.UpdateBar(value, weight, isNewBar);
   if(isNewBar)
      m_alerts.CheckAlerts();
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}

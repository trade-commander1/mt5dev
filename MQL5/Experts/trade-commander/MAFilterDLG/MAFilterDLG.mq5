//+------------------------------------------------------------------+
//| MAFilterDLG.mq5                                                   |
//| MA Filter Strategy with Dialog — human approval before execution  |
//| Uses Standard Library Controls (Include/Controls)                |
//+------------------------------------------------------------------+
#property copyright "NTCP"
#property version   "1.00"
#property strict

//--- Input parameters
input double LotSize            = 0.1;    // Lot size (base)
input int    MinLen              = 5;     // Shortest MA period
input int    MaxLen              = 21;    // Longest MA period
input int    NbrMa               = 4;     // Number of MAs (log-spaced)
input double MinSlopeFactor      = 1.0;    // Min slope vs average slope
input double MinLaminarLevel     = 0.8;   // Slope ordering threshold
input double MaxBandwidthFactor  = 1.0;   // Max BW / SMA(BW) ratio
input int    NH                  = 1000;  // BW historical SMA window
input double StdDevFactor        = 1.0;   // StdDev multiplier for exit
input int    ExitOption          = 0;     // 0=StdDev exit, 1=Slope exit
input bool   VolumeWeighted      = false; // Weight averages by volume
input bool   StrictSlopeOrder    = false; // Strict slope order
input bool   UseOHCLMean         = false; // Use OHCL Mean as input
input int    MagicNumber         = 0;     // 0 = hash from Symbol+File+TF
input double BaseVolume         = 0.01;   // Base volume for orders
input double MaxDepositLoad     = 50.0;   // Max deposit load %
input double EquityDrawDownThresholdPCT = 30.0;  // Equity drawdown threshold %
input bool   Debug                 = true;   // Show debug levels (Close-MA, Laminar, Bandwidth, Slope)

#include "MAFilterDialog.mqh"

CMAFilterDialog ExtDialog;

//+------------------------------------------------------------------+
int OnInit()
{
   if(!ExtDialog.Create(0, "MAFilterDLG", 0, 20, 20, 420, 420))
      return INIT_FAILED;
   if(!ExtDialog.Run())
      return INIT_FAILED;
   EventSetMillisecondTimer(500);
   Print("MAFilterDLG initialized — Controls Dialog");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   ExtDialog.Destroy(reason);
}

//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   ExtDialog.ChartEvent(id, lparam, dparam, sparam);
}

//+------------------------------------------------------------------+
void OnTimer()
{
   ExtDialog.OnBlinkTimer();
}

//+------------------------------------------------------------------+
void OnTick()
{
   ExtDialog.OnTickUpdate();
}
//+------------------------------------------------------------------+

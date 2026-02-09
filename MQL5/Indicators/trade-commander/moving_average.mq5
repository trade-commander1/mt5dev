//+------------------------------------------------------------------+
//|                                              moving_average.mq5   |
//|                         Copyright 2026, Trade-Commander.com       |
//|                               https://www.trade-commander.com     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
#property indicator_label1  "tcMA"
#property indicator_applied_price PRICE_CLOSE

#include <trade-commander/moving_average.mqh>

input int         InpMAPeriod = 14;             // MA Period
input TC_MA_TYPE  InpMAType   = TC_MA_SMA;      // MA Type
input bool        InpVWAP     = false;          // Use tick volume as weight

double maBuffer[];
tcMA   maInstance;
double calcWeight;
int    barsProcessed;

//+------------------------------------------------------------------+
int OnInit()
{
    SetIndexBuffer(0, maBuffer, INDICATOR_DATA);
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);

    // Calculate default weight (used when VWAP is off)
    if (InpMAType == TC_MA_SMA)
        calcWeight = 1.0;
    else
        calcWeight = 2.0 / (InpMAPeriod + 1.0);

    maInstance.setup(InpMAPeriod, 0, InpMAType);
    barsProcessed = 0;

    string typeName;
    switch (InpMAType)
    {
        case TC_MA_SMA:  typeName = "SMA";  break;
        case TC_MA_EMA:  typeName = "EMA";  break;
        case TC_MA_DEMA: typeName = "DEMA"; break;
        case TC_MA_TEMA: typeName = "TEMA"; break;
        default:         typeName = "MA";   break;
    }
    IndicatorSetString(INDICATOR_SHORTNAME,
        "tc" + typeName + "(" + IntegerToString(InpMAPeriod) + ")");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
    if (rates_total <= 0) return 0;

    if (prev_calculated == 0)
    {
        maInstance.setup(InpMAPeriod, 0, InpMAType);
        barsProcessed = 0;
        ArrayInitialize(maBuffer, EMPTY_VALUE);
    }

    // Get tick volumes if VWAP mode is enabled
    long tickVol[];
    if (InpVWAP)
        CopyTickVolume(Symbol(), Period(), 0, rates_total, tickVol);

    // Only feed completed bars into the accumulator
    int completedBars = rates_total - 1;

    for (int i = barsProcessed; i < completedBars; i++)
    {
        double weight = InpVWAP ? (double)tickVol[i] : calcWeight;
        maBuffer[i] = maInstance.update(price[i], weight);
    }

    barsProcessed = completedBars;

    // Current forming bar: show current MA value
    maBuffer[rates_total - 1] = maInstance.GetMA();
    

    return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}
//+------------------------------------------------------------------+

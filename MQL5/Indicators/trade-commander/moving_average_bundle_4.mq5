//+------------------------------------------------------------------+
//|                                     moving_average_bundle_4.mq5   |
//|                         Copyright 2026, Trade-Commander.com       |
//|                               https://www.trade-commander.com     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Trade-Commander.com"
#property link      "https://www.trade-commander.com"
#property version   "1.00"
#property strict
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   4
#include <trade-commander/moving_average.mqh>

#define MA_COUNT 4

input int         InpMinLen  = 10;              // Min MA Period
input int         InpMaxLen  = 100;             // Max MA Period
input TC_MA_TYPE  InpMAType  = TC_MA_SMA;       // MA Type
input bool        InpVWAP    = false;           // Volume Weighted

double Buffer0[], Buffer1[], Buffer2[], Buffer3[];

tcMA*  maInst[];
int    periods[];
double calcWeights[];
int    barsProcessed;

//+------------------------------------------------------------------+
//| HSV heat gradient from red (short) to blue (long)                |
//+------------------------------------------------------------------+
color HeatColor(int index, int total)
{
    if (total <= 1) return clrRed;

    double ratio = (double)index / (double)(total - 1);
    double hue   = ratio * 240.0;
    double hp    = hue / 60.0;
    double x     = 1.0 - MathAbs(fmod(hp, 2.0) - 1.0);

    double r = 0, g = 0, b = 0;

    if      (hp < 1.0) { r = 1.0; g = x;   b = 0;   }
    else if (hp < 2.0) { r = x;   g = 1.0; b = 0;   }
    else if (hp < 3.0) { r = 0;   g = 1.0; b = x;   }
    else if (hp < 4.0) { r = 0;   g = x;   b = 1.0; }
    else               { r = 0;   g = 0;   b = 1.0; }

    int ri = (int)MathRound(r * 255.0);
    int gi = (int)MathRound(g * 255.0);
    int bi = (int)MathRound(b * 255.0);

    return (color)((bi << 16) | (gi << 8) | ri);
}

//+------------------------------------------------------------------+
void SetBuffer(int idx, int bar, double val)
{
    switch (idx)
    {
        case 0: Buffer0[bar] = val; break;
        case 1: Buffer1[bar] = val; break;
        case 2: Buffer2[bar] = val; break;
        case 3: Buffer3[bar] = val; break;
    }
}

//+------------------------------------------------------------------+
void InitAllBuffers()
{
    ArrayInitialize(Buffer0, EMPTY_VALUE);
    ArrayInitialize(Buffer1, EMPTY_VALUE);
    ArrayInitialize(Buffer2, EMPTY_VALUE);
    ArrayInitialize(Buffer3, EMPTY_VALUE);
}

//+------------------------------------------------------------------+
int OnInit()
{
    SetIndexBuffer(0, Buffer0, INDICATOR_DATA);
    SetIndexBuffer(1, Buffer1, INDICATOR_DATA);
    SetIndexBuffer(2, Buffer2, INDICATOR_DATA);
    SetIndexBuffer(3, Buffer3, INDICATOR_DATA);

    int minLen = (int)MathMin(InpMinLen, InpMaxLen);
    int maxLen = (int)MathMax(InpMinLen, InpMaxLen);

    ArrayResize(maInst,  MA_COUNT);
    ArrayResize(periods, MA_COUNT);
    ArrayResize(calcWeights, MA_COUNT);

    for (int i = 0; i < MA_COUNT; i++)
    {
        periods[i] = (MA_COUNT == 1)
            ? minLen
            : minLen + i * (maxLen - minLen) / (MA_COUNT - 1);

        calcWeights[i] = (InpMAType == TC_MA_SMA)
            ? 1.0
            : 2.0 / (periods[i] + 1.0);

        maInst[i] = new tcMA();
        maInst[i].setup(periods[i], 0, InpMAType);

        PlotIndexSetInteger(i, PLOT_DRAW_TYPE,   DRAW_LINE);
        PlotIndexSetInteger(i, PLOT_LINE_STYLE,  STYLE_SOLID);
        PlotIndexSetInteger(i, PLOT_LINE_WIDTH,  1);
        PlotIndexSetInteger(i, PLOT_LINE_COLOR,  HeatColor(i, MA_COUNT));
        PlotIndexSetDouble(i,  PLOT_EMPTY_VALUE, EMPTY_VALUE);
        PlotIndexSetString(i,  PLOT_LABEL,
            "MA(" + IntegerToString(periods[i]) + ")");
    }

    barsProcessed = 0;
    IndicatorSetString(INDICATOR_SHORTNAME,
        "tcMABundle4(" + IntegerToString(minLen) + "-" + IntegerToString(maxLen) + ")");

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
    if (rates_total <= 0) return 0;

    if (prev_calculated == 0)
    {
        for (int j = 0; j < MA_COUNT; j++)
            maInst[j].setup(periods[j], 0, InpMAType);
        barsProcessed = 0;
        InitAllBuffers();
    }

    int completedBars = rates_total - 1;

    for (int i = barsProcessed; i < completedBars; i++)
    {
        for (int j = 0; j < MA_COUNT; j++)
        {
            double w = InpVWAP ? (double)tick_volume[i] : calcWeights[j];
            SetBuffer(j, i, maInst[j].update(close[i], w));
        }
    }

    barsProcessed = completedBars;

    for (int j = 0; j < MA_COUNT; j++)
        SetBuffer(j, rates_total - 1, maInst[j].GetMA());

    return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    for (int i = 0; i < MA_COUNT; i++)
    {
        if (maInst[i] != NULL)
        {
            delete maInst[i];
            maInst[i] = NULL;
        }
    }
}
//+------------------------------------------------------------------+

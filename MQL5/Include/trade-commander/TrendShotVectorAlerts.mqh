//+------------------------------------------------------------------+
//| TrendShotVectorAlerts.mqh                                         |
//| TrendShotVector alert logic (Entry Long/Short, Exit Long/Short)   |
//| encapsulated for use in Expert Advisors. Has tcMABundle as member.|
//| Copyright 2026, Trade-Commander.com                               |
//+------------------------------------------------------------------+
#property strict

#include <trade-commander\TrendShotVector.mqh>

//+------------------------------------------------------------------+
//| TrendShotVectorAlerts — Entry/Exit signals for EA use             |
//| Holds tcMABundle and TrendShotVector; evaluates signals on new bar|
//+------------------------------------------------------------------+
class TrendShotVectorAlerts
{
private:
   tcMABundle      m_bundle;
   TrendShotVector m_indicator;

   // Alert parameters
   double m_triggerLong;
   double m_triggerShort;
   double m_stdDevLevel;
   bool   m_enableAlert;
   bool   m_ohcl;
   bool   m_volumeWeighted;

   // State
   int    m_currentSig;           // 1=Long, -1=Short, 0=Flat
   bool   m_longAlertFired;
   bool   m_shortAlertFired;
   bool   m_longExitAlertFired;
   bool   m_shortExitAlertFired;

   string m_symbol;
   ENUM_TIMEFRAMES m_timeframe;

   // Last bar values (current after UpdateBar)
   double m_lastIndVal;
   double m_lastValue;
   double m_lastSlowestMA;
   double m_lastStdDev;
   bool   m_hasLastBar;

   // Last completed bar values (for CheckAlerts when isNewBar)
   double m_prevIndVal;
   double m_prevValue;
   double m_prevSlowestMA;
   double m_prevStdDev;
   bool   m_hasPrevBar;

   /**
    * Evaluate alert logic for last completed bar. Call only on new bar.
    */
   void EvaluateAlerts(double indVal, double value, double slowestMA, double stdDev)
   {
      if(!m_enableAlert) return;

      double band = m_stdDevLevel * stdDev;
      bool longExitCond = (value <= slowestMA - band);
      bool shortExitCond = (value >= slowestMA + band);

      if(longExitCond && m_currentSig > 0)
      {
         if(!m_longExitAlertFired)
         {
            OnLongExit();
            m_longExitAlertFired = true;
         }
         m_currentSig = 0;
         m_longAlertFired = false;
      }
      else if(shortExitCond && m_currentSig < 0)
      {
         if(!m_shortExitAlertFired)
         {
            OnShortExit();
            m_shortExitAlertFired = true;
         }
         m_currentSig = 0;
         m_shortAlertFired = false;
      }
      else if(indVal >= m_triggerLong)
      {
         if(m_currentSig <= 0)
         {
            m_currentSig = 1;
            m_longAlertFired = false;
            m_longExitAlertFired = false;
         }
         if(!m_longAlertFired)
         {
            OnLongEntry();
            m_longAlertFired = true;
         }
      }
      else if(indVal <= m_triggerShort)
      {
         if(m_currentSig >= 0)
         {
            m_currentSig = -1;
            m_shortAlertFired = false;
            m_shortExitAlertFired = false;
         }
         if(!m_shortAlertFired)
         {
            OnShortEntry();
            m_shortAlertFired = true;
         }
      }
   }

protected:
   /**
    * Override in derived class for custom handling. Default: Alert().
    */
   virtual void OnLongEntry()
   {
      Alert(m_symbol, " ", EnumToString(m_timeframe), " TrendShotVector Long");
   }

   virtual void OnShortEntry()
   {
      Alert(m_symbol, " ", EnumToString(m_timeframe), " TrendShotVector Short");
   }

   virtual void OnLongExit()
   {
      Alert(m_symbol, " ", EnumToString(m_timeframe), " TrendShotVector Long Exit");
   }

   virtual void OnShortExit()
   {
      Alert(m_symbol, " ", EnumToString(m_timeframe), " TrendShotVector Short Exit");
   }

public:
   TrendShotVectorAlerts() : m_triggerLong(3.0), m_triggerShort(-3.0), m_stdDevLevel(1.1),
      m_enableAlert(true), m_ohcl(false), m_volumeWeighted(false),
      m_currentSig(0), m_longAlertFired(false), m_shortAlertFired(false),
      m_longExitAlertFired(false), m_shortExitAlertFired(false),
      m_symbol(""), m_timeframe(PERIOD_CURRENT),
      m_lastIndVal(0), m_lastValue(0), m_lastSlowestMA(0), m_lastStdDev(0), m_hasLastBar(false),
      m_prevIndVal(0), m_prevValue(0), m_prevSlowestMA(0), m_prevStdDev(0), m_hasPrevBar(false) {}

   /**
    * Get reference to the tcMABundle (for direct access if needed).
    */
   tcMABundle* GetMABundle() { return GetPointer(m_bundle); }

   /**
    * Set MA bundle parameters (forward to TrendShotVector).
    */
   void SetMinLen(int v)       { m_indicator.SetMinLen(v); }
   void SetMaxLen(int v)       { m_indicator.SetMaxLen(v); }
   void SetNbrMA(int v)        { m_indicator.SetNbrMA(v); }
   void SetHistLen(int v)      { m_indicator.SetHistLen(v); }
   void SetDiffpoints(int v)   { m_indicator.SetDiffpoints(v); }
   void SetPointValue(double v){ m_indicator.SetPointValue(v); }
   void SetVolumeWeighted(bool v) { m_volumeWeighted = v; m_indicator.SetVolumeWeighted(v); }
   void SetOHCL(bool v)        { m_ohcl = v; m_indicator.SetOHCL(v); }
   void SetStrict(bool v)      { m_indicator.SetStrict(v); }

   void SetNormLaminar(double v)      { m_indicator.SetNormLaminar(v); }
   void SetNormSlopeLevel(double v)   { m_indicator.SetNormSlopeLevel(v); }
   void SetNormBandwidthLevel(double v){ m_indicator.SetNormBandwidthLevel(v); }
   void SetNormCloseMALevel(double v) { m_indicator.SetNormCloseMALevel(v); }

   void SetTriggerLong(double v)   { m_triggerLong = v; }
   void SetTriggerShort(double v)  { m_triggerShort = v; }
   void SetStdDevLevel(double v)   { m_stdDevLevel = v; }
   void SetEnableAlert(bool v)     { m_enableAlert = v; }

   void SetSymbol(string s)        { m_symbol = s; }
   void SetTimeframe(ENUM_TIMEFRAMES tf) { m_timeframe = tf; }

   /**
    * Initialize. Call SetMinLen, SetMaxLen, etc. before Setup.
    * Call once before ProcessBar() loop.
    */
   bool Setup(string symbol = "", ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT)
   {
      if(symbol != "") m_symbol = symbol;
      if(timeframe != PERIOD_CURRENT) m_timeframe = timeframe;
      if(m_symbol == "") m_symbol = _Symbol;
      if(m_timeframe == PERIOD_CURRENT) m_timeframe = Period();

      m_indicator.UseExternalBundle(GetPointer(m_bundle));
      int minLen = m_indicator.GetMinLen();
      int maxLen = m_indicator.GetMaxLen();
      int nbrMA = m_indicator.GetNbrMA();
      int histLen = m_indicator.GetHistLen();
      int minL = MathMin(minLen, maxLen);
      int maxL = MathMax(minLen, maxLen);
      int nbr = MathMax(1, nbrMA);
      m_bundle.setup(nbr, minL, maxL, histLen, TC_MA_SMA);
      return m_indicator.Setup();
   }

   /**
    * Update indicator with one bar's data. Call first.
    */
   void UpdateBar(double value, double weight, bool isNewBar)
   {
      if(isNewBar && m_hasLastBar)
      {
         m_prevIndVal = m_lastIndVal;
         m_prevValue = m_lastValue;
         m_prevSlowestMA = m_lastSlowestMA;
         m_prevStdDev = m_lastStdDev;
         m_hasPrevBar = true;
      }

      m_indicator.Update(value, weight, isNewBar);

      double sp, lv;
      double pointValue = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
      m_indicator.Calculate(value, sp, lv, pointValue);

      double indVal = m_indicator.GetIndicatorValue(sp, lv);
      double slowestMA = m_indicator.GetSlowestMA();
      double stdDev = m_indicator.GetSlowestStdDev();

      m_lastIndVal = indVal;
      m_lastValue = value;
      m_lastSlowestMA = slowestMA;
      m_lastStdDev = stdDev;
      m_hasLastBar = true;
   }

   /**
    * Evaluate Entry/Exit alerts for the last completed bar. Call after UpdateBar when isNewBar.
    */
   void CheckAlerts()
   {
      if(m_enableAlert && m_hasPrevBar)
         EvaluateAlerts(m_prevIndVal, m_prevValue, m_prevSlowestMA, m_prevStdDev);
   }

   /**
    * Get current signal state: 1=Long, -1=Short, 0=Flat.
    */
   int GetCurrentSignal() const { return m_currentSig; }

   /**
    * Get last computed indicator value.
    */
   double GetLastIndicatorValue() const { return m_lastIndVal; }

   /**
    * Get last computed slowest MA (for indicator buffers).
    */
   double GetLastSlowestMA() const { return m_lastSlowestMA; }

   /**
    * Get last computed StdDev (for indicator buffers).
    */
   double GetLastStdDev() const { return m_lastStdDev; }

   /**
    * Sync historical bars (oldest to newest). Call after Setup() in OnInit.
    * @param bars Number of bars to sync (default: InpHistLen or 1000)
    * @return true on success
    */
   bool SyncHistory(int bars = 0)
   {
      if(bars <= 0) bars = m_indicator.GetHistLen();
      MqlRates rates[];
      int copied = CopyRates(m_symbol, m_timeframe, 0, bars, rates);
      if(copied <= 0) return false;

      for(int i = 0; i < copied && !IsStopped(); i++)
      {
         double value = m_ohcl
            ? (rates[i].open + rates[i].high + rates[i].low + 2.0 * rates[i].close) / 5.0
            : rates[i].close;
         double weight = m_volumeWeighted ? (double)rates[i].tick_volume : 1.0;
         UpdateBar(value, weight, false);
      }
      return true;
   }
};

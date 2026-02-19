//+------------------------------------------------------------------+
//| TrendShotVector.mqh                                               |
//| Vector-based trend indicator using tcMABundle                      |
//| Scalar product (cosine similarity) of Market Vector vs Norm Vector|
//| Copyright 2026, Trade-Commander.com                              |
//+------------------------------------------------------------------+
#property strict

#include <trade-commander\moving_average.mqh>

#define TrendShotVector_components 4
//+------------------------------------------------------------------+
//| TrendShotVector — vector-based trend strength indicator           |
//| MV = (laminar_level, SlopeLevel, bandwidth_level, CloseMALevel)  |
//| NV = norm vector (configurable). SP = cosine similarity, LV = |MV|/|NV|
//+------------------------------------------------------------------+
class TrendShotVector
{
private:
   tcMABundle   m_bundle;
   tcMABundle*  m_pExternalBundle;  // If set, use this instead of m_bundle
   int          m_minLen;
   int          m_maxLen;
   int          m_nbrMA;
   int          m_histLen;
   int          m_diffpoints;
   double       m_pointValue;
   bool         m_volumeWeighted;
   bool         m_ohcl;
   bool         m_strict;
   double       m_nvLaminar;
   double       m_nvSlopeLevel;
   double       m_nvBandwidthLevel;
   double       m_nvCloseMALevel;
   double       m_LastIndicatorValue;
   // Latest market vector
   double         m_mv[TrendShotVector_components];
   // Norm vector
   double         m_nv[TrendShotVector_components];
public:
   TrendShotVector() : m_pExternalBundle(NULL), m_minLen(5), m_maxLen(21), m_nbrMA(4), m_histLen(1000),
      m_diffpoints(0), m_pointValue(0.0), m_volumeWeighted(false), m_ohcl(false), m_strict(false),
      m_nvLaminar(1.0), m_nvSlopeLevel(1.0), m_nvBandwidthLevel(1.0), m_nvCloseMALevel(1.0) ,m_LastIndicatorValue(0.0)
      {
         // Initialize market vector     
         for(int i=0;i < TrendShotVector_components;++i)
            m_mv[i]=0.0;

         // Initialize norm vector            
         int idx=-1;
         m_nv[++idx]= m_nvLaminar;
         m_nv[++idx]= m_nvSlopeLevel;
         m_nv[++idx]= m_nvBandwidthLevel;
         m_nv[++idx]= m_nvCloseMALevel;
      }

   /**
    * Use an external tcMABundle instead of the internal one. Call before Setup().
    * The external bundle must be set up by the caller.
    */
   void UseExternalBundle(tcMABundle* pBundle) { m_pExternalBundle = pBundle; }

   /**
    * Set MA bundle parameters.
    */
   void SetMinLen(int v)       { m_minLen = v; }
   void SetMaxLen(int v)       { m_maxLen = v; }
   void SetNbrMA(int v)        { m_nbrMA = v; }
   void SetHistLen(int v)      { m_histLen = v; }
   void SetDiffpoints(int v)   { m_diffpoints = v; }
   void SetPointValue(double v){ m_pointValue = v; }
   void SetVolumeWeighted(bool v) { m_volumeWeighted = v; }
   void SetOHCL(bool v)        { m_ohcl = v; }
   void SetStrict(bool v)      { m_strict = v; }

   /**
    * Set Norm Vector (NV) components. Default all = 1.
    */
   void SetNormLaminar(double v)      { m_nvLaminar = v; }
   void SetNormSlopeLevel(double v)  { m_nvSlopeLevel = v; }
   void SetNormBandwidthLevel(double v) { m_nvBandwidthLevel = v; }
   void SetNormCloseMALevel(double v){ m_nvCloseMALevel = v; }

   int GetMinLen() const { return m_minLen; }
   int GetMaxLen() const { return m_maxLen; }
   int GetNbrMA() const { return m_nbrMA; }
   int GetHistLen() const { return m_histLen; }
   
   double LastIndicatorValue() const {return m_LastIndicatorValue;}

   /**
    * Initialize the bundle. Call once before update loop.
    * If UseExternalBundle() was called, the external bundle must already be set up.
    */
   bool Setup()
   {
      if(m_pExternalBundle != NULL) return true;
      int minL = MathMin(m_minLen, m_maxLen);
      int maxL = MathMax(m_minLen, m_maxLen);
      int nbr = MathMax(1, m_nbrMA);
      m_bundle.setup(nbr, minL, maxL, m_histLen, TC_MA_SMA);
      return true;
   }

   /**
   * Dump market vector
   */
   void DumpMarketVector() const
   {      
      PrintFormat("lml=%.4f slp=%.3f bwl=%.3f cll=%.4f",m_mv[0],m_mv[1],m_mv[2],m_mv[3]);
   }
   /**
    * Update with new price data.
    */
   void Update(double value, double weight, bool isNewBar)
   {
      if(m_pExternalBundle != NULL)
         m_pExternalBundle.update(value, weight, isNewBar);
      else
         m_bundle.update(value, weight, isNewBar);
   }

   /**
    * Compute SP (scalar product / cosine similarity) and LV (length ratio).
    * @param value Current price value for CloseMALevel
    * @param spOut Output: scalar product (cosine similarity), range [-1, 1]
    * @param lvOut Output: length ratio |MV| / |NV|
    * @param pointValue Point value for bandwidth normalization when diffpoints>0. If 0, uses stored m_pointValue.
    */
   void Calculate(double value, double &spOut, double &lvOut, double pointValue = 0.0)
   {
      double pv = (pointValue > 0.0) ? pointValue : m_pointValue;

      double laminar, slopeLevel, bwLevel, closeMALevel;
      if(m_pExternalBundle != NULL)
      {
         laminar = m_pExternalBundle.laminar_level(m_strict, false);
         slopeLevel = m_pExternalBundle.SlopeLevel();
         bwLevel = m_pExternalBundle.bandwidth_level(m_diffpoints, pv);
         closeMALevel = m_pExternalBundle.CloseMALevel(value);
      }
      else
      {
         laminar = m_bundle.laminar_level(m_strict, false);
         slopeLevel = m_bundle.SlopeLevel();
         bwLevel = m_bundle.bandwidth_level(m_diffpoints, pv);
         closeMALevel = m_bundle.CloseMALevel(value);
      }
         
      // as BWL should be low for good signal, we need to amplify in vector
      //if(bwLevel > 0.0)
      //   bwLevel=1.0 / bwLevel;
         
      bwLevel = MathMax(0.0, 1.0 - bwLevel);
         
      if(MathAbs(laminar) < 1.0  || MathAbs(closeMALevel) < 1.0)
      {
         laminar=0.0;
         slopeLevel=0.0;
         bwLevel=0.0;
         closeMALevel=0.0;
      }
      
      // Market Vector
      //m_mv = { laminar, slopeLevel, bwLevel, closeMALevel };
      
      int idx=-1;
      m_mv[++idx]= laminar;
      m_mv[++idx]= slopeLevel;
      m_mv[++idx]= bwLevel;
      m_mv[++idx]= closeMALevel;      
      //double nv[] = { m_nvLaminar, m_nvSlopeLevel, m_nvBandwidthLevel, m_nvCloseMALevel };

      // Dot product MV · NV (raw scalar product)
      double dot = 0.0;
      for(int k = 0; k < TrendShotVector_components; k++)
         dot += m_mv[k] * m_nv[k];

      // |MV| and |NV|
      double lenMV = 0.0, lenNV2 = 0.0;
      for(int k = 0; k < TrendShotVector_components; k++)
      {
         lenMV += m_mv[k] * m_mv[k];
         lenNV2 += m_nv[k] * m_nv[k];
      }
      lenMV = MathSqrt(lenMV);
      double lenNV = MathSqrt(lenNV2);

      // SP = dot / |NV|² — projection of MV onto NV, can exceed ±1
      // SP >= 1 means strong Long, SP <= -1 means strong Short
      double sp = 0.0;
      if(lenNV2 > 1e-15)
         sp = dot / lenNV2;

      // Length ratio LV = |MV| / |NV|
      double lv = (lenNV > 1e-15) ? (lenMV / lenNV) : 0.0;

      spOut = sp;
      lvOut = lv;
      
      if(spOut >= 1.0)
         m_LastIndicatorValue=lv;
      else if(spOut <= -1.0)
         m_LastIndicatorValue=-lv;
      else
         m_LastIndicatorValue=0.0;
         
      /*
      if(MathAbs(m_LastIndicatorValue) >= 20)
      {
         DumpMarketVector();
         if(m_pExternalBundle != NULL)
            m_pExternalBundle.DumpSlopeAndBWL();
         else
            m_bundle.DumpSlopeAndBWL();
         
       }
       */
   }

   /**
    * Get slowest MA value (for exit condition).
    */
   double GetSlowestMA() const
   {
      int n;
      if(m_pExternalBundle != NULL)
      {
         n = m_pExternalBundle.GetNbrMA();
         return (n > 0) ? m_pExternalBundle.GetMAValue(n - 1) : 0.0;
      }
      n = m_bundle.GetNbrMA();
      return (n > 0) ? m_bundle.GetMAValue(n - 1) : 0.0;
   }

   /**
    * Get slowest MA StdDev (for exit condition).
    */
   double GetSlowestStdDev() const
   {
      int n;
      if(m_pExternalBundle != NULL)
      {
         n = m_pExternalBundle.GetNbrMA();
         return (n > 0) ? m_pExternalBundle.GetStdDev(n - 1) : 0.0;
      }
      n = m_bundle.GetNbrMA();
      return (n > 0) ? m_bundle.GetStdDev(n - 1) : 0.0;
   }

   /**
    * Get indicator output value: 1*LV if SP>=1, -1*LV if SP<=-1, else SP*LV.
    */
   double GetIndicatorValue(double sp, double lv)
   {
      if(sp >= 1.0)  return lv;
      if(sp <= -1.0) return -lv;
      return 0;//sp * lv;
   }
};
//+------------------------------------------------------------------+

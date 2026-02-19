//+------------------------------------------------------------------+
//|                                                      tcMA.mqh    |
//|                         Copyright 2026, Trade-Commander.com      |
//|                                   https://www.trade-commander.com|
//+------------------------------------------------------------------+
#property strict

// Moving average type enumeration
enum TC_MA_TYPE
{
    TC_MA_SMA  = 0,  // Simple Moving Average
    TC_MA_EMA  = 1,  // Exponential Moving Average
    TC_MA_DEMA = 2,  // Double Exponential Moving Average
    TC_MA_TEMA = 3   // Triple Exponential Moving Average
};

class tcMA
{
private:
    double sum;                // Accumulated sum of all values
    double ma;                 // Current Moving Average (MA)
    double ema1;               // First EMA pass (for EMA/DEMA/TEMA)
    double ema2;               // Second EMA pass (for DEMA/TEMA)
    double ema3;               // Third EMA pass (for TEMA)
    double minVal;             // Current minimum value
    double maxVal;             // Current maximum value
    double stdDev;             // Current standard deviation
    double slope;              // Current slope of the MA
    double signedDiff;         // Current signed difference between value and MA
    double m2;                 // Welford's running sum of squared deviations
    int count;                // Number of values used for calculation
    TC_MA_TYPE maType;         // Moving average type

    // Windowed SMA state
    int    maLength;           // Window size for SMA (0 = cumulative)
    double circBuf[];          // Circular buffer for windowed SMA
    double windowSum;          // Running sum of values in the window
    int    circIdx;            // Current write position in circular buffer

    // Snapshot before last update (for rollback when isNewBar=false)
    double snapSum, snapMa, snapEma1, snapEma2, snapEma3;
    double snapMinVal, snapMaxVal, snapStdDev, snapSlope, snapSignedDiff, snapM2;
    int    snapCount;
    double snapWindowSum, snapOldBufVal;
    int    snapCircIdx;

protected:
    tcMA* historicalMa;       // Pointer to the historical MA
    tcMA* historicalStdDev;   // Pointer to the historical standard deviation
    tcMA* historicalSlope;    // Pointer to the historical slope

    /**
     * Cleans up and deletes all historical MAs.
     */
    void cleanup()
    {
        if (historicalMa != NULL)
        {
            delete historicalMa;
            historicalMa = NULL;
        }
        if (historicalStdDev != NULL)
        {
            delete historicalStdDev;
            historicalStdDev = NULL;
        }
        if (historicalSlope != NULL)
        {
            delete historicalSlope;
            historicalSlope = NULL;
        }
    }

public:
    tcMA() : sum(0), ma(0), ema1(0), ema2(0), ema3(0),
             minVal(DBL_MAX), maxVal(-DBL_MAX), stdDev(0), slope(0), signedDiff(0), m2(0), count(0),
             maType(TC_MA_SMA), maLength(0), windowSum(0), circIdx(0),
             snapSum(0), snapMa(0), snapEma1(0), snapEma2(0), snapEma3(0),
             snapMinVal(DBL_MAX), snapMaxVal(-DBL_MAX), snapStdDev(0), snapSlope(0), snapSignedDiff(0), snapM2(0), snapCount(0),
             snapWindowSum(0), snapCircIdx(0), snapOldBufVal(0),
             historicalMa(NULL), historicalStdDev(NULL), historicalSlope(NULL) {}

    virtual ~tcMA()
    {
        cleanup();
    }

    /**
     * Updates the Moving Average (MA) with a new value and weight.
     *
     * @param value The new data point to be added to the MA calculation.
     * @param weight The weight/alpha factor. For SMA: if 1.0 plain average, otherwise weighted.
     *               For EMA/DEMA/TEMA: the EMA smoothing factor (alpha).
     * @return The updated Moving Average.
     */
    double update(double value, double weight=1.0,bool isNewBar=true)
    {
        if (isNewBar)
        {
            // Save snapshot before applying new value
            snapSum = sum; snapMa = ma; snapEma1 = ema1; snapEma2 = ema2; snapEma3 = ema3;
            snapMinVal = minVal; snapMaxVal = maxVal; snapStdDev = stdDev;
            snapSlope = slope; snapSignedDiff = signedDiff; snapM2 = m2; snapCount = count;
            snapWindowSum = windowSum; snapCircIdx = circIdx;
            if (maLength > 0) snapOldBufVal = circBuf[circIdx];
        }
        else
        {
            // Rollback to state before last update, then re-apply with new value
            sum = snapSum; ma = snapMa; ema1 = snapEma1; ema2 = snapEma2; ema3 = snapEma3;
            minVal = snapMinVal; maxVal = snapMaxVal; stdDev = snapStdDev;
            slope = snapSlope; signedDiff = snapSignedDiff; m2 = snapM2; count = snapCount;
            windowSum = snapWindowSum; circIdx = snapCircIdx;
            if (maLength > 0) circBuf[snapCircIdx] = snapOldBufVal;
        }

        if (count == 0)
        {
            // Initialize with the first value
            sum = value;
            ma = value;
            ema1 = value;
            ema2 = value;
            ema3 = value;
            minVal = value;
            maxVal = value;
            stdDev = 0;
            slope = 0;
            signedDiff = 0;
            // Initialize circular buffer for windowed SMA
            if (maLength > 0)
            {
                circBuf[0] = value;
                windowSum = value;
                circIdx = 1 % maLength;
            }
        }
        else
        {
            // Welford's online algorithm for standard deviation
            double oldMean = sum / count;
            double oldMa = ma;
            sum += value;
            int n = count + 1;
            double newMean = sum / n;
            double delta = value - oldMean;
            double delta2 = value - newMean;
            m2 += delta * delta2;
            stdDev = MathSqrt(m2 / n);

            // Update Moving Average based on type
            switch (maType)
            {
                case TC_MA_SMA:
                    if (weight != 1.0 && weight != 0.0)
                        ma = weight * value + (1.0 - weight) * ma;
                    else if (maLength > 0)
                    {
                        // Windowed SMA over the last maLength values
                        if (count >= maLength)
                            windowSum -= circBuf[circIdx];
                        windowSum += value;
                        circBuf[circIdx] = value;
                        circIdx = (circIdx + 1) % maLength;
                        ma = windowSum / MathMin(count + 1, maLength);
                    }
                    else
                        ma = newMean;
                    break;

                case TC_MA_EMA:
                    ema1 = weight * value + (1.0 - weight) * ema1;
                    ma = ema1;
                    break;

                case TC_MA_DEMA:
                    ema1 = weight * value + (1.0 - weight) * ema1;
                    ema2 = weight * ema1 + (1.0 - weight) * ema2;
                    ma = 2.0 * ema1 - ema2;
                    break;

                case TC_MA_TEMA:
                    ema1 = weight * value + (1.0 - weight) * ema1;
                    ema2 = weight * ema1 + (1.0 - weight) * ema2;
                    ema3 = weight * ema2 + (1.0 - weight) * ema3;
                    ma = 3.0 * ema1 - 3.0 * ema2 + ema3;
                    break;
            }

            // Update min and max values
            if (value < minVal) minVal = value;
            if (value > maxVal) maxVal = value;

            // Calculate slope
            slope = ma - oldMa;

            // Calculate signed difference
            signedDiff = value - ma;
        }

        if (historicalMa != NULL)
        {
            historicalMa.update(value, weight, isNewBar);
        }
        if (historicalStdDev != NULL)
        {
            historicalStdDev.update(stdDev, weight, isNewBar);
        }
        if (historicalSlope != NULL)
        {
            historicalSlope.update(MathAbs(slope), weight, isNewBar);
        }

        count++;
        return ma;
    }
   
    /**
     * Resets the Moving Average and sets a new length.
     *
     * @param length The new length of the MA.
     * @param lengthHist The length for the historical MA, stddev, and slope tcMA.
     * @param type The moving average type (SMA, EMA, DEMA, TEMA). Default is SMA.
     */
    void setup(int length, int lengthHist, TC_MA_TYPE type = TC_MA_SMA)
    {
        cleanup();

        maType = type;
        maLength = length;
        sum = 0;
        ma = 0;
        ema1 = 0;
        ema2 = 0;
        ema3 = 0;
        minVal = DBL_MAX;
        maxVal = -DBL_MAX;
        stdDev = 0;
        slope = 0;
        signedDiff = 0;
        m2 = 0;
        count = 0;
        windowSum = 0;
        circIdx = 0;
        if (maLength > 0)
        {
            ArrayResize(circBuf, maLength);
            ArrayInitialize(circBuf, 0.0);
        }

        snapSum = 0; snapMa = 0; snapEma1 = 0; snapEma2 = 0; snapEma3 = 0;
        snapMinVal = DBL_MAX; snapMaxVal = -DBL_MAX; snapStdDev = 0;
        snapSlope = 0; snapSignedDiff = 0; snapM2 = 0; snapCount = 0;
        snapWindowSum = 0; snapCircIdx = 0; snapOldBufVal = 0;

        if (lengthHist > 0)
        {
            historicalMa = new tcMA();
            historicalStdDev = new tcMA();
            historicalSlope = new tcMA();

            historicalMa.setup(lengthHist, 0, type);
            historicalStdDev.setup(lengthHist, 0, type);
            historicalSlope.setup(lengthHist, 0, type);
        }
    }

    /**
     * Retrieves the current Moving Average.
     *
     * @return The current Moving Average.
     */
    double GetMA() const { return ma; }

    /**
     * Retrieves the current minimum value.
     *
     * @return The current minimum value.
     */
    double GetMinValue() const { return minVal; }

    /**
     * Retrieves the current maximum value.
     *
     * @return The current maximum value.
     */
    double GetMaxValue() const { return maxVal; }

    /**
     * Retrieves the current standard deviation.
     *
     * @return The current standard deviation.
     */
    double GetStdDev() const { return stdDev; }

    /**
     * Retrieves the current slope of the Moving Average.
     *
     * @return The current slope of the Moving Average.
     */
    double GetSlope() const { return slope; }

    /**
     * Retrieves the signed difference between the value and the Moving Average.
     *
     * @return The signed difference between the value and the Moving Average.
     */
    double GetSignedDiff() const { return signedDiff; }

    /**
     * Retrieves the number of values processed.
     *
     * @return The current count.
     */
    int GetCount() const { return count; }

    /**
     * Retrieves the current MA type.
     *
     * @return The TC_MA_TYPE enum value.
     */
    TC_MA_TYPE GetMAType() const { return maType; }

    /**
     * Retrieves the historical average of the slope (when lengthHist > 0 in setup).
     */
    double GetHistoricalSlopeMA() const { return (historicalSlope != NULL) ? historicalSlope.GetMA() : 0.0; }

    /**
     * Returns the signed ratio of current slope vs historical slope.
     * Positive when current slope exceeds historical average, negative otherwise.
     * Returns 0 if historical slope is zero or unavailable.
     */
    double SlopeLevel() const
    {
        double histSlope = GetHistoricalSlopeMA();
        if (histSlope == 0.0) return 0.0;
        return GetSlope() / MathAbs(histSlope);
    }
    
    /**
     * Dump slope.
     */
    void DumpSlope(void)
    {
       PrintFormat("slope=%.4f hist-slope=%.4f",slope,historicalSlope.GetMA());
    }    
    
};

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                      tcMABundle.mqh|
//|                        Copyright 2026, Trade-Commander.com      |
//|                                       https://www.trade-commander.com|
//+------------------------------------------------------------------+
class tcMABundle
{
private:
    int nbrMA;
    tcMA* maArray[];
    tcMA* historicalBandwidth;   // Instance to track historical bandwidth
    TC_MA_TYPE maType;           // Moving average type for the bundle
    double bw;                   // Latest bandwidth
    double bwHist;               // Latest historical bandwidth    
protected:
    /**
     * Cleans up and deletes all MAs.
     */
    void cleanup()
    {
        for (int i = 0; i < nbrMA; ++i)
        {
            delete maArray[i];
        }
        ArrayFree(maArray);
        nbrMA = 0;

        if (historicalBandwidth != NULL)
        {
            delete historicalBandwidth;
            historicalBandwidth = NULL;
        }
    }

public:
    tcMABundle() : nbrMA(0), maType(TC_MA_SMA), historicalBandwidth(NULL),bw(0.0),bwHist(0.0) {}

    virtual ~tcMABundle()
    {
        cleanup();
    }

    /**
     * Sets up the tcMABundle with custom period lengths.
     *
     * @param periods Array of MA period lengths.
     * @param count Number of MAs (size of periods array).
     * @param HistLen The length for the historical MA, stddev, and slope tcMA (default is 1000).
     * @param type The moving average type (SMA, EMA, DEMA, TEMA). Default is SMA.
     */
    void setup(const int &periods[], int count, int HistLen = 1000, TC_MA_TYPE type = TC_MA_SMA)
    {
        cleanup();

        maType = type;
        nbrMA = count;
        ArrayResize(maArray, nbrMA);

        for (int i = 0; i < nbrMA; ++i)
        {
            maArray[i] = new tcMA();
            maArray[i].setup(periods[i], HistLen, type);
        }

        historicalBandwidth = new tcMA();
        historicalBandwidth.setup(HistLen, 0, type);
    }

    /**
     * Sets up the tcMABundle with a specified number of MAs and their lengths (linear spacing).
     *
     * @param NbrMA The number of MAs to create.
     * @param nMinLen The minimum length of the MAs.
     * @param nMaxLen The maximum length of the MAs.
     * @param HistLen The length for the historical MA, stddev, and slope tcMA (default is 1000).
     * @param type The moving average type (SMA, EMA, DEMA, TEMA). Default is SMA.
     */
    void setup(int NbrMA, int nMinLen, int nMaxLen, int HistLen = 1000, TC_MA_TYPE type = TC_MA_SMA)
    {
        cleanup();

        maType = type;
        nbrMA = NbrMA;
        ArrayResize(maArray, nbrMA);

        for (int i = 0; i < nbrMA; ++i)
        {
            maArray[i] = new tcMA();
            int length = (nbrMA == 1) ? nMinLen : nMinLen + i * (nMaxLen - nMinLen) / (nbrMA - 1);
            maArray[i].setup(length, HistLen, type);
        }

        // Initialize the historical bandwidth MA
        historicalBandwidth = new tcMA();
        historicalBandwidth.setup(HistLen, 0, type);
    }

    /**
     * Updates all MAs in the bundle with a new value and weight.
     *
     * @param value The new data point to be added to each MA calculation.
     * @param weight The weight factor for the moving average calculation.
     */
    void update(double value, double weight=1.0,bool isNewBar=true)
    {
        for (int i = 0; i < nbrMA; ++i)
        {
            maArray[i].update(value, weight, isNewBar);
        }

        // Update the historical bandwidth with the current bandwidth
        bw = calculate_bandwidth();
        bwHist=historicalBandwidth.update(bw, 1.0, isNewBar);
        //PrintFormat("bw=%.4f hist-bw=%.4f",bw,bwHist);
    }
    double bandwidth() const
    {
      return bw;
    }
    double bandwidth_hist() const
    {
      return bwHist;
    }
    double bandwidth_level(int diffpoints=0,double PointValue=0.0) const
    {
      if(diffpoints <= 0 || PointValue <= 0.0)
         return (bwHist > 0.0 ? bw /  bwHist : 0.0);
      else
      {
         return (bw / (PointValue*diffpoints*(nbrMA-1)));
      }
    }
    
    
    /**
     * Retrieves the laminar level of the MAs in the bundle.
     *
     * @param bstrict If true, breaks when direction changes strictly.
     * @param dump If true, prints debug information.
     * @return The laminar level, ranging from -1 to 1.
     */
    double laminar_level(bool bstrict = false, bool dump = false) const
    {
        int xorder = 0;
        double denom = nbrMA;

        if (nbrMA > 0)
        {
        
            denom = nbrMA + (nbrMA - 1);

            double prev_avg = 0.0;
            double prev_slope = 0.0; // unused in current logic block but kept

            for (int i = 0; i < nbrMA; ++i)
            {
                tcMA* pa = maArray[i];
                if (pa != NULL && pa.GetCount() > 0)
                {
                    double avg = pa.GetMA();
                    double avg_slope = pa.GetSlope();

                    if (avg_slope > 0.0)
                    {
                        if (bstrict == true && xorder < 0) break;
                        else xorder += 1;
                    }
                    else if (avg_slope < 0.0)
                    {
                        if (bstrict == true && xorder > 0) break;
                        else xorder -= 1;
                    }

                    if (i > 0)
                    {
                        if (prev_avg < avg)
                        {
                            if (bstrict == true && xorder > 0) break;
                            else xorder -= 1;
                        }
                        else if (prev_avg > avg)
                        {
                            if (bstrict == true && xorder < 0) break;
                            else xorder += 1;
                        }
                    }

                    if (dump)
                        PrintFormat("idx=%d avg=%.8f prev-avg=%.8f slope=%.8f prev-slope=%.8f order=%d ", i, avg, prev_avg, avg_slope, prev_slope, xorder);

                    prev_avg = avg;
                    prev_slope = avg_slope;
                }
            }
        }

        if (dump) PrintFormat("lamlevel=%.2f xorder=%d denom=%.2f", xorder / denom, xorder, denom);
        return (xorder / denom);
    }

    /**
     * Retrieves the current historical bandwidth.
     *
     * @return The current historical bandwidth MA.
     */
    tcMA* GetHistoricalBandwidth() const { return historicalBandwidth; }

    /**
     * Retrieves the current MA type of the bundle.
     *
     * @return The TC_MA_TYPE enum value.
     */
    TC_MA_TYPE GetMAType() const { return maType; }

    /**
     * Retrieves the number of MAs in the bundle.
     */
    int GetNbrMA() const { return nbrMA; }

    /**
     * Retrieves the MA value at the given index.
     */
    double GetMAValue(int i) const { return (i >= 0 && i < nbrMA && maArray[i] != NULL) ? maArray[i].GetMA() : 0.0; }

    /**
     * Retrieves the StdDev at the given index (slowest MA for exit band).
     */
    double GetStdDev(int i) const { return (i >= 0 && i < nbrMA && maArray[i] != NULL) ? maArray[i].GetStdDev() : 0.0; }

    /**
     * Retrieves the slope at the given index.
     */
    double GetSlope(int i) const { return (i >= 0 && i < nbrMA && maArray[i] != NULL) ? maArray[i].GetSlope() : 0.0; }

    /**
     * Retrieves the historical average of slope at the given index (for slope strength check).
     */
    double GetHistoricalSlopeMA(int i) const { return (i >= 0 && i < nbrMA && maArray[i] != NULL) ? maArray[i].GetHistoricalSlopeMA() : 0.0; }

    /**
     * Checks if the given price/input is strictly above all MAs in the bundle.
     */
    bool IsPriceAboveAllMAs(double price) const
    {
        for (int i = 0; i < nbrMA; ++i)
            if (maArray[i] != NULL && price <= maArray[i].GetMA()) return false;
        return nbrMA > 0;
    }

    /**
     * Checks if the given price/input is strictly below all MAs in the bundle.
     */
    bool IsPriceBelowAllMAs(double price) const
    {
        for (int i = 0; i < nbrMA; ++i)
            if (maArray[i] != NULL && price >= maArray[i].GetMA()) return false;
        return nbrMA > 0;
    }

    /**
     * Relative position of price to all mas.0=below all mas, nbrMA=above all mas
     */
    int RelativePricePosition(double price) const
    {
        int rel_pos=0;
        for (int i = 0; i < nbrMA; ++i)
        {
            if (maArray[i] != NULL && price >= maArray[i].GetMA()) ++rel_pos;
        }
        return rel_pos;
    }
 
    /**
     * Relative position of price to all mas.0=below all mas, nbrMA=above all mas
     */
    double RelativePricePositionLevel(double price) const
    {
        if(nbrMA >  0)
        {
           int rel_pos=0;
           for (int i = 0; i < nbrMA; ++i)
           {
               if (maArray[i] != NULL && price >= maArray[i].GetMA()) ++rel_pos;
           }
           return (double) rel_pos / (double) nbrMA;
        }
        return 0.0;
    }

    /**
     * Returns Close-MA level from -1 to 1.
     * 1 = value above all MAs, -1 = value below all MAs, else interpolated.
     */
    double CloseMALevel(double value) const
    {
        if (nbrMA == 0) return 0.0;
        bool aboveAll = IsPriceAboveAllMAs(value);
        bool belowAll = IsPriceBelowAllMAs(value);
        if (aboveAll) return 1.0;
        if (belowAll) return -1.0;        
        // Count how many MAs value is above; interpolate
        int above = 0;
        for (int i = 0; i < nbrMA; ++i)
            if (maArray[i] != NULL && value > maArray[i].GetMA()) above++;
        return 2.0 * (above / (double)nbrMA) - 1.0;
    }

    /**
     * Returns the average of all individual tcMA SlopeLevel values.
     */
    double SlopeLevel() const
    {
        if (nbrMA == 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < nbrMA; ++i)
        {
            if (maArray[i] != NULL) sum += maArray[i].SlopeLevel();
        }
        return sum / nbrMA;
    }
    
    /**
     * Dump slope.
     */
    void DumpSlopeAndBWL(void)
    {
      for (int i = 0; i < nbrMA; ++i)
          maArray[i].DumpSlope();
          
      PrintFormat("bw=%.4f hist-bw=%.4f",bw,bwHist);
    }    
    
protected:
    /**
     * Retrieves the bandwidth of the MAs in the bundle.
     *
     * @return The difference between the highest and lowest MA values.
     */
    double calculate_bandwidth()
    {
        if (nbrMA == 0) return 0;

        double maxMa = maArray[0].GetMA();
        double minMa = maArray[0].GetMA();

        for (int i = 1; i < nbrMA; ++i)
        {
            if (maArray[i].GetMA() > maxMa) maxMa = maArray[i].GetMA();
            if (maArray[i].GetMA() < minMa) minMa = maArray[i].GetMA();
        }

        return MathAbs(maxMa - minMa);
    }
    
};

//+------------------------------------------------------------------+

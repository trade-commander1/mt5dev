//+------------------------------------------------------------------+
//|                                            native_indicators.mqh |
//|                              Copyright 2021, trade-commander.com |
//|                                  https://www.trade-commander.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, trade-commander.com"
#property link      "https://www.trade-commander.com"

//---------------------------
//--native_ma
//---------------------------
class native_ma
// wrapps native ma
{
public:
	native_ma(int ma_period
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, int shift=0
				, ENUM_MA_METHOD method=MODE_SMA
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)
	{
	    _handle=INVALID_HANDLE;
	    setup(ma_period,symbol,tf,shift,method,appl_price);			
	}
	virtual ~native_ma()
	{
		if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 
	}
public:
    //! setup native ma
	bool setup(int ma_period
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, int shift=0
				, ENUM_MA_METHOD method=MODE_SMA
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)
    {
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
		_len=ma_period;
		
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		
		// release prev. handle
        if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 		
      shift_=shift;	
		// create ma handle		
		ResetLastError();
		_handle =iMA(_symbol,tf,ma_period,shift,method,appl_price);
        if(_handle == INVALID_HANDLE)
		{
		    PrintFormat("Failed to create MA handle %s len=%d period=%s.  code=%d",_symbol,ma_period,EnumToString(tf),GetLastError());
		}
		
		
		
		return (_handle!=INVALID_HANDLE) ;
    }
	//! get length
	inline int len(void) const {return _len;}
	inline int shift(void) const {return shift_;}
public:
	//!
	//! Get ma value[offset]
	//!
	double value(int offset=0,double fail=0.0)
	{
		if(_handle!=INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle,buffer_num,offset,1,val_buffer) > 0) 
				return val_buffer[0];
		}
		return fail;
	}
	//!
	//! Get ma slope
	//!
	double slope(int offset=0,int len=1,double fail=0.0)
	{
		if(_handle!=INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle,buffer_num,offset,len+1,val_buffer) > 1) 
			{
				double diff_sum=0.0;
				for(int i=0;i < len;++i)
				{
					diff_sum += val_buffer[len]-val_buffer[i];
				}
				diff_sum /= len;
				return (diff_sum / _point);
			}				
		}
		return fail;
	}

protected:
	//! ma len
	int		_len;
	//! Shift or offset
	int      shift_;
	//! ma symbol
	int		_handle;
	string	_symbol;
	double	_point;
};

//---------------------------
//--native_ma_arr
//---------------------------
class native_ma_arr
// wrapps native ma
{
public:
	native_ma_arr(string period_str="28,55"
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, int shift=0
				, ENUM_MA_METHOD method=MODE_SMA
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)								
	{
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
	    _tf=tf;
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		
	    _has_data=false;	
	    _last_ec=0;
      string lens[];
      _size=StringSplit(period_str,',',lens);
      if(_size > 0)
      {
          ArrayResize(_len,_size);
          ArrayInitialize(_len,0);
          
          ArrayResize(_handle,_size);
          ArrayInitialize(_handle,0);

          for(int i=0; i < _size; ++i)
          {
             _len[i]=(int) MathAbs(StringToInteger(lens[i]));
             _handle[i] =iMA(_symbol,tf,_len[i],shift,method,appl_price);
          }      
      }    		
	}
	virtual ~native_ma_arr()
	{
	   release();
	}
protected:
    //! release handles
    void release(void)
    {
        for(int i=0; i < _size; ++i)
        {
            if(_handle[i] != INVALID_HANDLE)
                IndicatorRelease(_handle[i]); 
        }
    }	
public:
    bool    check_for_data(void)     
    {
        if(_has_data == false)
        {
            const double fail=-1.0;
            for(int i=0; i < _size; ++i)
            {
                if(value(i,0,fail) == fail)
                    return false;
            }
            _has_data=true;
        }
        return _has_data;
    }
public:    
    //! get handle array size
    inline int size(void) const {return _size;}
    //! is idx valid
    bool idx_valid(int idx) const
    {
        if(_size > 0 && idx >= 0 && idx < _size)
            return true;
        return false;
    }
	//! get length of ma[idx]
	int len(int idx) const 
	{
	    if(idx_valid(idx) == true)
	        return _len[idx];
	    return -1;
	}
public:
   
	//!
	//! Get ma value[offset]
	//!
	double value(int idx, int offset=0,double fail=0.0)
	{
	    if(idx_valid(idx) == true && _handle[idx] != INVALID_HANDLE)
		{
			double    val_buffer[];
			int buffer_num=0;
			ResetLastError(); 
			if(CopyBuffer(_handle[idx],buffer_num,offset,1,val_buffer) < 0) 
			{
			    int ec=GetLastError();
			    if(_last_ec != ec)
			    {
			       // PrintFormat("Failed to copy data for %s %s. code %d",_symbol,EnumToString(_tf),GetLastError()); 
			        _last_ec=ec;

			    }
              //  IndicatorRelease(_handle[idx]); 
              //  _handle[idx] =iMA(_symbol,_tf,_len[idx],0,MODE_SMA,PRICE_CLOSE);			        
			    
			}
			else
			{
			    _last_ec=0;
				return val_buffer[0];
				
	        }
		}
		return fail;
	}
	//!
	//! Get ma slope
	//!
	double slope(int idx, int offset=0,int len=1,double fail=0.0,bool bpoint=true)
	{
		if(idx_valid(idx) == true && _handle[idx] != INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle[idx],buffer_num,offset,len+1,val_buffer) > 1) 
			{
				double diff_sum=0.0;
				for(int i=0;i < len;++i)
				{
					diff_sum += val_buffer[len]-val_buffer[i];
				}
				diff_sum /= len;
				if(bpoint == true)
				    return (diff_sum / _point);
				else
				    return diff_sum;
			}				
		}
		return fail;
	}

protected:
    //! array size
    int _size;
	//! ma len array
	int		_len[];
	//! ma handle array
	int		_handle[];
	//! ma symbol
	string	_symbol;
	//! timeframe
	ENUM_TIMEFRAMES _tf;
	
	double	_point;
	//! flag for intial data
	bool    _has_data;
	//! last errorcode
	int     _last_ec;
};

//---------------------------
//--native_stddev
//---------------------------
class native_stddev
// wrapps native ma
{
public:
	native_stddev(int ma_period
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, int shift=0
				, ENUM_MA_METHOD method=MODE_SMA
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)
	{
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
		_len=ma_period;
		
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		// create ma handle
		ResetLastError();
		_handle =iStdDev(_symbol,tf,ma_period,shift,method,appl_price);
		if(_handle == INVALID_HANDLE)
		{
		    PrintFormat("#error: failed to create std. dev. ec=%d",GetLastError());
		}
	}
	virtual ~native_stddev()
	{
		if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 


	}
public:
	//! get length
	inline int len(void) const {return _len;}
public:
	//!
	//! Get ma value[offset]
	//!
	double value(int offset=0,double fail=0.0)
	{
		if(_handle!=INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle,buffer_num,offset,1,val_buffer) > 0) 
				return val_buffer[0];
		}
		return fail;
	}
	//!
	//! Get ma slope
	//!
	double slope(int offset=0,int len=1,double fail=0.0)
	{
		if(_handle!=INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle,buffer_num,offset,len+1,val_buffer) > 1) 
			{
				double diff_sum=0.0;
				for(int i=0;i < len;++i)
				{
					diff_sum += val_buffer[len]-val_buffer[i];
				}
				diff_sum /= len;
				return (diff_sum / _point);
			}				
		}
		return fail;
	}

protected:
	//! ma len
	int		_len;
	//! ma symbol
	int		_handle;
	string	_symbol;
	double	_point;
};

//---------------------------
//--native_macd
//---------------------------
class native_macd
// wrapps native ma
{
public:
	native_macd( int                 slow_ema_period
   ,int                 fast_ema_period=-1
   ,int                 signal_period=-1
	
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, int shift=0
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)
	{
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
		
		_slow=slow_ema_period;
    	_fast=(fast_ema_period > 0 ?  fast_ema_period : slow_ema_period / 6);
    	
    	_signal=(signal_period > 0 ? signal_period : (_slow + _fast) / 2);

		
		
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		// create ma handle
		_handle =iMACD( _symbol,tf,_fast,_slow,_signal,appl_price);
        if(_handle == INVALID_HANDLE)
		{
		    PrintFormat("Failed to create MACD handle %s slow-len=%d fast-len=%d period=%s.  code=%d",_symbol,slow_ema_period,fast_ema_period,EnumToString(tf),GetLastError());
		}						
	}
	virtual ~native_macd()
	{
		if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 


	}
public:
	//! get length
	inline int fast(void) const {return _fast;}
	inline int slow(void) const {return _slow;}
	inline int signal(void) const {return _signal;}
public:
	//!
	//! Get ma value[offset]
	//!
	double value(int offset=0,double fail=0.0)
	{
		if(_handle!=INVALID_HANDLE) 
		{
			double    val_buffer[];
			int buffer_num=0;
			if(CopyBuffer(_handle,buffer_num,offset,1,val_buffer) > 0) 
				return val_buffer[0];
		}
		return fail;
	}
	

protected:
	//! macd lens
	int		_fast;
	int		_slow;
	int		_signal;
	//! ma symbol
	int		_handle;
	string	_symbol;
	double	_point;
};

//---------------------------
//--native_rsi
//---------------------------
class native_rsi
// wrapps native ma
{
public:
	native_rsi( int  ma_len
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT
				, ENUM_APPLIED_PRICE appl_price=PRICE_CLOSE)
	{
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
		
    	_ma_len=ma_len;
				
	    _applied_price=(int) appl_price;
	    _tf=tf;

			
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		_handle =iRSI( _symbol,_tf,_ma_len,_applied_price);		    		
	}
	virtual ~native_rsi()
	{
		if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 


	}
protected:
    //! creates indicator on request
    bool    create_inidicator(bool force=false)	
    {
      if(_handle != INVALID_HANDLE && force == true) 
      {
         IndicatorRelease(_handle); 
         _handle = INVALID_HANDLE;
      }
      if(_handle == INVALID_HANDLE)
      {
   		ResetLastError();
   		_handle =iRSI( _symbol,_tf,_ma_len,_applied_price);
   		if(_handle == INVALID_HANDLE)
   		{
   		    PrintFormat("Failed to create RSI handle %s len=%d period=%s.  code=%d",_symbol,_ma_len,EnumToString(_tf),GetLastError());
   		    return false;
   		}
         
      }
      return true;
    }
public:

	//! get length
	inline int ma_len(void) const {return _ma_len;}
	
public:
	//!
	//! Get ma value[offset]
	//!
	double value(int offset=0,double fail=0.0,bool force_recreate_indicator=false)
	{
		if(create_inidicator(force_recreate_indicator) == true) 
		{
			double    val_buffer[];
			int buffer_num=0;
			ResetLastError();
			int count=CopyBuffer(_handle,buffer_num,offset,1,val_buffer);
			if(count <= 0)
			{
			   
			  //PrintFormat("#error: can't get rsi. code=%d", GetLastError());
			   //IndicatorRelease(_handle); 
			   //_handle = INVALID_HANDLE;
				//return val_buffer[0];
				count=-1;
	        }
	        else
	            return val_buffer[0];
		}
		return fail;
	}
	

protected:
	//! macd lens
	int		_ma_len;	
	int		_handle;
	string	_symbol;
	double	_point;
	int     _applied_price;
	ENUM_TIMEFRAMES _tf;
}; // native_rsi



//---------------------------
//--native_atr
//---------------------------
class native_atr
// wrapps native ma
{
public:
	native_atr( int  ma_len
				, string symbol=""
				, ENUM_TIMEFRAMES tf=PERIOD_CURRENT)
	{
		if(StringLen(symbol) == 0)
			_symbol=_Symbol;
	    else
	        _symbol=symbol;
		
    	_ma_len=ma_len;
				
	    _tf=tf;

			
		_point=SymbolInfoDouble(_symbol,SYMBOL_POINT);
		_handle =iATR( _symbol,_tf,_ma_len);		    		
	}
	virtual ~native_atr()
	{
		if(_handle!=INVALID_HANDLE) 
      		IndicatorRelease(_handle); 


	}
protected:
    //! creates indicator on request
    bool    create_inidicator(bool force=false)	
    {
      if(_handle != INVALID_HANDLE && force == true) 
      {
         IndicatorRelease(_handle); 
         _handle = INVALID_HANDLE;
      }
      if(_handle == INVALID_HANDLE)
      {
   		ResetLastError();
   		_handle =iATR( _symbol,_tf,_ma_len);
   		if(_handle == INVALID_HANDLE)
   		{
   		    PrintFormat("Failed to create ATR handle %s len=%d period=%s.  code=%d",_symbol,_ma_len,EnumToString(_tf),GetLastError());
   		    return false;
   		}
         
      }
      return true;
    }
public:

	//! get length
	inline int ma_len(void) const {return _ma_len;}
	
public:
	//!
	//! Get ma value[offset]
	//!
	double value(int offset=0,double fail=0.0,bool force_recreate_indicator=false)
	{
		if(create_inidicator(force_recreate_indicator) == true) 
		{
			double    val_buffer[];
			int buffer_num=0;
			ResetLastError();
			int count=CopyBuffer(_handle,buffer_num,offset,1,val_buffer);
			if(count <= 0)
			{
			   
			  //PrintFormat("#error: can't get rsi. code=%d", GetLastError());
			   //IndicatorRelease(_handle); 
			   //_handle = INVALID_HANDLE;
				//return val_buffer[0];
				count=-1;
	        }
	        else
	            return val_buffer[0];
		}
		return fail;
	}
	

protected:
	//! macd lens
	int		_ma_len;	
	int		_handle;
	string	_symbol;
	double	_point;
	ENUM_TIMEFRAMES _tf;
};


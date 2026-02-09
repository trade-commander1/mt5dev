//+------------------------------------------------------------------+
//|                                                       macros.mqh |
//|                             Copyright 2014, trade-commander.com. |
//|                                   http://www.trade-commander.com |
//+------------------------------------------------------------------+

/// +----------------------------------------------------------------+
/// some frequently used macros
/// +----------------------------------------------------------------+

#property copyright "Copyright 2014, trade-commander.com."
#property link      "http://www.trade-commander.com"



#define RGBToClr(r,g,b)    ((uchar(b)<<16)|(uchar(g)<<8)|uchar(r))

/// [[ +---------------- keycodes --------------------+

#define VK_RETURN         13

#define VK_LBUTTON         0x01
#define VK_RBUTTON         0x02
#define VK_CANCEL         0x03
#define VK_MBUTTON         0x04
#define VK_XBUTTON1         0x05 // Middle mouse button (three-button mouse)
#define VK_XBUTTON2         0x06
/// defuned             0x07
#define VK_BACK         0x08
#define VK_TAB         0x09

#define KEY_NUMPAD_5       12
#define KEY_STR_LEFT       17
#define KEY_LEFT           37
#define KEY_UP             38
#define KEY_RIGHT          39
#define KEY_DOWN           40
#define KEY_INSERT         45
#define KEY_REMOVE         46
#define KEY_NUMLOCK_DOWN   98
#define KEY_NUMLOCK_LEFT  100
#define KEY_NUMLOCK_RIGHT 102
#define KEY_NUMLOCK_UP    104

#define KEY_NUMLOCK_0      96
#define KEY_NUMLOCK_1      97
#define KEY_NUMLOCK_2      98
#define KEY_NUMLOCK_3      99
#define KEY_NUMLOCK_4     100
#define KEY_NUMLOCK_5     101
#define KEY_NUMLOCK_6     102
#define KEY_NUMLOCK_7     103
#define KEY_NUMLOCK_8     104
#define KEY_NUMLOCK_9     105

#define KEY_NUMLOCK_MUL     106
#define KEY_NUMLOCK_DEL     110
#define KEY_NUMLOCK_DIV     111
/// ]] keycodes

// [[ constants
#define SECTIO_AUREA 1.6180339887
// ]]
#define TC_FILE_READ_ERROR "#ERROR#"
#define TC_LONG2BOOL(x) (x > 0 ? true : false)
#define TC_BOOL2LONG(x) (x == true ? 1 : 0)
#define bool2int(x) (x == true ? 1: 0)
#define int2bool(x) (x > 0 ? true : false)
#define TC_SIGN(x) (x > 0 ? 1 : x < 0 ? -1 : 0)

#define TC_ROW_SEP "\r\n"

// [[ pointer
#define TC_SAVE_DELETE_OBJ(x)   { if(x != NULL && CheckPointer(x) != POINTER_INVALID) {delete x;}x=NULL;}
#define TC_DEL_OBJ(x)   TC_SAVE_DELETE_OBJ(x)
#define TC_SAVE_DELETE_ONLY(x)   { if(CheckPointer(x) != POINTER_INVALID) {delete x;}}
#define TC_DEL(x)       TC_SAVE_DELETE_OBJ(x)
// ]] pointer



//--------------------------
//--MACROS
//--------------------------

#define TC_DEBUG_LINE PrintFormat("debug <%s::%s line %d>",__FILE__,__FUNCTION__,__LINE__)
#define TC_DEBUG_LINE_C(x) PrintFormat("debug <%s::%s line %d> %s",__FILE__,__FUNCTION__,__LINE__,x)

//#define TC_SAVE_DELETE_OBJ(x)   { if(CheckPointer(x) == POINTER_DYNAMIC) {delete x;x=NULL;}}
//#define TC_SAVE_DELETE_OBJ(x)   { if(x != NULL && CheckPointer(x) == POINTER_DYNAMIC) {delete x;x=NULL;}}
#define TC_POINTER_VALID(x)     (CheckPointer(x) != POINTER_INVALID ? true : false)
#define TC_ASSERT(x)     if(CheckPointer(x) == POINTER_INVALID) MessageBox(StringFormat("%s:%d",__FILE__,__LINE__),"Assertion failed")
#define PTHIS		GetPointer(this)
#define CAST_THIS(to_type)		(to_type*)GetPointer(this)

#define TC_AS_POINTER(x)     GetPointer(x)

#define TC_IGNORE_STRING "#"

#define FN_RISK_MODULE_FOLDER tc_default_company
// [[ signals
#define TC_SIGNAL_LONG 1
#define TC_SIGNAL_SHORT -1



#define SIDE_LONG   1
#define SIDE_SHORT  -1
#define SIDE_FLAT   0


#define SIG_LONG_EXIT INT_MAX
#define SIG_LONG 1
#define SIG_BUY SIG_LONG

#define SIG_FLAT 0
#define SIG_NONE SIG_FLAT

#define SIG_SHORT -1
#define SIG_SELL SIG_SHORT
#define SIG_SHORT_EXIT -INT_MAX

#define SIG_CLOSE_ALL -3
#define SIG_EXIT -4

#define SIG_IS_FLAT(sig) (sig==SIG_FLAT || sig==SIG_SHORT_EXIT || sig==SIG_LONG_EXIT)

#define SIG_STRING(sig) (sig == SIG_FLAT ? "FLAT" : sig == SIG_EXIT ? "EXIT" : sig == SIG_BUY ? "BUY" : "SELL" )




#define TC_EXIT_LONG SIG_LONG_EXIT
#define TC_SIDE_LONG TC_SIG_LONG
#define TC_SIDE_FLAT SIDE_FLAT
#define TC_SIDE_SHORT TC_SIG_SHORT
#define TC_EXIT_SHORT SIG_SHORT_EXIT


#define TC_SIG_NONE SIDE_FLAT
#define TC_SIG_LONG SIG_LONG
#define TC_SIG_EXIT_LONG SIG_LONG_EXIT
#define TC_SIG_SHORT SIG_SHORT
#define TC_SIG_EXIT_SHORT SIG_SHORT_EXIT
#define TC_SIG_CLOSE_ALL SIG_CLOSE_ALL
// ]] signals


#define IsTesting	 (MQLInfoInteger(MQL_TESTER) > 0 ? true : false)
#define is_testing	 IsTesting
#define AccountNumber AccountInfoInteger(ACCOUNT_LOGIN)
#define AccountCompany	AccountInfoString(ACCOUNT_COMPANY)
#define AccountLeverage AccountInfoInteger(ACCOUNT_LEVERAGE)
#define isHedgeAccount (ACCOUNT_MARGIN_MODE_RETAIL_HEDGING==(ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE))//    (AccountInfoInteger(ACCOUNT_HEDGE_ALLOWED) > 0 ? true : false)
#define isDemo         (ACCOUNT_TRADE_MODE_REAL != (ENUM_ACCOUNT_TRADE_MODE) AccountInfoInteger(ACCOUNT_TRADE_MODE))

#define AccountBalance			AccountInfoDouble(ACCOUNT_BALANCE)
#define AccountEquity			AccountInfoDouble(ACCOUNT_EQUITY)
#define AccountProfit			AccountInfoDouble(ACCOUNT_PROFIT)
#define AccountMarginLevel	   AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)
#define AccountMargin	      AccountInfoDouble(ACCOUNT_MARGIN)
#define ACC_LOGIN	            AccountInfoInteger(ACCOUNT_LOGIN)

#define DepositLoad	         (AccountEquity > 0.0 ? AccountMargin / AccountEquity : 0.0)
#define DepositLoadPct	      (AccountEquity > 0.0 ? AccountMargin / AccountEquity * 100.0 : 0.0)

#define Language              TerminalInfoString(TERMINAL_LANGUAGE)  

//#define SymbolProfit(symbol)	SymbolInfoDouble(ACCOUNT_PROFIT)
#define	bar_number				Bars(_Symbol,_Period)
#define	tcBars				    bar_number
#define	tc_bars				    tcBars
#define tc_max_bars             SeriesInfoInteger(_Symbol,PERIOD_CURRENT,SERIES_BARS_COUNT)

#define tcOpen(shift)             tc_market_info::open(shift)
#define tcClose(shift)            tc_market_info::close(shift)
#define tcHigh(shift)             tc_market_info::high(shift)
#define tcLow(shift)              tc_market_info::low(shift)
#define tcSpread(shift)              tc_market_info::spread(shift)


#define tcBar(shift)              tc_market_info::bar(shift)
#define tcBarPts(shift)           tc_market_info::bar_pts(shift)

#define tcTime(shift)             tc_market_info::time(shift)


#define tcAsk                     SymbolInfoDouble(Symbol(),SYMBOL_ASK)//tc_market_info::ask()
#define tcBid                     SymbolInfoDouble(Symbol(),SYMBOL_BID)//tc_market_info::bid()

#define tcsAsk(xsymbol)           SymbolInfoDouble(xsymbol,SYMBOL_ASK)//tc_market_info::ask(xsymbol)
#define tcsBid(xsymbol)           SymbolInfoDouble(xsymbol,SYMBOL_BID)//tc_market_info::bid(xsymbol)



#define tc_default_company	"trade-commander"



//---------------------------------------------------
// ENUMS
//---------------------------------------------------
enum tc_data_type
{ 
   dt_open=0,
   dt_close,
   dt_high,
   dt_low,
};

/// Runtime mode of an expert advisor
enum tcen_run_mode
{    
   ensd_backtest=0,        // Backtest
   ensd_user=1,            // User
   ensd_auto=2,            // Auto (Expert runs unattanded)
   ensd_signal_strength=3, // Only signal strength important
};

enum tc_stop_mode
{    
   
   stop_md_ma=0,      // stop by moving average penetration
   stop_md_hl,      // stop by high_low penetration
   stop_md_break_even,  // stop to break even
   stop_md_freeze,  // stop not changed dynamically
   stop_md_off,   // stop mode off
};


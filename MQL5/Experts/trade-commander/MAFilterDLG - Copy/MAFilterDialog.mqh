//+------------------------------------------------------------------+
//| MAFilterDialog.mqh                                                |
//| MA Filter Strategy Dialog — Controls-based UI                     |
//+------------------------------------------------------------------+
#include <Controls\Dialog.mqh>
#include <Controls\Button.mqh>
#include <Controls\Edit.mqh>
#include <Controls\RadioButton.mqh>
#include <trade-commander\moving_average.mqh>
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//--- Layout
#define INDENT_LEFT         11
#define INDENT_TOP          11
#define CONTROLS_GAP_X      5
#define CONTROLS_GAP_Y      5
#define BUTTON_WIDTH        80
#define BUTTON_HEIGHT       22
#define EDIT_WIDTH          60
#define EDIT_HEIGHT         20
#define RADIO_WIDTH         20   // Radio button width (circle only, left of edit)

//+------------------------------------------------------------------+
//| CMAFilterDialog                                                   |
//+------------------------------------------------------------------+
class CMAFilterDialog : public CAppDialog
{
private:
   // Row 1
   CButton        m_btnIgnore;
   CButton        m_btnManual;
   // Row 2
   CButton        m_btnOpenLong;
   CButton        m_btnLongLMT;
   CButton        m_btnOpenShort;
   CButton        m_btnShortLMT;
   // Row 3 — Radio left of each volume edit
   CRadioButton   m_radioVol[4];
   CEdit          m_editVol[4];
   int            m_radioVolIndex;
   // Row 4
   CButton        m_btnCancel;
   CButton        m_btnCancelAll;
   CButton        m_btnClose;
   CButton        m_btnCloseAll;

   // MA state — tcMABundle replaces iMA handles
   tcMABundle     m_maBundle;
   int            m_maPeriods[];
   int            m_nbrMa;
   bool           m_maBundleSynced;

   // Dialog state
   int            m_ignoreState;
   int            m_autoCount;
   bool           m_blinkLong;
   bool           m_blinkShort;
   bool           m_positionFromDialog;
   datetime       m_lastSignalBar;
   bool           m_blinkOn;
   double         m_volumes[4];
   double         m_equityHigh;

   CTrade         m_trade;
   CPositionInfo  m_posInfo;
   COrderInfo     m_orderInfo;

   // Helpers
   void           GenerateMAPeriods(int minLen, int maxLen, int count, int &periods[]);
   int            GetPositionDirection();
   bool           IsEquityDrawDownOK();
   double         GetSelectedVolume();
   bool           OpenPosition(int direction);
   bool           OpenLimitOrder(int direction);
   bool           CloseOurPositions();
   bool           CloseAllPositions();
   bool           CancelOurOrders();
   bool           CancelAllOrders();
   void           StopBlink();
   void           UpdateOpenButtons();
   void           UpdateIgnoreLabel();
   void           UpdateManualLabel();
   void           ApplyButtonColors();

   // Create methods
   bool           CreateRow1();
   bool           CreateRow2();
   bool           CreateRow3();
   bool           CreateRow4();

public:
                     CMAFilterDialog(void);
                    ~CMAFilterDialog(void);

   virtual bool      Create(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2);
   virtual bool      OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);

   // Called from EA OnTick
   void              OnTickUpdate();
   // Called from EA OnTimer for blink effect
   void              OnBlinkTimer();

protected:
   // Event handlers
   void              OnClickIgnore();
   void              OnClickManual();
   void              OnClickOpenLong();
   void              OnClickLongLMT();
   void              OnClickOpenShort();
   void              OnClickShortLMT();
   bool              OnChangeRadioVol(int index);
   void              OnClickCancel();
   void              OnClickCancelAll();
   void              OnClickClose();
   void              OnClickCloseAll();

   int               CheckSignal();
   bool              InitMA();
};

//+------------------------------------------------------------------+
//| Event map                                                         |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CMAFilterDialog)
   ON_EVENT(ON_CLICK, m_btnIgnore, OnClickIgnore)
   ON_EVENT(ON_CLICK, m_btnManual, OnClickManual)
   ON_EVENT(ON_CLICK, m_btnOpenLong, OnClickOpenLong)
   ON_EVENT(ON_CLICK, m_btnLongLMT, OnClickLongLMT)
   ON_EVENT(ON_CLICK, m_btnOpenShort, OnClickOpenShort)
   ON_EVENT(ON_CLICK, m_btnShortLMT, OnClickShortLMT)
   ON_INDEXED_EVENT(ON_CHANGE, m_radioVol, OnChangeRadioVol)
   ON_EVENT(ON_CLICK, m_btnCancel, OnClickCancel)
   ON_EVENT(ON_CLICK, m_btnCancelAll, OnClickCancelAll)
   ON_EVENT(ON_CLICK, m_btnClose, OnClickClose)
   ON_EVENT(ON_CLICK, m_btnCloseAll, OnClickCloseAll)
EVENT_MAP_END(CAppDialog)

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CMAFilterDialog::CMAFilterDialog(void) : m_nbrMa(0), m_ignoreState(0), m_autoCount(0), m_blinkLong(false), m_blinkShort(false),
   m_positionFromDialog(false), m_lastSignalBar(0), m_blinkOn(false), m_equityHigh(0), m_radioVolIndex(0), m_maBundleSynced(false)
{
   for(int i = 0; i < 4; i++) m_volumes[i] = 0;
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CMAFilterDialog::~CMAFilterDialog(void)
{
}

//+------------------------------------------------------------------+
//| Create                                                            |
//+------------------------------------------------------------------+
bool CMAFilterDialog::Create(const long chart, const string name, const int subwin, const int x1, const int y1, const int x2, const int y2)
{
   if(!CAppDialog::Create(chart, name, subwin, x1, y1, x2, y2))
      return false;
   if(!CreateRow1()) return false;
   if(!CreateRow2()) return false;
   if(!CreateRow3()) return false;
   if(!CreateRow4()) return false;
   if(!InitMA()) return false;
   m_trade.SetExpertMagicNumber(MagicNumber);
   m_trade.SetDeviationInPoints(10);
   long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      m_trade.SetTypeFilling(ORDER_FILLING_IOC);
   else if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      m_trade.SetTypeFilling(ORDER_FILLING_FOK);
   else
      m_trade.SetTypeFilling(ORDER_FILLING_RETURN);
   return true;
}

//+------------------------------------------------------------------+
//| OnBlinkTimer — called from EA OnTimer for blink effect            |
//+------------------------------------------------------------------+
void CMAFilterDialog::OnBlinkTimer()
{
   m_blinkOn = !m_blinkOn;
   UpdateOpenButtons();
}

//+------------------------------------------------------------------+
//| OnTickUpdate — called from EA OnTick for signal processing        |
//+------------------------------------------------------------------+
void CMAFilterDialog::OnTickUpdate()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   static datetime lastBarTime = 0;
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   m_positionFromDialog = (GetPositionDirection() != 0);
   int signal = CheckSignal();
   if(signal == 0) return;
   if(m_positionFromDialog) return;
   if(m_ignoreState == 1 && signal > 0) return;
   if(m_ignoreState == 2 && signal < 0) return;
   if(m_ignoreState == 3) return;
   if(m_lastSignalBar == currentBarTime && !m_blinkLong && !m_blinkShort) return;

   m_lastSignalBar = currentBarTime;
   if(m_autoCount > 0)
   {
      OpenPosition(signal > 0 ? 1 : -1);
      m_autoCount--;
      UpdateManualLabel();
   }
   else
   {
      if(signal > 0) m_blinkLong = true;
      else m_blinkShort = true;
      UpdateOpenButtons();
   }
}

//+------------------------------------------------------------------+
//| Create Row 1 — Ignore, Manual                                      |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow1()
{
   int x1 = INDENT_LEFT;
   int y1 = INDENT_TOP;
   int x2 = x1 + BUTTON_WIDTH;
   int y2 = y1 + BUTTON_HEIGHT;

   if(!m_btnIgnore.Create(m_chart_id, m_name + "Ignore", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnIgnore.Text("All");
   if(!Add(m_btnIgnore)) return false;

   x1 = x2 + CONTROLS_GAP_X;
   x2 = x1 + BUTTON_WIDTH;
   if(!m_btnManual.Create(m_chart_id, m_name + "Manual", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnManual.Text("Manual");
   if(!Add(m_btnManual)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 2 — Open Long, LMT, Open Short, LMT                    |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow2()
{
   int x1 = INDENT_LEFT;
   int y1 = INDENT_TOP + BUTTON_HEIGHT + CONTROLS_GAP_Y;
   int x2 = x1 + BUTTON_WIDTH;
   int y2 = y1 + BUTTON_HEIGHT;

   if(!m_btnOpenLong.Create(m_chart_id, m_name + "OpenLong", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnOpenLong.Text("Open Long");
   if(!Add(m_btnOpenLong)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnLongLMT.Create(m_chart_id, m_name + "LongLMT", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnLongLMT.Text("LMT");
   if(!Add(m_btnLongLMT)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnOpenShort.Create(m_chart_id, m_name + "OpenShort", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnOpenShort.Text("Open Short");
   if(!Add(m_btnOpenShort)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnShortLMT.Create(m_chart_id, m_name + "ShortLMT", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnShortLMT.Text("LMT");
   if(!Add(m_btnShortLMT)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 3 — Radio left of each volume edit (no separate area)  |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow3()
{
   int y1 = INDENT_TOP + 2 * (BUTTON_HEIGHT + CONTROLS_GAP_Y);
   int pairWidth = RADIO_WIDTH + 2 + EDIT_WIDTH;   // [Radio][Edit] per pair
   double volMin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   m_volumes[0] = MathMax(volMin, MathFloor(BaseVolume / volStep) * volStep);
   m_volumes[1] = MathMax(volMin, MathFloor(BaseVolume * 2 / volStep) * volStep);
   m_volumes[2] = MathMax(volMin, MathFloor(BaseVolume * 4 / volStep) * volStep);
   m_volumes[3] = MathMax(volMin, MathFloor(BaseVolume * 8 / volStep) * volStep);

   for(int i = 0; i < 4; i++)
   {
      int baseX = INDENT_LEFT + i * (pairWidth + CONTROLS_GAP_X);
      // Radio button left of edit
      int rx1 = baseX;
      int rx2 = rx1 + RADIO_WIDTH;
      if(!m_radioVol[i].Create(m_chart_id, m_name + "RadioVol" + IntegerToString(i), m_subwin, rx1, y1, rx2, y1 + EDIT_HEIGHT))
         return false;
      m_radioVol[i].Text("");   // No label, just the circle
      m_radioVol[i].State(i == 0);   // First selected by default
      if(!Add(m_radioVol[i])) return false;

      // Edit field right of radio
      int ex1 = rx2 + 2;
      int ex2 = ex1 + EDIT_WIDTH;
      if(!m_editVol[i].Create(m_chart_id, m_name + "Vol" + IntegerToString(i), m_subwin, ex1, y1, ex2, y1 + EDIT_HEIGHT))
         return false;
      m_editVol[i].Text(DoubleToString(m_volumes[i], 2));
      if(!Add(m_editVol[i])) return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 4 — Cancel, Cancel All, Close, Close All               |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow4()
{
   int x1 = INDENT_LEFT;
   int y1 = INDENT_TOP + 2 * (BUTTON_HEIGHT + CONTROLS_GAP_Y) + EDIT_HEIGHT + CONTROLS_GAP_Y;
   int x2 = x1 + BUTTON_WIDTH;
   int y2 = y1 + BUTTON_HEIGHT;

   if(!m_btnCancel.Create(m_chart_id, m_name + "Cancel", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnCancel.Text("Cancel");
   if(!Add(m_btnCancel)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnCancelAll.Create(m_chart_id, m_name + "CancelAll", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnCancelAll.Text("Cancel All");
   if(!Add(m_btnCancelAll)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnClose.Create(m_chart_id, m_name + "CloseX", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnClose.Text("Close");
   if(!Add(m_btnClose)) return false;

   x1 = x2 + CONTROLS_GAP_X; x2 = x1 + BUTTON_WIDTH;
   if(!m_btnCloseAll.Create(m_chart_id, m_name + "CloseXAll", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnCloseAll.Text("Close All");
   if(!Add(m_btnCloseAll)) return false;
   ApplyButtonColors();
   return true;
}

//+------------------------------------------------------------------+
//| Event handlers                                                    |
//+------------------------------------------------------------------+
void CMAFilterDialog::OnClickIgnore()
{
   m_ignoreState = (m_ignoreState + 1) % 4;
   UpdateIgnoreLabel();
}

void CMAFilterDialog::OnClickManual()
{
   m_autoCount = (m_autoCount + 1) % 4;
   UpdateManualLabel();
}

void CMAFilterDialog::OnClickOpenLong()
{
   if(!m_blinkLong) return;
   for(int i = 0; i < 4; i++) m_volumes[i] = StringToDouble(m_editVol[i].Text());
   OpenPosition(1);
   StopBlink();
}

void CMAFilterDialog::OnClickLongLMT()
{
   if(!m_blinkLong) return;
   for(int i = 0; i < 4; i++) m_volumes[i] = StringToDouble(m_editVol[i].Text());
   OpenLimitOrder(1);
   StopBlink();
}

void CMAFilterDialog::OnClickOpenShort()
{
   if(!m_blinkShort) return;
   for(int i = 0; i < 4; i++) m_volumes[i] = StringToDouble(m_editVol[i].Text());
   OpenPosition(-1);
   StopBlink();
}

void CMAFilterDialog::OnClickShortLMT()
{
   if(!m_blinkShort) return;
   for(int i = 0; i < 4; i++) m_volumes[i] = StringToDouble(m_editVol[i].Text());
   OpenLimitOrder(-1);
   StopBlink();
}

bool CMAFilterDialog::OnChangeRadioVol(int index)
{
   m_radioVolIndex = index;
   for(int i = 0; i < 4; i++)
      if(i != index) m_radioVol[i].State(false);
   return true;
}

void CMAFilterDialog::OnClickCancel()
{
   CancelOurOrders();
}

void CMAFilterDialog::OnClickCancelAll()
{
   CancelAllOrders();
}

void CMAFilterDialog::OnClickClose()
{
   CloseOurPositions();
}

void CMAFilterDialog::OnClickCloseAll()
{
   CloseAllPositions();
}

//+------------------------------------------------------------------+
//| Helpers                                                           |
//+------------------------------------------------------------------+
void CMAFilterDialog::GenerateMAPeriods(int minLen, int maxLen, int count, int &periods[])
{
   ArrayResize(periods, count);
   if(count < 2) { periods[0] = minLen; return; }
   double logRatio = MathLog((double)maxLen / minLen);
   int temp[]; ArrayResize(temp, count);
   int uniqueCount = 0;
   for(int i = 0; i < count; i++)
   {
      double raw = minLen * MathExp(i / (double)(count - 1) * logRatio);
      int rounded = (int)MathMax(2, MathRound(raw));
      bool isDup = false;
      for(int j = 0; j < uniqueCount; j++)
         if(temp[j] == rounded) { isDup = true; break; }
      if(!isDup) temp[uniqueCount++] = rounded;
   }
   ArrayResize(periods, uniqueCount);
   for(int i = 0; i < uniqueCount; i++) periods[i] = temp[i];
}

int CMAFilterDialog::GetPositionDirection()
{
   int dir = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol) continue;
      if(m_posInfo.Magic() != MagicNumber) continue;
      if(m_posInfo.PositionType() == POSITION_TYPE_BUY) dir += 1;
      else if(m_posInfo.PositionType() == POSITION_TYPE_SELL) dir -= 1;
   }
   return dir;
}

bool CMAFilterDialog::IsEquityDrawDownOK()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(m_equityHigh <= 0) m_equityHigh = equity;
   if(equity > m_equityHigh) m_equityHigh = equity;
   double threshold = m_equityHigh * (1.0 - EquityDrawDownThresholdPCT / 100.0);
   return equity >= threshold;
}

double CMAFilterDialog::GetSelectedVolume()
{
   int idx = m_radioVolIndex;
   if(idx < 0 || idx > 3) idx = 0;
   // Use volume from edit (in case user changed it)
   m_volumes[idx] = StringToDouble(m_editVol[idx].Text());
   double vol = m_volumes[idx];
   double volMin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volMax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   vol = MathMax(volMin, MathMin(volMax, vol));
   return MathFloor(vol / volStep) * volStep;
}

bool CMAFilterDialog::OpenPosition(int direction)
{
   if(!IsEquityDrawDownOK()) { Print("Blocked: equity drawdown"); return false; }
   double vol = GetSelectedVolume();
   double price = (direction > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string comment = (direction > 0) ? "MAFilterDLG Long" : "MAFilterDLG Short";
   bool ok = (direction > 0)
      ? m_trade.Buy(vol, _Symbol, price, 0, 0, comment)
      : m_trade.Sell(vol, _Symbol, price, 0, 0, comment);
   if(ok) { m_positionFromDialog = true; Print((direction > 0 ? "LONG" : "SHORT"), " opened"); }
   return ok;
}

bool CMAFilterDialog::OpenLimitOrder(int direction)
{
   if(!IsEquityDrawDownOK()) { Print("Blocked: equity drawdown"); return false; }
   double vol = GetSelectedVolume();
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double offset = 10 * point;
   double price = (direction > 0)
      ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) - offset
      : SymbolInfoDouble(_Symbol, SYMBOL_BID) + offset;
   price = NormalizeDouble(price, digits);
   string comment = (direction > 0) ? "MAFilterDLG Long LMT" : "MAFilterDLG Short LMT";
   bool ok = (direction > 0)
      ? m_trade.BuyLimit(vol, price, _Symbol, 0, 0, ORDER_TIME_GTC, 0, comment)
      : m_trade.SellLimit(vol, price, _Symbol, 0, 0, ORDER_TIME_GTC, 0, comment);
   if(ok) Print((direction > 0 ? "BUY LIMIT" : "SELL LIMIT"), " at ", price);
   return ok;
}

bool CMAFilterDialog::CloseOurPositions()
{
   bool ok = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol || m_posInfo.Magic() != MagicNumber) continue;
      if(!m_trade.PositionClose(m_posInfo.Ticket())) ok = false;
   }
   return ok;
}

bool CMAFilterDialog::CloseAllPositions()
{
   bool ok = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol) continue;
      if(!m_trade.PositionClose(m_posInfo.Ticket())) ok = false;
   }
   return ok;
}

bool CMAFilterDialog::CancelOurOrders()
{
   bool ok = true;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!m_orderInfo.SelectByIndex(i)) continue;
      if(m_orderInfo.Symbol() != _Symbol || m_orderInfo.Magic() != MagicNumber) continue;
      if(!m_trade.OrderDelete(m_orderInfo.Ticket())) ok = false;
   }
   return ok;
}

bool CMAFilterDialog::CancelAllOrders()
{
   bool ok = true;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!m_orderInfo.SelectByIndex(i)) continue;
      if(m_orderInfo.Symbol() != _Symbol) continue;
      if(!m_trade.OrderDelete(m_orderInfo.Ticket())) ok = false;
   }
   return ok;
}

void CMAFilterDialog::StopBlink()
{
   m_blinkLong = false;
   m_blinkShort = false;
   UpdateOpenButtons();
}

void CMAFilterDialog::UpdateOpenButtons()
{
   color cLong  = (m_blinkLong && m_blinkOn) ? clrYellow : clrDarkGreen;
   color cShort = (m_blinkShort && m_blinkOn) ? clrYellow : clrDarkRed;
   m_btnOpenLong.ColorBackground(cLong);
   m_btnLongLMT.ColorBackground((m_blinkLong && m_blinkOn) ? clrYellow : clrForestGreen);
   m_btnOpenShort.ColorBackground(cShort);
   m_btnShortLMT.ColorBackground((m_blinkShort && m_blinkOn) ? clrYellow : clrCrimson);
}

void CMAFilterDialog::UpdateIgnoreLabel()
{
   string lbl = "All";
   if(m_ignoreState == 1) lbl = "Ignore Long";
   else if(m_ignoreState == 2) lbl = "Ignore Short";
   else if(m_ignoreState == 3) lbl = "Ignore All";
   m_btnIgnore.Text(lbl);
   ApplyButtonColors();
}

void CMAFilterDialog::UpdateManualLabel()
{
   m_btnManual.Text(m_autoCount == 0 ? "Manual" : ("Auto " + IntegerToString(m_autoCount)));
   ApplyButtonColors();
}

//+------------------------------------------------------------------+
//| ApplyButtonColors — set BK/FG per user spec                       |
//+------------------------------------------------------------------+
void CMAFilterDialog::ApplyButtonColors()
{
   color cWhite = clrWhite;
   color cBlue = clrBlue;
   color cGray = clrGray;

   // Open Long: dark green BG, white FG
   m_btnOpenLong.ColorBackground(clrDarkGreen);
   m_btnOpenLong.Color(cWhite);
   // LMT (Long): lighter green BG, white FG
   m_btnLongLMT.ColorBackground(clrForestGreen);
   m_btnLongLMT.Color(cWhite);
   // Open Short: dark red BG, white FG
   m_btnOpenShort.ColorBackground(clrDarkRed);
   m_btnOpenShort.Color(cWhite);
   // LMT (Short): lighter red BG, white FG
   m_btnShortLMT.ColorBackground(clrCrimson);
   m_btnShortLMT.Color(cWhite);

   // Cancel: yellow BG, blue FG
   m_btnCancel.ColorBackground(clrYellow);
   m_btnCancel.Color(cBlue);
   // Cancel All: yellow BG, blue FG
   m_btnCancelAll.ColorBackground(clrYellow);
   m_btnCancelAll.Color(cBlue);
   // Close: dark red BG, white FG
   m_btnClose.ColorBackground(clrDarkRed);
   m_btnClose.Color(cWhite);
   // Close All: dark red BG, white FG
   m_btnCloseAll.ColorBackground(clrDarkRed);
   m_btnCloseAll.Color(cWhite);

   // Ignore button: All=dodgerblue, Ignore Long=darkred, Ignore Short=green, Ignore All=darkgray
   if(m_ignoreState == 0)
   {
      m_btnIgnore.ColorBackground(clrDodgerBlue);
      m_btnIgnore.Color(cWhite);
   }
   else if(m_ignoreState == 1)
   {
      m_btnIgnore.ColorBackground(clrDarkRed);
      m_btnIgnore.Color(cWhite);
   }
   else if(m_ignoreState == 2)
   {
      m_btnIgnore.ColorBackground(clrGreen);
      m_btnIgnore.Color(cWhite);
   }
   else
   {
      m_btnIgnore.ColorBackground(clrDarkGray);
      m_btnIgnore.Color(cGray);
   }

   // Manual: black BG white FG; Auto 1,2,3: dodgerblue BG white FG
   if(m_autoCount == 0)
   {
      m_btnManual.ColorBackground(clrBlack);
      m_btnManual.Color(cWhite);
   }
   else
   {
      m_btnManual.ColorBackground(clrDodgerBlue);
      m_btnManual.Color(cWhite);
   }
}

//+------------------------------------------------------------------+
//| InitMA                                                            |
//+------------------------------------------------------------------+
bool CMAFilterDialog::InitMA()
{
   GenerateMAPeriods(MinLen, MaxLen, NbrMa, m_maPeriods);
   m_nbrMa = ArraySize(m_maPeriods);
   if(m_nbrMa < 2) { Print("Need at least 2 MA periods"); return false; }
   m_maBundle.setup(m_maPeriods, m_nbrMa, NH, TC_MA_SMA);
   m_maBundleSynced = false;
   return true;
}

//+------------------------------------------------------------------+
//| CheckSignal — MAFilter logic using tcMABundle                      |
//| Returns: 1 = long signal, -1 = short signal, 0 = no signal        |
//+------------------------------------------------------------------+
int CMAFilterDialog::CheckSignal()
{
   int barsNeeded = NH + 2;
   if(Bars(_Symbol, PERIOD_CURRENT) < barsNeeded) return 0;

   // Sync tcMABundle: feed bars from oldest to newest (initial sync or incremental)
   if(!m_maBundleSynced)
   {
      for(int shift = barsNeeded; shift >= 1; shift--)
      {
         double o = iOpen(_Symbol, PERIOD_CURRENT, shift);
         double h = iHigh(_Symbol, PERIOD_CURRENT, shift);
         double l = iLow(_Symbol, PERIOD_CURRENT, shift);
         double c = iClose(_Symbol, PERIOD_CURRENT, shift);
         double value = UseOHCLMean ? (o + h + l + 2.0 * c) / 5.0 : c;
         double weight = VolumeWeighted ? (double)iVolume(_Symbol, PERIOD_CURRENT, shift) : 1.0;
         m_maBundle.update(value, weight, true);
      }
      m_maBundleSynced = true;
   }
   else
   {
      // Feed just the closed bar (shift 1)
      double o = iOpen(_Symbol, PERIOD_CURRENT, 1);
      double h = iHigh(_Symbol, PERIOD_CURRENT, 1);
      double l = iLow(_Symbol, PERIOD_CURRENT, 1);
      double c = iClose(_Symbol, PERIOD_CURRENT, 1);
      double value = UseOHCLMean ? (o + h + l + 2.0 * c) / 5.0 : c;
      double weight = VolumeWeighted ? (double)iVolume(_Symbol, PERIOD_CURRENT, 1) : 1.0;
      m_maBundle.update(value, weight, true);
   }

   double closePrice = iClose(_Symbol, PERIOD_CURRENT, 1);
   double value = UseOHCLMean ? (iOpen(_Symbol, PERIOD_CURRENT, 1) + iHigh(_Symbol, PERIOD_CURRENT, 1) + iLow(_Symbol, PERIOD_CURRENT, 1) + 2.0 * closePrice) / 5.0 : closePrice;

   // Price vs MAs: input above all MAs (bullish) or below all MAs (bearish)
   bool closeAboveAll = m_maBundle.IsPriceAboveAllMAs(value);
   bool closeBelowAll = m_maBundle.IsPriceBelowAllMAs(value);
   if(closeAboveAll == false && closeBelowAll == false)
      return 0;
   // Slope direction: all MAs rising (long) or all falling (short)
   bool allSlopesPos = true, allSlopesNeg = true;
   for(int k = 0; k < m_nbrMa; k++)
   {
      double sl = m_maBundle.GetSlope(k);
      if(sl <= 0) allSlopesPos = false;
      if(sl >= 0) allSlopesNeg = false;
   }

   // Slope strength: each MA slope must exceed MinSlopeFactor * avg historical slope
   bool slopesStrong = true;
   for(int k = 0; k < m_nbrMa; k++)
   {
      double avgAbsSlope = MathMax(1e-10, MathAbs(m_maBundle.GetHistoricalSlopeMA(k)));
      if(MathAbs(m_maBundle.GetSlope(k)) < MinSlopeFactor * avgAbsSlope) { slopesStrong = false; break; }
   }

   // Laminar level and bandwidth from tcMABundle
   double laminarLevel = m_maBundle.laminar_level(StrictSlopeOrder);
   bool laminarLong  = laminarLevel >= MinLaminarLevel;
   bool laminarShort = laminarLevel <= -MinLaminarLevel;

   double bwCurrent = m_maBundle.bandwidth();
   tcMA* histBw = m_maBundle.GetHistoricalBandwidth();
   double bwSma = (histBw != NULL) ? histBw.GetMA() : bwCurrent;
   double bwFactor = (bwSma > 1e-10) ? bwCurrent / bwSma : 1.0;
   bool bwOk = bwFactor <= MaxBandwidthFactor;

   int pos = GetPositionDirection();

   if(pos != 0)
   {
      bool exitSignal = false;
      if(ExitOption == 0)
      {
         double slowestMA = m_maBundle.GetMAValue(m_nbrMa - 1);
         double stdDevCurrent = m_maBundle.GetStdDev(m_nbrMa - 1);
         double band = StdDevFactor * stdDevCurrent;
         if(pos > 0 && closePrice < slowestMA - band) exitSignal = true;
         if(pos < 0 && closePrice > slowestMA + band) exitSignal = true;
      }
      else
      {
         if(pos > 0 && m_maBundle.GetSlope(0) < 0) exitSignal = true;
         if(pos < 0 && m_maBundle.GetSlope(0) > 0) exitSignal = true;
      }
      if(exitSignal) CloseOurPositions();
      return 0;
   }

   bool longEntry  = closeAboveAll && allSlopesPos && slopesStrong && laminarLong && bwOk;
   bool shortEntry = closeBelowAll && allSlopesNeg && slopesStrong && laminarShort && bwOk;
   if(longEntry) return 1;
   if(shortEntry) return -1;
   return 0;
}
//+------------------------------------------------------------------+

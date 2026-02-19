//+------------------------------------------------------------------+
//| MAFilterDialog.mqh                                                |
//| MA Filter Strategy Dialog — Controls-based UI                     |
//+------------------------------------------------------------------+
#include <tcControls\Dialog.mqh>
#include <tcControls\Button.mqh>
#include <tcControls\Edit.mqh>
#include <tcControls\RadioButton.mqh>
#include <tcControls\Label.mqh>
#include <tcControls\Panel.mqh>
#include <trade-commander\moving_average.mqh>
#include <trade-commander\hash.mqh>

#include <tcTrade\Trade.mqh>
#include <tcTrade\PositionInfo.mqh>
#include <tcTrade\OrderInfo.mqh>

//--- Layout
#define INDENT_LEFT         11
#define INDENT_TOP          11
#define CONTROLS_GAP_X      5
#define CONTROLS_GAP_Y      5
#define ROW_EXTRA           4    // Extra spacing between rows
#define GROUP_EXTRA         5    // Extra spacing above/below grouped rows
#define BUTTON_WIDTH        82   // Align LMT/Close All right edge with last volume field
#define BUTTON_HEIGHT       22
#define EDIT_WIDTH          60
#define EDIT_HEIGHT         20
#define RADIO_WIDTH         20   // Radio button width (circle only, left of edit)
#define ROW2_Y_OFFSET       (INDENT_TOP + BUTTON_HEIGHT + CONTROLS_GAP_Y + GROUP_EXTRA)
#define ROW3_Y_OFFSET       (ROW2_Y_OFFSET + BUTTON_HEIGHT + CONTROLS_GAP_Y)
#define ROW4_Y_OFFSET       (ROW3_Y_OFFSET + EDIT_HEIGHT + CONTROLS_GAP_Y + ROW_EXTRA)
#define ROW5_Y_OFFSET       (ROW4_Y_OFFSET + BUTTON_HEIGHT + CONTROLS_GAP_Y + GROUP_EXTRA)
#define ROW6_Y_OFFSET       (ROW5_Y_OFFSET + 18 + CONTROLS_GAP_Y + ROW_EXTRA)
#define DEPOSIT_LOAD_BAR_H  24
#define ROW7_Y_OFFSET       (ROW6_Y_OFFSET + 18 + CONTROLS_GAP_Y + GROUP_EXTRA)
#define TELEMETRY_EXTRA     10   // Extra space before telemetry (debug) block
#define ROW8_Y_OFFSET       (ROW7_Y_OFFSET + DEPOSIT_LOAD_BAR_H + CONTROLS_GAP_Y + TELEMETRY_EXTRA)
#define INFO_FONT_SIZE      (12)
#define TELEMETRY_FONT_SIZE (10)
#define LABEL_VAL_GAP       8    // Gap between label and value in Position/Profit rows
#define POS_LABEL_W         50  // Label width for Net/Long/Short (align values)
#define PROFIT_LABEL_W      55  // Label width for Sym/Sess/Profit (align values)

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
   // Row 5 — Position display (label white, value colored)
   CLabel         m_lblNetPos;
   CLabel         m_lblNetVal;
   CLabel         m_lblLongPos;
   CLabel         m_lblLongVal;
   CLabel         m_lblShortPos;
   CLabel         m_lblShortVal;
   // Row 6 — Profit display (label white, value colored)
   CLabel         m_lblSymbolProfit;
   CLabel         m_lblSymbolProfitVal;
   CLabel         m_lblSessionProfit;
   CLabel         m_lblSessionProfitVal;
   CLabel         m_lblProfit;
   CLabel         m_lblProfitVal;
   // Row 7 — Deposit Load progress bar
   CPanel         m_progressBg;
   CPanel         m_progressFill;
   CLabel         m_lblDepositLoad;
   // Row 8 — Debug levels vertical (label white, value colored)
   CLabel         m_lblCloseMALbl;
   CLabel         m_lblCloseMAVal;
   CLabel         m_lblLaminarLbl;
   CLabel         m_lblLaminarVal;
   CLabel         m_lblBandwidthLbl;
   CLabel         m_lblBandwidthVal;
   CLabel         m_lblSlopeLbl;
   CLabel         m_lblSlopeVal;
   CPanel         m_bgPanel;

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
   ulong          m_effectiveMagic;
   double         m_sessionStartEquity;
   double         m_volStep;
   int            m_volDigits;

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
   void           UpdateCancelCloseStates();
   void           UpdatePositionProfitDisplay();
   void           UpdateDepositLoadBar();
   void           UpdateDebugLevels();
   bool           HasOurPendingOrders();
   bool           HasAnyPendingOrders();
   bool           HasOurPosition();
   bool           HasAnyPosition();
   ulong          ResolveMagicNumber();

   // Create methods
   bool           CreateRow1();
   bool           CreateRow2();
   bool           CreateRow3();
   bool           CreateRow4();
   bool           CreateRow5();
   bool           CreateRow6();
   bool           CreateRow7();
   bool           CreateRow8();
   bool           CreateBackgroundPanel();

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
   m_positionFromDialog(false), m_lastSignalBar(0), m_blinkOn(false), m_equityHigh(0), m_radioVolIndex(0), m_maBundleSynced(false),
   m_effectiveMagic(0), m_sessionStartEquity(0)
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
   if(!CreateBackgroundPanel()) return false;
   if(!CreateRow1()) return false;
   if(!CreateRow2()) return false;
   if(!CreateRow3()) return false;
   if(!CreateRow4()) return false;
   if(!CreateRow5()) return false;
   if(!CreateRow6()) return false;
   if(!CreateRow7()) return false;
   if(!CreateRow8()) return false;
   if(!InitMA()) return false;
   m_effectiveMagic = ResolveMagicNumber();
   m_trade.SetExpertMagicNumber(m_effectiveMagic);
   m_trade.SetDeviationInPoints(10);
   m_sessionStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   m_volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   m_volDigits = (m_volStep > 0) ? (int)MathMax(0, -MathFloor(MathLog10(m_volStep) + 0.5)) : 2;
   Caption("MAFilterDLG Magic #" + StringFormat("%I64u", m_effectiveMagic));
   UpdateOpenButtons();
   UpdateCancelCloseStates();
   UpdatePositionProfitDisplay();
   UpdateDepositLoadBar();
   UpdateDebugLevels();
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
   m_positionFromDialog = (GetPositionDirection() != 0);
   UpdateCancelCloseStates();
   UpdatePositionProfitDisplay();
   UpdateDepositLoadBar();
   UpdateDebugLevels();

   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   static datetime lastBarTime = 0;
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

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
   m_btnIgnore.ColorBackground(clrBlack);
   m_btnIgnore.Color(clrWhite);
   if(!Add(m_btnIgnore)) return false;

   x1 = x2 + CONTROLS_GAP_X;
   x2 = x1 + BUTTON_WIDTH;
   if(!m_btnManual.Create(m_chart_id, m_name + "Manual", m_subwin, x1, y1, x2, y2))
      return false;
   m_btnManual.Text("Manual");
   m_btnManual.ColorBackground(clrBlack);
   m_btnManual.Color(clrWhite);
   if(!Add(m_btnManual)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 2 — Open Long, LMT, Open Short, LMT                    |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow2()
{
   int x1 = INDENT_LEFT;
   int y1 = ROW2_Y_OFFSET;
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
   int y1 = ROW3_Y_OFFSET;
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
      m_editVol[i].ColorBackground(clrBlack);
      m_editVol[i].Color(clrWhite);
      m_editVol[i].TextAlign(ALIGN_RIGHT);
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
   int y1 = ROW4_Y_OFFSET;
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
//| CreateBackgroundPanel — black background for dialog                |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateBackgroundPanel()
{
   if(!m_bgPanel.Create(m_chart_id, m_name + "Bg", m_subwin, 0, 0, ClientAreaWidth(), ClientAreaHeight()))
      return false;
   m_bgPanel.ColorBackground(clrBlack);
   m_bgPanel.ColorBorder(clrBlack);
   if(!Add(m_bgPanel)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 5 — Position display (Net, Long, Short)                 |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow5()
{
   int y1 = ROW5_Y_OFFSET;
   int lblH = 18;
   int w = (ClientAreaWidth() - 2 * INDENT_LEFT - 2 * CONTROLS_GAP_X) / 3;
   int valX = POS_LABEL_W + LABEL_VAL_GAP;

   if(!m_lblNetPos.Create(m_chart_id, m_name + "NetLbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + POS_LABEL_W, y1 + lblH))
      return false;
   m_lblNetPos.Text("Net:");
   m_lblNetPos.Color(clrWhite);
   m_lblNetPos.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblNetPos)) return false;
   if(!m_lblNetVal.Create(m_chart_id, m_name + "NetVal", m_subwin, INDENT_LEFT + valX, y1, INDENT_LEFT + w, y1 + lblH))
      return false;
   m_lblNetVal.Text("0");
   m_lblNetVal.Color(clrWhite);
   m_lblNetVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblNetVal)) return false;

   int x2 = INDENT_LEFT + w + CONTROLS_GAP_X;
   if(!m_lblLongPos.Create(m_chart_id, m_name + "LongLbl", m_subwin, x2, y1, x2 + POS_LABEL_W, y1 + lblH))
      return false;
   m_lblLongPos.Text("Long:");
   m_lblLongPos.Color(clrWhite);
   m_lblLongPos.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblLongPos)) return false;
   if(!m_lblLongVal.Create(m_chart_id, m_name + "LongVal", m_subwin, x2 + valX, y1, x2 + w, y1 + lblH))
      return false;
   m_lblLongVal.Text("0");
   m_lblLongVal.Color(clrWhite);
   m_lblLongVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblLongVal)) return false;

   int x3 = INDENT_LEFT + 2*w + 2*CONTROLS_GAP_X;
   if(!m_lblShortPos.Create(m_chart_id, m_name + "ShortLbl", m_subwin, x3, y1, x3 + POS_LABEL_W, y1 + lblH))
      return false;
   m_lblShortPos.Text("Short:");
   m_lblShortPos.Color(clrWhite);
   m_lblShortPos.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblShortPos)) return false;
   if(!m_lblShortVal.Create(m_chart_id, m_name + "ShortVal", m_subwin, x3 + valX, y1, ClientAreaWidth() - INDENT_LEFT, y1 + lblH))
      return false;
   m_lblShortVal.Text("0");
   m_lblShortVal.Color(clrWhite);
   m_lblShortVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblShortVal)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 6 — Profit display                                    |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow6()
{
   int y1 = ROW6_Y_OFFSET;
   int lblH = 18;
   int w = (ClientAreaWidth() - 2 * INDENT_LEFT - 2 * CONTROLS_GAP_X) / 3;
   int valX = PROFIT_LABEL_W + LABEL_VAL_GAP;

   if(!m_lblSymbolProfit.Create(m_chart_id, m_name + "SymLbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + PROFIT_LABEL_W, y1 + lblH))
      return false;
   m_lblSymbolProfit.Text("Sym:");
   m_lblSymbolProfit.Color(clrWhite);
   m_lblSymbolProfit.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblSymbolProfit)) return false;
   if(!m_lblSymbolProfitVal.Create(m_chart_id, m_name + "SymVal", m_subwin, INDENT_LEFT + valX, y1, INDENT_LEFT + w, y1 + lblH))
      return false;
   m_lblSymbolProfitVal.Text("0");
   m_lblSymbolProfitVal.Color(clrWhite);
   m_lblSymbolProfitVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblSymbolProfitVal)) return false;

   int x2 = INDENT_LEFT + w + CONTROLS_GAP_X;
   if(!m_lblSessionProfit.Create(m_chart_id, m_name + "SessLbl", m_subwin, x2, y1, x2 + PROFIT_LABEL_W, y1 + lblH))
      return false;
   m_lblSessionProfit.Text("Sess:");
   m_lblSessionProfit.Color(clrWhite);
   m_lblSessionProfit.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblSessionProfit)) return false;
   if(!m_lblSessionProfitVal.Create(m_chart_id, m_name + "SessVal", m_subwin, x2 + valX, y1, x2 + w, y1 + lblH))
      return false;
   m_lblSessionProfitVal.Text("0");
   m_lblSessionProfitVal.Color(clrWhite);
   m_lblSessionProfitVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblSessionProfitVal)) return false;

   int x3 = INDENT_LEFT + 2*w + 2*CONTROLS_GAP_X;
   if(!m_lblProfit.Create(m_chart_id, m_name + "ProfitLbl", m_subwin, x3, y1, x3 + PROFIT_LABEL_W, y1 + lblH))
      return false;
   m_lblProfit.Text("Profit:");
   m_lblProfit.Color(clrWhite);
   m_lblProfit.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblProfit)) return false;
   if(!m_lblProfitVal.Create(m_chart_id, m_name + "ProfitVal", m_subwin, x3 + valX, y1, ClientAreaWidth() - INDENT_LEFT, y1 + lblH))
      return false;
   m_lblProfitVal.Text("0");
   m_lblProfitVal.Color(clrWhite);
   m_lblProfitVal.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblProfitVal)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 7 — Deposit Load progress bar                          |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow7()
{
   int y1 = ROW7_Y_OFFSET;
   int barH = DEPOSIT_LOAD_BAR_H;
   int barW = ClientAreaWidth() - 2 * INDENT_LEFT;

   if(!m_progressBg.Create(m_chart_id, m_name + "ProgBg", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + barW, y1 + barH))
      return false;
   m_progressBg.ColorBackground(clrDarkGray);
   m_progressBg.ColorBorder(clrGray);
   if(!Add(m_progressBg)) return false;

   if(!m_progressFill.Create(m_chart_id, m_name + "ProgFill", m_subwin, INDENT_LEFT, y1, INDENT_LEFT, y1 + barH))
      return false;
   m_progressFill.ColorBackground(clrGreen);
   m_progressFill.ColorBorder(clrGreen);
   if(!Add(m_progressFill)) return false;

   if(!m_lblDepositLoad.Create(m_chart_id, m_name + "DepLoad", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + barW, y1 + barH))
      return false;
   m_lblDepositLoad.Text("Deposit Load 0 %");
   m_lblDepositLoad.Color(clrBlack);
   m_lblDepositLoad.FontSize(INFO_FONT_SIZE);
   if(!Add(m_lblDepositLoad)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Create Row 8 — Debug levels vertical (label + value per line)    |
//+------------------------------------------------------------------+
bool CMAFilterDialog::CreateRow8()
{
   int y1 = ROW8_Y_OFFSET;
   int lineH = 14;
   int labelW = 85;
   int valW = 60;

   if(!m_lblCloseMALbl.Create(m_chart_id, m_name + "CloseMALbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + labelW, y1 + lineH))
      return false;
   m_lblCloseMALbl.Text("Close-MA:");
   m_lblCloseMALbl.Color(clrWhite);
   m_lblCloseMALbl.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblCloseMALbl)) return false;
   if(!m_lblCloseMAVal.Create(m_chart_id, m_name + "CloseMAVal", m_subwin, INDENT_LEFT + labelW + 2, y1, INDENT_LEFT + labelW + valW, y1 + lineH))
      return false;
   m_lblCloseMAVal.Text("--");
   m_lblCloseMAVal.Color(clrWhite);
   m_lblCloseMAVal.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblCloseMAVal)) return false;

   y1 += lineH + 2;
   if(!m_lblLaminarLbl.Create(m_chart_id, m_name + "LaminarLbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + labelW, y1 + lineH))
      return false;
   m_lblLaminarLbl.Text("Laminar:");
   m_lblLaminarLbl.Color(clrWhite);
   m_lblLaminarLbl.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblLaminarLbl)) return false;
   if(!m_lblLaminarVal.Create(m_chart_id, m_name + "LaminarVal", m_subwin, INDENT_LEFT + labelW + 2, y1, INDENT_LEFT + labelW + valW, y1 + lineH))
      return false;
   m_lblLaminarVal.Text("--");
   m_lblLaminarVal.Color(clrWhite);
   m_lblLaminarVal.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblLaminarVal)) return false;

   y1 += lineH + 2;
   if(!m_lblBandwidthLbl.Create(m_chart_id, m_name + "BandwidthLbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + labelW, y1 + lineH))
      return false;
   m_lblBandwidthLbl.Text("Bandwidth:");
   m_lblBandwidthLbl.Color(clrWhite);
   m_lblBandwidthLbl.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblBandwidthLbl)) return false;
   if(!m_lblBandwidthVal.Create(m_chart_id, m_name + "BandwidthVal", m_subwin, INDENT_LEFT + labelW + 2, y1, INDENT_LEFT + labelW + valW, y1 + lineH))
      return false;
   m_lblBandwidthVal.Text("--");
   m_lblBandwidthVal.Color(clrWhite);
   m_lblBandwidthVal.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblBandwidthVal)) return false;

   y1 += lineH + 2;
   if(!m_lblSlopeLbl.Create(m_chart_id, m_name + "SlopeLbl", m_subwin, INDENT_LEFT, y1, INDENT_LEFT + labelW, y1 + lineH))
      return false;
   m_lblSlopeLbl.Text("Slope:");
   m_lblSlopeLbl.Color(clrWhite);
   m_lblSlopeLbl.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblSlopeLbl)) return false;
   if(!m_lblSlopeVal.Create(m_chart_id, m_name + "SlopeVal", m_subwin, INDENT_LEFT + labelW + 2, y1, INDENT_LEFT + labelW + valW, y1 + lineH))
      return false;
   m_lblSlopeVal.Text("--");
   m_lblSlopeVal.Color(clrWhite);
   m_lblSlopeVal.FontSize(TELEMETRY_FONT_SIZE);
   if(!Add(m_lblSlopeVal)) return false;

   if(!Debug)
   {
      m_lblCloseMALbl.Visible(false); m_lblCloseMAVal.Visible(false);
      m_lblLaminarLbl.Visible(false); m_lblLaminarVal.Visible(false);
      m_lblBandwidthLbl.Visible(false); m_lblBandwidthVal.Visible(false);
      m_lblSlopeLbl.Visible(false); m_lblSlopeVal.Visible(false);
   }
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
   if(!HasOurPendingOrders()) return;
   CancelOurOrders();
}

void CMAFilterDialog::OnClickCancelAll()
{
   if(!HasAnyPendingOrders()) return;
   CancelAllOrders();
}

void CMAFilterDialog::OnClickClose()
{
   if(!HasOurPosition()) return;
   CloseOurPositions();
}

void CMAFilterDialog::OnClickCloseAll()
{
   if(!HasAnyPosition()) return;
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

ulong CMAFilterDialog::ResolveMagicNumber()
{
   if(MagicNumber != 0) return (ulong)MagicNumber;
   string hashInput = _Symbol + MQLInfoString(MQL_PROGRAM_NAME) + IntegerToString((int)Period());
   return CHash::Hash(hashInput);
}

bool CMAFilterDialog::HasOurPendingOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!m_orderInfo.SelectByIndex(i)) continue;
      if(m_orderInfo.Symbol() != _Symbol || m_orderInfo.Magic() != m_effectiveMagic) continue;
      return true;
   }
   return false;
}

bool CMAFilterDialog::HasAnyPendingOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!m_orderInfo.SelectByIndex(i)) continue;
      if(m_orderInfo.Symbol() != _Symbol) continue;
      return true;
   }
   return false;
}

bool CMAFilterDialog::HasOurPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol || m_posInfo.Magic() != m_effectiveMagic) continue;
      return true;
   }
   return false;
}

bool CMAFilterDialog::HasAnyPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol) continue;
      return true;
   }
   return false;
}

void CMAFilterDialog::UpdateCancelCloseStates()
{
   bool hasOurOrders = HasOurPendingOrders();
   bool hasOurPos = HasOurPosition();
   bool hasAnyPos = HasAnyPosition();

   if(hasOurOrders)
   {
      m_btnCancel.Enable();
      m_btnCancel.ColorBackground(clrYellow);
      m_btnCancel.Color(clrBlue);
   }
   else
   {
      m_btnCancel.Disable();
      m_btnCancel.ColorBackground(clrDarkGray);
      m_btnCancel.Color(clrLightGray);
   }

   if(hasOurPos)
   {
      m_btnClose.Enable();
      m_btnClose.ColorBackground(clrDarkRed);
      m_btnClose.Color(clrWhite);
   }
   else
   {
      m_btnClose.Disable();
      m_btnClose.ColorBackground(clrDarkGray);
      m_btnClose.Color(clrLightGray);
   }

   if(hasAnyPos)
   {
      m_btnCloseAll.Enable();
      m_btnCloseAll.ColorBackground(clrDarkRed);
      m_btnCloseAll.Color(clrWhite);
   }
   else
   {
      m_btnCloseAll.Disable();
      m_btnCloseAll.ColorBackground(clrDarkGray);
      m_btnCloseAll.Color(clrLightGray);
   }

   bool hasAnyOrders = HasAnyPendingOrders();
   if(hasAnyOrders)
   {
      m_btnCancelAll.Enable();
      m_btnCancelAll.ColorBackground(clrYellow);
      m_btnCancelAll.Color(clrBlue);
   }
   else
   {
      m_btnCancelAll.Disable();
      m_btnCancelAll.ColorBackground(clrDarkGray);
      m_btnCancelAll.Color(clrLightGray);
   }
}

void CMAFilterDialog::UpdatePositionProfitDisplay()
{
   double longVol = 0, shortVol = 0;
   double symbolProfit = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol) continue;
      double vol = m_posInfo.Volume();
      if(m_posInfo.PositionType() == POSITION_TYPE_BUY) longVol += vol;
      else shortVol += vol;
      symbolProfit += m_posInfo.Profit() + m_posInfo.Swap() + m_posInfo.Commission();
   }

   double netVol = longVol - shortVol;
   double sessionProfit = AccountInfoDouble(ACCOUNT_EQUITY) - m_sessionStartEquity;
   double totalProfit = AccountInfoDouble(ACCOUNT_PROFIT);

   color cNet = (netVol > 0) ? clrGreen : (netVol < 0) ? clrRed : clrWhite;
   color cLong = (longVol > 0) ? clrGreen : clrWhite;
   color cShort = (shortVol > 0) ? clrRed : clrWhite;
   color cSym = (symbolProfit > 0) ? clrGreen : (symbolProfit < 0) ? clrRed : clrWhite;
   color cSess = (sessionProfit > 0) ? clrGreen : (sessionProfit < 0) ? clrRed : clrWhite;
   color cProf = (totalProfit > 0) ? clrGreen : (totalProfit < 0) ? clrRed : clrWhite;

   m_lblNetVal.Text(DoubleToString(netVol, m_volDigits));
   m_lblNetVal.Color(cNet);
   m_lblLongVal.Text(DoubleToString(longVol, m_volDigits));
   m_lblLongVal.Color(cLong);
   m_lblShortVal.Text(DoubleToString(shortVol, m_volDigits));
   m_lblShortVal.Color(cShort);

   m_lblSymbolProfitVal.Text(DoubleToString(symbolProfit, 2));
   m_lblSymbolProfitVal.Color(cSym);
   m_lblSessionProfitVal.Text(DoubleToString(sessionProfit, 2));
   m_lblSessionProfitVal.Color(cSess);
   m_lblProfitVal.Text(DoubleToString(totalProfit, 2));
   m_lblProfitVal.Color(cProf);
}

void CMAFilterDialog::UpdateDepositLoadBar()
{
   double margin = AccountInfoDouble(ACCOUNT_MARGIN);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double load = (equity > 1e-10) ? (margin / equity * 100.0) : 0;
   double ratio = MathMin(1.0, load / MathMax(1e-10, MaxDepositLoad));   // For color: 0..1 vs MaxDepositLoad
   double fillRatio = MathMin(1.0, load / 100.0);                         // For bar length: 0..100%

   int barW = ClientAreaWidth() - 2 * INDENT_LEFT;
   int fillW = (int)(fillRatio * barW);

   int fillX = ClientAreaLeft() + INDENT_LEFT;
   int fillY = ClientAreaTop() + ROW7_Y_OFFSET;
   m_progressFill.Move(fillX, fillY);
   m_progressFill.Size(fillW, DEPOSIT_LOAD_BAR_H);

   // Linear gradient: green (0) -> yellow (0.5) -> red (1.0)
   int r, g, b;
   if(ratio <= 0.5)
   {
      double t = ratio * 2.0;
      r = (int)(255 * t);
      g = 255;
      b = 0;
   }
   else
   {
      double t = (ratio - 0.5) * 2.0;
      r = 255;
      g = (int)(255 * (1.0 - t));
      b = 0;
   }
   color fillColor = (color)((b << 16) | (g << 8) | r);
   m_progressFill.ColorBackground(fillColor);
   m_progressFill.ColorBorder(fillColor);

   // Contrast: dark text on light BG, white text on dark BG (luminance threshold 128)
   double luminance = 0.299 * r + 0.587 * g + 0.114 * b;
   m_lblDepositLoad.Text("Deposit Load " + DoubleToString(load, 1) + " %");
   m_lblDepositLoad.Color((luminance > 128) ? clrBlack : clrWhite);
}

void CMAFilterDialog::UpdateDebugLevels()
{
   if(!Debug)
   {
      m_lblCloseMALbl.Visible(false); m_lblCloseMAVal.Visible(false);
      m_lblLaminarLbl.Visible(false); m_lblLaminarVal.Visible(false);
      m_lblBandwidthLbl.Visible(false); m_lblBandwidthVal.Visible(false);
      m_lblSlopeLbl.Visible(false); m_lblSlopeVal.Visible(false);
      return;
   }
   if(m_nbrMa == 0) return;
   m_lblCloseMALbl.Visible(true); m_lblCloseMAVal.Visible(true);
   m_lblLaminarLbl.Visible(true); m_lblLaminarVal.Visible(true);
   m_lblBandwidthLbl.Visible(true); m_lblBandwidthVal.Visible(true);
   m_lblSlopeLbl.Visible(true); m_lblSlopeVal.Visible(true);

   double value = UseOHCLMean
      ? (iOpen(_Symbol, PERIOD_CURRENT, 1) + iHigh(_Symbol, PERIOD_CURRENT, 1) + iLow(_Symbol, PERIOD_CURRENT, 1) + 2.0 * iClose(_Symbol, PERIOD_CURRENT, 1)) / 5.0
      : iClose(_Symbol, PERIOD_CURRENT, 1);

   double closeMA = m_maBundle.CloseMALevel(value);
   double laminar = m_maBundle.laminar_level(StrictSlopeOrder);
   double bw = m_maBundle.bandwidth();
   tcMA* histBw = m_maBundle.GetHistoricalBandwidth();
   double bwSma = (histBw != NULL) ? histBw.GetMA() : bw;
   double bwLevel = (bwSma > 1e-10) ? bw / bwSma : 0.0;
   double slopeLvl = m_maBundle.SlopeLevel();

   m_lblCloseMAVal.Text(DoubleToString(closeMA, 3));
   m_lblCloseMAVal.Color((closeMA < 0) ? clrRed : (closeMA > 0) ? clrGreen : clrWhite);

   m_lblLaminarVal.Text(DoubleToString(laminar, 3));
   m_lblLaminarVal.Color((laminar < 0) ? clrRed : (laminar > 0) ? clrGreen : clrWhite);

   m_lblBandwidthVal.Text(DoubleToString(bwLevel, 3));
   m_lblBandwidthVal.Color(clrWhite);

   m_lblSlopeVal.Text(DoubleToString(slopeLvl, 3));
   m_lblSlopeVal.Color((slopeLvl < 0) ? clrRed : (slopeLvl > 0) ? clrGreen : clrWhite);
}

int CMAFilterDialog::GetPositionDirection()
{
   int dir = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!m_posInfo.SelectByIndex(i)) continue;
      if(m_posInfo.Symbol() != _Symbol) continue;
      if(m_posInfo.Magic() != m_effectiveMagic) continue;
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
   string comment = "{\"magic\":" + StringFormat("%I64u", m_effectiveMagic) + "}";
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
   string comment = "{\"magic\":" + StringFormat("%I64u", m_effectiveMagic) + "}";
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
      if(m_orderInfo.Symbol() != _Symbol || m_orderInfo.Magic() != m_effectiveMagic) continue;
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
   bool hasLongSignal = m_blinkLong;
   bool hasShortSignal = m_blinkShort;

   if(hasLongSignal)
   {
      m_btnOpenLong.Enable();
      m_btnLongLMT.Enable();
      color cLong = (m_blinkOn) ? clrLime : clrDarkGreen;
      m_btnOpenLong.ColorBackground(cLong);
      m_btnLongLMT.ColorBackground((m_blinkOn) ? clrLime : clrForestGreen);
      m_btnOpenLong.Color(clrWhite);
      m_btnLongLMT.Color(clrWhite);
   }
   else
   {
      m_btnOpenLong.Disable();
      m_btnLongLMT.Disable();
      m_btnOpenLong.ColorBackground(clrDarkGray);
      m_btnLongLMT.ColorBackground(clrDarkGray);
      m_btnOpenLong.Color(clrLightGray);
      m_btnLongLMT.Color(clrLightGray);
   }

   if(hasShortSignal)
   {
      m_btnOpenShort.Enable();
      m_btnShortLMT.Enable();
      color cShort = (m_blinkOn) ? clrOrangeRed : clrDarkRed;
      m_btnOpenShort.ColorBackground(cShort);
      m_btnShortLMT.ColorBackground((m_blinkOn) ? clrOrangeRed : clrCrimson);
      m_btnOpenShort.Color(clrWhite);
      m_btnShortLMT.Color(clrWhite);
   }
   else
   {
      m_btnOpenShort.Disable();
      m_btnShortLMT.Disable();
      m_btnOpenShort.ColorBackground(clrDarkGray);
      m_btnShortLMT.ColorBackground(clrDarkGray);
      m_btnOpenShort.Color(clrLightGray);
      m_btnShortLMT.Color(clrLightGray);
   }
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

   // Open Long/Short/LMT, Cancel All — set by UpdateOpenButtons / UpdateCancelCloseStates

   // Cancel, Close, Close All — set by UpdateCancelCloseStates

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

   // Slope strength: bundle SlopeLevel (signed ratio vs historical) must exceed MinSlopeFactor
   double slopeLevel = m_maBundle.SlopeLevel();
   bool slopesStrongLong  = slopeLevel >= MinSlopeFactor;
   bool slopesStrongShort = slopeLevel <= -MinSlopeFactor;

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

   bool longEntry  = closeAboveAll && allSlopesPos && slopesStrongLong && laminarLong && bwOk;
   bool shortEntry = closeBelowAll && allSlopesNeg && slopesStrongShort && laminarShort && bwOk;
   if(longEntry) return 1;
   if(shortEntry) return -1;
   return 0;
}
//+------------------------------------------------------------------+

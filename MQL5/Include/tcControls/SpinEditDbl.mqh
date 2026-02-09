//+------------------------------------------------------------------+
//|                                                     SpinEdit.mqh |
//|                   Copyright 2009-2017, MetaQuotes Software Corp. |
//|                                              http://www.mql5.com |
//|                                   http://www.trade-commander.com |
//| trade-commander.com double version of CSpinEdit                  |
//+------------------------------------------------------------------+
#include "WndContainer.mqh"
#include "Edit.mqh"
#include "BmpButton.mqh"
//+------------------------------------------------------------------+
//| Resources                                                        |
//+------------------------------------------------------------------+
#resource "res\\SpinInc.bmp"
#resource "res\\SpinDec.bmp"
//+------------------------------------------------------------------+
//| Class CSpinEditDbl                                                  |
//| Usage: class that implements the "Up-Down" control               |
//+------------------------------------------------------------------+
class CSpinEditDbl : public CWndContainer
  {
public:
   //--- dependent controls
   // ce! public to catch edit events outside
   CEdit             m_edit;                // the entry field object
private:   
   CBmpButton        m_inc;                 // the "Increment button" object
   CBmpButton        m_dec;                 // the "Decrement button" object
   //--- adjusted parameters
   double            m_min_value;           // minimum value
   double            m_max_value;           // maximum value
   //--- state
   double            m_value;               // current value
   double            _step;                 // step value
   int               _digits;               // digits

public:
                     CSpinEditDbl(void);
                    ~CSpinEditDbl(void);
   //--- create
   virtual bool      Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2);
   //--- chart event handler
   virtual bool      OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam);
   //--- set up
   double            MinValue(void) const { return(m_min_value); }
   void              MinValue(const double value);
   double            MaxValue(void) const { return(m_max_value); }
   void              MaxValue(const double value);

   double            step(void) const { return(_step); }
   void              step(const double value);

   int               digits(void) const { return(_digits); }
   void              digits(const int value);

   //--- state
   double            Value(void) const { return(m_value); }
   bool              Value(double value,bool trigger_event=true);
   //--- methods for working with files
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);

// tc
public:
   color             Color(void) const { return m_edit.Color(); }
   bool              Color(const color value) {return m_edit.Color(value);}
   color             ColorBackground(void) const { return m_edit.ColorBackground(); }
   bool              ColorBackground(const color value) {return m_edit.ColorBackground(value);}
   color             ColorBorder(void) const { return m_edit.ColorBorder(); }
   bool              ColorBorder(const color value) {return m_edit.ColorBorder(value);}
   string            Font(void) const { return m_edit.Font(); }
   bool              Font(const string value) {return  m_edit.Font(value);}
   int               FontSize(void) const { return m_edit.FontSize(); }
   bool              FontSize(const int value) {return m_edit.FontSize(value);}

public:
    //! set value from edit and reset edit from normalized value
    bool edit_to_value(void)
    {
        double val=StringToDouble(m_edit.Text());
        long units=(_step != 0.0 ?  val / _step : 0);
        m_value=_step * units;
        // normalize
        m_edit.Text(DoubleToString(m_value,_digits)); 
        return true;    
    }
protected:
   //--- create dependent controls
   virtual bool      CreateEdit(void);
   virtual bool      CreateInc(void);
   virtual bool      CreateDec(void);
   //--- handlers of the dependent controls events
   virtual bool      OnClickInc(void);
   virtual bool      OnClickDec(void);
   virtual bool      OnEditChange(void);
   
   //--- internal event handlers
   virtual bool      OnChangeValue(bool trigger_event=true);
  };
//+------------------------------------------------------------------+
//| Common handler of chart events                                   |
//+------------------------------------------------------------------+
EVENT_MAP_BEGIN(CSpinEditDbl)
   ON_EVENT(ON_CLICK,m_inc,OnClickInc)
   ON_EVENT(ON_CLICK,m_dec,OnClickDec)
   ON_EVENT(ON_END_EDIT,m_edit,OnEditChange)
EVENT_MAP_END(CWndContainer)
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSpinEditDbl::CSpinEditDbl(void) : m_min_value(0),
                             m_max_value(0),
                             m_value(0),
                             _step(1.0),
                             _digits(1)
  {
  }
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSpinEditDbl::~CSpinEditDbl(void)
  {
  }
//+------------------------------------------------------------------+
//| Create a control                                                 |
//+------------------------------------------------------------------+
bool CSpinEditDbl::Create(const long chart,const string name,const int subwin,const int x1,const int y1,const int x2,const int y2)
  {
//--- check height
   if(y2-y1<CONTROLS_SPIN_MIN_HEIGHT)
      return(false);
//--- call method of the parent class
   if(!CWndContainer::Create(chart,name,subwin,x1,y1,x2,y2))
      return(false);
//--- create dependent controls
   if(!CreateEdit())
      return(false);
   if(!CreateInc())
      return(false);
   if(!CreateDec())
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Set current value                                                |
//+------------------------------------------------------------------+
bool CSpinEditDbl::Value(double value,bool trigger_event=true)
  {
//--- check value
   if(value<m_min_value)
      value=m_min_value;
   if(value>m_max_value)
      value=m_max_value;
//--- if value was changed
   if(m_value!=value || trigger_event)
     {
      m_value=value;
      //--- call virtual handler
        return(OnChangeValue(trigger_event));
     }
//--- value has not been changed
   return(false);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSpinEditDbl::Save(const int file_handle)
  {
//--- check
   if(file_handle==INVALID_HANDLE)
      return(false);
//---
   FileWriteInteger(file_handle,m_value);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSpinEditDbl::Load(const int file_handle)
  {
//--- check
   if(file_handle==INVALID_HANDLE)
      return(false);
//---
   if(!FileIsEnding(file_handle))
      Value(FileReadInteger(file_handle));
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Set minimum value                                                |
//+------------------------------------------------------------------+
void CSpinEditDbl::MinValue(const double value)
  {
//--- if value was changed
   if(m_min_value!=value)
     {
      m_min_value=value;
      //--- adjust the edit value
      Value(m_value);
     }
  }
//+------------------------------------------------------------------+
//| Set maximum value                                                |
//+------------------------------------------------------------------+
void CSpinEditDbl::MaxValue(const double value)
  {
//--- if value was changed
   if(m_max_value!=value)
     {
      m_max_value=value;
      //--- adjust the edit value
      Value(m_value);
     }
  }
  
//+------------------------------------------------------------------+
//| Set step value                                                |
//+------------------------------------------------------------------+
void CSpinEditDbl::step(const double value)
{
    _step=value;
}

//+------------------------------------------------------------------+
//| Set digits value                                                |
//+------------------------------------------------------------------+
void CSpinEditDbl::digits(const int value)
{
    _digits=value;
}

  
//+------------------------------------------------------------------+
//| Create the edit field                                            |
//+------------------------------------------------------------------+
bool CSpinEditDbl::CreateEdit(void)
  {
//--- create
   if(!m_edit.Create(m_chart_id,m_name+"Edit",m_subwin,0,0,Width(),Height()))
      return(false);
   if(!m_edit.Text(""))
      return(false);
  // if(!m_edit.ReadOnly(true))
  //    return(false);
   if(!Add(m_edit))
      return(false);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the "Increment" button                                    |
//+------------------------------------------------------------------+
bool CSpinEditDbl::CreateInc(void)
  {
//--- right align button (try to make equal offsets from top and bottom)
   int x1=Width()-(CONTROLS_BUTTON_SIZE+CONTROLS_SPIN_BUTTON_X_OFF);
   int y1=(Height()-2*CONTROLS_SPIN_BUTTON_SIZE)/2;
   int x2=x1+CONTROLS_BUTTON_SIZE;
   int y2=y1+CONTROLS_SPIN_BUTTON_SIZE;
//--- create
   if(!m_inc.Create(m_chart_id,m_name+"Inc",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_inc.BmpNames("::res\\SpinInc.bmp"))
      return(false);
   if(!Add(m_inc))
      return(false);
//--- property
   m_inc.PropFlags(WND_PROP_FLAG_CLICKS_BY_PRESS);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Create the "Decrement" button                                    |
//+------------------------------------------------------------------+
bool CSpinEditDbl::CreateDec(void)
  {
//--- right align button (try to make equal offsets from top and bottom)
   int x1=Width()-(CONTROLS_BUTTON_SIZE+CONTROLS_SPIN_BUTTON_X_OFF);
   int y1=(Height()-2*CONTROLS_SPIN_BUTTON_SIZE)/2+CONTROLS_SPIN_BUTTON_SIZE;
   int x2=x1+CONTROLS_BUTTON_SIZE;
   int y2=y1+CONTROLS_SPIN_BUTTON_SIZE;
//--- create
   if(!m_dec.Create(m_chart_id,m_name+"Dec",m_subwin,x1,y1,x2,y2))
      return(false);
   if(!m_dec.BmpNames("::res\\SpinDec.bmp"))
      return(false);
   if(!Add(m_dec))
      return(false);
//--- property
   m_dec.PropFlags(WND_PROP_FLAG_CLICKS_BY_PRESS);
//--- succeed
   return(true);
  }
//+------------------------------------------------------------------+
//| Handler of click on the "increment" button                       |
//+------------------------------------------------------------------+
bool CSpinEditDbl::OnClickInc(void)
  {
//--- try to increment current value
   return(Value(m_value+_step));
  }
//+------------------------------------------------------------------+
//| Handler of click on the "decrement" button                       |
//+------------------------------------------------------------------+
bool CSpinEditDbl::OnClickDec(void)
  {
//--- try to decrement current value
   return(Value(m_value-_step));
  }
//+------------------------------------------------------------------+
//| Handler of edit change event                      |
//+------------------------------------------------------------------+
bool CSpinEditDbl::OnEditChange(void)
{
    if(edit_to_value() == true)
    {
      OnChangeValue(true);
      return true;
    }
    return false;
}
  
//+------------------------------------------------------------------+
//| Handler of changing current state                                |
//+------------------------------------------------------------------+
bool CSpinEditDbl::OnChangeValue(bool trigger_event)
  {
//--- copy text to the edit field edit
   m_edit.Text(DoubleToString(m_value,_digits));
//--- send notification
    if(trigger_event == true)
    EventChartCustom(CONTROLS_SELF_MESSAGE,ON_CHANGE,m_id,0.0,m_name);
//--- handled
   return(true);
  }
//+------------------------------------------------------------------+

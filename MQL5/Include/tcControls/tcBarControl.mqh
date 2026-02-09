#include <tcControls/Edit.mqh>
#include <tcControls/Label.mqh>
#include <trade-commander/macros.mqh>


//------------------------
// tcBarControl
//------------------------
class tcBarControl
{
public:
   tcBarControl()
   {
      edBar_=new CEdit();
      lbText_=new CLabel();
      xmin_=0;
      xmax_=0;
      y_=0;
      prefix_="";
   }
   virtual ~tcBarControl()
   {
      TC_DEL(edBar_);
      TC_DEL(edBar_);
   }
public:
   CEdit* edBar(void) {return edBar_;}
   CLabel* lbText(void) {return lbText_;}
   color clrBack(void) const {return edBar_.ColorBackground();}
   
   inline void prefix(string x) {prefix_=x;}
public:

   virtual bool Create(const long chart,const string name,const int subwin, int x1, int y1, int x2, int y2=-1,bool bborder=false,string text="")
   {
      xmin_=x1;
      xmax_=x2;
      y_=y1;
      if(y2 < 0)
         y2=y1+20;
      /*
      // inner heigth lower
      if(bborder == true)
      {
         y1+=1;
         y2-=1;
      }
      */
      bool bret=edBar_.Create(chart,name+"B2",subwin,x1,y1,x1,y2);    
      edBar_.ReadOnly(true);

     // bret&=lbText_.Create(chart,name,subwin,x1,y1-1,x2,y2-1);    
      bret&=lbText_.Create(chart,name,subwin,x1+(xmax_-xmin_) / 2 - 50,y1-1,x2,y2-1);    
      lbText_.Text(text);
     // lbText_.TextAlign(ALIGN_CENTER);  
      //lbText_.ReadOnly(true);
      
      return bret;
   } 
   void setFont(string Font="Calibri",int Fontsize=10)
   {
      lbText_.Font(Font);
      lbText_.FontSize(Fontsize);
   }   
   void setColors(color clrFg,color clrBack,color clrBorder=clrNONE)
   {
      /*
      if(clrBorder == clrNONE)
         lbText_.ColorBorder(clrBack);
      else
         lbText_.ColorBorder(clrBorder);
      */
      //lbText_.ColorBackground(clrBack);   
      lbText_.Color(clrFg);
      
      edBar_.ColorBorder(clrBack);
      edBar_.ColorBackground(clrBack);
   }
public:
   void update(double percent, double percentBar=-1.0,color clrFg=clrNONE, color clrBk=clrNONE, bool bChartRedraw=false)
   {
      if(percent < 0)      
         percent=0.0;
      else if(percent > 100)      
         percent=100.0;

      if(percentBar < 0.0)
           percentBar=percent;      
      
      string text=StringFormat("%s %.2f",prefix_,percent);
      text += " %";
      lbText_.Text(text);
      
      int Width= (int) ((double) (xmax_-xmin_) * percentBar / 100.0);
      edBar_.Size(Width,edBar_.Height());
      
      
      //lbText_.Move(const int x,y_);
      
      if(clrFg != clrNONE)
         lbText_.Color(clrFg);
      if(clrBk != clrNONE)
         edBar_.ColorBackground(clrBk);
      if(bChartRedraw == true)
         ChartRedraw(lbText_.ChartID());
   }   
protected:
   int   xmin_;
   int   xmax_;
   int   y_;
   CEdit* edBar_;
   CLabel* lbText_;
   string prefix_;
};

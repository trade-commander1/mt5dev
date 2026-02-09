//+------------------------------------------------------------------+
//|                                                 export_deals.mq5 |
//|                              Copyright 2026, trade-commander.com |
//|                                  https://www.trade-commander.com |
//+------------------------------------------------------------------+

#property strict
#property script_show_inputs

//--- inputs: optional time range (leave as default to export all history)
input datetime InpFrom = 0;          // From date (0 = from earliest)
input datetime InpTo   = 0;          // To date   (0 = now)

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   datetime from_time = InpFrom;
   datetime to_time   = InpTo;

   // If not specified, use a very early start and current time as end
   if(from_time == 0)
      from_time = D'1970.01.01 00:00';
   if(to_time == 0)
      to_time = TimeCurrent();

   //--- select history range
   if(!HistorySelect(from_time, to_time))
   {
      Print("HistorySelect failed. Error = ", GetLastError());
      return;
   }

   int deals_total = HistoryDealsTotal();
   if(deals_total <= 0)
   {
      Print("No deals found in the selected period.");
      return;
   }

   //--- open CSV file in \MQL5\Files
   int file_handle = FileOpen("AllDeals.csv",
                              FILE_WRITE | FILE_CSV | FILE_ANSI,
                              ';');   // use ; as separator (change if you want ,)
   if(file_handle == INVALID_HANDLE)
   {
      Print("Failed to open file. Error = ", GetLastError());
      return;
   }

   //--- write header row
   FileWrite(file_handle,
             "Ticket",
             "Order",
             "Time",
             "Symbol",
             "Type",
             "Volume",
             "Price",
             "Profit",
             "Commission",
             "Swap",
             "Magic",
             "Comment");

   //--- iterate all deals
   for(int i = 0; i < deals_total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      ulong   order      = (ulong)HistoryDealGetInteger(ticket, DEAL_ORDER);
      datetime time      = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      string  symbol     = HistoryDealGetString(ticket, DEAL_SYMBOL);
      long    type       = HistoryDealGetInteger(ticket, DEAL_TYPE);
      double  volume     = HistoryDealGetDouble(ticket, DEAL_VOLUME);
      double  price      = HistoryDealGetDouble(ticket, DEAL_PRICE);
      double  profit     = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double  commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double  swap       = HistoryDealGetDouble(ticket, DEAL_SWAP);
      long    magic      = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      string  comment    = HistoryDealGetString(ticket, DEAL_COMMENT);

      //--- convert type to readable text
      string type_text;
      switch(type)
      {
         case DEAL_TYPE_BUY:             type_text = "BUY";              break;
         case DEAL_TYPE_SELL:            type_text = "SELL";             break;
         case DEAL_TYPE_BALANCE:         type_text = "BALANCE";          break;
         case DEAL_TYPE_CREDIT:          type_text = "CREDIT";           break;
         case DEAL_TYPE_CHARGE:          type_text = "CHARGE";           break;
         case DEAL_TYPE_CORRECTION:      type_text = "CORRECTION";       break;
         case DEAL_TYPE_BONUS:           type_text = "BONUS";            break;
         case DEAL_TYPE_COMMISSION:      type_text = "COMMISSION";       break;
         case DEAL_TYPE_COMMISSION_DAILY:type_text = "COMMISSION_DAILY"; break;
         case DEAL_TYPE_COMMISSION_MONTHLY:type_text="COMMISSION_MONTHLY";break;
         //case DEAL_TYPE_COMMISSION_AGENT:type_text = "COMMISSION_AGENT"; break;
         case DEAL_TYPE_INTEREST:        type_text = "INTEREST";         break;
         case DEAL_TYPE_BUY_CANCELED:    type_text = "BUY_CANCELED";     break;
         case DEAL_TYPE_SELL_CANCELED:   type_text = "SELL_CANCELED";    break;
         default:                        type_text = "OTHER";            break;
      }

      //--- write one CSV row
      FileWrite(file_handle,
                (long)ticket,
                (long)order,
                TimeToString(time, TIME_DATE | TIME_SECONDS),
                symbol,
                type_text,
                DoubleToString(volume, 2),
                DoubleToString(price,  _Digits),
                DoubleToString(profit, 2),
                DoubleToString(commission, 2),
                DoubleToString(swap, 2),
                magic,
                comment);
   }

   FileClose(file_handle);
   Print("Deals exported: ", deals_total,
         ". File: \\MQL5\\Files\\AllDeals.csv");
}
//+------------------------------------------------------------------+


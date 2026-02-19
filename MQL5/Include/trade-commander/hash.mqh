//+------------------------------------------------------------------+
//| hash.mqh                                                          |
//| String hasher returning ulong                                    |
//+------------------------------------------------------------------+
#property strict

//+------------------------------------------------------------------+
//| Static class for string hashing                                   |
//+------------------------------------------------------------------+
class CHash
{
public:
   /**
    * Hash a string and return ulong.
    */
   static ulong Hash(const string &str)
   {
      ulong h = 0;
      int len = StringLen(str);
      for(int i = 0; i < len; i++)
      {
         h = 31 * h + (ulong)StringGetCharacter(str, i);
         if(h == 0) h = 1;
      }
      return h;
   }
};
//+------------------------------------------------------------------+

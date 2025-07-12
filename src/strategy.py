from backtesting import Backtest, Strategy
from trend import generate_signals as EMA

def get_MSM_startegy(_ar_weights, data):

    class MSM_Strategy(Strategy):
        ar_weights = _ar_weights

        def init(self):
            self.signals = self.I(EMA, self.data, self.ar_weights)

        def next(self):
            price = self.data.Close[-1]
            #print(self.signals[-1])

            if not self.position and self.signals[-1]>0:
                self.buy()
                #print("Buying at ", price, self.position)
            elif self.position and self.signals[-1]<0:
                self.position.close()
                #print("Selling at", price, self.position)
    
    return MSM_Strategy
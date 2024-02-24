import backtrader as bt
from backtrader.feeds import GenericCSVData
import numpy as np


class GenericCSV_extend(GenericCSVData):
    """
        Extending the GenericCSVData to include new indicators
    """
    lines = ('index',)

    # Setting default parameters
    params = (('index', 5),
              ('nullvalue', float('NaN')))


class TotalValue(bt.analyzers.Analyzer):
    """
        Analyzer returning cash and market values
    """

    def create_analysis(self):
        self.rets = {}
        self.vals = 0.0

    def notify_cashvalue(self, cash, value):
        self.vals = (
            self.strategy.datetime.datetime().strftime('%Y-%m-%d'),
            value,
        )
        self.rets[len(self)] = self.vals

    def get_analysis(self):
        return self.rets
    


class MyStrategy(bt.Strategy):
    """
        Main class which used to conduct backtest

        Parameters:
            group: indicate which group is currently testing
            printlog: print log or not
            hedge: using hedge to sell the last group or not
            reverse: reverse the sort order or not
    """

    # Strategy parameters
    params = dict(
        account_value = [],
        total_trend = []
    )

    def __init__(self):
        self.index = self.datas[0].index
        self.order_refs = {}


    def next(self):
        self.params.account_value.append(self.broker.getvalue())
        close_price = [i.close[0] for i in self.datas]
        self.params.total_trend.append(np.mean(close_price))

        total_value = self.broker.getvalue()
        p_value = total_value*0.5
        size=int(p_value/self.datas[0].close[0])

        if (self.index[0]==1)&(self.getposition().size<0):
            self.close()
            order = self.buy(size=size)
            self.order_refs[order.ref] = "buy"
        elif (self.index[0]==1)&(self.getposition().size==0):
            order = self.buy(size=size)
            self.order_refs[order.ref] = "buy"

        if (self.index[0]==0)&(self.getposition().size!=0):
            self.close()

        if (self.index[0]==-1)&(self.getposition().size>0):
            self.close()
            order = self.sell(size=size)
            self.order_refs[order.ref] = "sell"
        elif (self.index[0]==-1)&(self.getposition().size==0):
            order = self.sell(size=size)
            self.order_refs[order.ref] = "sell"


    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        time = None or self.datas[0].datetime.time(0)
        print(f'{dt.isoformat()} {time.isoformat()},{txt}')

    # 记录交易执行情况（可省略，默认不输出结果）
    def notify_order(self, order):
        # 如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果order为buy/sell executed,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入:{order.data._name}\n价格:{order.executed.price:.2f},\
                发起：{self.order_refs[order.ref] if order.ref in self.order_refs else "close"},\
                成本:{order.executed.value:.2f},\
                手续费:{order.executed.comm:.2f}')

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出:{order.data._name}\n价格:{order.executed.price:.2f},\
                发起：{self.order_refs[order.ref] if order.ref in self.order_refs else "close"},\
                成本: {order.executed.value:.2f},\
                手续费{order.executed.comm:.2f}')
            if order.ref in self.order_refs: del self.order_refs[order.ref] 
            self.bar_executed = len(self)

        # 如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'交易失败:{order.data._name}, 原因:{order.status}, Value: {order.created.value}, Cash: {self.broker.getcash()}')
        self.order = None

    #记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self,trade):
        if trade.isclosed:
            self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')

    def stop(self):
        for a in self.datas:
            print(a._name)
        return
from backtraderHelper import *
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

def backTest(data, start_date, end_date, logger):
    """
        Conduct the backtest using given parameters

        params:
            data: directory of data source   
            start_date: starting data of data
            end_date: ending date of data         
            logger: the universal logger
    """
    cerebro = bt.Cerebro()

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='pnl')  # 返回收益率时序数据
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='_AnnualReturn')  # 年化收益率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='_DrawDown')  # 回撤
    cerebro.addanalyzer(TotalValue, _name="_TotalValue") # 账户总值

    cerebro.addstrategy(MyStrategy)

    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    data_df = GenericCSV_extend(
        dataname=data,
        fromdate=start_date,
        todate=end_date,
        dtformat = '%Y-%m-%d',
        datetime=0,
        open = 1,
        high = 2,
        low = 3,
        close=4,
        index=5,
        volume=-1,
        openinterest=-1
        )
    cerebro.adddata(data_df)

    # 将初始本金设为100w
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(0.00002)
    cerebro.addobserver(bt.observers.TimeReturn)

    logger.info('启动资金: %.2f' % cerebro.broker.getvalue())

    result = cerebro.run()

    # 以年化收益率1.7%作为无风险利率计算Sharpe ratio
    risk_free_rate = 0.017
    risk_free_rate_daily = (1 + risk_free_rate)**(1/252) - 1
    df = pd.DataFrame({"Account value":result[0].p.account_value})
    df["Time"] = pd.to_datetime(pd.read_csv(data)["Date"])
    df['Account value'] = df['Account value']-500000
    df['Return'] = df['Account value'].pct_change()
    # Calculate the excess returns by subtracting the daily risk-free rate from the daily returns
    df['Excess Return'] = df['Return'] - risk_free_rate_daily
    # Calculate the Sharpe Ratio
    sharpe_ratio = df['Excess Return'].mean() / df['Excess Return'].std()
    # Annualize the Sharpe Ratio
    sharpe_ratio = sharpe_ratio * np.sqrt(252)
    
    # 计算年化收益率
    annual_return = []
    for year in result[0].analyzers._AnnualReturn.get_analysis().items():
        annual_return.append(year[1])

    logger.info('期末价值: %.2f' % cerebro.broker.getvalue())
    logger.info(f'Annual Return: {np.mean(annual_return)*2}')
    logger.info(f'Sharp Ratio: {sharpe_ratio}')
    logger.info(f'Max DrawDown: {result[0].analyzers._DrawDown.get_analysis()["max"]["drawdown"]*2}%')

    # 计算市场以及策略账户总值
    account_value = np.array(result[0].p.account_value)
    account_value = account_value/(account_value[0]/2)-1
    total_trend = np.array(result[0].p.total_trend)
    total_trend = total_trend/total_trend[0]

    cerebro.plot(volume=False)

    plt.plot(df["Time"], account_value, label="Account value")
    plt.plot(df["Time"],total_trend, label="Market trend")
    plt.xticks(rotation=45)
    plt.title("Account value compared to market trend")
    plt.legend()
    plt.show()

    plt.plot(df["Time"], account_value-total_trend, label="Excess return")
    plt.xticks(rotation=45)
    plt.title("Excess return of strategy")
    plt.legend()
    plt.show()

    print("done")
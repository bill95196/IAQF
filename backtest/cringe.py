from main import backTest
import logging
import backtrader as bt

logging.basicConfig(filename=None, filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)

# data: 回测数据的directory
# start_date: 回测数据开始的日期
# end_date: 回测数据截至的日期
backTest(
            data='/Users/jinnia/Desktop/backtest/raw data/backtesting_data.csv',
            start_date=bt.datetime.datetime(2015, 1, 7),
            end_date=bt.datetime.datetime(2024, 1, 10),
            logger=logger)
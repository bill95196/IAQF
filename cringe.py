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
            data='./raw data/backtesting_data.csv',
            start_date=bt.datetime.datetime(2018, 1, 31),
            end_date=bt.datetime.datetime(2021, 12, 2),
            logger=logger)
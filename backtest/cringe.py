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
            data=r"C:\Users\azhe1\Desktop\IAQF\sp500_index.csv",
            start_date=bt.datetime.datetime(2007, 1, 12),
            end_date=bt.datetime.datetime(2023, 7, 26),
            logger=logger)
import datetime

import pandas as pd
import pandas_datareader.data as web
import pytest

def get_stock_data(ticker):
    """pull data from stooq"""
    df = web.DataReader(ticker, 'yahoo')
    return df

# scope="class" tears down the fixture only at the end of the last test in the
# class, so we avoid rerunning this step.
@pytest.fixture(scope="class")
def stock_df():
    # We only want to pull this data once for each TestCase since it is an
    # expensive operation
    df = get_stock_data('^DJI')
    return df

class TestGetStockData:
    def test_columns_present(self, stock_df):
        # ensures that the expected columns are all present
        assert "Open" in stock_df.columns
        assert "High" in stock_df.columns
        assert "Low" in stock_df.columns
        assert "Close" in stock_df.columns
        assert "Volume" in stock_df.columns

    def test_non_empty(self, stock_df):
        # ensures that there is more than one row of data
        assert len(stock_df.index) != 0

    def test_high_low(self, stock_df):
        # ensure high and low are the highest and lowest in the same row
        ohlc = stock_df[["Open","High","Low","Close"]]
        highest = ohlc.max(axis=1)
        lowest = ohlc.min(axis=1)
        assert ohlc.le(highest, axis=0).all(axis=None)
        assert ohlc.ge(lowest, axis=0).all(axis=None)

    def test_most_recent_within_week(self, stock_df):
        # most recent data was collected within the last week
        most_recent_date = pd.to_datetime(stock_df.index[-1])
        assert (datetime.datetime.today() - most_recent_date).days <= 7

import numpy as np
from PySide6.QtCore import QObject, Slot
import talib

class StockCalculate(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    @Slot('QVariantList', int, int, int, result='QVariantList')
    def calculate_bollinger_bands(self, close_prices, period=20, nbdevup=2, nbdevdn=2):
        """计算布林带并处理NaN值"""
        if len(close_prices) < period:
            return []

        upper, middle, lower = talib.BBANDS(
            np.array(close_prices),
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )

        result = []
        for i in range(len(close_prices)):
            item = {
                "upper": float(upper[i]) if not np.isnan(upper[i]) else None,
                "middle": float(middle[i]) if not np.isnan(middle[i]) else None,
                "lower": float(lower[i]) if not np.isnan(lower[i]) else None
            }
            result.append(item)
        
        return result
    
    @Slot('QVariantList', int, result='QVariantList')
    def calculate_ema(self, close_prices, period):
        """计算指数移动平均线"""
        if len(close_prices) < period:
            return []
        return talib.EMA(np.array(close_prices), timeperiod=period).tolist()
    
    @Slot('QVariantList', 'QVariantList', 'QVariantList', int, int, int, result='QVariantList')
    def calculate_kdj(self, high_prices, low_prices, close_prices, fastk_period=9, slowk_period=3, slowd_period=3):
        """计算KDJ"""
        min_length = fastk_period + slowd_period
        if len(close_prices) < min_length:
            return []

        high_array = np.array(high_prices, dtype=np.float64)
        low_array = np.array(low_prices, dtype=np.float64)
        close_array = np.array(close_prices, dtype=np.float64)
        
        slowk, slowd = talib.STOCH(
            high_array,
            low_array,
            close_array,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        j = 3 * slowk - 2 * slowd

        results = []
        for i in range(len(close_array)):
            # 处理NaN值并转换类型
            result_item = {
                "k": 0.0 if np.isnan(slowk[i]) else float(slowk[i]),
                "d": 0.0 if np.isnan(slowd[i]) else float(slowd[i]),
                "j": 0.0 if np.isnan(j[i]) else float(j[i])
            }
            results.append(result_item)
        return results
  
    @Slot('QVariantList', int, int, int, result='QVariantList')
    def calculate_macd(self, close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
        """计算MACD"""

        if len(close_prices) < slowperiod:
            return []

        close_array = np.array(close_prices, dtype=np.float64)

        # 计算MACD
        macd, signal, hist = talib.MACD(
            close_array,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )

        results = []
        for i in range(len(close_array)):
            result_item = {
                "macd": 0.0 if np.isnan(macd[i]) else float(macd[i]),
                "signal": 0.0 if np.isnan(signal[i]) else float(signal[i]),
                "histogram": 0.0 if np.isnan(hist[i]) else float(hist[i])
            }
            results.append(result_item)

        return results
    
    @Slot('QVariantList', int, result='QVariantList')
    def calculate_rsi(self, close_prices, period=14):

        """计算RSI"""
        if len(close_prices) < period:
            return []
        return talib.RSI(np.array(close_prices), timeperiod=period).tolist()
    
    @Slot('QVariantList', int, result='QVariantList')
    def calculate_sma(self, close_prices, period):
        """计算简单移动平均线"""
        if len(close_prices) < period:
            return []
        return talib.SMA(np.array(close_prices), timeperiod=period).tolist()   
import aiohttp, asyncio
import json
import numpy as np
import os
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
                
class StockGet(QObject):          
    def __init__(self, parent=None):
        super().__init__(parent)  
        self.base_url = "https://hq.sinajs.cn/list="
        self.history_url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        self.timeout = aiohttp.ClientTimeout(total=10)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/"
        }
        self.stock_data = []
        self.stock_data = []
        self.favorites_file = os.path.join(os.path.expanduser("~"), "Stocker.json")

    @Slot(result='QVariantList')
    def load_favorite_stocks(self):
        """从本地文件加载自选股列表"""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r', encoding='utf-8') as f:
                    favorites = json.load(f)
                    if isinstance(favorites, list):
                        return favorites
        except Exception as e:
            print(f"加载自选股失败: {str(e)}")
        return []
    
    @Slot('QVariantList', result=bool)
    def save_favorite_stocks(self, favorites):
        """保存自选股列表到本地文件"""
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump(favorites, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存自选股失败: {str(e)}")
            return False
    
    @Slot('QVariantList', str, bool, result='QVariantList')
    def sort_stock_data(self, data, field, ascending):
        """排序数据
        :param data: 要排序的数据列表
        :param field: 排序字段
        :param ascending: 是否升序
        :return: 排序后的列表
        """
        if not data:
            return []
        
        try:
            reverse = not ascending
            sorted_data = sorted(
                data,
                key=lambda x: x.get(field, 0),
                reverse=reverse
            )
            return sorted_data
        except Exception as e:
            print(f"排序错误: {str(e)}")
            return data

    @Slot('QVariantList', str, result='QVariantList')
    def filter_stock_data(self, data, search_text):
        """过滤数据
        :param data: 要过滤的数据列表
        :param search_text: 搜索文本
        :return: 过滤后的列表
        """
        if not data or not isinstance(data, list):
            return []
        
        if not search_text:
            return data
        
        try:
            search_lower = search_text.lower()
            filtered = [
                item for item in data
                if (search_lower in item.get("代码", "").lower() or 
                    search_lower in item.get("名称", "").lower())
            ]
            return filtered
        except Exception as e:
            print(f"过滤错误: {str(e)}")
            return data
        
    async def fetch_history_stock_data(self, session, stock_code, scale=240, datalen=365*5):
        """
        获取单支股票历史K线数据
        :param stock_code: 股票代码，如'sh600000'
        :param scale: K线周期，240=日K线，1200=周K线， 7200=月K线，60=60分钟，30=30分钟，15=15分钟，5=5分钟
        :param datalen: 获取的数据长度
        :return: 列表
        """
        params = {
            "symbol": stock_code,
            "scale": scale,
            "datalen": datalen
        }
        
        try:
            async with session.get(
                self.history_url,
                params=params,
                timeout=self.timeout,
                headers=self.headers,
                ssl=True
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    return self.parse_history_stock_data(text, stock_code)
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"请求历史数据失败: {str(e)}")
            return None

    def parse_history_stock_data(self, response_text, stock_code):
        """解析单支股票历史K线数据"""
        try:
            # 解析JSON格式的历史数据
            data = json.loads(response_text)
            
            # 转换数据格式
            history_data = []
            for item in data:
                history_data.append({
                    "代码": stock_code,
                    "日期": item["day"],
                    "开盘价": float(item["open"]),
                    "收盘价": float(item["close"]),
                    "最高价": float(item["high"]),
                    "最低价": float(item["low"]),
                    "成交量": int(item["volume"]),
                })
            return history_data
        except Exception as e:
            return None
    
    async def get_all_stocks(self):
        """使用新浪财经接口分页获取沪深A股列表"""
        stocks = []
        total_pages = 55  # 总页数
        batch_size = 10    # 每次并行请求的页数
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
                # 定义内部函数用于处理单页请求
                async def fetch_page(page):
                    url = (
                        f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                        f"Market_Center.getHQNodeData?page={page}&num=100"
                        f"&sort=symbol&asc=1&node=hs_a&_s_r_a=page"
                    )
                    
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                if not data:
                                    return []
                                
                                # 解析数据项
                                return [{
                                    "代码": item.get("symbol"),
                                    "名称": item.get("name"),
                                    "今开": float(item.get("open", 0)),
                                    "昨收": float(item.get("settlement", 0)),
                                    "最新价": float(item.get("trade", 0)),
                                    "最高": float(item.get("high", 0)),
                                    "最低": float(item.get("low", 0)),
                                    "涨幅": float(item.get("changepercent", 0)),
                                    "涨跌": float(item.get("pricechange", 0)),
                                    "成交量": int(item.get("volume", 0)),
                                    "成交额": float(item.get("amount", 0)),
                                    "市盈率": float(item.get("per", 0)),
                                    "市净率": float(item.get("pb", 0)),
                                    "总市值": float(item.get("mktcap", 0)),
                                    "流通市值": float(item.get("nmc", 0)),
                                    "换手率": float(item.get("turnoverratio", 0)),
                                    "更新时间": item.get("ticktime")
                                } for item in data]
                            else:
                                print(f"第 {page} 页请求失败，状态码: {response.status}")
                                return []
                    except Exception as e:
                        print(f"第 {page} 页请求异常: {str(e)}")
                        return []

                # 分批并行处理
                for batch_start in range(1, total_pages + 1, batch_size):
                    batch_end = min(batch_start + batch_size, total_pages + 1)
                    tasks = [fetch_page(page) for page in range(batch_start, batch_end)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            print(f"批量请求失败: {result}")
                            continue
                        stocks.extend(result)
        
        except Exception as e:
            print(f"获取股票列表主流程失败: {e}")
        
        print(f"总共获取 {len(stocks)} 条股票数据")
        return stocks
    
    async def async_get_history_stock_data(self, stock_code, scale=240, datalen=1000):
        """异步获取单支股票历史数据"""
        async with aiohttp.ClientSession() as session:
            return await self.fetch_history_stock_data(session, stock_code, scale, datalen)
        
    @Slot(str, int, int, result='QVariantList')
    def get_history_stock_data(self, stock_code, scale=240, datalen=1000):
        """获取单支股票历史数据"""
        return asyncio.run(self.async_get_history_stock_data(stock_code, scale, datalen))
    
    async def async_get_stock_data(self):
        """异步获取股票数据"""
        return await self.get_all_stocks()
    
    @Slot(result='QVariantList')
    def get_stock_data(self):
        """获取股票数据"""
        self.stock_data = asyncio.run(self.async_get_stock_data())
        return self.stock_data
    
        

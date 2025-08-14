import aiohttp, asyncio
import json
import os
from PySide6.QtCore import QObject, Slot
             
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
            data = json.loads(response_text)
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
        total_pages = 55 
        batch_size = 10
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
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
    
        

"""
新浪财经-沪深 A 股实时行情和历史K线数据获取
"""
import aiohttp
import asyncio
import json
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex, Slot, QTimer, QObject


class StockInfoGet(QObject):          
    def __init__(self, parent=None):
        super().__init__(parent)  
        self.base_url = "https://hq.sinajs.cn/list="
        self.history_url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        self.timeout = aiohttp.ClientTimeout(total=10)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/"
        }
    
    async def fetch_stock_data(self, session, stock_codes):
        """获取股票实时数据"""
        url = f"{self.base_url}{','.join(stock_codes)}"
        try:
            async with session.get(
                url,
                timeout=self.timeout,
                headers=self.headers,
                ssl=True
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    return self.parse_stock_data(text)
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"请求失败: {str(e)}")
            return None
        
    async def fetch_history_stock_data(self, session, stock_code, scale=240, datalen=365*5):
        """
        获取股票历史K线数据
        :param stock_code: 股票代码，如'sh600000'
        :param scale: K线周期，240=日K线，60=60分钟，30=30分钟，15=15分钟，5=5分钟
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

    def parse_stock_data(self, response_text):
        """解析新浪财经返回的数据"""
        stocks_data = []
        for line in response_text.splitlines():
            if not line:
                continue
            try:
                # 提取股票代码和数据部分
                code_part, data_part = line.split("=", 1)
                stock_code = code_part.split("_")[-1]
                
                # 清理数据部分
                data_str = data_part.strip().strip('"')
                if not data_str:
                    continue
                
                data_fields = data_str.split(",")
                if len(data_fields) < 30:  # 新浪返回数据通常有30+字段
                    continue
                
                stock_data = {
                    "代码": stock_code, 
                    "名称": data_fields[0],
                    "今开": float(data_fields[1]),
                    "昨收": float(data_fields[2]),
                    "最新价": float(data_fields[3]),
                    "最高": float(data_fields[4]),
                    "最低": float(data_fields[5]),
                    "成交量": int(data_fields[8]),
                    "成交额": float(data_fields[9]),
                    "日期": data_fields[30],
                    "时间": data_fields[31],
                }
                
                # 计算涨跌幅和涨跌额
                if stock_data["昨收"] != 0:
                    stock_data["涨跌"] = round(stock_data["最新价"] - stock_data["昨收"], 2)
                    stock_data["涨幅"] = round((stock_data["涨跌"] / stock_data["昨收"]) * 100, 2)
                else:
                    stock_data["涨跌"] = 0
                    stock_data["涨幅"] = 0
                
                stocks_data.append(stock_data)
            except Exception as e:
                print(f"解析数据失败: {str(e)}")
                continue
        return stocks_data
    
    def parse_history_stock_data(self, response_text, stock_code):
        """解析历史K线数据"""
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
    
    def get_all_stock_list(self):
        """获取所有沪深A股股票代码（包括上证A股、深证A股）"""
        stock_codes = []
        # 深证A股股票代码 (000, 001, 002, 003, 300 开头)
        stock_codes += [f"sz000{str(i).zfill(3)}" for i in range(1, 1000)]  # 000001-000999
        stock_codes += [f"sz001{str(i).zfill(3)}" for i in range(0, 1000)]  # 001000-001999
        stock_codes += [f"sz002{str(i).zfill(3)}" for i in range(0, 1000)]  # 002000-002999（中小板）
        stock_codes += [f"sz003{str(i).zfill(3)}" for i in range(0, 1000)]  # 003000-003999
        stock_codes += [f"sz300{str(i).zfill(3)}" for i in range(0, 1000)]  # 300000-300999（创业板）
        # 上证A股股票代码 (600, 601, 603, 605, 688 开头)
        stock_codes += [f"sh600{str(i).zfill(3)}" for i in range(0, 1000)]  # 600000-600999
        stock_codes += [f"sh601{str(i).zfill(3)}" for i in range(0, 1000)]  # 601000-601999
        stock_codes += [f"sh603{str(i).zfill(3)}" for i in range(0, 1000)]  # 603000-603999
        stock_codes += [f"sh605{str(i).zfill(3)}" for i in range(0, 1000)]  # 605000-605999
        stock_codes += [f"sh688{str(i).zfill(3)}" for i in range(0, 1000)]  # 688000-688999（科创板）

        
        return stock_codes
    
    async def get_all_stocks(self):
        """获取所有股票数据(返回DataFrame)"""
        batch_size = 100
        all_results = []
        stock_list=self.get_all_stock_list()
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i:i+batch_size]
                tasks.append(self.fetch_stock_data(session, batch))
            
            results = await asyncio.gather(*tasks)
            for result in results:
                if result:
                    all_results.extend(result)
        if not all_results:
            return pd.DataFrame()
        df = pd.DataFrame(all_results)
        return df
     
    async def async_get_history_stock_data(self, stock_code, scale=240, datalen=1000):
        """异步获取单支股票历史数据"""
        async with aiohttp.ClientSession() as session:
            return await self.fetch_history_stock_data(session, stock_code, scale, datalen)
        
    @Slot(str, int, int, result=list)
    def get_history_stock_data(self, stock_code, scale=240, datalen=1000):
        """获取单支股票历史数据"""
        return asyncio.run(self.async_get_history_stock_data(stock_code, scale, datalen))
    
    async def async_get_stock_data(self):
        """异步获取股票数据"""
        return await self.get_all_stocks()
    
    @Slot(result=list)
    def get_stock_data(self):
        """获取股票数据"""
        return asyncio.run(self.async_get_stock_data())
    
class StockInfoProcess(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data.copy()
        self._original_data = data.copy()
        self._roles = {Qt.UserRole + i + 1: col.encode('utf-8') for i, col in enumerate(self._data.columns)}

    def rowCount(self, parent=QModelIndex()):
        """返回数据行数"""
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        """返回数据列数"""
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """获取指定索引的数据"""
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        
        # 显示角色处理
        if role == Qt.DisplayRole:
            value = self._data.iloc[row, col]
            if isinstance(value, float):
                return f"{value:.2f}"  # 浮点数保留2位小数
            elif isinstance(value, int):
                return f"{value:,}"   # 整数添加千位分隔符
            return str(value)
        
        # 自定义角色处理
        if role >= Qt.UserRole + 1:
            column_name = self._roles.get(role, b'').decode('utf-8')
            if column_name in self._data.columns:
                value = self._data.iloc[row, self._data.columns.get_loc(column_name)]
                if isinstance(value, float):
                    return f"{value:.2f}"
                elif isinstance(value, int):
                    return f"{value:,}"
                return str(value)
        
        return None
    
    @Slot(int, result="QVariantMap")
    def get(self, row):
        """获取指定行的数据"""
        if 0 <= row < self.rowCount():
            return self._data.iloc[row].to_dict()
        return {}
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        """获取表头数据"""
        if role != Qt.DisplayRole:
            return None
        
        if orientation == Qt.Horizontal:
            return str(self._data.columns[section])  # 返回列名
        elif orientation == Qt.Vertical:
            return str(section + 1)   # 返回行号(从1开始)
        
        return None

    def roleNames(self):
        """返回角色名称字典"""
        return self._roles

    def flags(self, index):
        """设置项标志"""
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role):
        """设置数据"""
        if not index.isValid() or role != Qt.EditRole:
            return False
        
        try:
            row = index.row()
            col = index.column()
            original_value = self._data.iloc[row, col]
            
            # 根据原始值类型进行类型转换
            if isinstance(original_value, float):
                self._data.iloc[row, col] = float(value)
            elif isinstance(original_value, int):
                self._data.iloc[row, col] = int(value)
            else:
                self._data.iloc[row, col] = str(value)
            
            self.dataChanged.emit(index, index, [role])
            return True
        except ValueError:
            return False
        
    @Slot(str, bool)
    def sortByField(self, field: str, ascending: bool = True):
        """排序"""
        if field not in self._data.columns:
            return
        
        # 使用更高效的排序方法
        try:
            self.layoutAboutToBeChanged.emit()
            self._data = self._data.sort_values(
                by=field,
                ascending=ascending,
                kind='mergesort',  # 对于大数据量更稳定
                na_position='last'
            ).reset_index(drop=True)
            self.layoutChanged.emit()
        except Exception as e:
            print(f"排序错误: {str(e)}")

    @Slot(str)
    def setFilterString(self, filter_text: str):
        """过滤"""
        try:
            self.layoutAboutToBeChanged.emit()
            
            if not filter_text:
                self._data = self._original_data.copy()
            else:
                # 使用更高效的字符串操作
                mask = (
                    self._original_data['代码'].str.contains(filter_text, case=False) | 
                    self._original_data['名称'].str.contains(filter_text, case=False)
                )
                self._data = self._original_data[mask].copy()
            
            self._data.reset_index(drop=True, inplace=True)
            self.layoutChanged.emit()
        except Exception as e:
            print(f"过滤错误: {str(e)}")

class StockInfoUpdater:
    def __init__(self, stock_info_get: StockInfoGet, stock_info_process: StockInfoProcess):
        self.stock_info_get = stock_info_get
        self.stock_info_process = stock_info_process
        
        # 数据更新定时器（低频：5-10秒）
        self.fetch_timer = QTimer()
        self.fetch_timer.setTimerType(Qt.VeryCoarseTimer)
        self.fetch_timer.timeout.connect(self._safe_fetch)
        
        # 分批处理定时器（高频：300ms）
        self.batch_timer = QTimer()
        self.batch_timer.setInterval(300)
        self.batch_timer.timeout.connect(self._process_buffer)
        
        self._update_buffer = None
        self._is_fetching = False

    def _safe_fetch(self):
        """安全获取数据，避免重叠请求"""
        if not self._is_fetching:
            self._is_fetching = True
            try:
                asyncio.run(self._fetch_data())
            finally:
                self._is_fetching = False

    async def _fetch_data(self):
        """异步获取数据并存入缓冲区"""
        new_data = await self.stock_info_get.async_get_stock_data()
        if not new_data.empty:
            # 合并到缓冲区（去重）
            if self._update_buffer is not None:
                self._update_buffer = pd.concat([self._update_buffer, new_data]).drop_duplicates(subset=['代码'], keep='last')
            else:
                self._update_buffer = new_data.copy()
            
            if not self.batch_timer.isActive():
                self.batch_timer.start()

    def _process_buffer(self):
        """处理缓冲区中的分批数据"""
        if self._update_buffer is None or self._update_buffer.empty:
            self.batch_timer.stop()
            return

        batch_size = 50  # 动态批次大小可根据数据量调整
        update_rows = self._update_buffer.iloc[:batch_size]
        self._update_buffer = self._update_buffer.iloc[batch_size:] if len(self._update_buffer) > batch_size else None

        # 高效更新数据到模型
        if not update_rows.empty:
            update_rows = update_rows.set_index('代码')
            mask = self.stock_info_process._data['代码'].isin(update_rows.index)
            
            for col in update_rows.columns:
                if col in self.stock_info_process._data.columns:
                    updates = update_rows[col].to_dict()
                    self.stock_info_process._data.loc[mask, col] = self.stock_info_process._data.loc[mask, '代码'].map(updates)
            
            # 通知UI更新
            changed_indices = self.stock_info_process._data.index[mask]
            if not changed_indices.empty:
                top_left = self.stock_info_process.index(changed_indices.min(), 0)
                bottom_right = self.stock_info_process.index(changed_indices.max(), len(self.stock_info_process._data.columns) - 1)
                self.stock_info_process.dataChanged.emit(top_left, bottom_right)

    def start(self, fetch_interval: int = 5000):
        """启动定时器"""
        self.fetch_timer.start(fetch_interval) 
        
    def stop(self):
        """停止所有定时器"""
        self.fetch_timer.stop()
        self.batch_timer.stop()
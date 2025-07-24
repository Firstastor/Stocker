"""
新浪财经-沪深 A 股实时行情和历史K线数据获取
"""
import aiohttp, asyncio
import json
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
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

    fetch_stock_data = Signal(list)
    fetch_history_stock_data = Signal(list)

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

    @Slot(str, result='QVariantList')
    def filter_stock_data(self, search_text):
        """过滤数据"""
        if not self.stock_data:
            return []
        
        if not search_text:
            return self.stock_data
        
        try:
            search_lower = search_text.lower()
            filtered = [
                item for item in self.stock_data
                if (search_lower in item["代码"].lower() or 
                    search_lower in item["名称"].lower())
            ]
            return filtered
        except Exception as e:
            print(f"Filtering error: {str(e)}")
            return self.stock_data
     
    async def fetch_stock_data(self, session, stock_codes):
        """获取单支股票实时数据"""
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

    def parse_stock_data(self, response_text):
        """解析单支股票实时数据"""
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
        """获取所有股票数据"""
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
        return all_results
     
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
    
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class StockPrediction(QObject):
    """股票预测类（集成LightGBM和LSTM模型）"""
    training_progress = Signal(int, str)  # 训练进度信号（进度百分比, 状态消息）

    def __init__(self, parent=None):
        super().__init__(parent)
        # LightGBM相关
        self.lgb_model = None
        self.lgb_scaler = MinMaxScaler()
        
        # LSTM相关
        self.lstm_model = None
        self.lstm_scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== LightGBM 部分 ====================
    def train_lgbm(self, X_train, y_train, params=None):
        """训练LightGBM模型"""
        try:
            # 数据标准化
            X_train_scaled = self.lgb_scaler.fit_transform(X_train)
            
            # 默认参数（针对股票预测优化）
            if params is None:
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'num_leaves': 63,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
            
            # 训练模型
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            self.lgb_model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                callbacks=[self._lgb_callback]
            )
            return True
        except Exception as e:
            self.training_progress.emit(0, f"LightGBM训练错误: {str(e)}")
            return False

    def _lgb_callback(self, env):
        """LightGBM训练进度回调"""
        progress = int(env.iteration / env.end_iteration * 100)
        self.training_progress.emit(progress, "LightGBM训练中...")

    # ==================== LSTM 部分 ====================
    class LSTMModel(nn.Module):
        """PyTorch LSTM模型定义"""
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    def train_lstm(self, X_train, y_train, epochs=100, seq_length=30):
        """训练LSTM模型"""
        try:
            # 数据预处理
            X_train_scaled = self.lstm_scaler.fit_transform(X_train)
            y_train = y_train.reshape(-1, 1)
            
            # 转换为序列数据
            X, y = self._create_sequences(X_train_scaled, y_train, seq_length)
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # 初始化模型
            self.lstm_model = self.LSTMModel(
                input_size=X_train.shape[1],
                hidden_size=64,
                num_layers=2
            ).to(self.device)
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            
            # 训练循环
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.lstm_model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                # 发射训练进度信号
                progress = int((epoch + 1) / epochs * 100)
                self.training_progress.emit(
                    progress, 
                    f"LSTM训练中 (Epoch {epoch+1}/{epochs}) Loss: {loss.item():.4f}"
                )
            return True
        except Exception as e:
            self.training_progress.emit(0, f"LSTM训练错误: {str(e)}")
            return False

    def _create_sequences(self, data, targets, seq_length):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(targets[i+seq_length])
        return np.array(X), np.array(y)

    def predict_lstm(self, X_test, seq_length=30):
        """使用LSTM进行预测"""
        if self.lstm_model is None:
            raise ValueError("LSTM模型未训练！")
            
        # 数据预处理
        X_test_scaled = self.lstm_scaler.transform(X_test)
        
        # 转换为序列数据
        X_seq = []
        for i in range(len(X_test_scaled) - seq_length + 1):
            X_seq.append(X_test_scaled[i:i+seq_length])
        X_seq = np.array(X_seq)
        
        # 预测
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            preds = self.lstm_model(X_tensor).cpu().numpy()
        
        return preds.flatten()

    # ==================== 通用功能 ====================
    def save_model(self, model_type, filepath):
        """保存指定类型的模型"""
        if model_type == "lgbm":
            if self.lgb_model:
                self.lgb_model.save_model(filepath)
            else:
                raise ValueError("LightGBM模型未训练")
        elif model_type == "lstm":
            if self.lstm_model:
                torch.save({
                    'model_state_dict': self.lstm_model.state_dict(),
                    'scaler_params': {
                        'scale_': self.lstm_scaler.scale_,
                        'min_': self.lstm_scaler.min_
                    }
                }, filepath)
            else:
                raise ValueError("LSTM模型未训练")
        else:
            raise ValueError("不支持的模型类型")

    def load_model(self, model_type, filepath):
        """加载预训练模型"""
        if model_type == "lgbm":
            self.lgb_model = lgb.Booster(model_file=filepath)
        elif model_type == "lstm":
            checkpoint = torch.load(filepath)
            self.lstm_model = self.LSTMModel().to(self.device)
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.lstm_scaler.scale_ = checkpoint['scaler_params']['scale_']
            self.lstm_scaler.min_ = checkpoint['scaler_params']['min_']
        else:
            raise ValueError("不支持的模型类型")
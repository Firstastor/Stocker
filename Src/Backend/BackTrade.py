import aiohttp
import asyncio
import json
from lightgbm import LGBMRegressor
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import talib
from typing import Dict, List, Callable, Optional
from xgboost import XGBRegressor

class BackTrade:
    def __init__(self, df, entries=None, exits=None, init_cash=100000, fee=0.0002, slippage=0.0025):
        """
        回测交易系统
        参数:
        - df: 包含Date/open/high/low/close/volume的Polars DataFrame
        - entries: 买入信号 (Polars Series)
        - exits: 卖出信号 (Polars Series)
        - init_cash: 初始资金
        - fee: 交易手续费率
        - slippage: 滑点率
        """
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame必须包含这些列: {required_cols}")
        self.data = df.with_columns(pl.col('Date').str.strptime(pl.Date)).sort('Date')
        self.entries = entries.fill_null(False) if entries is not None else None
        self.exits = exits.fill_null(False) if exits is not None else None
        self.init_cash = init_cash
        self.fee = fee
        self.slip = slippage
        self.results = None
        self.trades = []
        self.stats = {}

    def _calculate_stats(self):
        equity = self.results['Equity']
        returns = equity.pct_change().fill_null(0)
        peak = equity.cum_max()
        drawdown = (peak - equity) / peak
        final_equity = equity[-1] if len(equity) > 0 else self.init_cash
        total_return = (final_equity / self.init_cash - 1) * 100
        annualized_return = (final_equity / self.init_cash) ** (252/len(equity)) - 1 if len(equity) > 0 else 0.0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
        max_dd = drawdown.max() * 100 if drawdown.max() is not None else 0.0
        stats = {
            'InitialCapital': self.init_cash,
            'FinalEquity': final_equity,
            'TotalReturn': total_return,
            'AnnualizedReturn': annualized_return,
            'Volatility': volatility,
            'MaxDrawdown': max_dd,
            'NumTrades': len(self.trades),
        }
        if len(self.trades) > 0:
            trades_df = pl.DataFrame(self.trades)
            winning_trades = trades_df.filter(pl.col('PnL') > 0)
            losing_trades = trades_df.filter(pl.col('PnL') < 0)
            stats.update({
                'WinRate': winning_trades.height / len(self.trades) * 100,
                'AvgReturn': trades_df['Return'].mean(),
                'AvgWinningReturn': winning_trades['Return'].mean() if winning_trades.height > 0 else 0,
                'AvgLosingReturn': losing_trades['Return'].mean() if losing_trades.height > 0 else 0,
                'ProfitFactor': abs(winning_trades['PnL'].sum() / losing_trades['PnL'].sum())
                    if losing_trades.height > 0 else float('inf'),
                'AvgTradeDuration': (trades_df['ExitDate'] - trades_df['EntryDate']).mean().total_seconds() / (60*60*24)
                    if trades_df.height > 0 else 0
            })
        downside_returns = returns.filter(returns < 0)
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        stats.update({
            'SharpeRatio': annualized_return / volatility if volatility > 0 else 0.0,
            'SortinoRatio': annualized_return / downside_volatility if downside_volatility > 0 else 0.0,
            'CalmarRatio': annualized_return / (max_dd / 100) if max_dd > 0 else 0.0,
        })
        self.stats = stats

    def run(self, size=1, sl=None, tp=None):
        """
        回测主逻辑
        参数:
        - size: 仓位比例
        - sl: 止损
        - tp: 止盈
        """
        if self.entries is None or self.exits is None:
            raise ValueError("必须提供entries和exits信号")
        dates = self.data['Date']
        price = self.data['Close']
        entries = self.entries
        exits = self.exits
        n = len(price)
        cash = np.full(n, self.init_cash, dtype=np.float64)
        position = np.zeros(n)
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss_price = 0.0
        take_profit_price = 0.0
        for i in range(n):
            current_price = price[i]
            sl_triggered = in_position and sl and (current_price <= stop_loss_price)
            tp_triggered = in_position and tp and (current_price >= take_profit_price)
            if in_position and (exits[i] or sl_triggered or tp_triggered or i == n-1):
                if sl_triggered:
                    exit_price = stop_loss_price * (1 - self.slip)
                    exit_type = 'stop_loss'
                elif tp_triggered:
                    exit_price = take_profit_price * (1 - self.slip)
                    exit_type = 'take_profit'
                else:
                    exit_price = current_price * (1 - self.slip)
                    exit_type = 'signal'
                pnl = position[i-1] * (exit_price - entry_price)
                fee = (position[i-1] * exit_price) * self.fee
                cash[i:] = round(cash[i-1] + (position[i-1] * exit_price - fee), 2)
                position[i:] = 0
                in_position = False
                self.trades.append({
                    'EntryDate': dates[entry_idx],
                    'ExitDate': dates[i],
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'Shares': int(position[i-1]),
                    'PnL': pnl,
                    'Return': round((exit_price / entry_price - 1) * 100, 4),
                    'Type': exit_type,
                })
            elif not in_position and entries[i]:
                entry_price = round(current_price * (1 + self.slip), 3)
                max_shares = int(((cash[i] * size) // (entry_price * (1 + self.fee))) // 100 * 100)
                if max_shares > 0:
                    fee = max_shares * entry_price * self.fee
                    cash[i:] = round(cash[i] - (max_shares * entry_price + fee), 2)
                    position[i:] = max_shares
                    in_position = True
                    entry_idx = i
                    stop_loss_price = entry_price * (1 - sl) if sl else 0.0
                    take_profit_price = entry_price * (1 + tp) if tp else 0.0
        equity = cash + position * price
        self.results = pl.DataFrame({
            'Date': dates,
            'Equity': equity,
            'Cash': cash,
            'Position': position,
            'Price': price
        })
        self._calculate_stats()
        return self

    def summary(self):
        """打印绩效摘要"""
        if not self.stats:
            raise ValueError("请先运行回测")
        print("回测结果摘要:")
        print(f"初始资金: {self.stats['InitialCapital']:.2f}")
        print(f"最终权益: {self.stats['FinalEquity']:.2f}")
        print(f"总收益率: {self.stats['TotalReturn']:.2f}%")
        print(f"年化收益率: {self.stats['AnnualizedReturn']*100:.2f}%")
        print(f"波动率: {self.stats['Volatility']:.4f}")
        print(f"最大回撤: {self.stats['MaxDrawdown']:.2f}%")
        if self.stats['NumTrades'] > 0:
            print("\n交易统计:")
            print(f"交易次数: {self.stats['NumTrades']}")
            print(f"胜率: {self.stats['WinRate']:.2f}%")
            print(f"平均收益率: {self.stats['AvgReturn']:.2f}%")
            print(f"平均盈利: {self.stats['AvgWinningReturn']:.2f}%") 
            print(f"平均亏损: {self.stats['AvgLosingReturn']:.2f}%") 
            print(f"盈亏比: {self.stats['ProfitFactor']:.2f}")
            print(f"平均持仓天数: {self.stats['AvgTradeDuration']:.1f}") 
        print("\n风险调整收益指标:")
        print(f"夏普比率: {self.stats['SharpeRatio']:.2f}")
        print(f"索提诺比率: {self.stats['SortinoRatio']:.2f}")
        print(f"卡玛比率: {self.stats['CalmarRatio']:.2f}")

    def plot(self, show_drawdown=True):
        """
        使用Plotly绘制K线+买卖信号+净值曲线
        """
        if self.results is None:
            raise ValueError("请先运行回测")
        df = self.data
        res = self.results
        trades = pl.DataFrame(self.trades) if self.trades else None
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("K线 & 交易信号", "策略净值")
        )
        candle = go.Candlestick(
            x=df['Date'].to_numpy(),
            open=df['Open'].to_numpy(),
            high=df['High'].to_numpy(),
            low=df['Low'].to_numpy(),
            close=df['Close'].to_numpy(),
            name='K线',
            increasing_line_color='#089981',
            decreasing_line_color='#f23645'
        )
        fig.add_trace(candle, row=1, col=1)
        if trades is not None and trades.height > 0:
            fig.add_trace(go.Scatter(
                x=trades['EntryDate'].to_numpy(),
                y=trades['EntryPrice'].to_numpy(),
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#00b746'),
                name='买入'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=trades['ExitDate'].to_numpy(),
                y=trades['ExitPrice'].to_numpy(),
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='#ff2e4d'),
                name='卖出'
            ), row=1, col=1)
        equity = res['Equity'].to_numpy()
        dates = res['Date'].to_numpy()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            line=dict(color='#2962FF', width=2),
            name='策略净值'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=dates,
            y=[self.init_cash] * len(equity),
            mode='lines',
            line=dict(color='#78909C', width=1, dash='dash'),
            name='初始资金'
        ), row=2, col=1)
        if show_drawdown and 'MaxDrawdown' in self.stats:
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_dd_idx = np.argmax(drawdown)
            peak_before = np.argmax(equity[:max_dd_idx]) if max_dd_idx > 0 else 0

            fig.add_vrect(
                x0=dates[peak_before],
                x1=dates[max_dd_idx],
                fillcolor="rgba(244, 67, 54, 0.2)",
                layer="below",
                line_width=0,
                row=2, col=1
            )
            fig.add_annotation(
                x=dates[max_dd_idx],
                y=equity[max_dd_idx],
                text=f"最大回撤: {self.stats['MaxDrawdown']:.2f}%",
                showarrow=True,
                arrowhead=1,
                font=dict(color='#ff5252'),
                row=2, col=1
            )
        fig.update_layout(
            title=dict(text='回测结果', x=0.5, xanchor='center'),
            template='plotly_dark',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            hovermode='x unified'
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text='价格', row=1, col=1)
        fig.update_yaxes(title_text='净值', row=2, col=1)
        return fig.show()

class BackTradeData:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn/"
        }
        self.timeout = aiohttp.ClientTimeout(total=10)
        self.stock_data = None

    @staticmethod
    def _parse_kline(response_text, stock_symbol):
        try:
            data = json.loads(response_text)
            if not data:
                print(f"{stock_symbol} is empty")
                return pl.DataFrame()
            df = pl.DataFrame(data)
            df = df.with_columns(pl.lit(stock_symbol).alias("Symbol"))
            mapping = {
                "day": "Date",
                "open": "Open",
                "close": "Close",
                "high": "High",
                "low": "Low",
                "volume": "Volume"
            }
            for old, new in mapping.items():
                if old in df.columns:
                    df = df.rename({old: new})
            for col in ['ma_price5', 'ma_volume5', 'ma_price10', 'ma_volume10', 'ma_price30', 'ma_volume30']:
                if col in df.columns:
                    df = df.drop(col)
            if df.height > 0:
                os.makedirs("Data", exist_ok=True)
                df.write_csv(f"Data/Stock_History_Data_{stock_symbol}.csv")
                print(f"{stock_symbol} is downloaded")
            else:
                print(f"{stock_symbol} is empty")
            return df
        except Exception as e:
            print(f"解析数据失败: {e}")
            return pl.DataFrame()

    async def fetch_stock_list(self, total_pages=55, batch_size=10):
        stocks = []
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
                            return [{
                                "代码": item.get("symbol"),
                                "名称": item.get("name"),
                            } for item in data] if data else []
                        else:
                            print(f"第 {page} 页请求失败，状态码: {response.status}")
                except Exception as e:
                    print(f"第 {page} 页请求异常: {str(e)}")
                return []

            for batch_start in range(1, total_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size, total_pages + 1)
                tasks = [fetch_page(page) for page in range(batch_start, batch_end)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        print(f"批量请求失败: {result}")
                        continue
                    stocks.extend(result)
        stocks_df = pl.DataFrame(stocks)
        print(f"总共获取 {stocks_df.height} 条股票数据")
        return stocks_df

    async def fetch_kline(self, session, stock_symbol, scale=240, datalen=3650):
        params = {
            "symbol": stock_symbol,
            "scale": scale,
            "datalen": datalen,
            "fq": 1
        }
        url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
        try:
            async with session.get(url, params=params, timeout=self.timeout, headers=self.headers, ssl=True) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_kline(text, stock_symbol)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"{stock_symbol} 请求历史数据失败: {str(e)}")
        return pl.DataFrame()

    async def fetch_and_save_all_kline(self, scale=240, datalen=3650, batch_size=20):
        stocks_df = await self.fetch_stock_list()
        all_stocks = list(stocks_df.iter_rows(named=True))
        results = []
        
        async with aiohttp.ClientSession(timeout=self.timeout, headers=self.headers) as session:
            for i in range(0, len(all_stocks), batch_size):
                batch = all_stocks[i:i + batch_size]
                
                # 处理当前批次
                tasks = [self.fetch_kline(session, row["代码"], scale, datalen) for row in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
                
                # 如果不是最后一批，添加延迟
                if i + batch_size < len(all_stocks):
                    await asyncio.sleep(0.1 * batch_size)  # 每批总延迟 = 0.5s × 批大小
        
        # 过滤有效结果
        dfs = [df for df in results if isinstance(df, pl.DataFrame) and df.height > 0]
        return dfs

    def download_all_stocks(self, scale=240, datalen=3650):
        asyncio.run(self.fetch_and_save_all_kline(scale, datalen))

    def download_single_stock(self, stock_symbol, scale=240, datalen=3650):
        async def runner():
            async with aiohttp.ClientSession() as session:
                await self.fetch_kline(session, stock_symbol, scale, datalen)
        asyncio.run(runner())

    def get_stock_list(self):
        self.stock_data = asyncio.run(self.fetch_stock_list())
        return self.stock_data

class BackTradeEvent:
    def __init__(self, init_cash: float = 100000, fee: float = 0.0002, 
                 slippage: float = 0.0025, start_date: str = '2019-12-31', 
                 end_date: Optional[str] = None):
        """
        初始化回测环境 - 递推优化版
        
        参数:
            init_cash: 初始资金
            fee: 交易手续费率
            slippage: 交易滑点率
            start_date: 回测开始日期
            end_date: 回测结束日期(可选)
        """
        self._load_initial_data(start_date, end_date)
            
        self.fee = fee
        self.slip = slippage
        self.trade_history = []
        self.positions: Dict[str, dict] = {} 
        self.current_date = start_date
        
        # 缓存股票数据 {symbol: df}
        self._data_cache: Dict[str, pl.DataFrame] = {}
        
        # 预计算所有日期并创建索引
        self.all_dates = self.df["Date"].to_list()
        self.date_index = {date: idx for idx, date in enumerate(self.all_dates)}
        self.init_cash = init_cash
        # 初始化每日数据存储
        self.daily_data = {
            'cash': [init_cash] * len(self.all_dates),
            'equity': [init_cash] * len(self.all_dates)
        }

    def _load_initial_data(self, start_date: str, end_date: Optional[str]):
        """加载初始日期数据"""
        test = pl.scan_csv('Data/Stock_History_Data_sz000001.csv')
        
        if end_date is None:
            self.df = (test.select('Date')
                       .filter(pl.col("Date") >= start_date)
                       .collect())
        elif end_date > start_date:
            self.end_date = end_date
            self.df = (test.select('Date')
                       .filter((pl.col("Date") >= start_date) & (pl.col("Date") < end_date))
                       .collect())
        else:
            raise ValueError("结束日期必须晚于开始日期")

    def _get_results(self) -> dict:
        """获取回测结果"""
        final_equity = self.daily_data['equity'][-1]
        return {
            '初始资金': self.init_cash,
            '最终资产': final_equity,
            '总收益率': f"{(final_equity / self.init_cash - 1) * 100:.2f}%",
            '交易次数': len(self.trade_history),
            '交易记录': self.trade_history,
            '期末持仓': self.positions
        }
    
    def _get_stock_data(self, symbol: str) -> pl.DataFrame:
        """获取股票数据(带缓存)"""
        if symbol not in self._data_cache:
            try:
                self._data_cache[symbol] = pl.read_csv(f'Data/Stock_History_Data_{symbol}.csv')
            except Exception as e:
                print(f"无法加载 {symbol} 的数据: {str(e)}")
                return pl.DataFrame()
        return self._data_cache[symbol]

    def _process_buy_order(self, date: str, symbol: str, price: float, prev_cash: float, size: float, current_idx: int):
        """处理买入订单"""
        entry_price = round(price * (1 + self.slip), 3)
        max_shares = (prev_cash * size) // (entry_price * (1 + self.fee))
        shares = int(max_shares // 100 * 100)
        
        if shares > 0:
            cost = shares * entry_price
            fee = cost * self.fee
            self.positions[symbol] = {
                'entry_date': date,
                'shares': shares,
                'entry_price': entry_price
            }
            self._update_cash(current_idx, -(cost + fee))
            print(f"{date} 买入 {symbol} {shares}股 @ {entry_price}")

    def _process_sell_order(self, date: str, symbol: str, price: float, current_idx: int):
        """处理卖出订单"""
        position = self.positions[symbol]
        exit_price = round(price * (1 - self.slip), 3)
        proceeds = position['shares'] * exit_price
        fee = proceeds * self.fee
        pnl = position['shares'] * (exit_price - position['entry_price'])
        
        self._update_cash(current_idx, proceeds - fee)

        self.trade_history.append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'pnl': pnl,
            'return_pct': round((exit_price / position['entry_price'] - 1) * 100, 4)
        })
        
        print(f"{date} 卖出 {symbol} {position['shares']}股 @ {exit_price}, 盈利: {pnl:.2f}")
        del self.positions[symbol]

    def _update_cash(self, current_idx: int, amount: float) -> None:
        """更新现金余额"""
        prev_cash = self.daily_data['cash'][current_idx-1] if current_idx > 0 else self.daily_data['cash'][0]
        self.daily_data['cash'][current_idx] = prev_cash + amount
        self._update_equity(current_idx)

    def _update_equity(self, current_idx: int) -> None:
        """更新总资产"""
        position_value = 0.0
        date = self.all_dates[current_idx]
        
        for symbol, pos in self.positions.items():
            df = self._get_stock_data(symbol)
            if not df.is_empty():
                row = df.filter(pl.col("Date") == date)
                if not row.is_empty():
                    position_value += pos['shares'] * row['Close'][0]
        self.daily_data['equity'][current_idx] = self.daily_data['cash'][current_idx] + position_value

    def order(self, date: str, symbol: str, action: str, size: float = 1.0) -> None:
        """
        执行交易订单 - 递推优化版
        
        参数:
            date: 交易日期
            symbol: 股票代码
            action: 'b'表示买入, 's'表示卖出
            size: 仓位大小(买入时为可用资金的比例)
        """
        try:
            df = self._get_stock_data(symbol)
            if df.is_empty():
                return
            
            row = df.filter(pl.col("Date") == date)
            if row.is_empty():
                return
            
            price = row['Close'][0]
            current_idx = self.date_index[date]
            prev_cash = self.daily_data['cash'][current_idx-1] if current_idx > 0 else self.daily_data['cash'][0]

            if action == 'b' and prev_cash > 0:
                self._process_buy_order(date, symbol, price, prev_cash, size, current_idx)
            elif action == 's' and symbol in self.positions:
                self._process_sell_order(date, symbol, price, current_idx)      

        except Exception as e:
            print(f"{date} {symbol}交易出错: {str(e)}")

    def run(self, 
            entry_strategy: Callable[[str], List[str]], 
            exit_strategy: Callable[[str], List[str]], 
            position_size: float = 1.0) -> None:
        """
        运行回测
        
        参数:
            entry_strategy: 入场策略函数，接收日期，返回要买入的股票列表
            exit_strategy: 出场策略函数，接收日期和当前持仓，返回要卖出的股票列表
            position_size: 每笔交易使用资金比例(0-1)
        """
        for i, date in enumerate(self.all_dates):
            self.current_date = date
            trade_flag = 1

            positions_to_exit = exit_strategy(date)
            for symbol in positions_to_exit:
                if symbol in self.positions:
                    self.order(date, symbol, 's')
                    trade_flag = 0

            symbols_to_buy = entry_strategy(date)
            for symbol in symbols_to_buy:
                if symbol not in self.positions:
                    self.order(date, symbol, 'b', position_size)
                    trade_flag = 0
            
            if trade_flag:
                self._update_cash(i,0)

    def summary(self) -> None:
        """打印回测摘要"""
        results = self._get_results()
        print("\n========== 回测结果 ==========")
        print(f"初始资金: {results['初始资金']:.2f}")
        print(f"最终资产: {results['最终资产']:.2f}")
        print(f"总收益率: {results['总收益率']}")
        print(f"交易次数: {results['交易次数']}")
        
        if results['交易次数'] > 0:
            avg_return = sum(t['return_pct'] for t in results['交易记录']) / results['交易次数']
            win_rate = sum(1 for t in results['交易记录'] if t['pnl'] > 0) / results['交易次数']
            print(f"平均收益率: {avg_return:.2f}%")
            print(f"胜率: {win_rate:.2%}")
               
class BackTradeModel:
    def __init__(self, df, entries_return=0.03, exits_return=0.01):
        df = pl.DataFrame()
        self.df = df.rename({
            'open': 'Open', 'o': 'Open', 'price_open': 'Open',
            'high': 'High', 'h': 'High', 'price_high': 'High',
            'low': 'Low', 'l': 'Low', 'price_low': 'Low',
            'close': 'Close', 'c': 'Close', 'price_close': 'Close',
            'volume': 'Volume', 'v': 'Volume', 'vol': 'Volume'
        }).drop_nulls()

        h, l, c, v = self.df["High"].to_numpy().astype(np.float64), self.df["Low"].to_numpy().astype(np.float64), self.df["Close"].to_numpy().astype(np.float64), self.df["Volume"].to_numpy().astype(np.float64)
        bb_u, bb_m, bb_l = talib.BBANDS(c, 20, 2, 2)
        fk, fd = talib.STOCHF(h, l, c, 14, 3)
        macd, macds, macdh = talib.MACD(c, 12, 26, 9)

        self.df = self.df.with_columns([
            pl.col("Close").pct_change().alias("Return"),
            pl.Series(talib.ADX(h, l, c, 14)).alias("ADX"),
            pl.Series(talib.ADXR(h, l, c, 14)).alias("ADXR"),
            pl.Series(talib.APO(c, 12, 26)).alias("APO"),
            pl.Series(talib.ATR(h, l, c, 14)).alias("ATR"),
            pl.Series(bb_u).alias("BB_Upper"),
            pl.Series(bb_m).alias("BB_Middle"),
            pl.Series(bb_l).alias("BB_Lower"),
            pl.Series(talib.CCI(h, l, c, 14)).alias("CCI"),
            pl.Series(fk).alias("FastStochK"),
            pl.Series(fd).alias("FastStochD"),
            pl.Series(talib.MA(c, 5)).alias("MA_5"),
            pl.Series(talib.MA(c, 10)).alias("MA_10"),
            pl.Series(macd).alias("MACD"),
            pl.Series(macds).alias("MACD_Signal"),
            pl.Series(macdh).alias("MACD_Hist"),
            pl.Series(talib.MOM(c, 10)).alias("MOM"),
            pl.Series(talib.NATR(h, l, c, 14)).alias("NATR"),
            pl.Series(talib.OBV(c, v)).alias("OBV"),
            pl.Series(talib.PPO(c, 12, 26)).alias("PPO"),
            pl.Series(talib.ROC(c, 10)).alias("ROC"),
            pl.Series(talib.RSI(c, 14)).alias("RSI"),
            pl.Series(talib.SAR(h, l, 0.02, 0.2)).alias("SAR"),
            pl.Series(talib.WILLR(h, l, c, 14)).alias("WilliamsR"),
            pl.col("Close").pct_change().shift(-1).alias("Future_Return")
        ])
        self.X = self.df.drop("Future_Return").drop_nans()
        self.X_pred = self.df.tail(1).drop("Future_Return")
        self.entries_return =entries_return
        self.exits_return =exits_return

    def _train_model(self, model_class):
        X = self.df.drop("Future_Return").to_numpy()
        y = self.df["Future_Return"].to_numpy()
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.99)),
            ('model', model_class())
        ])
        model.fit(X, y)
        
        y_pred = model.predict(self.X_pred.to_numpy())
        entries = y_pred > self.entries_return
        exits = y_pred < self.exits_return
        return entries, exits
        
    def GB(self):
        return self._train_model(GradientBoostingRegressor)
        
    def HGB(self):
        return self._train_model(HistGradientBoostingRegressor)
        
    def LGBM(self):
        return self._train_model(LGBMRegressor)
        
    def RF(self):
        return self._train_model(RandomForestRegressor)
        
    def XGB(self):
        return self._train_model(XGBRegressor)  

class BackTradeStrategy:
    def __init__(self, df: pl.DataFrame):
        """
        技术指标策略生成器
        参数：
            df: Polars DataFrame,包含OHLCV数据
        """
        self.df = df

    def adx(self, flag, type='trend', period=14, threshold=25):
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        close = self.df['Close'].to_numpy()

        adx = talib.ADX(high, low, close, timeperiod=period)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)

        if type == 'trend':
            entries = adx > threshold
            exits = adx > threshold
        elif type == 'cross':
            entries = (np.roll(plus_di, 1) > np.roll(minus_di, 1)) & (np.roll(plus_di, 2) <= np.roll(minus_di, 2))
            exits = (np.roll(minus_di, 1) > np.roll(plus_di, 1)) & (np.roll(minus_di, 2) <= np.roll(plus_di, 2))
        elif type == 'extreme':
            entries = (adx > 40) & (np.roll(adx, 1) > np.roll(adx, 2)) & (plus_di < minus_di)
            exits = (adx > 40) & (np.roll(adx, 1) > np.roll(adx, 2)) & (plus_di > minus_di)
        signals = entries if flag else exits
        return pl.Series('adx_signal', signals)

    def bband(self, flag, type='cross', timeperiod=20, nbdev=2, bandwidth_period=50):
        close = self.df['Close'].to_numpy()
        volume = self.df['Volume'].to_numpy()

        upper, middle, lower = talib.BBANDS(
            close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev
        )
        bandwidth = upper - lower
        pct_bandwidth = bandwidth / middle

        if type == 'cross':
            entries = (np.roll(close, 1) > np.roll(middle, 1)) & (np.roll(close, 2) <= np.roll(middle, 2))
            exits = (np.roll(close, 1) < np.roll(middle, 1)) & (np.roll(close, 2) >= np.roll(middle, 2))
        elif type == 'band':
            entries = (np.roll(close, 1) <= np.roll(lower, 1)) & (np.roll(close, 2) > np.roll(lower, 2))
            exits = (np.roll(close, 1) >= np.roll(upper, 1)) & (np.roll(close, 2) < np.roll(upper, 2))
        elif type == 'width':
            bandwidth_ma = pl.Series(pct_bandwidth).rolling_mean(bandwidth_period).to_numpy()
            entries = (np.roll(pct_bandwidth, 1) < np.roll(bandwidth_ma, 1)) & (close > middle)
            exits = close < middle
        elif type == 'squeeze':
            kc_middle = talib.SMA(close, 20)
            atr = talib.ATR(self.df['High'].to_numpy(), self.df['Low'].to_numpy(), close, 20)
            kc_upper = kc_middle + 1.5 * atr
            kc_lower = kc_middle - 1.5 * atr
            squeeze_on = (upper < kc_upper) & (lower > kc_lower)
            entries = (
                np.roll(squeeze_on, 1) & ~squeeze_on &
                (close > middle) &
                (volume > pl.Series(volume).rolling_mean(20).to_numpy())
            )
            exits = close < middle
        elif type == 'trend':
            entries = (np.roll(close, 1) > np.roll(upper, 1)) & (np.roll(close, 2) <= np.roll(upper, 2))
            exits = (np.roll(close, 1) < np.roll(lower, 1)) & (np.roll(close, 2) >= np.roll(lower, 2))
        signals = entries if flag else exits
        return pl.Series('bband_signal', signals)

    def cci(self, flag, type='break', period=14, overbought=100, oversold=-100, divergence_lookback=10):
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        close = self.df['Close'].to_numpy()
        cci = talib.CCI(high, low, close, timeperiod=period)

        if type == 'break':
            entries = (np.roll(cci, 1) < oversold) & (np.roll(cci, 2) >= oversold) & (np.roll(close, 1) > np.roll(close, 2))
            exits = (np.roll(cci, 1) > overbought) & (np.roll(cci, 2) <= overbought) & (np.roll(close, 1) < np.roll(close, 2))
        elif type == 'cross':
            entries = (np.roll(cci, 1) > 0) & (np.roll(cci, 2) <= 0) & (cci > np.roll(cci, 1))
            exits = (np.roll(cci, 1) < 0) & (np.roll(cci, 2) >= 0) & (cci < np.roll(cci, 1))
        elif type == 'divergence':
            if flag:
                price_lows = pl.Series(low).rolling_min(divergence_lookback).to_numpy()
                cci_lows = pl.Series(cci).rolling_min(divergence_lookback).to_numpy()
                entries = (low == price_lows) & (cci > cci_lows) & (np.roll(cci, 1) > np.roll(cci, 2)) & (cci < -50)
                signals = entries
            else:
                price_highs = pl.Series(high).rolling_max(divergence_lookback).to_numpy()
                cci_highs = pl.Series(cci).rolling_max(divergence_lookback).to_numpy()
                exits = (high == price_highs) & (cci < cci_highs) & (np.roll(cci, 1) < np.roll(cci, 2)) & (cci > 50)
                signals = exits
            return pl.Series('cci_signal', signals if signals is not None else np.zeros(len(cci), dtype=bool))
        elif type == 'swing':
            entries = (np.roll(cci, 2) < oversold) & (np.roll(cci, 1) > oversold) & (cci > np.roll(cci, 1)) & (close > np.roll(close, 1))
            exits = (np.roll(cci, 2) > overbought) & (np.roll(cci, 1) < overbought) & (cci < np.roll(cci, 1)) & (close < np.roll(close, 1))
        elif type == 'trend':
            cci_ma = pl.Series(cci).rolling_mean(5).to_numpy()
            entries = (cci > 0) & (cci > cci_ma) & (np.roll(cci, 1) > np.roll(cci, 2)) & (close > np.roll(close, 5))
            exits = (cci < 0) & (cci < cci_ma) & (np.roll(cci, 1) < np.roll(cci, 2)) & (close < np.roll(close, 5))
        signals = entries if flag else exits
        return pl.Series('cci_signal', signals)

    def cmo(self, flag, period=14, overbought=50, oversold=-50):
        close = self.df['Close'].to_numpy()
        cmo = talib.CMO(close, timeperiod=period)
        if flag:
            signals = (np.roll(cmo, 1) < oversold) & (cmo > np.roll(cmo, 1)) & (cmo > oversold)
        else:
            signals = (np.roll(cmo, 1) > overbought) & (cmo < np.roll(cmo, 1)) & (cmo < overbought)
        return pl.Series('cmo_signal', signals)

    def kdj(self, flag, type='cross', k_period=9, d_period=3, overbought=80, oversold=20, divergence_lookback=10):
        low = self.df['Low'].to_numpy()
        high = self.df['High'].to_numpy()
        close = self.df['Close'].to_numpy()
        lowest_low = pl.Series(low).rolling_min(k_period).to_numpy()
        highest_high = pl.Series(high).rolling_max(k_period).to_numpy()
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        K = pl.Series(rsv).ewm_mean(alpha=1/d_period, adjust=False).to_numpy()
        D = pl.Series(K).ewm_mean(alpha=1/d_period, adjust=False).to_numpy()
        J = 3 * K - 2 * D

        if type == 'cross':
            entries = (np.roll(K, 1) > np.roll(D, 1)) & (np.roll(K, 2) <= np.roll(D, 2)) & (np.roll(K, 1) < 50)
            exits = (np.roll(K, 1) < np.roll(D, 1)) & (np.roll(K, 2) >= np.roll(D, 2)) & (np.roll(K, 1) > 50)
        elif type == 'break':
            entries = (np.roll(K, 1) < oversold) & (np.roll(K, 2) >= oversold) & (np.roll(close, 1) > np.roll(close, 2))
            exits = (np.roll(K, 1) > overbought) & (np.roll(K, 2) <= overbought) & (np.roll(close, 1) < np.roll(close, 2))
        elif type == 'divergence':
            if flag:
                price_lows = pl.Series(low).rolling_min(divergence_lookback).to_numpy()
                k_lows = pl.Series(K).rolling_min(divergence_lookback).to_numpy()
                entries = (low == price_lows) & (K > k_lows) & (np.roll(K, 1) > np.roll(K, 2)) & (D < 30)
                signals = entries
            else:
                price_highs = pl.Series(high).rolling_max(divergence_lookback).to_numpy()
                k_highs = pl.Series(K).rolling_max(divergence_lookback).to_numpy()
                exits = (high == price_highs) & (K < k_highs) & (np.roll(K, 1) < np.roll(K, 2)) & (D > 70)
                signals = exits
            return pl.Series('kdj_signal', signals if signals is not None else np.zeros(len(K), dtype=bool))
        elif type == 'extreme':
            entries = (np.roll(J, 1) < 10) & (J > np.roll(J, 1)) & (close > np.roll(close, 1))
            exits = (np.roll(J, 1) > 90) & (J < np.roll(J, 1)) & (close < np.roll(close, 1))
        elif type == 'trend':
            k_ma = pl.Series(K).rolling_mean(5).to_numpy()
            entries = (K > D) & (D > 50) & (K > k_ma) & (close > np.roll(close, 5))
            exits = (K < D) & (D < 50) & (K < k_ma) & (close < np.roll(close, 5))
        signals = entries if flag else exits
        return pl.Series('kdj_signal', signals)

    def keltner(self, flag, period=20, multiplier=2, atr_period=10):
        close = self.df['Close'].to_numpy()
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        ema = talib.EMA(close, timeperiod=period)
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        if flag:
            signals = (np.roll(close, 1) > np.roll(upper, 1)) & (np.roll(close, 2) <= np.roll(upper, 2))
        else:
            signals = (np.roll(close, 1) < np.roll(lower, 1)) & (np.roll(close, 2) >= np.roll(lower, 2))
        return pl.Series('keltner_signal', signals)

    def ma(self, flag, type='cross', fast_period=5, slow_period=10):
        close = self.df['Close'].to_numpy()
        fast_ma = talib.MA(close, timeperiod=fast_period)
        slow_ma = talib.MA(close, timeperiod=slow_period)
        if type == 'cross':
            entries = (np.roll(fast_ma, 1) >= np.roll(slow_ma, 1)) & (np.roll(fast_ma, 2) < np.roll(slow_ma, 2))
            exits = (np.roll(fast_ma, 1) <= np.roll(slow_ma, 1)) & (np.roll(fast_ma, 2) > np.roll(slow_ma, 2))
        elif type == 'price':
            entries = (np.roll(close, 1) >= np.roll(slow_ma, 1)) & (np.roll(close, 2) < np.roll(slow_ma, 2))
            exits = (np.roll(close, 1) <= np.roll(slow_ma, 1)) & (np.roll(close, 2) > np.roll(slow_ma, 2))
        elif type == 'distance':
            distance = (fast_ma - slow_ma) / slow_ma
            distance_ma = pl.Series(distance).rolling_mean(slow_period).to_numpy()
            distance_std = pl.Series(distance).rolling_std(slow_period).to_numpy()
            entries = (np.roll(distance, 1) < (np.roll(distance_ma, 1) - np.roll(distance_std, 1))) & (distance > distance_ma)
            exits = distance < distance_ma
        elif type == 'slope':
            fast_slope = talib.LINEARREG_SLOPE(fast_ma, fast_period)
            slow_slope = talib.LINEARREG_SLOPE(slow_ma, slow_period)
            entries = (np.roll(fast_slope, 1) > 0) & (np.roll(slow_slope, 1) > 0) & (np.roll(fast_ma, 1) > np.roll(slow_ma, 1))
            exits = (fast_slope < 0) | (slow_slope < 0)
        signals = entries if flag else exits
        return pl.Series('ma_signal', signals)

    def macd(self, flag, type='cross', fastperiod=12, slowperiod=26, signalperiod=9, hist_threshold=0, divergence_lookback=5):
        close = self.df['Close'].to_numpy()
        macd_dif, macd_dea, macd_hist = talib.MACD(
            close,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        if type == 'cross':
            entries = (np.roll(macd_dif, 1) > np.roll(macd_dea, 1)) & (np.roll(macd_dif, 2) <= np.roll(macd_dea, 2)) & (np.roll(macd_dif, 1) > 0)
            exits = (np.roll(macd_dif, 1) < np.roll(macd_dea, 1)) & (np.roll(macd_dif, 2) >= np.roll(macd_dea, 2)) & (np.roll(macd_dif, 1) < 0)
        elif type == 'zero':
            entries = (np.roll(macd_dif, 1) > 0) & (np.roll(macd_dif, 2) <= 0)
            exits = (np.roll(macd_dif, 1) < 0) & (np.roll(macd_dif, 2) >= 0)
        elif type == 'hist':
            entries = (np.roll(macd_hist, 1) > hist_threshold) & (np.roll(macd_hist, 2) <= hist_threshold) & (np.roll(macd_dif, 1) > 0)
            exits = (np.roll(macd_hist, 1) < -hist_threshold) & (np.roll(macd_hist, 2) >= -hist_threshold) & (np.roll(macd_dif, 1) < 0)
        elif type == 'trend':
            hist_ma = pl.Series(macd_hist).rolling_mean(5).to_numpy()
            entries = (np.roll(macd_dif, 1) > np.roll(macd_dea, 1)) & (np.roll(macd_dea, 1) > 0) & (np.roll(macd_hist, 1) > np.roll(hist_ma, 1))
            exits = (np.roll(macd_dif, 1) < np.roll(macd_dea, 1)) & (np.roll(macd_dea, 1) < 0) & (np.roll(macd_hist, 1) < np.roll(hist_ma, 1))
        elif type == 'divergence':
            if flag:
                price_lows = pl.Series(self.df['Low'].to_numpy()).rolling_min(divergence_lookback).shift(1).to_numpy()
                macd_lows = pl.Series(macd_dif).rolling_min(divergence_lookback).shift(1).to_numpy()
                entries = (
                    (np.roll(self.df['Low'].to_numpy(), 1) == price_lows) &
                    (np.roll(macd_dif, 1) > macd_lows) &
                    (np.roll(close, 1) > np.roll(close, 2)) &
                    (np.roll(macd_dif, 1) > 0) & (np.roll(macd_hist, 1) > 0)
                )
                signals = entries
            else:
                price_highs = pl.Series(self.df['High'].to_numpy()).rolling_max(divergence_lookback).shift(1).to_numpy()
                macd_highs = pl.Series(macd_dif).rolling_max(divergence_lookback).shift(1).to_numpy()
                exits = (
                    (np.roll(self.df['High'].to_numpy(), 1) == price_highs) &
                    (np.roll(macd_dif, 1) < macd_highs) &
                    (np.roll(close, 1) < np.roll(close, 2)) &
                    (np.roll(macd_dif, 1) < 0) & (np.roll(macd_hist, 1) < 0)
                )
                signals = exits
            return pl.Series('macd_signal', signals if signals is not None else np.zeros(len(macd_dif), dtype=bool))
        signals = entries if flag else exits
        return pl.Series('macd_signal', signals)

    def mfi(self, flag, period=14, overbought=80, oversold=20):
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        close = self.df['Close'].to_numpy()
        volume = self.df['Volume'].to_numpy()
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        if flag:
            signals = (np.roll(mfi, 1) < oversold) & (mfi > np.roll(mfi, 1)) & (mfi > oversold)
        else:
            signals = (np.roll(mfi, 1) > overbought) & (mfi < np.roll(mfi, 1)) & (mfi < overbought)
        return pl.Series('mfi_signal', signals)

    def mom(self, flag, type='basic', period=10, ma_period=5, threshold=0, fast_period=12, slow_period=26, smooth_period=3):
        close = self.df['Close'].to_numpy()
        momentum = talib.MOM(close, timeperiod=period)
        if type == 'basic':
            mom_ma = talib.MA(momentum, timeperiod=ma_period)
            if flag:
                signals = (momentum > threshold) & (momentum > mom_ma) & (np.roll(momentum, 1) <= np.roll(mom_ma, 1))
            else:
                signals = (momentum < -threshold) & (momentum < mom_ma) & (np.roll(momentum, 1) >= np.roll(mom_ma, 1))
        elif type == 'oscillator':
            mom_fast = talib.MOM(close, timeperiod=fast_period)
            mom_slow = talib.MOM(close, timeperiod=slow_period)
            if flag:
                signals = (np.roll(mom_fast, 1) > np.roll(mom_slow, 1)) & (np.roll(mom_fast, 2) <= np.roll(mom_slow, 2))
            else:
                signals = (np.roll(mom_fast, 1) < np.roll(mom_slow, 1)) & (np.roll(mom_fast, 2) >= np.roll(mom_slow, 2))
        elif type == 'ma_crossover':
            price_ma = talib.MA(close, timeperiod=ma_period)
            if flag:
                signals = (np.roll(close, 1) > np.roll(price_ma, 1)) & (np.roll(close, 2) <= np.roll(price_ma, 2)) & (momentum > 0)
            else:
                signals = (np.roll(close, 1) < np.roll(price_ma, 1)) & (np.roll(close, 2) >= np.roll(price_ma, 2)) & (momentum < 0)
        elif type == 'percent':
            mom_pct = momentum / np.roll(close, period) * 100
            mom_ma = talib.MA(mom_pct, timeperiod=ma_period)
            if flag:
                signals = (mom_pct > threshold) & (mom_pct > mom_ma) & (np.roll(mom_pct, 1) <= np.roll(mom_ma, 1))
            else:
                signals = (mom_pct < -threshold) & (mom_pct < mom_ma) & (np.roll(mom_pct, 1) >= np.roll(mom_ma, 1))
        elif type == 'divergence':
            lookback = max(period, 10)
            if flag:
                price_lows = pl.Series(close).rolling_min(lookback).to_numpy()
                mom_lows = pl.Series(momentum).rolling_min(lookback).to_numpy()
                signals = (close == price_lows) & (momentum > mom_lows) & (np.roll(momentum, 1) > np.roll(momentum, 2))
            else:
                price_highs = pl.Series(close).rolling_max(lookback).to_numpy()
                mom_highs = pl.Series(momentum).rolling_max(lookback).to_numpy()
                signals = (close == price_highs) & (momentum < mom_highs) & (np.roll(momentum, 1) < np.roll(momentum, 2))
        elif type == 'volatility':
            atr = talib.ATR(self.df['High'].to_numpy(), self.df['Low'].to_numpy(), close, timeperiod=period)
            norm_mom = momentum / atr
            smooth_mom = talib.MA(norm_mom, timeperiod=smooth_period)
            if flag:
                signals = (norm_mom > threshold) & (norm_mom > smooth_mom) & (np.roll(norm_mom, 1) <= np.roll(smooth_mom, 1))
            else:
                signals = (norm_mom < -threshold) & (norm_mom < smooth_mom) & (np.roll(norm_mom, 1) >= np.roll(smooth_mom, 1))
        signals = signals if 'signals' in locals() else np.zeros(len(momentum), dtype=bool)
        return pl.Series('mom_signal', signals)

    def obv(self, flag, type='break', period=20, ma_period=30, divergence_lookback=10):
        close = self.df['Close'].to_numpy()
        volume = self.df['Volume'].to_numpy()
        obv = talib.OBV(close, volume)
        obv_smooth = pl.Series(obv).rolling_mean(period).to_numpy()
        obv_ma = pl.Series(obv).rolling_mean(ma_period).to_numpy()
        if type == 'break':
            if flag:
                prev_high = pl.Series(obv_smooth).rolling_max(divergence_lookback).to_numpy()
                entries = (obv_smooth >= prev_high) & (np.roll(obv_smooth, 1) < prev_high)
                signals = entries
            else:
                prev_low = pl.Series(obv_smooth).rolling_min(divergence_lookback).to_numpy()
                exits = (obv_smooth <= prev_low) & (np.roll(obv_smooth, 1) > prev_low)
                signals = exits
            return pl.Series('obv_signal', signals if signals is not None else np.zeros(len(obv), dtype=bool))
        elif type == 'ma':
            entries = (np.roll(obv_smooth, 1) > np.roll(obv_ma, 1)) & (np.roll(obv_smooth, 2) <= np.roll(obv_ma, 2)) & (close > np.roll(close, 1))
            exits = (np.roll(obv_smooth, 1) < np.roll(obv_ma, 1)) & (np.roll(obv_smooth, 2) >= np.roll(obv_ma, 2)) & (close < np.roll(close, 1))
        elif type == 'divergence':
            if flag:
                price_lows = pl.Series(self.df['Low'].to_numpy()).rolling_min(divergence_lookback).to_numpy()
                obv_lows = pl.Series(obv_smooth).rolling_min(divergence_lookback).to_numpy()
                entries = (self.df['Low'].to_numpy() == price_lows) & (obv_smooth > obv_lows) & (np.roll(obv_smooth, 1) > np.roll(obv_smooth, 2))
                signals = entries
            else:
                price_highs = pl.Series(self.df['High'].to_numpy()).rolling_max(divergence_lookback).to_numpy()
                obv_highs = pl.Series(obv_smooth).rolling_max(divergence_lookback).to_numpy()
                exits = (self.df['High'].to_numpy() == price_highs) & (obv_smooth < obv_highs) & (np.roll(obv_smooth, 1) < np.roll(obv_smooth, 2))
                signals = exits
            return pl.Series('obv_signal', signals if signals is not None else np.zeros(len(obv), dtype=bool))
        elif type == 'slope':
            # 5周期线性回归斜率
            def _slope(x):
                return np.polyfit(range(len(x)), x, 1)[0]
            slope = pl.Series(obv_smooth).rolling_apply(_slope, 5, closed="left").to_numpy()
            entries = (np.roll(slope, 1) > 0) & (np.roll(slope, 2) <= 0) & (volume > pl.Series(volume).rolling_mean(20).to_numpy())
            exits = (np.roll(slope, 1) < 0) & (np.roll(slope, 2) >= 0) & (volume < pl.Series(volume).rolling_mean(20).to_numpy())
        elif type == 'trend':
            obv_trend = pl.Series(obv_smooth).rolling_mean(3).to_numpy()
            entries = (obv_smooth > obv_ma) & (obv_smooth > obv_trend) & (close > np.roll(close, 5))
            exits = (obv_smooth < obv_ma) & (obv_smooth < obv_trend) & (close < np.roll(close, 5))
        signals = entries if flag else exits
        return pl.Series('obv_signal', signals)

    def rsi(self, flag, type='break', fast_period=6, slow_period=14, signal_period=12, overbought=70, oversold=30):
        close = self.df['Close'].to_numpy()
        rsi_slow = talib.RSI(close, timeperiod=slow_period)
        rsi_fast = talib.RSI(close, timeperiod=fast_period)
        if type == 'break':
            entries = (np.roll(rsi_slow, 1) < oversold) & (np.roll(rsi_slow, 2) >= oversold)
            exits = (np.roll(rsi_slow, 1) > overbought) & (np.roll(rsi_slow, 2) <= overbought)
        elif type == 'cross':
            rsi_signal = talib.RSI(close, timeperiod=signal_period)
            entries = (np.roll(rsi_fast, 1) >= np.roll(rsi_signal, 1)) & (np.roll(rsi_fast, 2) < np.roll(rsi_signal, 2)) & (np.roll(rsi_slow, 1) > 50)
            exits = (np.roll(rsi_fast, 1) <= np.roll(rsi_signal, 1)) & (np.roll(rsi_fast, 2) > np.roll(rsi_signal, 2)) & (np.roll(rsi_slow, 1) < 50)
        elif type == 'divergence':
            lookback = 10
            if flag:
                price_lows = pl.Series(self.df['Low'].to_numpy()).rolling_min(lookback).to_numpy()
                rsi_lows = pl.Series(rsi_slow).rolling_min(lookback).to_numpy()
                entries = (self.df['Low'].to_numpy() == price_lows) & (rsi_slow > rsi_lows) & (np.roll(rsi_slow, 1) > np.roll(rsi_slow, 2))
                signals = entries
            else:
                price_highs = pl.Series(self.df['High'].to_numpy()).rolling_max(lookback).to_numpy()
                rsi_highs = pl.Series(rsi_slow).rolling_max(lookback).to_numpy()
                exits = (self.df['High'].to_numpy() == price_highs) & (rsi_slow < rsi_highs) & (np.roll(rsi_slow, 1) < np.roll(rsi_slow, 2))
                signals = exits
            return pl.Series('rsi_signal', signals if signals is not None else np.zeros(len(rsi_slow), dtype=bool))
        elif type == 'swing':
            entries = (np.roll(rsi_slow, 1) < 30) & (np.roll(rsi_slow, 2) > 30) & (np.roll(rsi_slow, 3) > 30) & (np.roll(close, 1) > np.roll(close, 2))
            exits = (np.roll(rsi_slow, 1) > 70) & (np.roll(rsi_slow, 2) < 70) & (np.roll(rsi_slow, 3) < 70) & (np.roll(close, 1) < np.roll(close, 2))
        elif type == 'trend':
            rsi_ma = pl.Series(rsi_slow).rolling_mean(5).to_numpy()
            entries = (rsi_slow > 50) & (rsi_slow > rsi_ma) & (np.roll(rsi_slow, 1) > np.roll(rsi_slow, 2))
            exits = (rsi_slow < 50) & (rsi_slow < rsi_ma) & (np.roll(rsi_slow, 1) < np.roll(rsi_slow, 2))
        signals = entries if flag else exits
        return pl.Series('rsi_signal', signals)

    def sar(self, flag, acceleration=0.02, maximum=0.2):
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        close = self.df['Close'].to_numpy()
        sar = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        if flag:
            signals = (np.roll(close, 1) > np.roll(sar, 1)) & (np.roll(close, 2) <= np.roll(sar, 2))
        else:
            signals = (np.roll(close, 1) < np.roll(sar, 1)) & (np.roll(close, 2) >= np.roll(sar, 2))
        return pl.Series('sar_signal', signals)

    def trix(self, flag, period=12, signal=9):
        close = self.df['Close'].to_numpy()
        trix = talib.TRIX(close, timeperiod=period)
        trix_signal = talib.EMA(trix, timeperiod=signal)
        if flag:
            signals = (np.roll(trix, 1) > np.roll(trix_signal, 1)) & (np.roll(trix, 2) <= np.roll(trix_signal, 2))
        else:
            signals = (np.roll(trix, 1) < np.roll(trix_signal, 1)) & (np.roll(trix, 2) >= np.roll(trix_signal, 2))
        return pl.Series('trix_signal', signals)

    def williams(self, flag, period=14, overbought=-20, oversold=-80):
        high = self.df['High'].to_numpy()
        low = self.df['Low'].to_numpy()
        close = self.df['Close'].to_numpy()
        willr = talib.WILLR(high, low, close, timeperiod=period)
        if flag:
            signals = (np.roll(willr, 1) < oversold) & (willr > np.roll(willr, 1)) & (willr > oversold)
        else:
            signals = (np.roll(willr, 1) > overbought) & (willr < np.roll(willr, 1)) & (willr < overbought)
        return pl.Series('williams_signal', signals)


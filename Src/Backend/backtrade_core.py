import numpy as np
import polars as pl

class BackTrade:
    def __init__(self, 
                 df: pl.DataFrame, 
                 init_cash: float =100000.0, 
                 fee: float =0.0002, 
                 slippage: float =0.0025
                 ):
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
        self.data = df.with_columns(pl.col('Date').str.strptime(pl.Date)).sort('Date')
        self.init_cash = init_cash
        self.fee = fee
        self.slip = slippage
        self.trades = []

    def run(self, 
            entries: pl.DataFrame | None =None, 
            exits: pl.DataFrame | None =None, 
            sl: float | None=None, 
            tp: float | None=None,
            size: float =1.0
            ):
        """
        回测主逻辑
        参数:
        - size: 仓位比例
        - sl: 止损
        - tp: 止盈
        """
        if entries is None or exits is None:
            raise ValueError("必须提供entries和exits信号")
        dates = self.data['Date']
        price = self.data['Close']
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
                    exit_price = round(stop_loss_price * (1 - self.slip),4)
                    exit_type = 'stop_loss'
                elif tp_triggered:
                    exit_price = round(take_profit_price * (1 - self.slip),4)
                    exit_type = 'take_profit'
                else:
                    exit_price = round(current_price * (1 - self.slip),4)
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
                    'Return': round((exit_price / entry_price - 1) * 100, 2),
                    'Type': exit_type,
                })
            elif not in_position and entries[i]:
                entry_price = round(current_price * (1 + self.slip), 4)
                max_shares = int(((cash[i] * size) // (entry_price * (1 + self.fee))) // 100 * 100)
                if max_shares > 0:
                    fee = max_shares * entry_price * self.fee
                    cash[i:] -= round((max_shares * entry_price + fee), 2)
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
        self.trades = pl.DataFrame(self.trades)
        return self

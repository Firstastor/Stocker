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

    @Slot('QVariantList', result='QVariantMap')
    def calculate_trading_signals(self, history_data):
        """计算买卖信号强度（专业增强版）
        
        返回结构：
        {
            "买入信号": [
                {"名称": str, "强度": int(0-100), "描述": str},
                ...
            ],
            "卖出信号": [
                {"名称": str, "强度": int(0-100), "描述": str},
                ...
            ]
        }
        """
        # 初始化返回结构
        signals = {
            "买入信号": [],
            "卖出信号": []
        }

        if not history_data or len(history_data) < 30:
            return signals

        try:
            # 数据准备（带异常值处理）
            closes = np.array([d.get('收盘价', float('nan')) for d in history_data], dtype=np.float64)
            highs = np.array([d.get('最高价', float('nan')) for d in history_data], dtype=np.float64)
            lows = np.array([d.get('最低价', float('nan')) for d in history_data], dtype=np.float64)
            volumes = np.array([d.get('成交量', 0) for d in history_data], dtype=np.float64)
            opens = np.array([d.get('开盘价', float('nan')) for d in history_data], dtype=np.float64)
            
            # 数据有效性检查
            if np.isnan(closes).any() or np.isnan(highs).any() or np.isnan(lows).any():
                return signals

            # ================= 技术指标计算 =================
            # 均线系统
            ma5 = talib.SMA(closes, timeperiod=5)
            ma10 = talib.SMA(closes, timeperiod=10)
            ma20 = talib.SMA(closes, timeperiod=20)
            ma60 = talib.SMA(closes, timeperiod=60)
            
            # 动量指标
            rsi = talib.RSI(closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes)
            slowk, slowd = talib.STOCH(highs, lows, closes)
            cci = talib.CCI(highs, lows, closes, timeperiod=14)
            
            # 波动率指标
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            natr = talib.NATR(highs, lows, closes, timeperiod=14)
            
            # 成交量指标
            vol_ma5 = talib.SMA(volumes, timeperiod=5)
            vol_ma10 = talib.SMA(volumes, timeperiod=10)
            vol_ma20 = talib.SMA(volumes, timeperiod=20)
            obv = talib.OBV(closes, volumes)
            
            # ================= 信号生成逻辑 =================
            # 金叉信号
            golden_cross = (ma5[-1] > ma10[-1]) and (ma5[-2] <= ma10[-2])
            if golden_cross:
                # 改进后的强度计算
                trend_angle = np.arctan((ma5[-1] - ma5[-5]) / (ma5[-5] * 0.01)) * 180/np.pi
                trend_power = 25 * min(1, max(0, trend_angle / 30))
                
                volume_ratio = min(5, volumes[-1] / vol_ma5[-1])
                volume_boost = 20 * (1 - np.exp(-0.5*(volume_ratio-1)))
                
                rsi_weight = 1 - abs(rsi[-1] - 50)/50
                rsi_confirm = 15 * rsi_weight
                
                ma_spread = (ma5[-1] - ma10[-1]) / closes[-1] * 100
                spread_boost = 10 * min(1, max(0, ma_spread / 0.5))
                
                strength = min(95, max(40, 50 + trend_power + volume_boost + rsi_confirm + spread_boost))
                
                signals["买入信号"].append({
                    "名称": "金叉信号",
                    "强度": int(strength),
                    "描述": (
                        f"5日线({ma5[-1]:.2f})上穿10日线({ma10[-1]:.2f})\n"
                        f"趋势角度: {trend_angle:.1f}° | 成交量比率: {volume_ratio:.1f}x\n"
                        f"RSI确认: {rsi[-1]:.1f} | 均线间距: {ma_spread:.2f}%"
                    )
                })

            # 死叉信号
            death_cross = (ma5[-1] < ma10[-1]) and (ma5[-2] >= ma10[-2])
            if death_cross:
                trend_damage = 30 * (ma10[-1] - ma20[-1]) / ma20[-1] * (1 if ma5[-1] < ma20[-1] else 0.5)
                normalized_volatility = 25 * (atr[-1] / closes[-1]) / 0.02
                macd_diff = (macd_signal[-1] - macd[-1]) / closes[-1] * 1000
                macd_confirm = 15 * min(1, max(0, macd_diff / 0.5))
                volume_confirmation = 10 if volumes[-1] > vol_ma5[-1] else 0
                strength = min(95, max(40, 55 + trend_damage + normalized_volatility + macd_confirm + volume_confirmation))
                signals["卖出信号"].append({
                    "名称": "死叉信号",
                    "强度": int(strength),
                    "描述": (
                        f"5日线({ma5[-1]:.2f})下穿10日线({ma10[-1]:.2f})\n"
                        f"趋势破坏: {trend_damage:.1f}% | ATR波动率: {natr[-1]:.1f}%\n"
                        f"MACD差值: {macd_diff:.3f} | 成交量确认: {'是' if volume_confirmation>0 else '否'}"
                    )
                })

            # MACD背离信号
            self._add_macd_divergence(signals, closes, highs, lows, macd, macd_hist, atr)
            
            # RSI超买超卖信号
            self._add_rsi_signals(signals, rsi, closes, atr, volumes, vol_ma5)
            
            # 布林带信号
            self._add_bollinger_signals(signals, closes, highs, lows, volumes, vol_ma20, atr)
            
            # 成交量异动信号
            self._add_volume_signals(signals, volumes, vol_ma5, vol_ma10, closes, obv)

            return signals

        except Exception as e:
            print(f"信号计算异常: {str(e)}")
            return signals

    def _add_macd_divergence(self, signals, closes, highs, lows, macd, macd_hist, atr):
        """增强版MACD背离信号检测（完整修复版）"""
        min_data_points = 26  # 最小数据点数
        if len(closes) < min_data_points:
            return

        # ========== 配置参数 ==========
        lookback_long = 26    # 底背离检测窗口（长周期）
        lookback_short = 13   # 顶背离检测窗口（短周期）
        atr_threshold = 0.03  # ATR波动率阈值（3%）
        # =============================

        # 辅助函数：标准化差值计算（解决百分比失真问题）
        def safe_divergence(new_price, old_price, new_macd, old_macd):
            """返回标准化后的价格和MACD差值"""
            # 价格差值用ATR缩放（1ATR = 1个平均波动单位）
            price_diff = (new_price - old_price) / (atr[-1] if atr[-1] > 0 else 1.0)
            
            # MACD差值用历史波动缩放（20周期标准差）
            macd_std = np.std(macd[-20:]) if len(macd) >= 20 else 1.0
            macd_diff = (new_macd - old_macd) / (macd_std if macd_std > 0 else 1.0)
            
            return price_diff, macd_diff

        # 初始化信号容器
        buy_signal = None
        sell_signal = None

        # ========== 底背离检测（买入信号） ==========
        recent_closes_long = closes[-lookback_long:]
        recent_macd_long = macd[-lookback_long:]

        lowest_idx = np.argmin(recent_closes_long)
        macd_lowest_idx = np.argmin(recent_macd_long)

        if (lowest_idx < macd_lowest_idx and 
            recent_macd_long[-1] > recent_macd_long[macd_lowest_idx]):
            
            # 计算标准化差值（解决百分比失真）
            price_diff, macd_diff = safe_divergence(
                recent_closes_long[-1], recent_closes_long[lowest_idx],
                recent_macd_long[-1], recent_macd_long[macd_lowest_idx]
            )
            
            # 背离分数 = MACD相对强度 - 价格相对强度
            divergence_score = macd_diff - price_diff
            
            # 波动率调整（高波动降低信号强度）
            atr_ratio = atr[-1] / closes[-1]
            volatility_adjustment = -8 * min(1, atr_ratio / atr_threshold)
            
            # 动态强度计算（基础50分，±20分范围）
            strength = np.clip(50 + 10 * divergence_score + volatility_adjustment, 20, 95)
            
            buy_signal = {
                "名称": "MACD底背离",
                "强度": int(strength),
                "描述": (
                    f"价格新低但MACD未创新低\n"
                    f"价格差值: {price_diff:+.1f}ATR | MACD差值: {macd_diff:+.1f}σ\n"
                    f"背离分数: {divergence_score:+.1f} | ATR调整: {volatility_adjustment:+.1f}"
                )
            }

        # ========== 顶背离检测（卖出信号） ==========
        recent_closes_short = closes[-lookback_short:]
        recent_macd_short = macd[-lookback_short:]

        highest_idx = np.argmax(recent_closes_short)
        macd_highest_idx = np.argmax(recent_macd_short)

        if (highest_idx < macd_highest_idx and 
            recent_macd_short[-1] < recent_macd_short[macd_highest_idx]):
            
            price_diff, macd_diff = safe_divergence(
                recent_closes_short[-1], recent_closes_short[highest_idx],
                recent_macd_short[-1], recent_macd_short[macd_highest_idx]
            )
            
            divergence_score = macd_diff - price_diff
            
            # 波动率调整（与底背离逻辑一致）
            atr_ratio = atr[-1] / closes[-1]
            volatility_adjustment = -8 * min(1, atr_ratio / atr_threshold)
            
            strength = np.clip(50 + 10 * abs(divergence_score) + volatility_adjustment, 20, 95)
            
            sell_signal = {
                "名称": "MACD顶背离",
                "强度": int(strength),
                "描述": (
                    f"价格新高但MACD未创新高\n"
                    f"价格差值: {price_diff:+.1f}ATR | MACD差值: {macd_diff:+.1f}σ\n"
                    f"背离分数: {divergence_score:+.1f} | 波动加成: {volatility_adjustment:+.1f}"
                )
            }

        # ========== 信号冲突处理 ==========
        if buy_signal and sell_signal:
            # 策略1：选择强度更高的信号
            if buy_signal["强度"] >= sell_signal["强度"]:
                signals["买入信号"].append(buy_signal)
            else:
                signals["卖出信号"].append(sell_signal)
            
            # 策略2：标记为震荡行情（可选）
            # signals["震荡信号"] = {"名称": "双向背离", "强度": max(buy_strength, sell_strength)}
        else:
            if buy_signal:
                signals["买入信号"].append(buy_signal)
            if sell_signal:
                signals["卖出信号"].append(sell_signal)

    def _add_rsi_signals(self, signals, rsi, closes, atr=None, volumes=None, vol_ma=None):
        """增强版RSI信号检测（修复数组比较问题）"""
        # 参数配置
        config = {
            'oversold': 30,
            'overbought': 70,
            'slope_window': 3,
            'max_slope': 20,
            'volatility_thresh': 0.02,
            'volume_thresh': 1.1,
            'base_strength': 60
        }
        
        # 确保输入为numpy数组
        rsi = np.asarray(rsi)
        closes = np.asarray(closes)
        
        # 数据检查
        if len(rsi) < config['slope_window']:
            return
        
        # 辅助函数：安全获取最新值
        def get_last_valid(arr):
            return arr[~np.isnan(arr)][-1] if isinstance(arr, np.ndarray) else arr[-1]
        
        # 当前RSI值（确保是标量）
        current_rsi = get_last_valid(rsi)
        
        # 斜率计算（处理可能的NaN值）
        valid_rsi = rsi[~np.isnan(rsi)]
        if len(valid_rsi) >= config['slope_window']:
            rsi_segment = valid_rsi[-config['slope_window']:]
            rsi_slope = (rsi_segment[-1] - rsi_segment[0]) / (len(rsi_segment) - 1)
        else:
            rsi_slope = 0
        
        # 斜率限制
        rsi_slope = np.clip(rsi_slope, -config['max_slope'], config['max_slope'])
        
        # 波动率调整
        def get_volatility_adjust():
            if atr is None or len(atr) < 1:
                return 0
            current_atr = get_last_valid(atr)
            current_close = get_last_valid(closes)
            atr_ratio = current_atr / current_close
            return 10 * min(1, atr_ratio / config['volatility_thresh'])
        
        # 量能确认
        def get_volume_confirm():
            if volumes is None or vol_ma is None or len(volumes) < 1 or len(vol_ma) < 1:
                return 0
            current_volume = get_last_valid(volumes)
            current_vol_ma = get_last_valid(vol_ma)
            return 5 if current_volume > current_vol_ma * config['volume_thresh'] else 0
        
        # RSI超卖信号（买入）
        if current_rsi < config['oversold']:
            oversold_level = config['oversold'] - current_rsi
            strength = config['base_strength'] + oversold_level * 1.2
            
            if rsi_slope > 1:
                strength += min(15, rsi_slope * 0.8)
            
            strength -= get_volatility_adjust() * 0.5
            strength += get_volume_confirm()
            strength = np.clip(strength, 40, 95)
            
            signals["买入信号"].append({
                "名称": "RSI超卖",
                "强度": int(strength),
                "描述": (
                    f"RSI({current_rsi:.1f})超卖 | "
                    f"偏离程度: {oversold_level:.1f}点\n"
                    f"标准化斜率: {rsi_slope:.1f} | "
                    f"ATR比率: {get_last_valid(atr)/get_last_valid(closes)*100 if atr is not None else 0:.1f}%\n"
                    f"量能确认: {'是' if get_volume_confirm()>0 else '否'}"
                )
            })
        
        # RSI超买信号（卖出）
        elif current_rsi > config['overbought']:
            overbought_level = current_rsi - config['overbought']
            strength = config['base_strength'] + overbought_level * 1.0
            
            if rsi_slope < -1:
                strength += min(15, abs(rsi_slope) * 0.8)
            
            strength += get_volatility_adjust()
            strength = np.clip(strength, 40, 95)
            
            signals["卖出信号"].append({
                "名称": "RSI超买",
                "强度": int(strength),
                "描述": (
                    f"RSI({current_rsi:.1f})超买 | "
                    f"偏离程度: {overbought_level:.1f}点\n"
                    f"标准化斜率: {rsi_slope:.1f} | "
                    f"ATR比率: {get_last_valid(atr)/get_last_valid(closes)*100 if atr is not None else 0:.1f}%\n"
                    f"量能确认: {'是' if volumes is not None and get_last_valid(volumes) < get_last_valid(vol_ma)/config['volume_thresh'] else '否'}"
                )
            })

    def _add_bollinger_signals(self, signals, closes, highs, lows, volumes=None, vol_ma20=None, atr=None):
        """增强版布林带信号(含波动率调整)"""
        # 参数配置
        config = {
            'period': 20,               # 布林带周期
            'base_strength': 65,        # 基础信号强度
            'max_penetration': 30,      # 最大穿透深度(%) 
            'min_band_width': 0.01,     # 最小布林带宽度(相对于价格)
            'volume_threshold': 1.1,    # 量能确认阈值
            'volatility_window': 14,    # 波动率评估窗口
            'high_volatility': 0.02,    # 高波动阈值(ATR/Price)
            'low_volatility': 0.005      # 低波动阈值(ATR/Price)
        }
        
        # 计算布林带
        upper, middle, lower = talib.BBANDS(
            closes, 
            timeperiod=config['period'],
            nbdevup=2, 
            nbdevdn=2
        )
        
        # 数据有效性检查
        if np.isnan(upper[-1]) or np.isnan(lower[-1]):
            return
        
        band_width = upper[-1] - lower[-1]
        relative_width = band_width / closes[-1]
        
        # 过滤布林带宽度过窄的情况
        if relative_width < config['min_band_width']:
            return
        
        # 计算波动率调整因子
        def get_volatility_adjustment():
            if atr is None or len(atr) < config['volatility_window']:
                return 0
                
            atr_ratio = atr[-1] / closes[-1]
            
            # 买入信号波动率调整(高波动减弱信号，低波动增强)
            if closes[-1] < lower[-1]:
                if atr_ratio > config['high_volatility']:
                    return -10 * min(1, (atr_ratio - config['high_volatility']) / 0.01)
                elif atr_ratio < config['low_volatility']:
                    return 5 * min(1, (config['low_volatility'] - atr_ratio) / 0.0025)
            
            # 卖出信号波动率调整(高波动增强信号，低波动减弱)
            elif closes[-1] > upper[-1]:
                if atr_ratio > config['high_volatility']:
                    return 10 * min(1, (atr_ratio - config['high_volatility']) / 0.01)
                elif atr_ratio < config['low_volatility']:
                    return -5 * min(1, (config['low_volatility'] - atr_ratio) / 0.0025)
            
            return 0
        
        # 辅助函数：计算信号强度
        def calc_strength(penetration, is_buy):
            penetration_factor = min(config['max_penetration'], penetration)
            strength = config['base_strength'] + penetration_factor
            
            # 量能确认增强
            if volumes is not None and vol_ma20 is not None:
                vol_ratio = volumes[-1] / vol_ma20[-1]
                if is_buy and vol_ratio > config['volume_threshold']:
                    strength += 5
                elif not is_buy and vol_ratio < 1/config['volume_threshold']:
                    strength += 5
            
            # 波动率调整
            strength += get_volatility_adjustment()
            
            return min(95, max(40, strength))  # 确保强度在40-95范围内
        
        # 布林带下轨反弹（买入） - 需要连续两天确认
        if (closes[-1] < lower[-1] and 
            closes[-2] >= lower[-2] and 
            closes[-3] >= lower[-3]):
            
            penetration = (lower[-1] - closes[-1]) / band_width * 100
            strength = calc_strength(penetration, is_buy=True)
            vol_adjustment = get_volatility_adjustment()
            
            signals["买入信号"].append({
                "名称": "布林带下轨反弹",
                "强度": int(strength),
                "描述": (
                    f"价格从下轨({lower[-1]:.2f})反弹 | "
                    f"穿透深度: {penetration:.1f}%\n"
                    f"带宽: {relative_width*100:.1f}% | "
                    f"ATR比率: {atr[-1]/closes[-1]*100:.1f}%\n"
                    f"量能: {'放量' if volumes is not None and volumes[-1] > vol_ma20[-1] * config['volume_threshold'] else '正常'}"
                    f"{f' | 波动调整: {vol_adjustment:+.1f}' if vol_adjustment !=0 else ''}"
                )
            })
        
        # 布林带上轨回落（卖出） - 需要连续两天确认
        if (closes[-1] > upper[-1] and 
            closes[-2] <= upper[-2] and 
            closes[-3] <= upper[-3]):
            
            penetration = (closes[-1] - upper[-1]) / band_width * 100
            strength = calc_strength(penetration, is_buy=False)
            vol_adjustment = get_volatility_adjustment()
            
            signals["卖出信号"].append({
                "名称": "布林带上轨回落",
                "强度": int(strength),
                "描述": (
                    f"价格从上轨({upper[-1]:.2f})回落 | "
                    f"穿透深度: {penetration:.1f}%\n"
                    f"带宽: {relative_width*100:.1f}% | "
                    f"ATR比率: {atr[-1]/closes[-1]*100:.1f}%\n"
                    f"量能: {'缩量' if volumes is not None and volumes[-1] < vol_ma20[-1] / config['volume_threshold'] else '正常'}"
                    f"{f' | 波动调整: {vol_adjustment:+.1f}' if vol_adjustment !=0 else ''}"
                )
            })

    def _add_volume_signals(self, signals, volumes, vol_ma5, vol_ma10, closes, obv):
        """基于动态权重的成交量信号（无固定基础/最大强度）"""
        lookback_period = 5  # 统一回溯周期（5日）
        
        # 计算关键指标
        vol_ratio_ma5 = volumes[-1] / vol_ma5[-1]  # 成交量 / 5日均量
        vol_ratio_ma10 = volumes[-1] / vol_ma10[-1]  # 成交量 / 10日均量
        price_change_pct = (closes[-1] - closes[-lookback_period-1]) / closes[-lookback_period-1] * 100  # 5日价格变化百分比
        obv_trend = obv[-1] - obv[-lookback_period-1]  # 5日OBV变化

        # --- 信号1: 放量突破（买入）---
        if vol_ratio_ma5 > 2.0 and vol_ratio_ma10 > 1.5:
            # 动态强度 = 成交量强度(0~50) + OBV趋势强度(0~30) + 价格动量(0~20)
            strength = (
                0.5 * min(100, (vol_ratio_ma10 - 1) * 50)  # 成交量越强，分数越高（0~50）
                + 0.3 * min(100, obv_trend / (volumes[-1] + 1e-6) * 100)  # OBV趋势（0~30）
                + 0.2 * max(0, price_change_pct)  # 价格上涨则加分（0~20）
            )
            
            signals["买入信号"].append({
                "名称": "放量突破",
                "强度": int(strength),
                "描述": (
                    f"成交量暴增（5日{vol_ratio_ma5:.1f}x，10日{vol_ratio_ma10:.1f}x）\n"
                    f"OBV5日趋势: {obv_trend/1000:.1f}K\n"
                    f"价格5日变化: {price_change_pct:.1f}%"
                )
            })

        # --- 信号2: 缩量上涨（卖出）---
        elif price_change_pct > 0 and vol_ratio_ma5 < 0.8:
            # 动态强度 = 缩量程度(0~60) - 价格上涨幅度(0~40)
            strength = (
                0.6 * (100 - vol_ratio_ma5 * 100)  # 缩量越严重，分数越高（0~60）
                - 0.4 * min(100, price_change_pct)  # 涨幅越高，分数越低（-40~0）
            )
            strength = max(0, strength)  # 确保强度不为负
            
            signals["卖出信号"].append({
                "名称": "缩量上涨",
                "强度": int(strength),
                "描述": (
                    f"价格上涨{price_change_pct:.1f}%，但成交量仅5日均量的{vol_ratio_ma5:.1f}x\n"
                    f"量价背离，可能见顶"
                )
            })

        # --- 信号3: 放量下跌（卖出）---
        elif price_change_pct < 0 and vol_ratio_ma5 > 1.5:
            # 动态强度 = 放量程度(0~50) + 下跌幅度(0~50)
            strength = (
                0.5 * min(100, (vol_ratio_ma5 - 1) * 50)  # 放量越强，分数越高（0~50）
                + 0.5 * min(100, abs(price_change_pct))  # 跌幅越大，分数越高（0~50）
            )
            
            signals["卖出信号"].append({
                "名称": "放量下跌",
                "强度": int(strength),
                "描述": (
                    f"价格下跌{abs(price_change_pct):.1f}%，成交量达5日均量{vol_ratio_ma5:.1f}x\n"
                    f"资金出逃明显"
                )
            })
                
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
    
        

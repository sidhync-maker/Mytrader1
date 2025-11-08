
"""Streamlit Automated Options Trader — FYERS (PAPER MODE)

This ready-to-run app is preconfigured to operate in PAPER mode (no live orders sent to FYERS).
It simulates order placement and writes simulated trade events to `paper_trades_log.csv`.

How to run:
1) Install dependencies:
   pip install streamlit pandas numpy requests fyers-apiv2
   (FYERS SDK is optional for paper mode; it's not required)

2) Run:
   streamlit run streamlit_option_trader_fyers_paper.py

Configuration:
- Edit config_example.json if you want to store credentials (not required for paper mode).
- Paper mode will simulate LTP by using historical close prices or random small ticks.

IMPORTANT:
- This is a demo for validation and strategy development. Test thoroughly.
- Do NOT enable live mode unless you know what you are doing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import threading
import os
import csv
from typing import Optional, Dict, Any

st.set_page_config(page_title="Auto Options Trader — FYERS (PAPER)", layout="wide")

LOG_CSV = "paper_trades_log.csv"

# ----------------------------------------------------------------------------
# Simple paper-mode broker simulator (no network calls)
# ----------------------------------------------------------------------------
class PaperBroker:
    def __init__(self):
        # local order id counter & storage
        self._counter = 1000
        self.orders = {}  # order_id -> metadata

    def _next_id(self):
        self._counter += 1
        return str(self._counter)

    def get_historical(self, symbol: str, start: dt.datetime, end: dt.datetime, resolution: str = '5') -> pd.DataFrame:
        # Simple synthetic historical: create 5-min candles using a sinusoidal series around 20000 (useful for NIFTY)
        periods = int(((end - start).total_seconds()) // (60 * int(resolution or 5)))
        if periods <= 0:
            periods = 100
        idx = pd.date_range(end=end, periods=periods, freq=f"{resolution}T")
        base = 20000 + 50 * np.sin(np.linspace(0, 6.28, len(idx)))
        noise = np.random.normal(scale=5.0, size=len(idx))
        close = base + noise
        df = pd.DataFrame({
            "datetime": idx,
            "open": close + np.random.normal(0, 2, len(idx)),
            "high": close + np.random.uniform(0, 6, len(idx)),
            "low": close - np.random.uniform(0, 6, len(idx)),
            "close": close,
            "volume": np.random.randint(100, 1000, len(idx))
        })
        return df

    def ltp(self, symbol: str) -> float:
        # Return last price from the most recent candle if available, else synthetic
        now = dt.datetime.now()
        df = self.get_historical(symbol, now - dt.timedelta(days=1), now, resolution='5')
        if not df.empty:
            return float(df['close'].iloc[-1])
        return 20000.0 + np.random.normal(0, 10)

    def place_order(self, symbol: str, qty: int, side: str = 'BUY', order_type: str = 'MARKET', product_type: str = 'INTRADAY') -> Dict[str, Any]:
        order_id = self._next_id()
        ltp = self.ltp(symbol)
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "status": "FILLED",  # simulate immediate fill for demo
            "filled_price": ltp,
            "ts": dt.datetime.now().isoformat()
        }
        self.orders[order_id] = order
        # record to CSV (paper log)
        write_trade_log({
            "ts": order["ts"],
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": order["filled_price"],
            "order_type": order_type,
            "note": "PAPER-FILLED"
        })
        return order

    def modify_order(self, order_id: str, **kwargs):
        # In paper mode we update local metadata
        if order_id in self.orders:
            self.orders[order_id].update(kwargs)
            return {"success": True, "order_id": order_id}
        return {"error": "not found"}

# ----------------------------------------------------------------------------
# Utilities and strategy logic
# ----------------------------------------------------------------------------
def write_trade_log(row: Dict[str, Any]):
    header = ["ts","order_id","symbol","side","qty","price","order_type","note"]
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def pick_itm_option_strike(underlying_price: float, strike_step: int = 50) -> int:
    return int(round(underlying_price / strike_step) * strike_step)

def check_entry_conditions(df5: pd.DataFrame, yesterday_option_high: float, vwap_series: pd.Series) -> bool:
    if df5.shape[0] < 3:
        return False
    c_prev = df5.iloc[-3]
    c_second = df5.iloc[-2]
    vwap_second = vwap_series.iloc[-2]
    crosses_vwap = (c_second['open'] < vwap_second) and (c_second['high'] > vwap_second)
    rises_and_higher = (c_second['close'] > vwap_second) and (c_second['high'] > c_prev['high'])
    cond1 = crosses_vwap or rises_and_higher
    passes_prev_close = c_second['high'] > c_prev['close']
    cond_yest = yesterday_option_high > 0
    return cond1 and passes_prev_close and cond_yest

# ----------------------------------------------------------------------------
# OrderManager for paper mode
# ----------------------------------------------------------------------------
class OrderManager:
    def __init__(self, broker: PaperBroker, stop_loss_pct: float = 0.03, trailing_pct: float = 0.03):
        self.broker = broker
        self.stop_loss_pct = stop_loss_pct
        self.trailing_pct = trailing_pct
        self.active_orders = {}
        self._monitor_thread = None
        self._run_monitor = False

    def start_monitor(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._run_monitor = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self):
        self._run_monitor = False

    def place_buy_option(self, symbol: str, quantity: int = 1) -> Dict[str, Any]:
        resp = self.broker.place_order(symbol, quantity, side='BUY', order_type='MARKET')
        order_id = resp.get('order_id')
        ltp = resp.get('filled_price')
        sl_price = ltp * (1 - self.stop_loss_pct)
        self.active_orders[order_id] = {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': ltp,
            'sl_price': sl_price,
            'peak_price': ltp
        }
        return resp

    def _monitor_loop(self):
        while self._run_monitor:
            try:
                for oid, meta in list(self.active_orders.items()):
                    sym = meta['symbol']
                    ltp = self.broker.ltp(sym)
                    if ltp and ltp > meta.get('peak_price', 0):
                        meta['peak_price'] = ltp
                    if meta.get('peak_price'):
                        new_sl = meta['peak_price'] * (1 - self.trailing_pct)
                        if meta.get('sl_price') is None or new_sl > meta['sl_price']:
                            meta['sl_price'] = round(new_sl, 2)
                    if meta.get('sl_price') and ltp <= meta['sl_price']:
                        # exit
                        _ = self.broker.place_order(sym, meta['quantity'], side='SELL', order_type='MARKET')
                        write_trade_log({
                            "ts": dt.datetime.now().isoformat(),
                            "order_id": oid + "-EXIT",
                            "symbol": sym,
                            "side": "SELL",
                            "qty": meta['quantity'],
                            "price": ltp,
                            "order_type": "MARKET",
                            "note": "PAPER-AUTO-EXIT"
                        })
                        self.active_orders.pop(oid, None)
                time.sleep(2)
            except Exception as e:
                time.sleep(2)

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.title("Automated Options Trader — FYERS (PAPER)")
st.markdown("Paper/demo mode — no live orders will be sent. Simulated fills will be logged to `paper_trades_log.csv`.")

with st.sidebar:
    st.header("Paper Mode Settings")
    strike_step = st.number_input("Strike step (e.g., 50)", value=50, step=1)
    stop_loss_pct = st.number_input("Stop Loss %", value=3.0, min_value=0.1, max_value=50.0) / 100.0
    trailing_pct = st.number_input("Trailing SL %", value=3.0, min_value=0.1, max_value=50.0) / 100.0
    quantity = st.number_input("Quantity (lots/shares)", value=1, step=1)

st.markdown("Use the buttons below to run entry checks and to simulate a live buy if conditions match.")

broker = PaperBroker()
om = OrderManager(broker, stop_loss_pct=stop_loss_pct, trailing_pct=trailing_pct)
om.start_monitor()

col1, col2 = st.columns(2)
with col1:
    if st.button("Run entry-check now"):
        to_dt = dt.datetime.now()
        from_dt = to_dt - dt.timedelta(days=2)
        symbol_underlying = "NSE:NIFTY-50"
        df5 = broker.get_historical(symbol_underlying, from_dt, to_dt, resolution='5')
        df5 = df5.sort_values('datetime').reset_index(drop=True)
        df5['vwap'] = calculate_vwap(df5)
        yesterday = (dt.date.today() - dt.timedelta(days=1))
        yesterday_mask = df5['datetime'].dt.date == yesterday
        yesterday_high = df5.loc[yesterday_mask, 'high'].max() if yesterday_mask.sum() > 0 else 0
        ok = check_entry_conditions(df5, yesterday_high, df5['vwap'])
        st.write(f"Entry conditions met? {ok}")
        st.dataframe(df5.tail(6)[['datetime','open','high','low','close','volume','vwap']])

with col2:
    if st.button("Run full strategy & buy (PAPER)"):
        to_dt = dt.datetime.now()
        from_dt = to_dt - dt.timedelta(days=2)
        symbol_underlying = "NSE:NIFTY-50"
        df5 = broker.get_historical(symbol_underlying, from_dt, to_dt, resolution='5')
        df5 = df5.sort_values('datetime').reset_index(drop=True)
        df5['vwap'] = calculate_vwap(df5)
        yesterday = (dt.date.today() - dt.timedelta(days=1))
        yesterday_mask = df5['datetime'].dt.date == yesterday
        yesterday_high = df5.loc[yesterday_mask, 'high'].max() if yesterday_mask.sum() > 0 else 0
        ok = check_entry_conditions(df5, yesterday_high, df5['vwap'])
        if not ok:
            st.warning("Entry conditions NOT met. Aborting buy.")
        else:
            underlying_ltp = broker.ltp(symbol_underlying)
            strike = pick_itm_option_strike(underlying_ltp, strike_step)
            st.write(f"Selected strike {strike} based on underlying {underlying_ltp}")
            tradingsymbol = f"NIFTY{strike}CE"  # placeholder tradingsymbol for demo
            resp = om.place_buy_option(tradingsymbol, quantity=int(quantity))
            st.write(resp)

st.subheader("Active (simulated) Orders")
if om.active_orders:
    df_active = pd.DataFrame([{
        "order_id": oid,
        "symbol": m['symbol'],
        "entry_price": m.get('entry_price'),
        "sl_price": m.get('sl_price'),
        "quantity": m.get('quantity')
    } for oid,m in om.active_orders.items()])
    st.dataframe(df_active)
else:
    st.write("No active simulated orders.")

st.subheader("Paper trades log")
if os.path.exists(LOG_CSV):
    df_log = pd.read_csv(LOG_CSV)
    st.dataframe(df_log.tail(50))
else:
    st.write("No trades logged yet.")

st.markdown("---")
st.markdown("**Note:** This app simulates fills and orders for testing strategy logic. Replace the PaperBroker with the FyersWrapper to enable real API calls (live mode).")

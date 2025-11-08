
# FYERS Streamlit Options Trader — PAPER mode (Ready-to-run)

This package contains a Streamlit app preconfigured to run in **paper/demo** mode. It simulates order placement and logs simulated trade fills to `paper_trades_log.csv` so you can validate strategy logic without placing live orders.

## Files
- `streamlit_option_trader_fyers_paper.py` — Streamlit app (paper mode)
- `config_example.json` — an example configuration file (not required for paper mode)
- `paper_trades_log.csv` — simulated trade log (created when the app runs)
- `README.md` — this file

## Requirements
```bash
pip install streamlit pandas numpy requests
```

FYERS SDK is not required for paper mode. If you later want to enable live trading with FYERS, install the official FYERS SDK and replace the `PaperBroker` with `FyersWrapper` in the code (comment provided in the code).

## Running
```bash
streamlit run streamlit_option_trader_fyers_paper.py
```

## Warning
This is a simulation. Do not enable live trading until you have thoroughly tested the application with paper mode and confirmed behavior.

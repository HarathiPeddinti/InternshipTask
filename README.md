# Trading Behavior vs Market Sentiment Analysis

## Project Overview
This project analyzes how trader performance varies with market sentiment using trading data and the Fear-Greed Index.

---

## Dataset Description
1. Fear & Greed Index — Daily sentiment classification
2. Historical Trades — Trader transactions

---

## Methodology
- Loaded and cleaned datasets
- Converted timestamps to daily format
- Merged sentiment with trades
- Created metrics:
  - Daily PnL
  - Win Rate
  - Trade Size
- Segmented traders by activity & profitability
- Compared performance across sentiment categories

---

## How to Run

### Option 1 — Script
```bash
pip install -r requirements.txt
python script/analysis.py

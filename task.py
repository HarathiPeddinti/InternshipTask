

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sentiment = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data (1).csv")

print("\nSentiment Shape:", sentiment.shape)
print("Trades Shape:", trades.shape)

# 3. DATA INSPECTION
print("\n--- SENTIMENT INFO ---")
print(sentiment.info())
print(sentiment.isnull().sum())
print("Duplicates:", sentiment.duplicated().sum())

print("\n--- TRADES INFO ---")
print(trades.info())
print(trades.isnull().sum())
print("Duplicates:", trades.duplicated().sum())

# 4. DATA CLEANING
# Convert dates
sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date
trades["Timestamp IST"] = pd.to_datetime(trades["Timestamp IST"], format="%d-%m-%Y %H:%M")
trades["date"] = trades["Timestamp IST"].dt.date

# Drop duplicates
sentiment.drop_duplicates(inplace=True)
trades.drop_duplicates(inplace=True)

# 5. MERGE DATASETS
merged = trades.merge(
    sentiment[["date","classification"]],
    on="date",
    how="left"
)

print("\nMerged Shape:", merged.shape)

# 6. FEATURE ENGINEERING
# Win or Loss trade
merged["win"] = merged["Closed PnL"] > 0

# Trade size absolute
merged["abs_size"] = merged["Size USD"].abs()

# Daily PnL per trader
daily_pnl = (
    merged.groupby(["Account","date"])["Closed PnL"]
    .sum()
    .reset_index()
)

# Trades per day
trades_per_day = merged.groupby("date").size()

# Winrate per trader
winrate = merged.groupby("Account")["win"].mean()

# Total profit per trader
profit_per_trader = merged.groupby("Account")["Closed PnL"].sum()

# 7. ANALYSIS — PERFORMANCE VS SENTIMENT
print("\nAverage PnL by Sentiment:")
print(merged.groupby("classification")["Closed PnL"].mean())

print("\nWin Rate by Sentiment:")
print(merged.groupby("classification")["win"].mean())

print("\nAvg Trade Size by Sentiment:")
print(merged.groupby("classification")["abs_size"].mean())

# 8. BEHAVIOR ANALYSIS
# Long vs Short
long_short = pd.crosstab(merged["classification"], merged["Side"])
print("\nLong Short Distribution")
print(long_short)


# Trades per sentiment day
trades_sentiment = merged.groupby("classification").size()
print("\nTrades count per sentiment:")
print(trades_sentiment)

# 9. TRADER SEGMENTATION
# Activity segmentation
trade_counts = merged.groupby("Account").size()

def activity_level(x):
    if x > 100:
        return "High"
    elif x > 20:
        return "Medium"
    else:
        return "Low"

activity_map = trade_counts.map(activity_level)
merged["activity"] = merged["Account"].map(activity_map)


# Profitability segmentation
profit_map = profit_per_trader.map(
    lambda x: "Profitable" if x > 0 else "Losing"
)
merged["profitability"] = merged["Account"].map(profit_map)

# Segment comparison
print("\nPnL by Activity Group:")
print(merged.groupby("activity")["Closed PnL"].mean())

print("\nPnL by Profitability Group:")
print(merged.groupby("profitability")["Closed PnL"].mean())

# 10. VISUALIZATIONS
# Avg PnL vs Sentiment
plt.figure()
merged.groupby("classification")["Closed PnL"].mean().plot(kind="bar")
plt.title("Average PnL by Sentiment")
plt.ylabel("Average PnL")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Winrate vs Sentiment
plt.figure()
merged.groupby("classification")["win"].mean().plot(kind="bar")
plt.title("Win Rate by Sentiment")
plt.ylabel("Win Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Trade count vs Sentiment
plt.figure()
merged.groupby("classification").size().plot(kind="bar")
plt.title("Number of Trades by Sentiment")
plt.ylabel("Trades")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 11. KEY INSIGHTS AUTO-GENERATOR

best_sentiment = merged.groupby("classification")["Closed PnL"].mean().idxmax()
worst_sentiment = merged.groupby("classification")["Closed PnL"].mean().idxmin()

print("\n===== INSIGHTS =====")
print("Best performance sentiment:", best_sentiment)
print("Worst performance sentiment:", worst_sentiment)

print("\nStrategy Suggestions:")
print("1. Trade more during", best_sentiment, "market conditions.")
print("2. Reduce risk exposure during", worst_sentiment, "days.")
print("3. High-activity traders should be monitored for drawdowns.")

# 12. SAVE OUTPUT FILES
merged.to_csv("merged_dataset.csv", index=False)
daily_pnl.to_csv("daily_pnl.csv", index=False)

print("\nFiles saved:")
print("✔ merged_dataset.csv")
print("✔ daily_pnl.csv")

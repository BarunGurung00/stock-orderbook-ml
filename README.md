# stock-orderbook-ml

# Short-Term Stock Price Prediction via Order Book Machine Learning

This repository implements a quantitative machine learning workflow designed to predict short-term stock price movements using high-frequency order book dynamics and market flow metrics.

## 📌 Overview

Short-term stock price changes can often be inferred from liquidity imbalances within the limit order book. This project constructs order book features from raw tick data—specifically focusing on Level 1 (L1), Level 2 (L2), and Level 3 (L3) order book imbalances alongside Money Flow (MF) indicators—to classify the next immediate price direction into binary targets:
* **Up (`u`)**
* **Down (`d`)**

All feature vectors are normalized into a standardized $[-1, 1]$ range prior to model training.

---

## 🛠️ Key Features & Methodology

1. **Order Book Construction**: Converts raw tick data into structured L1, L2, and L3 depth snapshots.
2. **Feature Engineering**:
   * **Order Book Imbalance (OBI)** across multiple depth levels.
   * **Money Flow (MF)** indicators to measure volume-weighted buying/selling pressure.
3. **Data Preprocessing**: Feature scaling and normalization to a $[-1, 1]$ bounding box.
4. **Supervised Classification**: Training machine learning models to predict directional price shifts.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries.

```bash
pip install -r requirements.txt
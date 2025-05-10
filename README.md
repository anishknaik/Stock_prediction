# Stock Price Prediction Project

## Overview

This repository contains code, data, and reports for forecasting stock prices using a combination of time-series and machine learning models. The goal is to compare the performance of ARIMA, LSTM, and ensemble methods on historical stock data fetched via the Alpha Vantage API.

## Features

* **Data ingestion**: Automated download of historical stock prices using Alpha Vantage.
* **Data processing**: Cleaning, normalization, and feature engineering.
* **Modeling**:

  * ARIMA time-series forecasting
  * LSTM neural network prediction
  * Ensemble of ARIMA and LSTM
* **Evaluation**: Metrics and visualization of actual vs. predicted prices.
* **Reporting**: LaTeX report generation with figures and code snippets.

## Repository Structure

```
├── data/
│   ├── raw/                # Original downloaded CSV files
│   └── processed/          # Cleaned and feature-engineered data
├── graphs/                 # Generated plots (actual vs. predicted, metrics)
├── models/                 # Saved model weights and artifacts
├── notebooks/              # Jupyter notebooks for exploration and prototyping
├── reports/                # LaTeX source and compiled PDF report
├── src/                    # Python modules (data loading, preprocessing, modeling)
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── arima_model.py
│   ├── lstm_model.py
│   └── ensemble.py
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── .env.example            # Environment variables template
```

## Installation

### Prerequisites

* Python 3.8 or higher
* An Alpha Vantage API key (free at [https://www.alphavantage.co](https://www.alphavantage.co))

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your Alpha Vantage API key:

   ```ini
   ALPHA_VANTAGE_API_KEY=YOUR_API_KEY_HERE
   ```

## Usage

### 1. Download and preprocess data

```bash
python src/data_loader.py --symbol AAPL --output data/raw/AAPL.csv
python src/preprocess.py --input data/raw/AAPL.csv --output data/processed/AAPL_clean.csv
```

### 2. Train and evaluate models

```bash
# ARIMA
python src/arima_model.py --input data/processed/AAPL_clean.csv --output graphs/arima_forecast.png

# LSTM
python src/lstm_model.py --input data/processed/AAPL_clean.csv --output graphs/lstm_forecast.png

# Ensemble
python src/ensemble.py --input data/processed/AAPL_clean.csv \
    --arima-model models/arima_AAPL.pkl \
    --lstm-model models/lstm_AAPL.h5 \
    --output graphs/ensemble_forecast.png
```

### 3. Generate report

Navigate to the `reports/` directory and compile the LaTeX report:

```bash
cd reports
pdflatex stock_price_report.tex
```

## Results

* View generated plots in the `graphs/` folder.
* Find the compiled PDF report at `reports/stock_price_report.pdf`.

## Contributing

Contributions and suggestions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Anish Naik – [your.email@example.com](mailto:your.email@example.com)

Project repository: [https://github.com/](https://github.com/)<your-username>/stock-price-prediction

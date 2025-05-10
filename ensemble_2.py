import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
import matplotlib.pyplot as plt

# ─── 0) YOUR API KEY ──────────────────────────────────────────────────────────
API_KEY = "AEJG2DMCDEXNP66O"


def load_stock_data(
    api_key: str, symbol: str, interval: str = "daily", output_size: str = "full"
) -> pd.DataFrame:
    func_map = {
        "daily": "TIME_SERIES_DAILY",
        "weekly": "TIME_SERIES_WEEKLY",
        "monthly": "TIME_SERIES_MONTHLY",
    }
    func = func_map.get(interval, "TIME_SERIES_DAILY")
    params = {
        "function": func,
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": output_size,
    }
    if interval not in func_map:
        params["interval"] = interval
    r = requests.get("https://www.alphavantage.co/query", params=params)
    data = r.json()
    ts_key = [k for k in data if "Time Series" in k][-1]
    df = pd.DataFrame.from_dict(data[ts_key], orient="index").astype(float)
    df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        },
        inplace=True,
    )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # SMA, EMA, RSI, MACD
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # OBV
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    # Bollinger %B
    m = df["close"].rolling(20).mean()
    s = df["close"].rolling(20).std()
    df["BBpct"] = (df["close"] - m) / (2 * s)
    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df.fillna(method="bfill", inplace=True)
    return df


def create_sequences(values: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    for i in range(len(values) - seq_len - (horizon - 1)):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len + horizon - 1, 3])  # close price
    return np.array(X), np.array(y)


def build_lstm_model(seq_len: int, n_feat: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=(seq_len, n_feat), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


def build_cnn_model(seq_len: int, n_feat: int) -> Sequential:
    model = Sequential(
        [
            Conv1D(64, 3, activation="relu", input_shape=(seq_len, n_feat)),
            MaxPooling1D(2),
            Dropout(0.2),
            Conv1D(32, 3, activation="relu"),
            Flatten(),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


def get_oof_test_preds(models: dict, X_tr, y_tr, X_te, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    oof = {name: np.zeros(len(y_tr)) for name in models}
    test_preds = {}
    for tr_idx, val_idx in kf.split(X_tr):
        Xt, Xv = X_tr[tr_idx], X_tr[val_idx]
        yt = y_tr[tr_idx]
        for name, m in models.items():
            if name == "xgb":
                m.fit(Xt.reshape(len(Xt), -1), yt)
                oof[name][val_idx] = m.predict(Xv.reshape(len(Xv), -1))
            else:
                m.fit(
                    Xt,
                    yt,
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                )
                oof[name][val_idx] = m.predict(Xv).flatten()
    for name, m in models.items():
        if name == "xgb":
            m.fit(X_tr.reshape(len(X_tr), -1), y_tr)
            test_preds[name] = m.predict(X_te.reshape(len(X_te), -1))
        else:
            m.fit(
                X_tr,
                y_tr,
                epochs=20,
                batch_size=32,
                verbose=0,
                validation_split=0.1,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            )
            test_preds[name] = m.predict(X_te).flatten()
    return oof, test_preds


def main():
    api_key = API_KEY
    symbol = input("Symbol (e.g. AAPL): ").strip().upper()
    seq_len = int(input("Sequence length (days): ").strip())
    horizon = 1

    # 1) load & features
    df = load_stock_data(api_key, symbol)
    df = add_technical_indicators(df)
    feat_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "SMA_20",
        "EMA_20",
        "RSI",
        "MACD",
        "OBV",
        "BBpct",
        "ATR",
    ]
    data = df[feat_cols].values

    # 2) scale & sequences
    scaler = MinMaxScaler().fit(data)
    data_s = scaler.transform(data)
    X, y = create_sequences(data_s, seq_len, horizon)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3) base learners
    bases = {
        "lstm": build_lstm_model(seq_len, X.shape[2]),
        "cnn": build_cnn_model(seq_len, X.shape[2]),
        "xgb": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }
    oof, test_preds = get_oof_test_preds(bases, Xtr, ytr, Xte)

    # 4) regression stack
    meta_X_tr = np.column_stack([oof[n] for n in bases])
    meta_X_te = np.column_stack([test_preds[n] for n in bases])
    Mreg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    Mreg.fit(meta_X_tr, ytr)
    y_pred = Mreg.predict(meta_X_te)

    # invert
    inv = np.zeros((len(yte), data.shape[1]))
    inv[:, 3] = yte
    true = scaler.inverse_transform(inv)[:, 3]
    inv[:, 3] = y_pred
    pred = scaler.inverse_transform(inv)[:, 3]

    # regression metrics
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.3f}  MAPE: {mape:.2f}%")

    # 5) classification stack on direction
    # build directional labels
    ytr_dir = (ytr[1:] > ytr[:-1]).astype(int)
    yte_dir = (yte[1:] > yte[:-1]).astype(int)
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    clf.fit(meta_X_tr[:-1], ytr_dir)
    dir_pred = clf.predict(meta_X_te[:-1])
    dir_acc = accuracy_score(yte_dir, dir_pred) * 100
    print(f"Stacked Directional Accuracy: {dir_acc:.2f}%")

    # 6) tuned epsilon backtest
    epsilons = np.linspace(0, 0.02, 21)
    best_dir_eps, best_eps = 0, 0
    for eps in epsilons:
        pos = (pred[1:] > true[:-1] * (1 + eps)).astype(int)
        da = (pos == (true[1:] > true[:-1])).mean() * 100
        if da > best_dir_eps:
            best_dir_eps, best_eps = da, eps
    print(f"Best ε={best_eps:.4f} → Dir Acc={best_dir_eps:.2f}%")

    # 7) backtest
    pos = np.concatenate([[0], (pred[1:] > true[:-1] * (1 + best_eps)).astype(int)])
    ret = np.concatenate([[0], np.diff(true) / true[:-1]])
    strat = pos * (ret - 0.0005)
    bh = ret
    cum_s = np.cumprod(1 + strat) - 1
    cum_b = np.cumprod(1 + bh) - 1
    sharpe = strat.mean() / strat.std() * np.sqrt(252)
    print(f"Strategy Sharpe Ratio: {sharpe:.2f}")

    # plots
    plt.figure(figsize=(10, 5))
    plt.plot(true, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.title(f"Actual vs Predicted for {symbol}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-len(cum_s) :], cum_s, label="Strategy")
    plt.plot(df.index[-len(cum_b) :], cum_b, label="Buy & Hold")
    plt.title("Cumulative Returns")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

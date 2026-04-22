from __future__ import annotations

import pandas as pd
import requests

from .config import BINANCE_BASE_URL, INTERVAL, INTERVAL_MS, KLINES_ENDPOINT, MAX_LIMIT, SYMBOLS


class BinanceDataFetcher:
    def __init__(self, base_url: str = BINANCE_BASE_URL, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def fetch_klines(self, symbol: str, start_time_ms: int, end_time_ms: int, interval: str = INTERVAL) -> pd.DataFrame:
        if start_time_ms >= end_time_ms:
            raise ValueError("start_time_ms must be earlier than end_time_ms.")

        rows = []
        next_start = start_time_ms

        while next_start < end_time_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": int(next_start),
                # end_time_ms duoc hieu la moc chan ben phai, khong lay chinh moc do.
                "endTime": int(end_time_ms - 1),
                "limit": MAX_LIMIT,
            }
            url = f"{self.base_url}{KLINES_ENDPOINT}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()

            if not payload:
                break

            # Chi giu cac truong dung o buoc sau de giam RAM va dung luong file.
            rows.extend((item[0], item[4], item[5]) for item in payload)

            last_open_time = int(payload[-1][0])
            next_start = last_open_time + INTERVAL_MS

            if len(payload) < MAX_LIMIT:
                break

        df = pd.DataFrame(rows, columns=["open_time", "close", "volume"])
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "close", "volume"])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df[["timestamp", "close", "volume"]]

    def fetch_crypto_pair(self, start_time_ms: int, end_time_ms: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        btc = self.fetch_klines(SYMBOLS["BTC"], start_time_ms, end_time_ms)
        eth = self.fetch_klines(SYMBOLS["ETH"], start_time_ms, end_time_ms)
        return btc, eth

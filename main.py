import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px

class MeanReversionETF:
    def __init__(self, tickers, startdate, enddate):
        self.tickers = tickers
        self.equity = 100000
        self.position = {t: 'None' for t in tickers}
        self.open_price = {t: 0 for t in tickers}
        self.open_date = {t: None for t in tickers}
        self.last_price = {t: 0 for t in tickers}
        self.last_date = {t: None for t in tickers}
        self.prev_z = {t: 0 for t in tickers}
        self.rolling_prices = {t: [] for t in tickers}
        self.fees = 0 
        self.trades_info = []
        self.overall_pnl = []
        self.total_fees = 0
        self.position_size = self.equity / len(tickers)

        self.prices = yf.download(tickers, start=startdate, end=enddate)['Close']
        if isinstance(self.prices, pd.Series):
            self.prices = self.prices.to_frame()

        self.run_strategy()
        self.compile_results()

    def run_strategy(self):
        for date, row in self.prices.iterrows():
            for ticker in self.tickers:
                price = row.get(ticker)
                if np.isnan(price):
                    continue

                series = self.rolling_prices[ticker]
                series.append(price)
                if len(series) < 20:
                    continue
                elif len(series) > 20:
                    series.pop(0)

                mean = np.mean(series)
                std = np.std(series)
                if std == 0:
                    continue

                z = (price - mean) / std

                # Stop loss logic
                if self.position[ticker] != 'None':
                    current_pnl = (self.open_price[ticker] - price) if self.position[ticker] == 'Short' else (price - self.open_price[ticker])
                    if current_pnl < -0.10 * self.position_size:
                        pnl = current_pnl
                        self.equity += pnl
                        self.overall_pnl.append((self.position[ticker], self.open_date[ticker], date, ticker, pnl, self.fees))
                        self.position[ticker] = 'None'
                        continue

                # Exit logic
                if self.position[ticker] != 'None' and ((self.prev_z[ticker] > 0 and z <= 0) or (self.prev_z[ticker] < 0 and z >= 0)):
                    pnl = (self.open_price[ticker] - price) if self.position[ticker] == 'Short' else (price - self.open_price[ticker])
                    self.equity += pnl
                    self.overall_pnl.append((self.position[ticker], self.open_date[ticker], date, ticker, pnl, self.fees))
                    self.position[ticker] = 'None'

                # Entry logic
                if self.position[ticker] == 'None':
                    if z > 1:
                        self.position[ticker] = 'Short'
                        self.open_price[ticker] = price
                        self.open_date[ticker] = date
                        self.fees = 0.001 * price
                        self.total_fees += self.fees
                        self.equity -= self.fees
                    elif z < -1:
                        self.position[ticker] = 'Long'
                        self.open_price[ticker] = price
                        self.open_date[ticker] = date
                        self.fees = 0.001 * price
                        self.total_fees += self.fees
                        self.equity -= self.fees

                self.prev_z[ticker] = z
                self.last_price[ticker] = price
                self.last_date[ticker] = date

    def compile_results(self):
        for ticker in self.tickers:
            if self.position[ticker] != 'None':
                pnl = (self.open_price[ticker] - self.last_price[ticker]) if self.position[ticker] == 'Short' else (self.last_price[ticker] - self.open_price[ticker])
                self.equity += pnl
                self.overall_pnl.append((self.position[ticker], self.open_date[ticker], self.last_date[ticker], ticker, pnl, self.fees))

        records = pd.DataFrame(self.overall_pnl, columns=['Position', 'Open Date', 'Exit Date', 'Ticker', 'PnL', 'Fees'])

        total_pnl = records['PnL'].sum()
        total_return = (self.equity / 100000 - 1) * 100
        pnl_series = records['PnL']
        sharpe_ratio = pnl_series.mean() / pnl_series.std() if len(pnl_series) > 1 else np.nan
        info_ratio = pnl_series.mean() / pnl_series.std() if len(pnl_series) > 1 else np.nan
        winrate = (pnl_series > 0).mean() * 100
        equity_curve = 100000 + pnl_series.cumsum()
        roll_max = equity_curve.cummax()
        drawdowns = (roll_max - equity_curve) / roll_max
        max_drawdown = drawdowns.max() * 100

        print(f"\nFinal Equity:      {self.equity:,.2f}")
        print(f"Fees:              {self.total_fees :.2f}")
        print(f"Total Trades:      {len(records)}")
        print(f"Total PnL:         {total_pnl:,.2f}")
        print(f"Return:            {total_return:.2f}%")
        print(f"Sharpe Ratio:      {sharpe_ratio:.2f}")
        print(f"Information Ratio: {info_ratio:.2f}")
        print(f"Winrate:           {winrate:.2f}%")
        print(f"Max Drawdown:      {max_drawdown:.2f}%")

        print("\n", records)

        records = records.sort_values(by = "Exit Date")
        initial_equity = 100000
        records['Cumulative Fees'] = records['Fees'].fillna(0).cumsum()
        records['Equity'] = initial_equity + records['PnL'].cumsum() - records['Cumulative Fees']

        plt.figure(figsize=(10, 5))
        plt.plot(records['Exit Date'], records['Equity'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        fig1 = px.line(
        records,
        x='Exit Date',
        y='Equity',
        labels={'Exit Date': 'Date', 'Equity': 'Equity $'},
        title='Equity Curve'
        )

        records['Adjusted Equity'] = records['Equity'] - 100000

        fig2 = px.area(
            records, 
            x = 'Exit Date',
            y = 'Adjusted Equity',
            labels={'Exit Date': 'Date', 'Adjusted Equity': 'PnL'},
            title='PnL'
        )

        fig3 = px.line(
        records,
        x='Exit Date',
        y='Equity',
        labels={'Exit Date': 'Date', 'Equity': 'Equity $'},
        title='Equity Curve'
        ) 

        fig3.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
        buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all", label = "Max")
                ])
            )
        )

        fig1.show()
        fig2.show()
        fig3.show()

# Usage
if __name__ == '__main__':
    tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    startdate = '2020-01-01'
    enddate = None
    MeanReversionETF(tickers, startdate, enddate)

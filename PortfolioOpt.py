import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
yf.pdr_override()
from scipy import optimize
from datetime import date, timedelta




class PortfolioOpt:

    def __init__(self, portfolio, start, end=None, benchmark="^GSPC"):

        self.tickers = {}
        tickers_skipped = 0
        for k in portfolio:
            try:
                self.tickers[k] = yf.Ticker(k).info['shortName']
            except:
                tickers_skipped += 1
                continue
        # print(tickers_skipped)


        self.start = start
        if end is None:
            self.end = str(date.today() - timedelta(days=1))
        else:
            self.end = end


        str_lengths = [len(f"{v} ({k}):") for k,v in self.tickers.items()]
        self.max_str = max(str_lengths) + 1

        self.data = pdr.get_data_yahoo(list(self.tickers.keys()), start=self.start, end=self.end)["Adj Close"]
        self.log_ret = np.log(self.data/self.data.shift(1))

        self.benchmark = benchmark
        self.bench = pdr.get_data_yahoo(self.benchmark, start=self.start, end=self.end)["Adj Close"]
        self.bench_ret = np.log(self.bench/self.bench.shift(1))
        self.bench_ret.rename(self.benchmark, axis=1, inplace=True)

        self.bench_er = (self.bench_ret.mean() * 252).round(3)
        self.bench_vol = (self.bench_ret.std() * np.sqrt(252)).round(3)
        self.bench_sharpe = (self.bench_er/self.bench_vol).round(3)

    def refresh_data(self, start=None, end=None):

        if start is not None:
            self.start=start

        if end is not None:
            self.end=end
        else:
            self.end = str(date.today() - timedelta(days=1))

        self.data = pdr.get_data_yahoo(list(self.tickers.keys()), start=self.start, end=self.end)["Adj Close"]
        self.log_ret = np.log(self.data/self.data.shift(1))


        self.bench = pdr.get_data_yahoo(self.benchmark, start=self.start, end=self.end)["Adj Close"]
        self.bench_ret = np.log(self.bench/self.bench.shift(1))
        self.bench_ret.rename(self.benchmark, axis=1, inplace=True)


    def _get_ret_vol_sr(self, weights):
        """
        Calculates the returns, volatility, and sharpe of a portfolio with given weights
        """
        weights = np.array(weights)
        ret = np.sum(self.log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(self.log_ret.cov()*252, weights)))
        sr = ret/vol
        return np.array([ret, vol, sr])

    def _neg_sharpe(self, weights):
        return self._get_ret_vol_sr(weights)[2] * -1

    def _neg_returns(self, weights):
        return self._get_ret_vol_sr(weights)[0] * -1

    def _minimize_volatility(self, weights):
        return self._get_ret_vol_sr(weights)[1]

    def _neg_beta(self, weights, beta_vec):
        return np.dot(beta_vec, x) * -1



    def optimize_portfolio(self, opt_for="sharpe", bounds=None, print_results=True, **kwargs):
        """
        Optimize portfolio buy maximizing sharpe, maximizing returns, or minimizing volatility
        """

        cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
        if bounds is None:
            bounds = tuple((0,1) for _ in range(len(self.tickers)))
        init_guess = [1/len(self.tickers) for _ in range(len(self.tickers))]

        if opt_for == "sharpe":
            opt_results = optimize.minimize(self._neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "returns":
            opt_results = optimize.minimize(self._neg_returns, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "vol":
            opt_results = optimize.minimize(self._minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        else:
            raise ValueError(f"'opt_for' can only take the values 'sharpe', 'returns', or 'vol'")

        if print_results:

            self.print_results(opt_results, **kwargs)

        return opt_results


    def optimize_portfolio_returns(self, target_returns, opt_for="sharpe", bounds=None, print_results=True, **kwargs):
        """
        Optimize portfolio buy maximizing sharpe, maximizing returns, or minimizing volatility with a target return
        """

        cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1},
                {'type':'eq', 'fun': lambda x: self._get_ret_vol_sr(x)[0] - target_returns})
        if bounds is None:
            bounds = tuple((0,1) for _ in range(len(self.tickers)))
        init_guess = [1/len(self.tickers) for _ in range(len(self.tickers))]

        if opt_for == "sharpe":
            opt_results = optimize.minimize(self._neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "returns":
            opt_results = optimize.minimize(self._neg_returns, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "vol":
            opt_results = optimize.minimize(self._minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        else:
            raise ValueError(f"'opt_for' can only take the values 'sharpe', 'returns', or 'vol'")

        if print_results:

            self.print_results(opt_results, **kwargs)

        return opt_results


    def optimize_portfolio_vol(self, target_vol, opt_for="sharpe", bounds=None, print_results=True, **kwargs):
        """
        Optimize portfolio buy maximizing sharpe, maximizing returns, or minimizing volatility with a target volatility
        """

        cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1},
                {'type':'eq', 'fun': lambda x: self._get_ret_vol_sr(x)[1] - target_vol})
        if bounds is None:
            bounds = tuple((0,1) for _ in range(len(self.tickers)))
        init_guess = [1/len(self.tickers) for _ in range(len(self.tickers))]

        if opt_for == "sharpe":
            opt_results = optimize.minimize(self._neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "returns":
            opt_results = optimize.minimize(self._neg_returns, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "vol":
            opt_results = optimize.minimize(self._minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        else:
            raise ValueError(f"'opt_for' can only take the values 'sharpe', 'returns', or 'vol'")

        if print_results:

            self.print_results(opt_results, **kwargs)

        return opt_results

    def optimize_portfolio_beta(self, beta_target, opt_for="sharpe", bounds=None, print_results=True, **kwargs):
        """
        Optimize portfolio buy maximizing sharpe, maximizing returns, or minimizing volatility with a target beta with the benchmark
        """
        t = pd.concat([self.bench_ret, self.log_ret], axis=1)

        # Betas vector
        hist_covs = np.array(t.cov().iloc[0])
        bench_var = hist_covs[0]
        beta_vec = hist_covs[1:]/bench_var


        cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1},
                {'type':'eq', 'fun': lambda x: np.dot(beta_vec, x)-beta_target})
        if bounds is None:
            bounds = tuple((0,1) for _ in range(len(self.tickers)))
        init_guess = [1/len(self.tickers) for _ in range(len(self.tickers))]

        if opt_for == "sharpe":
            opt_results = optimize.minimize(self._neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "returns":
            opt_results = optimize.minimize(self._neg_returns, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        elif opt_for == "vol":
            opt_results = optimize.minimize(self._minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        else:
            raise ValueError(f"'opt_for' can only take the values 'sharpe', 'returns', or 'vol'")

        if print_results:

            self.print_results(opt_results, beta_target=beta_target, **kwargs)

        return opt_results




    def print_results(self, opt_results, print_zeros=False, percentages=True, shares=True, dollars=True, amount=1000, beta_target=None):
        """
        A function to print the results of the optimization.
        """


        print(f"Data start:" + " "*(self.max_str-len("Data start:")) + f"{self.start}")
        print(f"Data end:" + " "*(self.max_str-len("Data end:")) + f"{self.end}\n")

        print("\nPerformance (Annualized)")
        print(f"Returns:" + " "*(self.max_str-len("Returns:")) + f"{round(self._get_ret_vol_sr(opt_results['x'])[0],3)}")
        print(f"Vol:" + " "*(self.max_str-len("Vol:")) + f"{round(self._get_ret_vol_sr(opt_results['x'])[1],3)}")
        print(f"Sharpe ratio:" + " "*(self.max_str-len("Sharpe ratio:")) + f"{round(self._get_ret_vol_sr(opt_results['x'])[2],3)}")
        if beta_target is not None:
            print(f"Beta Target:" + " "*(self.max_str-len("Beta Target:")) + f"{beta_target}")

        print(f"\nBenchmark ({self.benchmark})")
        print(f"Returns:" + " "*(self.max_str-len("Returns:")) + f"{self.bench_er}")
        print(f"Vol:" + " "*(self.max_str-len(f"Vol:")) + f"{self.bench_vol}")
        print(f"Sharpe ratio:" + " "*(self.max_str-len(f"Sharpe ratio:")) + f"{self.bench_sharpe}")



        if percentages:
            print("\n" + "#"*(self.max_str+10))
            print("Optimal Percentages")
            for i, (k, v) in zip(opt_results['x'], self.tickers.items()):
                space = self.max_str - len(f"{v} ({k}):")

                value = round(i*100, 2)
                if value == 0 and not print_zeros:
                    continue

                print(f"{v} ({k}):" + " "*space + f"{value}")

        if shares:

            print("\n" + "#"*(self.max_str+10))
            print(f"Shares to buy (${amount})")
            dol_inv = opt_results['x'] * amount
            last_price = self.data.iloc[-1].values
            shares = np.floor(dol_inv/last_price)


            for i, (k,v) in zip(shares, self.tickers.items()):
                space = self.max_str - len(f"{v} ({k}):")

                if i == 0 and not print_zeros:
                    continue

                print(f"{v} ({k}):" + " "*space + f"{i}")


        if dollars:

            print("\n" + "#"*(self.max_str+10))
            print(f"Dollars to buy (${amount})")
            dol_inv = opt_results['x'] * amount

            for i, (k,v) in zip(dol_inv.round(2), self.tickers.items()):
                space = self.max_str - len(f"{v} ({k}):")

                if i == 0 and not print_zeros:
                    continue

                print(f"{v} ({k}):" + " "*space + f"{i}")



if __name__ == "__main__":

    # tickers = ['XAR', 'KBE', 'XBI', 'KCE', 'XHE', 'XHS', 'XHB', 'KIE', 'XWEB', 'XME', 'XES', 'XOP', 'XPH', 'KRE', 'XRT', 'XSD', 'XSW', 'XTL', 'XTN']
    # tickers = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    # tickers = ["FB", "AAPL", "AMZN", "NFLX", "GOOG"]
    # ssmif = pd.read_csv("ssmif_port.csv", header=None)
    # tickers = ssmif[0].values

    spdr = pd.read_csv("spdr_holdings-all.csv")
    tickers = spdr['Symbol'].unique()

    from dateutil.relativedelta import relativedelta
    start = str(date.today() - relativedelta(years=3))
    opt = PortfolioOpt(tickers, start=start)
    t = opt.optimize_portfolio()
    # print(t)
    # opt.print_results(t, amount=180)

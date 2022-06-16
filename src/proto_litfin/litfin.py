import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import investpy
import numpy as np
import requests
from pyjstat import pyjstat
import seaborn as sns
import time

import matplotlib.pyplot as plt

from IPython.display import Markdown, display

"""
Begin of compute_drawdowns_table function
"""


# Função auxiliar 1
def compute_time_period(timestamp_1, timestamp_2):

    year = timestamp_1.year - timestamp_2.year
    month = timestamp_1.month - timestamp_2.month
    day = timestamp_1.day - timestamp_2.day

    if month < 0:
        year = year - 1
        month = 12 + month

    if day == 0:
        day = -day

    if day < 0:
        month = month - 1
        if timestamp_1.month not in [1, 3, 5, 7, 8, 10, 12]:
            day = 31 + day
        else:
            day = 30 + day

    # Returns datetime object in years, month, days
    return str(year) + " Years, " + str(month) + " Months, " + str(day) + " Days"


# Função auxiliar 2
def compute_drawdowns_periods(df):

    # Input: df of max points in drawdowns (where dd == 0)

    drawdown_periods = list()

    for i in range(0, len(df.index)):

        drawdown_periods.append(compute_time_period(df.index[i], df.index[i - 1]))

    drawdown_periods = pd.DataFrame(drawdown_periods)

    return drawdown_periods


# Função auxiliar 3
def compute_max_drawdown_in_period(prices, timestamp_1, timestamp_2):

    df = prices[timestamp_1:timestamp_2]

    max_dd = compute_max_DD(df)

    return max_dd


# Função auxiliar 4
def compute_drawdowns_min(df, prices):

    # Input: df of max points in drawdowns (where dd == 0)

    drawdowns_min = list()

    for i in range(0, len(df.index) - 1):

        drawdowns_min.append(
            compute_max_drawdown_in_period(prices, df.index[i], df.index[i + 1])
        )

    drawdowns_min = pd.DataFrame(drawdowns_min)

    return drawdowns_min


# Multi_period_return (in CAGR)
def multi_period_return(df, years=1, days=252):
    shifted = df.shift(days * years)
    One_year = (((1 + (df - shifted) / shifted) ** (1 / years)) - 1) * 100
    return One_year


# Função principal
def compute_drawdowns_table(prices, number=5):

    # input: df of prices
    dd = compute_drawdowns(prices)

    max_points = dd[dd == 0].dropna()

    data = [0.0]

    # Create the pandas DataFrame
    new_data = pd.DataFrame(data, columns=["New_data"])

    new_data["Date"] = prices.index.max()

    new_data.set_index("Date", inplace=True)

    max_points = max_points.loc[~max_points.index.duplicated(keep="first")]

    max_points = pd.DataFrame(pd.concat([max_points, new_data], axis=1).iloc[:, 0])

    dp = compute_drawdowns_periods(max_points)

    dp.set_index(max_points.index, inplace=True)

    df = pd.concat([max_points, dp], axis=1)

    df.index.name = "Date"

    df.reset_index(inplace=True)

    df["End"] = df["Date"].shift(-1)

    df[0] = df[0].shift(-1)

    df["values"] = round(compute_drawdowns_min(max_points, prices), 2)

    df = df.sort_values(by="values")

    df["Number"] = range(1, len(df) + 1)

    df.reset_index(inplace=True)

    df.columns = ["index", "Begin", "point", "Length", "End", "Depth", "Number"]

    df = df[["Begin", "End", "Depth", "Length"]].head(number)

    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: str(x) + "%")

    df.set_index(np.arange(1, number + 1), inplace=True)

    df["End"] = df["End"].astype(str)

    df["Begin"] = df["Begin"].astype(str)

    for i in range(0, len(df["End"])):
        if df["End"].iloc[i] == str(prices.iloc[-1].name)[0:10]:
            df["End"].iloc[i] = str("N/A")

    return df


def compute_rolling_cagr(dataframe, years):
    index = dataframe.index + pd.DateOffset(years=years)

    start = dataframe.index[0]
    end = dataframe.index[-1]

    portfolio = dataframe.copy()
    portfolio.set_index(index, inplace=True)

    portfolio = portfolio[start:]

    rr = (dataframe.iloc[:, 0] / portfolio.iloc[:, 0] - 1) * 100
    rr = rr.loc[:end]

    return pd.DataFrame(((((rr / 100) + 1)) ** (1 / years)) - 1)


def color_negative_red(value):
    """
    Colors elements in a dateframe
    green if positive and red if
    negative. Does not color NaN
    values.
    """

    if value < 0:
        color = "red"
    elif value > 0:
        color = "green"
    else:
        color = "black"

    return "color: %s" % color


def compute_yearly_returns(
    dataframe,
    start="1900",
    end="2100",
    style="table",
    title="Yearly Returns",
    color=False,
    warning=True,
):
    """
    Style: table // string // chart
    """

    # Resampling to yearly (business year)
    yearly_quotes = dataframe.resample("BA").last()

    # Adding first quote (only if start is in the middle of the year)
    yearly_quotes = pd.concat([dataframe.iloc[:1], yearly_quotes])
    first_year = dataframe.index[0].year - 1
    last_year = dataframe.index[-1].year + 1

    # Returns
    yearly_returns = ((yearly_quotes / yearly_quotes.shift(1)) - 1) * 100
    yearly_returns = yearly_returns.set_index([list(range(first_year, last_year))])

    # Inverter o sentido das rows no dataframe ####
    yearly_returns = yearly_returns.loc[first_year + 1:last_year].transpose()
    yearly_returns = round(yearly_returns, 2)

    # As strings and percentages
    yearly_returns.columns = yearly_returns.columns.map(str)
    yearly_returns_numeric = yearly_returns.copy()

    if style == "table" and color is False:
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.format("{:.2%}")
        print_title(title)

    elif style == "table":
        yearly_returns = yearly_returns / 100
        yearly_returns = yearly_returns.style.applymap(color_negative_red).format(
            "{:.2%}"
        )
        print_title(title)

    elif style == "numeric":
        yearly_returns = yearly_returns_numeric.copy()

    elif style == "string":
        for column in yearly_returns:
            yearly_returns[column] = yearly_returns[column].apply(
                lambda x: str(x) + "%"
            )

    elif style == "chart":
        fig, ax = plt.subplots()
        fig.set_size_inches(
            yearly_returns_numeric.shape[1] * 1.25,
            yearly_returns_numeric.shape[0] + 0.5,
        )
        yearly_returns = sns.heatmap(
            yearly_returns_numeric,
            annot=True,
            cmap="RdYlGn",
            linewidths=0.2,
            fmt=".2f",
            cbar=False,
            center=0,
        )
        for t in yearly_returns.texts:
            t.set_text(t.get_text() + "%")
        plt.title(title)

    else:
        print("At least one parameter has a wrong input")

    return yearly_returns


def filter_by_date(dataframe, years=0):

    """
    Legacy function
    """

    last_date = dataframe.tail(1).index
    year_nr = last_date.year.values[0]
    month_nr = last_date.month.values[0]
    day_nr = last_date.day.values[0]

    if month_nr == 2 and day_nr == 29 and years % 4 != 0:
        new_date = str(year_nr - years) + "-" + str(month_nr) + "-" + str(day_nr - 1)
    else:
        new_date = str(year_nr - years) + "-" + str(month_nr) + "-" + str(day_nr)

    dataframe = pd.concat([dataframe.loc[:new_date].tail(1), dataframe.loc[new_date:]])
    # Delete repeated days
    dataframe = dataframe.loc[~dataframe.index.duplicated(keep="first")]

    return dataframe


def print_title(string):
    display(Markdown("**" + string + "**"))


def compute_drawdowns(dataframe):
    """
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    """
    return (dataframe / dataframe.cummax() - 1) * 100


colors_list = [
    "royalblue",
    "darkorange",
    "dimgrey",
    "rgb(86, 53, 171)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "#ffd166",
    "#62959c",
    "#b5179e",
    "rgb(148, 103, 189)",
    "rgb(140, 86, 75)",
    "rgb(227, 119, 194)",
    "rgb(127, 127, 127)",
    "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
]


def merge_time_series(df_1, df_2, how="outer"):
    df = df_1.merge(df_2, how=how, left_index=True, right_index=True)
    return df


def ichart(
    data,
    title="",
    colors=colors_list,
    yTitle="",
    xTitle="",
    style="normal",
    hovermode="x",
    yticksuffix="",
    ytickprefix="",
    ytickformat="",
    source_text="",
    y_position_source="-0.125",
    xticksuffix="",
    xtickprefix="",
    xtickformat="",
    dd_range=[-50, 0],
    y_axis_range_range=None,
):

    """
    style = normal, area, drawdowns_histogram
    colors = color_list or lightcolors
    hovermode = 'x', 'x unified', 'closest'
    y_position_source = -0.125 or bellow
    dd_range = [-50, 0]
    ytickformat =  ".1%"

    """
    fig = go.Figure()

    fig.update_layout(
        paper_bgcolor="#F5F6F9",
        plot_bgcolor="#F5F6F9",
        hovermode=hovermode,
        title=title,
        title_x=0.5,
        yaxis=dict(
            ticksuffix=yticksuffix,
            tickprefix=ytickprefix,
            tickfont=dict(color="#4D5663"),
            gridcolor="#E1E5ED",
            range=y_axis_range_range,
            titlefont=dict(color="#4D5663"),
            zerolinecolor="#E1E5ED",
            title=yTitle,
            showgrid=True,
            tickformat=ytickformat,
        ),
        xaxis=dict(
            title=xTitle,
            tickfont=dict(color="#4D5663"),
            gridcolor="#E1E5ED",
            titlefont=dict(color="#4D5663"),
            zerolinecolor="#E1E5ED",
            showgrid=True,
            tickformat=xtickformat,
            ticksuffix=xticksuffix,
            tickprefix=xtickprefix,
        ),
        images=[
            dict(
                name="watermark_1",
                source="https://raw.githubusercontent.com/LuisSousaSilva/Articles_and_studies/master/FP-cor-positivo.png",
                xref="paper",
                yref="paper",
                x=-0.07500,
                y=1.250,
                sizey=0.15,
                sizex=0.39,
                opacity=1,
                layer="below",
            )
        ],
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.5,
                y=y_position_source,
                xanchor="center",
                yanchor="top",
                text=source_text,
                showarrow=False,
                font=dict(family="Arial", size=12, color="rgb(150,150,150)"),
            )
        ],
    ),  # end

    if style == "normal":
        z = -1

        for i in data:
            z = z + 1
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[i],
                    mode="lines",
                    name=i,
                    line=dict(width=1.3, color=colors[z]),
                )
            )

    if style == "area":
        z = -1

        for i in data:
            z = z + 1
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[i],
                    hoverinfo="x+y",
                    mode="lines",
                    name=i,
                    line=dict(width=0.7, color=colors[z]),
                    stackgroup="one",  # define stack group
                )
            )

    if style == "drawdowns_histogram":
        fig.add_trace(
            go.Histogram(
                x=data.iloc[:, 0],
                histnorm="probability",
                marker=dict(
                    colorscale="RdBu",
                    reversescale=False,
                    cmin=-24,
                    cmax=0,
                    color=np.arange(start=dd_range[0], stop=dd_range[1]),
                    line=dict(color="white", width=0.2),
                ),
                opacity=0.75,
                cumulative=dict(enabled=True),
            )
        )

    return fig


def compute_growth_index(dataframe, initial_value=100, initial_cost=0, ending_cost=0):
    initial_cost = initial_cost / 100
    ending_cost = ending_cost / 100

    GR = ((1 + dataframe.pct_change()).cumprod()) * (initial_value * (1 - initial_cost))
    GR.iloc[0] = initial_value * (1 - initial_cost)
    GR.iloc[-1] = GR.iloc[-1] * (1 * (1 - ending_cost))
    return GR


def download_quotes_yahoo(
    tickers, growth_index=False, start_date="1970-01-01", end_date="2030-06-12"
):
    # Create empty DataFrame
    ETFs = pd.DataFrame()

    for ticker in tickers:
        # Download Quotes
        etf = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Get only Adj Close column as Dataframe
        etf = etf[["Adj Close"]]
        # Change Column name
        etf.columns = [ticker]
        # Merge time series de etf com ETFs
        ETFs = merge_time_series(ETFs, etf)

    # Start when all ETFs are on the market
    ETFs = ETFs.dropna()

    if growth_index:
        ETFs = compute_growth_index(ETFs)

    return ETFs


# Investpy functions


def search_investing_etf(isins=False, tickers=False, visual=""):
    etfs = investpy.get_etfs()

    if isins:
        for isin in isins:
            if visual == "jupyter":
                display(
                    etfs.loc[etfs["isin"] == isin.upper()][
                        [
                            "symbol",
                            "isin",
                            "stock_exchange",
                            "currency",
                            "name",
                            "country",
                        ]
                    ]
                )
            else:
                print(
                    etfs.loc[etfs["isin"] == isin.upper()][
                        [
                            "symbol",
                            "isin",
                            "stock_exchange",
                            "currency",
                            "name",
                            "country",
                        ]
                    ]
                )

    elif tickers:
        for ticker in tickers:
            if visual == "jupyter":
                display(
                    etfs.loc[etfs["symbol"] == ticker.upper()].sort_values(
                        by="def_stock_exchange", ascending=False
                    )[
                        [
                            "symbol",
                            "isin",
                            "stock_exchange",
                            "currency",
                            "name",
                            "country",
                        ]
                    ]
                )
            else:
                print(
                    etfs.loc[etfs["symbol"] == ticker.upper()].sort_values(
                        by="def_stock_exchange", ascending=False
                    )[
                        [
                            "symbol",
                            "isin",
                            "stock_exchange",
                            "currency",
                            "name",
                            "country",
                        ]
                    ]
                )

    else:
        print("Something went wrong with the function inputs")


def download_quotes_investing_etf(
    names,
    countries,
    colnames="",
    begin="1990-01-01",
    end="2025-01-01",
    merge="inner",
    growth_index=False,
):
    begin = pd.to_datetime(begin).strftime("%d/%m/%Y")
    end = pd.to_datetime(end).strftime("%d/%m/%Y")
    iteration = 0

    for i in range(len(names)):
        iteration += 1
        etf = investpy.get_etf_historical_data(
            etf=names[i], from_date=begin, to_date=end, country=countries[i]
        )[["Close"]]
        if iteration == 1:
            etfs = etf.copy()

        else:
            etfs = merge_time_series(etfs, etf, how=merge)

    if colnames:
        etfs.columns = colnames
    else:
        etfs.columns = names

    if growth_index:
        etfs = compute_growth_index(etfs)

    return etfs


# BdP Functions


def get_data_series_from_bdp(series, names):
    main_df = pd.DataFrame()

    for serie in series:
        BPSTAT_API_URL = "https://bpstat.bportugal.pt/data/v1"
        # get series_id from BdP link
        # (https://bpstat.bportugal.pt/serie/12532123)
        series_id = serie
        url = f"{BPSTAT_API_URL}/series/?lang=EN&series_ids={series_id}"
        series_info = requests.get(url).json()[0]

        domain_id = series_info["domain_ids"][0]
        dataset_id = series_info["dataset_id"]

        dataset_url = f"{BPSTAT_API_URL}/domains/{domain_id}/datasets/{dataset_id}/?lang=EN&series_ids={series_id}"
        dataset = pyjstat.Dataset.read(dataset_url)
        df = dataset.write("dataframe")
        df = df[["Date", "value"]]
        # Fazer o datetime index
        df["Date"] = pd.to_datetime(df["Date"])
        df.index = df["Date"]
        df.pop("Date")
        main_df = merge_time_series(main_df, df)

    main_df.columns = names

    return main_df


def compute_return(dataframe, years=""):
    """
    Function to compute drawdowns of a timeseries
    given a dataframe of prices
    """
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return (dataframe.iloc[-1] / dataframe.iloc[0] - 1) * 100

    else:
        return (dataframe.iloc[-1] / dataframe.iloc[0] - 1) * 100


def compute_max_DD(dataframe):
    return compute_drawdowns(dataframe).min()


def compute_cagr(dataframe, years=""):
    """
    Function to calculate CAGR given a dataframe of prices
    """
    if isinstance(years, int):
        years = years
        dataframe = filter_by_date(dataframe, years=years)
        return (
            (dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)
        )

    else:
        years = (
            len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq="D")) / 365
        )

    return (dataframe.iloc[-1].div(dataframe.iloc[0])).pow(1 / years).sub(1).mul(100)


def compute_mar(dataframe):
    """
    Function to calculate mar: Return Over Maximum Drawdown
    given a dataframe of prices
    """
    return compute_cagr(dataframe).div(compute_drawdowns(dataframe).min().abs())


def compute_StdDev(dataframe, freq="days"):
    """
    Function to calculate annualized standart deviation
    given a dataframe of prices. It takes into account the
    frequency of the data.
    """
    if freq == "days":
        return dataframe.pct_change().std().mul((np.sqrt(252))).mul(100)
    if freq == "months":
        return dataframe.pct_change().std().mul((np.sqrt(12))).mul(100)
    if freq == "quarters":
        return dataframe.pct_change().std().mul((np.sqrt(4))).mul(100)
    if freq == "years":
        return dataframe.pct_change().std().mul((np.sqrt(1))).mul(100)


def compute_sharpe(dataframe, years="", freq="days"):
    """
    Function to calculate the sharpe ratio given a dataframe of prices.
    """
    return compute_cagr(dataframe, years).div(compute_StdDev(dataframe, freq))


def compute_performance_table(dataframe, years="si", freq="days"):
    """
    Function to calculate a performance table given a dataframe of prices.
    Takes into account the frequency of the data.
    """

    if years == "si":
        years = (
            len(pd.date_range(dataframe.index[0], dataframe.index[-1], freq="D"))
            / 365.25
        )

        df = pd.DataFrame(
            [
                compute_cagr(dataframe, years),
                compute_return(dataframe),
                compute_StdDev(dataframe, freq),
                compute_sharpe(dataframe, years, freq),
                compute_max_DD(dataframe),
                compute_mar(dataframe),
            ]
        )
        df.index = ["CAGR", "Return", "StdDev", "Sharpe", "Max DD", "MAR"]

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df["Return"] = (df["Return"] / 100).apply("{:.2%}".format)
        df["CAGR"] = (df["CAGR"] / 100).apply("{:.2%}".format)
        df["StdDev"] = (df["StdDev"] / 100).apply("{:.2%}".format)
        df["Max DD"] = (df["Max DD"] / 100).apply("{:.2%}".format)

        start = str(dataframe.index[0])[0:10]
        end = str(dataframe.index[-1])[0:10]
        print_title(
            "Performance from "
            + start
            + " to "
            + end
            + " (≈ "
            + str(round(years, 1))
            + " years)"
        )

        # Return object
        return df

    if years == "ytd":

        df = filter_by_date(dataframe, "ytd")

        start = str(df.index[0])[0:10]
        end = str(df.index[-1])[0:10]

        df = pd.DataFrame(
            [
                compute_ytd_cagr(dataframe),
                compute_ytd_return(dataframe),
                compute_ytd_StdDev(dataframe),
                compute_ytd_sharpe(dataframe),
                compute_ytd_max_DD(dataframe),
                compute_ytd_mar(dataframe),
            ]
        )
        df.index = ["CAGR", "Return", "StdDev", "Sharpe", "Max DD", "MAR"]

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df["Return"] = (df["Return"] / 100).apply("{:.2%}".format)
        df["CAGR"] = "N/A"
        df["StdDev"] = (df["StdDev"] / 100).apply("{:.2%}".format)
        df["Max DD"] = (df["Max DD"] / 100).apply("{:.2%}".format)

        print_title("Performance from " + start + " to " + end + " (YTD)")

        # Return object
        return df

    else:
        dataframe = filter_by_date(dataframe, years)
        df = pd.DataFrame(
            [
                compute_cagr(dataframe, years=years),
                compute_return(dataframe),
                compute_StdDev(dataframe),
                compute_sharpe(dataframe),
                compute_max_DD(dataframe),
                compute_mar(dataframe),
            ]
        )
        df.index = ["CAGR", "Return", "StdDev", "Sharpe", "Max DD", "MAR"]

        df = round(df.transpose(), 2)

        # Colocar percentagens
        df["Return"] = (df["Return"] / 100).apply("{:.2%}".format)
        df["CAGR"] = (df["CAGR"] / 100).apply("{:.2%}".format)
        df["StdDev"] = (df["StdDev"] / 100).apply("{:.2%}".format)
        df["Max DD"] = (df["Max DD"] / 100).apply("{:.2%}".format)

        start = str(dataframe.index[0])[0:10]
        end = str(dataframe.index[-1])[0:10]

        if years == 1:
            print_title(
                "Performance from "
                + start
                + " to "
                + end
                + " ("
                + str(years)
                + " year)"
            )
        else:
            print_title(
                "Performance from "
                + start
                + " to "
                + end
                + " ("
                + str(years)
                + " years)"
            )

        return df

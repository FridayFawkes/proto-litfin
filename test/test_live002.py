#!/usr/bin/env python3

import proto_litfin as lf

if __name__ == "__main__":
    # Download de dados
    etfs_yahoo_finance = lf.download_quotes_yahoo(["IWDA.AS", "VWCE.DE"], growth_index=True, start_date="2000-01-01")

    # Gráfico comparativo
    lf.ichart(
        etfs_yahoo_finance, yticksuffix="€", title="Evolução de cada 100 €uros investidos"
    ).show()

    # etf_names = ["Vanguard FTSE All-World UCITS USD Acc", "iShares Core MSCI World UCITS"]
    # etf_countries = ["germany", "netherlands"]
    # etf_tickers = ["VWCE", "IWDA"]
    # TODO: Fix download_quotes_investing_etf()
    # ETFs_investing = lf.download_quotes_investing_etf(
    #     names=etf_names, countries=etf_countries, growth_index=True
    # )
    #
    # ETFs_investing.columns = etf_tickers
    #
    # lf.ichart(ETFs_investing).show()

    df = lf.get_data_series_from_bdp(
        series=["12532139", "12532125", "12532123"], names=["EUR/GBP", "EUR/CHF", "EUR/USD"]
    )

    lf.ichart(df).show()

    # https://www.investopedia.com/articles/forex/013015/why-switzerland-scrapped-euro.asp

    lf.ichart(lf.download_quotes_yahoo(["EURCHF=X"])).show()

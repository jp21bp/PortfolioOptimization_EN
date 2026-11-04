
CREATE DATABASE port_opt;

/* Creating Tables */
/* Stock adj close Table */
CREATE TABLE adj_close(
    full_date DATE,
    amxb FLOAT,
    axia3 FLOAT,
    bbas3 FLOAT,
    bimboa FLOAT,
    bsac FLOAT,
    cemexcpo FLOAT,
    cencosud FLOAT,
    cib FLOAT,
    femsaubd FLOAT,
	gcarsoa1 FLOAT,
    ggb FLOAT,
    isa FLOAT,
    pac FLOAT,
    pbr FLOAT,
    rent3 FLOAT,
    scco FLOAT,
    sqm FLOAT,
    vale FLOAT,
    viv FLOAT,
    walmex FLOAT,
    wege3 FLOAT,
    PRIMARY KEY (full_date)
);

/* Stock dimensions Table */
CREATE TABLE stock_dim (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
	asset TEXT,
	market_value NUMERIC,
	weight FLOAT,
	notional_value NUMERIC,
	shares INT,
	price FLOAT,
	located TEXT,
	exchange TEXT,
	currency TEXT,
	fx_rate FLOAT,
	market_currency VARCHAR(3),
	accural TEXT
);

/* Copying CSV to Postgresql */
\copy adj_close(
    full_date,
    amxb,
    axia3,
    bbas3,
    bimboa,
    bsac,
    cemexcpo,
    cencosud,
    cib,
    femsaubd,
	gcarsoa1,
    ggb,
    isa,
    pac,
    pbr,
    rent3,
    ssco,
    sqm,
    vale,
    viv,
    walmex,
    wege3  
) FROM 'C:\Datos\Limpios\top40_adj_close.csv' 
DELIMITER ',' CSV HEADER;

\copy stock_dim (
    ticker,
    company_name,
    sector,
	asset,
	market_value,
	weight,
	notional_value,
	shares,
	price,
	located,
	exchange,
	currency,
	fx_rate,
	market_currency,
	accural
) FROM 'C:\Datos\Limpios\top40_alphabetized_EN.csv' 
DELIMITER ',' CSV HEADER;


/* Transforming wide data into long format */
CREATE OR REPLACE VIEW vw_stock_prices AS
SELECT
    a.full_date,
    v.ticker,
    v.adj_close
FROM adj_close a
CROSS JOIN LATERAL (
    VALUES
        ('AMXB.MX', a.amxb),
        ('AXIA3.SA', a.axia3),
        ('BBAS3.SA', a.bbas3),
        ('BIMBOA.MX', a.bimboa),
        ('BSAC', a.bsac),
        ('CEMEXCPO.MX', a.cemexcpo),
        ('CENCOSUD.SN', a.cencosud),
        ('CIB', a.cib),
        ('FEMSAUBD.MX', a.femsaubd),
        ('GCARSOA1.MX', a.gcarsoa1),
        ('GGB', a.ggb),
        ('ISA.CL', a.isa),
        ('PAC', a.pac),
        ('PBR', a.pbr),
        ('RENT3.SA', a.rent3),
        ('SCCO', a.scco),
        ('SQM', a.sqm),
        ('VALE', a.vale),
        ('VIV', a.viv),
        ('WALMEX.MX', a.walmex),
        ('WEGE3.SA', a.wege3)
) AS v(ticker, adj_close);

/* Verifying long format */
SELECT *
FROM vw_stock_prices
WHERE ticker = 'VALE'
ORDER BY full_date;



/* General Overview */
SELECT
    MIN(full_date) AS first_trading_date,
    MAX(full_date) AS last_trading_date,
    COUNT(DISTINCT full_date) AS trading_days,
    COUNT(DISTINCT ticker) AS number_of_stocks,
    COUNT(*) AS total_records
FROM vw_stock_prices;


/* Stock Classification*/
SELECT
    sector,
    located,
    COUNT(*) AS number_of_companies,
    ROUND(AVG(market_value), 2) AS avg_market_value,
    ROUND(SUM(market_value), 2) AS total_market_value
FROM stock_dim
GROUP BY
    sector,
    located
ORDER BY
    sector,
    total_market_value DESC;


/* Stock daily returns */
CREATE OR REPLACE VIEW vw_stock_returns AS
WITH previous_prices AS (
    SELECT
        full_date,
        ticker,
        adj_close,
        LAG(adj_close) OVER (
            PARTITION BY ticker
            ORDER BY full_date
        ) AS previous_price
    FROM vw_stock_prices
)
SELECT
    full_date,
    ticker,
    adj_close,
    previous_price,
    CASE
        WHEN previous_price > 0
        THEN (adj_close / previous_price) - 1
        ELSE NULL
    END AS daily_return
FROM previous_prices;

/* Verifying prev calculations */
SELECT *
FROM vw_stock_returns
WHERE ticker='SCCO';




/* Sharpe ratio */
SELECT
    ticker,
    COUNT(daily_return) AS observations,
    ROUND(
        AVG(daily_return)::NUMERIC * 252,
        4
    ) AS annualized_return,
    ROUND(
        (STDDEV_SAMP(daily_return)* SQRT(252))::NUMERIC,
        4
    ) AS annualized_volatility,
    ROUND(
        (
            AVG(daily_return)
            / NULLIF(STDDEV_SAMP(daily_return), 0)
        )::NUMERIC ,
        4
    ) AS sharpe_ratio
FROM vw_stock_returns
WHERE daily_return IS NOT NULL
GROUP BY ticker
HAVING COUNT(daily_return) > 100
ORDER BY sharpe_ratio DESC;



/* Market Events */
SELECT
    ticker,
    CASE
        WHEN full_date < DATE '2008-09-01'
            THEN 'Pre-Financial Crisis'
        WHEN full_date < DATE '2010-01-01'
            THEN 'Financial Crisis'
        WHEN full_date < DATE '2020-03-01'
            THEN 'Pre-COVID'
        WHEN full_date < DATE '2022-01-01'
            THEN 'COVID Period'
        ELSE 'Post-COVID'
    END AS market_period,
    COUNT(*) AS observations,
    ROUND(
        AVG(daily_return)::NUMERIC * 100,
        4
    ) AS avg_daily_return_pct,
    ROUND(
        (STDDEV_SAMP(daily_return) * SQRT(252))::NUMERIC,
        4
    ) AS annualized_volatility,
	ROUND(
        (
            AVG(daily_return)
            / NULLIF(STDDEV_SAMP(daily_return), 0)
        )::NUMERIC ,
        4
    ) AS sharpe_ratio
FROM vw_stock_returns
WHERE daily_return IS NOT NULL
GROUP BY
    ticker,
    market_period
ORDER BY
    ticker,
    MIN(full_date);
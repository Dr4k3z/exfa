BEGIN;

CREATE SCHEMA IF NOT EXISTS lseg;

CREATE TABLE IF NOT EXISTS lseg.equity_classification (
    asset text PRIMARY KEY,
    country_of_headquarters text,
    trbc_economic_sector_name text,
    trbc_business_sector_name text,
    trbc_industry_group_name text,
    trbc_industry_name text,
    trbc_activity_name text
);

CREATE TABLE IF NOT EXISTS lseg.ref_prices (
    date date,
    asset text,
    close double precision,
    high double precision,
    low double precision,
    bid double precision,
    ask double precision,
    vwap double precision,
    volume double precision,
    PRIMARY KEY (date, asset)
);

CREATE TABLE IF NOT EXISTS lseg.value_fundamentals (
    date date,
    asset text,
    shares_outstanding bigint,
    market_cap double precision,
    revenue double precision,
    book_value double precision,
    operating_cash_flow double precision,
    PRIMARY KEY (date, asset)
);

COMMIT;

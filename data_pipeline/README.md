# Module 1 — Data Pipeline

This module implements the Zepto Data & AI Platform capstone requirements for the
25-mark Data Pipeline module.

## What this module does

The pipeline performs the complete:

**scrape → clean → convert → store → query → read into pandas → reproduce JOIN**

workflow using `requests`, `BeautifulSoup`, `pandas`, and SQLite.

### Data source

Public scraping-practice website:

`https://books.toscrape.com/`

The script dynamically discovers at least 3 categories from the website and
scrapes every pagination page in those categories.

## Required fixed currency conversion

This project uses the assignment-defined fixed baseline:

**1 GBP = 105.50 INR**

Therefore:

`price_inr = price_gbp * 105.50`

No exchange-rate API or network lookup is used for currency conversion.

## Files

```text
data_pipeline/
├── run_pipeline.py
├── queries.sql
├── requirements.txt
├── README.md
├── data_pipeline.db              # generated after running the pipeline
└── outputs/
    ├── cleaned_books.csv         # generated
    ├── sql_query_results.txt     # generated
    ├── join_comparison.txt       # generated
    └── validation_report.txt     # generated
```

## Installation

From inside this folder:

```bash
python -m pip install -r requirements.txt
```

## Run end-to-end

```bash
python run_pipeline.py
```

The script will:

1. Discover 3 categories.
2. Scrape all paginated books in those categories.
3. Require at least 60 clean book rows.
4. Clean `price` into `price_gbp` as float.
5. Convert `One`–`Five` into integer `rating` 1–5.
6. Convert availability text into boolean `in_stock`.
7. Compute `price_inr` using 1 GBP = 105.50 INR.
8. Handle numeric parsing failures with median imputation.
9. Drop rows whose availability cannot safely be interpreted.
10. Create the normalized SQLite schema.
11. Insert categories and books.
12. Execute all SQL queries in `queries.sql`.
13. Save SQL query output.
14. Load JOIN data with `pd.read_sql_query()`.
15. Reproduce the JOIN using `pandas.merge()`.
16. Compare both results.
17. Generate an acceptance-criteria validation report.

## Cleaning decisions

### Price

The currency symbol and other non-numeric characters are removed with a
regular expression and the result is converted to `float`.

If a price cannot be parsed, the median valid `price_gbp` is used.

### Rating

The source text values `One`, `Two`, `Three`, `Four`, and `Five` are mapped to
integers 1–5.

If a rating cannot be parsed, the median rating is rounded to the nearest
integer and used.

### Availability

Text containing `In stock` is converted to `True`. Text containing
`Out of stock` is converted to `False`.

An unrecognized availability value is dropped rather than guessed, because a
boolean inventory state cannot be safely inferred from unknown text.

### Currency

`price_inr` is calculated only from the required fixed project rate:

```text
price_inr = price_gbp * 105.50
```

## SQLite design

The database is normalized into two related tables:

```sql
categories(
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)

books(
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
)
```

The relationship is:

`categories.category_id` → `books.category_id`

SQLite stores the Python boolean values as `0/1`, while the cleaned pandas
DataFrame keeps `in_stock` as a boolean column.

## SQL requirements covered

`queries.sql` demonstrates:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- IN
- JOIN
- GROUP BY

The pipeline executes the queries and saves their printed/tabular output to:

`outputs/sql_query_results.txt`

## SQL and pandas JOIN equivalence

The pipeline executes a SQL JOIN and separately loads the source tables into
pandas and performs:

```python
books_df.merge(
    categories_df,
    on="category_id",
    how="inner"
)
```

The normalized outputs are compared and the result is saved to:

`outputs/join_comparison.txt`

## Acceptance validation

After the pipeline completes, inspect:

`outputs/validation_report.txt`

The report checks:

- at least 60 books
- at least 3 categories
- `price_gbp` is float
- `price_inr` is float
- `rating` is integer 1–5
- `in_stock` is boolean
- both SQLite tables exist
- the books table has a foreign key to categories
- the database contains the required number of books/categories
- SQL JOIN and pandas `merge()` results match

## Reproducibility

The SQLite database is generated from scratch by `run_pipeline.py`. This means
the repository does not depend on manually created database records.

The script is the authoritative recreation path for the database.

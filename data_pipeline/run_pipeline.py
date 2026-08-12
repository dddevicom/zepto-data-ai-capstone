"""
Zepto Capstone - Module 1: Data Pipeline

End-to-end pipeline:
1. Scrape books.toscrape.com with requests + BeautifulSoup.
2. Clean/convert fields.
3. Convert GBP to INR using the required fixed rate: 1 GBP = 105.50 INR.
4. Create a normalized SQLite database with categories -> books PK/FK.
5. Execute the required SQL queries and save their outputs.
6. Read SQL results with pandas.read_sql().
7. Reproduce the JOIN with pandas.merge() and verify both results match.
8. Run acceptance-criteria validation.

Run:
    python run_pipeline.py
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
MIN_BOOKS = 60
MIN_CATEGORIES = 3

MODULE_DIR = Path(__file__).resolve().parent
DB_PATH = MODULE_DIR / "data_pipeline.db"
OUTPUT_DIR = MODULE_DIR / "outputs"
CLEANED_CSV = OUTPUT_DIR / "cleaned_books.csv"
SQL_RESULTS = OUTPUT_DIR / "sql_query_results.txt"
JOIN_COMPARISON = OUTPUT_DIR / "join_comparison.txt"
VALIDATION_REPORT = OUTPUT_DIR / "validation_report.txt"
SQL_FILE = MODULE_DIR / "queries.sql"


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def build_session() -> requests.Session:
    """Create a requests session with retries and a browser-like user agent."""
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            )
        }
    )
    return session


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch a page and return its parsed HTML."""
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def absolute_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE_URL.rstrip("/") + "/" + href.lstrip("/")


def get_category_urls(session: requests.Session, number_of_categories: int = 3) -> list[tuple[str, str]]:
    """Discover category URLs from the website rather than hard-coding them."""
    soup = get_soup(session, BASE_URL)
    links = soup.select("ul.nav-list ul li a")

    categories: list[tuple[str, str]] = []
    seen = set()

    for link in links:
        name = link.get_text(" ", strip=True)
        href = link.get("href")
        if not href:
            continue

        url = absolute_url(href)
        if name and url not in seen:
            categories.append((name, url))
            seen.add(url)

        if len(categories) >= number_of_categories:
            break

    if len(categories) < number_of_categories:
        raise RuntimeError(
            f"Could discover only {len(categories)} categories; "
            f"at least {number_of_categories} are required."
        )

    return categories


def parse_book_cards(soup: BeautifulSoup, category_name: str) -> list[dict]:
    """Extract required raw fields from a category listing page."""
    rows = []

    for card in soup.select("article.product_pod"):
        title_link = card.select_one("h3 a")
        price_tag = card.select_one("p.price_color")
        availability_tag = card.select_one("p.instock.availability")
        rating_tag = card.select_one("p.star-rating")

        title = (
            title_link.get("title", "").strip()
            if title_link
            else ""
        )
        if not title and title_link:
            title = title_link.get_text(" ", strip=True)

        rating_class = ""
        if rating_tag:
            classes = rating_tag.get("class", [])
            rating_class = next(
                (c for c in classes if c in RATING_MAP),
                "",
            )

        rows.append(
            {
                "title": title,
                "price": price_tag.get_text(" ", strip=True) if price_tag else "",
                "star_rating": rating_class,
                "availability": (
                    availability_tag.get_text(" ", strip=True)
                    if availability_tag
                    else ""
                ),
                "category": category_name,
            }
        )

    return rows


def next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    next_link = soup.select_one("li.next a")
    if not next_link or not next_link.get("href"):
        return None

    current = current_url.rsplit("/", 1)[0] + "/"
    href = next_link["href"]
    return absolute_url(current + href)


def scrape_category(
    session: requests.Session,
    category_name: str,
    category_url: str,
) -> list[dict]:
    """Scrape every pagination page for one category."""
    rows: list[dict] = []
    url: str | None = category_url
    visited = set()

    while url and url not in visited:
        visited.add(url)
        soup = get_soup(session, url)
        rows.extend(parse_book_cards(soup, category_name))
        url = next_page_url(soup, url)

    return rows


def scrape_books(number_of_categories: int = 3) -> pd.DataFrame:
    """Scrape all books across at least the requested number of categories."""
    session = build_session()
    categories = get_category_urls(session, number_of_categories)

    print("Categories selected:")
    for name, url in categories:
        print(f"  - {name}: {url}")

    all_rows: list[dict] = []

    for category_name, category_url in categories:
        print(f"Scraping category: {category_name}")
        category_rows = scrape_category(session, category_name, category_url)
        print(f"  Books scraped: {len(category_rows)}")
        all_rows.extend(category_rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError("Scraping returned zero book rows.")

    return df


def parse_price(value: object) -> float:
    """Parse a GBP price string into a float; return NA when invalid."""
    if pd.isna(value):
        return float("nan")

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else float("nan")


def parse_rating(value: object) -> float:
    """Convert One..Five to 1..5; return NA for unexpected values."""
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if text in RATING_MAP:
        return float(RATING_MAP[text])

    # Also tolerate the complete CSS class string, e.g. "star-rating Three".
    for word, score in RATING_MAP.items():
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            return float(score)

    return float("nan")


def parse_stock(value: object) -> object:
    """Convert availability text to boolean; return NA for unrecognized text."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False

    return pd.NA


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and type the required fields.

    Numeric parse failures use median imputation as permitted by the assignment.
    Unrecognized availability values are dropped because a boolean stock state
    cannot be inferred safely from unknown text.
    """
    df = raw_df.copy()

    required_columns = [
        "title",
        "price",
        "star_rating",
        "availability",
        "category",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required scraped columns: {missing}")

    for column in ["title", "category"]:
        df[column] = df[column].astype("string").str.strip()

    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_stock)

    # Remove rows with missing essential text fields.
    before_text_drop = len(df)
    df = df.dropna(subset=["title", "category"])
    df = df[(df["title"] != "") & (df["category"] != "")]
    text_rows_dropped = before_text_drop - len(df)

    # Numeric median imputation.
    price_missing = int(df["price_gbp"].isna().sum())
    rating_missing = int(df["rating"].isna().sum())

    if df["price_gbp"].notna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    if df["rating"].notna().any():
        rating_median = int(round(df["rating"].median()))
        df["rating"] = df["rating"].fillna(rating_median)

    # Rating must remain an integer from 1 to 5.
    df["rating"] = df["rating"].clip(1, 5).round().astype("int64")

    # Unknown availability is unsafe to infer, so drop those rows.
    stock_missing = int(df["in_stock"].isna().sum())
    df = df.dropna(subset=["in_stock"])
    df["in_stock"] = df["in_stock"].astype(bool)

    # Required fixed project-defined conversion.
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).astype(float)

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].reset_index(drop=True)

    decisions = {
        "text_rows_dropped": text_rows_dropped,
        "price_median_imputed": price_missing,
        "rating_median_imputed": rating_missing,
        "availability_rows_dropped": stock_missing,
    }

    return df, decisions


def create_database(df: pd.DataFrame) -> None:
    """Create the normalized SQLite schema and load the cleaned data."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        );
        """
    )

    category_df = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )
    category_df["category_id"] = range(1, len(category_df) + 1)

    for row in category_df.itertuples(index=False):
        conn.execute(
            "INSERT INTO categories(category_id, category_name) VALUES (?, ?)",
            (int(row.category_id), str(row.category)),
        )

    category_map = dict(
        zip(category_df["category"], category_df["category_id"])
    )

    insert_rows = [
        (
            row.title,
            float(row.price_gbp),
            float(row.price_inr),
            int(row.rating),
            int(bool(row.in_stock)),
            int(category_map[row.category]),
        )
        for row in df.itertuples(index=False)
    ]

    conn.executemany(
        """
        INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        insert_rows,
    )

    conn.commit()
    conn.close()


def load_queries() -> list[tuple[str, str]]:
    """Read named SQL queries from queries.sql."""
    text = SQL_FILE.read_text(encoding="utf-8")
    sections = re.split(r"\n--\s*QUERY:\s*", text)

    queries = []
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            continue
        name = lines[0].strip()
        sql = "\n".join(lines[1:]).strip()
        sql = sql.split(";")[0].strip()
        queries.append((name, sql))

    return queries


def execute_sql_queries() -> dict[str, pd.DataFrame]:
    """Execute all required SQL queries and save their outputs."""
    conn = sqlite3.connect(DB_PATH)
    results: dict[str, pd.DataFrame] = {}

    with SQL_RESULTS.open("w", encoding="utf-8") as output:
        output.write("ZEPT0 MODULE 1 - SQL QUERY OUTPUTS\n")
        output.write("=" * 80 + "\n\n")

        for index, (name, sql) in enumerate(load_queries(), start=1):
            result = pd.read_sql_query(sql, conn)
            results[name] = result

            output.write(f"QUERY {index}: {name}\n")
            output.write("-" * 80 + "\n")
            output.write(sql + "\n\n")
            output.write(result.to_string(index=False))
            output.write("\n\n")

    conn.close()
    return results


def reproduce_join_with_pandas() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Compare SQL JOIN output with pandas.merge() output."""
    conn = sqlite3.connect(DB_PATH)

    sql_join = """
    SELECT
        b.title,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock,
        c.category_name
    FROM books AS b
    JOIN categories AS c
        ON b.category_id = c.category_id
    ORDER BY b.book_id
    """

    sql_df = pd.read_sql_query(sql_join, conn)

    books_df = pd.read_sql_query(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        ORDER BY book_id
        """,
        conn,
    )

    categories_df = pd.read_sql_query(
        """
        SELECT category_id, category_name
        FROM categories
        ORDER BY category_id
        """,
        conn,
    )

    merged_df = books_df.merge(
        categories_df,
        on="category_id",
        how="inner",
    )[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    conn.close()

    # Normalize dtypes for an exact value comparison.
    left = sql_df.copy()
    right = merged_df.copy()

    left["in_stock"] = left["in_stock"].astype(int)
    right["in_stock"] = right["in_stock"].astype(int)

    left = left.sort_values(["title", "price_gbp"]).reset_index(drop=True)
    right = right.sort_values(["title", "price_gbp"]).reset_index(drop=True)

    match = left.equals(right)

    with JOIN_COMPARISON.open("w", encoding="utf-8") as output:
        output.write("SQL JOIN RESULT\n")
        output.write("=" * 80 + "\n")
        output.write(left.to_string(index=False))
        output.write("\n\nPANDAS MERGE RESULT\n")
        output.write("=" * 80 + "\n")
        output.write(right.to_string(index=False))
        output.write("\n\n")
        output.write(f"SQL JOIN == pandas.merge(): {match}\n")

    return left, right, match


def validate_acceptance_criteria(
    df: pd.DataFrame,
    join_match: bool,
) -> bool:
    """Validate the explicit acceptance criteria from the assignment."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    checks: list[tuple[str, bool]] = []

    checks.append(("At least 60 book rows", len(df) >= MIN_BOOKS))
    checks.append(
        ("At least 3 categories", df["category"].nunique() >= MIN_CATEGORIES)
    )
    checks.append(
        ("price_gbp is float", pd.api.types.is_float_dtype(df["price_gbp"]))
    )
    checks.append(
        ("price_inr is float", pd.api.types.is_float_dtype(df["price_inr"]))
    )
    checks.append(
        ("rating is integer 1-5",
         pd.api.types.is_integer_dtype(df["rating"])
         and df["rating"].between(1, 5).all())
    )
    checks.append(
        ("in_stock is boolean", pd.api.types.is_bool_dtype(df["in_stock"]))
    )

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    checks.append(("categories table exists", "categories" in tables))
    checks.append(("books table exists", "books" in tables))

    fk_rows = conn.execute("PRAGMA foreign_key_list(books)").fetchall()
    checks.append(
        (
            "books has FK to categories",
            any(row[2] == "categories" for row in fk_rows),
        )
    )

    category_count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    book_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    checks.append(("Database has >= 60 books", book_count >= MIN_BOOKS))
    checks.append(("Database has >= 3 categories", category_count >= MIN_CATEGORIES))
    checks.append(("SQL JOIN and pandas.merge match", join_match))

    conn.close()

    passed = all(value for _, value in checks)

    with VALIDATION_REPORT.open("w", encoding="utf-8") as output:
        output.write("MODULE 1 ACCEPTANCE VALIDATION\n")
        output.write("=" * 80 + "\n")
        output.write(f"Required GBP -> INR rate: 1 GBP = {GBP_TO_INR:.2f} INR\n\n")

        for name, value in checks:
            output.write(f"[{'PASS' if value else 'FAIL'}] {name}\n")

        output.write("\n")
        output.write(
            f"Overall result: {'PASS' if passed else 'FAIL'}\n"
        )

    return passed


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== ZEPTO MODULE 1: DATA PIPELINE ===\n")

    print("1/6 Scraping...")
    raw_df = scrape_books(number_of_categories=MIN_CATEGORIES)
    print(f"Raw rows scraped: {len(raw_df)}")

    print("\n2/6 Cleaning and conversion...")
    clean_df, decisions = clean_data(raw_df)
    print(f"Clean rows: {len(clean_df)}")
    print(f"Cleaning decisions: {decisions}")

    if len(clean_df) < MIN_BOOKS:
        raise RuntimeError(
            f"Only {len(clean_df)} clean rows remain; at least {MIN_BOOKS} are required."
        )

    if clean_df["category"].nunique() < MIN_CATEGORIES:
        raise RuntimeError(
            f"Only {clean_df['category'].nunique()} categories remain; "
            f"at least {MIN_CATEGORIES} are required."
        )

    clean_df.to_csv(CLEANED_CSV, index=False)

    print("\n3/6 Creating normalized SQLite database...")
    create_database(clean_df)
    print(f"Database created: {DB_PATH}")

    print("\n4/6 Executing required SQL queries...")
    results = execute_sql_queries()
    print(f"Executed {len(results)} SQL queries.")
    print(f"SQL output saved: {SQL_RESULTS}")

    print("\n5/6 Reproducing JOIN with pandas.merge()...")
    _, _, join_match = reproduce_join_with_pandas()
    print(f"SQL JOIN == pandas.merge(): {join_match}")
    print(f"Comparison saved: {JOIN_COMPARISON}")

    print("\n6/6 Validating acceptance criteria...")
    passed = validate_acceptance_criteria(clean_df, join_match)
    print(f"Validation report: {VALIDATION_REPORT}")

    print("\n=== SUMMARY ===")
    print(f"Books: {len(clean_df)}")
    print(f"Categories: {clean_df['category'].nunique()}")
    print(f"GBP -> INR: 1 GBP = {GBP_TO_INR:.2f} INR")
    print(f"SQL queries executed: {len(results)}")
    print(f"JOIN comparison match: {join_match}")
    print(f"Acceptance criteria: {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

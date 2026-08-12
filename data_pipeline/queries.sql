-- Zepto Capstone Module 1
-- SQL queries for data pipeline validation and analysis
-- Required SQL demonstrations: SELECT, WHERE, ORDER BY, LIMIT,
-- DISTINCT, BETWEEN, and JOIN.

-- QUERY: High-rated books
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC, price_gbp ASC;

-- QUERY: Ten most expensive books
SELECT title, price_gbp, price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10;

-- QUERY: Distinct ratings
SELECT DISTINCT rating
FROM books
ORDER BY rating;

-- QUERY: Books in a GBP price range
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;

-- QUERY: Books from selected categories using IN
SELECT
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
WHERE c.category_name IN (
    SELECT category_name
    FROM categories
    ORDER BY category_name
    LIMIT 3
)
ORDER BY c.category_name, b.title;

-- QUERY: Category join summary
SELECT
    c.category_name,
    COUNT(b.book_id) AS book_count,
    ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
FROM categories AS c
JOIN books AS b
    ON b.category_id = c.category_id
GROUP BY c.category_id, c.category_name
ORDER BY book_count DESC, c.category_name;

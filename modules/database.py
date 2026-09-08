import os
import sqlite3
from datetime import datetime
from typing import Any


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# Project root is one level above modules/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_DIR = os.path.join(_PROJECT_ROOT, "data", "db")
DB_PATH = os.path.join(DB_DIR, "finbot_products.db")


# ---------------------------------------------------------
# Controlled product taxonomy
# ---------------------------------------------------------

PRODUCT_DOMAINS = {
    "Banking",
    "Lending",
    "Investments",
    "Insurance",
    "Cards",
    "Retirement",
}


# ---------------------------------------------------------
# Database connection
# ---------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection and create data/db/ if needed."""
    os.makedirs(DB_DIR, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


# ---------------------------------------------------------
# Database initialization
# ---------------------------------------------------------

def init_db() -> None:
    """Create the products table if it does not already exist."""

    connection = get_connection()

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT NOT NULL,

                product_domain TEXT NOT NULL,
                product_type TEXT NOT NULL,

                target_customer TEXT,
                region TEXT,

                description TEXT,
                features TEXT,
                benefits TEXT,
                eligibility TEXT,
                price TEXT,

                source_document TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:
        connection.close()


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def _validate_product(product: dict[str, Any]) -> None:
    """Validate required fields and controlled product domain."""

    required_fields = [
        "product_id",
        "product_name",
        "product_domain",
        "product_type",
    ]

    missing = [
        field
        for field in required_fields
        if not product.get(field)
    ]

    if missing:
        raise ValueError(
            f"Missing required product fields: {', '.join(missing)}"
        )

    product_domain = product["product_domain"]

    if product_domain not in PRODUCT_DOMAINS:
        raise ValueError(
            f"Invalid product_domain '{product_domain}'. "
            f"Expected one of: {', '.join(sorted(PRODUCT_DOMAINS))}"
        )


# ---------------------------------------------------------
# Insert / update one product
# ---------------------------------------------------------

def upsert_product(product: dict[str, Any]) -> None:
    """
    Insert a product or update it if product_id already exists.

    The LLM never generates SQL.
    All SQL is predefined and parameterized.
    """

    _validate_product(product)

    connection = get_connection()

    now = datetime.now().isoformat(timespec="seconds")

    try:
        existing = connection.execute(
            """
            SELECT created_at
            FROM products
            WHERE product_id = ?
            """,
            (product["product_id"],),
        ).fetchone()

        created_at = existing["created_at"] if existing else now

        connection.execute(
            """
            INSERT INTO products (
                product_id,
                product_name,
                product_domain,
                product_type,
                target_customer,
                region,
                description,
                features,
                benefits,
                eligibility,
                price,
                source_document,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(product_id)
            DO UPDATE SET
                product_name = excluded.product_name,
                product_domain = excluded.product_domain,
                product_type = excluded.product_type,
                target_customer = excluded.target_customer,
                region = excluded.region,
                description = excluded.description,
                features = excluded.features,
                benefits = excluded.benefits,
                eligibility = excluded.eligibility,
                price = excluded.price,
                source_document = excluded.source_document,
                updated_at = excluded.updated_at
            """,
            (
                product["product_id"],
                product["product_name"],
                product["product_domain"],
                product["product_type"],
                product.get("target_customer"),
                product.get("region"),
                product.get("description"),
                product.get("features"),
                product.get("benefits"),
                product.get("eligibility"),
                product.get("price"),
                product.get("source_document"),
                created_at,
                now,
            ),
        )

        connection.commit()

    finally:
        connection.close()


# ---------------------------------------------------------
# Insert / update multiple products
# ---------------------------------------------------------

def upsert_products(products: list[dict[str, Any]]) -> None:
    """Insert or update multiple products in a single transaction."""

    connection = get_connection()
    now = datetime.now().isoformat(timespec="seconds")

    try:
        for product in products:

            _validate_product(product)

            existing = connection.execute(
                """
                SELECT created_at
                FROM products
                WHERE product_id = ?
                """,
                (product["product_id"],),
            ).fetchone()

            created_at = existing["created_at"] if existing else now

            connection.execute(
                """
                INSERT INTO products (
                    product_id,
                    product_name,
                    product_domain,
                    product_type,
                    target_customer,
                    region,
                    description,
                    features,
                    benefits,
                    eligibility,
                    price,
                    source_document,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(product_id)
                DO UPDATE SET
                    product_name = excluded.product_name,
                    product_domain = excluded.product_domain,
                    product_type = excluded.product_type,
                    target_customer = excluded.target_customer,
                    region = excluded.region,
                    description = excluded.description,
                    features = excluded.features,
                    benefits = excluded.benefits,
                    eligibility = excluded.eligibility,
                    price = excluded.price,
                    source_document = excluded.source_document,
                    updated_at = excluded.updated_at
                """,
                (
                    product["product_id"],
                    product["product_name"],
                    product["product_domain"],
                    product["product_type"],
                    product.get("target_customer"),
                    product.get("region"),
                    product.get("description"),
                    product.get("features"),
                    product.get("benefits"),
                    product.get("eligibility"),
                    product.get("price"),
                    product.get("source_document"),
                    created_at,
                    now,
                ),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ---------------------------------------------------------
# Product retrieval
# ---------------------------------------------------------

def get_products(
    product_domain: str | None = None,
    product_type: str | None = None,
    target_customer: str | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return products matching deterministic filters.

    No Text-to-SQL.
    The backend builds the SQL query itself.
    """

    connection = get_connection()

    try:
        query = """
            SELECT *
            FROM products
            WHERE 1 = 1
        """

        parameters: list[Any] = []

        if product_domain:
            query += " AND product_domain = ?"
            parameters.append(product_domain)

        if product_type:
            query += " AND product_type = ?"
            parameters.append(product_type)

        if target_customer:
            query += " AND target_customer = ?"
            parameters.append(target_customer)

        if region:
            query += " AND region = ?"
            parameters.append(region)

        query += " ORDER BY product_name"

        cursor = connection.execute(query, parameters)

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()


# ---------------------------------------------------------
# Get one product
# ---------------------------------------------------------

def get_product(product_id: str) -> dict[str, Any] | None:
    """Return one product by product_id."""

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            SELECT *
            FROM products
            WHERE product_id = ?
            """,
            (product_id,),
        )

        row = cursor.fetchone()

        return dict(row) if row else None

    finally:
        connection.close()


# ---------------------------------------------------------
# List all products
# ---------------------------------------------------------

def list_products() -> list[dict[str, Any]]:
    """Return all products in the catalogue."""

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            SELECT *
            FROM products
            ORDER BY product_id
            """
        )

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()


# ---------------------------------------------------------
# Delete products from a source document
# ---------------------------------------------------------

def delete_products_by_source(source_document: str) -> int:
    """
    Delete all products originating from a source document.

    Useful when replacing/re-ingesting an updated catalogue PDF.
    """

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            DELETE FROM products
            WHERE source_document = ?
            """,
            (source_document,),
        )

        connection.commit()

        return cursor.rowcount

    finally:
        connection.close()


# ---------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------

if __name__ == "__main__":

    init_db()

    print(f"Database path: {DB_PATH}")

    products = list_products()

    print(f"Products currently stored: {len(products)}")

    for product in products:

        print(
            f"{product['product_id']} | "
            f"{product['product_name']} | "
            f"{product['product_domain']} | "
            f"{product['product_type']}"
        )
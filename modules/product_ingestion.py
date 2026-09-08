import os
import re
from pypdf import PdfReader

from modules.database import init_db, upsert_products


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_products(text: str, source_document: str) -> list[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    product_id_positions = [
        i for i, line in enumerate(lines)
        if re.fullmatch(r"FIN-P\d{3}", line)
    ]

    products = []

    fields = {
        "Domain": "product_domain",
        "Product Type": "product_type",
        "Target Customer": "target_customer",
        "Region": "region",
        "Price / Range": "price",
        "Features": "features",
        "Benefits": "benefits",
        "Eligibility": "eligibility",
        "Semantic Description": "description",
    }

    for n, pid_index in enumerate(product_id_positions):

        # Structure in PDF:
        # Product Name
        # Product ID
        # FIN-Pxxx
        product_name = lines[pid_index - 2]

        # End before the next product name.
        if n + 1 < len(product_id_positions):
            next_pid_index = product_id_positions[n + 1]
            end = next_pid_index - 2
        else:
            end = len(lines)

        block = lines[pid_index:end]

        product = {
            "product_id": lines[pid_index],
            "product_name": product_name,
            "source_document": source_document,
        }

        current_field = None
        values = {}

        for line in block:
            if line in fields:
                current_field = fields[line]
                values[current_field] = []
            elif current_field:
                values[current_field].append(line)

        for key, value in values.items():
            product[key] = " ".join(value)

        products.append(product)

    return products

def ingest_product_catalogue(pdf_path: str) -> None:
    init_db()

    text = extract_text(pdf_path)
    source_document = os.path.basename(pdf_path)

    products = parse_products(text, source_document)

    if not products:
        raise ValueError("No products found in PDF.")

    upsert_products(products)

    print(f"Ingested {len(products)} products.")
    for product in products:
        print(
            f"{product['product_id']} | "
            f"{product['product_name']} | "
            f"{product['product_domain']}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python modules/product_ingestion.py "
            "<catalogue.pdf>"
        )
        raise SystemExit(1)

    ingest_product_catalogue(sys.argv[1])
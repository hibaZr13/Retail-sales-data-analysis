"""
clean.py - cleaning script for the Retail Sales Analysis project

Put this file in the project folder (next to the data/ folder) and run:
    python clean.py

Reads (never changes):  data/customers.csv, invoice_items.csv, products.csv, purchases.csv
Writes:                 data/clean/  (cleaned tables, review CSV, cleaning_report.md, comparison.md)

Steps (same order as my cleaning report):
    1. Load the raw tables and keep a copy of each untouched
    2. purchases: turn `date` into a datetime
    3. Delete exact duplicate rows (invoice_items, purchases)
    4. invoice_items: flag repeated invoice + product lines, zero and near-zero prices
    5. purchases: flag repeated invoice + product lines
    6. products: standardize categories + keyword check ("proposed_category")
    7. Merge the tables (invoice header table, one row per invoice)
    8. Compare cleaned data with the original

Open decisions from my report (big quantities, invoice price vs products price,
postage/fee lines, the two ID ranges) are NOT handled here on purpose.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data"
CLEAN_DIR = RAW_DIR / "clean"

NEAR_ZERO_LIMIT = 0.10

CATEGORY_MAP = {
    "home decor": "Home Decor",
    "kitchen & dining": "Kitchen & Dining",
    "kitchen": "Kitchen & Dining",
    "party & festive": "Party & Festive",
    "seasonal": "Party & Festive",
    "toys & games": "Toys & Games",
    "stationery & craft": "Stationery & Craft",
    "apparel & accessories": "Apparel & Accessories",
    "garden & outdoor": "Garden & Outdoor",
    "gardening": "Garden & Outdoor",
    "health, beauty & personal care": "Personal Care & Beauty",
    "personal care": "Personal Care & Beauty",
    "cleaning & hygiene": "Cleaning & Household",
    "household": "Cleaning & Household",
    "diy & tools": "DIY & Tools",
    "produce": "Produce",
    "dairy": "Dairy",
    "beverages": "Beverages",
    "bakery & snacks": "Bakery & Snacks",
    "meat": "Meat",
    "condiments & spices": "Condiments & Spices",
    "pantry": "Pantry",
    "pet care": "Pet Care",
    "miscellaneous": "Miscellaneous",
    "administrative": "Non-merchandise",
}

KEYWORDS = {
    "Home Decor": ["wall", "mirror", "frame", "clock", "candle", "holder", "lamp", "lampshade", "vase",
                   "ornament", "photo", "sign", "heart", "flower", "hanging"],
    "Kitchen & Dining": ["mug", "cup", "glass", "bowl", "plate", "tray", "cutlery", "spoon", "fork", "knife",
                         "jar", "teapot", "kettle", "cake", "baking", "cookware", "utensil", "tea towel"],
    "Party & Festive": ["party", "garland", "bunting", "ribbon", "balloon", "decoration", "honeycomb",
                        "confetti", "banner", "halloween", "christmas", "easter", "birthday", "wedding"],
    "Toys & Games": ["toy", "doll", "dolly", "jigsaw", "puzzle", "playing cards", "game", "dinosaur",
                     "childrens", "child's", "kids", "soft toy", "teddy"],
    "Stationery & Craft": ["notebook", "paper", "card", "greeting", "pencil", "pen", "envelope", "sticker",
                           "craft", "wrapping", "gift wrap", "book", "stamp"],
    "Apparel & Accessories": ["necklace", "earrings", "bracelet", "ring", "purse", "handbag", "scarf", "hair",
                              "charm", "brooch", "keyring", "bag"],
    "Garden & Outdoor": ["garden", "watering", "plant", "seed", "soil", "fertilizer", "parasol", "picnic",
                         "bird feeder", "plant pot", "toadstool", "herb"],
    "Personal Care & Beauty": ["soap", "shampoo", "cosmetics", "makeup", "skin", "beauty", "perfume",
                               "deodorant", "moisturiser", "toothbrush", "toothpaste", "sanitary", "hair spray"],
    "Cleaning & Household": ["cleaner", "detergent", "softener", "decalcifier", "abrasive", "dishwashing",
                             "cling film", "bin bag", "napkin", "tissue", "washing", "disinfectant"],
    "DIY & Tools": ["screwdriver", "spirit level", "tape measure", "drill", "screw", "hammer", "tool", "repair"],
    "Produce": ["fruit", "vegetable", "grapes", "citrus", "berries", "potato", "onion", "tomato", "carrot", "apple"],
    "Dairy": ["milk", "cheese", "butter", "yogurt", "cream", "curd", "whipped", "sour cream"],
    "Beverages": ["wine", "beer", "coffee", "tea", "water", "soda", "juice", "prosecco", "whisky", "brandy", "cocoa"],
    "Bakery & Snacks": ["bread", "cake", "waffle", "biscuit", "cookie", "pastry", "roll", "bun", "snack",
                        "cereal", "dessert"],
    "Meat": ["meat", "sausage", "chicken", "beef", "pork", "ham", "bacon", "hamburger", "liver"],
    "Condiments & Spices": ["sauce", "honey", "mustard", "vinegar", "jam", "syrup", "salt", "spice", "ketchup",
                            "dressing", "seasoning"],
    "Pantry": ["oil", "flour", "rice", "pasta", "lentils", "beans", "sugar"],
    "Pet Care": ["pet", "dog food", "cat food", "animal food", "pet care"],
}

AMBIGUOUS = ["bag", "candle", "tea", "bottle", "heart", "box", "flower", "holder", "light", "paper", "gift"]


# ----------------------------------------------------------------------
# Markdown reporting
# ----------------------------------------------------------------------
report_sections = []


def md_table(data):
    """Convert a DataFrame to a Markdown table without extra dependencies."""
    if data.empty:
        return "_None._"

    data = data.fillna("").astype(str)
    headers = [str(column).replace("|", "\\|") for column in data.columns]
    rows = [[value.replace("|", "\\|").replace("\n", " ") for value in row]
            for row in data.values.tolist()]

    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows]
    ])


def add_report(title, summary, tables=None):
    """Add a step summary and supporting record tables to the report."""
    section = [f"## {title}", "", summary]

    for heading, data in (tables or {}).items():
        section.extend(["", f"### {heading}", "", md_table(data)])

    report_sections.append("\n".join(section))


def log_step(step, table, rows_before, rows_after, sales_before=None, sales_after=None, note=""):
    """Print the existing concise progress message."""
    print(f"[{step}] {table}: {rows_before:,} -> {rows_after:,} rows  {note}")


# ----------------------------------------------------------------------
# 1. Load the raw tables and keep untouched copies
# ----------------------------------------------------------------------
customers_raw = pd.read_csv(RAW_DIR / "customers.csv")
invoice_items_raw = pd.read_csv(RAW_DIR / "invoice_items.csv")
products_raw = pd.read_csv(RAW_DIR / "products.csv")
purchases_raw = pd.read_csv(RAW_DIR / "purchases.csv")

customers = customers_raw.copy()
invoice_items = invoice_items_raw.copy()
products = products_raw.copy()
purchases = purchases_raw.copy()

# Checks on the raw data
for name, table in [("customers", customers), ("invoice_items", invoice_items),
                    ("products", products), ("purchases", purchases)]:
    n_missing = int(table.isna().sum().sum())
    if n_missing > 0:
        raise ValueError(f"{name} has {n_missing} missing values, my notes said there were none")

if not ((invoice_items["quantity"] > 0).all() and (invoice_items["price"] >= 0).all()):
    raise ValueError("invoice_items has a quantity <= 0 or a negative price, check before cleaning")

# line_total should be price x quantity
calculated_total = (invoice_items["price"] * invoice_items["quantity"]).round(2)
n_wrong_totals = int((invoice_items["line_total"].round(2) != calculated_total).sum())
print(f"line_total different from price x quantity: {n_wrong_totals} rows")

sales_raw = invoice_items_raw["line_total"].sum()
print(f"Raw total sales: {sales_raw:,.2f}\n")

log_step("baseline", "invoice_items", len(invoice_items), len(invoice_items), sales_raw, sales_raw)
log_step("baseline", "purchases", len(purchases), len(purchases))

add_report("1. Raw data baseline",
           f"Loaded four raw tables without modifying the source files.\n\n"
           f"- customers: {len(customers):,} rows\n"
           f"- invoice_items: {len(invoice_items):,} rows; total line sales: {sales_raw:,.2f}\n"
           f"- products: {len(products):,} rows\n"
           f"- purchases: {len(purchases):,} rows\n"
           f"- Missing values: none found by the initial check\n"
           f"- Invalid quantity/negative price check: passed\n"
           f"- line_total versus price x quantity mismatches: {n_wrong_totals:,}")


# ----------------------------------------------------------------------
# 2. purchases: date from string to datetime
# ----------------------------------------------------------------------
purchases["date"] = pd.to_datetime(purchases["date"], format="%Y-%m-%d")

add_report("2. Purchase dates",
           f"Converted all {len(purchases):,} purchase dates using the required YYYY-MM-DD format.\n\n"
           f"- Resulting dtype: {purchases['date'].dtype}\n"
           f"- Date range: {purchases['date'].min().date()} to {purchases['date'].max().date()}\n"
           "- Invalid dates: 0 (the script stops if a value does not match the format).")

print(f"date type: {purchases['date'].dtype}, from {purchases['date'].min().date()} to {purchases['date'].max().date()}\n")


# ----------------------------------------------------------------------
# 3. Delete exact duplicate rows
# ----------------------------------------------------------------------
before_rows = len(invoice_items)
before_sales = invoice_items["line_total"].sum()

removed_invoice_duplicates = invoice_items.loc[invoice_items.duplicated(keep="first")].copy()

invoice_items = invoice_items.drop_duplicates().reset_index(drop=True)

log_step("drop exact duplicates", "invoice_items", before_rows, len(invoice_items),
         before_sales, invoice_items["line_total"].sum(),
         note=f"{before_rows - len(invoice_items):,} rows deleted")

before_rows = len(purchases)

removed_purchase_duplicates = purchases.loc[purchases.duplicated(keep="first")].copy()

purchases = purchases.drop_duplicates().reset_index(drop=True)

log_step("drop exact duplicates", "purchases", before_rows, len(purchases),
         note=f"{before_rows - len(purchases):,} rows deleted")

add_report("3. Exact duplicate removal",
           f"Removed only fully identical rows, keeping the first occurrence.\n\n"
           f"- invoice_items: {len(removed_invoice_duplicates):,} rows removed "
           f"({len(invoice_items):,} rows remain)\n"
           f"- purchases: {len(removed_purchase_duplicates):,} rows removed "
           f"({len(purchases):,} rows remain)",
           {"Removed invoice_items records": removed_invoice_duplicates,
            "Removed purchases records": removed_purchase_duplicates})

print("\nThe two tables no longer have the same number of rows, so I do not compare them row by row anymore.\n")


# ----------------------------------------------------------------------
# 4. invoice_items: flags (nothing is deleted here)
# ----------------------------------------------------------------------
key = ["InvoiceID", "product_id"]

n_lines = invoice_items.groupby(key)["quantity"].transform("size")
n_quantities = invoice_items.groupby(key)["quantity"].transform("nunique")
n_prices = invoice_items.groupby(key)["price"].transform("nunique")

invoice_items["dup_diff_qty"] = (n_quantities > 1) & (n_prices == 1)
invoice_items["dup_same_qty_diff_price"] = (n_quantities == 1) & (n_prices > 1)
invoice_items["dup_diff_qty_diff_price"] = (n_quantities > 1) & (n_prices > 1)

any_flag = (invoice_items["dup_diff_qty"] | invoice_items["dup_same_qty_diff_price"]
            | invoice_items["dup_diff_qty_diff_price"])

n_not_flagged = int(((n_lines > 1) & ~any_flag).sum())

if n_not_flagged > 0:
    raise ValueError(f"{n_not_flagged} repeated lines are not in any of the 3 cases, check the duplicates step")

invoice_items["is_free"] = invoice_items["price"] == 0
invoice_items["near_zero_price"] = (invoice_items["price"] > 0) & (invoice_items["price"] < NEAR_ZERO_LIMIT)

print("invoice_items flags (rows / invoice + product pairs):")

for flag in ["dup_diff_qty", "dup_same_qty_diff_price", "dup_diff_qty_diff_price"]:
    n_rows = int(invoice_items[flag].sum())
    n_pairs = len(invoice_items.loc[invoice_items[flag], key].drop_duplicates())
    print(f"  {flag}: {n_rows:,} rows, {n_pairs:,} pairs")

print(f"  is_free: {int(invoice_items['is_free'].sum()):,} rows")
print(f"  near_zero_price: {int(invoice_items['near_zero_price'].sum()):,} rows (0 < price < {NEAR_ZERO_LIMIT})\n")

duplicate_details = invoice_items.loc[any_flag, key + ["quantity", "price", "line_total",
    "dup_diff_qty", "dup_same_qty_diff_price", "dup_diff_qty_diff_price"]].sort_values(key)

price_details = invoice_items.loc[invoice_items["is_free"] | invoice_items["near_zero_price"],
                                  key + ["quantity", "price", "line_total", "is_free", "near_zero_price"]]

add_report("4. Invoice item checks",
           f"No rows were deleted in this step; these are review flags.\n\n"
           f"- Different quantity, same price: {int(invoice_items['dup_diff_qty'].sum()):,} rows\n"
           f"- Same quantity, different price: {int(invoice_items['dup_same_qty_diff_price'].sum()):,} rows\n"
           f"- Different quantity and price: {int(invoice_items['dup_diff_qty_diff_price'].sum()):,} rows\n"
           f"- Zero-price rows: {int(invoice_items['is_free'].sum()):,}\n"
           f"- Near-zero-price rows (0 < price < {NEAR_ZERO_LIMIT}): {int(invoice_items['near_zero_price'].sum()):,}",
           {"Repeated invoice + product records": duplicate_details,
            "Zero and near-zero price records": price_details})


# ----------------------------------------------------------------------
# 5. purchases: flag repeated invoice + product lines with different quantity
# ----------------------------------------------------------------------
n_quantities_purchases = purchases.groupby(key)["quantity"].transform("nunique")

purchases["dup_diff_qty"] = n_quantities_purchases > 1

n_pairs = len(purchases.loc[purchases["dup_diff_qty"], key].drop_duplicates())

print(f"purchases dup_diff_qty: {int(purchases['dup_diff_qty'].sum()):,} rows, {n_pairs:,} pairs\n")

purchase_flag_details = purchases.loc[purchases["dup_diff_qty"],
                                      key + ["quantity", "date", "CustomerID"]].sort_values(key)

add_report("5. Purchase repeated-line checks",
           f"No rows were deleted. Found {int(purchases['dup_diff_qty'].sum()):,} rows "
           f"across {n_pairs:,} invoice + product pairs with different quantities.",
           {"Flagged purchase records": purchase_flag_details})


# ----------------------------------------------------------------------
# 6. products: standardize categories + keyword check
# ----------------------------------------------------------------------
category_key = (products["category"].str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.lower())

products["category_clean"] = category_key.map(CATEGORY_MAP)

unknown_labels = products.loc[products["category_clean"].isna(), "category"].unique()

if len(unknown_labels) > 0:
    raise ValueError(f"Category labels not in CATEGORY_MAP: {list(unknown_labels)}")

print(f"Categories: {products['category'].nunique()} old labels -> {products['category_clean'].nunique()} final categories")

patterns = {
    category: {word: re.compile(r"\b" + re.escape(word) + r"(?:s|es)?\b")
               for word in words if word not in AMBIGUOUS}
    for category, words in KEYWORDS.items()
}


def match_keywords(item_name):
    """Return {category: [matched keywords]} for one item name."""
    text = item_name.lower().strip()
    found = {}

    for category, word_patterns in patterns.items():
        matched = [word for word, pattern in word_patterns.items() if pattern.search(text)]

        if matched:
            found[category] = matched

    return found


def best_category(found):
    """Category with the most matched keywords. Nothing if there is no match or a tie."""
    if not found:
        return None

    scores = {category: len(words) for category, words in found.items()}
    top_score = max(scores.values())
    top_categories = [category for category, score in scores.items() if score == top_score]

    return top_categories[0] if len(top_categories) == 1 else None


all_matches = products["item"].apply(match_keywords)

products["keyword_category"] = all_matches.apply(best_category)

products["keyword_matches"] = all_matches.apply(
    lambda found: "; ".join(f"{category}: {', '.join(words)}" for category, words in found.items()))

disagree = products["keyword_category"].notna() & (products["keyword_category"] != products["category_clean"])

products["proposed_category"] = products["keyword_category"].where(disagree)

n_no_match = int((all_matches.apply(len) == 0).sum())
n_tie = int(((all_matches.apply(len) > 0) & products["keyword_category"].isna()).sum())

print(f"Keyword check on {len(products):,} products: {int(disagree.sum()):,} proposals, "
      f"{n_no_match:,} with no keyword, {n_tie:,} ties (no proposal)")

category_counts = (products.groupby(["category", "category_clean"]).size()
                   .reset_index(name="product_count").sort_values(["category_clean", "category"]))

proposal_details = products.loc[disagree,
    ["product_id", "item", "category", "category_clean", "proposed_category", "keyword_matches"]]

tie_details = products.loc[(all_matches.apply(len) > 0) & products["keyword_category"].isna(),
                           ["product_id", "item", "category", "category_clean", "keyword_matches"]]

no_match_details = products.loc[all_matches.apply(len) == 0,
                                ["product_id", "item", "category", "category_clean"]]

add_report("6. Product categories and keyword review",
           f"Standardized {products['category'].nunique():,} original category labels into "
           f"{products['category_clean'].nunique():,} final categories. No product rows were removed.\n\n"
           f"- Products with a proposed category change: {int(disagree.sum()):,}\n"
           f"- Products with no keyword match: {n_no_match:,}\n"
           f"- Products with tied keyword results (no proposal): {n_tie:,}\n"
           "- Keyword suggestions are indicative and require manual review.",
           {"Original labels mapped to final categories": category_counts,
            "Products with proposed category changes": proposal_details,
            "Products with keyword ties": tie_details,
            "Products with no keyword match": no_match_details})

non_merch = products.loc[products["category_clean"] == "Non-merchandise", ["product_id", "item", "price"]]

print("\nItems in Non-merchandise (check: are they all really not products?):")
print(non_merch.to_string(index=False))
print()

report_sections.append("### Non-merchandise products to inspect\n\n" + md_table(non_merch))


# ----------------------------------------------------------------------
# 7. Merge the tables
# ----------------------------------------------------------------------
per_invoice = purchases.groupby("InvoiceID").agg(
    n_dates=("date", "nunique"),
    n_customers=("CustomerID", "nunique")
)

if not ((per_invoice["n_dates"] == 1).all() and (per_invoice["n_customers"] == 1).all()):
    raise ValueError("Some invoices have more than one date or customer, header table is not possible")

invoice_header = purchases[["InvoiceID", "date", "CustomerID"]].drop_duplicates("InvoiceID")

products_to_merge = products[["product_id", "item", "category", "category_clean", "price"]].rename(
    columns={"price": "list_price"})

rows_before_merge = len(invoice_items)

sales = invoice_items.merge(invoice_header, on="InvoiceID", how="left", validate="many_to_one")
sales = sales.merge(products_to_merge, on="product_id", how="left", validate="many_to_one")
sales = sales.merge(customers, on="CustomerID", how="left", validate="many_to_one")

if len(sales) != rows_before_merge:
    raise ValueError(f"Merge changed the number of rows: {rows_before_merge} -> {len(sales)}")

if not np.isclose(sales["line_total"].sum(), invoice_items["line_total"].sum()):
    raise ValueError("Merge changed total sales")

n_unmatched = int(sales[["date", "CustomerID", "item", "customer_type"]].isna().sum().sum())

if n_unmatched > 0:
    raise ValueError(f"{n_unmatched} values did not match in the merge")

log_step("merge", "sales", rows_before_merge, len(sales),
         invoice_items["line_total"].sum(), sales["line_total"].sum())

add_report("7. Merge results",
           f"Merged invoice items with the one-row-per-invoice header, products and customers.\n\n"
           f"- Rows before merge: {rows_before_merge:,}\n"
           f"- Rows after merge: {len(sales):,}\n"
           f"- Unmatched values: {n_unmatched:,}\n"
           f"- Total sales before merge: {invoice_items['line_total'].sum():,.2f}\n"
           f"- Total sales after merge: {sales['line_total'].sum():,.2f}\n"
           "- Validation: row count and sales total unchanged; all required merge fields matched.")


# ----------------------------------------------------------------------
# 8. Compare cleaned data with the original
# ----------------------------------------------------------------------
comparison = pd.DataFrame({
    "metric": ["invoice_items rows", "purchases rows", "products rows", "customers rows",
               "invoices (in invoice_items)", "products sold", "total sales"],
    "original": [len(invoice_items_raw), len(purchases_raw), len(products_raw), len(customers_raw),
                 invoice_items_raw["InvoiceID"].nunique(), invoice_items_raw["product_id"].nunique(),
                 round(invoice_items_raw["line_total"].sum(), 2)],
    "cleaned": [len(invoice_items), len(purchases), len(products), len(customers),
                invoice_items["InvoiceID"].nunique(), invoice_items["product_id"].nunique(),
                round(invoice_items["line_total"].sum(), 2)],
})

comparison["difference"] = (comparison["cleaned"] - comparison["original"]).round(2)

print("\nOriginal vs cleaned:")
print(comparison.to_string(index=False))

if invoice_items_raw["InvoiceID"].nunique() != invoice_items["InvoiceID"].nunique():
    raise ValueError("Some invoices disappeared during cleaning")


# ----------------------------------------------------------------------
# Save (the raw files in data/ are not touched)
# ----------------------------------------------------------------------
CLEAN_DIR.mkdir(exist_ok=True)

customers.to_csv(CLEAN_DIR / "customers_clean.csv", index=False)
invoice_items.to_csv(CLEAN_DIR / "invoice_items_clean.csv", index=False)
products.to_csv(CLEAN_DIR / "products_clean.csv", index=False)
purchases.to_csv(CLEAN_DIR / "purchases_clean.csv", index=False)
sales.to_csv(CLEAN_DIR / "sales_clean.csv", index=False)

# Products to review by hand (keywords disagree with the current category)
review_columns = ["product_id", "item", "category", "category_clean", "proposed_category", "keyword_matches"]

products.loc[disagree, review_columns].sort_values(["category_clean", "item"]).to_csv(
    CLEAN_DIR / "products_to_review.csv", index=False)

# Original vs cleaned report
comparison_report = [
    "# Original vs cleaned data comparison", "",
    "This report compares the original raw tables with the cleaned tables. "
    "Only exact duplicate rows were removed; review flags do not delete records.", "",
    md_table(comparison), "",
    "## Interpretation", "",
    f"- invoice_items: {len(invoice_items_raw):,} original rows -> {len(invoice_items):,} cleaned rows.",
    f"- purchases: {len(purchases_raw):,} original rows -> {len(purchases):,} cleaned rows.",
    f"- products and customers: no rows removed ({len(products):,} and {len(customers):,} rows respectively).",
    f"- Total sales: {invoice_items_raw['line_total'].sum():,.2f} original -> "
    f"{invoice_items['line_total'].sum():,.2f} cleaned.",
    "- The number of distinct invoices was checked and must remain unchanged."
]

(CLEAN_DIR / "comparison.md").write_text("\n".join(comparison_report), encoding="utf-8")

# Full step-by-step cleaning report
cleaning_report = [
    "# Data cleaning report", "",
    "Generated by `clean.py`. Raw CSV files are never modified.", "",
    *report_sections, ""
]

(CLEAN_DIR / "cleaning_report.md").write_text("\n\n".join(cleaning_report), encoding="utf-8")

print(f"\nDone. Files saved in {CLEAN_DIR}")
"""
generate_data.py  —  v2.0
--------------------------
Generates a realistic but intentionally messy synthetic retail sales
dataset with missing values, duplicates, outliers, and inconsistent
categoricals.

New vs v1
---------
  customer_id       – unique ID enabling repeat-purchase analysis
  city / state      – mapped to region (geographic granularity)
  loyalty_tier      – Bronze / Silver / Gold (correlated with spend)
  return_flag       – Electronics returns more likely
  shipping_days     – 1-10 days (region + category dependent)
  Seasonal pricing  – Electronics spike Oct–Dec
  Dataset size      – 3,000 rows (up from 2,000)
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ─────────────────────────────────────────────────────────────
N_ROWS       = 3_000
N_CUSTOMERS  = 800       # pool size — enables repeat purchases

REGION_CITY_STATE: dict = {
    "North":   [("Delhi", "Delhi"), ("Chandigarh", "Punjab"),
                ("Lucknow", "Uttar Pradesh"), ("Jaipur", "Rajasthan"),
                ("Amritsar", "Punjab")],
    "South":   [("Bengaluru", "Karnataka"), ("Chennai", "Tamil Nadu"),
                ("Hyderabad", "Telangana"), ("Kochi", "Kerala"),
                ("Coimbatore", "Tamil Nadu")],
    "East":    [("Kolkata", "West Bengal"), ("Bhubaneswar", "Odisha"),
                ("Patna", "Bihar"), ("Guwahati", "Assam"),
                ("Ranchi", "Jharkhand")],
    "West":    [("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"),
                ("Ahmedabad", "Gujarat"), ("Surat", "Gujarat"),
                ("Nashik", "Maharashtra")],
    "Central": [("Bhopal", "Madhya Pradesh"), ("Nagpur", "Maharashtra"),
                ("Raipur", "Chhattisgarh"), ("Indore", "Madhya Pradesh"),
                ("Jabalpur", "Madhya Pradesh")],
}

CATEGORIES      = ["Electronics", "Clothing", "Groceries",
                   "Furniture", "Sports", "Books"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Cash", "UPI", "Net Banking"]

PRODUCTS: dict = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones",
                    "Smart Watch", "Bluetooth Speaker", "Power Bank", "LED Monitor"],
    "Clothing":    ["T-Shirt", "Jeans", "Jacket", "Dress", "Sneakers",
                    "Formal Shirt", "Kurti", "Sports Shorts"],
    "Groceries":   ["Rice (5kg)", "Cooking Oil", "Sugar (1kg)", "Tea Powder",
                    "Biscuits", "Pulses (1kg)", "Ghee 500ml", "Atta (10kg)"],
    "Furniture":   ["Office Chair", "Study Table", "Bookshelf", "Sofa",
                    "Wardrobe", "Bed Frame", "Coffee Table", "Dining Set"],
    "Sports":      ["Cricket Bat", "Football", "Yoga Mat", "Dumbbells",
                    "Cycling Gloves", "Badminton Racket", "Running Shoes", "Jump Rope"],
    "Books":       ["Data Science 101", "Python Crash Course", "Fiction Novel",
                    "Business Guide", "Self Help", "Machine Learning A-Z",
                    "History of India", "Clean Code"],
}

PRICE_RANGES: dict = {
    "Electronics": (3_000, 80_000),
    "Clothing":    (200,    5_000),
    "Groceries":   (30,       800),
    "Furniture":   (2_500,  50_000),
    "Sports":      (150,    8_000),
    "Books":       (100,    1_500),
}

# Festive season price boost (months → multiplier)
SEASONAL_BOOST: dict = {
    "Electronics": {10: 1.40, 11: 1.50, 12: 1.60},
    "Clothing":    {10: 1.20, 11: 1.30, 12: 1.25},
}

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun",
    "Kavya", "Rahul", "Divya", "Amit", "Pooja", "Raj", "Meera",
    "Suresh", "Nisha", "Kiran", "Deepa", "Sanjay", "Lakshmi",
    "Aditi", "Kunal", "Riya", "Manish", "Swati", "Tarun", "Neha",
    "Vinod", "Geeta", "Harish",
]
LAST_NAMES = [
    "Sharma", "Patel", "Verma", "Singh", "Gupta", "Kumar", "Joshi",
    "Nair", "Reddy", "Mehta", "Shah", "Pillai", "Iyer", "Das", "Rao",
    "Kapoor", "Malhotra", "Tiwari", "Bose", "Chatterjee",
]

START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 12, 31)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


def _make_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _loyalty_tier(total_spend: float) -> str:
    if total_spend >= 50_000:
        return "Gold"
    if total_spend >= 15_000:
        return "Silver"
    return "Bronze"


def _shipping_days(region: str, category: str) -> int:
    base = {"North": 3, "South": 4, "East": 5, "West": 3, "Central": 4}
    extra = {"Electronics": 1, "Furniture": 2, "Groceries": 0,
              "Clothing": 0, "Sports": 1, "Books": 0}
    days = base.get(region, 3) + extra.get(category, 0) + random.randint(-1, 2)
    return max(1, min(10, days))


# ── Core Generation ───────────────────────────────────────────────────────────
def generate_clean_data(n: int = N_ROWS) -> pd.DataFrame:
    """Generate a clean, consistent retail dataset of n rows."""
    # Build a reusable customer pool (enables repeat purchases)
    customer_ids    = [f"CUST-{5000 + i}" for i in range(N_CUSTOMERS)]
    customer_names  = {cid: _make_name()                             for cid in customer_ids}
    customer_gender = {cid: random.choice(["Male", "Female"])        for cid in customer_ids}
    customer_age    = {
        cid: max(18, min(70, int(np.random.normal(35, 10))))
        for cid in customer_ids
    }
    cumulative_spend = {cid: 0.0 for cid in customer_ids}

    records = []
    for i in range(n):
        category   = random.choice(CATEGORIES)
        product    = random.choice(PRODUCTS[category])
        lo, hi     = PRICE_RANGES[category]
        order_date = _random_date(START_DATE, END_DATE)
        month      = order_date.month

        boost      = SEASONAL_BOOST.get(category, {}).get(month, 1.0)
        unit_price = round(np.random.uniform(lo, hi) * boost, 2)
        quantity   = int(np.random.choice([1, 2, 3, 4, 5],
                                           p=[0.40, 0.28, 0.17, 0.10, 0.05]))
        discount   = round(random.choice([0, 0, 0, 5, 10, 15, 20, 25]) / 100, 2)
        total_sales = round(unit_price * quantity * (1 - discount), 2)

        cid        = random.choice(customer_ids)
        region     = random.choice(list(REGION_CITY_STATE.keys()))
        city, state = random.choice(REGION_CITY_STATE[region])

        # Return probability: Electronics 15%, others 5%
        return_prob = 0.15 if category == "Electronics" else 0.05
        return_flag = random.random() < return_prob

        cumulative_spend[cid] += total_sales

        records.append({
            "order_id":          f"ORD-{10000 + i}",
            "customer_id":       cid,
            "customer_name":     customer_names[cid],
            "age":               customer_age[cid],
            "gender":            customer_gender[cid],
            "city":              city,
            "state":             state,
            "region":            region,
            "loyalty_tier":      _loyalty_tier(cumulative_spend[cid]),
            "product_category":  category,
            "product_name":      product,
            "quantity":          quantity,
            "unit_price":        unit_price,
            "discount":          discount,
            "total_sales":       total_sales,
            "order_date":        order_date.strftime("%Y-%m-%d"),
            "payment_method":    random.choice(PAYMENT_METHODS),
            "customer_rating":   round(max(1.0, min(5.0, np.random.normal(3.8, 0.7))), 1),
            "return_flag":       return_flag,
            "shipping_days":     _shipping_days(region, category),
        })

    return pd.DataFrame(records)


def introduce_mess(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic data quality issues for pipeline demonstration."""
    df  = df.copy()
    n   = len(df)
    rng = np.random.default_rng(SEED)

    # 1. Missing values (~8% spread across key columns)
    miss_config = {
        "age":              0.05,
        "gender":           0.04,
        "customer_rating":  0.07,
        "discount":         0.03,
        "unit_price":       0.02,
        "payment_method":   0.04,
        "region":           0.03,
        "city":             0.02,
        "shipping_days":    0.02,
    }
    for col, rate in miss_config.items():
        idx = rng.choice(df.index, size=int(n * rate), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Inconsistent gender encoding (~30% of each gender)
    for orig, dirty_pool in [
        ("Male",   ["male", "M", "MALE", "m"]),
        ("Female", ["female", "F", "FEMALE", "f"]),
    ]:
        mask = df["gender"] == orig
        dirty_idx = rng.choice(df[mask].index,
                                size=int(mask.sum() * 0.30), replace=False)
        df.loc[dirty_idx, "gender"] = rng.choice(dirty_pool, size=len(dirty_idx))

    # 3. Outlier prices (data-entry / fraud errors)
    spike_idx = rng.choice(df.index, size=20, replace=False)
    df.loc[spike_idx, "unit_price"] = rng.uniform(200_000, 999_999, size=20)

    # 4. Negative quantities (returns entered incorrectly)
    neg_idx = rng.choice(df.index, size=25, replace=False)
    df.loc[neg_idx, "quantity"] = rng.choice([-1, -2, -3], size=25)

    # 5. Duplicate rows (~3%)
    dup_rows = df.sample(n=int(n * 0.03), random_state=SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 6. Invalid dates (10 rows)
    bad_date_idx = rng.choice(df.index, size=12, replace=False)
    df.loc[bad_date_idx, "order_date"] = "99/99/9999"

    # 7. Out-of-range ratings
    bad_rating_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[bad_rating_idx, "customer_rating"] = rng.choice([0.0, 6.0, 7.5, -1.0],
                                                             size=15)

    # 8. Trailing / leading whitespace in categorical columns
    ws_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[ws_idx, "product_category"] = " " + df.loc[ws_idx, "product_category"] + " "

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ── Entry Point ───────────────────────────────────────────────────────────────
def generate_and_save(out_dir: Optional[str] = None) -> str:
    """Generate, corrupt, and save the raw dataset. Returns the file path."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "raw_sales_data.csv")

    print("  Generating synthetic retail sales dataset …")
    clean_df = generate_clean_data(N_ROWS)
    messy_df = introduce_mess(clean_df)
    messy_df.to_csv(path, index=False)

    print(f"  Raw dataset saved  → {os.path.abspath(path)}")
    print(f"  Shape              : {messy_df.shape[0]:,} rows × {messy_df.shape[1]} columns")
    print(f"  Nulls              : {messy_df.isnull().sum().sum():,} total missing values")
    print(f"  Duplicates         : {messy_df.duplicated().sum():,} exact duplicate rows")
    return path


if __name__ == "__main__":
    generate_and_save()

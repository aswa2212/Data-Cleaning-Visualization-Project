import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to python path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from data_cleaner import (
    DataQualityReport,
    step_remove_duplicates,
    step_fix_dates,
    step_standardise_categoricals,
    step_fix_invalid_values,
    step_impute_missing,
    step_remove_outliers,
    step_recompute_derived,
    step_type_cast,
    step_validate,
)

class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataframe that contains messy data matching our constraints
        self.report = DataQualityReport()
        self.test_data = pd.DataFrame({
            "order_id": [1, 2, 2, 3, 4, 5],
            "customer_id": ["C1", "C2", "C2", "C3", "C4", "C5"],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "NaT", None],
            "gender": ["Male", "female", "female", "F", None, "InvalidGender"],
            "age": [25.0, 30.0, 30.0, np.nan, 45.0, 50.0],
            "quantity": [2, -5, -5, 10, 1, 3],
            "unit_price": [10.0, 20.0, 20.0, 15.0, np.nan, 100.0],
            "discount": [0.0, 0.1, 0.1, 0.05, 0.0, 0.2],
            "customer_rating": [4.5, 3.0, 3.0, 6.0, np.nan, 4.0],
            "shipping_days": [2.0, 3.0, 3.0, np.nan, 5.0, 1.0],
            "payment_method": ["UPI", "Card", "Card", "UPI", "Cash", None],
            "region": ["North", "South", "South", "East", "West", None],
            "city": ["Delhi", "Mumbai", "Mumbai", "Kolkata", "Bengaluru", None],
            "product_category": ["Electronics", "Clothing", "Clothing", "Home", "Home", "Books"],
            "product_name": ["Phone", "T-shirt", "T-shirt", "Lamp", "Rug", "Novel"]
        })

    def test_remove_duplicates(self):
        df = step_remove_duplicates(self.test_data.copy(), self.report)
        self.assertEqual(len(df), 5)
        self.assertEqual(self.report.duplicates_removed, 1)

    def test_fix_dates(self):
        df = step_fix_dates(self.test_data.copy(), self.report)
        self.assertFalse(df["order_date"].isna().any())
        self.assertEqual(self.report.invalid_dates, 2)

    def test_gender_standardisation(self):
        df = step_standardise_categoricals(self.test_data.copy(), self.report)
        # female -> Female, MALE -> Male, F -> Female, Male -> Male
        # InvalidGender -> kept as-is, None -> None
        valid_genders = df["gender"].dropna().unique()
        self.assertIn("Male", valid_genders)
        self.assertIn("Female", valid_genders)
        self.assertIn("InvalidGender", valid_genders)
        self.assertEqual(self.report.gender_variants_fixed, 3)
        self.assertEqual(self.report.unknown_genders, 1)

    def test_no_negative_qty(self):
        df = step_fix_invalid_values(self.test_data.copy(), self.report)
        self.assertTrue((df["quantity"] >= 0).all())
        self.assertEqual(self.report.negative_qty_fixed, 2)

    def test_rating_range(self):
        df = step_fix_invalid_values(self.test_data.copy(), self.report)
        # rating 6.0 is out of bounds, should be made NaN
        self.assertTrue(df.loc[df["order_id"] == 3, "customer_rating"].isna().all())
        self.assertEqual(self.report.bad_ratings_nulled, 1)

    def test_no_nulls_after_impute(self):
        df = self.test_data.copy()
        # To test impute properly, first run steps up to impute
        df = step_remove_duplicates(df, self.report)
        df = step_fix_dates(df, self.report)
        df = step_standardise_categoricals(df, self.report)
        df = step_fix_invalid_values(df, self.report)
        df = step_impute_missing(df, self.report)
        
        # Verify no missing values in key columns
        self.assertFalse(df["age"].isna().any())
        self.assertFalse(df["unit_price"].isna().any())
        self.assertFalse(df["customer_rating"].isna().any())
        self.assertFalse(df["shipping_days"].isna().any())
        self.assertFalse(df["gender"].isna().any())
        self.assertFalse(df["payment_method"].isna().any())
        self.assertFalse(df["region"].isna().any())

if __name__ == "__main__":
    unittest.main()

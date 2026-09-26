import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==========================================
# KONFIGURASI KOLOM
# ==========================================

NUMERIC_COLUMNS = [
    "usia",
    "berat_badan_kg",
    "tinggi_badan_cm",
    "tekanan_darah_sistolik",
    "tekanan_darah_diastolik",
    "suhu_tubuh_c",
    "durasi_tidur_jam",
]

CATEGORICAL_COLUMNS = [
    "jenis_kelamin",
    "aktivitas_fisik",
    "merokok",
    "tingkat_stres",
]

TARGET_COLUMN = "kategori_keluhan"


# ==========================================
# 1. DATASET INFORMATION
# ==========================================

def get_dataset_info(df):

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
    }


# ==========================================
# 2. MISSING VALUE
# ==========================================

def check_missing_values(df):

    result = []

    for column in df.columns:

        count = int(df[column].isna().sum())

        percentage = round(
            (count / len(df)) * 100,
            2,
        )

        result.append({
            "column": column,
            "missing_count": count,
            "missing_percentage": percentage,
        })

    return result


# ==========================================
# 3. DUPLICATE
# ==========================================

def check_duplicates(df):

    return {
        "duplicate_rows": int(
            df.duplicated().sum()
        )
    }


# ==========================================
# 4. DATA INCONSISTENCY
# ==========================================

def check_inconsistent_data(df):

    result = {}

    expected_categories = {
        "jenis_kelamin": [
            "Laki-laki",
            "Perempuan",
        ],
        "aktivitas_fisik": [
            "Rendah",
            "Sedang",
            "Tinggi",
        ],
        "merokok": [
            "Ya",
            "Tidak",
        ],
        "tingkat_stres": [
            "Rendah",
            "Sedang",
            "Tinggi",
        ],
    }

    for column, valid_values in expected_categories.items():

        if column not in df.columns:
            continue

        invalid_mask = (
            df[column].notna()
            & ~df[column].isin(valid_values)
        )

        invalid_values = (
            df.loc[invalid_mask, column]
            .drop_duplicates()
            .tolist()
        )

        result[column] = {
            "valid_values": valid_values,
            "invalid_count": int(
                invalid_mask.sum()
            ),
            "invalid_values": invalid_values,
        }

    return result


# ==========================================
# 5. OUTLIER - IQR
# ==========================================

def check_outliers(df):

    result = {}

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        series = df[column].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        result[column] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(
                float(lower_bound), 2
            ),
            "upper_bound": round(
                float(upper_bound), 2
            ),
            "outlier_count": int(
                outlier_mask.sum()
            ),
        }

    return result


# ==========================================
# 6. HANDLE MISSING VALUE
# ==========================================

def handle_missing_values(df):

    df = df.copy()

    imputation_log = []

    # --------------------------------------
    # MEDIAN
    # --------------------------------------

    median_columns = [
        "usia",
        "berat_badan_kg",
        "tinggi_badan_cm",
        "tekanan_darah_sistolik",
        "tekanan_darah_diastolik",
    ]

    for column in median_columns:

        if column not in df.columns:
            continue

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        value = df[column].median()

        df[column] = df[column].fillna(value)

        imputation_log.append({
            "column": column,
            "method": "median",
            "value": round(float(value), 2),
            "filled_count": missing_count,
        })

    # --------------------------------------
    # MEAN
    # --------------------------------------

    mean_columns = [
        "suhu_tubuh_c",
        "durasi_tidur_jam",
    ]

    for column in mean_columns:

        if column not in df.columns:
            continue

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        value = df[column].mean()

        df[column] = df[column].fillna(value)

        imputation_log.append({
            "column": column,
            "method": "mean",
            "value": round(float(value), 2),
            "filled_count": missing_count,
        })

    # --------------------------------------
    # MODUS
    # --------------------------------------

    for column in CATEGORICAL_COLUMNS:

        if column not in df.columns:
            continue

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        mode = df[column].mode()

        if mode.empty:
            continue

        value = mode.iloc[0]

        df[column] = df[column].fillna(value)

        imputation_log.append({
            "column": column,
            "method": "mode",
            "value": value,
            "filled_count": missing_count,
        })

    return df, imputation_log


# ==========================================
# 7. HANDLE INCONSISTENT DATA
# ==========================================

def handle_inconsistent_data(df):

    df = df.copy()

    mappings = {

        "jenis_kelamin": {
            "L": "Laki-laki",
            "Male": "Laki-laki",
            "M": "Laki-laki",
            "P": "Perempuan",
            "Female": "Perempuan",
        },

        "aktivitas_fisik": {
            "low": "Rendah",
            "Low": "Rendah",
            "medium": "Sedang",
            "Medium": "Sedang",
            "high": "Tinggi",
            "High": "Tinggi",
        },

        "merokok": {
            "Yes": "Ya",
            "yes": "Ya",
            "No": "Tidak",
            "no": "Tidak",
        },

        "tingkat_stres": {
            "low": "Rendah",
            "Low": "Rendah",
            "medium": "Sedang",
            "Medium": "Sedang",
            "high": "Tinggi",
            "High": "Tinggi",
        },
    }

    cleaning_log = []

    for column, mapping in mappings.items():

        if column not in df.columns:
            continue

        before = df[column].copy()

        df[column] = df[column].replace(mapping)

        changed = int(
            (before != df[column])
            .fillna(False)
            .sum()
        )

        if changed > 0:
            cleaning_log.append({
                "column": column,
                "changed_count": changed,
                "action": "standardisasi kategori",
            })

    return df, cleaning_log


# ==========================================
# 8. HANDLE DUPLICATE
# ==========================================

def handle_duplicates(df):

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates().copy()

    return df, duplicate_count


# ==========================================
# 9. HANDLE OUTLIER
# ==========================================

def handle_outliers(df):

    df = df.copy()

    outlier_log = []

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        mask = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        # Winsorization sederhana:
        # nilai outlier diganti dengan batas IQR

        df.loc[
            df[column] < lower,
            column
        ] = lower

        df.loc[
            df[column] > upper,
            column
        ] = upper

        outlier_log.append({
            "column": column,
            "outlier_count": count,
            "lower_bound": round(
                float(lower), 2
            ),
            "upper_bound": round(
                float(upper), 2
            ),
            "action": "capping menggunakan batas IQR",
        })

    return df, outlier_log


# ==========================================
# 10. FEATURE ENGINEERING
# ==========================================

def create_features(df):

    df = df.copy()

    # BMI
    df["bmi"] = (
        df["berat_badan_kg"]
        / (
            df["tinggi_badan_cm"] / 100
        ) ** 2
    )

    df["bmi"] = df["bmi"].round(2)

    return df


# ==========================================
# 11. ENCODING
# ==========================================

def encode_data(df):

    df = df.copy()

    columns_to_encode = [
        "jenis_kelamin",
        "aktivitas_fisik",
        "merokok",
        "tingkat_stres",
    ]

    df = pd.get_dummies(
        df,
        columns=columns_to_encode,
        drop_first=True,
        dtype=int,
    )

    return df


# ==========================================
# 12. NORMALIZATION / STANDARDIZATION
# ==========================================

def standardize_data(df):

    df = df.copy()

    scaler = StandardScaler()

    columns = [
        "usia",
        "berat_badan_kg",
        "tinggi_badan_cm",
        "tekanan_darah_sistolik",
        "tekanan_darah_diastolik",
        "suhu_tubuh_c",
        "durasi_tidur_jam",
        "bmi",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    df[available_columns] = scaler.fit_transform(
        df[available_columns]
    )

    return df


# ==========================================
# 13. FULL PREPROCESSING
# ==========================================

def preprocess_dataset(df):

    # BEFORE
    before = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicates": int(
            df.duplicated().sum()
        ),
    }

    # --------------------------------------
    # Missing
    # --------------------------------------

    df, imputation_log = handle_missing_values(df)

    # --------------------------------------
    # Inconsistent
    # --------------------------------------

    df, inconsistent_log = handle_inconsistent_data(df)

    # --------------------------------------
    # Duplicate
    # --------------------------------------

    df, duplicate_count = handle_duplicates(df)

    # --------------------------------------
    # Outlier
    # --------------------------------------

    df, outlier_log = handle_outliers(df)

    # --------------------------------------
    # Feature Engineering
    # --------------------------------------

    df = create_features(df)

    # --------------------------------------
    # Encoding
    # --------------------------------------

    df = encode_data(df)

    # --------------------------------------
    # Standardization
    # --------------------------------------

    df = standardize_data(df)

    # AFTER
    after = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicates": int(
            df.duplicated().sum()
        ),
    }

    return {
        "data": df,
        "before": before,
        "after": after,
        "imputation": imputation_log,
        "inconsistent_data": inconsistent_log,
        "duplicates_removed": duplicate_count,
        "outliers": outlier_log,
    }
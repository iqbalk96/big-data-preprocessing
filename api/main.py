from io import BytesIO

import pandas as pd

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from preprocessing import (
    get_dataset_info,
    check_missing_values,
    check_duplicates,
    check_inconsistent_data,
    check_outliers,
    preprocess_dataset,
)


app = FastAPI(
    title="Healthcare Dataset Preprocessing API",
    description="API untuk preprocessing dataset kesehatan",
    version="1.0.0",
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Healthcare Dataset Preprocessing API",
        "status": "running",
    }


# ==========================================
# PREPROCESS
# ==========================================

@app.post("/preprocess")
async def preprocess(
    file: UploadFile = File(...)
):

    # ==========================================
    # VALIDASI
    # ==========================================

    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="File harus berformat CSV.",
        )

    # ==========================================
    # READ CSV
    # ==========================================

    try:

        contents = await file.read()

        df = pd.read_csv(
            BytesIO(contents)
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"CSV tidak dapat dibaca: {str(error)}",
        )

    # ==========================================
    # DATASET INFORMATION
    # ==========================================

    dataset_info = get_dataset_info(df)

    # ==========================================
    # QUALITY CHECK
    # ==========================================

    missing_values = check_missing_values(df)

    duplicates = check_duplicates(df)

    inconsistent_data = check_inconsistent_data(df)

    outliers = check_outliers(df)

    # ==========================================
    # PREPROCESSING
    # ==========================================

    result = preprocess_dataset(df)

    processed_df = result["data"]

    # ==========================================
    # PREVIEW BEFORE
    # ==========================================

    before_preview = (
        df.head(5)
        .astype(object)
        .where(
            pd.notna(df.head(5)),
            None,
        )
        .to_dict(orient="records")
    )

    # ==========================================
    # PREVIEW AFTER
    # ==========================================

    after_preview = (
        processed_df.head(5)
        .astype(object)
        .where(
            pd.notna(processed_df.head(5)),
            None,
        )
        .to_dict(orient="records")
    )

    # ==========================================
    # RESPONSE
    # ==========================================

    return {

        "message": "Dataset berhasil diproses.",

        "filename": file.filename,

        "dataset": dataset_info,

        "quality_check": {

            "missing_values": missing_values,

            "duplicates": duplicates,

            "inconsistent_data": inconsistent_data,

            "outliers": outliers,
        },

        "cleaning": {

            "imputation": result["imputation"],

            "inconsistent_data": result[
                "inconsistent_data"
            ],

            "duplicates_removed": result[
                "duplicates_removed"
            ],

            "outliers": result["outliers"],
        },

        "transformation": {

            "encoding": "One-Hot Encoding",

            "standardization": "StandardScaler",
        },

        "feature_engineering": [

            "BMI"
        ],

        "comparison": {

            "before": result["before"],

            "after": result["after"],
        },

        "preview": {

            "before": before_preview,

            "after": after_preview,
        },
    }
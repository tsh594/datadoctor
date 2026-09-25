from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
from datadoctor import Doctor
import io

app = FastAPI(
    title="Data Doctor API",
    description="An API for automated data cleaning and ML pipeline preparation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Welcome to the Data Doctor API",
        "version": "0.1.0",
    }


@app.post("/clean/")
async def clean_data(file: UploadFile = File(...)):
    """
    Accepts a CSV file, runs diagnose() and treat() on it,
    and returns the diagnosis report plus a preview of the cleaned data.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.StringIO(contents.decode("latin-1")))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    doctor = Doctor(df)
    report = doctor.diagnose()
    cleaned_df = doctor.treat()

    preview = cleaned_df.head(100).to_dict(orient="records")

    return {
        "message": "Data cleaned successfully",
        "diagnosis": report,
        "cleaned_preview": preview,
        "cleaned_row_count": len(cleaned_df),
        "cleaned_column_count": len(cleaned_df.columns),
    }


@app.post("/clean/csv/")
async def clean_data_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV file and returns the cleaned data as a downloadable CSV.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV file.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

    doctor = Doctor(df)
    doctor.diagnose()
    cleaned_df = doctor.treat()

    stream = io.StringIO()
    cleaned_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=cleaned_{file.filename}"
        },
    )
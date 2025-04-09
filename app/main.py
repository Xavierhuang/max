from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Body, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
import os
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import tempfile
import shutil
import logging
import traceback
import re
import io
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.figure import Figure
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

from app.llm import llm_assistant

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DataCleaner API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store datasets in memory (in a real app, use a database or file storage)
datasets = {}
transformation_logs = {}

# Create absolute path for data directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define Pydantic models for API
class NaturalLanguageTransform(BaseModel):
    query: str

class LLMQuery(BaseModel):
    query: str

class LLMResponse(BaseModel):
    response: str
    suggested_operation: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    analysis_type: str  # 'descriptive', 'correlation', 'regression', 'comparison'
    target_column: Optional[str] = None  # For regression: dependent variable
    feature_columns: Optional[List[str]] = None  # For regression: independent variables
    group_column: Optional[str] = None  # For comparison analysis (t-test, ANOVA)
    value_column: Optional[str] = None  # For comparison analysis (t-test, ANOVA)

class AnalysisResponse(BaseModel):
    result: Dict[str, Any]
    plots: Optional[List[str]] = None  # Base64 encoded plot images
    error: Optional[str] = None

# OpenAI client initialization with error handling
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set. AI features will not work.")
        openai_client = None
    else:
        openai_client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    openai_client = None

@app.get("/")
async def read_root():
    return {"message": "Welcome to DataCleaner API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Write the uploaded file to the temporary file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        logger.info(f"File uploaded successfully: {file.filename}")

        # Generate a unique dataset ID
        dataset_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Save the original file
        file_path = os.path.join(DATA_DIR, f"{dataset_id}_original{file_extension}")
        shutil.move(temp_path, file_path)

        # Read the file into a pandas DataFrame
        try:
            # Determine how to read the file based on its extension
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t')
            else:
                # Default to csv
                df = pd.read_csv(file_path)

            logger.info(f"File read successfully. Shape: {df.shape}")
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=error_msg)

        # Convert DataFrame to a list of dictionaries (one per row)
        try:
            # Handle non-serializable data types (like datetime)
            df_json = df.replace({np.nan: None})
            for col in df_json.columns:
                if df_json[col].dtype.name == 'datetime64[ns]':
                    df_json[col] = df_json[col].astype(str)
            
            # Convert to records
            data = df_json.to_dict(orient='records')
            
            # Get column names
            columns = list(df.columns)
            
            logger.info(f"Data processed successfully. Columns: {columns}")
        except Exception as e:
            error_msg = f"Error converting data: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

        # Store the dataset in memory and on disk
        datasets[dataset_id] = {
            "df": df,
            "file_path": file_path,
            "filename": file.filename,
            "operation_log": []
        }
        
        # Return the dataset ID and a preview of the data
        return {
            "dataset_id": dataset_id,
            "columns": columns,
            "data": data,
            "rows": len(df)
        }
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        file.file.close()

@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, sample_size: int = 100):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    
    # Get a sample of the data for preview
    sample = df.head(sample_size)
    
    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": json.loads(sample.to_json(orient="records")),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }

@app.post("/transform/{dataset_id}")
async def transform_data(dataset_id: str, operation: Dict[str, Any] = Body(...)):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = datasets[dataset_id]["df"].copy()
        rows_before = len(df)
        op_type = operation.get("type")

        # Apply the requested transformation
        if op_type == "drop_columns":
            columns = operation.get("columns", [])
            df = df.drop(columns=columns)

        elif op_type == "rename_column":
            old_name = operation.get("old_name")
            new_name = operation.get("new_name")
            df = df.rename(columns={old_name: new_name})

        elif op_type == "fill_na":
            column = operation.get("column")
            method = operation.get("method")
            value = operation.get("value")

            if method == "value":
                df[column] = df[column].fillna(value)
            elif method == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif method == "median":
                df[column] = df[column].fillna(df[column].median())
            elif method == "mode":
                df[column] = df[column].fillna(df[column].mode()[0])
            elif method == "ffill":
                df[column] = df[column].ffill()
            elif method == "bfill":
                df[column] = df[column].bfill()

        elif op_type == "change_type":
            column = operation.get("column")
            new_type = operation.get("new_type")

            if new_type == "int":
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif new_type == "float":
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif new_type == "string":
                df[column] = df[column].astype(str)
            elif new_type == "datetime":
                df[column] = pd.to_datetime(df[column], errors='coerce')

        elif op_type == "filter_rows":
            column = operation.get("column")
            condition = operation.get("condition")
            value = operation.get("value")

            if condition == "equals":
                df = df[df[column] == value]
            elif condition == "not_equals":
                df = df[df[column] != value]
            elif condition == "greater_than":
                df = df[df[column] > float(value)]
            elif condition == "less_than":
                df = df[df[column] < float(value)]
            elif condition == "contains":
                df = df[df[column].astype(str).str.contains(str(value))]

        elif op_type == "remove_duplicates":
            columns = operation.get("columns")
            if columns:
                df = df.drop_duplicates(subset=columns)
            else:
                df = df.drop_duplicates()

        # Calculate affected rows
        rows_after = len(df)
        rows_affected = abs(rows_after - rows_before)

        # Log the operation
        datasets[dataset_id]["operation_log"].append(operation)
        
        # Update the dataset in memory
        datasets[dataset_id]["df"] = df
        
        # Convert DataFrame to a list of dictionaries for the response
        df_json = df.replace({np.nan: None})
        for col in df_json.columns:
            if df_json[col].dtype.name == 'datetime64[ns]':
                df_json[col] = df_json[col].astype(str)
        
        data = df_json.to_dict(orient='records')
        columns = list(df.columns)
        
        return {
            "columns": columns,
            "data": data,
            "rows": len(df),
            "affected_rows": rows_affected
        }
    except Exception as e:
        error_msg = f"Error applying transformation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/natural-language-transform/{dataset_id}")
def natural_language_transform(dataset_id: str, transform_request: NaturalLanguageTransform):
    """
    Endpoint to process natural language transformation requests
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    query = transform_request.query
    
    try:
        # Use LLM to parse the natural language query
        parsed_operation = llm_assistant.parse_natural_language_transformation(query, list(df.columns))
        
        if parsed_operation.get("operation_type") == "unknown":
            return {
                "success": False,
                "error": parsed_operation.get("error", "Could not parse the transformation request"),
                "api_key_missing": not llm_assistant.has_api_key
            }
        
        # Convert the parsed operation to our internal format
        operation = {
            "type": parsed_operation["operation_type"],
            **parsed_operation.get("parameters", {})
        }
        
        # Process the operation using our existing transform endpoint
        return transform_data(dataset_id, operation)
    except Exception as e:
        print(f"Error in natural language transformation: {str(e)}")
        
        # Try to parse common phrases without LLM
        simple_parsed = {
            "type": "unknown",
            "error": "Failed to parse request and LLM assistance is unavailable."
        }
        
        # Simple pattern matching for common operations
        query_lower = query.lower()
        
        if "duplicates" in query_lower or "duplicate rows" in query_lower:
            simple_parsed = {"type": "remove_duplicates"}
        elif any(x in query_lower for x in ["drop", "remove"]) and "column" in query_lower:
            for col in df.columns:
                if col.lower() in query_lower:
                    simple_parsed = {"type": "drop_columns", "columns": [col]}
                    break
        
        if simple_parsed["type"] != "unknown":
            try:
                return transform_data(dataset_id, simple_parsed)
            except Exception:
                pass
        
        return {
            "success": False,
            "error": "Could not process natural language request. Please try using the manual transformations instead.",
            "api_key_missing": True
        }

@app.post("/analyze/{dataset_id}")
async def analyze_dataset(dataset_id: str, analysis_request: dict = Body(...)):
    """
    Perform statistical analysis on the dataset.
    
    Parameters:
    - dataset_id: The ID of the dataset to analyze
    - analysis_request: The analysis configuration
      - analysis_type: Type of analysis ('descriptive', 'correlation', 'regression', 'comparison')
      - target_column: For regression, the column to predict
      - feature_columns: For regression, the columns used as features
      - group_column: For comparison, the column defining groups
      - value_column: For comparison, the numeric column to compare
    """
    try:
        # Log the received dataset ID for debugging
        logger.info(f"Received analysis request for dataset ID: {dataset_id}")
        logger.info(f"Available datasets: {list(datasets.keys())}")
        
        # Check if dataset exists
        if dataset_id not in datasets:
            logger.error(f"Dataset {dataset_id} not found in datasets dictionary")
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Get the dataframe directly from memory
        df = datasets[dataset_id]["df"]
        
        # Get analysis parameters
        analysis_type = analysis_request.get("analysis_type")
        
        # Initialize result
        result = {}
        
        if analysis_type == "descriptive":
            # Get descriptive statistics for numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            result["summary"] = numeric_df.describe().to_dict()
        
        elif analysis_type == "correlation":
            # Calculate correlation matrix for numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                return {"error": "Need at least 2 numeric columns for correlation analysis"}
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr().to_dict()
            result["correlation_matrix"] = corr_matrix
        
        elif analysis_type == "regression":
            target_column = analysis_request.get("target_column")
            feature_columns = analysis_request.get("feature_columns", [])
            
            if not target_column or not feature_columns:
                return {"error": "Target column and at least one feature column required"}
            
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found in dataset"}
            
            # Verify all feature columns exist
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                return {"error": f"Feature columns not found: {', '.join(missing_columns)}"}
            
            # Check if enough data for regression
            if len(df) < 2:
                return {"error": "Not enough data for regression analysis"}
            
            # Create X and y datasets
            y = df[target_column]
            X = df[feature_columns]
            
            # Count missing or infinite values before cleaning
            missing_count_before = X.isna().sum().sum() + np.isinf(X).sum().sum()
            y_missing_before = y.isna().sum() + np.isinf(y).sum()
            
            # Clean data - remove rows with NaN or infinite values
            valid_rows = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1) | y.isna() | np.isinf(y))
            
            if valid_rows.sum() < 2:
                return {"error": "Not enough valid data for regression after removing missing values"}
            
            # Filter data to valid rows only
            X = X[valid_rows]
            y = y[valid_rows]
            
            # Log how many rows were removed
            rows_removed = len(df) - len(X)
            
            # Add constant for intercept
            X = sm.add_constant(X)
            
            try:
                # Fit regression model
                model = sm.OLS(y, X).fit()
                
                # Prepare summary
                coefficients = model.params.to_dict()
                std_errors = model.bse.to_dict()
                p_values = model.pvalues.to_dict()
                
                result["summary"] = {
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                    "f_statistic": model.fvalue,
                    "p_value": model.f_pvalue,
                    "coefficients": coefficients,
                    "std_errors": std_errors,
                    "p_values": p_values,
                    "data_cleaning": {
                        "total_rows": len(df),
                        "rows_used": len(X),
                        "rows_removed": rows_removed,
                        "missing_values_before": int(missing_count_before + y_missing_before)
                    }
                }
            except Exception as e:
                logger.error(f"Regression failed: {str(e)}")
                logger.error(traceback.format_exc())
                return {"error": f"Regression failed: {str(e)}"}
        
        elif analysis_type == "comparison":
            group_column = analysis_request.get("group_column")
            value_column = analysis_request.get("value_column")
            
            if not group_column or not value_column:
                return {"error": "Group column and value column required"}
            
            if group_column not in df.columns:
                return {"error": f"Group column '{group_column}' not found in dataset"}
                
            if value_column not in df.columns:
                return {"error": f"Value column '{value_column}' not found in dataset"}
            
            # Get unique groups
            groups = df[group_column].unique()
            
            if len(groups) < 2:
                return {"error": "Need at least 2 groups for comparison"}
            
            if len(groups) == 2:
                # For 2 groups, perform t-test
                group1 = df[df[group_column] == groups[0]][value_column].dropna()
                group2 = df[df[group_column] == groups[1]][value_column].dropna()
                
                if len(group1) < 2 or len(group2) < 2:
                    return {"error": "Not enough data in one or both groups"}
                
                try:
                    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    result["t_test"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "group1_mean": group1.mean(),
                        "group1_std": group1.std(),
                        "group1_count": len(group1),
                        "group2_mean": group2.mean(),
                        "group2_std": group2.std(),
                        "group2_count": len(group2)
                    }
                except Exception as e:
                    return {"error": f"T-test failed: {str(e)}"}
            else:
                # For 3+ groups, perform ANOVA
                group_stats = {}
                groups_data = []
                
                for group in groups:
                    group_data = df[df[group_column] == group][value_column].dropna()
                    if len(group_data) < 2:
                        continue
                    
                    groups_data.append(group_data)
                    group_stats[str(group)] = {
                        "mean": group_data.mean(),
                        "std": group_data.std(),
                        "count": len(group_data)
                    }
                
                if len(groups_data) < 2:
                    return {"error": "Not enough valid groups with data"}
                
                try:
                    f_stat, p_value = stats.f_oneway(*groups_data)
                    
                    result["anova"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "group_stats": group_stats
                    }
                except Exception as e:
                    return {"error": f"ANOVA failed: {str(e)}"}
        
        else:
            return {"error": f"Unsupported analysis type: {analysis_type}"}
        
        return {"result": result}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/suggest-transformations/{dataset_id}")
def suggest_transformations(dataset_id: str):
    """
    Get AI-suggested transformations for the dataset
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = datasets[dataset_id]["df"]
    
    try:
        analysis = llm_assistant.detect_data_issues(df)
        
        # Check if the OpenAI API key is available
        api_key_missing = not llm_assistant.has_api_key
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "suggested_transformations": analysis.get("transformations", []),
            "api_key_missing": api_key_missing
        }
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        
        # Use fallback approach
        fallback_analysis = llm_assistant._fallback_detect_data_issues(df)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "suggested_transformations": fallback_analysis.get("transformations", []),
            "api_key_missing": True
        }

@app.get("/transformations/{dataset_id}")
def get_transformations(dataset_id: str):
    if dataset_id not in transformation_logs:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"transformations": transformation_logs[dataset_id]}

@app.post("/export/{dataset_id}")
async def export_data(dataset_id: str, format: str = "csv"):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = datasets[dataset_id]["df"]
        base_name = os.path.splitext(os.path.basename(datasets[dataset_id]["file_path"]))[0]
        output_dir = os.path.join(DATA_DIR, "exports")
        os.makedirs(output_dir, exist_ok=True)

        # Export the data based on the requested format
        if format == "csv":
            output_path = os.path.join(output_dir, f"{base_name}.csv")
            df.to_csv(output_path, index=False)
        elif format == "xlsx":
            output_path = os.path.join(output_dir, f"{base_name}.xlsx")
            df.to_excel(output_path, index=False)
        elif format == "json":
            output_path = os.path.join(output_dir, f"{base_name}.json")
            df.to_json(output_path, orient="records", date_format="iso")
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

        return FileResponse(
            path=output_path,
            filename=os.path.basename(output_path),
            media_type="application/octet-stream"
        )
    except Exception as e:
        error_msg = f"Error exporting data: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/undo/{dataset_id}")
async def undo_operation(dataset_id: str, params: Dict[str, Any] = Body({})):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        # Check if we should restore to original
        restore_to_original = params.get("restore_to_original", False)

        if restore_to_original:
            # Just reload the original file
            file_path = datasets[dataset_id]["file_path"]
            file_extension = os.path.splitext(file_path)[1].lower()

            # Read the original file
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t')
            else:
                # Default to csv
                df = pd.read_csv(file_path)
            
            # Reset operation log
            datasets[dataset_id]["operation_log"] = []
        else:
            # Check if there are operations to undo
            if not datasets[dataset_id]["operation_log"]:
                raise HTTPException(status_code=400, detail="No operations to undo")

            # Remove the last operation from the log
            datasets[dataset_id]["operation_log"].pop()

            # Reload the original file
            file_path = datasets[dataset_id]["file_path"]
            file_extension = os.path.splitext(file_path)[1].lower()

            # Read the original file
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension in ['.tsv', '.txt']:
                df = pd.read_csv(file_path, sep='\t')
            else:
                # Default to csv
                df = pd.read_csv(file_path)

            # Re-apply all operations except the last one
            for operation in datasets[dataset_id]["operation_log"]:
                op_type = operation.get("type")

                # Apply the transformation (same code as in transform_data)
                if op_type == "drop_columns":
                    columns = operation.get("columns", [])
                    df = df.drop(columns=columns)

                elif op_type == "rename_column":
                    old_name = operation.get("old_name")
                    new_name = operation.get("new_name")
                    df = df.rename(columns={old_name: new_name})

                elif op_type == "fill_na":
                    column = operation.get("column")
                    method = operation.get("method")
                    value = operation.get("value")

                    if method == "value":
                        df[column] = df[column].fillna(value)
                    elif method == "mean":
                        df[column] = df[column].fillna(df[column].mean())
                    elif method == "median":
                        df[column] = df[column].fillna(df[column].median())
                    elif method == "mode":
                        df[column] = df[column].fillna(df[column].mode()[0])
                    elif method == "ffill":
                        df[column] = df[column].ffill()
                    elif method == "bfill":
                        df[column] = df[column].bfill()

                elif op_type == "change_type":
                    column = operation.get("column")
                    new_type = operation.get("new_type")

                    if new_type == "int":
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif new_type == "float":
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif new_type == "string":
                        df[column] = df[column].astype(str)
                    elif new_type == "datetime":
                        df[column] = pd.to_datetime(df[column], errors='coerce')

                elif op_type == "filter_rows":
                    column = operation.get("column")
                    condition = operation.get("condition")
                    value = operation.get("value")

                    if condition == "equals":
                        df = df[df[column] == value]
                    elif condition == "not_equals":
                        df = df[df[column] != value]
                    elif condition == "greater_than":
                        df = df[df[column] > float(value)]
                    elif condition == "less_than":
                        df = df[df[column] < float(value)]
                    elif condition == "contains":
                        df = df[df[column].astype(str).str.contains(str(value))]

                elif op_type == "remove_duplicates":
                    columns = operation.get("columns")
                    if columns:
                        df = df.drop_duplicates(subset=columns)
                    else:
                        df = df.drop_duplicates()

        # Update the dataset in memory
        datasets[dataset_id]["df"] = df
        
        # Convert DataFrame to a list of dictionaries for the response
        df_json = df.replace({np.nan: None})
        for col in df_json.columns:
            if df_json[col].dtype.name == 'datetime64[ns]':
                df_json[col] = df_json[col].astype(str)
        
        data = df_json.to_dict(orient='records')
        columns = list(df.columns)
        
        return {
            "columns": columns,
            "data": data,
            "rows": len(df)
        }
    except Exception as e:
        error_msg = f"Error undoing operation: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/datasets")
async def list_datasets():
    try:
        result = []
        for dataset_id, dataset in datasets.items():
            result.append({
                "dataset_id": dataset_id,
                "filename": dataset["filename"],
                "rows": len(dataset["df"]),
                "columns": list(dataset["df"].columns)
            })
        return {"datasets": result}
    except Exception as e:
        error_msg = f"Error listing datasets: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/ai-assistant/{dataset_id}")
async def ai_assistant(dataset_id: str, query_data: LLMQuery):
    """
    Process a natural language query about the dataset and return insights or suggested operations
    """
    try:
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        df = datasets[dataset_id]["df"]
        query = query_data.query.lower()
        
        # Simple keyword-based response system
        # In a real implementation, this would use an actual LLM API
        response = "I'm not sure how to help with that query."
        suggested_operation = None
        insights = None
        
        # Extract basic dataset statistics
        row_count = len(df)
        column_count = len(df.columns)
        
        # Process missing values query
        if "missing" in query or "null" in query or "na" in query:
            missing_info = {}
            for column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    missing_info[column] = {
                        "count": int(missing_count),
                        "percentage": round(missing_count / len(df) * 100, 2)
                    }
            
            if missing_info:
                response = f"I found missing values in the following columns:\n"
                for column, info in missing_info.items():
                    response += f"- {column}: {info['count']} missing values ({info['percentage']}%)\n"
                response += "\nWould you like me to help you fill these missing values?"
            else:
                response = "Good news! I didn't find any missing values in your dataset."
                
            insights = {
                "rowCount": row_count,
                "columnCount": column_count,
                "missingValues": missing_info
            }
        
        # Process statistics query
        elif "average" in query or "mean" in query or "statistic" in query:
            # Try to identify which column the user is asking about
            target_column = None
            for col in df.columns:
                if col.lower() in query:
                    target_column = col
                    break
            
            if target_column:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    avg_value = df[target_column].mean()
                    response = f"The average {target_column} is {avg_value:.2f}"
                    
                    # Include more statistics
                    insights = {
                        "column": target_column,
                        "statistics": {
                            "mean": round(float(df[target_column].mean()), 2),
                            "median": round(float(df[target_column].median()), 2),
                            "min": round(float(df[target_column].min()), 2),
                            "max": round(float(df[target_column].max()), 2),
                            "std": round(float(df[target_column].std()), 2)
                        }
                    }
                else:
                    response = f"The column '{target_column}' is not numeric, so I cannot calculate the average."
            else:
                # Summarize numeric columns
                numeric_stats = {}
                for column in df.select_dtypes(include=np.number).columns:
                    numeric_stats[column] = {
                        "mean": round(float(df[column].mean()), 2),
                        "median": round(float(df[column].median()), 2),
                        "min": round(float(df[column].min()), 2),
                        "max": round(float(df[column].max()), 2)
                    }
                
                if numeric_stats:
                    response = "Here are statistics for numeric columns:\n"
                    for column, stats in numeric_stats.items():
                        response += f"- {column}: mean={stats['mean']}, median={stats['median']}, min={stats['min']}, max={stats['max']}\n"
                    
                    insights = {
                        "numericColumns": numeric_stats
                    }
                else:
                    response = "I couldn't find any numeric columns to calculate statistics on."
        
        # Process fill missing values query
        elif "fill" in query:
            # Try to identify which column and method/value
            target_column = None
            for col in df.columns:
                if col.lower() in query:
                    target_column = col
                    break
            
            # Determine fill method
            fill_method = "value"  # Default
            fill_value = "0"  # Default
            
            if "mean" in query or "average" in query:
                fill_method = "mean"
            elif "median" in query:
                fill_method = "median"
            elif "mode" in query:
                fill_method = "mode"
            elif "zero" in query or "0" in query:
                fill_value = "0"
            
            if target_column:
                # Check if column has missing values
                missing_count = df[target_column].isna().sum()
                if missing_count > 0:
                    response = f"I'll fill {missing_count} missing values in the '{target_column}' column using {fill_method} method."
                    if fill_method == "value":
                        response = f"I'll fill {missing_count} missing values in the '{target_column}' column with {fill_value}."
                    
                    suggested_operation = {
                        "type": "fill_na",
                        "column": target_column,
                        "method": fill_method
                    }
                    
                    if fill_method == "value":
                        suggested_operation["value"] = fill_value
                else:
                    response = f"The column '{target_column}' doesn't have any missing values."
            else:
                # Find columns with missing values
                missing_cols = []
                for col in df.columns:
                    if df[col].isna().sum() > 0:
                        missing_cols.append(col)
                
                if missing_cols:
                    response = "I found missing values in these columns: " + ", ".join(missing_cols)
                    response += "\nWhich column would you like me to fill?"
                else:
                    response = "I didn't find any missing values in your dataset."
        
        # Process data type conversion query
        elif "convert" in query or "change type" in query:
            # Try to identify which column and target type
            target_column = None
            for col in df.columns:
                if col.lower() in query:
                    target_column = col
                    break
            
            # Determine target data type
            target_type = None
            if "int" in query or "integer" in query:
                target_type = "int"
            elif "float" in query or "decimal" in query or "number" in query:
                target_type = "float"
            elif "date" in query or "time" in query:
                target_type = "datetime"
            elif "string" in query or "text" in query:
                target_type = "string"
            
            if target_column and target_type:
                current_type = df[target_column].dtype
                response = f"I'll convert column '{target_column}' from {current_type} to {target_type} type."
                
                suggested_operation = {
                    "type": "change_type",
                    "column": target_column,
                    "new_type": target_type
                }
            elif target_column:
                current_type = df[target_column].dtype
                response = f"The column '{target_column}' is currently {current_type} type. What type would you like to convert it to?"
            elif target_type:
                response = f"Which column would you like to convert to {target_type} type?"
            else:
                response = "I need to know which column you want to convert and to what data type."
                
        # Process data cleaning recommendations
        elif "clean" in query or "fix" in query or "recommend" in query:
            suggestions = []
            
            # Check for missing values
            for column in df.columns:
                missing_count = df[column].isna().sum()
                if missing_count > 0:
                    missing_pct = missing_count / len(df) * 100
                    if missing_pct < 50:  # Less than 50% missing
                        suggestions.append({
                            "type": "fill_missing",
                            "column": column,
                            "description": f"Fill missing values in column '{column}' ({missing_count} missing values, {missing_pct:.1f}%)"
                        })
                    else:  # More than 50% missing, maybe drop column
                        suggestions.append({
                            "type": "drop_column",
                            "column": column,
                            "description": f"Consider dropping column '{column}' ({missing_count} missing values, {missing_pct:.1f}%)"
                        })
            
            # Check for data type issues
            for column in df.columns:
                # Check for potentially misclassified numeric columns
                if df[column].dtype == object:
                    # Check if column contains mostly numeric values
                    try:
                        numeric_count = pd.to_numeric(df[column], errors='coerce').notna().sum()
                        if numeric_count / len(df) > 0.8:  # More than 80% convertible to numeric
                            suggestions.append({
                                "type": "change_type",
                                "column": column,
                                "description": f"Convert '{column}' to numeric type (currently string but contains mostly numbers)"
                            })
                    except:
                        pass
                        
                # Check for date columns misclassified as strings
                if df[column].dtype == object:
                    date_patterns = ['^\d{4}-\d{2}-\d{2}', '^\d{2}/\d{2}/\d{4}', '^\d{2}-\d{2}-\d{4}']
                    for pattern in date_patterns:
                        date_matches = df[column].astype(str).str.match(pattern, na=False).sum()
                        if date_matches / len(df) > 0.8:  # More than 80% match date pattern
                            suggestions.append({
                                "type": "change_type",
                                "column": column,
                                "description": f"Convert '{column}' to datetime type (currently string but contains date formats)"
                            })
                            break
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                suggestions.append({
                    "type": "remove_duplicates",
                    "description": f"Remove {duplicate_count} duplicate rows"
                })
            
            if suggestions:
                response = "Here are my recommendations for cleaning your data:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"{i}. {suggestion['description']}\n"
                
                insights = {
                    "rowCount": row_count,
                    "columnCount": column_count,
                    "suggestions": suggestions
                }
            else:
                response = "Your data looks pretty clean! I don't see any obvious issues that need fixing."
        
        # Process financial analysis query
        elif "financial" in query or "finance" in query or "income" in query or "balance" in query:
            response = "Financial summary has been disabled as it's not needed at this time."
            insights = None
        
        # Default response for unrecognized queries
        else:
            response = "I'm not sure how to help with that query. You can ask about missing values, statistics, data cleaning recommendations, or specific transformations like converting data types."
        
        return LLMResponse(
            response=response,
            suggested_operation=suggested_operation,
            insights=insights
        )
    
    except Exception as e:
        logger.error(f"Error in AI assistant: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Handle NLP requests for analysis through AI Assistant
@app.post("/analyze-nlp/{dataset_id}")
def analyze_dataset_nlp(dataset_id: str, query: Dict[str, str]):
    try:
        user_query = query.get("query", "").lower()
        
        # Load dataset to get column names
        dataset_path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
        if not os.path.exists(dataset_path):
            return {"error": f"Dataset {dataset_id} not found"}
        
        df = pd.read_csv(dataset_path)
        column_names = df.columns.tolist()
        
        # Extract analysis type from query
        analysis_type = None
        if any(term in user_query for term in ["describe", "summary", "statistics", "stats"]):
            analysis_type = "descriptive"
        elif any(term in user_query for term in ["correlation", "correlate", "relationship"]):
            analysis_type = "correlation"
        elif any(term in user_query for term in ["regression", "predict", "model", "relationship between"]):
            analysis_type = "regression"
        elif any(term in user_query for term in ["compare", "difference", "t-test", "anova", "comparison"]):
            analysis_type = "comparison"
        
        # Prepare analysis request
        analysis_request = {"analysis_type": analysis_type} if analysis_type else None
        
        # Extract column names from query
        if analysis_type == "regression":
            # Look for "predict X using Y, Z" or "regression of X on Y, Z" patterns
            target_column = None
            feature_columns = []
            
            # Check for target column
            predict_patterns = ["predict", "regression of", "model"]
            for pattern in predict_patterns:
                if pattern in user_query:
                    parts = user_query.split(pattern)[1].split("using" if "using" in user_query else "on")
                    if len(parts) > 0:
                        target_candidate = parts[0].strip()
                        # Find the closest matching column name
                        for col in column_names:
                            if col.lower() in target_candidate:
                                target_column = col
                                break
                    
                    # Extract feature columns
                    if len(parts) > 1:
                        features_text = parts[1].strip()
                        for col in column_names:
                            if col.lower() in features_text.lower():
                                feature_columns.append(col)
            
            if target_column and feature_columns:
                analysis_request.update({
                    "target_column": target_column,
                    "feature_columns": feature_columns
                })
        
        elif analysis_type == "comparison":
            # Look for "compare X across Y" or "difference in X between groups in Y"
            value_column = None
            group_column = None
            
            compare_patterns = ["compare", "difference in", "groups in", "across"]
            for pattern in compare_patterns:
                if pattern in user_query:
                    parts = user_query.split(pattern)
                    if len(parts) > 1:
                        # Find columns in the text
                        for col in column_names:
                            if col.lower() in user_query.lower():
                                # First column found is likely the value column
                                if not value_column:
                                    value_column = col
                                # Second column is likely the group column
                                elif not group_column:
                                    group_column = col
                                    break
            
            if value_column and group_column:
                analysis_request.update({
                    "value_column": value_column,
                    "group_column": group_column
                })
        
        if not analysis_request:
            return {"error": "Could not determine analysis type or required parameters from query"}
        
        # Call the analysis endpoint with the constructed request
        return {
            "analysis_request": analysis_request,
            "message": "Analysis request ready to be submitted"
        }
    
    except Exception as e:
        logger.error(f"Error processing NLP query: {str(e)}")
        return {"error": f"Error processing query: {str(e)}"}

@app.post("/api/generate-ai-insights")
async def generate_ai_insights(dataset_id: str, analysis_type: str = "general"):
    try:
        if not openai_client:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "OpenAI client not initialized. Please check your API key."}
            )
            
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
        
        df = datasets[dataset_id]["df"]
        
        # Extract dataset information
        num_rows, num_cols = df.shape
        column_types = {col: str(df[col].dtype) for col in df.columns}
        missing_values = df.isnull().sum().to_dict()
        
        # Get sample data (first 5 rows)
        sample_data = df.head(5).to_dict(orient="records")
        
        # Create a prompt based on the analysis type
        if analysis_type == "general":
            prompt = f"""
            Please analyze this dataset and provide key insights and recommendations:
            
            Dataset Information:
            - Number of rows: {num_rows}
            - Number of columns: {num_cols}
            - Column types: {column_types}
            - Missing values: {missing_values}
            
            Sample data:
            {json.dumps(sample_data, indent=2)}
            
            Please provide:
            1. A summary of the dataset
            2. Key observations about data quality
            3. Recommended data cleaning steps
            4. Potential analysis directions
            """
        elif analysis_type == "cleaning":
            prompt = f"""
            Please recommend data cleaning strategies for this dataset:
            
            Dataset Information:
            - Number of rows: {num_rows}
            - Number of columns: {num_cols}
            - Column types: {column_types}
            - Missing values: {missing_values}
            
            Sample data:
            {json.dumps(sample_data, indent=2)}
            
            Please provide detailed recommendations for:
            1. Handling missing values for each column
            2. Detecting and handling outliers
            3. Appropriate data transformations
            4. Feature engineering opportunities
            """
        else:  # analysis_type == "insights"
            # Calculate basic statistics
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                basic_stats = df[numeric_cols].describe().to_dict()
            else:
                basic_stats = {}
                
            prompt = f"""
            Please provide advanced analytical insights for this dataset:
            
            Dataset Information:
            - Number of rows: {num_rows}
            - Number of columns: {num_cols}
            - Column types: {column_types}
            - Basic statistics: {json.dumps(basic_stats, indent=2)}
            
            Sample data:
            {json.dumps(sample_data, indent=2)}
            
            Please provide:
            1. Key patterns and trends in the data
            2. Statistical significance of observed relationships
            3. Suggested hypothesis tests or modeling approaches
            4. Business or domain-specific insights
            """
        
        # Call OpenAI API with error handling - using chat completions
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4.5-preview",  # Change back to the original model
                messages=[
                    {"role": "system", "content": "You are a helpful data analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the response text
            insights_text = completion.choices[0].message.content
            
            # Return the insights
            return {
                "success": True,
                "insights": insights_text,
                "dataset_id": dataset_id,
                "analysis_type": analysis_type
            }
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Error from OpenAI API: {str(e)}"}
            )
        
    except Exception as e:
        print(f"Error generating AI insights: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error generating AI insights: {str(e)}"}
        )

@app.post("/api/generate-report")
async def generate_report(dataset_id: str, report_type: str, target_column: Optional[str] = None):
    try:
        if dataset_id not in datasets:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
        
        df = datasets[dataset_id]["df"]
        visualizations = []
        
        # Generate base report
        if report_type == "summary":
            # ... existing summary report code ...
            
            # Generate summary visualizations
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                plt.figure(figsize=(10, 6))
                missing_values = df.isnull().sum().sort_values(ascending=False)
                sns.barplot(x=missing_values.index, y=missing_values.values)
                plt.title('Missing Values by Column')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations.append({
                    "title": "Missing Values by Column",
                    "image": fig_to_base64(plt.gcf())
                })
                plt.close()
            
            # Histograms for top numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                    visualizations.append({
                        "title": f"Distribution of {col}",
                        "image": fig_to_base64(plt.gcf())
                    })
                    plt.close()
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 10))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                visualizations.append({
                    "title": "Correlation Heatmap",
                    "image": fig_to_base64(plt.gcf())
                })
                plt.close()
            
            # Pie charts for categorical columns with few unique values
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if df[col].nunique() <= 10:  # Only for columns with 10 or fewer unique values
                    plt.figure(figsize=(10, 6))
                    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                    plt.title(f'Distribution of {col}')
                    plt.ylabel('')
                    plt.tight_layout()
                    visualizations.append({
                        "title": f"Distribution of {col}",
                        "image": fig_to_base64(plt.gcf())
                    })
                    plt.close()
            
            # Generate LLM insights for the summary using chat completions
            ai_insights = ""
            if openai_client:
                try:
                    prompt = f"""
                    Please analyze this dataset summary and provide key insights:
                    
                    Dataset Information:
                    - Number of rows: {df.shape[0]}
                    - Number of columns: {df.shape[1]}
                    - Columns: {', '.join(df.columns.tolist())}
                    
                    Numeric columns statistics:
                    {df.describe().to_string() if not df.select_dtypes(include=['number']).empty else "No numeric columns"}
                    
                    Missing values:
                    {df.isnull().sum().to_string()}
                    
                    Please provide:
                    1. A summary of data quality issues
                    2. Key observations about the distributions
                    3. Potential relationships between variables
                    4. Suggested next steps for analysis
                    """
                    
                    completion = openai_client.chat.completions.create(
                        model="gpt-4.5-preview",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    ai_insights = completion.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API error in summary report: {str(e)}")
                    ai_insights = "Unable to generate AI insights due to an API error."
            else:
                ai_insights = "AI insights not available. Please check your OpenAI API key."
            
            report = {
                "dataset_id": dataset_id,
                "report_type": "summary",
                "timestamp": datetime.now().isoformat(),
                "data_shape": {
                    "rows": df.shape[0],
                    "columns": df.shape[1]
                },
                "column_types": {col: str(df[col].dtype) for col in df.columns},
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {},
                "categorical_summary": {
                    col: df[col].value_counts().to_dict() 
                    for col in df.select_dtypes(include=['object']).columns
                    if df[col].nunique() <= 10
                },
                "visualizations": visualizations,
                "ai_insights": ai_insights
            }
            
        elif report_type == "prediction" and target_column:
            # ... existing prediction report code ...
            
            # Generate prediction visualizations
            # Actual vs Predicted
            if 'y_test' in locals() and 'y_pred' in locals():
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted Values')
                plt.tight_layout()
                visualizations.append({
                    "title": "Actual vs Predicted Values",
                    "image": fig_to_base64(plt.gcf())
                })
                plt.close()
                
                # Residuals plot
                plt.figure(figsize=(10, 6))
                residuals = y_test - y_pred
                plt.scatter(y_pred, residuals)
                plt.axhline(y=0, color='k', linestyle='--')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title('Residuals Plot')
                plt.tight_layout()
                visualizations.append({
                    "title": "Residuals Plot",
                    "image": fig_to_base64(plt.gcf())
                })
                plt.close()
                
                # Feature importance
                if 'model' in locals() and hasattr(model, 'coef_'):
                    feature_importance = pd.Series(model.coef_, index=X.columns)
                    plt.figure(figsize=(12, 8))
                    feature_importance.abs().sort_values(ascending=False).plot(kind='bar')
                    plt.title('Feature Importance')
                    plt.tight_layout()
                    visualizations.append({
                        "title": "Feature Importance",
                        "image": fig_to_base64(plt.gcf())
                    })
                    plt.close()
            
            # Generate LLM insights for the prediction model using chat completions
            ai_insights = ""
            if 'model_metrics' in locals() and openai_client:
                try:
                    prompt = f"""
                    Please analyze this prediction model results and provide insights:
                    
                    Model Information:
                    - Target Variable: {target_column}
                    - Model Type: {model.__class__.__name__}
                    - R² Score: {model_metrics.get('r2_score', 'N/A')}
                    - RMSE: {model_metrics.get('rmse', 'N/A')}
                    
                    Feature Importance:
                    {feature_importance.abs().sort_values(ascending=False).to_string() if 'feature_importance' in locals() else "Not available"}
                    
                    Please provide:
                    1. An interpretation of the model's performance
                    2. Analysis of the most important features
                    3. Suggestions for model improvement
                    4. Practical applications of this model
                    """
                    
                    completion = openai_client.chat.completions.create(
                        model="gpt-4.5-preview",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    
                    ai_insights = completion.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API error in prediction report: {str(e)}")
                    ai_insights = "Unable to generate AI insights due to an API error."
            else:
                ai_insights = "AI insights not available. Please check your OpenAI API key or model metrics."
            
            report = {
                "dataset_id": dataset_id,
                "report_type": "prediction",
                "target_column": target_column,
                "timestamp": datetime.now().isoformat(),
                "model_type": model.__class__.__name__,
                "metrics": model_metrics,
                "visualizations": visualizations,
                "ai_insights": ai_insights
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid report type or missing target column")
        
        return {"success": True, "report": report}
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error generating report: {str(e)}"}
        )

# Helper function to convert matplotlib figure to base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
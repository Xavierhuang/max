import os
import openai
import pandas as pd
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# If no API key is provided, we'll run in a limited mode
HAS_API_KEY = bool(openai.api_key)

class LLMAssistant:
    """Class that handles all interactions with the LLM"""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.has_api_key = HAS_API_KEY
    
    def detect_data_issues(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """
        Detect potential data quality issues using LLM
        """
        if not self.has_api_key:
            return self._fallback_detect_data_issues(df)
            
        # Sample the dataframe to avoid sending too much data
        sample_df = df.head(sample_size)
        
        # Convert the sample to JSON for the prompt
        sample_json = sample_df.to_json(orient="records", date_format="iso")
        
        # Calculate basic statistics
        stats = {}
        for col in df.columns:
            stats[col] = {
                "dtype": str(df[col].dtype),
                "missing_count": df[col].isna().sum(),
                "missing_percentage": round(df[col].isna().mean() * 100, 2),
                "unique_count": df[col].nunique()
            }
            
            # Add numeric stats where applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col].update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                })
        
        # Create the prompt
        prompt = f"""
        You are a data scientist analyzing a dataset. Below is a sample of the data and some statistics.
        
        Data sample (first {sample_size} rows):
        {sample_json}
        
        Column statistics:
        {json.dumps(stats, indent=2)}
        
        Please analyze this data and provide the following in JSON format:
        1. Data quality issues detected (missing values, outliers, inconsistent formats)
        2. Column type suggestions (if a column seems to be the wrong type)
        3. Transformation recommendations (specific cleaning operations that would improve the data)
        4. General observations about the dataset
        
        Return only a valid JSON with these keys: "issues", "type_suggestions", "transformations", "observations".
        Each should be a list of objects with clear descriptions of the issues and suggestions.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality assistant that analyzes datasets and provides recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            print(f"Error using LLM to detect data issues: {str(e)}")
            return self._fallback_detect_data_issues(df)
    
    def _fallback_detect_data_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback method for when the API key is not available or there's an error
        Uses basic pandas operations to detect common issues
        """
        issues = []
        type_suggestions = []
        transformations = []
        observations = []
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_percent = round((missing_count / len(df)) * 100, 2)
                issues.append({
                    "column": col,
                    "issue": "missing_values",
                    "description": f"Column has {missing_count} missing values ({missing_percent}%)"
                })
                
                if missing_percent < 30:
                    transformations.append({
                        "column": col,
                        "operation": "fill_na",
                        "description": f"Fill missing values in {col}",
                        "method": "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
                    })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                "issue": "duplicates",
                "description": f"Dataset has {duplicate_count} duplicate rows"
            })
            transformations.append({
                "operation": "remove_duplicates",
                "description": "Remove duplicate rows"
            })
        
        # Check for potential type issues
        for col in df.columns:
            # Check if column has string values that could be dates
            if df[col].dtype == 'object':
                # Check for email-like columns
                if any(str(x).count('@') == 1 for x in df[col].dropna().head(10)):
                    observations.append({
                        "column": col,
                        "observation": "This column appears to contain email addresses"
                    })
                
                # Check if numeric column is stored as string
                try:
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    if numeric_conversion.notna().mean() > 0.8:  # More than 80% can be converted to numbers
                        type_suggestions.append({
                            "column": col,
                            "current_type": str(df[col].dtype),
                            "suggested_type": "float" if any('.' in str(x) for x in df[col].dropna().head(10)) else "int",
                            "description": "Column contains mostly numeric values but is stored as string"
                        })
                except:
                    pass
                
                # Check if date-like columns
                try:
                    date_conversion = pd.to_datetime(df[col], errors='coerce')
                    if date_conversion.notna().mean() > 0.8:  # More than 80% can be converted to dates
                        type_suggestions.append({
                            "column": col,
                            "current_type": str(df[col].dtype),
                            "suggested_type": "datetime",
                            "description": "Column contains date-like values but is stored as string"
                        })
                except:
                    pass
        
        # General observations
        observations.append({
            "observation": f"Dataset has {len(df)} rows and {len(df.columns)} columns"
        })
        
        if df.shape[0] > 1000:
            observations.append({
                "observation": "This is a relatively large dataset, consider sampling for performance"
            })
        
        return {
            "issues": issues,
            "type_suggestions": type_suggestions,
            "transformations": transformations,
            "observations": observations
        }
    
    def parse_natural_language_transformation(self, query: str, df_columns: List[str]) -> Dict[str, Any]:
        """
        Parse a natural language transformation request into a structured operation
        """
        if not self.has_api_key:
            return self._fallback_parse_transformation(query, df_columns)
            
        column_list = ", ".join(df_columns)
        
        prompt = f"""
        You are an assistant that translates natural language data transformation requests into structured operations.
        
        Available columns in the dataset: {column_list}
        
        User request: "{query}"
        
        Translate this request into a specific data transformation operation from the following types:
        - drop_columns: Remove columns from the dataset
        - rename_column: Rename a column
        - fill_na: Fill missing values in a column
        - change_type: Change the data type of a column
        - filter_rows: Filter rows based on a condition
        - remove_duplicates: Remove duplicate rows
        
        Return a JSON object with the following structure:
        {{
            "operation_type": "one of the types listed above",
            "parameters": {{
                // parameters specific to the operation type
                // e.g., "columns" for drop_columns, "column" and "method" for fill_na, etc.
            }}
        }}
        
        For example, if the request is "remove the age column", you should return:
        {{
            "operation_type": "drop_columns",
            "parameters": {{
                "columns": ["age"]
            }}
        }}
        
        Only return valid JSON. If you cannot match the request to any operation, return:
        {{ "operation_type": "unknown", "error": "explanation of why it's not recognized" }}
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data transformation assistant. Keep your responses concise and in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            print(f"Error using LLM to parse transformation: {str(e)}")
            return self._fallback_parse_transformation(query, df_columns)
    
    def _fallback_parse_transformation(self, query: str, df_columns: List[str]) -> Dict[str, Any]:
        """
        Fallback method for when the API key is not available
        Uses simple keyword matching to try to interpret the command
        """
        query = query.lower()
        
        # Look for drop columns operation
        if any(x in query for x in ["drop", "remove", "delete"]) and any(x in query for x in ["column", "field"]):
            # Try to find mentioned columns
            mentioned_columns = [col for col in df_columns if col.lower() in query]
            if mentioned_columns:
                return {
                    "operation_type": "drop_columns",
                    "parameters": {
                        "columns": mentioned_columns
                    }
                }
        
        # Look for rename column operation
        if any(x in query for x in ["rename", "change name", "name"]) and any(x in query for x in ["column", "field"]):
            for col in df_columns:
                if col.lower() in query:
                    # Very simplistic - just tries to find "to X" pattern
                    to_parts = query.split(" to ")
                    if len(to_parts) > 1:
                        new_name_part = to_parts[1].strip().split()
                        if new_name_part:
                            new_name = new_name_part[0].strip(",.;:")
                            return {
                                "operation_type": "rename_column",
                                "parameters": {
                                    "old_name": col,
                                    "new_name": new_name
                                }
                            }
        
        # Look for fill missing values
        if any(x in query for x in ["fill", "replace", "missing", "null", "na"]):
            for col in df_columns:
                if col.lower() in query:
                    method = "value"
                    if "mean" in query:
                        method = "mean"
                    elif "median" in query:
                        method = "median"
                    elif "mode" in query:
                        method = "mode"
                    
                    return {
                        "operation_type": "fill_na",
                        "parameters": {
                            "column": col,
                            "method": method,
                            "value": "0" if method == "value" else None
                        }
                    }
        
        # Default response if no match
        return {
            "operation_type": "unknown",
            "error": "Unable to parse the request. Please try again with a clearer instruction."
        }
    
    def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive profile of the dataset
        """
        if not self.has_api_key:
            return self._fallback_data_profile(df)
            
        # Prepare a summary of the data
        sample_size = min(100, len(df))
        sample_df = df.head(sample_size)
        
        # Basic statistics
        stats = {}
        for col in df.columns:
            stats[col] = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_percentage": round(df[col].isna().mean() * 100, 2),
                "unique_count": int(df[col].nunique())
            }
            
            # Add type-specific stats
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col].update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                })
                
                # Add percentiles
                try:
                    stats[col]["percentiles"] = {
                        "25%": float(df[col].quantile(0.25)),
                        "50%": float(df[col].quantile(0.5)),
                        "75%": float(df[col].quantile(0.75))
                    }
                except:
                    pass
        
        # Create correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlations = {}
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(2)
            for col1 in numeric_cols:
                correlations[col1] = {}
                for col2 in numeric_cols:
                    if col1 != col2:
                        correlations[col1][col2] = float(corr_matrix.loc[col1, col2])
        
        # Prepare for the LLM
        prompt = f"""
        You are a data scientist analyzing a dataset. Here are some statistics about the dataset:
        
        Basic stats:
        {json.dumps(stats, indent=2)}
        
        Correlations between numeric columns:
        {json.dumps(correlations, indent=2)}
        
        Based on this information, generate a comprehensive data profile with the following sections:
        1. Overview of the dataset
        2. Data quality assessment
        3. Distribution of key variables
        4. Relationships between variables
        5. Recommendations for data preparation
        
        Return the result as a JSON object with these sections as keys.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data profiling assistant that creates comprehensive statistical summaries of datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error generating data profile with LLM: {str(e)}")
            return self._fallback_data_profile(df)
    
    def _fallback_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback method for generating a data profile without using an LLM
        """
        profile = {
            "overview": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB"
            },
            "data_quality": {
                "missing_values": {},
                "duplicate_rows": int(df.duplicated().sum())
            },
            "distributions": {},
            "relationships": {},
            "recommendations": []
        }
        
        # Data quality: missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                profile["data_quality"]["missing_values"][col] = {
                    "count": int(missing),
                    "percentage": f"{(missing / len(df)) * 100:.2f}%"
                }
        
        # Distributions for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            profile["distributions"][col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
            }
        
        # Relationships: correlations between numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().abs()
            
            # Find pairs with high correlation
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr.iloc[i, j]) > 0.7:  # Threshold for "high" correlation
                        high_corr_pairs.append({
                            "column1": numeric_cols[i],
                            "column2": numeric_cols[j],
                            "correlation": round(float(corr.iloc[i, j]), 2)
                        })
            
            profile["relationships"]["high_correlations"] = high_corr_pairs
        
        # Generate some basic recommendations
        if profile["data_quality"]["duplicate_rows"] > 0:
            profile["recommendations"].append({
                "type": "remove_duplicates",
                "description": f"Remove {profile['data_quality']['duplicate_rows']} duplicate rows"
            })
        
        for col, missing in profile["data_quality"].get("missing_values", {}).items():
            if float(missing["percentage"].strip("%")) < 30:
                profile["recommendations"].append({
                    "type": "handle_missing",
                    "column": col,
                    "description": f"Fill missing values in {col}"
                })
            elif float(missing["percentage"].strip("%")) > 80:
                profile["recommendations"].append({
                    "type": "drop_column",
                    "column": col,
                    "description": f"Consider dropping {col} as it has more than 80% missing values"
                })
        
        return profile

# Initialize the LLM assistant
llm_assistant = LLMAssistant() 
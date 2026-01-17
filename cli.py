import os
import sys
import time
import json
import pandas as pd
import numpy as np
import warnings
import questionary
import logging
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.syntax import Syntax
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats

# ======================== SETUP ========================
warnings.filterwarnings('ignore')
load_dotenv()
console = Console()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== CSV TYPE DETECTOR ========================
class CSVTypeDetector:
    """Automatically detects CSV type and adapts agent strategies"""
    
    DOMAIN_PATTERNS = {
        "water": ["water", "groundwater", "aquifer", "precipitation", "rainfall", "well", "borehole", "depth"],
        "health": ["disease", "patient", "hospital", "medical", "diagnosis", "symptom", "treatment", "age"],
        "agriculture": ["crop", "yield", "soil", "fertilizer", "pesticide", "harvest", "farm", "irrigation"],
        "finance": ["price", "revenue", "cost", "profit", "transaction", "payment", "loan", "interest"],
        "education": ["student", "grade", "score", "test", "exam", "course", "enrollment", "gpa"],
        "census": ["population", "age", "gender", "income", "education", "employment", "housing"],
        "sales": ["product", "quantity", "revenue", "customer", "region", "category", "discount", "sales"],
        "energy": ["consumption", "generation", "renewable", "coal", "solar", "wind", "grid", "kwh"]
    }
    
    @staticmethod
    def detect_domain(df):
        """Detect CSV domain based on column names and content"""
        cols_lower = " ".join(df.columns).lower()
        
        scores = {}
        for domain, keywords in CSVTypeDetector.DOMAIN_PATTERNS.items():
            scores[domain] = sum(1 for kw in keywords if kw in cols_lower)
        
        top_domain = max(scores, key=scores.get)
        confidence = scores[top_domain] / len(CSVTypeDetector.DOMAIN_PATTERNS[top_domain]) * 100
        
        return top_domain, confidence
    
    @staticmethod
    def get_adaptive_strategy(domain, df):
        """Get domain-specific data cleaning strategy"""
        strategies = {
            "water": {
                "priority_features": ["water_quality", "quantity", "depth", "temperature"],
                "outlier_threshold": 2.5,
                "missing_strategy": "median",  # Continuous data
                "categorical_encoding": "label"
            },
            "health": {
                "priority_features": ["diagnosis", "age", "symptoms"],
                "outlier_threshold": 3.0,
                "missing_strategy": "mode",  # Categorical preferred
                "categorical_encoding": "one_hot",
                "privacy_alert": True
            },
            "agriculture": {
                "priority_features": ["yield", "crop_type", "soil_type"],
                "outlier_threshold": 2.5,
                "missing_strategy": "median",
                "categorical_encoding": "label"
            },
            "finance": {
                "priority_features": ["amount", "transaction_type", "account"],
                "outlier_threshold": 3.0,
                "missing_strategy": "mean",
                "categorical_encoding": "one_hot",
                "privacy_alert": True
            },
            "education": {
                "priority_features": ["student_id", "score", "course"],
                "outlier_threshold": 2.5,
                "missing_strategy": "median",
                "categorical_encoding": "label"
            },
            "census": {
                "priority_features": ["age", "gender", "income"],
                "outlier_threshold": 3.0,
                "missing_strategy": "mode",
                "categorical_encoding": "label",
                "privacy_alert": True
            },
            "sales": {
                "priority_features": ["product", "quantity", "revenue"],
                "outlier_threshold": 2.5,
                "missing_strategy": "mean",
                "categorical_encoding": "one_hot"
            },
            "energy": {
                "priority_features": ["consumption", "generation", "source"],
                "outlier_threshold": 2.5,
                "missing_strategy": "median",
                "categorical_encoding": "label"
            },
            "generic": {
                "priority_features": [],
                "outlier_threshold": 2.5,
                "missing_strategy": "median",
                "categorical_encoding": "label"
            }
        }
        return strategies.get(domain, strategies["generic"])

# ======================== AGENT: ARCHITECT (Planner) ========================
class Architect:
    """
    Senior Data Scientist who analyzes dataset metadata and creates comprehensive 
    data preparation strategies using OpenRouter API.
    Works with ANY CSV type through adaptive strategies.
    """
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.df = None
        self.filename = ""
        self.history = []
        self.correlation_matrix = None
        self.csv_domain = None
        self.csv_strategy = None

    def assess_dataset(self, df, filename="dataset"):
        """Perform comprehensive dataset analysis with domain detection"""
        self.df = df
        self.filename = filename
        
        # Detect CSV domain and get adaptive strategy
        self.csv_domain, domain_confidence = CSVTypeDetector.detect_domain(df)
        self.csv_strategy = CSVTypeDetector.get_adaptive_strategy(self.csv_domain, df)
        
        console.print(f"[cyan]📍 Detected Domain: {self.csv_domain.upper()} (confidence: {domain_confidence:.0f}%)[/cyan]")
        
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "numeric_cols": df.select_dtypes(include=np.number).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
            "statistics": {},
            "correlation_issues": [],
            "csv_domain": self.csv_domain,
            "domain_confidence": domain_confidence,
            "adaptive_strategy": self.csv_strategy,
        }

        # Detailed statistics for numeric columns
        for col in analysis["numeric_cols"]:
            if not df[col].isnull().all():
                analysis["statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                }

        # Correlation analysis
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) > 1:
            self.correlation_matrix = numeric_df.corr().abs()
            upper = self.correlation_matrix.where(
                np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
            )
            for column in upper.columns:
                correlated = upper[column][upper[column] > 0.90]
                if len(correlated) > 0:
                    analysis["correlation_issues"].append({
                        "column": column,
                        "correlated_with": correlated.index.tolist(),
                        "correlation_values": correlated.values.tolist()
                    })

        logger.info(f"Dataset assessment complete: {analysis['shape']} | Domain: {self.csv_domain}")
        return analysis

    def create_plan(self, analysis, user_request="Prepare for ML"):
        """Generate comprehensive cleaning plan via OpenRouter API (domain-aware)"""
        
        # Build context
        context = self._build_context(analysis)
        
        # Add domain-specific instructions to system prompt
        domain_instructions = f"""
DOMAIN-SPECIFIC GUIDANCE:
- Detected Domain: {self.csv_domain.upper()}
- Priority Features: {', '.join(self.csv_strategy.get('priority_features', []))}
- Recommended Outlier Threshold: {self.csv_strategy.get('outlier_threshold')}
- Missing Value Strategy: {self.csv_strategy.get('missing_strategy')}
- Categorical Encoding: {self.csv_strategy.get('categorical_encoding')}
- Privacy Alert: {'YES - be very careful with PII' if self.csv_strategy.get('privacy_alert') else 'No special privacy concerns'}
"""
        
        system_prompt = """You are an Expert Data Scientist and ML Strategy Architect.
        
CRITICAL RULES:
1. Analyze the provided dataset metadata thoroughly
2. Identify ALL data quality issues relevant to ML readiness
3. Create logically sequenced transformation steps
4. Provide clear reasoning with confidence scores (0-100)
5. Consider downstream ML implications
6. Never recommend dropping the target column
7. If a column is already clean (scaled, no nulls, properly encoded), return action: "None"
8. Check if Mean≈0 and Std≈1 means column is ALREADY standardized
9. Return ONLY valid JSON with no additional text

""" + domain_instructions + """

Return a JSON object with this exact structure:
{
  "user_intent_summary": "string",
  "dataset_assessment": {
    "total_rows": number,
    "total_columns": number,
    "csv_domain": "string",
    "ml_readiness_score": number (0-100),
    "critical_issues": ["issue1", "issue2"],
    "target_column_identified": "string or null"
  },
  "recommended_operations": [
    {
      "step": number,
      "operation": "string (drop|impute_median|impute_mean|impute_mode|log_transform|standard_scale|minmax_scale|cap_outliers|one_hot|label_encode|remove_duplicates)",
      "target_columns": ["col1", "col2"],
      "method": "string",
      "reasoning": "string",
      "confidence": number (0-100),
      "estimated_impact": "string",
      "safety_notes": "string",
      "is_destructive": boolean
    }
  ],
  "ml_readiness_prediction": number (0-100),
  "next_steps": "string"
}"""

        try:
            with console.status("[bold magenta]🧠 Architect analyzing dataset...[/bold magenta]", spinner="dots"):
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://github.com",
                        "X-Title": "AI-Data-Engineering-System"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"User Request: {user_request}\n\n{context}"}
                        ],
                        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),
                        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 2000)),
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    logger.error(f"API Error: {response.status_code} - {response.text}")
                    console.print(f"[red]❌ API Error: {response.status_code}[/red]")
                    return None
                
                response_data = response.json()
                plan = json.loads(response_data['choices'][0]['message']['content'])
            
            # Filter "None" actions and normalize operations
            normalized_ops = []
            for op in plan.get("recommended_operations", []):
                if op.get("operation", "").lower() not in ["none", "null", "skip"]:
                    # Normalize operation names
                    op_name = op.get("operation", "").lower()
                    # Map common variations to standard names
                    if "impute" in op_name and "mean" in op_name:
                        op["operation"] = "impute_mean"
                        op["method"] = "mean"
                    elif "impute" in op_name and "median" in op_name:
                        op["operation"] = "impute_median"
                        op["method"] = "median"
                    elif "impute" in op_name and ("mode" in op_name or "frequent" in op_name):
                        op["operation"] = "impute_mode"
                        op["method"] = "most_frequent"
                    # Ensure required fields
                    if "target_columns" not in op:
                        op["target_columns"] = op.get("columns", [])
                    normalized_ops.append(op)
            
            plan["recommended_operations"] = normalized_ops
            
            logger.info(f"Plan created with {len(plan['recommended_operations'])} operations for {self.csv_domain} domain")
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            console.print(f"[red]❌ Architect failed: {e}[/red]")
            return None

    def _build_context(self, analysis):
        """Build detailed context for LLM (domain-aware)"""
        buffer = [
            f"📊 Dataset: {self.filename}",
            f"📍 Domain: {analysis.get('csv_domain', 'generic').upper()}",
            f"📏 Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns",
            f"🔄 Duplicates: {analysis['duplicates']}",
            "",
            "📋 COLUMN ANALYSIS:",
        ]

        for col in self.df.columns:
            dtype = analysis["dtypes"].get(col, "unknown")
            missing = analysis["missing_values"].get(col, 0)
            missing_pct = analysis["missing_percentage"].get(col, 0)
            
            if col in analysis["numeric_cols"]:
                stats_info = analysis["statistics"].get(col, {})
                buffer.append(
                    f"  • {col} (Type: {dtype}, Missing: {missing} [{missing_pct:.1f}%], "
                    f"Mean: {stats_info.get('mean', 0):.2f}, Std: {stats_info.get('std', 0):.2f})"
                )
            else:
                unique = self.df[col].nunique()
                samples = self.df[col].dropna().unique()[:3]
                buffer.append(
                    f"  • {col} (Type: {dtype}, Missing: {missing} [{missing_pct:.1f}%], "
                    f"Unique: {unique}, Samples: {samples})"
                )

        if analysis["correlation_issues"]:
            buffer.append("\n⚠️ CORRELATION ALERTS (Multicollinearity):")
            for issue in analysis["correlation_issues"]:
                buffer.append(f"  • {issue['column']} correlates with {issue['correlated_with']}")

        return "\n".join(buffer)


# ======================== AGENT: ENGINEER (Coder) ========================
class Engineer:
    """
    Expert ML Engineer who translates strategic plans into safe, efficient Python code.
    """
    def __init__(self):
        self.execution_log = []
        self.df_before = None
        self.df_after = None

    def execute_plan(self, df, plan):
        """Execute all operations from the plan"""
        self.df_before = df.copy()
        df_working = df.copy()
        
        execution_results = {
            "status": "success",
            "operations_executed": [],
            "errors": [],
            "before_shape": df.shape,
            "after_shape": None,
        }

        for i, operation in enumerate(plan.get("recommended_operations", [])):
            try:
                result = self._execute_operation(df_working, operation)
                if result["success"]:
                    df_working = result.get("df", df_working)
                    execution_results["operations_executed"].append({
                        "step": i + 1,
                        "operation": operation.get("operation"),
                        "target_columns": operation.get("target_columns"),
                        "result": result.get("message"),
                        "rows_affected": result.get("rows_affected", 0),
                        "success": True,
                    })
                    console.print(f"[green]✔ Step {i + 1}: {result['message']}[/green]")
                else:
                    execution_results["errors"].append({
                        "step": i + 1,
                        "operation": operation.get("operation"),
                        "error": result.get("error"),
                    })
                    console.print(f"[yellow]⚠ Step {i + 1}: {result['error']}[/yellow]")
            except Exception as e:
                logger.error(f"Operation {i+1} failed: {e}")
                execution_results["errors"].append({
                    "step": i + 1,
                    "operation": operation.get("operation"),
                    "error": str(e),
                })
                console.print(f"[red]✖ Step {i + 1} failed: {e}[/red]")

        self.df_after = df_working
        execution_results["after_shape"] = df_working.shape
        execution_results["null_count_before"] = df.isnull().sum().sum()
        execution_results["null_count_after"] = df_working.isnull().sum().sum()
        
        logger.info(f"Execution complete: {len(execution_results['operations_executed'])} operations")
        return df_working, execution_results

    def _execute_operation(self, df, operation):
        """Execute individual operation"""
        op_type = operation.get("operation")
        # Handle both "target_columns" and "columns" field names
        cols = operation.get("target_columns") or operation.get("columns") or []
        method = operation.get("method", "")

        try:
            if op_type == "drop":
                cols_to_drop = [c for c in cols if c in df.columns]
                df.drop(columns=cols_to_drop, inplace=True)
                return {
                    "success": True,
                    "df": df,
                    "message": f"Dropped {len(cols_to_drop)} columns",
                    "rows_affected": 0,
                }

            elif op_type == "remove_duplicates":
                before = len(df)
                df.drop_duplicates(inplace=True)
                removed = before - len(df)
                return {
                    "success": True,
                    "df": df,
                    "message": f"Removed {removed} duplicate rows",
                    "rows_affected": removed,
                }

            elif op_type in ["impute_median", "impute_mean", "impute_mode"]:
                valid_cols = [c for c in cols if c in df.columns]
                if not valid_cols:
                    return {"success": False, "error": f"No valid columns found: {cols}"}
                
                # Extract strategy correctly
                if op_type == "impute_mean":
                    strategy = "mean"
                elif op_type == "impute_median":
                    strategy = "median"
                elif op_type == "impute_mode":
                    strategy = "most_frequent"
                else:
                    strategy = method.lower() if method else "mean"
                    # Clean up strategy names
                    strategy = strategy.replace("imputation", "").replace("_", "").strip()
                    if "mean" in strategy:
                        strategy = "mean"
                    elif "median" in strategy:
                        strategy = "median"
                    elif "mode" in strategy or "frequent" in strategy:
                        strategy = "most_frequent"
                    else:
                        strategy = "mean"  # Default to mean
                
                # Separate numeric and non-numeric columns
                numeric_cols = []
                non_numeric_cols = []
                for col in valid_cols:
                    try:
                        # Try to convert to numeric
                        pd.to_numeric(df[col], errors='raise')
                        numeric_cols.append(col)
                    except (ValueError, TypeError):
                        # Non-numeric column - use label encoding instead
                        non_numeric_cols.append(col)
                
                # Impute numeric columns
                if numeric_cols:
                    imputer = SimpleImputer(strategy=strategy)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                
                # Label encode non-numeric columns with nulls
                for col in non_numeric_cols:
                    if df[col].isnull().any():
                        le = LabelEncoder()
                        # Fill nulls with "MISSING" first
                        df[col] = df[col].fillna("MISSING")
                        df[col] = le.fit_transform(df[col].astype(str))
                
                total_cols_processed = len(numeric_cols) + len(non_numeric_cols)
                return {
                    "success": True,
                    "df": df,
                    "message": f"Imputed {len(numeric_cols)} numeric + encoded {len(non_numeric_cols)} categorical",
                    "rows_affected": total_cols_processed,
                }

            elif op_type == "log_transform":
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if (df[col] <= 0).any():
                            df[col] += abs(df[col].min()) + 1
                        df[col] = np.log(df[col])
                return {
                    "success": True,
                    "df": df,
                    "message": f"Applied log transformation to {len(cols)} columns",
                    "rows_affected": len(cols),
                }

            elif op_type == "cap_outliers":
                for col in cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df[col] = np.where(df[col] < lower, lower, df[col])
                        df[col] = np.where(df[col] > upper, upper, df[col])
                return {
                    "success": True,
                    "df": df,
                    "message": f"Capped outliers in {len(cols)} columns",
                    "rows_affected": len(cols),
                }

            elif op_type == "standard_scale":
                valid_cols = [c for c in cols if c in df.columns]
                if not valid_cols:
                    return {"success": False, "error": f"No valid numeric columns: {cols}"}
                scaler = StandardScaler()
                df[valid_cols] = scaler.fit_transform(df[valid_cols])
                return {
                    "success": True,
                    "df": df,
                    "message": f"Standard scaled {len(valid_cols)} columns (mean≈0, std≈1)",
                    "rows_affected": len(valid_cols),
                }

            elif op_type == "minmax_scale":
                valid_cols = [c for c in cols if c in df.columns]
                if not valid_cols:
                    return {"success": False, "error": f"No valid numeric columns: {cols}"}
                scaler = MinMaxScaler()
                df[valid_cols] = scaler.fit_transform(df[valid_cols])
                return {
                    "success": True,
                    "df": df,
                    "message": f"MinMax scaled {len(valid_cols)} columns (0-1 range)",
                    "rows_affected": len(valid_cols),
                }

            elif op_type == "one_hot":
                # Check if one-hot would create too many columns
                # For large datasets with high cardinality, use label encoding instead
                safe_cols = []
                label_cols = []
                
                for col in cols:
                    if col in df.columns:
                        unique_count = df[col].nunique()
                        # If more than 20 unique values, use label encoding to avoid memory issues
                        if unique_count > 20:
                            label_cols.append(col)
                        else:
                            safe_cols.append(col)
                
                # One-hot encode low-cardinality columns
                if safe_cols:
                    dummies = pd.get_dummies(df[safe_cols], prefix=safe_cols, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=safe_cols, inplace=True)
                
                # Label encode high-cardinality columns
                for col in label_cols:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                
                return {
                    "success": True,
                    "df": df,
                    "message": f"Encoded {len(safe_cols)} cols (one-hot) + {len(label_cols)} cols (label)",
                    "rows_affected": len(cols),
                }

            elif op_type == "label_encode":
                for col in cols:
                    if col in df.columns:
                        le = LabelEncoder()
                        df[col] = df[col].astype(str)
                        df[col] = le.fit_transform(df[col])
                return {
                    "success": True,
                    "df": df,
                    "message": f"Label encoded {len(cols)} categorical columns",
                    "rows_affected": len(cols),
                }

            else:
                return {"success": False, "error": f"Unknown operation: {op_type}"}

        except Exception as e:
            logger.error(f"Operation {op_type} failed: {e}")
            return {"success": False, "error": str(e)}


# ======================== AGENT: OBSERVER (Reviewer) ========================
class Observer:
    """
    Data Quality Auditor who validates transformations and communicates results.
    """
    def __init__(self):
        self.confidence = 0

    def validate_execution(self, df_before, df_after, execution_results, plan):
        """Validate that transformations succeeded"""
        validation = {
            "overall_success": True,
            "checks": [],
            "ml_readiness_score": 50,
            "issues_remaining": [],
        }

        # Check 1: No data loss (unless intentional)
        destructive_ops = [
            op for op in plan.get("recommended_operations", [])
            if op.get("is_destructive")
        ]
        if len(df_after) == 0:
            validation["checks"].append({"check": "Data integrity", "status": False, "message": "Dataset is empty!"})
            validation["overall_success"] = False
        else:
            validation["checks"].append({"check": "Data integrity", "status": True, "message": "Data preserved"})

        # Check 2: Target column present
        if "target_column" in plan.get("dataset_assessment", {}):
            target = plan["dataset_assessment"].get("target_column_identified")
            if target and target in df_after.columns:
                validation["checks"].append({"check": "Target column", "status": True, "message": f"'{target}' present"})
            elif target:
                validation["checks"].append({"check": "Target column", "status": False, "message": f"'{target}' missing!"})
                validation["overall_success"] = False

        # Check 3: Missing values
        null_before = df_before.isnull().sum().sum()
        null_after = df_after.isnull().sum().sum()
        null_reduction = null_before - null_after
        validation["checks"].append({
            "check": "Missing values",
            "status": null_after == 0 or null_reduction > 0,
            "message": f"Reduced from {null_before} to {null_after} nulls"
        })

        # Check 4: Operations executed
        executed = len(execution_results.get("operations_executed", []))
        errors = len(execution_results.get("errors", []))
        validation["checks"].append({
            "check": "Execution",
            "status": errors == 0,
            "message": f"{executed} operations executed, {errors} errors"
        })

        # Calculate ML readiness
        base_score = plan.get("ml_readiness_prediction", 50)
        if null_after == 0:
            base_score += 20
        if errors == 0:
            base_score += 15
        else:
            base_score -= 15
        validation["ml_readiness_score"] = min(100, max(0, base_score))

        # Calculate confidence
        self.confidence = 85 if validation["overall_success"] else 45

        return validation

    def generate_report(self, execution_results, validation, plan):
        """Generate user-friendly report"""
        report = {
            "summary": "TRANSFORMATION COMPLETE",
            "operations_executed": execution_results.get("operations_executed", []),
            "before": execution_results.get("before_shape"),
            "after": execution_results.get("after_shape"),
            "ml_readiness_score": validation.get("ml_readiness_score", 0),
            "confidence": self.confidence,
            "user_message": self._build_user_message(execution_results, validation),
        }
        return report

    def _build_user_message(self, execution_results, validation):
        """Build friendly user message"""
        msg = []
        msg.append("🎯 TRANSFORMATION SUMMARY\n")
        
        # What I did
        msg.append("What I Did:")
        for op in execution_results.get("operations_executed", []):
            msg.append(f"  {op['step']}. {op['result']}")

        # Impact
        before_shape = execution_results.get("before_shape")
        after_shape = execution_results.get("after_shape")
        msg.append(f"\nDataset Impact:")
        msg.append(f"  • Rows: {before_shape[0]} → {after_shape[0]}")
        msg.append(f"  • Columns: {before_shape[1]} → {after_shape[1]}")
        msg.append(f"  • Null values: {execution_results['null_count_before']} → {execution_results['null_count_after']}")

        # ML Readiness
        score = validation.get("ml_readiness_score", 0)
        status = "✓ READY" if score >= 95 else "⚠ NEEDS WORK"
        msg.append(f"\n📊 ML Readiness: {score}/100 {status}")

        # Confidence
        msg.append(f"\nConfidence: {self.confidence}%")

        return "\n".join(msg)


# ======================== ON-DEMAND AGENTS (ARCHITECTURE & VISUALIZER) ========================
class OnDemandArchitecture:
    """On-Demand Architecture Agent via API"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandArchitecture._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandArchitecture._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def analyze_architecture(self, df, domain):
        """Call on-demand architecture analysis API"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare data payload with proper JSON serialization
            sample_data = df.head(5).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            payload = {
                "dataset": {
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "domain": domain,
                    "rows_sample": sample_data,
                    "missing_count": int(df.isnull().sum().sum())
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        # Fetch results using execution ID
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df, domain)
                    
                    # Extract architecture data - handle different response formats
                    architecture = result.get("architecture") or result.get("data") or result
                    recommendations = result.get("recommendations") or result.get("insights") or []
                    
                    # If we got valid data, return it
                    if architecture or recommendations:
                        return {
                            "success": True,
                            "architecture": architecture if isinstance(architecture, dict) else {"analysis": str(architecture)},
                            "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default analysis
                        return {
                            "success": True,
                            "architecture": self._generate_default_architecture(df, domain),
                            "recommendations": self._generate_default_recommendations(df, domain),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "fallback": "Using default architecture analysis"
                    }
        except Exception as e:
            logger.error(f"On-Demand Architecture error: {e}")
            return {"success": False, "error": str(e), "fallback": "Using default architecture analysis"}
    
    def _fetch_execution_results(self, execution_id, headers, df, domain, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        architecture = result.get("architecture") or result.get("data") or result
                        recommendations = result.get("recommendations") or result.get("insights") or []
                        
                        if architecture or recommendations:
                            return {
                                "success": True,
                                "architecture": architecture if isinstance(architecture, dict) else {"analysis": str(architecture)},
                                "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        # Still processing
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        # Results not ready yet
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "architecture": self._generate_default_architecture(df, domain),
            "recommendations": self._generate_default_recommendations(df, domain),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_architecture(df, domain):
        """Generate default architecture analysis when API returns empty"""
        return {
            "dataset_shape": list(df.shape),
            "columns_count": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "domain": domain,
            "analysis": f"Dataset with {len(df.columns)} columns and {df.shape[0]} rows. Domain: {domain}"
        }
    
    @staticmethod
    def _generate_default_recommendations(df, domain):
        """Generate default recommendations when API returns empty"""
        recommendations = [
            f"Dataset contains {len(df.columns)} features",
            f"Total records: {df.shape[0]}",
            f"Missing values: {int(df.isnull().sum().sum())}",
            f"Domain classification: {domain}",
            "Recommend feature scaling for numeric columns",
            "Encode categorical variables before modeling"
        ]
        return recommendations


class OnDemandVisualizer:
    """On-Demand Visualizer Agent - Generates charts from CSV data"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandVisualizer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandVisualizer._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def generate_charts(self, df, filename):
        """Call on-demand visualizer to generate charts"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare data payload with proper JSON serialization
            sample_data = df.head(10).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            payload = {
                "dataset": {
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                    "sample_data": sample_data,
                    "total_missing": int(df.isnull().sum().sum())
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        # Fetch results using execution ID
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df)
                    
                    # Extract chart data - handle different response formats
                    charts = result.get("charts") or result.get("data") or []
                    visualizations = result.get("visualizations") or []
                    insights = result.get("insights") or result.get("analysis") or []
                    
                    # If we got valid data, return it
                    if charts or visualizations or insights:
                        return {
                            "success": True,
                            "charts": charts if isinstance(charts, list) else [str(charts)],
                            "visualizations": visualizations if isinstance(visualizations, list) else [str(visualizations)],
                            "insights": insights if isinstance(insights, list) else [str(insights)],
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default visualizations
                        return {
                            "success": True,
                            "charts": self._generate_default_charts(df),
                            "visualizations": self._generate_default_visualizations(df),
                            "insights": self._generate_default_insights(df),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "message": "Could not generate visualizations"
                    }
        except Exception as e:
            logger.error(f"On-Demand Visualizer error: {e}")
            return {"success": False, "error": str(e), "message": "Visualization generation failed"}
    
    def _fetch_execution_results(self, execution_id, headers, df, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        charts = result.get("charts") or result.get("data") or []
                        visualizations = result.get("visualizations") or []
                        insights = result.get("insights") or result.get("analysis") or []
                        
                        if charts or visualizations or insights:
                            return {
                                "success": True,
                                "charts": charts if isinstance(charts, list) else [str(charts)],
                                "visualizations": visualizations if isinstance(visualizations, list) else [str(visualizations)],
                                "insights": insights if isinstance(insights, list) else [str(insights)],
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        # Still processing
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        # Results not ready yet
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "charts": self._generate_default_charts(df),
            "visualizations": self._generate_default_visualizations(df),
            "insights": self._generate_default_insights(df),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_charts(df):
        """Generate default chart descriptions"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        charts = []
        for col in numeric_cols[:3]:
            charts.append(f"Distribution chart: {col}")
        for col in categorical_cols[:3]:
            charts.append(f"Bar chart: {col} frequencies")
        return charts if charts else ["Data distribution overview", "Feature correlation matrix"]
    
    @staticmethod
    def _generate_default_visualizations(df):
        """Generate default visualization descriptions"""
        return [
            f"Dataset Overview: {len(df)} rows × {len(df.columns)} columns",
            f"Data Types: {len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical",
            f"Missing Values: {int(df.isnull().sum().sum())} total",
            "Feature distributions and relationships"
        ]
    
    @staticmethod
    def _generate_default_insights(df):
        """Generate default data insights"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        insights = [
            f"Dataset contains {len(df)} observations",
            f"Total features: {len(df.columns)}",
            f"Missing data: {int(df.isnull().sum().sum())} values",
        ]
        
        if len(numeric_cols) > 0:
            insights.append(f"Numeric columns: {', '.join(numeric_cols.tolist()[:5])}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Categorical columns: {', '.join(categorical_cols.tolist()[:5])}")
        
        return insights


class OnDemandSummarizer:
    """On-Demand Summarizer Agent - Summarizes dataset"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandSummarizer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandSummarizer._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def summarize_dataset(self, df, filename):
        """Call on-demand summarizer to generate dataset summary"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare data payload with proper JSON serialization
            sample_data = df.head(10).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            payload = {
                "dataset": {
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                    "sample_data": sample_data,
                    "total_missing": int(df.isnull().sum().sum())
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        # Fetch results using execution ID
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df)
                    
                    # Extract summary data - handle different response formats
                    summary = result.get("summary") or result.get("data") or {}
                    key_findings = result.get("key_findings") or result.get("findings") or []
                    recommendations = result.get("recommendations") or []
                    
                    # If we got valid data, return it
                    if summary or key_findings or recommendations:
                        return {
                            "success": True,
                            "summary": summary if isinstance(summary, dict) else {"overview": str(summary)},
                            "key_findings": key_findings if isinstance(key_findings, list) else [str(key_findings)],
                            "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default summary
                        return {
                            "success": True,
                            "summary": self._generate_default_summary(df),
                            "key_findings": self._generate_default_findings(df),
                            "recommendations": self._generate_default_recommendations(df),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "message": "Could not generate summary"
                    }
        except Exception as e:
            logger.error(f"On-Demand Summarizer error: {e}")
            return {"success": False, "error": str(e), "message": "Summary generation failed"}
    
    def _fetch_execution_results(self, execution_id, headers, df, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        summary = result.get("summary") or result.get("data") or {}
                        key_findings = result.get("key_findings") or result.get("findings") or []
                        recommendations = result.get("recommendations") or []
                        
                        if summary or key_findings or recommendations:
                            return {
                                "success": True,
                                "summary": summary if isinstance(summary, dict) else {"overview": str(summary)},
                                "key_findings": key_findings if isinstance(key_findings, list) else [str(key_findings)],
                                "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        # Still processing
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        # Results not ready yet
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "summary": self._generate_default_summary(df),
            "key_findings": self._generate_default_findings(df),
            "recommendations": self._generate_default_recommendations(df),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_summary(df):
        """Generate default dataset summary"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(df.select_dtypes(include=['object']).columns),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": round((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2)
        }
    
    @staticmethod
    def _generate_default_findings(df):
        """Generate default key findings"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        findings = [
            f"Dataset size: {len(df)} rows × {len(df.columns)} columns",
            f"Memory usage: {round(df.memory_usage(deep=True).sum() / 1024**2, 2)} MB",
        ]
        
        if len(numeric_cols) > 0:
            findings.append(f"Numeric features: {len(numeric_cols)} ({', '.join(numeric_cols.tolist()[:3])}...)")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            findings.append(f"Categorical features: {len(categorical_cols)} ({', '.join(categorical_cols.tolist()[:3])}...)")
        
        missing = df.isnull().sum().sum()
        if missing > 0:
            findings.append(f"Missing values: {int(missing)} ({round((missing / (len(df) * len(df.columns))) * 100, 2)}%)")
        
        duplicates = len(df[df.duplicated()])
        if duplicates > 0:
            findings.append(f"Duplicate rows: {duplicates}")
        
        return findings
    
    @staticmethod
    def _generate_default_recommendations(df):
        """Generate default recommendations"""
        recommendations = []
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            recommendations.append(f"Handle {int(missing)} missing values using imputation or removal")
        
        # Check for duplicates
        duplicates = len(df[df.duplicated()])
        if duplicates > 0:
            recommendations.append(f"Remove {duplicates} duplicate rows")
        
        # Check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            recommendations.append(f"Normalize or scale {len(numeric_cols)} numeric features for ML")
        
        # Check categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            recommendations.append(f"Encode {len(categorical_cols)} categorical features")
        
        if not recommendations:
            recommendations.append("Dataset appears clean and ready for analysis")
        
        return recommendations


class OnDemandJargonTranslator:
    """On-Demand Jargon Translator Agent - Translates technical column names to business terms"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandJargonTranslator._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandJargonTranslator._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def translate_columns(self, df, filename):
        """Call on-demand jargon translator to translate column names"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare data payload with proper JSON serialization
            sample_data = df.head(5).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            # Create column samples with proper serialization
            column_samples = {}
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        col_sample = df[col].head(3).fillna('NULL').tolist()
                        column_samples[col] = col_sample
                except:
                    pass
            
            payload = {
                "dataset": {
                    "filename": filename,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "shape": list(df.shape),
                    "sample_data": sample_data,
                    "column_samples": column_samples
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        # Fetch results using execution ID
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df)
                    
                    # Extract translation data - handle different response formats
                    translations = result.get("translations") or result.get("column_mappings") or {}
                    insights = result.get("business_insights") or result.get("insights") or []
                    
                    # If we got valid data, return it
                    if translations or insights:
                        return {
                            "success": True,
                            "translations": translations if isinstance(translations, dict) else {"overview": str(translations)},
                            "business_insights": insights if isinstance(insights, list) else [str(insights)],
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default translations
                        return {
                            "success": True,
                            "translations": self._generate_default_translations(df),
                            "business_insights": self._generate_default_insights(df),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "message": "Could not translate column names"
                    }
        except Exception as e:
            logger.error(f"On-Demand Jargon Translator error: {e}")
            return {"success": False, "error": str(e), "message": "Translation failed"}
    
    def _fetch_execution_results(self, execution_id, headers, df, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        translations = result.get("translations") or result.get("column_mappings") or {}
                        insights = result.get("business_insights") or result.get("insights") or []
                        
                        if translations or insights:
                            return {
                                "success": True,
                                "translations": translations if isinstance(translations, dict) else {"overview": str(translations)},
                                "business_insights": insights if isinstance(insights, list) else [str(insights)],
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        # Still processing
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        # Results not ready yet
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "translations": self._generate_default_translations(df),
            "business_insights": self._generate_default_insights(df),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_translations(df):
        """Generate default column translations"""
        translations = {}
        for col in df.columns:
            # Simple pattern-based translation
            business_term = col.replace('_', ' ').title()
            
            # Apply some common patterns
            if 'avg' in col.lower():
                business_term = business_term.replace('Avg', 'Average')
            if 'rev' in col.lower():
                business_term = business_term.replace('Rev', 'Revenue')
            if 'd7' in col.lower():
                business_term = business_term.replace('D7', '(Weekly)')
            if 'd30' in col.lower():
                business_term = business_term.replace('D30', '(Monthly)')
            if 'per_user' in col.lower():
                business_term = business_term.replace('Per User', 'Per User')
            if 'count' in col.lower():
                business_term = business_term.replace('Count', 'Total Count')
            
            translations[col] = business_term
        
        return translations
    
    @staticmethod
    def _generate_default_insights(df):
        """Generate default business insights from columns"""
        insights = []
        
        # Analyze column names for business context
        revenue_cols = [col for col in df.columns if 'rev' in col.lower() or 'revenue' in col.lower()]
        user_cols = [col for col in df.columns if 'user' in col.lower() or 'customer' in col.lower()]
        time_cols = [col for col in df.columns if any(t in col.lower() for t in ['d7', 'd30', 'daily', 'weekly', 'monthly'])]
        
        if revenue_cols:
            insights.append(f"Revenue metrics detected: {len(revenue_cols)} column(s) tracking financial performance")
        
        if user_cols:
            insights.append(f"User/Customer data found: {len(user_cols)} column(s) measuring user engagement")
        
        if time_cols:
            insights.append(f"Time-based metrics identified: {len(time_cols)} column(s) with temporal dimensions")
        
        if not insights:
            insights.append(f"Dataset contains {len(df.columns)} business metrics requiring clarification")
        
        return insights


class OnDemandTopColumnsAnalyzer:
    """On-Demand Top Columns Analyzer - Identifies most important/impactful columns"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandTopColumnsAnalyzer._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandTopColumnsAnalyzer._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def analyze_top_columns(self, df, filename):
        """Analyze and rank columns by importance/impact"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare comprehensive data payload
            sample_data = df.head(10).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            # Calculate column statistics for ranking
            column_stats = {}
            for col in df.columns:
                stats = {
                    "dtype": str(df[col].dtype),
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                    "unique_count": int(df[col].nunique()),
                    "memory_usage": int(df[col].memory_usage(deep=True))
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    stats["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
                    stats["std"] = float(df[col].std()) if not df[col].isna().all() else None
                    stats["min"] = float(df[col].min()) if not df[col].isna().all() else None
                    stats["max"] = float(df[col].max()) if not df[col].isna().all() else None
                    stats["median"] = float(df[col].median()) if not df[col].isna().all() else None
                
                column_stats[col] = stats
            
            payload = {
                "dataset": {
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "column_stats": column_stats,
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                    "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                    "sample_data": sample_data,
                    "total_missing": int(df.isnull().sum().sum()),
                    "duplicates": len(df[df.duplicated()])
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df)
                    
                    # Extract top columns data
                    top_columns = result.get("top_columns") or result.get("important_columns") or []
                    column_ranking = result.get("column_ranking") or result.get("rankings") or {}
                    recommendations = result.get("recommendations") or result.get("insights") or []
                    
                    # If we got valid data, return it
                    if top_columns or column_ranking or recommendations:
                        return {
                            "success": True,
                            "top_columns": top_columns if isinstance(top_columns, list) else [str(top_columns)],
                            "column_ranking": column_ranking if isinstance(column_ranking, dict) else {},
                            "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default analysis
                        return {
                            "success": True,
                            "top_columns": self._generate_default_top_columns(df),
                            "column_ranking": self._generate_default_ranking(df),
                            "recommendations": self._generate_default_recommendations(df),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "message": "Could not analyze top columns"
                    }
        except Exception as e:
            logger.error(f"On-Demand Top Columns Analyzer error: {e}")
            return {"success": False, "error": str(e), "message": "Top columns analysis failed"}
    
    def _fetch_execution_results(self, execution_id, headers, df, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        top_columns = result.get("top_columns") or result.get("important_columns") or []
                        column_ranking = result.get("column_ranking") or result.get("rankings") or {}
                        recommendations = result.get("recommendations") or result.get("insights") or []
                        
                        if top_columns or column_ranking or recommendations:
                            return {
                                "success": True,
                                "top_columns": top_columns if isinstance(top_columns, list) else [str(top_columns)],
                                "column_ranking": column_ranking if isinstance(column_ranking, dict) else {},
                                "recommendations": recommendations if isinstance(recommendations, list) else [str(recommendations)],
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "top_columns": self._generate_default_top_columns(df),
            "column_ranking": self._generate_default_ranking(df),
            "recommendations": self._generate_default_recommendations(df),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_top_columns(df):
        """Generate default top columns based on data quality and variance"""
        # Score columns by importance
        scores = {}
        
        for col in df.columns:
            score = 0
            
            # Low missing values = important
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            score += max(0, 100 - missing_pct)
            
            # Higher variance/uniqueness = important (for numeric/categorical)
            if df[col].dtype in ['int64', 'float64']:
                variance = df[col].var()
                score += min(50, variance / df[col].std() if df[col].std() > 0 else 0)
            else:
                unique_ratio = (df[col].nunique() / len(df)) * 100
                score += min(50, unique_ratio)
            
            # Less duplicates = important
            duplicates = len(df[df.duplicated(subset=[col])])
            score += max(0, 20 - (duplicates / len(df)) * 100)
            
            scores[col] = score
        
        # Sort by score
        sorted_cols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [col for col, score in sorted_cols[:max(1, len(df.columns) // 3)]]
    
    @staticmethod
    def _generate_default_ranking(df):
        """Generate default column ranking with scores"""
        ranking = {}
        
        for col in df.columns:
            score = 0
            details = []
            
            # Data quality score
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            quality_score = 100 - missing_pct
            score += quality_score * 0.3
            if quality_score == 100:
                details.append("✓ No missing values")
            elif quality_score > 90:
                details.append("✓ High data quality")
            
            # Variance score
            if df[col].dtype in ['int64', 'float64']:
                variance = df[col].var()
                variance_score = min(100, (variance / (df[col].std() + 1e-6)) * 10)
                score += variance_score * 0.3
                details.append(f"Variance: {variance:.2f}")
            else:
                unique_pct = (df[col].nunique() / len(df)) * 100
                score += unique_pct * 0.3
                details.append(f"Unique: {unique_pct:.1f}%")
            
            # Cardinality score
            unique_count = df[col].nunique()
            if unique_count < 10:
                details.append("Low cardinality")
            elif unique_count > len(df) * 0.8:
                details.append("Very high cardinality")
            
            ranking[col] = {
                "importance_score": min(100, score),
                "details": details
            }
        
        return ranking
    
    @staticmethod
    def _generate_default_recommendations(df):
        """Generate default recommendations based on columns"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            recommendations.append(f"Focus on {len(numeric_cols)} numeric features for correlation analysis")
        
        if len(categorical_cols) > 0:
            recommendations.append(f"Encode {len(categorical_cols)} categorical features for modeling")
        
        # Check for low-variance columns
        for col in numeric_cols:
            if df[col].std() < 0.1:
                recommendations.append(f"Consider dropping '{col}' - very low variance")
        
        # Check for high cardinality
        for col in categorical_cols:
            if df[col].nunique() > 50:
                recommendations.append(f"'{col}' has high cardinality ({df[col].nunique()} unique values) - may need grouping")
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                recommendations.append(f"Drop '{col}' - only has one unique value (no variance)")
        
        if not recommendations:
            recommendations.append("Dataset appears well-balanced with good feature diversity")
        
        return recommendations


class OnDemandReversibilityChecker:
    """On-Demand Reversibility Checker - Identifies which transformations are reversible"""
    
    def __init__(self, api_key, workflow_id):
        self.api_key = api_key
        self.workflow_id = workflow_id
        self.base_url = "https://api.on-demand.io/automation/api/workflow"
        self.endpoint = f"{self.base_url}/{workflow_id}/execute"
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert non-JSON-serializable values to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: OnDemandReversibilityChecker._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OnDemandReversibilityChecker._make_json_serializable(item) for item in obj]
        elif pd.isna(obj) or (isinstance(obj, float) and np.isinf(obj)):
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def check_reversibility(self, df, filename, transformations=None):
        """Check which transformations are reversible"""
        try:
            headers = {"apikey": self.api_key}
            
            # Prepare data payload with proper JSON serialization
            sample_data = df.head(5).to_dict(orient='records')
            sample_data = self._make_json_serializable(sample_data)
            
            # Create column samples with proper serialization
            column_samples = {}
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        col_sample = df[col].head(3).fillna('NULL').tolist()
                        column_samples[col] = col_sample
                except:
                    pass
            
            payload = {
                "dataset": {
                    "filename": filename,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "shape": list(df.shape),
                    "sample_data": sample_data,
                    "column_samples": column_samples,
                    "transformations": transformations or []
                }
            }
            
            # Initial API call
            with requests.post(self.endpoint, json=payload, headers=headers, timeout=30) as response:
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if this is async response with executionID
                    execution_id = result.get("executionID")
                    if execution_id:
                        # Fetch results using execution ID
                        logger.info(f"Async execution started: {execution_id}")
                        return self._fetch_execution_results(execution_id, headers, df)
                    
                    # Extract reversibility data
                    reversible = result.get("reversible_transformations") or result.get("reversible") or []
                    irreversible = result.get("irreversible_transformations") or result.get("irreversible") or []
                    analysis = result.get("analysis") or result.get("details") or {}
                    
                    # If we got valid data, return it
                    if reversible or irreversible or analysis:
                        return {
                            "success": True,
                            "reversible": reversible if isinstance(reversible, list) else [str(reversible)],
                            "irreversible": irreversible if isinstance(irreversible, list) else [str(irreversible)],
                            "analysis": analysis if isinstance(analysis, dict) else {"overview": str(analysis)},
                            "timestamp": datetime.now().isoformat(),
                            "source": "on-demand"
                        }
                    else:
                        # Empty response - return success with default analysis
                        return {
                            "success": True,
                            "reversible": self._generate_default_reversible(df),
                            "irreversible": self._generate_default_irreversible(df),
                            "analysis": self._generate_default_analysis(df),
                            "timestamp": datetime.now().isoformat(),
                            "source": "fallback"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"API returned {response.status_code}",
                        "message": "Could not check reversibility"
                    }
        except Exception as e:
            logger.error(f"On-Demand Reversibility Checker error: {e}")
            return {"success": False, "error": str(e), "message": "Reversibility check failed"}
    
    def _fetch_execution_results(self, execution_id, headers, df, max_retries=5):
        """Fetch results from async execution"""
        result_endpoint = f"{self.base_url}/{execution_id}/result"
        
        for attempt in range(max_retries):
            try:
                with requests.get(result_endpoint, headers=headers, timeout=10) as response:
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Extract data from result
                        reversible = result.get("reversible_transformations") or result.get("reversible") or []
                        irreversible = result.get("irreversible_transformations") or result.get("irreversible") or []
                        analysis = result.get("analysis") or result.get("details") or {}
                        
                        if reversible or irreversible or analysis:
                            return {
                                "success": True,
                                "reversible": reversible if isinstance(reversible, list) else [str(reversible)],
                                "irreversible": irreversible if isinstance(irreversible, list) else [str(irreversible)],
                                "analysis": analysis if isinstance(analysis, dict) else {"overview": str(analysis)},
                                "timestamp": datetime.now().isoformat(),
                                "source": "on-demand"
                            }
                    elif response.status_code == 202:
                        time.sleep(1)
                        continue
                    elif response.status_code == 404:
                        time.sleep(1)
                        continue
            except Exception as e:
                logger.debug(f"Fetch attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        # If we couldn't get results, return fallback
        return {
            "success": True,
            "reversible": self._generate_default_reversible(df),
            "irreversible": self._generate_default_irreversible(df),
            "analysis": self._generate_default_analysis(df),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    @staticmethod
    def _generate_default_reversible(df):
        """Generate default reversible transformations"""
        return [
            "✓ StandardScaler/Normalization - Can be reversed with stored parameters (mean, std)",
            "✓ MinMax Scaling - Can be reversed using min/max values",
            "✓ Log Transform - Can be reversed with inverse exponential",
            "✓ One-Hot Encoding - Can be decoded if original categories are known",
            "✓ Label Encoding - Can be reversed with mapping dictionary",
        ]
    
    @staticmethod
    def _generate_default_irreversible(df):
        """Generate default irreversible transformations"""
        return [
            "✗ Dropping Columns - Cannot recover deleted data",
            "✗ Removing Rows - Cannot recover deleted observations",
            "✗ Binning/Discretization - Loses granular information",
            "✗ Aggregation - Cannot recover individual records",
            "✗ Deduplication - Cannot recover duplicate records",
        ]
    
    @staticmethod
    def _generate_default_analysis(df):
        """Generate default reversibility analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return {
            "summary": "Reversibility analysis for model interpretation",
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "recommendation": "Store transformation parameters for all reversible operations to enable debugging",
            "critical_insight": "Irreversible operations should be documented for model transparency"
        }


# ======================== MAIN UI INTERFACE ========================
class DataEngineeringSystem:
    """
    Main orchestrator that coordinates all three agents using OpenRouter API.
    Features: Multi-CSV type support, Human-in-the-loop control, Domain adaptation.
    """
    
    def __init__(self):
        self.version = "5.0.0"
        self.architect = None
        self.engineer = Engineer()
        self.observer = Observer()
        # On-Demand Agents
        self.ondemand_api_key = os.getenv("ONDEMAND_API_KEY")
        self.ondemand_architecture = None
        self.ondemand_visualizer = None
        self.ondemand_summarizer = None
        self.ondemand_jargon_translator = None
        self.ondemand_reversibility = None
        self.ondemand_top_columns = None
        if self.ondemand_api_key:
            architecture_workflow = os.getenv("ONDEMAND_ARCHITECTURE_WORKFLOW")
            visualizer_workflow = os.getenv("ONDEMAND_VISUALIZER_WORKFLOW")
            summarizer_workflow = os.getenv("ONDEMAND_SUMMARIZER_WORKFLOW")
            jargon_translator_workflow = os.getenv("ONDEMAND_JARGON_TRANSLATOR_WORKFLOW")
            reversibility_workflow = os.getenv("ONDEMAND_REVERSIBILITY_WORKFLOW")
            top_columns_workflow = os.getenv("ONDEMAND_TOP_COLUMNS_WORKFLOW", architecture_workflow)  # Use top columns workflow if available
            self.ondemand_architecture = OnDemandArchitecture(self.ondemand_api_key, architecture_workflow)
            self.ondemand_visualizer = OnDemandVisualizer(self.ondemand_api_key, visualizer_workflow)
            self.ondemand_summarizer = OnDemandSummarizer(self.ondemand_api_key, summarizer_workflow)
            self.ondemand_jargon_translator = OnDemandJargonTranslator(self.ondemand_api_key, jargon_translator_workflow)
            self.ondemand_reversibility = OnDemandReversibilityChecker(self.ondemand_api_key, reversibility_workflow)
            self.ondemand_top_columns = OnDemandTopColumnsAnalyzer(self.ondemand_api_key, top_columns_workflow)
        # Original configuration
        self.api_key = None
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
        self.df = None
        self.filename = ""
        self.plan = None
        self.execution_results = None
        self.validation = None
        self.csv_domain = None
        self.csv_strategy = None

    def setup_authentication(self):
        """Initialize OpenRouter API with API key"""
        console.clear()
        console.rule("[bold cyan]AI DATA ENGINEERING SYSTEM[/bold cyan]")
        console.print("[dim]Version 5.0.0 | Multi-Agent ML Data Preparation[/dim]\n")
        
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            api_key = questionary.password("Enter OpenRouter API Key (or set OPENROUTER_API_KEY):", qmark="🔒").ask()
        
        if not api_key:
            console.print("[red]API Key required. Exiting.[/red]")
            sys.exit()

        try:
            with console.status("[cyan]Authenticating with OpenRouter...[/cyan]", spinner="dots"):
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com",
                        "X-Title": "AI-Data-Engineering-System"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"Authentication failed: {response.status_code}")
            
            self.api_key = api_key
            console.print("[green]✓ Authentication Successful[/green]")
            console.print(f"[dim]Model: {self.model}[/dim]")
            self.architect = Architect(self.api_key, self.base_url, self.model)
            time.sleep(1)
        except Exception as e:
            console.print(f"[red]❌ Authentication Failed: {e}[/red]")
            logger.error(f"Authentication failed: {e}")
            sys.exit()

    def load_dataset(self):
        """Load CSV file with automatic domain detection"""
        self.show_banner()
        console.rule("[bold cyan]LOAD DATASET[/bold cyan]")
        
        while True:
            path = questionary.text("\nEnter CSV file path (or 'back' to return):").ask()
            if not path:
                return False
            
            path = path.strip().strip('"').strip("'")
            if path.lower() in ['exit', 'back']:
                return False
            
            try:
                self.df = pd.read_csv(path)
                self.filename = os.path.basename(path)
                
                # Auto-detect domain
                self.csv_domain, confidence = CSVTypeDetector.detect_domain(self.df)
                self.csv_strategy = CSVTypeDetector.get_adaptive_strategy(self.csv_domain, self.df)
                
                console.print(f"\n[green]✓ Loaded:[/green] {self.filename}")
                console.print(f"[dim]Shape:[/dim] {self.df.shape[0]} rows × {self.df.shape[1]} columns")
                console.print(f"[cyan]Domain:[/cyan] {self.csv_domain.upper()} ({confidence:.0f}% confidence)")
                console.print(f"[dim]Strategy:[/dim] {self.csv_strategy.get('missing_strategy')} imputation, {self.csv_strategy.get('categorical_encoding')} encoding")
                
                if self.csv_strategy.get('privacy_alert'):
                    console.print(f"[yellow]⚠ Privacy Alert:[/yellow] This domain may contain PII. Handle carefully!")
                
                time.sleep(1)
                return True
            except FileNotFoundError:
                console.print("[yellow]⚠ File not found. Try again.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ Error loading file:[/yellow] {e}")
                logger.error(f"File loading failed: {e}")

    def show_banner(self):
        """Display system banner"""
        console.clear()
        console.rule("[bold cyan]🤖 AI DATA ENGINEERING SYSTEM[/bold cyan]")
        console.print("[dim]v5.0.0 | Multi-Agent ML Data Preparation[/dim]")

    def show_main_menu(self):
        """Main menu with hybrid interface: type commands OR use arrows/numbers"""
        while True:
            self.show_banner()
            
            # Show current state
            if self.df is not None:
                console.print(f"\n[dim]Dataset:[/dim] {self.filename} | [dim]Rows:[/dim] {self.df.shape[0]} | [dim]Columns:[/dim] {self.df.shape[1]}")
                if self.csv_domain:
                    plan_status = "✓" if self.plan else "✗"
                    console.print(f"[dim]Domain:[/dim] {self.csv_domain.upper()} | [dim]Plan:[/dim] {plan_status}")
            else:
                console.print("\n[dim]No dataset loaded[/dim]")
            
            # Show numbered menu options
            console.print("\n[bold cyan]Commands:[/bold cyan]")
            menu_options = [
                ("1", "load", "Load Dataset"),
                ("2", "analyze", "Analyze & Create Plan"),
                ("3", "run", "Execute Plan"),
                ("4", "preview", "Preview Data"),
                ("5", "save", "Save Dataset"),
                ("6", "auto", "Auto Workflow"),
                ("7", "architecture", "On-Demand Architecture Analysis"),
                ("8", "visualize", "Generate Visualizations"),
                ("9", "summarize", "Dataset Summary"),
                ("10", "translate", "Jargon Translator"),
                ("11", "reversibility", "Reversibility Checker"),
                ("12", "topcolumns", "Top Columns Analyzer"),
                ("13", "help", "Help"),
                ("14", "exit", "Exit"),
            ]
            
            for num, cmd, desc in menu_options:
                console.print(f"  {num}. {cmd:10} → {desc}")
            
            console.print("[dim]Type: command name, number (1-13), or custom request[/dim]")
            console.print("[dim]Examples: load  |  3  |  'impute age with mean'[/dim]\n")
            
            cmd = questionary.text("[cyan]What would you like to do?[/cyan] >").ask()
            
            if not cmd:
                continue
            
            cmd = cmd.strip().lower()
            
            # Map number to command
            number_map = {"1": "load", "2": "analyze", "3": "run", "4": "preview", 
                         "5": "save", "6": "auto", "7": "architecture", "8": "visualize", "9": "summarize", "10": "translate", "11": "reversibility", "12": "topcolumns", "13": "help", "14": "exit"}
            if cmd in number_map:
                cmd = number_map[cmd]
            
            # Handle commands
            if cmd == "load":
                self.load_dataset()
            elif cmd == "analyze":
                if self.df is not None:
                    self.analyze_dataset()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "run":
                if self.plan is not None:
                    self.execute_plan()
                elif self.df is not None:
                    console.print("[yellow]⚠ Create a plan first with 'analyze'[/yellow]")
                    time.sleep(1)
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "preview":
                if self.df is not None:
                    self.preview_data()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "save":
                if self.df is not None:
                    self.save_dataset()
                else:
                    console.print("[yellow]⚠ No data to save[/yellow]")
                    time.sleep(1)
            elif cmd == "auto":
                self.auto_workflow()
            elif cmd == "architecture":
                if self.df is not None:
                    self.run_ondemand_architecture()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "visualize":
                if self.df is not None:
                    self.run_ondemand_visualizer()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "summarize":
                if self.df is not None:
                    self.run_ondemand_summarizer()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "translate":
                if self.df is not None:
                    self.run_jargon_translator()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "reversibility":
                if self.df is not None:
                    self.run_ondemand_reversibility()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "topcolumns":
                if self.df is not None:
                    self.run_top_columns_analyzer()
                else:
                    console.print("[yellow]⚠ Load a dataset first[/yellow]")
                    time.sleep(1)
            elif cmd == "help":
                self.show_help()
            elif cmd == "exit":
                if questionary.confirm("Exit?").ask():
                    console.print("[green]Goodbye![/green]")
                    sys.exit()
            else:
                self.handle_custom_command(cmd)

    def analyze_dataset(self):
        """AI Architect analyzes data and creates transformation plan"""
        self.show_banner()
        console.rule("[bold cyan]ARCHITECT ANALYSIS[/bold cyan]")
        
        analysis = self.architect.assess_dataset(self.df, self.filename)
        
        user_request = questionary.text(
            "\nWhat's your data preparation goal? (or press Enter for ML-read" \
            "y):",
            default="Prepare dataset for machine learning"
        ).ask()
        
        self.plan = self.architect.create_plan(analysis, user_request)
        
        if self.plan:
            self.display_plan()
            Prompt.ask("\nPress Enter to continue")

    def display_plan(self):
        """Show plan to user"""
        console.print("\n[bold cyan]RECOMMENDED OPERATIONS:[/bold cyan]\n")
        
        assessment = self.plan.get("dataset_assessment", {})
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        
        for key, value in assessment.items():
            if key != "critical_issues":
                table.add_row(str(key), str(value))
        
        console.print(table)

        if assessment.get("critical_issues"):
            console.print("\n[bold yellow]Critical Issues:[/bold yellow]")
            for issue in assessment["critical_issues"]:
                console.print(f"  • {issue}")

        console.print("\n[bold cyan]Operations to Execute:[/bold cyan]")
        for i, op in enumerate(self.plan.get("recommended_operations", []), 1):
            console.print(f"  {i}. {op['operation']} on {op['target_columns']}")
            console.print(f"     → {op['reasoning'][:80]}...")

    def execute_plan(self):
        """Run Engineer & Observer agents"""
        self.show_banner()
        console.rule("[bold cyan]EXECUTION[/bold cyan]")
        
        if not self.plan:
            console.print("[yellow]⚠ No plan to execute[/yellow]")
            return

        console.print("\n[cyan]Executing operations...[/cyan]\n")
        self.df, self.execution_results = self.engineer.execute_plan(self.df, self.plan)

        console.print("\n[cyan]Validating results...[/cyan]\n")
        self.validation = self.observer.validate_execution(
            self.engineer.df_before,
            self.df,
            self.execution_results,
            self.plan
        )

        report = self.observer.generate_report(self.execution_results, self.validation, self.plan)
        console.print(Panel(report["user_message"], title="RESULTS", border_style="cyan"))

        ml_score = self.validation.get("ml_readiness_score", 0)
        if ml_score >= 95:
            console.print("\n[green]✓ DATASET IS ML-READY![/green]")
        elif ml_score >= 70:
            console.print(f"\n[yellow]ML Readiness: {ml_score}/100 - More work recommended[/yellow]")
        else:
            console.print(f"\n[red]ML Readiness: {ml_score}/100 - Significant issues remain[/red]")

        Prompt.ask("\nPress Enter to continue")

    def preview_data(self):
        """Show data preview"""
        self.show_banner()
        console.rule("[bold cyan]DATA PREVIEW[/bold cyan]")
        
        console.print(f"\n[bold]Dataset:[/bold] {self.filename}")
        console.print(f"[dim]Shape:[/dim] {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
        
        console.print("[bold cyan]First 5 rows:[/bold cyan]")
        console.print(self.df.head().to_string())
        
        console.print("\n[bold cyan]Data types:[/bold cyan]")
        console.print(self.df.dtypes.to_string())
        
        console.print("\n[bold cyan]Missing values:[/bold cyan]")
        console.print(self.df.isnull().sum().to_string())

        Prompt.ask("\nPress Enter to continue")

    def save_dataset(self):
        """Save processed dataset"""
        self.show_banner()
        console.rule("[bold cyan]SAVE DATASET[/bold cyan]")
        
        default_name = f"cleaned_{self.filename}"
        filename = questionary.text("\nOutput filename:", default=default_name).ask()
        
        try:
            self.df.to_csv(filename, index=False)
            console.print(f"\n[green]✓ Saved to {filename}[/green]")
            logger.info(f"Dataset saved: {filename}")
        except Exception as e:
            console.print(f"\n[yellow]⚠ Save failed: {e}[/yellow]")
            logger.error(f"Save failed: {e}")

        Prompt.ask("\nPress Enter to continue")

    def run_ondemand_architecture(self):
        """Run On-Demand Architecture Analysis"""
        if not self.ondemand_architecture:
            console.print("[red]❌ On-Demand Architecture not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]ON-DEMAND ARCHITECTURE ANALYSIS[/bold cyan]")
        
        with console.status("[cyan]Analyzing architecture with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_architecture.analyze_architecture(self.df, self.csv_domain or "general")
        
        if result.get("success"):
            console.print("\n[green]✓ Architecture Analysis Complete[/green]")
            console.print(Panel(
                f"""[bold cyan]ARCHITECTURE INSIGHTS[/bold cyan]
[yellow]{json.dumps(result.get('architecture', {}), indent=2)}[/yellow]

[bold cyan]RECOMMENDATIONS[/bold cyan]
{chr(10).join(['• ' + rec for rec in result.get('recommendations', [])])}
""",
                title="On-Demand Analysis",
                border_style="cyan"
            ))
        else:
            console.print(f"[yellow]⚠ {result.get('fallback', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")

    def run_ondemand_visualizer(self):
        """Run On-Demand Visualizer - Generate Charts"""
        if not self.ondemand_visualizer:
            console.print("[red]❌ On-Demand Visualizer not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]ON-DEMAND VISUALIZER - CHART GENERATION[/bold cyan]")
        
        with console.status("[cyan]Generating visualizations with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_visualizer.generate_charts(self.df, self.filename)
        
        if result.get("success"):
            console.print("\n[green]✓ Visualizations Generated[/green]")
            
            # Display charts info
            charts = result.get("charts", [])
            if charts:
                console.print("\n[bold cyan]GENERATED CHARTS:[/bold cyan]")
                for i, chart in enumerate(charts, 1):
                    console.print(f"  {i}. {chart}")
            
            # Display insights
            insights = result.get("insights", [])
            if insights:
                console.print("\n[bold cyan]DATA INSIGHTS:[/bold cyan]")
                for insight in insights:
                    console.print(f"  • {insight}")
            
            console.print(Panel(
                f"""[bold cyan]VISUALIZATION SUMMARY[/bold cyan]
Charts Generated: {len(charts)}
Insights Found: {len(insights)}
Data Points: {len(self.df)}
Columns Analyzed: {len(self.df.columns)}
""",
                title="On-Demand Visualizer",
                border_style="cyan"
            ))
            
            # Generate and save actual visual charts
            console.print("\n[cyan]Generating visual charts...[/cyan]")
            try:
                chart_files = self._generate_visual_charts(charts)
                if chart_files:
                    console.print(f"\n[green]✓ Visual charts saved:[/green]")
                    for chart_file in chart_files:
                        console.print(f"  • {chart_file}")
            except Exception as e:
                logger.debug(f"Visual chart generation skipped: {e}")
        else:
            console.print(f"[yellow]⚠ {result.get('message', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_ondemand_summarizer(self):
        """Run On-Demand Summarizer - Generate Dataset Summary"""
        if not self.ondemand_summarizer:
            console.print("[red]❌ On-Demand Summarizer not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]ON-DEMAND DATASET SUMMARIZER[/bold cyan]")
        
        with console.status("[cyan]Generating summary with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_summarizer.summarize_dataset(self.df, self.filename)
        
        if result.get("success"):
            console.print("\n[green]✓ Dataset Summary Generated[/green]")
            
            # Display summary
            summary = result.get("summary", {})
            if summary:
                console.print("\n[bold cyan]DATASET OVERVIEW:[/bold cyan]")
                for key, value in summary.items():
                    console.print(f"  {key.replace('_', ' ').title()}: {value}")
            
            # Display key findings
            findings = result.get("key_findings", [])
            if findings:
                console.print("\n[bold cyan]KEY FINDINGS:[/bold cyan]")
                for finding in findings:
                    console.print(f"  • {finding}")
            
            # Display recommendations
            recommendations = result.get("recommendations", [])
            if recommendations:
                console.print("\n[bold cyan]RECOMMENDATIONS:[/bold cyan]")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"  {i}. {rec}")
            
            console.print(Panel(
                f"""[bold cyan]SUMMARY REPORT[/bold cyan]
File: {self.filename}
Rows: {len(self.df)}
Columns: {len(self.df.columns)}
Findings: {len(findings)}
Recommendations: {len(recommendations)}
""",
                title="Dataset Summary",
                border_style="cyan"
            ))
        else:
            console.print(f"[yellow]⚠ {result.get('message', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_jargon_translator(self):
        """Run On-Demand Jargon Translator - Translate technical column names"""
        if not self.ondemand_jargon_translator:
            console.print("[red]❌ On-Demand Jargon Translator not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]🗣️ JARGON TRANSLATOR - BUSINESS-FRIENDLY COLUMN NAMES[/bold cyan]")
        
        with console.status("[cyan]Translating technical terms with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_jargon_translator.translate_columns(self.df, self.filename)
        
        if result.get("success"):
            console.print("\n[green]✓ Column Translation Complete[/green]")
            
            # Display translations
            translations = result.get("translations", {})
            if translations:
                console.print("\n[bold cyan]TECHNICAL → BUSINESS TERMINOLOGY:[/bold cyan]")
                for technical, business in translations.items():
                    console.print(f"  {technical:30} → {business}")
            
            # Display business insights
            insights = result.get("business_insights", [])
            if insights:
                console.print("\n[bold cyan]BUSINESS INSIGHTS:[/bold cyan]")
                for insight in insights:
                    console.print(f"  💡 {insight}")
            
            console.print(Panel(
                f"""[bold cyan]TRANSLATION SUMMARY[/bold cyan]
File: {self.filename}
Total Columns: {len(self.df.columns)}
Translated: {len(translations)}
Business Insights: {len(insights)}
Accessibility: Executive-Ready ✓
""",
                title="Jargon Translation",
                border_style="cyan"
            ))
        else:
            console.print(f"[yellow]⚠ {result.get('message', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_ondemand_reversibility(self):
        """Run On-Demand Reversibility Checker - Identify reversible vs irreversible transformations"""
        if not self.ondemand_reversibility:
            console.print("[red]❌ On-Demand Reversibility Checker not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]🔄 REVERSIBILITY CHECKER - TRANSFORMATION IMPACT ANALYSIS[/bold cyan]")
        
        with console.status("[cyan]Analyzing transformation reversibility with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_reversibility.check_reversibility(self.df, self.filename, self.plan or {})
        
        if result.get("success"):
            console.print("\n[green]✓ Reversibility Analysis Complete[/green]")
            
            # Display reversible transformations
            reversible = result.get("reversible", [])
            if reversible:
                console.print("\n[bold green]REVERSIBLE TRANSFORMATIONS (Can be undone):[/bold green]")
                for item in reversible:
                    console.print(f"  {item}")
            
            # Display irreversible transformations
            irreversible = result.get("irreversible", [])
            if irreversible:
                console.print("\n[bold red]IRREVERSIBLE TRANSFORMATIONS (Cannot be undone):[/bold red]")
                for item in irreversible:
                    console.print(f"  {item}")
            
            # Display analysis
            analysis = result.get("analysis", {})
            if analysis:
                console.print("\n[bold cyan]REVERSIBILITY ANALYSIS:[/bold cyan]")
                for key, value in analysis.items():
                    console.print(f"  {key}: {value}")
            
            console.print(Panel(
                f"""[bold cyan]REVERSIBILITY SUMMARY[/bold cyan]
File: {self.filename}
Total Columns: {len(self.df.columns)}
Reversible Transformations: {len(reversible)}
Irreversible Transformations: {len(irreversible)}
Model Interpretation: Critical for Debugging ✓
""",
                title="Reversibility Check",
                border_style="cyan"
            ))
        else:
            console.print(f"[yellow]⚠ {result.get('message', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def run_top_columns_analyzer(self):
        """Run On-Demand Top Columns Analyzer - Identify most important columns"""
        if not self.ondemand_top_columns:
            console.print("[red]❌ On-Demand Top Columns Analyzer not configured[/red]")
            Prompt.ask("Press Enter to continue")
            return
        
        self.show_banner()
        console.rule("[bold cyan]⭐ TOP COLUMNS ANALYZER - FEATURE IMPORTANCE RANKING[/bold cyan]")
        
        with console.status("[cyan]Analyzing top columns with On-Demand API...[/cyan]", spinner="dots"):
            result = self.ondemand_top_columns.analyze_top_columns(self.df, self.filename)
        
        if result.get("success"):
            console.print("\n[green]✓ Top Columns Analysis Complete[/green]")
            
            # Display top columns
            top_cols = result.get("top_columns", [])
            if top_cols:
                console.print("\n[bold yellow]⭐ MOST IMPORTANT COLUMNS (By Impact):[/bold yellow]")
                for i, col in enumerate(top_cols, 1):
                    console.print(f"  {i}. {col}")
            
            # Display detailed ranking
            ranking = result.get("column_ranking", {})
            if ranking:
                console.print("\n[bold cyan]COLUMN IMPORTANCE RANKING:[/bold cyan]")
                
                # Sort by importance score
                sorted_ranking = sorted(ranking.items(), 
                    key=lambda x: x[1].get("importance_score", 0) if isinstance(x[1], dict) else 0, 
                    reverse=True)
                
                for col, info in sorted_ranking[:10]:  # Show top 10
                    if isinstance(info, dict):
                        score = info.get("importance_score", 0)
                        details = info.get("details", [])
                        bar_length = int(score / 10)
                        bar = "█" * bar_length + "░" * (10 - bar_length)
                        console.print(f"  {col:30} [{bar}] {score:.1f}%")
                        if details:
                            for detail in details[:2]:
                                console.print(f"    └─ {detail}")
            
            # Display recommendations
            recommendations = result.get("recommendations", [])
            if recommendations:
                console.print("\n[bold cyan]FEATURE ENGINEERING RECOMMENDATIONS:[/bold cyan]")
                for i, rec in enumerate(recommendations, 1):
                    console.print(f"  {i}. {rec}")
            
            # Create visual summary
            console.print(Panel(
                f"""[bold cyan]TOP COLUMNS SUMMARY[/bold cyan]
File: {self.filename}
Total Columns: {len(self.df.columns)}
Top Columns Identified: {len(top_cols)}
Analysis Source: {result.get('source', 'API')}

[bold]Key Findings:[/bold]
• Dataset contains {len(self.df)} rows
• {len(self.df.select_dtypes(include=[np.number]).columns)} numeric features
• {len(self.df.select_dtypes(include=['object']).columns)} categorical features
• Total missing values: {int(self.df.isnull().sum().sum())}
""",
                title="Top Columns Analysis",
                border_style="cyan"
            ))
            
            # Option to focus on top columns
            focus = questionary.confirm("\nCreate subset with top columns only?").ask()
            if focus:
                if top_cols:
                    cols_to_keep = [col for col in top_cols if col in self.df.columns]
                    self.df = self.df[cols_to_keep]
                    console.print(f"[green]✔ Dataset focused to {len(cols_to_keep)} top columns[/green]")
        else:
            console.print(f"[yellow]⚠ {result.get('message', result.get('error'))}[/yellow]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def _generate_visual_charts(self, chart_descriptions):
        """Generate actual visual charts from descriptions"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            logger.debug("Matplotlib not available for chart generation")
            return []
        
        chart_files = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        try:
            # Create a figure with subplots for numeric distributions
            if numeric_cols:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'Data Distributions - {self.filename}', fontsize=14, fontweight='bold')
                axes = axes.flatten()
                
                for idx, col in enumerate(numeric_cols[:4]):
                    if idx < len(axes):
                        self.df[col].hist(ax=axes[idx], bins=20, color='steelblue', edgecolor='black')
                        axes[idx].set_title(f'Distribution: {col}')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Frequency')
                
                # Hide unused subplots
                for idx in range(len(numeric_cols[:4]), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                dist_file = f"charts_distribution_{self.filename.replace('.csv', '')}.png"
                dist_path = f"D:\\aadhar\\{dist_file}"
                plt.savefig(dist_path, dpi=100, bbox_inches='tight')
                plt.close()
                chart_files.append(dist_file)
            
            # Create a figure for categorical distributions
            if categorical_cols:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'Categorical Analysis - {self.filename}', fontsize=14, fontweight='bold')
                axes = axes.flatten()
                
                for idx, col in enumerate(categorical_cols[:4]):
                    if idx < len(axes):
                        value_counts = self.df[col].value_counts().head(10)
                        value_counts.plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')
                        axes[idx].set_title(f'Bar Chart: {col}')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Count')
                        axes[idx].tick_params(axis='x', rotation=45)
                
                # Hide unused subplots
                for idx in range(len(categorical_cols[:4]), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                cat_file = f"charts_categorical_{self.filename.replace('.csv', '')}.png"
                cat_path = f"D:\\aadhar\\{cat_file}"
                plt.savefig(cat_path, dpi=100, bbox_inches='tight')
                plt.close()
                chart_files.append(cat_file)
            
            # Create correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                try:
                    import seaborn as sns
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation = self.df[numeric_cols].corr()
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax, 
                               fmt='.2f', square=True, cbar_kws={"shrink": 0.8})
                    ax.set_title(f'Correlation Matrix - {self.filename}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    corr_file = f"charts_correlation_{self.filename.replace('.csv', '')}.png"
                    corr_path = f"D:\\aadhar\\{corr_file}"
                    plt.savefig(corr_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    chart_files.append(corr_file)
                except ImportError:
                    logger.debug("Seaborn not available for correlation heatmap")
        
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
        
        return chart_files

    def show_help(self):
        """Display help menu"""
        console.print(Panel("""
[bold cyan]AI DATA ENGINEERING SYSTEM v5.0[/bold cyan]
[dim]Multi-Agent ML Data Preparation with On-Demand Agents[/dim]

[bold]COMMANDS[/bold]
  load         Load CSV file
  analyze      AI analyzes & creates transformation plan
  run          Execute the plan
  preview      View dataset
  save         Save cleaned data
  auto         Full automatic workflow
  architecture On-Demand Architecture Analysis
  visualize    Generate Visualizations & Charts
  help         Show this help
  exit         Quit

[bold]CUSTOM COMMANDS[/bold]
Type natural language requests:
  • "impute age with mean"
  • "remove duplicates"
  • "encode categorical columns"
  • "detect outliers"
  • "scale numeric features"
  • "drop unused columns"
  • "show correlations"

[bold]WORKFLOW[/bold]
1. Load CSV dataset
2. AI Architect analyzes data
3. Review suggested operations
4. AI Engineer executes plan
5. AI Observer validates results
6. Save cleaned dataset

[bold]AI CAPABILITIES[/bold]
✓ Missing value handling (mean, median, mode)
✓ Categorical encoding (label, one-hot)
✓ Feature scaling (standard, minmax)
✓ Outlier detection & handling
✓ Duplicate removal
✓ Correlation analysis
✓ ML readiness scoring (0-100)

[bold]HUMAN-IN-THE-LOOP[/bold]
At each step you can:
✓ Approve or reject AI decisions
✓ Run custom commands
✓ Take full manual control
✓ Adjust parameters

[bold]SETUP[/bold]
Set in .env file:
  OPENROUTER_API_KEY=your_key
  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
  LLM_MODEL=openai/gpt-3.5-turbo
        """, title="HELP", border_style="cyan"))
        Prompt.ask("\nPress Enter to return")

    def auto_workflow(self):
        """Automatic full workflow"""
        console.print("[bold cyan]🤖 Starting automatic workflow...[/bold cyan]")
        
        # Step 1: Load
        if self.df is None:
            console.print("\n[bold]Step 1: Loading dataset[/bold]")
            if not self.load_dataset():
                return
        
        # Step 2: Analyze
        console.print("\n[bold]Step 2: Analyzing data[/bold]")
        self.analyze_dataset()
        
        # Step 3: Execute
        if self.plan:
            proceed = questionary.confirm("Execute transformation plan?").ask()
            if proceed:
                console.print("\n[bold]Step 3: Running transformations[/bold]")
                self.execute_plan()
        
        # Step 4: Save
        if self.df is not None:
            save = questionary.confirm("Save cleaned dataset?").ask()
            if save:
                console.print("\n[bold]Step 4: Saving data[/bold]")
                self.save_dataset()
        
        console.print("\n[green]✔ Workflow complete![/green]")

    def handle_custom_command(self, cmd):
        """Handle custom natural language commands"""
        if self.df is None:
            console.print("[yellow]⚠ Load a dataset first[/yellow]")
            return
        
        # Common custom commands
        if "impute" in cmd.lower():
            self.custom_impute_command(cmd)
        elif "remove" in cmd.lower() and "duplicate" in cmd.lower():
            self.remove_duplicates_ui()
        elif "encode" in cmd.lower() or "categorical" in cmd.lower():
            self.custom_encode_command(cmd)
        elif "scale" in cmd.lower() or "normalize" in cmd.lower():
            self.custom_scale_command(cmd)
        elif "outlier" in cmd.lower():
            self.detect_outliers_ui()
        elif "correlation" in cmd.lower():
            self.show_correlations()
        elif "info" in cmd.lower() or "describe" in cmd.lower():
            self.preview_data()
        elif "drop" in cmd.lower():
            self.custom_drop_command(cmd)
        else:
            console.print(f"[dim]Command not recognized. Type 'help' for available commands.[/dim]")

    def custom_impute_command(self, cmd):
        """Handle impute commands like 'impute age with mean'"""
        try:
            # Parse command
            parts = cmd.lower().split()
            if "column" in parts:
                idx = parts.index("column") + 1
                column = parts[idx] if idx < len(parts) else None
            else:
                # Guess column from text
                for col in self.df.columns:
                    if col.lower() in cmd.lower():
                        column = col
                        break
                else:
                    column = None
            
            if not column or column not in self.df.columns:
                console.print("[yellow]⚠ Specify which column to impute[/yellow]")
                column = questionary.select("Select column:", 
                    choices=[c for c in self.df.columns if self.df[c].isnull().sum() > 0]).ask()
            
            # Get method
            method = "mean" if "mean" in cmd else "median" if "median" in cmd else "mode" if "mode" in cmd else None
            if not method:
                method = questionary.select("Imputation method:", 
                    choices=["mean", "median", "mode", "forward_fill", "drop"]).ask()
            
            # Execute
            if method == "mean":
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            elif method == "median":
                self.df[column].fillna(self.df[column].median(), inplace=True)
            elif method == "mode":
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            elif method == "forward_fill":
                self.df[column].fillna(method='ffill', inplace=True)
            elif method == "drop":
                self.df.dropna(subset=[column], inplace=True)
            
            console.print(f"[green]✔ Applied {method} imputation to '{column}'[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def custom_encode_command(self, cmd):
        """Handle categorical encoding commands"""
        try:
            categorical = self.df.select_dtypes(include='object').columns.tolist()
            if not categorical:
                console.print("[yellow]⚠ No categorical columns found[/yellow]")
                return
            
            cols = questionary.checkbox("Select columns to encode:", 
                choices=categorical).ask()
            
            method = "label" if "label" in cmd else "one_hot" if "one" in cmd else None
            if not method:
                method = questionary.select("Encoding method:", 
                    choices=["label", "one_hot"]).ask()
            
            for col in cols:
                if method == "label":
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                elif method == "one_hot":
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
            
            console.print(f"[green]✔ Applied {method} encoding to {len(cols)} columns[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def custom_scale_command(self, cmd):
        """Handle scaling commands"""
        try:
            numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                console.print("[yellow]⚠ No numeric columns found[/yellow]")
                return
            
            cols = questionary.checkbox("Select columns to scale:", 
                choices=numeric).ask()
            
            method = "standard" if "standard" in cmd else "minmax" if "minmax" in cmd or "min-max" in cmd else None
            if not method:
                method = questionary.select("Scaling method:", 
                    choices=["standard", "minmax"]).ask()
            
            if method == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            self.df[cols] = scaler.fit_transform(self.df[cols])
            console.print(f"[green]✔ Applied {method} scaling to {len(cols)} columns[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def remove_duplicates_ui(self):
        """Remove duplicate rows"""
        try:
            before = len(self.df)
            self.df.drop_duplicates(inplace=True)
            after = len(self.df)
            console.print(f"[green]✔ Removed {before - after} duplicate rows[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def detect_outliers_ui(self):
        """Detect outliers using IQR"""
        try:
            numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                console.print("[yellow]⚠ No numeric columns[/yellow]")
                return
            
            col = questionary.select("Select column:", choices=numeric).ask()
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            console.print(f"[cyan]Found {len(outliers)} outliers in '{col}'[/cyan]")
            console.print(f"[dim]Range: {lower:.2f} to {upper:.2f}[/dim]")
            
            if len(outliers) > 0:
                action = questionary.select("Action:", 
                    choices=["Keep", "Remove", "Cap values", "Show"]).ask()
                
                if action == "Remove":
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                    console.print(f"[green]✔ Removed outliers[/green]")
                elif action == "Cap values":
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                    console.print(f"[green]✔ Capped outliers[/green]")
                elif action == "Show":
                    console.print(outliers[[col]].head(20))
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def show_correlations(self):
        """Show correlation heatmap"""
        try:
            numeric = self.df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                console.print("[yellow]⚠ Need at least 2 numeric columns[/yellow]")
                return
            
            corr = numeric.corr()
            console.print("\n[bold]Correlation Matrix:[/bold]")
            console.print(corr.to_string())
            
            # Find strong correlations
            strong = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        strong.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            
            if strong:
                console.print("\n[bold cyan]Strong Correlations (>0.7):[/bold cyan]")
                for col1, col2, val in strong:
                    console.print(f"  {col1} ↔ {col2}: {val:.2f}")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def custom_drop_command(self, cmd):
        """Handle column drop commands"""
        try:
            cols = questionary.checkbox("Select columns to drop:", 
                choices=self.df.columns).ask()
            if cols:
                self.df.drop(columns=cols, inplace=True)
                console.print(f"[green]✔ Dropped {len(cols)} columns[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def run(self):
        """Main entry point"""
        self.setup_authentication()
        self.show_main_menu()


# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    try:
        app = DataEngineeringSystem()
        app.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
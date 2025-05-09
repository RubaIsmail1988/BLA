#تحسينات عامة على الهيكل
# Import libraries at the beginning in organized groups
# Standard library
import warnings
from pathlib import Path

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler, 
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Global configurations
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn')

#معالجة البيانات مع تحسين معالجة الأخطاء
def load_data(file_path):
    """Load dataset with error handling"""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found at {file_path}")
            
        df = pd.read_csv(file_path)
        print("Data loaded successfully with shape:", df.shape)
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Handle all data preprocessing with error checking"""
    if df is None or df.empty:
        raise ValueError("Empty dataframe provided for preprocessing")
    
    try:
        # Drop Loan_ID if exists
        if 'Loan_ID' in df.columns:
            df.drop("Loan_ID", axis=1, inplace=True)
        
        # Convert "3+" to numerical
        if 'Dependents' in df.columns:
            df["Dependents"] = df["Dependents"].replace("3+", 3)
        
        # Handle missing values
        cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 
                   'Credit_History', 'Loan_Amount_Term']
        num_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
        
        for col in cat_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        for col in num_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Remove outliers
        df = remove_outliers(df, 'ApplicantIncome')
        df = remove_outliers(df, 'LoanAmount')
        
        return df
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None

def remove_outliers(df, column, factor=1.5):
    """Remove outliers with safety checks"""
    if column not in df.columns:
        print(f"Column {column} not found in dataframe")
        return df
        
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column {column} is not numeric")
        return df
        
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    before = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after = len(df)
    
    print(f"Removed {before - after} outliers from {column}")
    return df

# إنشاء محسنPipeline  مع معالجة الأخطاء
def create_pipeline(model_type='random_forest'):
    """Create a robust ML pipeline with error handling"""
    
    # Define numeric and categorical features
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 
                      'LoanAmount', 'Loan_Amount_Term']
    categorical_features = ['Gender', 'Married', 'Dependents', 
                          'Education', 'Self_Employed', 'Property_Area']
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Model selection
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

#تقييم النموذج مع معالجة الأخطاء 
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with comprehensive metrics"""
    try:
        y_pred = model.predict(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        #
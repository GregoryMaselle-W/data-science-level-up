import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """
    A comprehensive preprocessor that handles data for multiple model types:
    - Tree-based: LightGBM, XGBoost, Random Forest (can handle categories natively)
    - Linear: Logistic Regression (needs scaling and encoding)
    """
    
    def __init__(self, model_type='tree'):
        """
        model_type: 'tree' for tree-based models, 'linear' for linear models, 'both' for compatible preprocessing
        """
        self.model_type = model_type
        self.label_encoders = {}
        self.scaler = None
        self.one_hot_encoder = None
        self.feature_names = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.high_cardinality_threshold = 20  # For deciding encoding strategy
        
    def identify_column_types(self, df, exclude_cols=['TX_ID', 'TX_FRAUD', 'TX_TS']):
        """Automatically identify categorical and numerical columns"""
        
        # Exclude identifying columns and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in feature_cols:
            if df[col].dtype in ['object', 'category', 'bool']:
                if col == 'DAY_OF_MONTH':
                    self.numerical_columns.append(col)
                else:
                    self.categorical_columns.append(col)
            elif df[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
                # Check if it's actually categorical (low unique values)
                if df[col].nunique() < 10 and col not in ['MONTHS_UNTIL_EXPIRY', 'HOUR', 'DAY_OF_WEEK']:
                    self.categorical_columns.append(col)
                else:
                    self.numerical_columns.append(col)
        
        print(f"Identified {len(self.categorical_columns)} categorical columns")
        print(f"Identified {len(self.numerical_columns)} numerical columns")
        
        return self.categorical_columns, self.numerical_columns
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately by column type"""
        df = df.copy()

        # Numerical columns: fill with median or -999 for tree models
        for col in self.numerical_columns:
            if df[col].isnull().any():
                if self.model_type == 'tree':
                    # Tree models can handle special values
                    df[col].fillna(-999, inplace=True)
                else:
                    # Linear models need proper imputation
                    df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns: fill with 'missing'
        for col in self.categorical_columns:
            if df[col].dtype.name == 'category':
                if 'missing' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['missing'])
                    df[col].fillna('missing', inplace=True)
            else:
                df[col].fillna('missing', inplace=True)
        
        return df
    
    def encode_categoricals(self, df, fit=True):
        """Encode categorical variables based on model type"""
        df = df.copy()
        
        if self.model_type == 'tree':
            # Use Label Encoding for tree-based models
            for col in self.categorical_columns:
                if col in df.columns:
                    print(f"  Encoding column: {col}")  # Debug - remove later
                    
                    if fit:
                        # Fit and transform
                        le = LabelEncoder()
                        df[col] = df[col].astype(str)
                        df[col] = le.fit_transform(df[col])
                        self.label_encoders[col] = le
                    else:
                        # Transform only (for test data) - VECTORIZED VERSION
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            df[col] = df[col].astype(str)
                            
                            # VECTORIZED approach - much faster!
                            # Create mapping dictionary
                            mapping = {cat: i for i, cat in enumerate(le.classes_)}
                            # Map with default value for unknown
                            df[col] = df[col].map(mapping).fillna(-1).astype(int)
                            
        elif self.model_type in ['linear', 'both']:
            # Use One-Hot Encoding for linear models
            # But use Label Encoding for high cardinality features
            
            low_cardinality_cats = []
            high_cardinality_cats = []
            
            for col in self.categorical_columns:
                if col in df.columns:
                    if df[col].nunique() > self.high_cardinality_threshold:
                        high_cardinality_cats.append(col)
                    else:
                        low_cardinality_cats.append(col)
            
            # Label encode high cardinality
            for col in high_cardinality_cats:
                if fit:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        
                        # VECTORIZED approach for high cardinality too!
                        mapping = {cat: i for i, cat in enumerate(le.classes_)}
                        df[col] = df[col].map(mapping).fillna(-1).astype(int)
            
            # One-hot encode low cardinality
            if len(low_cardinality_cats) > 0:
                if fit:
                    self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = self.one_hot_encoder.fit_transform(df[low_cardinality_cats])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=self.one_hot_encoder.get_feature_names_out(low_cardinality_cats),
                        index=df.index
                    )
                else:
                    encoded = self.one_hot_encoder.transform(df[low_cardinality_cats])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=self.one_hot_encoder.get_feature_names_out(low_cardinality_cats),
                        index=df.index
                    )
                
                # Drop original columns and add encoded ones
                df = df.drop(columns=low_cardinality_cats)
                df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features for linear models"""
        df = df.copy()
        
        if self.model_type in ['linear', 'both']:
            # Get current numerical columns (some may have been encoded)
            current_numerical = [col for col in df.columns 
                                if col in self.numerical_columns 
                                and col not in ['TX_ID', 'TX_FRAUD']]
            
            if len(current_numerical) > 0:
                if fit:
                    # Use RobustScaler for better handling of outliers in fraud data
                    self.scaler = RobustScaler()
                    df[current_numerical] = self.scaler.fit_transform(df[current_numerical])
                else:
                    if self.scaler is not None:
                        df[current_numerical] = self.scaler.transform(df[current_numerical])
        
        return df
    
    def create_additional_features(self, df):
        """Create ratio and interaction features that work well for fraud detection"""
        df = df.copy()
        
        # Transaction amount ratios (if columns exist)
        if 'TX_AMOUNT' in df.columns:
            if 'AVERAGE_TICKET_SALE_AMOUNT' in df.columns:
                df['AMOUNT_TO_AVG_RATIO'] = df['TX_AMOUNT'] / (df['AVERAGE_TICKET_SALE_AMOUNT'] + 0.01)
            
            if 'ANNUAL_TURNOVER' in df.columns:
                df['AMOUNT_TO_TURNOVER_RATIO'] = df['TX_AMOUNT'] / (df['ANNUAL_TURNOVER'] / 365 + 0.01)
        
        # Risk score combinations
        if 'TERMINAL_FRAUD_RATE' in df.columns and 'EUCLIDEAN_DISTANCE' in df.columns:
            df['RISK_SCORE'] = df['TERMINAL_FRAUD_RATE'] * np.log1p(df['EUCLIDEAN_DISTANCE'])
        
        # Time-based risk
        if 'IS_WEEKEND' in df.columns and 'IS_BUSINESS_HOURS' in df.columns:
            df['OFF_HOURS_TRANSACTION'] = ((df['IS_WEEKEND'] == 1) | 
                                           (df['IS_BUSINESS_HOURS'] == 0)).astype(int)
        
        return df
    
    def fit_transform(self, df, y=None):
        """Fit preprocessor and transform training data"""
        print(f"Preprocessing for {self.model_type} models...")
        
        # Store original shape
        original_shape = df.shape
        print(f"Original shape: {original_shape}")
        
        # Identify column types
        self.identify_column_types(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create additional features
        df = self.create_additional_features(df)
        
        # Encode categoricals
        df = self.encode_categoricals(df, fit=True)
        
        # Scale features (if needed)
        df = self.scale_features(df, fit=True)
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                             if col not in ['TX_ID', 'TX_FRAUD', 'TX_TS']]
        
        print(f"Final shape: {df.shape}")
        print(f"Features created: {len(self.feature_names)}")
        
        return df
    
    def transform(self, df):
        """Transform test data using fitted preprocessor"""
        print(f"Transform started - DataFrame shape: {df.shape}")
        
        # Handle missing values
        print("Step 1: Handling missing values...")
        df = self.handle_missing_values(df)
        print(f"After missing values - shape: {df.shape}")
        
        # Create additional features
        print("Step 2: Creating additional features...")
        df = self.create_additional_features(df)
        print(f"After additional features - shape: {df.shape}")
        
        # Encode categoricals
        print("Step 3: Encoding categoricals...")
        df = self.encode_categoricals(df, fit=False)
        print(f"After encoding - shape: {df.shape}")
        
        # Scale features (if needed)
        print("Step 4: Scaling features...")
        df = self.scale_features(df, fit=False)
        print(f"After scaling - shape: {df.shape}")
        
        print('Transform completed!')
        
        return df
    
    def get_feature_importance_df(self, model, feature_names=None):
        """Helper to get feature importance from tree models"""
        if feature_names is None:
            feature_names = self.feature_names
            
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            print("Model doesn't have feature_importances_ attribute")
            return None


# Example usage function
def prepare_data_for_modeling(train_df, test_df, model_type='tree'):
    """
    Prepare train and test data for modeling
    
    model_type options:
    - 'tree': For XGBoost, LightGBM, Random Forest
    - 'linear': For Logistic Regression, SVM
    - 'both': Creates compatible features for all models
    """
    
    # Initialize preprocessor
    preprocessor = FraudDataPreprocessor(model_type=model_type)
    
    # Separate target if it exists
    if 'TX_FRAUD' in train_df.columns:
        y_train = train_df['TX_FRAUD'].values
        train_df = train_df.drop('TX_FRAUD', axis=1)
    else:
        y_train = None
    
    # Fit and transform training data
    train_processed = preprocessor.fit_transform(train_df, y_train)
    
    # Transform test data
    test_processed = preprocessor.transform(test_df)
    
    # Get feature columns (excluding ID and timestamp)
    feature_cols = [col for col in train_processed.columns 
                   if col not in ['TX_ID', 'TX_TS', 'TX_FRAUD']]
    
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    print(f"\nFinal dataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    if y_train is not None:
        print(f"y_train: {y_train.shape}")
        print(f"Fraud rate: {y_train.mean():.2%}")
    
    return X_train, X_test, y_train, preprocessor


# Model-specific configurations
def get_model_config(model_type, X_train, y_train):
    """Get model-specific configurations and parameters"""
    
    configs = {
        'xgboost': {
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
                'use_label_encoder': False,
                'random_state': 42
            },
            'needs_encoding': True,  # XGBoost needs numeric encoding
            'can_handle_missing': True
        },
        'lightgbm': {
            'params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'force_col_wise': True
            },
            'needs_encoding': False,  # LightGBM can handle categories directly
            'can_handle_missing': True
        },
        'random_forest': {
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            },
            'needs_encoding': True,
            'can_handle_missing': False
        },
        'logistic_regression': {
            'params': {
                'penalty': 'l2',
                'C': 1.0,
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            },
            'needs_encoding': True,
            'needs_scaling': True,
            'can_handle_missing': False
        }
    }
    
    return configs.get(model_type, configs['xgboost'])
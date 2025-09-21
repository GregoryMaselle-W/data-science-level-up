import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, roc_curve, f1_score,average_precision_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from FraudDataPreprocessor import FraudDataPreprocessor
warnings.filterwarnings('ignore')

# Assuming the FraudDataPreprocessor class from previous artifact is imported
# Also assuming your feature engineering function process_data_for_training is available

class FraudModelTrainer:
    """
    Comprehensive fraud detection model trainer supporting multiple algorithms
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.preprocessors = {}
        self.results = {}
        
    def prepare_validation_split(self, X, y, val_size=0.2):
        """Create stratified train/validation split"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Fraud rate: {y_train.mean():.2%}")
        print(f"Validation set: {X_val.shape}, Fraud rate: {y_val.mean():.2%}")
        
        return X_train, X_val, y_train, y_val
    
    def handle_imbalance(self, X_train, y_train, method='none'):
        """
        Handle class imbalance using various techniques
        
        method: 'none', 'smote', 'undersample'
        """
        if method == 'smote':
            print("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {X_resampled.shape}, Fraud rate: {y_resampled.mean():.2%}")
            return X_resampled, y_resampled
            
        elif method == 'undersample':
            print("Applying random undersampling...")
            rus = RandomUnderSampler(random_state=self.random_state, sampling_strategy=1)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
            print(f"After undersampling: {X_resampled.shape}, Fraud rate: {y_resampled.mean():.2%}")
            return X_resampled, y_resampled
            
        else:
            return X_train, y_train
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, preprocessor, use_undersampling=False):
        """Train XGBoost model with early stopping"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # XGBoost parameters optimized for fraud detection
        # MOVED early_stopping_rounds HERE for newer XGBoost versions
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'n_estimators': 500, 
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
            'random_state': self.random_state,
            'tree_method': 'hist',  # Faster training
            'enable_categorical': False,  # We already encoded
            'early_stopping_rounds': 50  # MOVED HERE!
        }
        
        model = xgb.XGBClassifier(**params)

        if use_undersampling:
            X_train, y_train = self.handle_imbalance(X_train, y_train, method='undersample')
        # Train with early stopping (no early_stopping_rounds in fit())
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=10  # Your verbose setting
        )
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)

        print(f"XGBoost Validation AUC: {auc_score:.4f}")
        print(f"XGBoost Validation F1: {f1:.4f}")
        
        # Handle different version attributes
        try:
            print(f"Best iteration: {model.best_iteration}")
        except AttributeError:
            print(f"Best iteration: {model.best_iteration_}")
        
        self.models['xgboost'] = model
        self.preprocessors['xgboost'] = preprocessor
        self.results['xgboost'] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        
        return model, auc_score
    
    def train_with_undersampling(X_train, y_train, X_val, y_val):
        # Undersample training data
        rus = RandomUnderSampler(sampling_strategy=0.2, random_state=42)  # 20% fraud
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
        
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=10,
            # Don't use scale_pos_weight with undersampled data
            random_state=42
        )
        
        model.fit(
            X_train_us, y_train_us,
            eval_set=[(X_val, y_val)],  # Original distribution!
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50)]
        )
        
        return model

    def train_lightgbm(self, X_train, y_train, X_val, y_val, preprocessor, use_undersampling=False):
        """Train LightGBM model with early stopping"""
        print("\n" + "="*50)
        print("Training LightGBM Model")
        print("="*50)
        
        # LightGBM parameters optimized for fraud detection
        # params = {
        #     'objective': 'binary',
        #     'metric': 'auc',
        #     'boosting_type': 'gbdt',
        #     'num_leaves': 31,
        #     'min_data_in_leaf': 100,
        #     'feature_fraction': 0.8,
        #     'bagging_fraction': 0.8,
        #     'bagging_freq': 5,
        #     'learning_rate': 0.01,
        #     'n_estimators': 1000,  # Reduced for faster training
        #     'class_weight': 'balanced',
        #     'random_state': self.random_state,
        #     'force_col_wise': True,
        #     'verbose': -1
        # }

        params = {
            'n_estimators':5000,          # Large number, will rely on early stopping
            'learning_rate':0.02,         # Small learning rate → better generalization
            'max_depth':10,               # Slightly deeper trees for complex patterns
            'num_leaves':128,             # More leaves to capture subtle splits
            'min_child_samples':30,       # Minimum samples per leaf → helps prevent overfitting
            'subsample':0.8,              # Row sampling → adds randomness, reduces overfit
            'colsample_bytree':0.8,       # Feature sampling per tree → adds robustness
            'reg_alpha':2.0,              # L1 regularization → reduces overfitting
            'reg_lambda':2.0,             # L2 regularization → reduces overfitting
            'scale_pos_weight':2,        # Handle imbalance (fraud is rare). Adjust based on ratio
            'random_state':42
        }
        
        model = lgb.LGBMClassifier(**params)
        if use_undersampling:
            X_train, y_train = self.handle_imbalance(X_train, y_train, method='undersample')

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]  # Changed to show every 10
        )
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        
        print(f"LightGBM Validation AUC: {auc_score:.4f}")
        print(f"LightGBM Validation F1: {f1:.4f}")
        print(f"Best iteration: {model.best_iteration_}")
        
        self.models['lightgbm'] = model
        self.preprocessors['lightgbm'] = preprocessor
        self.results['lightgbm'] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        
        return model, auc_score
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, preprocessor, use_smote=False):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest Model")
        print("="*50)
        
        # Apply SMOTE if requested (RF can't handle imbalance as well as boosting)
        if use_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train, method='smote')
        
        # Random Forest parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'class_weight': 'balanced_subsample',
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': 1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        
        
        print(f"Random Forest Validation AUC: {auc_score:.4f}")
        print(f"Random Forest Validation F1: {f1:.4f}")
        
        self.models['random_forest'] = model
        self.preprocessors['random_forest'] = preprocessor
        self.results['random_forest'] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        
        return model, auc_score
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val, preprocessor, use_smote=False):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression Model")
        print("="*50)
        
        # Logistic Regression parameters
        params = {
            'penalty': 'l2',
            'C': 0.1,  # Regularization strength
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': self.random_state,
            'solver': 'liblinear'
        }

        if use_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train, method='smote')
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        print(f"Logistic Regression Validation AUC: {auc_score:.4f}")
        
        self.models['logistic_regression'] = model
        self.preprocessors['logistic_regression'] = preprocessor
        self.results['logistic_regression'] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred_proba
        }
        
        return model, auc_score
    
    def plot_feature_importance(self, model_name='xgboost', top_n=20):
        """Plot feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]
        preprocessor = self.preprocessors[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': preprocessor.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, y='feature', x='importance')
            plt.title(f'Top {top_n} Features - {model_name.upper()}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
    
    def compare_models(self, y_val):
        """Compare all trained models"""
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        comparison_df = pd.DataFrame([
            {'Model': name, 'AUC': results['auc']} 
            for name, results in self.results.items()
        ]).sort_values('AUC', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(y_val, results['predictions'])
            plt.plot(fpr, tpr, label=f"{name} (AUC: {results['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
terminal_stats = None

def process_data_for_training(targetFile = 'transactions_train.csv', isTest = False):
    global terminal_stats
    customers_df = pd.read_csv(file_path + "customers.csv")
    merchants_df = pd.read_csv(file_path + "merchants.csv")
    terminals_df = pd.read_csv(file_path + "terminals.csv")
    transactions_df = pd.read_csv(file_path + targetFile)

    merged_train = transactions_df.copy()
    merged_train = merged_train.merge(customers_df, on='CUSTOMER_ID', how='left')
    merged_train = merged_train.merge(terminals_df, on='TERMINAL_ID', how='left')
    merged_train = merged_train.merge(merchants_df, on='MERCHANT_ID', how='left')

    merged_train.columns = merged_train.columns.str.upper()

    # Time-based Features

    def categorise_transaction_frequency(seconds):
        if pd.isna(seconds):
            return "first"
        elif seconds < 60:           
            return "under_1_min"
        elif seconds < 3600:         
            return "under_1_hour"
        elif seconds < 86400:        
            return "under_1_day"
        elif seconds < 604800:
            return "under_1_week"
        elif seconds < 2.592e+6:
            return "under_1_month"
        elif seconds < 1.555e+7:
            return "under_6_months" 
        else:
            return "over_6_months"
        
    def categorise_time_until_expiration(months):
        if months <= 0:           
            return "expired"
        elif months < 1:         
            return "under_1_month"
        elif months < 3:        
            return "under_3_months"
        elif months < 6:
            return "under_6_months"
        elif months < 12:
            return "under_1_year"
        else:
            return "over_1_year"
        
    def months_until_expiry(row):
        return (row["CARD_EXPIRY_DATE"].year - row["TX_TS"].year) * 12 + \
            (row["CARD_EXPIRY_DATE"].month - row["TX_TS"].month)

    merged_train["TX_TS"] = pd.to_datetime(merged_train["TX_TS"])
    merged_train["DAY_OF_WEEK"] = merged_train['TX_TS'].dt.day_of_week
    merged_train["HOUR"] = merged_train['TX_TS'].dt.hour
    merged_train["IS_WEEKEND"] = merged_train['TX_TS'].dt.day_of_week >= 5
    merged_train["DAY_OF_MONTH"] = merged_train['TX_TS'].dt.day
    merged_train_sorted = merged_train.sort_values(["CUSTOMER_ID", "TX_TS"])
    merged_train["TIME_SINCE_LAST_TRANSACTION"] = merged_train_sorted.groupby("CUSTOMER_ID")["TX_TS"].diff().dt.total_seconds()
    merged_train["IS_FIRST_TRANSACTION"] = merged_train["TIME_SINCE_LAST_TRANSACTION"].isna().astype(int)
    merged_train["TIME_SINCE_LAST_TRANSACTION"] = merged_train["TIME_SINCE_LAST_TRANSACTION"].fillna(0)
    merged_train["WINDOW_AFTER_LAST_TRANSACTION_CATEGORY"] = merged_train["TIME_SINCE_LAST_TRANSACTION"].apply(categorise_transaction_frequency).astype("category")
    merged_train["IS_BUSINESS_HOURS"] = (merged_train["TX_TS"].dt.hour >=8) | (merged_train["TX_TS"].dt.hour <=17)
    merged_train["CARD_EXPIRY_DATE"] = pd.to_datetime(
        "01/" + merged_train["CARD_EXPIRY_DATE"].astype(str), format="%d/%m/%y"
    )
    merged_train["CARD_EXPIRY_DATE"] = merged_train["CARD_EXPIRY_DATE"].dt.to_period('M').dt.end_time
    merged_train["MONTHS_UNTIL_EXPIRY"] = merged_train.apply(months_until_expiry, axis=1)
    merged_train["EXPIRY_CATEGORY"] = merged_train["MONTHS_UNTIL_EXPIRY"].apply(categorise_time_until_expiration).astype("category")

    # Geographical Features

    coord_cols = [col for col in merged_train.columns if any(coord in col for coord in ['X_', 'Y_'])]
    print(f"  Found coordinate columns: {coord_cols}")

    if len(coord_cols) >= 4:
        # Find customer and terminal coordinates
        customer_coords = [col for col in coord_cols if 'CUSTOMER' in col]
        terminal_coords = [col for col in coord_cols if 'TERMINAL' in col]
        
        if len(customer_coords) >= 2 and len(terminal_coords) >= 2:
            customer_x = [col for col in customer_coords if 'X' in col][0]
            customer_y = [col for col in customer_coords if 'Y' in col][0]
            terminal_x = [col for col in terminal_coords if 'X' in col][0]
            terminal_y = [col for col in terminal_coords if 'Y' in col][0]
            
            # Distance calculations
            merged_train['EUCLIDEAN_DISTANCE'] = np.sqrt(
                (merged_train[customer_x] - merged_train[terminal_x])**2 + 
                (merged_train[customer_y] - merged_train[terminal_y])**2
            )
            
            merged_train['MANHATTAN_DISTANCE'] = (
                abs(merged_train[customer_x] - merged_train[terminal_x]) + 
                abs(merged_train[customer_y] - merged_train[terminal_y])
            )
            
            # Distance categories
            merged_train['DISTANCE_CATEGORY'] = pd.cut(merged_train['EUCLIDEAN_DISTANCE'],
                                                bins=[0, 10, 25, 50, 100, float('inf')],
                                                labels=['very_close', 'close', 'medium', 'far', 'very_far'])
            
            # Quadrant analysis
            merged_train['CUSTOMER_QUADRANT'] = (
                (merged_train[customer_x] >= 50).astype(int) * 2 + 
                (merged_train[customer_y] >= 50).astype(int)
            )
            
            merged_train['TERMINAL_QUADRANT'] = (
                (merged_train[terminal_x] >= 50).astype(int) * 2 + 
                (merged_train[terminal_y] >= 50).astype(int)
            )
            
            merged_train['SAME_QUADRANT'] = (merged_train['CUSTOMER_QUADRANT'] == merged_train['TERMINAL_QUADRANT']).astype(int)
        else:
            print(f"  Insufficient coordinate data")

# We get terminal aggregate features since we noticed that terminal location is a big indicator of fraud
    if (isTest == False):
        terminal_stats = (
            merged_train.groupby("TERMINAL_ID")["TX_FRAUD"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={
                "mean": "TERMINAL_FRAUD_RATE",
                "sum": "TERMINAL_FRAUD_COUNT",
                "count": "TERMINAL_TX_COUNT"
            })
        )

        merged_train = merged_train.merge(terminal_stats, on="TERMINAL_ID", how="left")
    else:
        # Use precomputed stats (must exist already)
        if terminal_stats is None:
            raise ValueError("terminal_stats not initialized. Run on training data first.")
        
        merged_train = merged_train.merge(terminal_stats, on="TERMINAL_ID", how="left")
        
        # Fill missing values for unseen terminals
        merged_train.fillna({
            "TERMINAL_FRAUD_RATE": 0,  # or global fraud rate
            "TERMINAL_FRAUD_COUNT": 0,
            "TERMINAL_TX_COUNT": 0
        }, inplace=True)
    
    drop_cols = []
    if (isTest):
        drop_cols = [     
            'CARD_DATA',    
            'CUSTOMER_ID',  
            'TERMINAL_ID',
            'MERCHANT_ID',
            'TX_TS',
            'CARD_EXPIRY_DATE',
            'ACQUIRER_ID',
            'LEGAL_NAME',
            'X_TERMINAL_ID',
            'Y_TERMINAL__ID'
        ]
    else:
        drop_cols = [  
            'TX_ID',   
            'CARD_DATA',    
            'CUSTOMER_ID',  
            'TERMINAL_ID',
            'MERCHANT_ID',
            'TX_TS',
            'CARD_EXPIRY_DATE',
            'ACQUIRER_ID',
            'LEGAL_NAME',
            'X_TERMINAL_ID',
            'Y_TERMINAL__ID'
        ]
    merged_train = merged_train.drop(columns=drop_cols, errors='ignore')

    return merged_train


# ===============================================
# MAIN EXECUTION PIPELINE
# ===============================================

def run_complete_fraud_detection_pipeline(file_path):
    """
    Complete pipeline from raw data to trained models
    """
    
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # =========================================
    # STEP 1: Load and engineer features
    # =========================================
    print("\n[STEP 1] Loading data and engineering features...")
    
    # Your existing feature engineering code
    global terminal_stats
    terminal_stats = None
    
    # Process training data with your feature engineering
    train_df = process_data_for_training(
        targetFile='transactions_train.csv', 
        isTest=False
    )
    
    # Process test data
    test_df = process_data_for_training(
        targetFile='transactions_test.csv', 
        isTest=True
    )
    
    # =========================================
    # STEP 2: Initialize trainer
    # =========================================
    trainer = FraudModelTrainer(random_state=42)
    
    # =========================================
    # STEP 3: Train XGBoost
    # =========================================
    print("\n[STEP 3] Training XGBoost...")
    
    # Prepare data for tree models
    preprocessor_tree = FraudDataPreprocessor(model_type='tree')
    
    print(train_df.columns)
    # Separate target
    y_train = train_df['TX_FRAUD'].values
    X_train_full = train_df.drop('TX_FRAUD', axis=1)
    
    # Preprocess
    X_train_tree = preprocessor_tree.fit_transform(X_train_full)
    X_test_tree = preprocessor_tree.transform(test_df)
    
    # Get feature columns
    feature_cols = [col for col in X_train_tree.columns 
                   if col not in ['TX_ID', 'TX_TS', 'TX_FRAUD']]
    X_train_tree = X_train_tree[feature_cols]
    X_test_tree = X_test_tree[feature_cols]
    
    # Create validation split
    X_tr, X_val, y_tr, y_val = trainer.prepare_validation_split(X_train_tree, y_train)
    
    # Train XGBoost
    print(f"Training data shape: {X_tr.shape}")
    print(f"Training data memory usage: {X_tr.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data type of X_tr: {type(X_tr)}")
    print(f"Starting XGBoost training...")
    for col in X_tr.columns:
        print(f"{col}: {X_tr[col].dtype}")
    xgb_model, xgb_auc = trainer.train_xgboost(X_tr, y_tr, X_val, y_val, preprocessor_tree, True)
    
    # =========================================
    # STEP 4: Train LightGBM
    # =========================================
    print("\n[STEP 4] Training LightGBM...")
    
    # LightGBM can use the same preprocessed data
    lgb_model, lgb_auc = trainer.train_lightgbm(X_tr, y_tr, X_val, y_val, preprocessor_tree, True)
    
    # =========================================
    # STEP 5: Train Random Forest
    # =========================================
    #print("\n[STEP 5] Training Random Forest...")
    
    # Random Forest with SMOTE to handle imbalance
    rf_model, rf_auc = trainer.train_random_forest(
        X_tr, y_tr, X_val, y_val, preprocessor_tree, use_smote=True
    )
    
    # =========================================
    # STEP 6: Train Logistic Regression
    # =========================================
    print("\n[STEP 6] Training Logistic Regression...")
    
    # Prepare data for linear models
    preprocessor_linear = FraudDataPreprocessor(model_type='linear')
    
    # Preprocess for linear models
    X_train_linear = preprocessor_linear.fit_transform(X_train_full)
    X_test_linear = preprocessor_linear.transform(test_df)
    
    # Get feature columns
    feature_cols_linear = [col for col in X_train_linear.columns 
                          if col not in ['TX_ID', 'TX_TS', 'TX_FRAUD']]
    X_train_linear = X_train_linear[feature_cols_linear]
    X_test_linear = X_test_linear[feature_cols_linear]
    
    # Create validation split
    X_tr_linear, X_val_linear, y_tr_linear, y_val_linear = trainer.prepare_validation_split(
        X_train_linear, y_train
    )
    
    # Train Logistic Regression
    lr_model, lr_auc = trainer.train_logistic_regression(
        X_tr_linear, y_tr_linear, X_val_linear, y_val_linear, preprocessor_linear, False
    )
    
    # =========================================
    # STEP 7: Compare models
    # =========================================
    print("\n[STEP 7] Comparing all models...")
    comparison = trainer.compare_models(y_val)
    
    # =========================================
    # STEP 8: Generate predictions for best model
    # =========================================
    print("\n[STEP 8] Generating predictions for submission...")
    
    best_model_name = comparison.iloc[0]['Model']
    print(f"\nBest model: {best_model_name} with AUC: {comparison.iloc[0]['AUC']:.4f}")
    
    # Get predictions based on best model
    if best_model_name in ['xgboost', 'lightgbm', 'random_forest']:
        best_model = trainer.models[best_model_name]
        predictions = best_model.predict_proba(X_test_tree)[:, 1]
    else:  # logistic regression
        best_model = trainer.models[best_model_name]
        predictions = best_model.predict_proba(X_test_linear)[:, 1]
    
    # Create submission file
    submission = pd.DataFrame({
        'TX_ID': test_df['TX_ID'],
        'TX_FRAUD': predictions
    })
    
    submission.to_csv('fraud_predictions.csv', index=False)
    print(f"\nPredictions saved to fraud_predictions.csv")
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # =========================================
    # STEP 9: Feature importance for best tree model
    # =========================================
    if best_model_name in ['xgboost', 'lightgbm', 'random_forest']:
        print(f"\n[STEP 9] Top features for {best_model_name}...")
        trainer.plot_feature_importance(best_model_name, top_n=20)
    
    evaluation_df = evaluate_all_models(trainer, X_val, y_val)
    print(evaluation_df.to_string())
    
    return trainer, X_test_tree, X_test_linear, submission, test_df


# ===============================================
# ENSEMBLE APPROACH (OPTIONAL)
# ===============================================

def create_ensemble_predictions(trainer, X_test_tree, X_test_linear, test_df):
    """
    Create ensemble predictions by averaging multiple models
    """
    print("\n" + "="*60)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*60)
    
    predictions_dict = {}
    
    # Get predictions from each model
    for name, model in trainer.models.items():
        if name in ['xgboost', 'lightgbm', 'random_forest']:
            preds = model.predict_proba(X_test_tree)[:, 1]
        else:  # logistic regression
            preds = model.predict_proba(X_test_linear)[:, 1]
        
        predictions_dict[name] = preds
        print(f"{name}: mean={preds.mean():.4f}, std={preds.std():.4f}")
    
    # Weighted ensemble (weights based on validation AUC)
    weights = {name: results['auc'] for name, results in trainer.results.items()}
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    ensemble_preds = sum(predictions_dict[name] * weights[name] 
                        for name in predictions_dict)
    
    print(f"\nEnsemble: mean={ensemble_preds.mean():.4f}, std={ensemble_preds.std():.4f}")
    
    # Create submission
    ensemble_submission = pd.DataFrame({
        'TX_ID': test_df['TX_ID'],
        'TX_FRAUD': ensemble_preds
    })
    
    ensemble_submission.to_csv('fraud_predictions_ensemble.csv', index=False)
    print("Ensemble predictions saved to fraud_predictions_ensemble.csv")
    
    return ensemble_submission

def find_optimal_thresholds(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # F1-optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])  # Exclude last point
    best_f1_threshold = thresholds[best_f1_idx]
    
    # Business-optimal (e.g., maintain 90% recall)
    target_recall = 0.9
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        best_precision_idx = np.argmax(precision[:-1][valid_idx])
        business_threshold = thresholds[valid_idx[best_precision_idx]]
    else:
        business_threshold = best_f1_threshold
    
    return {
        'f1_optimal': best_f1_threshold,
        'f1_score': f1_scores[best_f1_idx],
        'business_optimal': business_threshold,
        'precision_at_90_recall': precision[:-1][valid_idx[best_precision_idx]] if len(valid_idx) > 0 else 0
    }

def calculate_key_metrics(y_true, y_pred_proba, threshold=0.5):

    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Precision-Recall at threshold
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Average Precision (area under PR curve)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'avg_precision': avg_precision
    }

def precision_at_recall_levels(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    results = {}
    for target_recall in [0.5, 0.7, 0.8, 0.9, 0.95]:
        idx = np.where(recall >= target_recall)[0]
        if len(idx) > 0:
            results[f'precision_at_{int(target_recall*100)}_recall'] = precision[idx[0]]
        else:
            results[f'precision_at_{int(target_recall*100)}_recall'] = 0
    
    return results

def top_k_metrics(y_true, y_pred_proba, k_values=[100, 500, 1000]):
    # Sort by predicted probability
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    results = {}
    for k in k_values:
        if k <= len(y_true):
            frauds_in_top_k = y_true_sorted[:k].sum()
            precision_at_k = frauds_in_top_k / k
            recall_at_k = frauds_in_top_k / y_true.sum()
            
            results[f'top_{k}_precision'] = precision_at_k
            results[f'top_{k}_recall'] = recall_at_k
            results[f'frauds_in_top_{k}'] = frauds_in_top_k
    
    return results

def fraud_business_metrics(y_true, y_pred_proba, threshold, avg_fraud_loss=100):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'true_positives': tp,  # Frauds caught
        'false_positives': fp,  # Good transactions blocked
        'false_negatives': fn,  # Frauds missed
        'true_negatives': tn,   # Good transactions passed
        
        # Business metrics
        'fraud_catch_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        
        # Money metrics
        'fraud_loss_prevented': tp * avg_fraud_loss,
        'fraud_loss_incurred': fn * avg_fraud_loss,
        'customer_friction_count': fp,
        
        # Operational metrics
        'alerts_per_100': (tp + fp) / len(y_true) * 100,
        'workload_reduction': 1 - (tp + fp) / len(y_true)
    }
    
    return metrics

def evaluate_all_models(trainer, X_val, y_val):
    results_df = []
    
    for model_name in trainer.results.keys():
        y_pred_proba = trainer.results[model_name]['predictions']
        
        # Find optimal threshold
        thresholds = find_optimal_thresholds(y_val, y_pred_proba)
        optimal_threshold = thresholds['f1_optimal']
        
        # Calculate metrics at optimal threshold
        metrics = fraud_business_metrics(y_val, y_pred_proba, optimal_threshold)
        pr_metrics = calculate_key_metrics(y_val, y_pred_proba, optimal_threshold)
        topk = top_k_metrics(y_val, y_pred_proba)
        
        results_df.append({
            'Model': model_name,
            'AUC': trainer.results[model_name]['auc'],
            'Avg_Precision': pr_metrics['avg_precision'],
            'Optimal_Threshold': optimal_threshold,
            'Precision': pr_metrics['precision'],
            'Recall': pr_metrics['recall'],
            'F1': pr_metrics['f1'],
            'Fraud_Catch_Rate': metrics['fraud_catch_rate'],
            'FPR': metrics['false_positive_rate'],
            'Alerts_per_100': metrics['alerts_per_100'],
            'Top_1000_Precision': topk['top_1000_precision']
        })
    
    return pd.DataFrame(results_df)
# ===============================================
# USAGE EXAMPLE
# ===============================================

if __name__ == "__main__":
    # Set your file path
    file_path = "../../bbd-payments-hackathon-2025/Payments Fraud DataSet/"
    
    # Run complete pipeline
    trainer, X_test_tree, X_test_linear, submission, test_df = run_complete_fraud_detection_pipeline(file_path)
    
    # Optional: Create ensemble predictions
    ensemble_submission = create_ensemble_predictions(
        trainer, X_test_tree, X_test_linear, test_df
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
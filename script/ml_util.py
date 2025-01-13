#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
from icecream import ic
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer, matthews_corrcoef
import json
import joblib

def ranksum_compare(data,class_name):
    from scipy.stats import ranksums
    from statsmodels.stats.multitest import multipletests
    r_p = data[class_name == 1]
    r_n = data[class_name == 0]
    pvalues = []
    mean_p = []
    mean_n = []
    cohen_d = []
    #print("try to do Wilcoxon rank-sum test...")
    for feature_name in data.columns:
        c_p = r_p.loc[:,feature_name].reset_index(drop=True)
        c_n = r_n.loc[:,feature_name].reset_index(drop=True)
        P_N_p_value = ranksums(c_p, c_n)[1]
        #calculating cohons_d
        m_p, m_n = np.mean(c_p), np.mean(c_n)
        s_p, s_n = np.std(c_p, ddof=1), np.std(c_n, ddof=1)
        #v_p, v_n = np.var(c_p, ddof=1), np.var(c_n, ddof=1)
        n_p, n_n = len(c_p), len(c_n)
        pooled_std = np.sqrt(((n_p - 1) * s_p ** 2 + (n_n - 1) * s_n ** 2) / (n_p + n_n - 2))
        pooled_std = pooled_std if pooled_std != 0 else 1
        #pooled_std = np.sqrt((v_p + v_n) / 2)
        cohen_d.append((m_p - m_n) / pooled_std)
        pvalues.append(P_N_p_value)
        mean_p.append(m_p)
        mean_n.append(m_n)
    P_N_correction = multipletests(pvalues, alpha=float(0.05), method='fdr_bh', is_sorted=False, returnsorted=False)
    #ranksum_df = pd.DataFrame({'feature': data.columns, 'mean_p': mean_p, 'mean_n': mean_n,'p_value': pvalues, 'fdr': P_N_correction[1], 'significant' : P_N_correction[0]})
    ranksum_df = pd.DataFrame({'Feature': data.columns, 'Mean_p': mean_p, 'Mean_n': mean_n, 'Cohens_d': cohen_d,'P_value': pvalues, 'FDR': P_N_correction[1], 'Significant': P_N_correction[0]})
    return ranksum_df.sort_values(by='FDR', ascending=True).reset_index(drop=True)

def filter_significant_features(X_df, significant_features, r_threshold = 0.7, filter=50):
    from scipy.stats import pearsonr
    filtered_features = set()
    non_k_mers_features = [fea for fea in significant_features if 'K_mers' not in fea]
    significant_feature_Kmers = [fea for fea in significant_features if 'K_mers' in fea]
    size_K_mers = len(significant_feature_Kmers)
    corr_matrix = X_df[significant_feature_Kmers].corr().abs()

    for i in range(size_K_mers):
        feature_1 = significant_feature_Kmers[i]
        if feature_1 not in filtered_features:
            for j in range(i + 1, size_K_mers):
                feature_2 = significant_feature_Kmers[j]
                if feature_2 not in filtered_features:
                    pearson_cor = corr_matrix.iloc[i, j]
                    if pearson_cor > r_threshold:
                        filtered_features.add(feature_2)

    #feature_keep = list(set(significant_feature_Kmers) - filtered_features)
    feature_keep = [feature for feature in significant_features if feature not in filtered_features]
    feature_keep = non_k_mers_features + feature_keep[:filter]
    return feature_keep


def find_best_threshold(y_true, y_probs):
    window_size = 9
    thresholds = np.arange(0, 1.01, 0.01)
    f1_scores = []

    # Calculate F1 scores for each threshold
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        f1_scores.append(score)

    # Apply moving average to smooth the F1 scores
    f1_scores = np.array(f1_scores)
    smoothed_f1_scores = np.convolve(f1_scores, np.ones(window_size)/window_size, mode='valid')

    # Adjust thresholds array to match the smoothed F1 scores length
    adjusted_thresholds = thresholds[:len(smoothed_f1_scores)]
    #ic(smoothed_f1_scores)
    # Find the threshold corresponding to the maximum smoothed F1 score
    best_idx = np.argmax(smoothed_f1_scores)
    best_threshold = adjusted_thresholds[best_idx]
    best_f1 = smoothed_f1_scores[best_idx]

    return best_threshold, best_f1


def ml_training_and_testing(X_df, y, model_name, model_dir):
    #save files to the model directory
    out = os.path.join(model_dir, f'Model_{model_name}')
    #out = f'Model_{model_name}'
    if not os.path.exists(out):
        os.makedirs(out)

    # Split data into training and test sets. Do sampling for training set
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
    #X_train_up, y_train_up = SMOTE(random_state=42).fit_resample(X_train, y_train)
    X_test_down, y_test_down = RandomUnderSampler(random_state=42).fit_resample(X_test, y_test)
    X_train_down, y_train_down = RandomUnderSampler().fit_resample(X_train, y_train)

    # Evaluate and select features
    evaluated_features_df = ranksum_compare(X_train_down, y_train_down)
    significant_features_df = evaluated_features_df[evaluated_features_df.Significant == True].copy()
    #ic(significant_features_df)
    significant_features = significant_features_df['Feature'].to_list()
    features_selected = filter_significant_features(X_df, significant_features, r_threshold = 0.7, filter = 50)
    significant_features_df.loc[:, 'Selected'] = significant_features_df['Feature'].isin(features_selected).astype(int)
    out_features_file = f'{out}/Important_features.tsv'
    significant_features_df.to_csv(out_features_file, sep='\t', index=False)

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(X_df[features_selected])
    out_scaler_file = f'{out}/Feature_scaler.pkl'
    joblib.dump(scaler, out_scaler_file)
    scaler_df = pd.DataFrame({
        'FeatureName': scaler.feature_names_in_,
        'Scale': scaler.scale_,
        'Min': scaler.min_
    })
    out_scaler_df = f'{out}/Feature_scaler_attributes.tsv'
    scaler_df.to_csv(out_scaler_df, sep='\t', index=False)

    X_train_scaled = scaler.transform(X_train[features_selected])
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features_selected)

    X_test_scaled = scaler.transform(X_test[features_selected])
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features_selected)

    X_test_down_scaled = scaler.transform(X_test_down[features_selected])
    X_test_down_scaled_df = pd.DataFrame(X_test_down_scaled, columns=features_selected)


    # Define classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=5000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, class_weight='balanced')
    }

    current_directory = os.path.dirname(os.path.abspath(__file__))
    ML_grids_param_file = f'{current_directory}/ML_grids_param.json'
    with open(ML_grids_param_file, 'r') as file:
        param_grids = json.load(file)
    
    # Perform grid search for each classifier
    best_classifiers = {}
    cv = StratifiedKFold(n_splits=5)
    scorers = {
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba'),
        'mcc': make_scorer(matthews_corrcoef)
    }

    results = []
    for name, clf in classifiers.items():
        #print(f'Processing {name}...')
        param_grid = param_grids[name]
        # Adjust param_grid to use the pipeline step name
        param_grid = {f'classifier__{key}': value for key, value in param_grid.items()}

        pipeline = Pipeline([
            #('smote', SMOTE(random_state=42)),
            ('down', RandomUnderSampler(random_state=42)),
            ('classifier', clf)
        ])
        # RandomUnderSampler(random_state=42) 
        #grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring=scorers, refit='f1', return_train_score=False, n_jobs=-1)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorers, refit='f1', return_train_score=False, n_jobs=-1)
        grid_search.fit(X_train_scaled_df, y_train)
        best_classifiers[name] = grid_search.best_estimator_

        # Save the best model
        model_filename = f'{out}/{name}_model.pkl'
        joblib.dump(best_classifiers[name], model_filename)

        #best_params = grid_search.best_params_
        best_params = {key.split('__')[1]: value for key, value in grid_search.best_params_.items()}
        mean_f1_cv = grid_search.cv_results_["mean_test_f1"][grid_search.best_index_]
        mean_rocauc_cv = grid_search.cv_results_["mean_test_roc_auc"][grid_search.best_index_]
        std_f1_cv = grid_search.cv_results_["std_test_f1"][grid_search.best_index_]
        std_rocauc_cv = grid_search.cv_results_["std_test_roc_auc"][grid_search.best_index_]
        mean_mcc_cv = grid_search.cv_results_["mean_test_mcc"][grid_search.best_index_]
        std_mcc_cv = grid_search.cv_results_["std_test_mcc"][grid_search.best_index_]
        print(f'Model_name: {model_name}')
        print(f'{name}: Best Parameters: {best_params}')
        print(f'  Mean F1 Score (CV): {mean_f1_cv:.4f}')
        print(f'  Mean AUC-ROC (CV): {mean_rocauc_cv:.4f}')
        print(f'  Best MCC (CV): {mean_mcc_cv:.4f}')


        # Evaluate on the test set
        y_test_pred = best_classifiers[name].predict(X_test_scaled_df)
        y_test_pred_proba = best_classifiers[name].predict_proba(X_test_scaled_df)[:, 1] if hasattr(best_classifiers[name], "predict_proba") else None
        f1_test = f1_score(y_test, y_test_pred)
        aucroc_test = roc_auc_score(y_test, y_test_pred_proba) if y_test_pred_proba is not None else 'N/A'
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        conf_matrix_test = confusion_matrix(y_test, y_test_pred)
        mean_prob_pos = np.mean(y_test_pred_proba[y_test_pred==1])
        mean_prob_neg = np.mean(y_test_pred_proba[y_test_pred==0])

        # Evaluate on balance test set
        y_test_down_pred = best_classifiers[name].predict(X_test_down_scaled_df)
        y_test_down_pred_proba = best_classifiers[name].predict_proba(X_test_down_scaled_df)[:, 1] if hasattr(best_classifiers[name], "predict_proba") else None

        f1_test_down_threshold, f1_test_down_bestF1 = find_best_threshold(y_test_down, y_test_down_pred_proba)
        #ic(f1_test_down_threshold, f1_test_down_bestF1)
        f1_test_down = f1_score(y_test_down, y_test_down_pred)
        aucroc_test_down = roc_auc_score(y_test_down, y_test_down_pred_proba) if y_test_down_pred_proba is not None else 'N/A'
        mcc_test_down = matthews_corrcoef(y_test_down, y_test_down_pred)
        conf_matrix_test_down = confusion_matrix(y_test_down, y_test_down_pred)
        accuracy_test_down = accuracy_score(y_test_down, y_test_down_pred)
        recall_test_down = recall_score(y_test_down, y_test_down_pred)
        precision_test_down = precision_score(y_test_down, y_test_down_pred)

        #calculate the random F1 and mcc
        random_f1s = []
        random_mccs = []
        for i in range(10):
            y_test_shuffled = np.random.permutation(y_test)
            random_f1 = f1_score(y_test, y_test_shuffled)
            random_f1s.append(random_f1)
            random_mcc = matthews_corrcoef(y_test, y_test_shuffled)
            random_mccs.append(random_mcc)
        
        print(f'{name} Model Evaluation:')
        #print(f'  F1: {f1_test:.4f}')
        #print(f'  AUC-ROC: {aucroc_test:.4f}')
        #print(f'  MCC: {mcc_test:.4f}')
        #print(f'  Random F1: {np.mean(random_f1s):.4f}')
        #print(f'  Random MCC: {np.mean(random_mccs):.4f}')
        print(f'  Balanced F1: {f1_test_down:.4f}')
        print(f'  Balanced AUC-ROC: {aucroc_test_down:.4f}')
        print(f'  Balanced MCC: {mcc_test_down:.4f}')
        #print(f'  Mean_proba_pos: {mean_prob_pos:.4f}')
        #print(f'  Mean_proba_neg: {mean_prob_neg:.4f}')

        print(f'  Confusion Matrix:\n{conf_matrix_test}\n')

        # Extract the classifier from the pipeline
        best_clf = best_classifiers[name].named_steps['classifier']
        # Get feature importances or coefficients
        if hasattr(best_clf, 'feature_importances_'):
            importances = best_clf.feature_importances_
        elif hasattr(best_clf, 'coef_'):
            importances = np.abs(best_clf.coef_[0])
        else:
            importances = []

        importances_df = pd.DataFrame({"Feature": features_selected, "Importances": importances})
        out_imp = f'{out}/{name}_model_importances.tsv'
        importances_df.to_csv(out_imp, sep='\t', index=False)

        result = {
            "Model": model_name,
            "Classifier": name,
            "BestParameters": best_params,
            "MeanF1": round(mean_f1_cv,4),
            "StdF1": round(std_f1_cv,4),
            "MeanAUCROC": round(mean_rocauc_cv,4),
            "StdAUCROC": round(std_rocauc_cv,4),
            "MeanMCC": round(mean_mcc_cv,4),
            "StdMCC": round(std_mcc_cv,4),
            "TestF1": round(f1_test,4),
            "TestAUCROC": round(aucroc_test,4),
            "TestMCC": round(mcc_test,4),
            "TestConfusionMatrix": f"{conf_matrix_test[0][0]} {conf_matrix_test[0][1]} {conf_matrix_test[1][0]} {conf_matrix_test[1][1]}",
            "Mean_proba_pos": round(mean_prob_pos,4),
            "Mean_proba_neg": round(mean_prob_neg,4),
            "RandomF1": round(np.mean(random_f1s),4),
            "RandomMCC": round(np.mean(random_mccs),4),
            "BalancedF1": round(f1_test_down,4),
            "BalancedAUCROC": round(aucroc_test_down,4),
            "BalancedMCC": round(mcc_test_down,4),
            "BalancedAccuracy": round(accuracy_test_down,4),
            "BalancedRecall": round(recall_test_down,4),
            "BalancedPrecision": round(precision_test_down,4),
            "Threshold": round(f1_test_down_threshold,4),
            "BalancedConfusionMatrix": f"{conf_matrix_test_down[0][0]} {conf_matrix_test_down[0][1]} {conf_matrix_test_down[1][0]} {conf_matrix_test_down[1][1]}"
        }
        results.append(result)

    out_summary = f'{out}/Model_evaluation_results.tsv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_summary, sep='\t', index=False)

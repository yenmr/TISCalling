#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import argparse
import ml_util
from icecream import ic
#import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, make_scorer, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler


def main(args):

    features_file_path = args.feature
    model_file_path = args.models
    features_file = os.path.basename(features_file_path)
    feature_dir = "feature"
    
    features_file_path = os.path.join(feature_dir, features_file_path)
    query_name = features_file.replace('.txt', '').replace('Features_', '')
    features_ori = pd.read_csv(features_file_path, sep = '\t')

    model_dir = "model"
    model_paths = model_file_path.split(",")
    ic(model_paths)

    results = []

    for model_path in model_paths:
        features = features_ori.copy()
        #model_path = 'Model_Sl_CDSATG'
        model_name = model_path.replace('Model_', '')

        sp = model_name.split('_')[0]
        clf_names = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'SVM']
        #model_name = features_file.replace('.txt', '').replace('Features_', '')

        # Remove all columns with PWM except PWM_keep and rename to PWM
        PWM_keep = f'PWM_{model_name}'
        pwm_columns = [col for col in features.columns if 'PWM' in col]
        if PWM_keep in pwm_columns:
            columns_to_drop = [col for col in pwm_columns if col != PWM_keep]
            features = features.drop(columns=columns_to_drop)
            features = features.rename(columns={PWM_keep: 'PWM'})

        # Remove all comumns name with TIS_codon except TIS_codon_keep and rename to TIS_codon 
        TIS_codon_keep = f'TIS_codon_{model_name}'
        TIS_codon_columns = [col for col in features.columns if 'TIS_codon' in col]
        if PWM_keep in pwm_columns:
            columns_to_drop = [col for col in TIS_codon_columns if col != TIS_codon_keep]
            features = features.drop(columns=columns_to_drop)
            features = features.rename(columns={TIS_codon_keep: 'TIS_codon'})    

        # Decode tag and drop tag from features table
        tags = features['ID'].str.split('|', expand=True)
        tags.columns = ['Gene', 'Pos','Tag' ,'y']
        tags['y'] = tags['y'].astype(int)
        X_df = features.drop(columns='ID')
        y = np.array(tags['y'].tolist())

        # Select features for testing
        #target_scaler = f'{model_path}/Feature_scaler.pkl'
        target_scaler = os.path.join(model_dir, f'{model_path}/Feature_scaler.pkl')
        scaler = joblib.load(target_scaler)
        selected_features_name = scaler.get_feature_names_out().tolist()
        X_df_selected = X_df[selected_features_name]
        X_scaled = scaler.transform(X_df_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features_name)
        X_test, y_test = X_scaled_df, y
        #X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
        #X_train_res, y_train_res = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
        X_test_res, y_test_res = RandomUnderSampler(random_state=42).fit_resample(X_test, y_test)
        

        for clf_name in clf_names:

            #model_filename = f'{model_path}/{clf_name}_model.pkl'
            model_filename = os.path.join(model_dir, f'{model_path}/{clf_name}_model.pkl')
            best_classifier = joblib.load(model_filename)

            # Evaluate on the test set
            y_test_pred = best_classifier.predict(X_test)
            y_test_pred_proba = best_classifier.predict_proba(X_test)[:, 1] if hasattr(best_classifier, "predict_proba") else None

            f1_test = f1_score(y_test, y_test_pred)
            aucroc_test = roc_auc_score(y_test, y_test_pred_proba) if y_test_pred_proba is not None else 'N/A'
            mcc_test = matthews_corrcoef(y_test, y_test_pred)
            conf_matrix_test = confusion_matrix(y_test, y_test_pred)

            y_test_res_pred = best_classifier.predict(X_test_res)
            y_test_res_pred_proba = best_classifier.predict_proba(X_test_res)[:, 1] if hasattr(best_classifier, "predict_proba") else None
            f1_test_res = f1_score(y_test_res, y_test_res_pred)
            aucroc_test_res = roc_auc_score(y_test_res, y_test_res_pred_proba) if y_test_res_pred_proba is not None else 'N/A'
            mcc_test_res = matthews_corrcoef(y_test_res, y_test_res_pred)
            accuracy_test_res = accuracy_score(y_test_res, y_test_res_pred)
            recall_test_res = recall_score(y_test_res, y_test_res_pred)
            precision_test_res = precision_score(y_test_res, y_test_res_pred)

            conf_matrix_test_res = confusion_matrix(y_test_res, y_test_res_pred)

            #understnad the random F1 and mcc
            random_f1s = []
            random_mccs = []
            for i in range(10):
                y_test_shuffled = np.random.permutation(y_test)
                random_f1 = f1_score(y_test, y_test_shuffled)
                random_f1s.append(random_f1)
                random_mcc = matthews_corrcoef(y_test, y_test_shuffled)
                random_mccs.append(random_mcc)
            print(f'{query_name}')
            print(f'{model_name}')
            print(f'{clf_name} Test Set Evaluation:')
            print(f'  F1: {f1_test:.4f}')
            print(f'  AUC-ROC: {aucroc_test:.4f}')
            print(f'  MCC: {mcc_test:.4f}')
            print(f'  Random F1: {np.mean(random_f1s):.4f}')
            print(f'  Random MCC: {np.mean(random_mccs):.4f}')
            print(f'  Confusion Matrix:\n{conf_matrix_test}')
            print(f'  Balanced F1: {f1_test_res:.4f}')
            print(f'  Balanced AUC-ROC: {aucroc_test_res:.4f}')
            print(f'  Balanced MCC: {mcc_test_res:.4f}')
            print(f'  Confusion Matrix:\n{conf_matrix_test_res}\n')
            result = {
                "Query": query_name,
                "Model": model_name,
                "Classifier": clf_name,
                #"MeanF1Score": round(mean_f1_cv,4),
                #"StdF1Score": round(std_f1_cv,4),
                #"MeanAUC-ROC": round(mean_rocauc_cv,4),
                #"StdAUC-ROC": round(std_rocauc_cv,4),
                #"MeanMCC": round(mean_mcc_cv,4),
                #"StdMCC": round(std_mcc_cv,4),
                "TestF1": round(f1_test,4),
                "TestAUCROC": round(aucroc_test,4),
                "TestMCC": round(mcc_test,4),
                "ConfusionMatrix": f"{conf_matrix_test[0][0]} {conf_matrix_test[0][1]} {conf_matrix_test[1][0]} {conf_matrix_test[1][1]}",
                "RandomF1": round(np.mean(random_f1s),4),
                "RandomMCC": round(np.mean(random_mccs),4),
                "BalancedF1": round(f1_test_res,4),
                "BalancedAUCROC": round(aucroc_test_res,4),
                "BalancedMCC": round(mcc_test_res,4),
                "BalancedAccuracy": round(accuracy_test_res,4),
                "BalancedRecall": round(recall_test_res,4),
                "BalancedPrecision": round(precision_test_res,4),
                "BalancedConfusionMatrix": f"{conf_matrix_test_res[0][0]} {conf_matrix_test_res[0][1]} {conf_matrix_test_res[1][0]} {conf_matrix_test_res[1][1]}"
            }
            results.append(result)

    out_summary = f'Cross_evaluation_result_{query_name}.tsv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_summary, sep='\t', index=False)


    #ml_util.ml_training_and_testing(X_df, y, model_name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prepare input features table for learning model.')
    parser.add_argument('-feature', required=True, help='Path to the feature table file.')
    parser.add_argument('-models', required=True, help='Path to the model path. Separate by comma ",".')
    args = parser.parse_args()
    main(args)


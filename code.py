
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from model.models import Neural_Network_FeedForward_Regularizer
from model.metric import *
from model.utils import delay_matrix_features, delay_matrix, concatenate_data
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# GPU Configs
def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU available, using CPU")

configure_gpus()

# Cons
RANDOM_SEED_BASE = 2308
TF_DATATYPE = 'float64'
NP_DATATYPE = np.float64
DATA_PATH = "/******************************/New_Features_Data"
FOLDERSAVE = '/**********************************/ensemble_NoFix'
WINDOW_MAVG = 10
TEST_FILES = 25
EPOCHS = 10
LEARNING_RATE = 1e-4

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': ***,
    'max_depth': *,
    'alpha': ***,
    'colsample_bytree': ********,
    'learning_rate': *********
}

# 
feature_logic = ['BlsAsw', 'Strategy_Hydraulic_FL', 'Strategy_Hydraulic_FR', 'Strategy_Rfp_ABS', 'PW', 'state']
feature_for_mavg = ['sPushrod', 'BlsAsw', 'Strategy_PumpUnit_N_Min', 'mueF', 'axF', 'EvRR_Current', 'EvRL_Current',
                    'EvFL_Current', 'EvFR_Current', 'Strategy_Hydraulic_RR', 'Strategy_Hydraulic_RL', 'Strategy_Hydraulic_FL', 
                    'Strategy_Hydraulic_FR', 'Strategy_Rfp_ABS', 'PW', 'state', 'vF_avg', 'MotRpm', 'SasInCor', 'ThrottlePos', 
                    'vPushrod', 'MProp', 'Vol_Acc_BC1', 'Vol_Acc_BC2', 'vGiF', 'PressSent1X1']

input_feature_with_delays = ['BlsAsw_MAVG', 'sPushrod_MAVG', 'mueF_MAVG', 'PW_MAVG', 'ThrottlePos_MAVG', 'axF_MAVG',
                             'vPushrod_MAVG', 'MotRpm_MAVG', 'vF_avg_MAVG', 'SasInCor_MAVG', 'MProp_MAVG', 'vGiF_MAVG']

feature_target = ['PressSent1X1_MAVG']
feature_mavg_cols = [f'{col}_MAVG' for col in feature_for_mavg]
input_feature_without_delays = [val for val in feature_mavg_cols if val not in input_feature_with_delays + feature_target]

# Adding XGBoost preds to input feat
input_feature_with_delays.append('XGBoost_Prediction')

#
def moving_avg_window_func(data, window_mavg):
    temp_z = np.zeros(window_mavg-1)
    temp_n = np.concatenate((temp_z, data.flatten()))
    data_window = [temp_n[i:i+window_mavg].mean() for i in range(len(temp_n) - window_mavg+1)]
    return np.array(data_window)

def scale_function(x, x_mean, x_std):
    return (x - x_mean) / x_std

def rescale_function(x, x_mean, x_std):
    return x * x_std + x_mean

def Threshold_R2_SCORE(r2_score):
    return 10.0 * max([r2_score - 0.9, 0.0])

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_SEED_BASE)
    model.fit(X_train, y_train)
    return model

def get_xgboost_predictions(model, X):
    return model.predict(X).reshape(-1, 1)

# Main func
def process_fold(fold_no):
    tf.keras.backend.clear_session()
    
    fold_name = f'fold_{fold_no}'
    path_save_results = os.path.join(FOLDERSAVE.strip(), fold_name.strip())
    os.makedirs(path_save_results, exist_ok=True)
    
    RANDOM_SEED = RANDOM_SEED_BASE + fold_no
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load and preprocess data
    files_list = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]
    idx_files = np.random.choice(len(files_list), len(files_list), replace=False)
    files_list_shuffle = [files_list[val] for val in idx_files]
    
    df_train, df_test, train_filenames, test_filenames = load_and_preprocess_data(files_list_shuffle)
    
    # features for XGBoost
    X_train_xgb, y_train_xgb, X_scaler, y_scaler = prepare_features_for_xgboost(df_train)
    
    # Train XGBoost 
    xgb_model = train_xgboost_model(X_train_xgb, y_train_xgb)
    
    # XGBoost predictions for train and test data
    df_train = add_xgboost_predictions(df_train, xgb_model, X_scaler)
    df_test = add_xgboost_predictions(df_test, xgb_model, X_scaler)
    
    # features for ANN
    input_train, target_train, input_mean, input_std, output_mean, output_std = prepare_features_and_target(df_train)
    
    # Train ANN
    model = train_model(input_train, target_train, input_mean, input_std)
    
    # 
    r2_score_train, rmse_train, mae_train, max_abs_error_train = evaluate_data(df_train, train_filenames, model, input_mean, input_std, output_mean, output_std, 'Train', path_save_results)
    r2_score_test, rmse_test, mae_test, max_abs_error_test = evaluate_data(df_test, test_filenames, model, input_mean, input_std, output_mean, output_std, 'Test', path_save_results)
    
    # Save results
    save_results(path_save_results, fold_name, train_filenames, test_filenames, 
                 r2_score_train, rmse_train, mae_train, max_abs_error_train,
                 r2_score_test, rmse_test, mae_test, max_abs_error_test)
    
    return test_filenames, r2_score_test

def load_and_preprocess_data(files_list_shuffle):
    always_train_files = [

   **********
  **********
  ********


    ]

    df_train, df_test = [], []
    train_filenames, test_filenames = [], []
    
    other_files = [f for f in files_list_shuffle if f not in always_train_files]
    np.random.shuffle(other_files)

    total_files = len(files_list_shuffle)
    additional_train_files = total_files - TEST_FILES - len(always_train_files)

    train_files = always_train_files + other_files[:additional_train_files]
    test_files = other_files[additional_train_files:]

    for filename in train_files + test_files:
        fileload = os.path.join(DATA_PATH, filename)
        df = pd.read_csv(fileload).filter(['TIME','sPushrod', 'BlsAsw', 'Strategy_PumpUnit_N_Min', 'mueF', 'axF', 'EvRR_Current', 'EvRL_Current',
                    'EvFL_Current', 'EvFR_Current', 'Strategy_Hydraulic_RR', 'Strategy_Hydraulic_RL', 'Strategy_Hydraulic_FL', 
                    'Strategy_Hydraulic_FR', 'Strategy_Rfp_ABS', 'PW', 'state', 'MotRpm', 'SasInCor', 'ThrottlePos', 
                    'vPushrod', 'MProp', 'Vol_Acc_BC1', 'Vol_Acc_BC2', 'vGiF', 'PressSent1X1'])
        vf = pd.read_csv(fileload).filter(['vF_FL','vF_FR','vF_RL','vF_RR'])
        df['vF_avg'] = 0.25 * (vf['vF_FL'] + vf['vF_FR'] + vf['vF_RL'] + vf['vF_RR'])
        
        for col in feature_for_mavg:
            col_name = f'{col}_MAVG'
            window_mavg_val = 1 if col in feature_logic else WINDOW_MAVG
            df[col_name] = moving_avg_window_func(df[col].values.flatten(), window_mavg_val)
        
        if filename in train_files:
            df_train.append(df)
            train_filenames.append(filename)
        else:
            df_test.append(df)
            test_filenames.append(filename)
    
    return df_train, df_test, train_filenames, test_filenames

def prepare_features_for_xgboost(df_data):
    X = pd.concat([df[input_feature_with_delays[:-1]] for df in df_data])  # Exclude 'XGBoost_Prediction'
    y = pd.concat([df[feature_target] for df in df_data])
    
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    return X_scaled, y_scaled, X_scaler, y_scaler

def add_xgboost_predictions(df_list, xgb_model, X_scaler):
    for df in df_list:
        X = df[input_feature_with_delays[:-1]]  # Exclude 'XGBoost_Prediction'
        X_scaled = X_scaler.transform(X)
        xgb_predictions = get_xgboost_predictions(xgb_model, X_scaled)
        df['XGBoost_Prediction'] = xgb_predictions
    return df_list

def prepare_features_and_target(df_train):
    delay_win = 25
    input_train_dim = len(input_feature_with_delays) * (delay_win + 1) + len(input_feature_without_delays)
    
    input_train = np.array([]).reshape(0, input_train_dim)
    target_train = np.array([]).reshape(0, 1)
    
    for data in df_train:
        features_delays_values = data[input_feature_with_delays].values
        features_non_delays_values = data[input_feature_without_delays].values
        target = data[feature_target].values
        
        delay_input_features = delay_matrix_features(features_delays_values, delay_win)
        non_delay_input_features = features_non_delays_values[delay_win:, :]
        
        delay_target_mat = delay_matrix(target, delay_win)
        target_value = delay_target_mat[:, -1:]
        
        input_concat = np.concatenate((non_delay_input_features, delay_input_features), axis=1)
        input_train = np.concatenate((input_train, input_concat), axis=0)
        target_train = np.concatenate((target_train, target_value), axis=0)
    
    input_mean, input_std = calculate_mean_std(df_train, input_feature_with_delays, input_feature_without_delays, delay_win)
    output_mean = target_train.mean()
    output_std = target_train.std()
    
    input_train_scaled = scale_function(input_train, input_mean, input_std)
    target_train_scaled = scale_function(target_train, output_mean, output_std)
    
    return input_train_scaled, target_train_scaled, input_mean, input_std, output_mean, output_std

def calculate_mean_std(df_train, input_feature_with_delays, input_feature_without_delays, delay_win):
    features_delays_values_all = np.array([]).reshape(0, len(input_feature_with_delays))
    features_non_delays_values_all = np.array([]).reshape(0, len(input_feature_without_delays))
    
    for data in df_train:
        features_delays_values = data[input_feature_with_delays].values
        features_non_delays_values = data[input_feature_without_delays].values
        
        features_delays_values_all = np.concatenate((features_delays_values_all, features_delays_values), axis=0)
        features_non_delays_values_all = np.concatenate((features_non_delays_values_all, features_non_delays_values), axis=0)
    
    features_delays_values_mean = features_delays_values_all.mean(0)
    features_delays_values_std = features_delays_values_all.std(0)
    features_non_delays_values_mean = features_non_delays_values_all.mean(0)
    features_non_delays_values_std = features_non_delays_values_all.std(0)
    
    input_mean = np.array(list(features_non_delays_values_mean) + list(np.tile(features_delays_values_mean, (delay_win + 1, 1)).T.flatten()))
    input_std = np.array(list(features_non_delays_values_std) + list(np.tile(features_delays_values_std, (delay_win + 1, 1)).T.flatten()))
    
    return input_mean, input_std

def train_model(input_train, target_train, input_mean, input_std):
    bsz_data = **
    layers = [input_train.shape[1], ***,***,***,**]
    input_val_scaled = None #scale_function(input_val,input_mean,input_std)
    target_val_scaled = None #scale_function(target_val,output_mean,output_std)

    layers[0] = input_train.shape[1]
    
    
    model = Neural_Network_FeedForward_Regularizer(
        input_train, target_train, bsz_data, 
        input_val_scaled, 
        target_val_scaled, 
        bsz_data_val=*,
        validate=False, stab_epsilon=**, shuffle=True, 
        layers=layers, lr_rate=LEARNING_RATE, save_step=1, tf_datatype=TF_DATATYPE
    )
    
    model.ADAM_optimizer.learning_rate.assign(LEARNING_RATE)
    model.train_model(EPOCHS)
    
    return model

def evaluate_data(df_data, filenames, model, input_mean, input_std, output_mean, output_std, data_type, path_save_results):
    path_save_results_data = os.path.join(path_save_results, f'{data_type}_predictions')
    os.makedirs(path_save_results_data, exist_ok=True)
    
    r2_scores, rmses, maes, max_abs_errors = [], [], [], []
    
    delay_win = **
    
    for data, filename in zip(df_data, filenames):
        features_delays_values = data[input_feature_with_delays].values
        features_non_delays_values = data[input_feature_without_delays].values
        target = data[feature_target].values
        
        delay_input_features = delay_matrix_features(features_delays_values, delay_win)
        non_delay_input_features = features_non_delays_values[delay_win:, :]
        
        input_model = concatenate_data(non_delay_input_features, delay_input_features)
        input_model_scaled = scale_function(input_model, input_mean, input_std)
        model_pred = model.predict(input_model_scaled)
        target_pred = rescale_function(model_pred, output_mean, output_std)
        
        target_true = target[delay_win:]
        time = data['TIME'].values[delay_win:]
        
        condition_brk = (data['BlsAsw'] == ***) & (data['sPushrod'] > ***)
        time_break = np.zeros(len(data['TIME']))
        time_break[condition_brk] = 1
        
        target_true_brk = target_true.flatten()[condition_brk[delay_win:]]
        target_pred_brk = target_pred.flatten()[condition_brk[delay_win:]]
        
        if sum(condition_brk) != 0:
            r2_score = R2_SCORE(target_true_brk, target_pred_brk)
            rmse = RMSE(target_true_brk, target_pred_brk)
            mae = MAE(target_true_brk, target_pred_brk)
            max_abs_error = MAX_ABSOLUTE_ERROR(target_true_brk, target_pred_brk)
        else:
            r2_score = rmse = mae = max_abs_error = np.nan
        
        r2_scores.append(r2_score)
        rmses.append(rmse)
        maes.append(mae)
        max_abs_errors.append(max_abs_error)
        
        # Plots
        plt.figure()
        plt.plot(data['TIME'], time_break * max(np.abs(target_true).max(), np.abs(target_pred).max()), linestyle='--', c='k')
        plt.title(f"{filename[:-4]} [ R2 : {round(r2_score, 3)} ]")
        plt.plot(time, target_true.flatten(), label='True')
        plt.plot(time, target_pred.flatten(), label='Pred')
        plt.xlabel('t')
        plt.ylabel(feature_target[0])
        plt.legend()
        plt.savefig(os.path.join(path_save_results_data, f"{filename[:-4]}_prediction.png"))
        plt.close()
    
    return r2_scores, rmses, maes, max_abs_errors

def save_results(path_save_results, fold_name, train_filenames, test_filenames, 
                 r2_score_train, rmse_train, mae_train, max_abs_error_train,
                 r2_score_test, rmse_test, mae_test, max_abs_error_test):
    
    # 
    file_create_name = os.path.join(path_save_results, f'file_infos_{fold_name}.txt')
    with open(file_create_name, "w") as file_create:
        file_create.write(f"Train files: {len(train_filenames)}\n")
        for filename in train_filenames:
            file_create.write(f"{filename}\n")
        file_create.write('-'*100 + "\n")
        file_create.write(f"Test files: {len(test_filenames)}\n")
        for filename in test_filenames:
            file_create.write(f"{filename}\n")
        file_create.write('-'*100 + "\n")
    
    # test results
    df_results_test = pd.DataFrame({
        'Filename': test_filenames,
        'R2': r2_score_test,
        'RMSE': rmse_test,
        'MAE': mae_test,
        'MAX ABS. ERROR': max_abs_error_test
    })
    df_results_test.to_csv(os.path.join(path_save_results, f'test_results_{fold_name}.csv'), index=False)
    
    # train results
    df_results_train = pd.DataFrame({
        'Filename': train_filenames,
        'R2': r2_score_train,
        'RMSE': rmse_train,
        'MAE': mae_train,
        'MAX ABS. ERROR': max_abs_error_train
    })
    df_results_train.to_csv(os.path.join(path_save_results, f'train_results_{fold_name}.csv'), index=False)

def generate_final_report(all_test_filenames, all_r2_scores):
    # 
    file_scores = {filename: [np.nan] * 100 for filename in set(sum(all_test_filenames, []))}
    
    #
    for fold, (filenames, scores) in enumerate(zip(all_test_filenames, all_r2_scores), 1):
        for filename, score in zip(filenames, scores):
            file_scores[filename][fold-1] = score
    
    # 
    df_final = pd.DataFrame.from_dict(file_scores, orient='index', 
                                      columns=[f'fold_{i}' for i in range(1,101 )])
    df_final.index.name = 'test_filename'
    
    # 
    df_final['mean_r2'] = df_final.mean(axis=1)
    df_final = df_final.sort_values('mean_r2', ascending=False)
    df_final = df_final.drop('mean_r2', axis=1)
    
    # final report
    df_final.to_csv(os.path.join(FOLDERSAVE, 'final_report_assemble_************.csv'))
    print("Final report generated and saved as 'final_report.csv'")

# Main execution
if __name__ == "__main__":
    all_test_filenames = []
    all_r2_scores = []
    
    for fold_no in range(1, 101):  # 
        print(f"Processing fold {fold_no}")
        test_filenames, r2_scores = process_fold(fold_no)
        all_test_filenames.append(test_filenames)
        all_r2_scores.append(r2_scores)
    
    generate_final_report(all_test_filenames, all_r2_scores)


import numpy as np
import pandas as pd
import os

def calculate_metrics(y_true, y_pred):
    """
    Tính toán 4 chỉ số đánh giá: MAE, RMSE, MAPE, sMAPE.
    
    Tham số:
    y_true (array-like): Mảng chứa các giá trị thực tế.
    y_pred (array-like): Mảng chứa các giá trị dự đoán từ mô hình.
    
    Trả về:
    dict: Dictionary chứa kết quả của 4 chỉ số.
    """
    # Chuyển sang Numpy array mà không ép cứng kiểu dữ liệu
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Hằng số
    epsilon = 1e-10
    
    # 1. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 2. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # 3. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # 4. sMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'sMAPE': float(smape)
    }
def evaluate_and_save_metrics(y_true, y_pred, output_path='metrics.csv'):
    """
    Tính toán 4 chỉ số và lưu kết quả vào file CSV.
    
    Tham số:
    y_true (array-like): Mảng chứa các giá trị thực tế.
    y_pred (array-like): Mảng chứa các giá trị dự đoán.
    output_path (str): Đường dẫn lưu file CSV (mặc định: 'metrics.csv').
    """
    metrics_dict = calculate_metrics(y_true, y_pred)
    
    df_metrics = pd.DataFrame([metrics_dict])
    df_metrics.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[INFO] Đã tính toán xong và lưu 4 chỉ số đánh giá vào: {output_path}")
    
    return metrics_dict

# Khối code chạy thử nghiệm
if __name__ == "__main__":
    y_thuc_te = [100, 150, 200, 250, 300]
    y_du_doan = [105, 145, 190, 260, 295]
    
    ket_qua = evaluate_and_save_metrics(y_thuc_te, y_du_doan)
    print("Kết quả chi tiết:", ket_qua)

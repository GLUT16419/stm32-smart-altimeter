#!/usr/bin/env python
#
# Neural Network Training for Pressure Filtering
# Supports: Keras, TFLite, ONNX export
#

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import onnxruntime as ort


class NeuralNetworkTrainer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
    
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df['pressure_pa'].values
    
    def create_dataset(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(np.median(data[i-self.window_size:i+1]))
        return np.array(X), np.array(y)
    
    def preprocess_data(self, data):
        data = data.reshape(-1, 1)
        self.scaler.fit(data)
        scaled_data = self.scaler.transform(data).flatten()
        return scaled_data
    
    def build_model(self, model_type='bp', input_dim=5):
        if model_type == 'bp':
            self.model = Sequential([
                Dense(16, activation='relu', input_shape=(input_dim,)),
                Dense(8, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'bp_large':
            self.model = Sequential([
                Dense(32, activation='relu', input_shape=(input_dim,)),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'cnn':
            self.model = Sequential([
                Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'lstm':
            self.model = Sequential([
                LSTM(16, return_sequences=False, input_shape=(input_dim, 1)),
                Dense(8, activation='relu'),
                Dense(1, activation='linear')
            ])
        elif model_type == 'gru':
            self.model = Sequential([
                GRU(16, return_sequences=False, input_shape=(input_dim, 1)),
                Dense(8, activation='relu'),
                Dense(1, activation='linear')
            ])
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test, verbose=0)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_unscaled = mae * (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        rmse_unscaled = rmse * (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        
        print(f"Test Loss: {loss:.6f}")
        print(f"Test MAE: {mae:.6f} (scaled)")
        print(f"Test RMSE: {rmse:.6f} (scaled)")
        print(f"Test MAE: {mae_unscaled:.2f} Pa (unscaled)")
        print(f"Test RMSE: {rmse_unscaled:.2f} Pa (unscaled)")
        
        return y_pred
    
    def predict(self, input_data):
        input_scaled = self.scaler.transform(input_data.reshape(-1, 1)).flatten()
        predictions = []
        
        for i in range(self.window_size, len(input_scaled)):
            X = input_scaled[i-self.window_size:i].reshape(1, -1)
            y_pred_scaled = self.model.predict(X, verbose=0)[0][0]
            predictions.append(y_pred_scaled)
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_unscaled = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions_unscaled
    
    def plot_results(self, y_test, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='True Value', alpha=0.5)
        plt.plot(y_pred, label='Predicted Value', linewidth=2)
        plt.title('Neural Network Filtering Results')
        plt.xlabel('Sample Index')
        plt.ylabel('Pressure (Pa)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_models(self, model_name='pressure_filter'):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        keras_path = os.path.join(models_dir, f"{model_name}.h5")
        tflite_path = os.path.join(models_dir, f"{model_name}.tflite")
        onnx_path = os.path.join(models_dir, f"{model_name}.onnx")
        
        self.model.save(keras_path)
        print(f"Keras model saved to: {keras_path}")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        # 若 LSTM/GRU 转换失败，可尝试启用：
        # converter.target_spec.supported_ops = [
        #     tf.lite.OpsSet.TFLITE_BUILTINS,
        #     tf.lite.OpsSet.SELECT_TF_OPS
        # ]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_path}")
        
        tf.saved_model.save(self.model, "temp_model")
        os.system(f"python -m tf2onnx.convert --saved-model temp_model --output {onnx_path}")
        import shutil
        shutil.rmtree("temp_model", ignore_errors=True)
        print(f"ONNX model saved to: {onnx_path}")
        
        scaler_info = {
            'min': float(self.scaler.data_min_[0]),
            'max': float(self.scaler.data_max_[0]),
            'range': float(self.scaler.data_max_[0] - self.scaler.data_min_[0])
        }
        
        import json
        scaler_path = os.path.join(models_dir, f"{model_name}_scaler.json")
        with open(scaler_path, 'w') as f:
            json.dump(scaler_info, f, indent=2)
        print(f"Scaler info saved to: {scaler_path}")
        
        print("\nAll models exported successfully!")


def main():
    print("=== Neural Network Training for Pressure Filtering ===\n")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please run data_collector.py first to collect data.")
        return
    
    ms5611_files = [f for f in os.listdir(data_dir) if f.startswith('ms5611_')]
    bmp280_files = [f for f in os.listdir(data_dir) if f.startswith('bmp280_')]
    
    print(f"Found {len(ms5611_files)} MS5611 data files")
    print(f"Found {len(bmp280_files)} BMP280 data files")
    
    sensor_choice = input("\nChoose sensor to train (1=MS5611, 2=BMP280): ")
    if sensor_choice == '1':
        files = ms5611_files
        sensor_name = 'MS5611'
    elif sensor_choice == '2':
        files = bmp280_files
        sensor_name = 'BMP280'
    else:
        print("Invalid choice")
        return
    
    if not files:
        print(f"No {sensor_name} data files found")
        return
    
    print(f"\nAvailable {sensor_name} files:")
    for i, f in enumerate(files):
        print(f"  {i+1}. {f}")
    
    file_choice = int(input("Enter file number: ")) - 1
    if 0 <= file_choice < len(files):
        file_path = os.path.join(data_dir, files[file_choice])
    else:
        print("Invalid choice")
        return
    
    model_type = 'bp'
    
    print(f"\nLoading data from {file_path}...")
    
    trainer = NeuralNetworkTrainer(window_size=5)
    
    raw_data = trainer.load_data(file_path)
    print(f"Loaded {len(raw_data)} samples")
    
    scaled_data = trainer.preprocess_data(raw_data)
    X, y = trainer.create_dataset(scaled_data)
    
    print(f"Created {len(X)} training samples")
    
    if model_type in ['cnn', 'lstm', 'gru']:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print(f"\nBuilding {model_type} model...")
    trainer.build_model(model_type=model_type, input_dim=trainer.window_size)
    trainer.model.summary()
    
    print("\nTraining...")
    trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    print("\nEvaluating...")
    y_pred = trainer.evaluate(X_test, y_test)
    
    print("\nExporting models (Keras, TFLite, ONNX)...")
    model_name = f"{sensor_name.lower()}_{model_type}_filter"
    trainer.export_models(model_name=model_name)
    
    print("\nPlotting results...")
    trainer.plot_training_history()
    trainer.plot_results(y_test, y_pred)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

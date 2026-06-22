import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense


# ======================================================
# GLOBALS
# ======================================================
df = None
model = None
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


# ======================================================
# LOAD DATASET
# ======================================================
def load_dataset():
    """
    Opens a file dialog for the user to select a CSV dataset,
    validates it, loads it into a global DataFrame, and updates
    the status label with a quick summary.
    """
    global df

    file_path = filedialog.askopenfilename(
        title="Select Rainfall Dataset",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_path:
        return  # user cancelled — not an error, so no popup

    try:
        loaded_df = pd.read_csv(file_path)

        if loaded_df.empty:
            messagebox.showerror("Error", "The selected file is empty!")
            return

        numeric_cols = loaded_df.select_dtypes(include=[np.number]).shape[1]
        if numeric_cols < 2:
            messagebox.showerror(
                "Error",
                "Dataset needs at least 2 numeric columns "
                "(features + rainfall target)."
            )
            return

        df = loaded_df
        status_label.config(text=f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        messagebox.showinfo("Success", f"Dataset Loaded!\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

    except Exception as e:
        messagebox.showerror("Error", f"Could not read the file:\n{e}")


# ======================================================
# TRAIN MODEL
# ======================================================
def train_model():
    """
    Validates that a dataset is loaded, then kicks off training on a
    background thread so the UI stays responsive (60 epochs would
    otherwise freeze the window). Buttons are disabled while training.
    """
    global df

    if df is None:
        messagebox.showerror("Error", "Please upload a dataset first!")
        return

    btn_train.config(state="disabled")
    btn_upload.config(state="disabled")
    btn_predict.config(state="disabled")
    status_label.config(text="Training model... please wait")

    threading.Thread(target=_train_model_worker, daemon=True).start()


def _train_model_worker():
    """
    Does the actual preprocessing + training off the main thread.
    Never touches Tkinter widgets directly — hands results back to
    the main thread via root.after() since Tkinter isn't thread-safe.
    """
    global df, model, scaler_x, scaler_y

    try:
        data = df.copy()
        data.fillna(data.mean(numeric_only=True), inplace=True)

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1:]

        if not all(np.issubdtype(dt, np.number) for dt in X.dtypes):
            raise ValueError("All feature columns must be numeric.")

        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )

        new_model = Sequential([
            Dense(32, activation="relu", input_shape=(X.shape[1],)),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        new_model.compile(optimizer="adam", loss="mse")
        new_model.fit(X_train, y_train, epochs=60, verbose=0)

        preds = new_model.predict(X_test)
        preds_rescaled = scaler_y.inverse_transform(preds)
        y_test_rescaled = scaler_y.inverse_transform(y_test)

        accuracy = r2_score(y_test_rescaled, preds_rescaled) * 100

        model = new_model
        root.after(0, _on_training_success, accuracy)

    except Exception as e:
        root.after(0, _on_training_error, str(e))


def _on_training_success(accuracy):
    """Re-enables the UI and reports the trained model's accuracy."""
    btn_train.config(state="normal")
    btn_upload.config(state="normal")
    btn_predict.config(state="normal")
    status_label.config(text=f"Model trained — Accuracy: {accuracy:.2f}%")
    messagebox.showinfo("Training Complete", f"Model trained!\nAccuracy: {accuracy:.2f} %")


def _on_training_error(error_msg):
    """Re-enables the UI and reports why training failed."""
    btn_train.config(state="normal")
    btn_upload.config(state="normal")
    btn_predict.config(state="normal")
    status_label.config(text="Training failed")
    messagebox.showerror("Training Error", f"Something went wrong:\n{error_msg}")


# ======================================================
# PREDICT RAINFALL FROM USER INPUT
# ======================================================
def make_prediction():
    """
    Reads the 7 input fields, validates them as numbers, scales them
    with the same scaler used in training, and shows the prediction.
    """
    global model

    if model is None:
        messagebox.showerror("Error", "Train the model first!")
        return

    try:
        vals = [
            float(entry_year.get()),
            float(entry_month.get()),
            float(entry_temp.get()),
            float(entry_hum.get()),
            float(entry_wind.get()),
            float(entry_press.get()),
            float(entry_cloud.get()),
        ]
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numbers in every field!")
        return

    try:
        X_new = np.array([vals])
        X_scaled = scaler_x.transform(X_new)
        pred_scaled = model.predict(X_scaled)
        rainfall = scaler_y.inverse_transform(pred_scaled)[0][0]
        result_label.config(text=f"Predicted Rainfall: {rainfall:.2f} mm")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Could not generate a prediction:\n{e}")


# ======================================================
# TKINTER UI
# ======================================================
root = tk.Tk()
root.title("WEATHER RAINFALL PREDICTION SYSTEM USING NEURAL NETWORK ANALYSIS")
root.geometry("520x700")
root.configure(bg="#e3f2fd")

title = tk.Label(root, text=" WEATHER RAINFALL PREDICTION SYSTEM USING NEURAL NETWORK ANALYSIS",
                 font=("Arial", 16, "bold"), bg="#e3f2fd", wraplength=480, justify="center")
title.pack(pady=10)


# ---------------- Buttons ----------------
btn_frame = tk.Frame(root, bg="#e3f2fd")
btn_frame.pack(pady=10)

btn_upload = tk.Button(btn_frame, text="Upload Dataset", width=18,
                       bg="#4caf50", fg="white", font=("Arial", 12),
                       command=load_dataset)
btn_upload.grid(row=0, column=0, padx=5)

btn_train = tk.Button(btn_frame, text="Train Model", width=18,
                      bg="#2196f3", fg="white", font=("Arial", 12),
                      command=train_model)
btn_train.grid(row=0, column=1, padx=5)


# ---------------- Status ----------------
status_label = tk.Label(root, text="No dataset loaded",
                        font=("Arial", 10, "italic"), bg="#e3f2fd", fg="#555")
status_label.pack(pady=(0, 10))


# ---------------- Input Fields ----------------
input_frame = tk.Frame(root, bg="#e3f2fd")
input_frame.pack(pady=10)

labels = ["Year", "Month", "Temperature", "Humidity",
          "Wind Speed", "Pressure", "Cloud Cover"]

entries = []

for i, t in enumerate(labels):
    lbl = tk.Label(input_frame, text=t + ":", font=("Arial", 12), bg="#e3f2fd")
    lbl.grid(row=i, column=0, pady=5, sticky="w")

    ent = tk.Entry(input_frame, width=15, font=("Arial", 12))
    ent.grid(row=i, column=1, pady=5)
    entries.append(ent)

entry_year, entry_month, entry_temp, entry_hum, entry_wind, entry_press, entry_cloud = entries


# ---------------- Predict Button ----------------
btn_predict = tk.Button(root, text="Predict Rainfall", width=20,
                        bg="#ff5722", fg="white", font=("Arial", 14),
                        command=make_prediction)
btn_predict.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here",
                        font=("Arial", 14, "bold"), bg="#e3f2fd", fg="#000")
result_label.pack(pady=10)


if __name__ == "__main__":
    root.mainloop()

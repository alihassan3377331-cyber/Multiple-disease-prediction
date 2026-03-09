# Multiple Disease Prediction System GUI
# Is file ko app.py ke naam se save karein aur run karein.

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class TargetedDiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩺 Targeted Disease Prediction System")
        self.root.geometry("950x700")
        self.root.configure(bg="#f0f4f8")
        
        # Variables
        self.models = {}
        self.feature_columns = []
        self.input_entries = {}
        self.disease_mapping_reverse = {}
        self.unique_diseases_list = []
        self.is_trained = False

        # Custom Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("TCombobox", font=("Segoe UI", 10))

        self.setup_ui()

    def setup_ui(self):
        # --- HEADER ---
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=20)
        header_frame.pack(fill=tk.X)
        
        title_lbl = tk.Label(header_frame, text="Targeted Disease Screening via Blood Samples", 
                             font=("Segoe UI", 18, "bold"), bg="#2c3e50", fg="white")
        title_lbl.pack()

        # --- TOP CONTROL PANEL ---
        control_frame = tk.Frame(self.root, bg="#f0f4f8", pady=15)
        control_frame.pack(fill=tk.X, padx=20)

        self.train_btn = ttk.Button(control_frame, text="📥 Download Data & Train Models", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.status_lbl = tk.Label(control_frame, text="Status: Waiting for data...", bg="#f0f4f8", fg="#e74c3c", font=("Segoe UI", 11, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=20)
        
        # --- MAIN CONTENT AREA ---
        self.main_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left: Target Disease & Blood Test Inputs
        left_panel = tk.Frame(self.main_frame, bg="#f0f4f8")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        # 1. Target Disease Selection (Nayi Requirement)
        target_frame = tk.LabelFrame(left_panel, text="1. Select Target Disease to Check", bg="#ffffff", font=("Segoe UI", 11, "bold"), fg="#e67e22", padx=10, pady=10)
        target_frame.pack(fill=tk.X, pady=(0, 10))

        self.target_disease_var = tk.StringVar()
        self.target_disease_dropdown = ttk.Combobox(target_frame, textvariable=self.target_disease_var, state="readonly", width=40)
        self.target_disease_dropdown.pack(side=tk.LEFT, padx=10)
        self.target_disease_dropdown.set("Train models first to load diseases...")

        # 2. Blood Test Inputs (Scrollable)
        self.input_frame_container = tk.LabelFrame(left_panel, text="2. Enter Patient Blood Test Results", bg="#ffffff", font=("Segoe UI", 11, "bold"), fg="#2980b9", padx=10, pady=10)
        self.input_frame_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.input_frame_container, bg="#ffffff", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.input_frame_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#ffffff")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right: Prediction Panel
        self.right_panel = tk.Frame(self.main_frame, bg="#ffffff", width=280, relief="flat", bd=1)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)

        panel_title = tk.Label(self.right_panel, text="Prediction Center", bg="#ffffff", font=("Segoe UI", 12, "bold"), fg="#2c3e50")
        panel_title.pack(pady=15)

        model_lbl = tk.Label(self.right_panel, text="Select Algorithm:", bg="#ffffff", font=("Segoe UI", 10, "bold"), fg="#7f8c8d")
        model_lbl.pack(anchor="w", padx=20)

        self.model_var = tk.StringVar(value="Decision Tree")
        self.model_dropdown = ttk.Combobox(self.right_panel, textvariable=self.model_var, values=["Decision Tree", "KNN"], state="readonly")
        self.model_dropdown.pack(fill=tk.X, padx=20, pady=(5, 25))

        self.predict_btn = ttk.Button(self.right_panel, text="🔍 Check Target Disease", command=self.make_prediction, state=tk.DISABLED)
        self.predict_btn.pack(fill=tk.X, padx=20, pady=10, ipady=5)

        tk.Label(self.right_panel, text="Diagnosis Result:", bg="#ffffff", font=("Segoe UI", 10, "bold"), fg="#7f8c8d").pack(anchor="w", padx=20, pady=(20, 0))

        self.result_lbl = tk.Label(self.right_panel, text="---", bg="#ecf0f1", fg="#2c3e50", 
                                   font=("Segoe UI", 14, "bold"), relief="flat", pady=30, wraplength=220)
        self.result_lbl.pack(fill=tk.X, padx=20, pady=10)

        self.actual_pred_lbl = tk.Label(self.right_panel, text="", bg="#ffffff", fg="#7f8c8d", font=("Segoe UI", 9, "italic"), wraplength=220)
        self.actual_pred_lbl.pack(fill=tk.X, padx=20)

    # --- TRAINING LOGIC ---
    def start_training(self):
        self.train_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Status: Downloading Data & Training...", fg="#f39c12")
        threading.Thread(target=self.train_models_backend, daemon=True).start()

    def train_models_backend(self):
        try:
            # 1. Load Data
            path = kagglehub.dataset_download("ehababoelnaga/multiple-disease-prediction")
            disease1 = pd.read_csv(os.path.join(path, "Blood_samples_dataset_balanced_2(f).csv"))
            disease2 = pd.read_csv(os.path.join(path, "blood_samples_dataset_test.csv"))
            disease = pd.concat([disease1, disease2])

            # 2. Preprocess & Map Diseases
            unique_diseases = disease['Disease'].unique()
            self.unique_diseases_list = list(unique_diseases)
            
            disease_mapping = {disease_name: i for i, disease_name in enumerate(unique_diseases)}
            self.disease_mapping_reverse = {i: disease_name for i, disease_name in enumerate(unique_diseases)}
            
            disease['Disease'] = disease['Disease'].replace(disease_mapping)

            x = disease.drop('Disease', axis=1)
            y = disease['Disease']
            self.feature_columns = x.columns.tolist()

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # 3. Train Models
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(x_train, y_train)
            knn_acc = accuracy_score(y_test, knn.predict(x_test))

            dtc = DecisionTreeClassifier()
            dtc.fit(x_train, y_train)
            dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

            self.models = {"KNN": knn, "Decision Tree": dtc}

            self.root.after(0, self.training_complete, dtc_acc, knn_acc)

        except Exception as e:
            self.root.after(0, self.training_failed, str(e))

    def training_complete(self, dtc_acc, knn_acc):
        self.status_lbl.config(text=f"✅ Ready! (DT: {dtc_acc*100:.1f}% | KNN: {knn_acc*100:.1f}%)", fg="#27ae60")
        self.is_trained = True
        self.predict_btn.config(state=tk.NORMAL)
        
        # Populate Target Disease Dropdown
        self.target_disease_dropdown['values'] = self.unique_diseases_list
        if self.unique_diseases_list:
            self.target_disease_dropdown.current(0)
        
        self.generate_input_fields()
        messagebox.showinfo("Success", "System is ready!\nStep 1: Select a Target Disease.\nStep 2: Enter Blood Values.")

    def training_failed(self, error_msg):
        self.train_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="Status: Training Failed!", fg="#c0392b")
        messagebox.showerror("Error", f"Failed to train models.\nError: {error_msg}")

    # --- DYNAMIC GRID UI GENERATION ---
    def generate_input_fields(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.input_entries.clear()
        columns = 3
        for i, feature in enumerate(self.feature_columns):
            row = i // columns
            col = i % columns

            cell_frame = tk.Frame(self.scrollable_frame, bg="#ffffff", padx=10, pady=8)
            cell_frame.grid(row=row, column=col, sticky="w")
            
            lbl = tk.Label(cell_frame, text=feature, width=18, anchor="w", bg="#ffffff", font=("Segoe UI", 9, "bold"), fg="#34495e")
            lbl.pack(side=tk.TOP, anchor="w")
            
            ent = ttk.Entry(cell_frame, width=22)
            ent.pack(side=tk.TOP, pady=(2,0))
            ent.insert(0, "0.0") 
            self.input_entries[feature] = ent

    # --- PREDICTION LOGIC ---
    def make_prediction(self):
        if not self.is_trained:
            messagebox.showwarning("Warning", "Please train the models first!")
            return

        target_disease = self.target_disease_var.get()
        if not target_disease or target_disease == "Train models first to load diseases...":
            messagebox.showerror("Selection Error", "Please select a target disease to check.")
            return

        input_data = []
        try:
            for feature in self.feature_columns:
                val = self.input_entries[feature].get()
                input_data.append(float(val))
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers.")
            return

        input_array = np.array([input_data])
        selected_model = self.model_var.get()
        model = self.models[selected_model]

        # Model Prediction
        prediction_idx = model.predict(input_array)[0]
        predicted_disease = self.disease_mapping_reverse.get(prediction_idx, "Unknown")

        # Compare Prediction with User's Target Disease
        if predicted_disease == target_disease:
            self.result_lbl.config(text=f"POSITIVE\nfor {target_disease}", fg="#c0392b", bg="#fadbd8") # Red for Positive
            self.actual_pred_lbl.config(text="")
        else:
            self.result_lbl.config(text=f"NEGATIVE\nfor {target_disease}", fg="#27ae60", bg="#d5f5e3") # Green for Negative
            self.actual_pred_lbl.config(text=f"(Model suggests signs of: {predicted_disease})")

if __name__ == "__main__":
    root = tk.Tk()
    app = TargetedDiseasePredictorApp(root)
    root.mainloop()
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.popup import Popup

# Configuration
MODEL_FILE = "../modelFiles/diabetes_model.pkl"
SCALER_FILE = "../modelFiles/diabetes_scaler.pkl"
DATASET_FILE = "../dataFiles/dataset.csv"

# Dataset generation & training
def synthesize_pima_like_dataset(n_samples=800, random_state=42):
    rng = np.random.RandomState(random_state)

    pregnancies = rng.poisson(1.5, size=n_samples) # typically small integers
    age = rng.randint(18, 85, size=n_samples)
    pedigree = np.round(rng.uniform(0.05, 2.5, size=n_samples), 3)

    # glucose - positive skewed
    glucose = np.clip(rng.normal(120, 30, size=n_samples), 40, 300).astype(int)

    # blood pressure typical adult values
    bloodpressure = np.clip(rng.normal(70, 12, size=n_samples), 30, 140).astype(int)

    # skin thickness (mm)
    skinthickness = np.clip(rng.normal(20, 10, size=n_samples), 7, 99).astype(int)

    # insulin (μU/ml) - skewed and sometimes zero
    insulin = np.clip(rng.normal(80, 90, size=n_samples), 0, 900).astype(int)

    # BMI computed from a generated height and weight for realism
    heights_cm = np.clip(rng.normal(162, 10, size=n_samples), 140, 200)
    weights_kg = np.clip(rng.normal(75, 18, size=n_samples), 40, 200)
    bmi = np.round(weights_kg / ((heights_cm / 100.0) ** 2), 1)

    # Create a logistic-ish probability for having diabetes
    # This is synthetic: higher glucose, BMI, age, pedigree raise risk
    logit = (
        0.03 * (glucose - 100) +
        0.05 * (bmi - 25) +
        0.02 * (age - 45) +
        0.8 * pedigree +
        0.2 * (pregnancies > 2).astype(int) +
        rng.normal(0, 1.0, size=n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    outcome = (prob > rng.uniform(0, 1, size=n_samples)).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
        "Outcome": outcome
    })
    return df

def train_and_save_model(df, model_file=MODEL_FILE, scaler_file=SCALER_FILE):

    # Train a RandomForestClassifier on the provided DataFrame and save model + scaler.
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # split to simulate realistic training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, random_state=42, min_samples_leaf=3)
    model.fit(X_train_scaled, y_train)

    # Save
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    # Save dataset to CSV for inspection
    df.to_csv(DATASET_FILE, index=False)

    return model, scaler

def ensure_model_exists():

    # If model/scaler files exist, load them. Otherwise, synthesize dataset and train.
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("Loaded existing model and scaler.")
        return model, scaler
    else:
        print("Model not found. Generating dataset and training model...")
        df = synthesize_pima_like_dataset()
        model, scaler = train_and_save_model(df)
        print(f"Training complete. Model saved to {MODEL_FILE}. Dataset saved to {DATASET_FILE}.")
        return model, scaler


# Kivy UI
class DiabetesPredictorUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=8, padding=12, **kwargs)
        Window.size = (520, 680)

        # Load or create model
        self.model, self.scaler = ensure_model_exists()

        header = Label(text="Diabetes Prediction (Kivy demo)", size_hint=(1, None), height=40, font_size=18)
        self.add_widget(header)

        # Grid for inputs
        grid = GridLayout(cols=2, spacing=6, size_hint=(1, None))
        grid.bind(minimum_height=grid.setter('height'))

        # INPUT FIELDS: label, TextInput
        self.inputs = {}

        def add_input(name, hint, numeric=True, default=""):
            lbl = Label(text=name, size_hint=(0.6, None), height=36)
            ti = TextInput(text=default, hint_text=hint, multiline=False, size_hint=(0.4, None), height=36)
            if numeric:
                ti.input_filter = 'float'
            grid.add_widget(lbl)
            grid.add_widget(ti)
            self.inputs[name] = ti

        # Height and weight (we compute BMI)
        add_input("Height (cm)", "e.g. 165", numeric=True, default="165")
        add_input("Weight (kg)", "e.g. 70", numeric=True, default="70")

        # Other features (we provide reasonable defaults)
        add_input("Pregnancies", "0-15", numeric=True, default="0")
        add_input("Glucose (mg/dL)", "e.g. 120", numeric=True, default="120")
        add_input("BloodPressure (mm Hg)", "e.g. 70", numeric=True, default="70")
        add_input("SkinThickness (mm)", "e.g. 20", numeric=True, default="20")
        add_input("Insulin (mu U/ml)", "e.g. 80", numeric=True, default="80")
        add_input("DiabetesPedigreeFunction", "e.g. 0.5", numeric=True, default="0.5")
        add_input("Age", "e.g. 45", numeric=True, default="45")

        # Make grid scrollable height-friendly by wrapping in a layout (keeps code simple)
        self.add_widget(grid)

        # Output Label
        self.output_label = Label(text="Fill inputs and click Predict", size_hint=(1, None), height=80)
        self.add_widget(self.output_label)

        # Buttons
        btn_layout = BoxLayout(size_hint=(1, None), height=48, spacing=8)
        predict_btn = Button(text="Predict", on_release=self.on_predict)
        clear_btn = Button(text="Clear", on_release=self.on_clear)
        retrain_btn = Button(text="Retrain model", on_release=self.on_retrain)
        btn_layout.add_widget(predict_btn)
        btn_layout.add_widget(clear_btn)
        btn_layout.add_widget(retrain_btn)

        self.add_widget(btn_layout)

        # small help/instructions
        help_text = (
            "Notes:\n"
            "- Height & Weight -> BMI computed automatically.\n"
            "- If you don't know a value (e.g. Insulin), leave it default.\n"
            "- 'Retrain model' regenerates a synthetic dataset and retrains the model locally."
        )
        self.add_widget(Label(text=help_text, size_hint=(1, None), height=120))

    def on_clear(self, instance):
        for k, ti in self.inputs.items():
            # reset to sensible defaults
            if "Height" in k:
                ti.text = "165"
            elif "Weight" in k:
                ti.text = "70"
            elif "Pregnancies" in k:
                ti.text = "0"
            elif "Glucose" in k:
                ti.text = "120"
            elif "BloodPressure" in k:
                ti.text = "70"
            elif "SkinThickness" in k:
                ti.text = "20"
            elif "Insulin" in k:
                ti.text = "80"
            elif "DiabetesPedigreeFunction" in k:
                ti.text = "0.5"
            elif "Age" in k:
                ti.text = "45"
        self.output_label.text = "Cleared inputs."

    def on_retrain(self, instance):
        # retrain model using a newly synthesized dataset
        df = synthesize_pima_like_dataset()
        self.model, self.scaler = train_and_save_model(df)
        self.output_label.text = "Model retrained on new synthetic dataset."

    def _read_input_value(self, name, cast=float, fallback=0.0):
        txt = self.inputs[name].text.strip()
        if txt == "":
            return fallback
        try:
            return cast(txt)
        except Exception:
            return fallback

    def on_predict(self, instance):
        try:
            # Read height & weight, compute BMI
            height_cm = self._read_input_value("Height (cm)", float, 165.0)
            weight_kg = self._read_input_value("Weight (kg)", float, 70.0)
            if height_cm <= 0:
                raise ValueError("Height must be > 0")
            bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)

            # Build feature vector in the order used in training
            pregnancies = int(round(self._read_input_value("Pregnancies", float, 0)))
            glucose = float(self._read_input_value("Glucose (mg/dL)", float, 120))
            bp = float(self._read_input_value("BloodPressure (mm Hg)", float, 70))
            skin = float(self._read_input_value("SkinThickness (mm)", float, 20))
            insulin = float(self._read_input_value("Insulin (mu U/ml)", float, 80))
            pedigree = float(self._read_input_value("DiabetesPedigreeFunction", float, 0.5))
            age = int(round(self._read_input_value("Age", float, 45)))

            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
            features_scaled = self.scaler.transform(features)

            prob = self.model.predict_proba(features_scaled)[0][1]  # probability of class 1
            label = "Positive for Diabetes (high risk)" if prob >= 0.5 else "Negative for Diabetes (low risk)"
            prob_pct = round(prob * 100, 1)

            advice = (
                f"[{prob_pct}%] {label}\n\n"
            )

            # Put BMI in output too
            self.output_label.text = f"BMI: {bmi} — Prediction: {prob_pct}% chance -> {label}"
            # Show a popup with more detail
            popup = Popup(title="Prediction result", content=Label(text=advice), size_hint=(0.9, 0.6))
            popup.open()

        except Exception as e:
            popup = Popup(title="Error", content=Label(text=str(e)), size_hint=(0.8, 0.4))
            popup.open()


# App wrapper
class DiabetesApp(App):
    def build(self):
        self.title = "Diabetes Prediction Demo"
        return DiabetesPredictorUI()

if __name__ == "__main__":
    DiabetesApp().run()

"""
EcoVision - Refined SHAP Explainability Module
----------------------------------------------

This module provides:
1. Global feature importance (bar + summary)
2. Local explanation for a single prediction
3. Human-readable explanation text
4. Model-agnostic structure (tree-based models)

Designed for:
- Urban Heat Island (UHI)
- Flood Risk
- Plastic Waste Risk

Author: EcoVision Project
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ==============================
# SHAP EXPLAINER CLASS
# ==============================

class SHAPExplainer:
    def __init__(self, model_name: str, data_file: str, feature_columns: list):
        """
        model_name: trained model filename
        data_file: processed dataset filename
        feature_columns: input feature names
        """

        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.data_path = os.path.join(DATA_DIR, data_file)
        self.features = feature_columns

        self._load_model()
        self._load_data()
        self._init_explainer()

    # ==============================
    # INTERNAL LOADERS
    # ==============================

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ Data not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.X = df[self.features]

    def _init_explainer(self):
        """
        Uses TreeExplainer for efficiency & correctness
        """
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(self.X)

    # ==============================
    # GLOBAL EXPLANATIONS
    # ==============================

    def global_importance_bar(self):
        """
        Global mean absolute SHAP value bar plot
        """
        shap.plots.bar(
            self.shap_values,
            max_display=len(self.features),
            show=False
        )
        plt.title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.show()

    def global_summary(self):
        """
        SHAP summary plot (distribution & direction)
        """
        shap.plots.beeswarm(
            self.shap_values,
            show=False
        )
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.show()

    def global_text_summary(self, top_k: int = 3) -> str:
        """
        Returns text summary of most influential features
        """
        mean_abs = np.abs(self.shap_values.values).mean(axis=0)
        importance = sorted(
            zip(self.features, mean_abs),
            key=lambda x: x[1],
            reverse=True
        )

        text = "Global SHAP Explanation:\n"
        for feature, value in importance[:top_k]:
            text += f"- {feature} is a major driver (avg impact={value:.3f})\n"

        return text

    # ==============================
    # LOCAL EXPLANATIONS
    # ==============================

    def local_waterfall(self, index: int = 0):
        """
        Waterfall plot for a single prediction
        """
        shap.plots.waterfall(
            self.shap_values[index],
            show=False
        )
        plt.title(f"Local SHAP Explanation (Instance {index})")
        plt.tight_layout()
        plt.show()

    def local_force(self, index: int = 0):
        """
        Force plot (matplotlib-friendly)
        """
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[index].values,
            self.X.iloc[index],
            matplotlib=True,
            show=True
        )

    def local_text_summary(self, index: int = 0, top_k: int = 3) -> str:
        """
        Human-readable explanation for a single prediction
        """
        values = self.shap_values[index].values
        contributions = sorted(
            zip(self.features, values),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        text = f"Local explanation for instance {index}:\n"
        for feature, val in contributions[:top_k]:
            direction = "increased" if val > 0 else "decreased"
            text += f"- {feature} {direction} the prediction (impact={val:.3f})\n"

        return text

# ==============================
# MODULE TEST (SAFE TO RUN)
# ==============================

if __name__ == "__main__":
    """
    Example: UHI SHAP refinement test
    """

    uhi_features = ["LST", "NDVI", "NDBI", "Albedo", "LULC"]

    explainer = SHAPExplainer(
        model_name="uhi_random_forest.pkl",
        data_file="uhi_clean.csv",
        feature_columns=uhi_features
    )

    print("\n📊 GLOBAL SHAP BAR")
    explainer.global_importance_bar()

    print("\n📈 GLOBAL SHAP SUMMARY")
    explainer.global_summary()

    print("\n🧠 GLOBAL TEXT SUMMARY")
    print(explainer.global_text_summary())

    print("\n🧩 LOCAL WATERFALL")
    explainer.local_waterfall(index=0)

    print("\n📄 LOCAL TEXT SUMMARY")
    print(explainer.local_text_summary(index=0))

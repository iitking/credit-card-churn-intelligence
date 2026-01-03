from setuptools import setup, find_packages

setup(
    name="credit-card-churn-intelligence",
    version="1.0.0",
    description="End-to-End Enterprise Credit Card Churn Prediction & Retention Intelligence System",
    author="Antigravity AI Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "joblib>=1.2.0",
        "streamlit>=1.25.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pydantic>=1.10.0",
        "plotly>=5.10.0",
    ],
)

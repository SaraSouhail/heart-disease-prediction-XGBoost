# ❤️ Heart Disease Prediction using XGBoost

## 📘 Overview
This project implements a comprehensive machine learning pipeline to predict the likelihood of heart disease using the **XGBoost** algorithm. It was developed as part of a Master’s program in **Artificial Intelligence and Data Science**. The project covers the entire workflow from data acquisition and preprocessing to model training, evaluation, and interpretation, making it a complete example of applying ML to healthcare data.

## ✨ Features
- **Data Acquisition**: Automated download and extraction of the heart disease dataset from Kaggle.
- **Exploratory Data Analysis (EDA)**: Visualizations for target distribution, missing values, and correlations.
- **Advanced Preprocessing**: Handling missing values using iterative imputation (regression for continuous, classification for categorical), outlier detection and treatment via Z-score.
- **Feature Engineering**: Encoding categorical variables and correlation analysis.
- **Model Training**: XGBoost classifier with optimized hyperparameters for binary/multi-class classification.
- **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and confusion matrix.
- **Interpretability**: Feature importance visualization to understand key predictors.
- **Predictions**: Example predictions on new patient data.

## 🧠 Technologies Used
- **Programming Language**: Python 3.x
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebook / Google Colab
- **Other**: Kaggle API for dataset download

## 📦 Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SaraSouhail/heart-disease-prediction-xgboost.git
   cd heart-disease-prediction-xgboost
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Install the required packages using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost kaggle
   ```

3. **Set Up Kaggle API** (for dataset download):
   - Download your Kaggle API key from your Kaggle account settings.
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows).
   - Run the notebook cells to download the dataset automatically.

## 📊 Dataset
- **Source**: [Heart Disease Data on Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Description**: The dataset is based on the UCI Heart Disease dataset, containing patient attributes such as age, sex, chest pain type, blood pressure, cholesterol, etc., and a target variable indicating the presence and severity of heart disease (0-4 scale).
- **Size**: Approximately 920 samples with 15 features.
- **Preprocessing**: Includes handling missing values, outliers, and encoding categorical features.

## 🚀 Usage
1. **Open the Notebook**:
   - Launch Jupyter Notebook or Google Colab.
   - Open `Prédiction de Maladies Cardiaques (XGBoost).ipynb`.

2. **Run the Cells Sequentially**:
   - The notebook is self-contained and will download the dataset, perform preprocessing, train the model, and evaluate it.
   - Key sections include data loading, EDA, imputation, outlier treatment, encoding, training, and visualization.

3. **Customize and Experiment**:
   - Modify hyperparameters in the XGBoost model cell.
   - Add new features or try different algorithms.

## 🏗️ Model Details
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Task**: Multi-class classification (predicting heart disease severity: 0-4)
- **Hyperparameters**:
  - `objective`: 'binary:logistic' (adapted for multi-class)
  - `learning_rate`: 0.01
  - `n_estimators`: 20
  - `max_depth`: 3
  - `min_child_weight`: 2
  - `random_state`: 30
- **Training**: 80% train / 20% test split with stratification.
- **Preprocessing**: Iterative imputation for missing values, Z-score outlier removal, label encoding for categoricals.

## 📈 Evaluation
- **Metrics**:
  - Accuracy: Overall correctness of predictions.
  - Precision, Recall, F1-Score: Per-class performance.
  - Confusion Matrix: Visual breakdown of true positives, false positives, etc.
- **Results**: The model achieves strong performance, with detailed reports in the notebook.

## 📊 Results
- Achieved high accuracy on the test set.
- Feature importance highlights key factors like `thal`, `ca`, `oldpeak`, etc.
- Visualizations include correlation heatmaps, distribution plots, and confusion matrices.
- Example predictions demonstrate real-world application.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements, bug fixes, or new features.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact
For questions or feedback, feel free to reach out via GitHub issues.


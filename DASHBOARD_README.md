# 🤖 AutoML Dashboard - Quick Start

## 🚀 Launch the Dashboard

### Method 1: Using the launcher script (Recommended)
```bash
python run_automl_dashboard.py
```

### Method 2: Manual launch
```bash
source automl_venv/bin/activate
streamlit run streamlit_app.py
```

The dashboard will open in your browser at: http://localhost:8501

## 📋 How to Use

1. **📁 Upload Dataset**: Click "Upload Dataset" in the sidebar
   - Supports CSV, Excel (.xlsx, .xls) files
   - View data exploration automatically

2. **🎯 Select Target**: Choose the column you want to predict

3. **⚙️ Configure Settings**:
   - **Task Type**: Classification or Regression (auto-detected)
   - **Time Limit**: How long to train models (1-30 minutes)
   - **Advanced Settings**: Validation split, code/plot options

4. **🚀 Train Models**: Click "Train AutoML Model"
   - Training happens in the main screen
   - Progress bar shows current step
   - Multiple models tested automatically

5. **📊 View Results**:
   - **Best model** and performance metrics
   - **Feature importance** charts
   - **Data visualizations** and analysis
   - **Generated Python code** for reproduction
   - **Download options** for model and results

## 📂 Sample Data

If you don't have data, click the sample download buttons:
- 🌸 **Iris Classification**: Species prediction from flower measurements
- 🏠 **Housing Regression**: Price prediction from house features

## 🔧 Features

### 📊 **Data Analysis**
- Automatic data profiling
- Missing value analysis
- Column type detection
- Distribution visualizations

### 🤖 **AutoML Training**
- Multiple algorithms tested:
  - Random Forest
  - Linear/Logistic Regression
  - XGBoost (if installed)
  - LightGBM (if installed)
- Automatic preprocessing:
  - Missing value handling
  - Categorical encoding
  - Feature scaling
  - Feature selection
- Hyperparameter tuning
- Cross-validation

### 📈 **Results & Visualizations**
- Performance metrics
- Feature importance plots
- Target distribution analysis
- Correlation analysis
- Model comparison

### 💻 **Code Generation**
- Complete Python code for:
  - Model training
  - Making predictions
  - Model analysis
  - Requirements file

### 💾 **Export Options**
- Save trained model locally
- Download results summary
- Copy generated code

## ❗ Troubleshooting

### "Training failed: could not convert string to float"
- This happens with categorical target variables in correlation analysis
- The dashboard now handles this automatically
- If you see this error, try refreshing the page

### "Virtual environment not found"
- Make sure you're in the correct directory
- Run: `source automl_venv/bin/activate` first

### Dashboard won't start
- Check if port 8501 is in use
- Try: `streamlit run streamlit_app.py --server.port 8502`

## 📊 Supported Data Formats

### ✅ **Input Files**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

### ✅ **Data Types**
- **Numerical**: Integers, floats, decimals
- **Categorical**: Text, strings, categories
- **Mixed**: Combination of both

### ✅ **Task Types**
- **Classification**: Predict categories/classes
- **Regression**: Predict numerical values

## 🎯 Tips for Best Results

1. **Clean your data**: Remove obvious errors before upload
2. **Choose clear target**: Make sure your target column is what you want to predict
3. **Sufficient data**: At least 100+ rows for good results
4. **Reasonable features**: More relevant features = better predictions
5. **Allow time**: Give the system 5-10 minutes for best results

## 🆘 Need Help?

If you encounter issues:
1. Check the console output for error messages
2. Try refreshing the dashboard
3. Restart the dashboard with the launcher script
4. Check that your data file is properly formatted

---

**Enjoy building ML models with AutoML! 🤖✨**
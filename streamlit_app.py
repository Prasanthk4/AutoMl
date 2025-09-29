"""
Streamlit Dashboard for AutoML System
Upload datasets, analyze data, train models, and get code + visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from automl import AutoML
    from automl.data_analysis import AdvancedDataAnalyzer
    from automl.model_comparison import ModelComparison, ABTestFramework
    from automl.comparison_visualizations import ComparisonVisualizer
    from automl.experiment_tracking import ExperimentTracker, ExperimentDatabase
    from automl.experiment_analytics import ExperimentHistory, ExperimentAnalytics
except ImportError as e:
    st.error(f"Failed to import AutoML: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Dark Theme Global Styles */
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    .main .block-container {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
    }
    
    /* Professional Dark Header */
    .main-header {
        text-align: center;
        padding: 40px;
        background: linear-gradient(135deg, #1f2937, #374151, #1f2937);
        color: #f9fafb;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid #374151;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        color: #f9fafb;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        margin: 15px 0 0 0;
        font-size: 1.1rem;
        color: #d1d5db;
        font-weight: 400;
    }
    
    /* Dark Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #161b22, #21262d);
    }
    
    /* Professional Dark Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #374151, #4b5563);
        color: #f9fafb;
        border-radius: 8px;
        border: 1px solid #4b5563;
        padding: 12px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-transform: none;
        letter-spacing: 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4b5563, #6b7280);
        border-color: #6b7280;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Primary Button (Train Model) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669, #10b981);
        border-color: #10b981;
        font-size: 16px;
        padding: 14px 20px;
        font-weight: 600;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #10b981, #34d399);
        border-color: #34d399;
    }
    
    /* Professional Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid #374151;
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        border-color: #4b5563;
    }
    
    [data-testid="metric-container"] > div {
        color: #f9fafb !important;
        font-weight: 500;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 1.8rem !important;
        color: #10b981 !important;
        font-weight: 700;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #d1d5db !important;
        font-size: 0.9rem !important;
        font-weight: 500;
    }
    
    /* Professional Code Headers */
    .code-header {
        background: linear-gradient(135deg, #1f2937, #374151);
        color: #f9fafb;
        padding: 12px 16px;
        border-radius: 8px 8px 0 0;
        margin: 0;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #374151;
        border-bottom: none;
    }
    
    /* Dark Theme Headers */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #f9fafb;
        font-weight: 600;
    }
    
    /* Professional Progress Bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #059669, #10b981);
        height: 8px;
        border-radius: 4px;
    }
    
    /* Dark Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #374151, #4b5563);
        color: #f9fafb;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #4b5563;
    }
    
    /* Professional File Uploader */
    .stFileUploader {
        border: 2px dashed #4b5563;
        border-radius: 12px;
        background: rgba(55, 65, 81, 0.3);
        padding: 24px;
    }
    
    .stFileUploader label {
        color: #d1d5db !important;
    }
    
    /* Dark Selectboxes */
    .stSelectbox > div > div {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        color: #f9fafb;
    }
    
    /* Professional Messages */
    .stSuccess {
        background: linear-gradient(135deg, #065f46, #059669) !important;
        border-radius: 8px !important;
        border: 1px solid #10b981 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #991b1b, #dc2626) !important;
        border-radius: 8px !important;
        border: 1px solid #ef4444 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #92400e, #d97706) !important;
        border-radius: 8px !important;
        border: 1px solid #f59e0b !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
        border-radius: 8px !important;
        border: 1px solid #60a5fa !important;
    }
    
    /* Dark Cards */
    .element-container {
        transition: all 0.2s ease;
    }
    
    .element-container:hover {
        transform: translateY(-1px);
    }
    
    /* Professional Text */
    .professional-text {
        color: #10b981;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Sidebar Styling */
    .css-1lcbmhc {
        background: linear-gradient(180deg, #161b22, #21262d);
        border-right: 1px solid #30363d;
    }
    
    /* Professional Data Frames */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        background: #1f2937;
    }
    
    /* Dark Text Inputs */
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
    }
    
    /* Dark Number Inputs */
    .stNumberInput > div > div > input {
        background-color: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
    }
    
    /* Dark Sliders */
    .stSlider > div > div > div > div {
        background: #374151;
    }
    
    /* Professional Typography */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    /* Remove Streamlit Branding Color */
    .css-1rs6os {
        background: #161b22;
    }
    
    /* Professional Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'training_requested' not in st.session_state:
        st.session_state.training_requested = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'automl'
    
    # Navigation tabs at the top
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🤖 AutoML Training", use_container_width=True, 
                    type="primary" if st.session_state.current_page == 'automl' else "secondary"):
            st.session_state.current_page = 'automl'
            st.session_state.training_requested = False
            st.rerun()
    
    with col2:
        if st.button("📊 Experiment History", use_container_width=True,
                    type="primary" if st.session_state.current_page == 'experiments' else "secondary"):
            st.session_state.current_page = 'experiments'
            st.session_state.training_requested = False
            st.rerun()
    
    with col3:
        if st.button("📈 Analytics", use_container_width=True,
                    type="primary" if st.session_state.current_page == 'analytics' else "secondary"):
            st.session_state.current_page = 'analytics'
            st.session_state.training_requested = False
            st.rerun()
    
    st.divider()
    
    # Conditional header based on current page
    if st.session_state.current_page == 'automl':
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AutoML System Dashboard</h1>
            <p>Enterprise-grade automated machine learning platform for data scientists and engineers</p>
            <div style="margin-top: 20px; font-size: 0.95rem; color: #10b981; font-weight: 500;">
                🎯 <strong>4 Algorithms</strong> • 📊 <strong>Auto-Selection</strong> • ⚡ <strong>Production Ready</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.current_page == 'experiments':
        st.markdown("""
        <div class="main-header">
            <h1>📊 Experiment History</h1>
            <p>Track, analyze, and compare your AutoML experiments</p>
            <div style="margin-top: 20px; font-size: 0.95rem; color: #10b981; font-weight: 500;">
                📈 <strong>Performance Tracking</strong> • 🔍 <strong>Experiment Search</strong> • ⚖️ <strong>Model Comparison</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:  # analytics
        st.markdown("""
        <div class="main-header">
            <h1>📈 Analytics Dashboard</h1>
            <p>Advanced analytics and insights from your experiments</p>
            <div style="margin-top: 20px; font-size: 0.95rem; color: #10b981; font-weight: 500;">
                📊 <strong>Trend Analysis</strong> • 🎯 <strong>Optimization Insights</strong> • 📋 <strong>Performance Reports</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional Sidebar (only show for AutoML page)
    if st.session_state.current_page == 'automl':
        with st.sidebar:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1f2937, #374151);
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 24px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.4);
                border: 1px solid #374151;
            ">
                <h2 style="color: #f9fafb; margin: 0; font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em;">
                    🔧 Configuration Panel
                </h2>
                <p style="color: #d1d5db; margin: 8px 0 0 0; font-size: 0.9rem; font-weight: 400;">
                    Configure your ML pipeline
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # File upload
            uploaded_file = st.file_uploader(
                "📁 Upload Dataset",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV, Excel (.xlsx), or Excel (.xls) files"
            )
            
            if uploaded_file:
                # Load and cache the data
                df = load_data(uploaded_file)
                if df is not None:
                    st.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    # Task configuration with professional header
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #374151, #4b5563);
                        padding: 16px;
                        border-radius: 8px;
                        margin: 20px 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                        border: 1px solid #4b5563;
                    ">
                        <h3 style="color: #f9fafb; margin: 0; text-align: center; font-weight: 600; font-size: 1.1rem;">
                            ⚙️ ML Configuration
                        </h3>
                        <p style="color: #d1d5db; margin: 4px 0 0 0; text-align: center; font-size: 0.85rem;">
                            Select target and configure training
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Target selection
                    target_column = st.selectbox(
                        "🎯 Select Target Column",
                        options=df.columns.tolist(),
                        help="Choose the column you want to predict"
                    )
                    
                    # Task type
                    if target_column:
                        # Auto-detect task type
                        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                        unique_values = df[target_column].nunique()
                        
                        if is_numeric and unique_values > 10:
                            default_task = "regression"
                        else:
                            default_task = "classification"
                        
                        task_type = st.selectbox(
                            "🔍 Task Type",
                            options=["classification", "regression"],
                            index=0 if default_task == "classification" else 1,
                            help="Choose the type of machine learning task"
                        )
                        
                        # Fast Mode Selection
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #059669, #10b981);
                            padding: 12px;
                            border-radius: 6px;
                            margin: 16px 0;
                            text-align: center;
                            border: 1px solid #10b981;
                        ">
                            <p style="color: white; margin: 0; font-weight: 600; font-size: 0.95rem;">
                                🚀 Training Mode
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    training_mode = st.radio(
                        "Choose Training Speed:",
                        options=["⚡ Fast Mode (1-2 min)", "🎯 Balanced Mode (3-5 min)", "🔬 Thorough Mode (5-10 min)"],
                        index=0,
                        help="Fast mode uses fewer trials and models for quick results. Balanced mode provides good accuracy. Thorough mode maximizes performance."
                    )
                    
                    # Set time limits based on mode
                    if "Fast Mode" in training_mode:
                        time_limit = 2
                        mode_config = "fast"
                    elif "Balanced Mode" in training_mode:
                        time_limit = 5
                        mode_config = "balanced"
                    else:
                        time_limit = 10
                        mode_config = "thorough"
                    
                    # Advanced settings
                    with st.expander("🔧 Advanced Settings"):
                        if mode_config != "fast":  # Allow customization for non-fast modes
                            time_limit = st.slider(
                                "⏱️ Time Limit (minutes)",
                                min_value=1,
                                max_value=15,
                                value=time_limit,
                                help="Maximum time for model training"
                            )
                        else:
                            st.info(f"⚡ Fast mode: Fixed at {time_limit} minutes for optimal speed")
                        
                        validation_split = st.slider(
                            "📊 Validation Split",
                            min_value=0.1,
                            max_value=0.5,
                            value=0.2,
                            help="Fraction of data used for validation"
                        )
                        
                        show_code = st.checkbox("📝 Show Generated Code", value=True)
                        show_plots = st.checkbox("📈 Show Visualizations", value=True)
                    
                    # Train button
                    if st.button("🚀 Train AutoML Model", type="primary"):
                        train_automl_model(df, target_column, task_type, time_limit * 60, validation_split, show_code, show_plots, mode_config)
                    
                    # Reset button
                    if st.session_state.get('training_requested', False):
                        st.divider()
                        if st.button("🔄 Reset to Data Exploration", type="secondary"):
                            st.session_state.training_requested = False
                            st.rerun()
            else:
                st.error("❌ Failed to load the dataset")
    else:
        # For experiment history and analytics pages, show a minimal sidebar
        with st.sidebar:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1f2937, #374151);
                padding: 24px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 24px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.4);
                border: 1px solid #374151;
            ">
                <h2 style="color: #f9fafb; margin: 0; font-size: 1.4rem; font-weight: 600; letter-spacing: -0.01em;">
                    📈 Navigation
                </h2>
                <p style="color: #d1d5db; margin: 8px 0 0 0; font-size: 0.9rem; font-weight: 400;">
                    Use the tabs above to navigate
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("📊 Switch to AutoML Training tab to upload datasets and train models.")
    
    # Main content routing based on selected page
    if st.session_state.current_page == 'experiments':
        show_experiment_history_dashboard()
    elif st.session_state.current_page == 'analytics':
        show_analytics_dashboard()
    else:  # automl page
        # Main content for AutoML
        if not uploaded_file:
            show_welcome_page()
        else:
            df = load_data(uploaded_file)
            if df is not None:
                # Check if training was requested
                if st.session_state.get('training_requested', False):
                    execute_training_in_main_area()
                else:
                    # Get target column from sidebar state if available
                    target_column = None
                    # Try to get target column from the current session
                    for key, value in st.session_state.items():
                        if 'target_column' in str(key).lower():
                            if value in df.columns:
                                target_column = value
                                break
                    
                    # If no target found in session state, check if there's a logical target
                    if not target_column:
                        # Look for common target column names
                        common_targets = ['target', 'label', 'class', 'output', 'y', 'price', 'species', 'approved']
                        for col in df.columns:
                            if col.lower() in common_targets:
                                target_column = col
                                break
                    
                    show_advanced_data_analysis_dashboard(df, target_column)

def execute_training_in_main_area():
    """Execute AutoML training in the main content area"""
    
    params = st.session_state.training_params
    df = params['df']
    target_column = params['target_column']
    task_type = params['task_type']
    time_limit = params['time_limit']
    validation_split = params['validation_split']
    show_code = params['show_code']
    show_plots = params['show_plots']
    mode_config = params.get('mode_config', 'balanced')
    
    # Show progress with cancel option
    st.header("🚀 Training AutoML Model...")
    
    # Add cancel button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("❌ Cancel Training", type="secondary"):
            st.session_state.training_requested = False
            st.info("✋ Training cancelled by user")
            st.rerun()
    
    with col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Initialize AutoML
    status_text.text("🔧 Initializing AutoML system...")
    progress_bar.progress(10)
    
    try:
        # Configure AutoML based on training mode
        if mode_config == 'fast':
            custom_config = {
                'models': {
                    'include_models': ['random_forest'],  # Only one fast model
                    'n_jobs': 1,  # Single core to avoid overload
                    'cv_folds': 2,
                },
                'tuning': {
                    'n_trials': 5,  # Very few trials
                    'timeout': 60,  # 1 minute max
                    'cv_folds': 2,
                },
                'evaluation': {
                    'cv_folds': 2,
                }
            }
        elif mode_config == 'thorough':
            custom_config = {
                'models': {
                    'n_jobs': 2,  # Limited cores
                    'cv_folds': 5,
                },
                'tuning': {
                    'n_trials': 50,  # More trials
                    'timeout': 600,  # 10 minutes
                    'cv_folds': 3,
                },
                'evaluation': {
                    'cv_folds': 5,
                }
            }
        else:  # balanced mode (default)
            custom_config = None  # Use the optimized defaults from config.py
            
        automl = AutoML(
            target=target_column,
            task_type=task_type,
            time_limit=int(time_limit),
            validation_split=validation_split,
            config=custom_config,
            verbose=False  # Disable verbose for cleaner UI
        )
        
        progress_bar.progress(20)
        status_text.text("📊 Loading and profiling data...")
        
        # Create temporary file for AutoML
        temp_file = "temp_dataset.csv"
        df.to_csv(temp_file, index=False)
        
        progress_bar.progress(40)
        status_text.text("🤖 Training models... This may take a few minutes...")
        
        # Train the model with timeout handling
        import signal
        import time as time_module
        
        # Set a reasonable timeout for the entire training process
        max_training_time = min(int(time_limit), 600)  # Max 10 minutes
        
        start_training = time_module.time()
        try:
            automl.fit(temp_file)
            actual_time = time_module.time() - start_training
            
            if actual_time > max_training_time * 0.8:
                st.warning(f"⚠️ Training took {actual_time:.1f}s (near timeout limit)")
                
        except Exception as training_error:
            elapsed = time_module.time() - start_training
            if elapsed > max_training_time * 0.9:
                raise TimeoutError(f"Training timeout after {elapsed:.1f}s - try reducing time limit or dataset size")
            else:
                raise training_error
        
        progress_bar.progress(80)
        status_text.text("📈 Generating results...")
        
        # Get results
        results = automl.get_results()
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        show_results(automl, results, df, target_column, task_type, show_code, show_plots)
        
        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)
        
        # Clear the training request from session state
        st.session_state.training_requested = False
        
        # Add button to go back to data exploration
        if st.button("🔙 Back to Data Exploration"):
            st.session_state.training_requested = False
            st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)
        
        # Add button to go back
        if st.button("🔙 Back to Data Exploration"):
            st.session_state.training_requested = False
            st.rerun()

@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        error_msg = str(e).lower()
        if "no columns to parse" in error_msg or "empty" in error_msg:
            st.error("📋 Unable to parse the file. Please ensure:")
            st.markdown("""
            - The file contains data with proper headers
            - For CSV files: columns are properly separated
            - For Excel files: data starts from the first row
            - The file is not empty or corrupted
            
            After uploading a valid file, select your target column and task type from the sidebar.
            """)
        else:
            st.error(f"❌ Error loading file: {e}")
        return None

def show_welcome_page():
    """Show welcome page when no file is uploaded"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            padding: 30px;
            border-radius: 25px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border: 3px solid rgba(255,255,255,0.3);
        ">
            <h2 style="color: white; margin: 0 0 15px 0; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🚀✨ Welcome to AutoML System! ✨🚀
            </h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0;">
                🎆 Build ML models automatically with our intelligent system! 🎆
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(45deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
            padding: 25px;
            border-radius: 20px;
            border: 2px solid rgba(78, 205, 196, 0.3);
            margin-bottom: 25px;
        ">
        
        #### 📋 How it works:
        
        #### 📋 Steps:
        1. **Upload** your dataset (CSV or Excel)
        2. **Select** your target column 
        3. **Configure** ML settings
        4. **Train** models automatically
        5. **Get** results, code, and visualizations
        
        #### 🎯 What you'll get:
        - 📊 **Data Analysis**: Comprehensive data profiling
        - 🤖 **Best Model**: Automatically selected ML model
        - 📈 **Visualizations**: Feature importance, performance plots
        - 💻 **Python Code**: Ready-to-use code for your model
        - 📋 **Detailed Results**: Performance metrics and insights
        
        #### 🔧 Supported Features:
        - **Classification**: Predict categories (iris species, spam detection, etc.)
        - **Regression**: Predict numbers (house prices, sales, etc.)
        - **Multiple Models**: Random Forest, Linear models, XGBoost, LightGBM
        - **Auto Preprocessing**: Missing values, encoding, scaling
        - **Hyperparameter Tuning**: Automatic optimization
        
        ---
        **👈 Start by uploading a dataset in the sidebar!**
        """)
        
        # Show sample datasets with colorful header
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #ff9a56, #ff6b6b);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin: 25px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            border: 2px solid rgba(255,255,255,0.3);
        ">
            <h3 style="color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                📂✨ Don't have data? Try our colorful samples! ✨📂
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🌸 Download Iris Classification Sample"):
                sample_data = create_sample_classification_data()
                st.download_button(
                    label="📥 Download iris_sample.csv",
                    data=sample_data.to_csv(index=False),
                    file_name="iris_sample.csv",
                    mime="text/csv"
                )
        
        with col_b:
            if st.button("🏠 Download Housing Regression Sample"):
                sample_data = create_sample_regression_data()
                st.download_button(
                    label="📥 Download housing_sample.csv", 
                    data=sample_data.to_csv(index=False),
                    file_name="housing_sample.csv",
                    mime="text/csv"
                )

def show_advanced_data_analysis_dashboard(df, target_column=None):
    """Show advanced data analysis dashboard with automated EDA"""
    
    # Initialize data analyzer
    analyzer = AdvancedDataAnalyzer(df, target_column)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 32px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid #374151;
    ">
        <h1 style="color: #f9fafb; margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em;">
            📊 Advanced Data Analysis Dashboard
        </h1>
        <p style="color: #d1d5db; margin: 12px 0 0 0; font-size: 1rem; font-weight: 400;">
            🔍 Automated EDA • 📊 Quality Assessment • 🎯 Smart Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Overview", 
        "🎯 Data Quality", 
        "📊 Correlations", 
        "🔍 Distributions", 
        "🚨 Outliers",
        "⚔️ Model Comparison"
    ])
    
    with tab1:
        show_data_overview_tab(df, analyzer)
    
    with tab2:
        show_data_quality_tab(df, analyzer)
    
    with tab3:
        show_correlations_tab(df, analyzer)
    
    with tab4:
        show_distributions_tab(df, analyzer)
    
    with tab5:
        show_outliers_tab(df, analyzer)
    
        with tab6:
            show_model_comparison_tab(df, target_column)

def show_data_overview_tab(df, analyzer):
    """Show data overview tab"""
    # Dataset overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Rows", f"{df.shape[0]:,}")
    
    with col2:
        st.metric("📋 Columns", df.shape[1])
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("❓ Missing %", f"{missing_pct:.1f}%")
    
    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("💾 Memory", f"{memory_mb:.1f} MB")
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    summary_stats = analyzer.get_summary_statistics()
    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

def show_data_quality_tab(df, analyzer):
    """Show data quality analysis tab"""
    st.subheader("🎯 Data Quality Assessment")
    
    with st.spinner("Analyzing data quality..."):
        quality_metrics = analyzer.compute_data_quality_score()
    
    # Overall quality score
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        score = quality_metrics['overall_quality_score']
        quality_level = quality_metrics['quality_level']
        
        # Quality score with color coding
        if score >= 90:
            color = "green"
            emoji = "🏆"
        elif score >= 80:
            color = "blue"
            emoji = "🎆"
        elif score >= 70:
            color = "orange"
            emoji = "⚠️"
        else:
            color = "red"
            emoji = "🚨"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1f2937, #374151);
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            border: 1px solid #374151;
        ">
            <h2 style="color: {color}; margin: 0; font-size: 3rem;">{emoji}</h2>
            <h3 style="color: #f9fafb; margin: 8px 0; font-size: 2rem;">{score:.1f}/100</h3>
            <p style="color: #d1d5db; margin: 0; font-size: 1.1rem; font-weight: 600;">{quality_level} Quality</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed quality metrics
    st.subheader("🔍 Quality Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Completeness
        completeness = quality_metrics['completeness'] * 100
        st.metric("🏡 Completeness", f"{completeness:.1f}%", 
                 help="Percentage of non-missing values")
        
        # Consistency
        consistency = quality_metrics['consistency'] * 100
        st.metric("🔄 Consistency", f"{consistency:.1f}%", 
                 help="Data type consistency")
        
        # Cardinality score
        cardinality = quality_metrics['cardinality_score'] * 100
        st.metric("🎠 Cardinality", f"{cardinality:.1f}%", 
                 help="Appropriate uniqueness levels")
    
    with col2:
        # Uniqueness
        uniqueness = quality_metrics['uniqueness'] * 100
        st.metric("🎯 Uniqueness", f"{uniqueness:.1f}%", 
                 help="Percentage of unique rows")
        
        # Validity
        validity = quality_metrics['validity'] * 100
        st.metric("✅ Validity", f"{validity:.1f}%", 
                 help="Data within expected ranges")
        
        # Overall issues
        total_issues = quality_metrics['duplicate_rows']
        st.metric("🚨 Issues Found", total_issues, 
                 help="Total data quality issues")
    
    # Smart recommendations
    st.subheader("🧪 Smart Recommendations")
    recommendations = analyzer.generate_smart_recommendations()
    
    if recommendations:
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**: {rec['description']}\n\n💡 *{rec['action']}*")
            elif rec['type'] == 'error':
                st.error(f"**{rec['title']}**: {rec['description']}\n\n💡 *{rec['action']}*")
            else:
                st.info(f"**{rec['title']}**: {rec['description']}\n\n💡 *{rec['action']}*")
    else:
        st.success("🎉 No major data quality issues detected!")

def show_correlations_tab(df, analyzer):
    """Show correlation analysis tab"""
    st.subheader("📊 Correlation Analysis")
    
    with st.spinner("Computing correlations..."):
        correlations = analyzer.analyze_correlations()
    
    # Correlation heatmap
    if 'numeric_correlation_matrix' in correlations:
        st.write("**Feature Correlation Matrix**")
        corr_fig = analyzer.create_correlation_heatmap()
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
        
        # High correlation pairs
        if correlations.get('high_correlation_pairs'):
            st.write("**🚨 Highly Correlated Features (>0.8)**")
            high_corr_df = pd.DataFrame(correlations['high_correlation_pairs'])
            high_corr_df['correlation'] = high_corr_df['correlation'].round(3)
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            
            st.warning("💡 **Tip**: Highly correlated features might cause multicollinearity. Consider removing one from each pair.")
        else:
            st.success("✅ No highly correlated feature pairs found.")
    
    # Target correlations
    if 'target_correlations' in correlations and correlations['target_correlations']:
        st.write("**🎯 Target Variable Correlations**")
        target_corr_data = correlations['target_correlations'][:15]  # Top 15
        
        # Create DataFrame for display
        target_corr_df = pd.DataFrame(target_corr_data)
        
        # Format correlation values
        if 'correlation' in target_corr_df.columns:
            target_corr_df['correlation'] = target_corr_df['correlation'].round(4)
            target_corr_df['abs_correlation'] = target_corr_df['abs_correlation'].round(4)
            target_corr_df = target_corr_df[['feature', 'correlation', 'abs_correlation']]
            target_corr_df.columns = ['Feature', 'Correlation', 'Abs Correlation']
        elif 'mutual_info_score' in target_corr_df.columns:
            target_corr_df['mutual_info_score'] = target_corr_df['mutual_info_score'].round(4)
            target_corr_df = target_corr_df[['feature', 'mutual_info_score', 'feature_type']]
            target_corr_df.columns = ['Feature', 'Mutual Info Score', 'Type']
        
        st.dataframe(target_corr_df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 No target variable specified for correlation analysis.")

def show_distributions_tab(df, analyzer):
    """Show distribution analysis tab"""
    st.subheader("🔍 Distribution Analysis")
    
    # Distribution plots
    dist_fig = analyzer.create_distribution_plots()
    if dist_fig:
        st.plotly_chart(dist_fig, use_container_width=True)
    
    # Target analysis
    target_fig = analyzer.create_target_analysis_plot()
    if target_fig:
        st.plotly_chart(target_fig, use_container_width=True)
    
    # Distribution statistics
    with st.spinner("Analyzing distributions..."):
        distributions = analyzer.analyze_distributions()
    
    if distributions:
        st.subheader("📊 Distribution Statistics")
        
        for feature, stats in distributions.items():
            with st.expander(f"📈 {feature} Distribution Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Statistics:**")
                    st.metric("Mean", f"{stats['mean']:.3f}")
                    st.metric("Median", f"{stats['median']:.3f}")
                    st.metric("Std Dev", f"{stats['std']:.3f}")
                
                with col2:
                    st.write("**Shape Analysis:**")
                    st.metric("Skewness", f"{stats['skewness']:.3f}")
                    st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
                    st.write(f"**Distribution Shape:** {stats['distribution_shape']}")
                
                # Normality test results
                if 'is_normal' in stats:
                    normality = "✅ Normal" if stats['is_normal'] else "❌ Non-normal"
                    st.write(f"**Normality Test:** {normality}")

def show_outliers_tab(df, analyzer):
    """Show outlier analysis tab"""
    st.subheader("🚨 Outlier Detection")
    
    # Outlier visualization
    outlier_fig = analyzer.create_outlier_plot()
    if outlier_fig:
        st.plotly_chart(outlier_fig, use_container_width=True)
    
    # Detailed outlier analysis
    with st.spinner("Detecting outliers..."):
        outliers = analyzer.detect_outliers()
    
    if outliers.get('iqr_outliers'):
        st.subheader("🔍 IQR-Based Outlier Analysis")
        
        outlier_summary = []
        for feature, outlier_info in outliers['iqr_outliers'].items():
            outlier_summary.append({
                'Feature': feature,
                'Outliers': outlier_info['count'],
                'Percentage': f"{outlier_info['percentage']:.2f}%",
                'Lower Bound': f"{outlier_info['bounds']['lower']:.3f}",
                'Upper Bound': f"{outlier_info['bounds']['upper']:.3f}"
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    
    # Multivariate outliers
    if outliers.get('multivariate_outliers'):
        st.subheader("🌀 Multivariate Outlier Analysis")
        mv_outliers = outliers['multivariate_outliers']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Multivariate Outliers", mv_outliers['count'])
        with col2:
            st.metric("Percentage", f"{mv_outliers['percentage']:.2f}%")
        
        if mv_outliers['percentage'] > 5:
            st.warning("💡 **High multivariate outlier rate detected.** Consider outlier treatment or robust algorithms.")
        else:
            st.success("✅ Multivariate outlier rate is within acceptable limits.")

def show_model_comparison_tab(df, target_column):
    """Show model comparison and A/B testing tab"""
    st.subheader("⚔️ Model Comparison & A/B Testing")
    
    if target_column is None:
        st.info("🎯 Please select a target column to enable model comparison. Use the sidebar configuration.")
        return
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1f2937, #374151);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        border: 1px solid #374151;
    ">
        <h3 style="color: #f9fafb; margin: 0 0 12px 0; font-weight: 600; font-size: 1.2rem;">
            ⚔️ Advanced Model Comparison
        </h3>
        <p style="color: #d1d5db; margin: 0; font-size: 0.95rem;">
            Compare multiple ML models with statistical significance testing, A/B testing framework, 
            and comprehensive performance analysis including ROC curves, learning curves, and bootstrap confidence intervals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison mode selection
    comparison_mode = st.radio(
        "Choose comparison mode:",
        [
            "🏁 Quick Model Comparison",
            "🧪 A/B Testing Framework",
            "📊 Advanced Statistical Analysis"
        ],
        help="Quick comparison trains multiple models, A/B testing compares two models with controlled experiments, Advanced analysis provides comprehensive statistical testing."
    )
    
    if "🏁 Quick Model Comparison" in comparison_mode:
        show_quick_model_comparison(df, target_column)
    elif "🧪 A/B Testing Framework" in comparison_mode:
        show_ab_testing_interface(df, target_column)
    else:
        show_advanced_statistical_analysis(df, target_column)

def show_quick_model_comparison(df, target_column):
    """Show quick model comparison interface"""
    st.write("**🏁 Quick Model Comparison**")
    st.info("Compare performance of multiple models with automated training and statistical significance testing.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Configuration:**")
        
        # Task type detection
        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        unique_values = df[target_column].nunique()
        
        if is_numeric and unique_values > 10:
            default_task = "regression"
        else:
            default_task = "classification"
        
        task_type = st.selectbox(
            "Task Type:",
            ["classification", "regression"],
            index=0 if default_task == "classification" else 1
        )
        
        cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        test_size = st.slider("Test set size:", 0.1, 0.4, 0.2)
        
        # Model selection
        available_models = {
            "Random Forest": "rf",
            "Logistic Regression / Linear Regression": "linear",
            "Gradient Boosting": "gb",
            "Support Vector Machine": "svm",
            "XGBoost": "xgb",
            "LightGBM": "lgb"
        }
        
        selected_models = st.multiselect(
            "Select models to compare:",
            list(available_models.keys()),
            default=["Random Forest", "Logistic Regression / Linear Regression", "Gradient Boosting"]
        )
    
    with col2:
        st.write("**Dataset Info:**")
        st.metric("Samples", f"{len(df):,}")
        st.metric("Features", len(df.columns) - 1)
        st.metric("Target", target_column)
        st.metric("Target Type", task_type.title())
        
        if task_type == "classification":
            classes = df[target_column].nunique()
            st.metric("Classes", classes)
        
    if st.button("🚀 Start Model Comparison", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to compare.")
            return
            
        run_model_comparison(df, target_column, task_type, cv_folds, test_size, selected_models, available_models)

def run_model_comparison(df, target_column, task_type, cv_folds, test_size, selected_models, available_models):
    """Run the actual model comparison"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    
    with st.spinner("🔄 Preparing data and training models..."):
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables (simple label encoding for demo)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_processed = X.copy()
        
        for col in categorical_cols:
            X_processed[col] = pd.factorize(X_processed[col])[0]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42,
            stratify=y if task_type == 'classification' else None
        )
        
        # Initialize comparison framework
        comparison = ModelComparison(
            task_type=task_type,
            cv_folds=cv_folds,
            random_state=42
        )
        
        # Add selected models
        for model_name in selected_models:
            model_code = available_models[model_name]
            
            if task_type == 'classification':
                if model_code == 'rf':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_code == 'linear':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_code == 'gb':
                    model = GradientBoostingClassifier(random_state=42)
                elif model_code == 'svm':
                    model = SVC(probability=True, random_state=42)
                elif model_code == 'xgb':
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier(random_state=42)
                    except ImportError:
                        st.warning(f"XGBoost not available. Skipping {model_name}.")
                        continue
                elif model_code == 'lgb':
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                    except ImportError:
                        st.warning(f"LightGBM not available. Skipping {model_name}.")
                        continue
            else:  # regression
                if model_code == 'rf':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_code == 'linear':
                    model = LinearRegression()
                elif model_code == 'gb':
                    model = GradientBoostingRegressor(random_state=42)
                elif model_code == 'svm':
                    model = SVR()
                elif model_code == 'xgb':
                    try:
                        import xgboost as xgb
                        model = xgb.XGBRegressor(random_state=42)
                    except ImportError:
                        st.warning(f"XGBoost not available. Skipping {model_name}.")
                        continue
                elif model_code == 'lgb':
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
                    except ImportError:
                        st.warning(f"LightGBM not available. Skipping {model_name}.")
                        continue
            
            comparison.add_model(model, model_name)
        
        # Run comparison
        try:
            results = comparison.compare_models(X_train, y_train, X_test, y_test)
            
            # Display results
            display_comparison_results(results, X_test, y_test, task_type, comparison)
            
        except Exception as e:
            st.error(f"Error during model comparison: {str(e)}")
            st.exception(e)

def display_comparison_results(results, X_test, y_test, task_type, comparison):
    """Display comprehensive comparison results"""
    st.success("✅ Model comparison completed successfully!")
    
    # Key metrics
    st.subheader("🏆 Comparison Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model", results.best_model)
    with col2:
        st.metric("📊 Models Compared", len(results.model_results))
    with col3:
        significant_tests = sum(1 for test in results.statistical_tests.values() if test.get('significant', False))
        st.metric("⚙️ Significant Differences", f"{significant_tests}/{len(results.statistical_tests)}")
    
    # Performance comparison table
    st.subheader("📊 Performance Comparison")
    st.dataframe(results.comparison_metrics, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.subheader("📈 Interactive Visualizations")
    
    viz = ComparisonVisualizer(dark_theme=True)
    
    # Performance comparison chart
    perf_fig = viz.create_performance_comparison_chart(results)
    st.plotly_chart(perf_fig, use_container_width=True)
    
    # Statistical significance
    if len(results.model_results) > 1:
        sig_fig = viz.create_statistical_significance_chart(results)
        st.plotly_chart(sig_fig, use_container_width=True)
    
    # ROC curves for classification
    if task_type == 'classification':
        roc_fig = viz.create_roc_curves(results.model_results, X_test, y_test)
        st.plotly_chart(roc_fig, use_container_width=True)
        
        pr_fig = viz.create_precision_recall_curves(results.model_results, X_test, y_test)
        st.plotly_chart(pr_fig, use_container_width=True)
    
    # Performance radar chart
    radar_fig = viz.create_performance_radar_chart(results)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Summary dashboard
    summary_fig = viz.create_model_comparison_summary(results)
    st.plotly_chart(summary_fig, use_container_width=True)
    
    # Statistical report
    with st.expander("📄 Detailed Statistical Report"):
        report = comparison.create_comparison_report(results)
        st.text(report)

def show_ab_testing_interface(df, target_column):
    """Show A/B testing interface"""
    st.write("**🧪 A/B Testing Framework**")
    st.info("Compare two models using controlled A/B testing with bootstrap confidence intervals and statistical significance testing.")
    
    # Coming soon placeholder
    st.warning("🚧 A/B Testing interface is under development. Available in the next update!")
    
    st.markdown("""
    **Features planned for A/B Testing:**
    - 🎯 Controlled experiment design
    - 📊 Bootstrap confidence intervals
    - ⚙️ Statistical power analysis
    - 📈 Sample size calculator
    - 📉 Effect size analysis
    - 📊 Bayesian A/B testing options
    """)

def show_advanced_statistical_analysis(df, target_column):
    """Show advanced statistical analysis interface"""
    st.write("**📊 Advanced Statistical Analysis**")
    st.info("Comprehensive statistical testing with learning curves, validation curves, and advanced model diagnostics.")
    
    # Coming soon placeholder
    st.warning("🚧 Advanced Statistical Analysis interface is under development. Available in the next update!")
    
    st.markdown("""
    **Features planned for Advanced Analysis:**
    - 📈 Learning curves analysis
    - 🎯 Validation curves for hyperparameters
    - ⚙️ Cross-validation analysis
    - 📊 Bias-variance decomposition
    - 📉 Model complexity analysis
    - 🧠 Feature importance stability
    """)
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #374151, #4b5563);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            border: 1px solid #4b5563;
        ">
            <h3 style="color: #f9fafb; margin: 0; text-align: center; font-weight: 600; font-size: 1.1rem;">
                🔢 Numerical Columns
            </h3>
        </div>
        """, unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                with st.expander(f"📊 {col}"):
                    # Create two columns for stats and visualization
                    col_left, col_right = st.columns([1, 2])
                    
                    with col_left:
                        st.write("**📈 Statistics**")
                        col_stats = df[col].describe()
                        # Format statistics nicely
                        stats_df = pd.DataFrame({
                            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                            'Value': [
                                f"{col_stats['count']:.0f}",
                                f"{col_stats['mean']:.3f}",
                                f"{col_stats['std']:.3f}",
                                f"{col_stats['min']:.3f}",
                                f"{col_stats['25%']:.3f}",
                                f"{col_stats['50%']:.3f}",
                                f"{col_stats['75%']:.3f}",
                                f"{col_stats['max']:.3f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Add data quality info
                        missing_pct = (df[col].isnull().sum() / len(df)) * 100
                        outliers = len(df[(df[col] < (col_stats['25%'] - 1.5 * (col_stats['75%'] - col_stats['25%']))) | 
                                         (df[col] > (col_stats['75%'] + 1.5 * (col_stats['75%'] - col_stats['25%'])))])
                        
                        st.write("**🔍 Data Quality**")
                        st.metric("Missing %", f"{missing_pct:.1f}%")
                        st.metric("Outliers", outliers)
                    
                    with col_right:
                        # Enhanced histogram with better styling
                        fig = px.histogram(
                            df, x=col, 
                            title=f"Distribution of {col}",
                            nbins=30,
                            marginal="box",  # Add box plot on top
                            color_discrete_sequence=['#10b981']
                        )
                        
                        # Customize layout for dark theme
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f9fafb'),
                            title_font_size=16,
                            showlegend=False
                        )
                        
                        fig.update_xaxes(
                            gridcolor='rgba(75, 85, 99, 0.3)',
                            title_font_size=14
                        )
                        fig.update_yaxes(
                            gridcolor='rgba(75, 85, 99, 0.3)',
                            title_font_size=14
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns found")
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #374151, #4b5563);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
            border: 1px solid #4b5563;
        ">
            <h3 style="color: #f9fafb; margin: 0; text-align: center; font-weight: 600; font-size: 1.1rem;">
                🗺️ Categorical Columns
            </h3>
        </div>
        """, unsafe_allow_html=True)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            for col in cat_cols:
                with st.expander(f"📋 {col}"):
                    col_left, col_right = st.columns([1, 2])
                    
                    with col_left:
                        st.write("**📊 Category Info**")
                        
                        # Basic stats
                        unique_count = df[col].nunique()
                        missing_pct = (df[col].isnull().sum() / len(df)) * 100
                        most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                        
                        stats_df = pd.DataFrame({
                            'Metric': ['Unique Values', 'Missing %', 'Most Common'],
                            'Value': [unique_count, f"{missing_pct:.1f}%", str(most_common)]
                        })
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                        
                        # Top values table
                        st.write("**🔝 Top 10 Values**")
                        value_counts = df[col].value_counts().head(10)
                        top_values_df = pd.DataFrame({
                            'Category': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': [f"{(count/len(df))*100:.1f}%" for count in value_counts.values]
                        })
                        st.dataframe(top_values_df, use_container_width=True, hide_index=True)
                    
                    with col_right:
                        # Enhanced bar chart with better colors and styling
                        value_counts = df[col].value_counts().head(15)  # Show more categories
                        
                        # Create a more sophisticated bar chart
                        fig = px.bar(
                            x=value_counts.values,
                            y=value_counts.index,
                            orientation='h',
                            title=f"Distribution of {col}",
                            labels={'x': 'Count', 'y': col},
                            color=value_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        
                        # Customize for dark theme
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f9fafb'),
                            title_font_size=16,
                            coloraxis_showscale=False,
                            height=max(400, len(value_counts) * 25)
                        )
                        
                        fig.update_xaxes(
                            gridcolor='rgba(75, 85, 99, 0.3)',
                            title_font_size=14
                        )
                        fig.update_yaxes(
                            gridcolor='rgba(75, 85, 99, 0.3)',
                            title_font_size=14
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add pie chart if not too many categories
                        if len(value_counts) <= 8:
                            st.write("**🍰 Proportion View**")
                            pie_fig = px.pie(
                                values=value_counts.values,
                                names=value_counts.index,
                                title=f"Proportion of {col}",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            
                            pie_fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f9fafb'),
                                title_font_size=14
                            )
                            
                            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No categorical columns found")
    
    # Add correlation heatmap and advanced insights
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #374151, #4b5563);
        padding: 16px;
        border-radius: 8px;
        margin: 20px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid #4b5563;
    ">
        <h3 style="color: #f9fafb; margin: 0; text-align: center; font-weight: 600; font-size: 1.1rem;">
            📊 Advanced Data Insights
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            st.write("**🔥 Correlation Heatmap**")
            correlation_matrix = numeric_df.corr()
            
            # Create enhanced heatmap
            fig_heatmap = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            
            fig_heatmap.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                title_font_size=14,
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Need at least 2 numerical columns for correlation analysis")
    
    with col2:
        # Data quality overview
        st.write("**🔍 Data Quality Overview**")
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': [f"{(count/len(df))*100:.1f}%" for count in missing_data.values]
            })
            
            # Missing data bar chart
            fig_missing = px.bar(
                missing_df,
                x='Missing %',
                y='Column',
                orientation='h',
                title='Missing Data by Column',
                color='Missing Count',
                color_continuous_scale='Reds'
            )
            
            fig_missing.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                title_font_size=14,
                coloraxis_showscale=False,
                height=max(300, len(missing_data) * 30)
            )
            
            fig_missing.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
            fig_missing.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
            
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing data detected!")
        
        # Data types summary
        st.write("**📋 Data Types Summary**")
        dtype_counts = df.dtypes.value_counts()
        dtype_df = pd.DataFrame({
            'Data Type': dtype_counts.index.astype(str),
            'Count': dtype_counts.values
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

def train_automl_model(df, target_column, task_type, time_limit, validation_split, show_code, show_plots, mode_config="balanced"):
    """Train AutoML model and show results"""
    
    # Store the training request in session state to show results in main area
    st.session_state.training_requested = True
    st.session_state.training_params = {
        'df': df,
        'target_column': target_column,
        'task_type': task_type,
        'time_limit': time_limit,
        'validation_split': validation_split,
        'show_code': show_code,
        'show_plots': show_plots,
        'mode_config': mode_config
    }
    
    # Trigger rerun to show training in main area
    st.rerun()

def show_results(automl, results, df, target_column, task_type, show_code, show_plots):
    """Show comprehensive results from AutoML training"""
    
    st.header("🎉 AutoML Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Best Model", results['best_model_name'])
    
    with col2:
        metric_name = "Accuracy" if task_type == "classification" else "R² Score"
        st.metric(f"📊 {metric_name}", f"{results['best_score']:.4f}")
    
    with col3:
        st.metric("⏱️ Training Time", f"{results['training_time']:.1f}s")
    
    with col4:
        st.metric("🔢 Features Used", results['feature_count'])
    
    # Performance details
    st.subheader("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📋 Performance Metrics:**")
        metrics = results['metrics']
        
        # Show key metrics based on task type
        if task_type == "classification":
            key_metrics = ['cv_mean_score', 'cv_accuracy_mean', 'cv_precision_macro_mean', 'cv_recall_macro_mean', 'cv_f1_macro_mean']
        else:
            key_metrics = ['cv_mean_score', 'cv_r2_mean', 'cv_mean_squared_error_mean', 'cv_mean_absolute_error_mean']
        
        metrics_df = pd.DataFrame([
            {"Metric": k.replace('cv_', '').replace('_mean', '').replace('_', ' ').title(), 
             "Value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)}
            for k, v in metrics.items() if k in key_metrics
        ])
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**🌟 Feature Importance Analysis:**")
        feature_importance = results['feature_importance'].head(15)  # Show more features
        
        # Enhanced feature importance plot
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title="Top 15 Most Important Features",
            labels={'importance': 'Importance Score', 'feature': 'Features'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        # Professional styling
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb', size=12),
            title_font_size=16,
            coloraxis_showscale=False,
            height=max(400, len(feature_importance) * 25)
        )
        
        fig.update_xaxes(
            gridcolor='rgba(75, 85, 99, 0.3)',
            title_font_size=14,
            tickfont_size=11
        )
        fig.update_yaxes(
            gridcolor='rgba(75, 85, 99, 0.3)',
            title_font_size=14,
            tickfont_size=11
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add feature importance table
        st.write("**📊 Top Features Table:**")
        importance_table = feature_importance.copy()
        importance_table['Importance %'] = (importance_table['importance'] / importance_table['importance'].sum() * 100).round(2)
        importance_table['Rank'] = range(1, len(importance_table) + 1)
        display_table = importance_table[['Rank', 'feature', 'Importance %']].rename(columns={'feature': 'Feature'})
        st.dataframe(display_table, use_container_width=True, hide_index=True)
    
    # Model Interpretability & Explainability
    show_interpretability_analysis(automl, df, target_column, task_type, results)
    
    # Visualizations
    if show_plots:
        show_visualizations(automl, df, target_column, task_type, results)
    
    # Generated Code
    if show_code:
        show_generated_code(results, target_column, task_type)
    
    # Model download
    st.subheader("💾 Download Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Save Model"):
            try:
                model_path = "trained_automl_model"
                automl.save_model(model_path)
                st.success(f"✅ Model saved to {model_path}/")
                st.info("The model has been saved locally. You can use the generated code to load and use it.")
            except Exception as e:
                st.error(f"❌ Failed to save model: {e}")
    
    with col2:
        # Create downloadable results summary
        results_summary = create_results_summary(results, task_type)
        st.download_button(
            label="📊 Download Results Summary",
            data=results_summary,
            file_name="automl_results_summary.txt",
            mime="text/plain"
        )

def show_visualizations(automl, df, target_column, task_type, results):
    """Show comprehensive visualizations"""
    
    st.subheader("📊 Advanced Model Visualizations")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🎯 Target Distribution**")
        if task_type == "classification":
            # Enhanced classification target distribution
            target_counts = df[target_column].value_counts()
            fig = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title=f"Distribution of {target_column}",
                labels={'x': target_column, 'y': 'Count'},
                color=target_counts.values,
                color_continuous_scale='Plasma'
            )
        else:
            # Enhanced regression target distribution with KDE
            fig = px.histogram(
                df, x=target_column, 
                title=f"Distribution of {target_column}", 
                nbins=30,
                marginal="violin",  # Add violin plot
                color_discrete_sequence=['#10b981']
            )
        
        # Apply dark theme styling
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f9fafb'),
            title_font_size=14,
            coloraxis_showscale=False
        )
        fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
        fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**📈 Feature-Target Relationships**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        if numeric_cols and pd.api.types.is_numeric_dtype(df[target_column]):
            # Enhanced correlation analysis for numeric targets
            correlations = []
            for col in numeric_cols[:10]:  # Top 10 features
                try:
                    corr = df[col].corr(df[target_column])
                    if not pd.isna(corr):
                        correlations.append({
                            "Feature": col, 
                            "Correlation": corr,
                            "Abs_Correlation": abs(corr)
                        })
                except:
                    continue
            
            if correlations:
                corr_df = pd.DataFrame(correlations).sort_values("Abs_Correlation", ascending=False)
                
                # Enhanced correlation plot with better colors
                fig = px.bar(
                    corr_df, 
                    x="Correlation", 
                    y="Feature", 
                    orientation='h',
                    title="Feature Correlation with Target",
                    color="Correlation",
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f9fafb'),
                    title_font_size=14,
                    coloraxis_showscale=True
                )
                fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Enhanced box plots for categorical targets
            if numeric_cols and len(numeric_cols) > 0:
                # Show distribution of top correlated feature by target categories
                first_numeric = numeric_cols[0]
                fig = px.box(
                    df, x=target_column, y=first_numeric, 
                    title=f"Distribution of {first_numeric} by {target_column}",
                    color=target_column,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f9fafb'),
                    title_font_size=14
                )
                fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric features available for analysis")
    
    with col3:
        st.write("**🏆 Model Performance Insights**")
        
        # Get model metrics
        metrics = results['metrics']
        best_model = results['best_model_name']
        
        # Create performance radar chart for classification
        if task_type == "classification" and 'cv_accuracy_mean' in metrics:
            import math
            
            # Performance metrics for radar chart
            perf_metrics = {
                'Accuracy': metrics.get('cv_accuracy_mean', 0),
                'Precision': metrics.get('cv_precision_macro_mean', 0),
                'Recall': metrics.get('cv_recall_macro_mean', 0),
                'F1-Score': metrics.get('cv_f1_macro_mean', 0)
            }
            
            # Create radar chart
            categories = list(perf_metrics.keys())
            values = list(perf_metrics.values())
            
            # Add first value at end to close the circle
            categories += [categories[0]]
            values += [values[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=best_model,
                line=dict(color='#10b981', width=2),
                fillcolor='rgba(16, 185, 129, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='rgba(75, 85, 99, 0.3)',
                        tickfont=dict(color='#f9fafb', size=10)
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#f9fafb', size=11),
                        gridcolor='rgba(75, 85, 99, 0.3)'
                    )
                ),
                showlegend=False,
                title=f"Performance Radar: {best_model}",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f9fafb'),
                title_font_size=14,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # For regression, show a different performance visualization
            perf_data = {
                'Metric': [],
                'Score': []
            }
            
            key_metrics = ['cv_r2_mean', 'cv_mean_squared_error_mean', 'cv_mean_absolute_error_mean']
            metric_names = ['R² Score', 'MSE', 'MAE']
            
            for key, name in zip(key_metrics, metric_names):
                if key in metrics:
                    perf_data['Metric'].append(name)
                    perf_data['Score'].append(metrics[key])
            
            if perf_data['Metric']:
                perf_df = pd.DataFrame(perf_data)
                fig = px.bar(
                    perf_df,
                    x='Score',
                    y='Metric',
                    orientation='h',
                    title=f"Performance: {best_model}",
                    color='Score',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f9fafb'),
                    title_font_size=14,
                    coloraxis_showscale=False,
                    height=300
                )
                fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("📊 Performance metrics will appear here")
    
    # Enhanced missing values analysis
    if df.isnull().sum().sum() > 0:
        st.markdown("""
        <div style="margin: 30px 0; padding: 20px; background: linear-gradient(135deg, #374151, #4b5563); border-radius: 12px; border: 1px solid #4b5563;">
            <h4 style="color: #f9fafb; margin: 0 0 15px 0; text-align: center;">🔍 Missing Data Pattern Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Missing Values by Column**")
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing_Count']
            missing_data = missing_data[missing_data['Missing_Count'] > 0]
            missing_data['Missing_Percentage'] = (missing_data['Missing_Count'] / len(df)) * 100
            
            if len(missing_data) > 0:
                fig = px.bar(
                    missing_data, 
                    x='Missing_Percentage', 
                    y='Column',
                    orientation='h',
                    title='Missing Data Percentage by Column',
                    color='Missing_Count',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f9fafb'),
                    title_font_size=14,
                    coloraxis_showscale=False,
                    height=max(300, len(missing_data) * 30)
                )
                fig.update_xaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                fig.update_yaxes(gridcolor='rgba(75, 85, 99, 0.3)')
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Missing data heatmap
            st.write("**🔥 Missing Data Heatmap**")
            missing_matrix = df.isnull().astype(int)
            
            if len(missing_matrix.columns) <= 20:  # Only show heatmap for reasonable number of columns
                fig_heatmap = px.imshow(
                    missing_matrix.T,
                    aspect="auto",
                    title="Missing Values Pattern",
                    color_continuous_scale='Reds',
                    labels={'x': 'Rows', 'y': 'Columns'}
                )
                
                fig_heatmap.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f9fafb'),
                    title_font_size=14,
                    height=400
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Too many columns for heatmap visualization")

def show_generated_code(results, target_column, task_type):
    """Show generated Python code"""
    
    st.subheader("💻 Generated Python Code")
    
    # Training code
    st.markdown('<div class="code-header">🚀 Model Training Code</div>', unsafe_allow_html=True)
    
    training_code = f"""
# AutoML Model Training Code
# Generated automatically by AutoML System

import pandas as pd
from pathlib import Path
import sys

# Add AutoML to path (adjust path as needed)
sys.path.append('path/to/automl/src')
from automl import AutoML

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # or pd.read_excel('your_dataset.xlsx')

# Initialize AutoML
automl = AutoML(
    target='{target_column}',
    task_type='{task_type}',
    time_limit=300,  # 5 minutes
    validation_split=0.2,
    verbose=True
)

# Train the model
print("Training AutoML model...")
automl.fit(df)

# Get results
results = automl.get_results()
print(f"Best model: {{results['best_model_name']}}")
print(f"Best score: {{results['best_score']:.4f}}")

# Save the model
automl.save_model('my_automl_model')
"""
    
    st.code(training_code, language='python')
    
    # Prediction code
    st.markdown('<div class="code-header">🔮 Prediction Code</div>', unsafe_allow_html=True)
    
    prediction_code = f"""
# Making Predictions with Trained Model

# Option 1: Use the trained AutoML object
new_data = pd.read_csv('new_data.csv')
predictions = automl.predict(new_data)
print("Predictions:", predictions)

# Option 2: Load saved model
from automl import AutoML
loaded_automl = AutoML.load_model('my_automl_model')
predictions = loaded_automl.predict(new_data)

# For classification: Get prediction probabilities
{'if task_type == "classification":' if task_type == 'classification' else '# '}
{'probabilities = automl.predict_proba(new_data)' if task_type == 'classification' else '# probabilities = automl.predict_proba(new_data)'}

# Get feature importance
feature_importance = automl.get_feature_importance()
print("Top features:")
print(feature_importance.head(10))
"""
    
    st.code(prediction_code, language='python')
    
    # Model details code
    st.markdown('<div class="code-header">📊 Model Analysis Code</div>', unsafe_allow_html=True)
    
    analysis_code = f"""
# Detailed Model Analysis

# Get comprehensive results
results = automl.get_results()

# Performance metrics
metrics = results['metrics']
print("Performance Metrics:")
for metric, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"{{metric}}: {{value:.4f}}")

# Feature importance
feature_importance = results['feature_importance']
print("\\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{{i+1}}. {{row['feature']}}: {{row['importance']:.4f}}")

# Best model details
best_model = automl.get_best_model()
print(f"\\nBest Model Type: {{type(best_model).__name__}}")
print(f"Model Parameters: {{best_model.get_params()}}")

# Training history
history = results['training_history']
print(f"\\nTraining completed in {{results['training_time']:.2f}} seconds")
print(f"Features used: {{results['feature_count']}}")
"""
    
    st.code(analysis_code, language='python')
    
    # Requirements
    st.markdown('<div class="code-header">📦 Requirements</div>', unsafe_allow_html=True)
    
    requirements = """
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
openpyxl>=3.1.0  # For Excel files

# Optional (for better models)
# xgboost>=1.7.0
# lightgbm>=4.0.0
# catboost>=1.2.0
"""
    
    st.code(requirements, language='text')

def show_interpretability_analysis(automl, df, target_column, task_type, results):
    """Show comprehensive model interpretability analysis"""
    
    st.subheader("🧠 Model Interpretability & Explainability")
    
    if not hasattr(automl, 'interpreter') or automl.interpreter is None:
        st.warning("⚠️ Model interpreter not available. This might happen with certain model types.")
        return
    
    # Create tabs for different interpretability analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 SHAP Analysis", 
        "🔍 LIME Explanations", 
        "📊 Feature Analysis", 
        "🧪 What-If Analysis"
    ])
    
    with tab1:
        st.write("**🎯 SHAP (SHapley Additive exPlanations) Analysis**")
        st.info("SHAP explains model predictions by computing the contribution of each feature to the prediction.")
        
        try:
            # Global feature importance
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Global Feature Importance**")
                max_features = st.slider("Number of features to display", 5, 20, 15, key="shap_features")
                
                with st.spinner("Computing SHAP values..."):
                    shap_fig = automl.analyze_feature_importance('shap', max_features)
                    if shap_fig:
                        st.plotly_chart(shap_fig, use_container_width=True)
                    else:
                        st.error("Failed to generate SHAP analysis")
            
            with col2:
                st.write("**Individual Prediction Explanation**")
                
                # Sample selection for individual explanation
                sample_idx = st.selectbox(
                    "Select a sample to explain:",
                    options=range(min(10, len(df))),
                    format_func=lambda x: f"Sample {x+1}",
                    key="shap_sample"
                )
                
                if st.button("Explain This Prediction", key="explain_shap"):
                    with st.spinner("Generating SHAP explanation..."):
                        try:
                            # Get the sample (excluding target column)
                            sample_data = df.drop(columns=[target_column]).iloc[sample_idx]
                            explanation = automl.explain_prediction(sample_data, method='shap')
                            
                            if 'shap_plot' in explanation and explanation['shap_plot']:
                                st.plotly_chart(explanation['shap_plot'], use_container_width=True)
                                
                                # Show actual vs predicted
                                actual = df[target_column].iloc[sample_idx]
                                predicted = automl.predict(pd.DataFrame([sample_data]))[0]
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Actual Value", f"{actual}")
                                with col_b:
                                    st.metric("Predicted Value", f"{predicted:.4f}")
                            else:
                                st.error("Failed to generate SHAP explanation")
                                
                        except Exception as e:
                            st.error(f"Error generating explanation: {e}")
        
        except Exception as e:
            st.error(f"SHAP analysis error: {e}")
    
    with tab2:
        st.write("**🔍 LIME (Local Interpretable Model-agnostic Explanations)**")
        st.info("LIME explains individual predictions by approximating the model locally with an interpretable model.")
        
        try:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Settings**")
                lime_sample_idx = st.selectbox(
                    "Select sample for LIME explanation:",
                    options=range(min(10, len(df))),
                    format_func=lambda x: f"Sample {x+1}",
                    key="lime_sample"
                )
                
                num_features = st.slider("Features to explain", 3, 15, 8, key="lime_features")
                
                if st.button("Generate LIME Explanation", key="explain_lime"):
                    with st.spinner("Computing LIME explanation..."):
                        try:
                            sample_data = df.drop(columns=[target_column]).iloc[lime_sample_idx]
                            explanation = automl.explain_prediction(sample_data, method='lime')
                            
                            if 'lime_plot' in explanation and explanation['lime_plot']:
                                st.session_state.lime_explanation = explanation
                            else:
                                st.error("Failed to generate LIME explanation")
                        except Exception as e:
                            st.error(f"LIME error: {e}")
            
            with col2:
                if 'lime_explanation' in st.session_state:
                    explanation = st.session_state.lime_explanation
                    st.plotly_chart(explanation['lime_plot'], use_container_width=True)
                    
                    # Show prediction details
                    if 'lime_data' in explanation:
                        lime_data = explanation['lime_data']
                        st.write(f"**Prediction:** {lime_data['prediction']:.4f}")
                        
                        # Feature contributions table
                        contrib_df = pd.DataFrame({
                            'Feature': lime_data['features'],
                            'Impact': lime_data['impacts']
                        })
                        contrib_df['Impact'] = contrib_df['Impact'].round(4)
                        st.write("**Feature Contributions:**")
                        st.dataframe(contrib_df, hide_index=True)
        
        except Exception as e:
            st.error(f"LIME analysis error: {e}")
    
    with tab3:
        st.write("**📊 Advanced Feature Analysis**")
        
        # Feature selection for analysis
        available_features = automl.feature_names if hasattr(automl, 'feature_names') else list(df.columns)
        if target_column in available_features:
            available_features.remove(target_column)
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Partial Dependence", "Feature Interactions"],
            key="feature_analysis_type"
        )
        
        if analysis_type == "Partial Dependence":
            st.write("**Partial Dependence Analysis**")
            st.info("Shows how a feature affects predictions on average, marginalizing over other features.")
            
            selected_feature = st.selectbox(
                "Select feature for partial dependence:",
                available_features,
                key="pd_feature"
            )
            
            if st.button("Generate Partial Dependence Plot", key="pd_plot"):
                with st.spinner("Computing partial dependence..."):
                    try:
                        pd_fig = automl.plot_partial_dependence(selected_feature)
                        if pd_fig:
                            st.plotly_chart(pd_fig, use_container_width=True)
                        else:
                            st.error("Failed to generate partial dependence plot")
                    except Exception as e:
                        st.error(f"Partial dependence error: {e}")
        
        elif analysis_type == "Feature Interactions":
            st.write("**Feature Interaction Analysis**")
            st.info("Shows how two features interact to affect predictions.")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox(
                    "First feature:",
                    available_features,
                    key="interaction_f1"
                )
            with col2:
                feature2 = st.selectbox(
                    "Second feature:",
                    available_features,
                    key="interaction_f2"
                )
            
            if feature1 != feature2 and st.button("Analyze Interaction", key="interaction_plot"):
                with st.spinner("Computing feature interaction..."):
                    try:
                        interaction_fig = automl.analyze_feature_interactions(feature1, feature2)
                        if interaction_fig:
                            st.plotly_chart(interaction_fig, use_container_width=True)
                        else:
                            st.error("Failed to generate interaction plot")
                    except Exception as e:
                        st.error(f"Feature interaction error: {e}")
    
    with tab4:
        st.write("**🧪 What-If Analysis**")
        st.info("Explore how changing a feature value affects the prediction for a specific instance.")
        
        try:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Configuration**")
                
                # Base instance selection
                base_sample_idx = st.selectbox(
                    "Select base sample:",
                    options=range(min(10, len(df))),
                    format_func=lambda x: f"Sample {x+1}",
                    key="whatif_sample"
                )
                
                # Feature to analyze
                whatif_feature = st.selectbox(
                    "Feature to analyze:",
                    available_features,
                    key="whatif_feature"
                )
                
                # Get current feature value
                base_instance = df.drop(columns=[target_column]).iloc[base_sample_idx]
                current_value = base_instance[whatif_feature]
                
                st.write(f"**Current value:** {current_value}")
                
                # Value range for analysis
                if pd.api.types.is_numeric_dtype(df[whatif_feature]):
                    feature_min = float(df[whatif_feature].min())
                    feature_max = float(df[whatif_feature].max())
                    feature_range = feature_max - feature_min
                    
                    # Create range around current value
                    range_min = max(feature_min, current_value - feature_range * 0.3)
                    range_max = min(feature_max, current_value + feature_range * 0.3)
                    
                    st.write("**Analysis Range:**")
                    start_val = st.number_input("Start value", value=float(range_min), key="whatif_start")
                    end_val = st.number_input("End value", value=float(range_max), key="whatif_end")
                    num_steps = st.slider("Number of steps", 5, 20, 10, key="whatif_steps")
                    
                    if st.button("Run What-If Analysis", key="run_whatif"):
                        if start_val < end_val:
                            values_to_test = np.linspace(start_val, end_val, num_steps)
                            
                            with st.spinner("Running what-if analysis..."):
                                try:
                                    whatif_fig = automl.what_if_analysis(
                                        base_instance, whatif_feature, values_to_test.tolist()
                                    )
                                    if whatif_fig:
                                        st.session_state.whatif_result = whatif_fig
                                    else:
                                        st.error("Failed to generate what-if analysis")
                                except Exception as e:
                                    st.error(f"What-if analysis error: {e}")
                        else:
                            st.error("Start value must be less than end value")
                else:
                    st.warning("What-if analysis is currently only supported for numeric features")
            
            with col2:
                if 'whatif_result' in st.session_state:
                    st.plotly_chart(st.session_state.whatif_result, use_container_width=True)
        
        except Exception as e:
            st.error(f"What-if analysis error: {e}")
            
def create_sample_classification_data():
    """Create sample classification dataset"""
    np.random.seed(42)
    n_samples = 150
    
    data = []
    species = ['setosa', 'versicolor', 'virginica']
    
    for i, species_name in enumerate(species):
        for _ in range(n_samples // 3):
            if species_name == 'setosa':
                sepal_length = np.random.normal(5.0, 0.4)
                sepal_width = np.random.normal(3.4, 0.4) 
                petal_length = np.random.normal(1.5, 0.2)
                petal_width = np.random.normal(0.2, 0.1)
            elif species_name == 'versicolor':
                sepal_length = np.random.normal(6.0, 0.5)
                sepal_width = np.random.normal(2.8, 0.3)
                petal_length = np.random.normal(4.2, 0.5) 
                petal_width = np.random.normal(1.3, 0.2)
            else:  # virginica
                sepal_length = np.random.normal(6.5, 0.6)
                sepal_width = np.random.normal(3.0, 0.3)
                petal_length = np.random.normal(5.5, 0.6)
                petal_width = np.random.normal(2.0, 0.3)
                
            data.append({
                'sepal_length': sepal_length,
                'sepal_width': sepal_width, 
                'petal_length': petal_length,
                'petal_width': petal_width,
                'species': species_name
            })
    
    return pd.DataFrame(data)

def create_sample_regression_data():
    """Create sample regression dataset"""
    np.random.seed(42)
    n_samples = 200
    
    data = []
    for _ in range(n_samples):
        bedrooms = np.random.randint(1, 6)
        bathrooms = np.random.randint(1, 4)
        sqft = np.random.normal(1800 + bedrooms * 300, 400)
        age = np.random.randint(0, 50)
        location = np.random.choice(['downtown', 'suburb', 'rural'])
        
        location_multiplier = {'downtown': 1.5, 'suburb': 1.0, 'rural': 0.7}[location]
        base_price = 150000 + sqft * 80 + bedrooms * 20000 + bathrooms * 15000
        age_discount = max(0, 1 - age * 0.01)
        price = base_price * location_multiplier * age_discount
        price *= np.random.normal(1.0, 0.1)
        price = max(price, 50000)
        
        data.append({
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft': sqft,
            'age': age,
            'location': location,
            'price': round(price, -3)
        })
    
    return pd.DataFrame(data)

def create_results_summary(results, task_type):
    """Create downloadable results summary"""
    summary = f"""
AutoML Results Summary
=====================

Best Model: {results['best_model_name']}
Task Type: {task_type}
Best Score: {results['best_score']:.4f}
Training Time: {results['training_time']:.2f} seconds
Features Used: {results['feature_count']}

Performance Metrics:
"""
    
    for metric, value in results['metrics'].items():
        if isinstance(value, (int, float)):
            summary += f"  {metric}: {value:.4f}\n"
    
    summary += f"""
Top 10 Feature Importance:
"""
    
    for i, (_, row) in enumerate(results['feature_importance'].head(10).iterrows()):
        summary += f"  {i+1}. {row['feature']}: {row['importance']:.4f}\n"
    
    return summary

def show_experiment_history_dashboard():
    """Show experiment history dashboard with browsing and comparison tools"""
    try:
        # Initialize experiment tracker
        tracker = ExperimentTracker()
        history = ExperimentHistory(tracker)
        
        # Get all experiments
        all_experiments = tracker.db.list_experiments(limit=100)
        
        if not all_experiments:
            st.info("📋 No experiments found. Train some models first to see experiment history!")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Experiment Browser", 
            "🔍 Search & Filter", 
            "⚖️ Compare Experiments", 
            "📊 Performance Trends"
        ])
        
        with tab1:
            show_experiment_browser(all_experiments, tracker)
        
        with tab2:
            show_experiment_search(history)
        
        with tab3:
            show_experiment_comparison(all_experiments, history)
        
        with tab4:
            show_performance_trends(history)
            
    except Exception as e:
        st.error(f"❌ Failed to load experiment history: {e}")
        st.info("📝 Try running some AutoML experiments first to build up history.")

def show_experiment_browser(experiments, tracker):
    """Show experiment browser with detailed view"""
    st.subheader("📋 Experiment History")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🧪 Total Experiments", len(experiments))
    
    with col2:
        classification_count = sum(1 for exp in experiments if exp.get('task_type') == 'classification')
        st.metric("🎯 Classification", classification_count)
    
    with col3:
        regression_count = sum(1 for exp in experiments if exp.get('task_type') == 'regression')
        st.metric("📊 Regression", regression_count)
    
    with col4:
        completed_count = sum(1 for exp in experiments if exp.get('status') == 'completed')
        st.metric("✅ Completed", completed_count)
    
    st.divider()
    
    # Recent experiments table
    st.subheader("🕰️ Recent Experiments")
    
    # Format experiment data for display
    display_data = []
    for exp in experiments[:20]:  # Show latest 20
        display_data.append({
            'ID': exp['experiment_id'][:8] + '...',
            'Name': exp['experiment_name'],
            'Task Type': exp.get('task_type', 'Unknown').title(),
            'Dataset': exp.get('dataset_name', 'Unknown'),
            'Status': exp.get('status', 'Unknown').title(),
            'Duration': f"{exp.get('duration', 0):.1f}s",
            'Date': pd.to_datetime(exp['timestamp']).strftime('%Y-%m-%d %H:%M') if exp.get('timestamp') else 'Unknown'
        })
    
    if display_data:
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Experiment details
        st.subheader("🔍 Experiment Details")
        
        selected_exp_idx = st.selectbox(
            "Select experiment to view details:",
            options=range(len(experiments[:20])),
            format_func=lambda x: f"{experiments[x]['experiment_name']} ({experiments[x]['experiment_id'][:8]}...)",
            key="exp_selector"
        )
        
        if selected_exp_idx is not None:
            selected_exp_id = experiments[selected_exp_idx]['experiment_id']
            show_experiment_details(selected_exp_id, tracker)
    else:
        st.info("No experiments available")

def show_experiment_details(experiment_id, tracker):
    """Show detailed view of a single experiment"""
    try:
        experiment = tracker.db.load_experiment(experiment_id)
        if not experiment:
            st.error("Experiment not found")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            st.write(f"**Name:** {experiment.config.experiment_name}")
            st.write(f"**Task Type:** {experiment.config.task_type.title()}")
            st.write(f"**Dataset:** {experiment.config.dataset_name}")
            st.write(f"**Target Column:** {experiment.config.target_column}")
            st.write(f"**Status:** {experiment.status.title()}")
            st.write(f"**Duration:** {experiment.duration:.1f}s")
            
        with col2:
            st.markdown("**Performance Metrics**")
            st.write(f"**Primary Metric:** {experiment.metrics.primary_metric_name}")
            st.write(f"**Primary Score:** {experiment.metrics.primary_metric:.4f}")
            
            if experiment.metrics.additional_metrics:
                for name, value in list(experiment.metrics.additional_metrics.items())[:5]:
                    st.write(f"**{name}:** {value:.4f}")
        
        # Parameters
        if experiment.parameters:
            st.markdown("**Parameters**")
            param_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)} 
                for k, v in experiment.parameters.items()
            ])
            st.dataframe(param_df, hide_index=True, use_container_width=True)
        
        # Model file info
        if experiment.artifacts.model_path:
            st.markdown("**Model Artifact**")
            st.info(f"💾 Model saved at: {experiment.artifacts.model_path}")
            
    except Exception as e:
        st.error(f"Error loading experiment details: {e}")

def show_experiment_search(history):
    """Show experiment search and filtering interface"""
    st.subheader("🔍 Advanced Search & Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text search
        search_query = st.text_input(
            "🔍 Search experiments",
            placeholder="Search in experiment names and descriptions...",
            help="Search for text in experiment names and descriptions"
        )
        
        # Task type filter
        task_type_filter = st.selectbox(
            "🎯 Task Type Filter",
            options=['All', 'Classification', 'Regression'],
            help="Filter experiments by task type"
        )
    
    with col2:
        # Status filter
        status_filter = st.selectbox(
            "✅ Status Filter",
            options=['All', 'Completed', 'Failed', 'Running'],
            help="Filter experiments by completion status"
        )
        
        # Date range
        date_filter = st.selectbox(
            "📅 Date Range",
            options=['All Time', 'Last 24 Hours', 'Last Week', 'Last Month'],
            help="Filter experiments by date range"
        )
    
    # Apply filters button
    if st.button("🔍 Apply Filters", type="primary"):
        with st.spinner("Searching experiments..."):
            try:
                # Build filters
                filters = {}
                if task_type_filter != 'All':
                    filters['task_type'] = task_type_filter.lower()
                if status_filter != 'All':
                    filters['status'] = status_filter.lower()
                
                # Date range
                date_range = None
                if date_filter != 'All Time':
                    from datetime import datetime, timedelta
                    now = datetime.now()
                    if date_filter == 'Last 24 Hours':
                        start_date = now - timedelta(days=1)
                    elif date_filter == 'Last Week':
                        start_date = now - timedelta(weeks=1)
                    elif date_filter == 'Last Month':
                        start_date = now - timedelta(days=30)
                    date_range = (start_date, now)
                
                # Search
                results = history.search_experiments(
                    query=search_query if search_query else None,
                    filters=filters,
                    date_range=date_range,
                    limit=50
                )
                
                st.success(f"✅ Found {len(results)} experiments matching your criteria")
                
                if results:
                    # Display results
                    display_data = []
                    for exp in results:
                        display_data.append({
                            'Name': exp['experiment_name'],
                            'Task Type': exp.get('task_type', 'Unknown').title(),
                            'Status': exp.get('status', 'Unknown').title(),
                            'Duration': f"{exp.get('duration', 0):.1f}s",
                            'Date': pd.to_datetime(exp['timestamp']).strftime('%Y-%m-%d %H:%M') if exp.get('timestamp') else 'Unknown'
                        })
                    
                    df_results = pd.DataFrame(display_data)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"Search failed: {e}")

def show_experiment_comparison(experiments, history):
    """Show experiment comparison interface"""
    st.subheader("⚖️ Compare Experiments")
    
    if len(experiments) < 2:
        st.warning("Need at least 2 experiments for comparison")
        return
    
    # Select experiments to compare
    st.markdown("**Select experiments to compare:**")
    
    exp_options = [f"{exp['experiment_name']} ({exp['experiment_id'][:8]}...)" for exp in experiments[:20]]
    
    selected_experiments = st.multiselect(
        "Choose experiments (select 2-5):",
        options=exp_options,
        help="Select between 2 and 5 experiments for comparison"
    )
    
    if len(selected_experiments) >= 2:
        if st.button("⚖️ Compare Selected Experiments", type="primary"):
            with st.spinner("Comparing experiments..."):
                try:
                    # Get experiment IDs
                    selected_indices = [exp_options.index(exp) for exp in selected_experiments]
                    experiment_ids = [experiments[i]['experiment_id'] for i in selected_indices]
                    
                    # Get comparison data
                    comparison_data = history.compare_experiments(experiment_ids)
                    
                    if comparison_data:
                        st.success("✅ Comparison completed!")
                        
                        # Performance comparison
                        st.subheader("🏆 Performance Comparison")
                        
                        perf_data = []
                        for exp in comparison_data['performance_ranking']:
                            perf_data.append({
                                'Rank': len(perf_data) + 1,
                                'Experiment': exp.config.experiment_name,
                                'Primary Metric': f"{exp.metrics.primary_metric:.4f}",
                                'Training Time': f"{exp.duration:.1f}s",
                                'Task Type': exp.config.task_type.title()
                            })
                        
                        df_perf = pd.DataFrame(perf_data)
                        st.dataframe(df_perf, use_container_width=True, hide_index=True)
                        
                        # Parameter differences
                        if comparison_data['different_parameters']:
                            st.subheader("🔧 Parameter Differences")
                            
                            param_data = []
                            for param, values in comparison_data['different_parameters'].items():
                                row = {'Parameter': param}
                                for exp_id, value in values.items():
                                    exp_name = next((exp.config.experiment_name for exp in comparison_data['performance_ranking'] 
                                                   if exp.experiment_id == exp_id), exp_id[:8])
                                    row[exp_name] = str(value)
                                param_data.append(row)
                            
                            if param_data:
                                df_params = pd.DataFrame(param_data)
                                st.dataframe(df_params, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
    
    elif len(selected_experiments) == 1:
        st.info("Please select at least one more experiment for comparison")

def show_performance_trends(history):
    """Show performance trends and analytics"""
    st.subheader("📊 Performance Trends & Analytics")
    
    try:
        analytics = ExperimentAnalytics(history)
        
        # Performance timeline
        st.markdown("**Performance Over Time**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task_type_filter = st.selectbox(
                "Task Type:",
                options=[None, 'classification', 'regression'],
                format_func=lambda x: 'All' if x is None else x.title(),
                key="trends_task_filter"
            )
        
        with col2:
            if st.button("📊 Generate Performance Timeline", type="primary"):
                with st.spinner("Generating timeline..."):
                    try:
                        timeline_fig = analytics.create_performance_timeline(
                            task_type=task_type_filter
                        )
                        if timeline_fig:
                            st.plotly_chart(timeline_fig, use_container_width=True)
                        else:
                            st.info("No data available for the selected filters")
                    except Exception as e:
                        st.error(f"Failed to generate timeline: {e}")
        
        # Metrics history
        st.markdown("**Metrics History**")
        
        try:
            metrics_df = history.get_experiment_metrics_history(limit=50)
            if not metrics_df.empty:
                # Group by metric name for display
                unique_metrics = metrics_df['metric_name'].unique()
                
                if len(unique_metrics) > 0:
                    selected_metric = st.selectbox(
                        "Select metric to analyze:",
                        options=unique_metrics,
                        key="metric_selector"
                    )
                    
                    metric_data = metrics_df[metrics_df['metric_name'] == selected_metric]
                    
                    if not metric_data.empty:
                        # Create simple visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=metric_data['timestamp'],
                            y=metric_data['metric_value'],
                            mode='lines+markers',
                            name=selected_metric,
                            line=dict(color='#10b981', width=3),
                            marker=dict(size=8, color='#10b981')
                        ))
                        
                        fig.update_layout(
                            title=f'{selected_metric.title()} Over Time',
                            xaxis_title='Date',
                            yaxis_title=selected_metric.replace('_', ' ').title(),
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Best Score", f"{metric_data['metric_value'].max():.4f}")
                        with col2:
                            st.metric("Average Score", f"{metric_data['metric_value'].mean():.4f}")
                        with col3:
                            st.metric("Latest Score", f"{metric_data['metric_value'].iloc[-1]:.4f}")
                        with col4:
                            improvement = metric_data['metric_value'].iloc[-1] - metric_data['metric_value'].iloc[0] if len(metric_data) > 1 else 0
                            st.metric("Improvement", f"{improvement:+.4f}")
            else:
                st.info("No metrics data available")
                
        except Exception as e:
            st.error(f"Failed to load metrics history: {e}")
            
    except Exception as e:
        st.error(f"Analytics error: {e}")

def show_analytics_dashboard():
    """Show advanced analytics dashboard"""
    try:
        # Initialize analytics components
        tracker = ExperimentTracker()
        history = ExperimentHistory(tracker)
        analytics = ExperimentAnalytics(history)
        
        # Get experiments for analysis
        all_experiments = tracker.db.list_experiments(limit=100)
        
        if not all_experiments:
            st.info("📋 No experiments found. Train some models first to see analytics!")
            return
        
        st.subheader("📈 Advanced Analytics")
        
        # Analytics tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Performance Analytics",
            "🔧 Hyperparameter Analysis", 
            "📋 Export & Reports"
        ])
        
        with tab1:
            show_performance_analytics(analytics, history)
        
        with tab2:
            show_hyperparameter_analysis(history)
        
        with tab3:
            show_export_reports(history, all_experiments)
            
    except Exception as e:
        st.error(f"❌ Failed to load analytics: {e}")
        st.info("📝 Try running some AutoML experiments first to generate analytics.")

def show_performance_analytics(analytics, history):
    """Show performance analytics section"""
    st.markdown("**Performance Analysis & Trends**")
    
    # Performance timeline with more options
    col1, col2 = st.columns(2)
    
    with col1:
        task_filter = st.selectbox(
            "Filter by task type:",
            options=[None, 'classification', 'regression'],
            format_func=lambda x: 'All Tasks' if x is None else x.title(),
            key="perf_task_filter"
        )
    
    with col2:
        time_range = st.selectbox(
            "Time range:",
            options=['All Time', 'Last 7 Days', 'Last 30 Days'],
            key="perf_time_range"
        )
    
    if st.button("📊 Generate Advanced Timeline", type="primary"):
        with st.spinner("Generating advanced performance timeline..."):
            try:
                # Calculate date range
                date_range = None
                if time_range != 'All Time':
                    from datetime import datetime, timedelta
                    now = datetime.now()
                    if time_range == 'Last 7 Days':
                        start_date = now - timedelta(days=7)
                    elif time_range == 'Last 30 Days':
                        start_date = now - timedelta(days=30)
                    date_range = (start_date, now)
                
                timeline_fig = analytics.create_performance_timeline(
                    task_type=task_filter,
                    date_range=date_range
                )
                
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.info("No data available for the selected filters")
                    
            except Exception as e:
                st.error(f"Failed to generate timeline: {e}")
    
    # Best performing experiments
    st.markdown("**🏆 Top Performing Experiments**")
    
    try:
        experiments = history.search_experiments(limit=10, sort_by="timestamp", ascending=False)
        if experiments:
            # Get detailed experiment info and sort by performance
            detailed_exps = []
            for exp in experiments[:20]:  # Limit to prevent long loading
                try:
                    full_exp = history.tracker.db.load_experiment(exp['experiment_id'])
                    if full_exp and hasattr(full_exp.metrics, 'primary_metric'):
                        detailed_exps.append(full_exp)
                except:
                    continue
            
            if detailed_exps:
                # Sort by primary metric (descending for better scores)
                detailed_exps.sort(key=lambda x: x.metrics.primary_metric, reverse=True)
                
                # Display top 5
                top_data = []
                for i, exp in enumerate(detailed_exps[:5]):
                    top_data.append({
                        'Rank': i + 1,
                        'Experiment': exp.config.experiment_name,
                        'Task': exp.config.task_type.title(),
                        'Primary Metric': f"{exp.metrics.primary_metric:.4f}",
                        'Training Time': f"{exp.duration:.1f}s",
                        'Dataset': exp.config.dataset_name
                    })
                
                df_top = pd.DataFrame(top_data)
                st.dataframe(df_top, use_container_width=True, hide_index=True)
                
    except Exception as e:
        st.error(f"Failed to load top experiments: {e}")

def show_hyperparameter_analysis(history):
    """Show hyperparameter analysis section"""
    st.markdown("**🔧 Hyperparameter Impact Analysis**")
    
    st.info("🚧 Hyperparameter analysis is coming soon! This will show which parameters have the biggest impact on model performance.")
    
    # For now, show parameter distribution
    try:
        experiments = history.search_experiments(limit=50)
        if experiments:
            st.markdown("**Parameter Distribution Across Experiments**")
            
            # Collect all parameters
            all_params = {}
            for exp in experiments[:10]:  # Limit for performance
                try:
                    full_exp = history.tracker.db.load_experiment(exp['experiment_id'])
                    if full_exp:
                        for param, value in full_exp.parameters.items():
                            if param not in all_params:
                                all_params[param] = []
                            all_params[param].append(str(value))
                except:
                    continue
            
            if all_params:
                param_summary = []
                for param, values in all_params.items():
                    unique_values = len(set(values))
                    most_common = max(set(values), key=values.count) if values else "N/A"
                    param_summary.append({
                        'Parameter': param,
                        'Unique Values': unique_values,
                        'Most Common': most_common,
                        'Total Uses': len(values)
                    })
                
                df_params = pd.DataFrame(param_summary)
                st.dataframe(df_params, use_container_width=True, hide_index=True)
                
    except Exception as e:
        st.error(f"Parameter analysis error: {e}")

def show_export_reports(history, experiments):
    """Show export and reporting section"""
    st.markdown("**📋 Export & Reporting**")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export format:",
            options=['CSV', 'JSON'],
            help="Choose format for exporting experiment data"
        )
    
    with col2:
        include_params = st.checkbox("Include parameters", value=True)
        include_metrics = st.checkbox("Include metrics", value=True)
    
    if st.button("📥 Export Experiment Data", type="primary"):
        with st.spinner("Preparing export..."):
            try:
                # Get experiment IDs
                experiment_ids = [exp['experiment_id'] for exp in experiments]
                
                # Export data
                export_data = history.export_experiments(
                    experiment_ids=experiment_ids,
                    export_format=export_format.lower(),
                    include_parameters=include_params,
                    include_metrics=include_metrics
                )
                
                if export_format.lower() == 'csv':
                    csv_data = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="automl_experiments.csv",
                        mime="text/csv"
                    )
                    st.success("✅ CSV export prepared!")
                    
                elif export_format.lower() == 'json':
                    st.download_button(
                        label="📥 Download JSON",
                        data=export_data,
                        file_name="automl_experiments.json",
                        mime="application/json"
                    )
                    st.success("✅ JSON export prepared!")
                    
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    # Summary report
    st.markdown("**📋 Summary Report**")
    
    if experiments:
        # Generate summary statistics
        total_experiments = len(experiments)
        completed = sum(1 for exp in experiments if exp.get('status') == 'completed')
        classification_count = sum(1 for exp in experiments if exp.get('task_type') == 'classification')
        regression_count = sum(1 for exp in experiments if exp.get('task_type') == 'regression')
        
        # Create summary
        summary_text = f"""
# AutoML Experiment Summary Report

## Overview
- **Total Experiments:** {total_experiments}
- **Completed:** {completed} ({completed/total_experiments*100:.1f}%)
- **Classification Tasks:** {classification_count}
- **Regression Tasks:** {regression_count}

## Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_text,
            file_name="automl_summary_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()

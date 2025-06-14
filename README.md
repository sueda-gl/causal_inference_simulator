# 🎯 Recommendation Impact Simulator

**Advanced Causal Inference Engine for AI Visibility Analysis**

A comprehensive Streamlit-based application that uses advanced causal inference techniques to analyze the true impact of marketing actions on brand visibility in AI recommendations. This simulator separates correlation from causation to provide actionable insights for marketing strategy optimization.


## 🌟 Features

### 🔍 **What-If Analysis**
- Predict the impact of marketing actions for specific brands
- Interactive brand and action selection
- Real-time ROI calculations with confidence intervals
- Business scenario modeling

### 📈 **Causal Discovery**
- Multiple causal inference methods (DoWhy, EconML)
- Treatment effect estimation with statistical validation
- Confidence intervals and p-value calculations
- Ground truth comparison for validation

### 🎯 **Heterogeneous Effects Analysis**
- Subgroup analysis by brand characteristics
- Effect modification detection
- Statistical significance testing
- Personalized treatment recommendations

### 💰 **Business Impact Analysis**
- Revenue and ROI projections
- Payback period calculations
- Scenario comparison (optimistic, pessimistic, competitive)
- Interactive business assumption modeling

### 📋 **Action Recommendations**
- AI-powered optimal action selection
- Budget constraint optimization
- Implementation roadmaps
- Monitoring and KPI tracking plans

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd recommendation-impact-simulator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser:**
Navigate to `http://localhost:8501` to access the application.

## 📦 Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning algorithms

### Causal Inference
- **dowhy**: Causal inference library
- **econml**: Heterogeneous treatment effects
- **statsmodels**: Statistical modeling

### Visualization
- **plotly**: Interactive plotting
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization

### Data Validation
- **pydantic**: Data validation and settings management

### Utilities
- **loguru**: Advanced logging
- **python-dotenv**: Environment variable management

## 🏗️ Project Structure

```
recommendation-impact-simulator/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── .env                          # Environment variables (create this)
├── src/                          # Source code
│   ├── __init__.py
│   ├── business/                 # Business logic
│   │   ├── recommender.py       # Action recommendation engine
│   │   └── roi_calculator.py    # ROI calculation logic
│   ├── causal/                  # Causal inference
│   │   ├── engine.py           # Main causal inference engine
│   │   └── effects.py          # Heterogeneous effects analysis
│   ├── config/                 # Configuration
│   │   └── settings.py         # Application settings
│   ├── data/                   # Data handling
│   │   ├── generator.py        # Synthetic data generation
│   │   └── schemas.py          # Data validation schemas
│   ├── prediction/             # Prediction models
│   │   └── predictor.py        # What-if prediction engine
│   ├── utils/                  # Utilities
│   │   ├── helpers.py          # Helper functions
│   │   └── logger.py           # Logging configuration
│   └── visualization/          # Charts and dashboards
│       ├── dashboards.py       # Dashboard components
│       └── plots.py            # Plotting functions
└── logs/                       # Application logs (auto-generated)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Application Settings
APP_NAME="Recommendation Impact Simulator"
APP_VERSION="0.1.0"
DEBUG=True

# Data Settings
DEFAULT_N_BRANDS=50
DEFAULT_N_TIME_PERIODS=52
DEFAULT_RANDOM_SEED=42

# Analysis Settings
DEFAULT_CONFIDENCE_LEVEL=0.95
DEFAULT_BOOTSTRAP_ITERATIONS=100

# Logging
LOG_LEVEL=INFO
ENABLE_FILE_LOGGING=True
```

### Application Settings

Modify `src/config/settings.py` to customize default parameters:

```python
class Settings(BaseSettings):
    # Adjust these values based on your needs
    n_brands: int = 50
    n_time_periods: int = 52
    confidence_level: float = 0.95
    bootstrap_iterations: int = 100
    # ... other settings
```

## 📊 Usage Guide

### 1. **Data Generation**
The application automatically generates synthetic data with embedded causal relationships. You can adjust:
- Number of brands (10-100)
- Time period in weeks (20-104)
- Random seed for reproducibility

### 2. **What-If Analysis**
1. Select a brand from the dropdown
2. Choose marketing actions to analyze
3. Set business assumptions (search volume, conversion rates, costs)
4. Run analysis to see predicted impact and ROI

### 3. **Causal Discovery**
1. Select treatments to analyze
2. Choose outcome variable
3. Run causal analysis to discover true causal effects
4. Compare with ground truth for validation

### 4. **Heterogeneous Effects**
1. Complete causal discovery first
2. Select brand characteristics for subgroup analysis
3. Identify which brands benefit most from each treatment

### 5. **Business Impact**
1. Set business scenario parameters
2. Calculate ROI for each treatment
3. Compare scenarios (optimistic, pessimistic, competitive)
4. Export results for reporting

### 6. **Action Recommendations**
1. Select target brand
2. Set budget constraints
3. Generate AI-powered recommendations
4. Review implementation roadmap and monitoring plan

## 🔬 Technical Details

### Causal Inference Methods

The application implements multiple causal inference approaches:

1. **Difference-in-Differences**: For time-series analysis
2. **Propensity Score Matching**: For observational data
3. **Instrumental Variables**: When available
4. **Linear Regression**: Baseline comparison
5. **Inverse Probability Weighting**: For confounding control

### Statistical Validation

- Bootstrap confidence intervals
- Permutation tests
- Cross-validation
- Sensitivity analysis

### Data Quality Checks

- Treatment balance assessment
- Missing data handling
- Outlier detection
- Covariate balance validation

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Reduce dataset size in sidebar
   n_brands = 25  # Instead of 50
   n_weeks = 26   # Instead of 52
   ```

3. **Slow Performance**
   ```python
   # Reduce bootstrap iterations
   bootstrap_iterations = 50  # Instead of 100
   ```

4. **Path Issues**
   - Ensure you're running from the project root directory
   - Check that all `__init__.py` files are present

### Debug Mode

Enable debug mode in `.env`:
```bash
DEBUG=True
LOG_LEVEL=DEBUG
```

## 📈 Performance Optimization

### For Large Datasets
- Use data sampling for initial exploration
- Reduce bootstrap iterations for faster results
- Consider parallel processing for multiple treatments

### Memory Management
- Monitor memory usage with large brand datasets
- Use data chunking for very large time series
- Clear Streamlit cache if needed: `Ctrl+R` or restart app

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Submit a pull request**

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd recommendation-impact-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/  # If tests are available

# Run the application
streamlit run app.py
```



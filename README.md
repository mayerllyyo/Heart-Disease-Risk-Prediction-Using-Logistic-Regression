# Heart Disease Risk Prediction Using Logistic Regression

This project implements logistic regression from scratch for heart disease prediction. The assignment focuses on understanding the mathematical foundations of binary classification by building a complete pipeline: data preprocessing, gradient descent optimization, decision boundary visualization, L2 regularization, and AWS SageMaker deployment. Rather than using black-box libraries, we implement core algorithms to understand how logistic regression learns patterns from clinical data.

## Dataset Description

Dataset: Heart Disease UCI  
Source: [UCI Machine Learning Repository via Kaggle](https://www.kaggle.com/datasets/neurocipher/heartdisease)  
Size: 270 patients (after cleaning)  
Classes: 2 (Disease Presence: 44.44%, Disease Absence: 55.56%)

### Features

The dataset includes 13 clinical features, of which 6 were selected for model training:

- Age: Patient age in years (29-77)
- Cholesterol: Serum cholesterol in mg/dL (126-564)
- BP: Resting blood pressure in mm Hg (94-200)
- Max_HR: Maximum heart rate achieved (71-202)
- ST_depression: ST depression induced by exercise (0.0-6.2)
- Number_of_vessels_fluro: Number of major vessels colored by fluoroscopy (0-3)

Additional features in the dataset include chest pain type, EKG results, fasting blood sugar, exercise angina, slope of ST segment, and thallium stress test results.

### Data Preprocessing

- Train/Test Split: 70/30 stratified split (189 train, 81 test)
- Normalization: Z-score standardization applied to all features
- Target Encoding: Binary (1 = Disease Presence, 0 = Disease Absence)
- Missing Values: None detected in the cleaned dataset



## Model Implementation

### Logistic Regression from Scratch

Mathematical Foundation:

Sigmoid Function:
```
σ(z) = 1 / (1 + e^(-z))
where z = w·x + b
```

Cost Function (Binary Cross-Entropy):
```
J(w,b) = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Gradient Descent Updates:
```
w := w - α·(1/m)·X^T·(ŷ - y)
b := b - α·(1/m)·Σ(ŷ - y)
```

Hyperparameters:
- Learning Rate (α): 0.01
- Iterations: 1000
- Convergence: Final cost = 0.4407

Implementation Details:
- Pure NumPy implementation (no scikit-learn for training)
- Vectorized operations for efficiency
- Cost tracking across iterations for convergence monitoring



## Results Summary

### Model Performance (Without Regularization)

| Metric     | Training Set | Test Set |
|---------|--------------|----------|
| Accuracy   | 77.78%       | 71.60%   |
| Precision  | 79.17%       | 72.41%   |
| Recall     | 67.86%       | 58.33%   |
| F1-Score   | 73.08%       | 64.62%   |

Confusion Matrix (Test Set):
- True Positives (TP): 21
- False Positives (FP): 8
- False Negatives (FN): 15
- True Negatives (TN): 37

Optimized Weights:
```
Age:              0.0659
Cholesterol:      0.1797
BP:               0.4177
Max_HR:          -0.6337
ST_depression:    0.6916
Vessels:          0.7049
Bias (b):        -0.1934
```

Interpretation: 
- Negative weight for Max_HR indicates lower heart rates correlate with disease risk
- Strong positive weights for ST_depression and vessels suggest these are powerful predictors
- BP shows moderate positive correlation with risk


## Decision Boundary Visualization

Three feature pairs were analyzed to understand model separability:

### 1. Age vs Cholesterol
Boundary Equation: `0.0659·Age + 0.1797·Cholesterol - 0.1934 = 0`

Observations:
- Moderate linear separability
- Higher cholesterol levels (>250 mg/dL) increase disease likelihood
- Age shows weaker individual correlation but combines with cholesterol effectively

### 2. BP vs Max_HR
Boundary Equation: `0.4177·BP - 0.6337·Max_HR - 0.1934 = 0`

Observations:
- Strongest separation among the three pairs
- Clear inverse relationship: low Max_HR at high BP indicates high risk
- Decision boundary effectively separates classes with minimal overlap

### 3. ST_depression vs Number_of_vessels_fluro
Boundary Equation: `0.6916·ST_depression + 0.7049·Vessels - 0.1934 = 0`

Observations:
- Best linear separability
- Both features are powerful predictors (highest weights in the model)
- ST_depression > 1.5 combined with vessels ≥ 1 strongly indicates disease

Conclusion: While no single feature pair perfectly separates classes, the model learns effective decision boundaries by combining weighted contributions from all six features.

## Regularization Experiments

### Methodology
L2 regularization was applied to prevent overfitting by penalizing large weights:

Regularized Cost:
```
J_reg(w,b) = J(w,b) + (λ/2m)·||w||²
```

Regularized Gradient:
```
dw += (λ/m)·w
```

### Results

| Lambda (λ) | Test Accuracy | Weight Magnitude (||w||) |
|------------|---------------|--------------------------|
| 0.000      | 75.31%        | 1.5458                   |
| 0.001      | 75.31%        | 1.5457                   |
| 0.010      | 75.31%        | 1.5452                   |
| 0.100      | 75.31%        | 1.5400                   |
| 1.000      | 75.31%        | 1.4905                   |

Findings:
- Test accuracy remained stable across all λ values (~75.3%)
- Weight magnitude decreased with higher λ (1.55 → 1.49)
- λ = 1.0 provides best generalization potential due to weight shrinkage
- No significant overfitting in baseline model (train: 77.78%, test: 75.31%)

Optimal Choice: λ = 0.1 balances regularization strength without over-constraining the model.

## Deployment in SageMaker

The trained logistic regression model was deployed using AWS SageMaker as a real-time inference endpoint.  
The deployment process consisted of the following steps:

### 1. Model Creation
The trained model artifacts and inference script were packaged and registered as a SageMaker model.

![](assets/deploy_model.png)

### 2. Endpoint Configuration and Creation
The registered model was selected and deployed as a real-time endpoint using a managed instance.

![](assets/select_our_model.png)

### 3. Endpoint Testing
The deployed endpoint was tested using a JSON request to verify correct inference behavior.

![](assets/test_endpoint.png)

### Sample Inference

**Input**
```json
{
  "features": [55, 250, 140, 160, 1.2, 1]
}
```

**Output**

```json
{
  "probability": 0.62,
  "prediction": 1,
  "risk_level": "HIGH RISK"
}
```

The endpoint returns the predicted probability of heart disease, along with a binary classification and a human-readable risk interpretation.


## Technical Stack

- Python 3.x - Core programming language
- NumPy - Matrix operations and gradient descent implementation
- Pandas - Data manipulation and exploratory analysis
- Matplotlib - Visualization of distributions and decision boundaries
- Amazon SageMaker - Model deployment and endpoint management
- Jupyter Notebook - Interactive development environment


## How to Run

### Local Execution
```bash
# Clone repository
git clone https://github.com/mayerllyyo/heart-disease-lr.git
cd heart-disease-lr

# Install dependencies
pip install numpy pandas matplotlib jupyter

# Launch notebook
jupyter notebook heart_disease_lr_analysis.ipynb
```

### AWS SageMaker Deployment
1. Upload `heart_disease_lr_analysis.ipynb` to SageMaker Studio
2. Execute cells 1-10 to train the model
3. Run deployment cells (11-13) to create the endpoint
4. Test with custom patient data in the final inference cell


## Author

- **Mayerlly Suárez Correa** [mayerllyyo](https://github.com/mayerllyyo)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
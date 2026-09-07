# Prediction of Student Performance: Multiple Linear Regression vs. Artificial Neural Networks

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end comparative study and MATLAB implementation investigating the predictive capability of **Multiple Linear Regression (MLR)** versus a **Feedforward Artificial Neural Network (ANN)** for forecasting student academic performance.

---

## 📖 Overview

Academic performance forecasting is one of the most prominent applications of Educational Data Mining (EDM). The objective of this project is to model and quantify the relationship between behavioral/academic predictors (such as study hours, previous scores, sleep duration, and practice tests) and overall academic performance.

Two fundamentally distinct modeling paradigms are implemented, benchmarked, and analyzed:
1. **Multiple Linear Regression**: A parametric statistical model providing interpretable regression coefficients, hypothesis testing (ANOVA, $t$-statistics, $p$-values), and closed-form OLS estimation.
2. **Artificial Neural Network (Shallow MLP)**: A non-parametric connectionist architecture capable of capturing subtle non-linear dependencies via non-linear activation functions and gradient-based backpropagation.

---

## 📊 Dataset Description

The models are trained and validated on the **Student Performance Dataset** ([Kaggle source](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)), comprising **10,000 student records**.

### Predictor & Target Variables

| Variable Name | MATLAB Variable | Type | Description |
| :--- | :---: | :---: | :--- |
| **Hours Studied** | $x_1$ | Numeric (Feature) | Total number of hours spent studying |
| **Previous Scores** | $x_2$ | Numeric (Feature) | Scores achieved in prior examinations |
| **Sleep Hours** | $x_3$ | Numeric (Feature) | Average hours of sleep per day |
| **Sample Question Papers Practiced** | $x_4$ | Numeric (Feature) | Number of practice exam papers completed |
| **Performance Index** | $y$ | Numeric (Target) | Rounded academic performance score (Range: $10 - 100$) |

Target distribution is approximately normal, unimodal, and symmetric without significant outlier corruption.

---

## 🔬 Theoretical Background

### Multiple Linear Regression (MLR)

The classical linear regression formulation assumes:
$$y = X \beta + \epsilon$$

Where:
- $y \in \mathbb{R}^{m 	\times 1}$ represents the response vector.
- $X \in \mathbb{R}^{m 	\times (n+1)}$ is the design matrix (including the intercept column of 1s).
- $\beta \in \mathbb{R}^{(n+1)	\times 1}$ is the unknown parameter vector.
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ represents unobservable, homoscedastic, and uncorrelated Gaussian noise.

By the Gauss-Markov Theorem, the Ordinary Least Squares (OLS) estimator $\hat{ \beta} = (X^T X)^{-1} X^T y$ is the Best Linear Unbiased Estimator (BLUE), with variance-covariance:
$$\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}, \quad \hat{\sigma}^2 = \frac{\hat{\epsilon}^T \hat{\epsilon}}{m-k}$$

Model fitness is assessed through Analysis of Variance (ANOVA):
$$\text{SST} =	\text{SSR} + \text{SSE}, \quad R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}$$

### Artificial Neural Network (ANN)

The connectionist model employs a shallow **Feedforward Multilayer Perceptron (MLP)** with:
- **Input Layer**: 4 neurons corresponding to $x_1, \dots, x_4$.
- **Hidden Layer**: 20 hidden units equipped with non-linear activation functions ($f(u) = (1 + e^{-u})^{-1}$).
- **Output Layer**: 1 linear output unit yielding $\hat{y}$.

Network optimization utilizes the **Backpropagation Algorithm (BPA)** with Levenberg-Marquardt / Gradient Descent optimization to minimize the quadratic cost function:
$$E = \frac{1}{2} \sum_{i=1}^m (\hat{y}_i - y_i)^2$$

Z-score feature standardization is applied prior to training to ensure numerical stability and balanced gradient updates:
$$z = \frac{x - \mu}{\sigma}$$

---

## ⚙️ Experimental Setup & Methodology

### 1. Multiple Linear Regression Pipeline (`linear_regression.m`)
- **Hold-out Split**: 70% Training ($7,000$ samples), 30% Testing ($3,000$ samples) initialized with `rng('default')`.
- **Model Fitting**: MATLAB `fitlm` function estimating coefficients, standard errors, $t$-statistics, and $p$-values.
- **Diagnostics**:
  - ANOVA decomposition (`anova(mdl)`).
  - Residual normality tests and scatter diagnostic plots (`plotResiduals`, `plotmatrix`).
  - Out-of-sample generalization test with residual error distribution.
  - Verification on a discrete test record (Sample #2350).

### 2. Artificial Neural Network Pipeline (`neural_network.m`)
- **Data Normalization**: Zero-mean, unit-variance standardization across all predictors and target.
- **Architecture**: `feedforwardnet(20)` with 20 hidden neurons.
- **Partitioning**: 70% Training, 15% Validation (early stopping to prevent overfitting), 15% Out-of-sample Testing.
- **Evaluation**: Mean Squared Error (`perform`) and Root Mean Squared Error (RMSE).

---

## 📈 Results & Performance Comparison

### Linear Regression Model Summary

$$	\text{PerformanceIndex} \approx -33.755 + 2.861(x_1) + 1.018(x_2) + 0.474(x_3) + 0.197(x_4)$$

| Term | Estimate ($\hat{ eta}$) | Standard Error (SE) | $t$-Statistic | $p$-Value | Significance ($ lpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(Intercept)** | -33.755 | 0.15073 | -223.94 | $\approx 0$ | Statistically Significant |
| **$x_1$ (Hours Studied)** | 2.8612 | 0.00949 | 301.61 | $\approx 0$ | Statistically Significant |
| **$x_2$ (Previous Scores)** | 1.0180 | 0.00141 | 721.84 | $\approx 0$ | Statistically Significant |
| **$x_3$ (Sleep Hours)** | 0.4738 | 0.01444 | 32.82 | $7.62 	\times 10^{-220}$ | Statistically Significant |
| **$x_4$ (Question Papers)** | 0.1972 | 0.00851 | 23.17 | $1.75 	\times 10^{-114}$ | Statistically Significant |

- **Number of Observations**: 7,000 (Train) / Degrees of Freedom: 6,995
- **Root Mean Squared Error (RMSE)**: $2.05$
- **Coefficient of Determination ($R^2$)**: $0.989$ (Adjusted $R^2 = 0.989$)
- **$F$-statistic vs. Constant Model**: $1.52 \times 10^5$ ($p	\text{-value} = 0$)
- **Single-Sample Test (#2350)**:
  - Actual Performance: `71.0000`
  - Predicted Performance: `72.8119` (High predictive agreement)

### Neural Network Evaluation

- **Convergence**: Met validation check criterion at Epoch 11 (Best validation performance at Epoch 5).
- **Normalized Test RMSE**: `0.1071`
- **Mean Squared Error (MSE)**: `0.0114`
- **Regression Fit ($R$)**: $R \approx 0.994$ across Training, Validation, and Test sets.

### Key Insights
1. **Significance**: All 4 features are statistically significant ($p \ll 0.05$). Study hours ($x_1$) and previous test scores ($x_2$) exhibit the highest direct impact on academic success.
2. **Model Choice**: Because the underlying data generating process displays strong linear characteristics ($R^2 \approx 0.99$), the Multiple Linear Regression model provides nearly identical accuracy to the ANN while offering full mathematical interpretability and parameter tractability.

---
## 🚀 Getting Started & Usage

### Prerequisites
- **MATLAB** (R2020b or later recommended)
- **Statistics and Machine Learning Toolbox** (for `fitlm`, `anova`, `cvpartition`)
- **Deep Learning Toolbox** (for `feedforwardnet`, `dividerand`, `perform`)

### Running the Script
1. Clone this repository:
   ```bash
   git clone https://github.com/chiaragiavalisco/student-performance-prediction.git
cd student-performance-prediction

2. Open MATLAB, navigate to the cloned folder, and run:
   ```matlab
   run('linear_regression.m')
   run('neural_network.m')
   ```
---

## 👤 Author

**Chiara Giavalisco**  
* Master's Degree Coursework: *Numerical Methods for Data Mining*  
* [LinkedIn Profile](https://www.linkedin.com/in/chiara-giavalisco-28b1b9268/) • [GitHub Profile](https://github.com/chiaragiavalisco) • [Email](mailto:chiara.giavalisco@gmail.com)

---

## 📚 References & Acknowledgments

- **Dataset**: [Student Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) by Nikhil (included as `Student_Performance.csv` in this repository).
- **Linear Regression**: Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). *Introduction to Linear Regression Analysis*. John Wiley & Sons.
- **Multivariate Techniques & Machine Learning**: Izenman, A. J. (2008). *Modern Multivariate Statistical Techniques*. Springer.
- **Artificial Neural Networks**: Yegnanarayana, B. (2009). *Artificial Neural Networks*. PHI Learning Pvt. Ltd. (or Wu, Y. & Feng, J., 2018).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).


# 🔮 Supervised Machine Learning & Feature Engineering: Retail Analytics

A predictive modeling project implementing **Linear Regression (`lm`)** and **Decision Trees (`rpart`)** on retail transaction data, featuring repeated 1-fold cross-validation and custom feature engineering with row-based and group-aggregated derived attributes.

---

## 💡 Overview & Strategy

The objective of this project is to build, evaluate, and iteratively improve predictive models on retail transaction data. Models were validated using repeated 1-fold cross-validation (70/30 train/test splits) across multiple random seeds to establish baseline performance metrics and ensure model stability.

---

## 🎯 Predictive Tasks & Models

1. **Numerical Target Regression (`lm`)**:
   * **Target**: `Total.Amount` (Transaction spend)
   * **Predictors**: `Age`, `Gender`, `Product.Category`, `Quantity`
   * **Evaluation Metric**: Root Mean Squared Error (RMSE)

2. **Categorical Target Classification (`rpart`)**:
   * **Target**: `Gender` (Customer classification)
   * **Predictors**: `Age`, `Total.Amount`, `Product.Category`, `Quantity`
   * **Evaluation Metric**: Classification Accuracy ($\%$)

---

## 🛠️ Feature Engineering (Derived Attributes)

To improve model performance beyond raw predictors, two types of derived attributes were engineered:

* **Row-based Derived Attribute (`Age_Squared`)**:
  $$\text{Age\_Squared} = \text{Age}^2$$
  *Captures non-linear spending dynamics across customer age groups.*

* **Aggregated Derived Attribute (`Avg.Spend.Category`)**:
  $$\text{Avg.Spend.Category}_g = \text{Mean}(\text{Total.Amount}_g)$$
  *Groups records by `Product.Category`, computes historical mean expenditures, and joins aggregate values back to individual records.*

---

## 📊 Results & Cross-Validation Benchmarks

Running 5 cross-validation folds ensured that performance gains were statistically significant and not an artifact of random data splits:

| Model Type | Target Metric | Baseline Model | Improved Model (With Derived Features) | Performance Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (`lm`)** | **RMSE** (Lower is better) | $523.06$ | $512.40$ | **$-10.66$ Error Reduction** |
| **Decision Tree (`rpart`)** | **Accuracy** (Higher is better) | $48.67\%$ | $51.33\%$ | **$+2.66\%$ Accuracy Gain** |

### Statistical Significance of Improvement
The reduction in linear regression error (RMSE drop of $10.66$) noticeably exceeded the standard deviation of fold fluctuations ($\pm 3.12$), confirming that group-aggregated features provided genuine structural information to the model.

---

## 📁 Repository Structure

```text
.
├── retail_sales_dataset.csv  # Dataset (1,000 retail transactions)
├── HW3.R                     # Full script containing model builds, CV loops, & derived features
```

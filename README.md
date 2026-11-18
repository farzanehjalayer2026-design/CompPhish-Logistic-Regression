# CompPhish – Phishing Website Detection Using Logistic Regression

## Project Overview

This academic project investigates the application of Logistic Regression to phishing website detection using the CompPhish dataset, with an emphasis on feature selection, classification performance, and model interpretability.

**Author:** Farzaneh Sadat Jalayer  
**Supervisor:** Dr. Mohammad Reza Soltan Aghaei  
**Course:** Machine Learning  
**Program:** Ph.D. in Software Engineering  
**University:** Islamic Azad University, Isfahan (Khorasgan) Branch  
**Year:** 2026

## Dataset

- Original dataset: CompPhish
- Total records: 15,358
- Original features: 70
- Candidate features after initial exclusion: 68
- Selected features for the reduced model: 20
- Experimental sample size: 1,000
- Phishing samples: 500
- Legitimate samples: 500
- Training set: 800 samples (80%)
- Testing set: 200 samples (20%)
- Random state: 42

## Methodology

1. Balanced random sampling
2. Data preprocessing
3. Feature standardization
4. L1-based feature selection
5. Logistic Regression classification
6. Five-fold stratified cross-validation
7. Performance evaluation
8. Confusion matrix analysis

## Experimental Results

The previously reported results for the reduced 20-feature model are:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 94.00% |
| Precision | 94.90% |
| Recall | 93.00% |
| F1-Score | 93.94% |
| Specificity | 95.00% |

## Confusion Matrix

| Actual / Predicted | Legitimate | Phishing |
|---|---:|---:|
| Legitimate | 95 | 5 |
| Phishing | 7 | 93 |

## Feature Selection

The reduced model uses 20 of the 68 candidate features, representing approximately a 70.59% reduction in input dimensionality.

The objective is to balance classification performance, model simplicity, and interpretability.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Google Colab
- Jupyter Notebook

## Limitations and Future Work

- Limited experimental sample size
- Need for independent external validation
- Evaluation against emerging phishing techniques
- Future comparison with KNN, Decision Tree, Bayesian classifiers, and ensemble learning methods

## Reference

Goenka, R., Chawla, M., & Tiwari, N. (2026). CompPhish: A comprehensive dataset for diversified phishing detection. Data in Brief, 69, 113219.

https://doi.org/10.1016/j.dib.2026.113219

## Acknowledgments

Special thanks to Dr. Mohammad Reza Soltan Aghaei for his academic guidance.

This project is intended for academic and educational purposes.

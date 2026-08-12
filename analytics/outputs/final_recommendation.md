## Final Recommendation

Recommended classifier: **Logistic Regression**. It has the strongest combined F1/AUC in the reported comparison (F1=0.734, AUC=0.861).

Classification and regression metrics are kept as separate metric groups because they measure different tasks and are not directly comparable. For deployment, the complete saved classification pipeline is preferred because preprocessing and the final estimator are stored together.
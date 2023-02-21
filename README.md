# prostate_cancer_genetic_association_risk_pred

Extreme gradient boosting machine learning algorithm identifies genome-wide interacting genetic variants in prostate cancer  

I applied the XGBoost algorithm and an iterative SNP selection algorithm to find the top features (SNPs) that best predict the risk of developing prostate cancer with a Support Vector Machine (SVM). 

The number of subjects was 907 and input features were 1,798,727 after appropriate quality control. 
The algorithm involved 10 trials of 5-fold cross-validation for optimization of the dataset's hyperparameters and the prediction task's second module (utilizing SVM). The model achieved AUC-ROC cure of 0.66, 0.57 and 0.55 on the Train, Dev and Test sets respectively.

The area under the Precision-Recall Curve was 0.69, 0.60 and 0.57 on the Train, Dev and Test sets respectively. 

Furthermore, the final number of risk predictive variants was 2798 which were associated with 847 Ensembl genes. Interaction analysis showed that Nodes were 339 and the edges were 622 in the gene interaction network. This shows evidence that the non-linear Machine learning approach offers great possibilities for understanding the genetic basis of complex diseases.
![image](https://user-images.githubusercontent.com/24875399/220415648-30ccfbf4-9dcb-4174-a0d3-f4d77b86d579.png)



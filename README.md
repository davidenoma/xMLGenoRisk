# Extreme gradient boosting and SVM machine learning algorithm identifies genome-wide interacting genetic variants in prostate cancer

Genome-wide association studies (GWAS) identify the variants (Single Nucleotide polymorphisms) associated with a disease phenotype within populations. These genetic differences are important in variations in incidence and mortalities, especially for Prostate cancer in the African population. Given the complexity of cancer, it is imperative to identify the variants that contribute to the development of the disease. The standard univariate analysis employed in GWAS may not capture the non-linear additive interactions between variants, which might affect the risk of developing Prostate cancer. This is because the interactions in complex diseases such as prostate cancer are usually non-linear and would benefit from a non-linear Machine Learning gradient boosting viz XGBoost (extreme gradient boosting). 
We applied the XGBoost algorithm and an iterative SNP selection algorithm to find the top features (SNPs) that best predict the risk of developing prostate cancer with a Support Vector Machine (SVM). The number of subjects was 907, and input features were 1,798,727 after appropriate quality control. The algorithm involved ten trials of 5-fold cross-validation for optimization of the hyperparameters on the dataset and the second module (utilizing SVM) of the prediction task. 
The model achieved AUC-ROC cure of 0.66, 0.57 and 0.55 on the Train, Dev and Test sets, respectively. The area under the Precision-Recall Curve was 0.69, 0.60 and 0.57 on the Train, Dev and Test sets, respectively. Furthermore, the final number of predictive risk variants was 2798, associated with 847 Ensembl genes. Interaction analysis showed that Nodes were 339 and the edges were 622 in the gene interaction network. 
![image](https://github.com/davidenoma/prostate_cancer_genetic_association_risk_pred/assets/24875399/b5359f61-2245-4234-9ce5-5974c48d740d)



<BR>
<img width="600" alt="image" src="https://github.com/davidenoma/prostate_cancer_genetic_association_risk_pred/assets/24875399/e1af6399-5aa3-4775-8c7b-dd854a01ff8b">

A. Using a 4:1 proportion, divide the genotyped data into training fold and test data. A 5-fold stratified CV is used to partition further the training fold data: one-fold (validation data) is used to assess the set of identified SNPs generated by module 2, and the other four folds are combined to form a training set data for the XGBoost model, which is used to find initial candidate Prostate Cancer risk-predictive SNPs (module 1). <br>
B. Using training fold data to optimize the XGBoost hyperparameters. <br>
C.	Module 1: Creating an initial list of possible Prostate Cancer risk-predictive SNPs using training set data to construct an XGBoost model <br>
D. Module 2: Using the initial list of candidate SNPs obtained from C and the validation data, an adaptive iterative SNP selection method. The top interacting SNPs producing the best Prostate Cancer risk prediction accuracy on the validation data are chosen after SNPs are re-ranked (see Algorithm 1). <br>
E. Using an SVM classifier, the top discovered interacting SNPs from (D) are adopted to forecast the Prostate Cancer risk on the test data. <br>
F. Performance values are averaged across all trials to get the final accuracy in the test set. <br>

<br>
The number of subjects was 907 and input features were 1,798,727 after appropriate quality control. 
The algorithm involved 10 trials of 5-fold cross-validation for optimization of the dataset's hyperparameters and the prediction task's second module (utilizing SVM). The model achieved AUC-ROC cure of 0.66, 0.57 and 0.55 on the Train, Dev and Test sets respectively.

![image](https://user-images.githubusercontent.com/24875399/220415941-b333861f-bc6b-4d79-90a5-a5972e5ba8ca.png)
![image](https://user-images.githubusercontent.com/24875399/220415972-b626ac8c-e8da-451f-8889-7d61658e79cf.png)

The area under the Precision-Recall Curve was 0.69, 0.60 and 0.57 on the Train, Dev and Test sets respectively. 

Furthermore, the final number of risk predictive variants was 2798 which were associated with 847 Ensembl genes. Interaction analysis showed that Nodes were 339 and the edges were 622 in the gene interaction network. This shows evidence that the non-linear Machine learning approach offers great possibilities for understanding the genetic basis of complex diseases.
![image](https://user-images.githubusercontent.com/24875399/220415648-30ccfbf4-9dcb-4174-a0d3-f4d77b86d579.png)



genotype_generated = pd.read_csv('../Xsubset.csv', header=None)
phenotype_generated = pd.read_csv('../hapmap_phenotype_recoded',header=None)
genotype_generated = genotype_generated.to_numpy()
phenotype_generated = phenotype_generated.to_numpy()
phenotype_generated = phenotype_generated.reshape(len(phenotype_generated),)

# Create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train ,y_test = train_test_split(genotype_generated,phenotype_generated,test_size=0.2)
### Step 1 ###



#SET PATH TO PLINK
export PATH=$PATH:$PATH:/{PATH_TO_PLINK}

#SET ALIAS
alias plink=""


#ENSURE GENOTYPE FILES' LOCATION
cd HOME/{user}/{path/folder containing your files}

# Investigate missingness per individual and per SNP and make histograms.
plink --bfile HapMap_3_r3_1 --missing

Rscript --no-save hist_miss.R
#Script for QC 
#EUR is the target genotype prefix
plink \
    --bfile EUR \
    --maf 0.01 \
    --hwe 1e-6 \
    --geno 0.01 \
    --mind 0.01 \
    --write-snplist \
    --make-just-fam \
    --out EUR.QC




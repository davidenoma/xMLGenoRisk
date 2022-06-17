#POST QC

#plink recode
plink --file Ghana_Prostate_11 --recodeA --out Ghana_Prostate
#create phenotype file
awk '{print $6}' Ghana_Prostate_11.fam > Ghana_Prostate_recode.pheno
#create the genotype inpute
awk '{for(i=7;i<=50;i++) printf $i" "; print ""}' ~/Association_GWAS/Ghana_Prostate_recode.raw > genotype_file_recode_final.geno
#sps list
python3 prepare_header_snps.py genotype_file_recode_final.geno ~/Association_GWAS/Ghana_Prostate_recode.pheno

#
bash /xampp/


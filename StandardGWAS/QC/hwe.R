hwe<-read.table (file="plink.hwe", header=TRUE)
pdf("histhwe.pdf")
hist(hwe[,9],load_data="Histogram HWE")
dev.off()

hwe_zoom<-read.table (file="plinkzoomhwe.hwe", header=TRUE)
pdf("histhwe_below_theshold.pdf")
hist(hwe_zoom[,9],load_data="Histogram HWE: strongly deviating SNPs only")
dev.off()

maf_freq <- read.table("MAF_check.frq", header =TRUE, as.is=T)
pdf("MAF_distribution.pdf")
hist(maf_freq[,5],load_data = "MAF distribution", xlab = "MAF")
dev.off()



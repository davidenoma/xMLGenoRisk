gender <- read.table("plink.sexcheck", header=T,as.is=T)

pdf("Gender_check.pdf")
hist(gender[,6],load_data="Gender", xlab="F")
dev.off()

pdf("Men_check.pdf")
male=subset(gender, gender$PEDSEX==1)
hist(male[,6],load_data="Men",xlab="F")
dev.off()

pdf("Women_check.pdf")
female=subset(gender, gender$PEDSEX==2)
hist(female[,6],load_data="Women",xlab="F")
dev.off()


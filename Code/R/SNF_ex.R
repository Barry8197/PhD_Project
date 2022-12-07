library(SNFtool)
library(dplyr)


require(data.table)
df <- fread("~/PPMI_Data/DNAm/beta_post_Funnorm_PPMI_EPICn524final030618.csv" ,   rownames=1)
df <- df %>% as.data.frame()
row.names(df) <- df$V1
df <- subset(df , select = -c(V1))
DNAm <- df

meta <- read.csv('~/PPMI_Data/DNAm/dnam_meta.csv')
meta <- meta[match(unique(meta$PATNO) , meta$PATNO),]
meta$Disease_Status <- factor(gsub(" ", "", meta$Disease_Status), levels = c("HealthyControl", "IdiopathicPD", "GeneticPD", "GeneticUnaffected"))
meta$PD_status <- factor(meta$Disease_Status, levels = c("Unaffected", "Affected"))
meta$PD_status[meta$Disease_Status %in% c("HealthyControl", "GeneticUnaffected")] <- "Unaffected"
meta$PD_status[meta$Disease_Status %in% c("IdiopathicPD", "GeneticPD")] <- "Affected"
meta <- meta[complete.cases(meta$PD_status),]
affected_sample <- sample(rownames(meta[meta$PD_status == 'Affected',]) , sum(meta$PD_status == 'Unaffected'))
unaffected <- rownames(meta[meta$PD_status == 'Unaffected',])
meta <- meta[append(affected_sample , unaffected) , ]
write.csv(meta , 'dnam_meta_final.csv')

#count_mtx <- cbind(count_mtx_tmp[sample(colnames(count_mtx_tmp[,2:length(count_mtx_tmp)]) , 190)] , count_mtx) 

DNAm <- DNAm[,unique(meta$Sentrix)]

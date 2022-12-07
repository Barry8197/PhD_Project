library(igraph)
library(DESeq2)
library(dplyr)

TS <- 'V04'
DS <- 'gPD'
load('~/mres_project_bryan/PPMI_IP/gPD_HC/preprocessedData/V04/preprocessed2.RData')
genes_of_interest <- colnames(read.csv('~/PhD_Project/Data/Expr_Data/deg_expr.csv'))[-c(1,2,210,211)]

vsd <- vst(dds)
res <- results(dds)
mat <- datExpr
mat <- mat[head(order(res$padj), 30), ]
mat <- mat - rowMeans(mat)
corr_mat <- as.matrix(as.dist(cor(mat, method="pearson")))
heatmap(corr_mat)
mat <- cbind(t(mat) , datMeta[,'PD_status'])
write.csv(mat , './PhD_Project/Data/Expr_Data/matExpr_padj.csv')


datExpr.pca <- prcomp(t(datExpr))
mat <- t(datExpr.pca$x[,1:150])
corr_mat <- as.matrix(as.dist(cor(mat, method="pearson")))
heatmap(corr_mat)
mat <- cbind(t(mat) , datMeta[,'PD_status'])
write.csv(mat , './PhD_Project/Data/Expr_Data/matExpr_pca.csv')

g <- graph.adjacency(
  corr_mat,
  mode="undirected",
  weighted=TRUE,
  diag=FALSE
)


g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)

# Colour negative correlation edges as blue
E(g)[which(E(g)$weight<0)]$color <- "darkblue"

# Colour positive correlation edges as red
E(g)[which(E(g)$weight>0)]$color <- "darkred"

# Convert edge weights to absolute values
E(g)$weight <- abs(E(g)$weight)

# Change arrow size
# For directed graphs only
#E(g)$arrow.size <- 1.0

# Remove edges below absolute Pearson correlation 0.8
g <- delete_edges(g, E(g)[which(E(g)$weight<0.3)])

# Remove any vertices remaining that have no edges
g <- delete.vertices(g, degree(g)==0)

# Assign names to the graph vertices (optional)
V(g)$class <- datMeta$PD_status

# Change shape of graph vertices
V(g)$shape <- "sphere"

# Change colour of graph vertices
V(g)$color <- V(g)$class

# Change colour of vertex frames
V(g)$vertex.frame.color <- "white"

# Scale the size of the vertices to be proportional to the level of expression of each gene represented by each vertex
# Multiply scaled vales by a factor of 10
scale01 <- function(x){(x-min(x))/(max(x)-min(x))}
vSizes <- (scale01(apply(mat, 1, mean)) + 1.0) * 10

# Amplify or decrease the width of the edges
edgeweights <- E(g)$weight * 2.0

# Convert the graph adjacency object into a minimum spanning tree based on Prim's algorithm
mst <- mst(g, algorithm="prim")

# Plot the tree object
plot(
  g,
  layout=layout.fruchterman.reingold,
  edge.curved=TRUE,
  vertex.size=vSizes,
  vertex.label.dist=-0.5,
  vertex.label.color="black",
  asp=FALSE,
  vertex.label.cex=0.6,
  edge.width=edgeweights,
  edge.arrow.mode=0,
  main="gPD vs. HC Correlation Network")


#write_graph(mst , './PhD_Project/Data/Graphs/first_graph.pajek' , 'pajek')

write.csv(as_long_data_frame(g) , file = './PhD_Project/Data/Graphs/first_graph.csv')


var_genes <- order(apply(datExpr, 1, var) , decreasing = TRUE)[1:200]
corr_mat <- as.matrix(as.dist(cor(datExpr[var_genes,], method="pearson")))
heatmap(corr_mat)


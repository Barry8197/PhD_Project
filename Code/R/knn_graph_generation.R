library(igraph)
library(DESeq2)
library(dplyr)

load('~/mres_project_bryan/PPMI_IP/gPD_HC/preprocessedData/V04/preprocessed2.RData')

make.knn.graph<-function(D,k){
  # calculate euclidean distances between cells
  dist<-as.matrix(dist(D))
  # make a list of edges to k nearest neighbors for each cell
  edges <- mat.or.vec(0,2)
  for (i in 1:nrow(dist)){
    # find closes neighbours
    matches <- setdiff(order(dist[i,],decreasing = F)[1:(k+1)],i)
    # add edges in both directions
    edges <- rbind(edges,cbind(rep(i,k),matches))  
    edges <- rbind(edges,cbind(matches,rep(i,k)))  
  }
  # create a graph from the edgelist
  graph <- graph_from_edgelist(edges,directed=F)
  V(graph)$frame.color <- NA
  # make a layout for visualizing in 2D
  set.seed(1)
  g.layout<-layout_with_fr(graph)
  return(list(graph=graph,layout=g.layout))        
}

vsd <- vst(dds)
res <- results(dds)
mat <- datExpr
mat <- mat[head(order(res$padj), 30), ]
mat <- mat - rowMeans(mat)
corr_mat <- as.matrix(as.dist(cor(mat, method="pearson")))
heatmap(corr_mat)

g <- make.knn.graph(corr_mat , 15)

plot.igraph(g$graph,layout=g$layout,vertex.color=datMeta[,'PD_status'],
            vertex.size=5,vertex.label=NA,main="padj KNN network")

g <- g$graph

g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)


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
vSizes <- (scale01(apply(corr_mat, 1, mean)) + 1.0) * 10

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
  main="gPD vs. HC Correlation Network")

write.csv(as_long_data_frame(g) , file = './PhD_Project/Data/Graphs/first_graph.csv')

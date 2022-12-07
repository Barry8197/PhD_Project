library(SNFtool)
library(igraph)

gExpr_graph <- read.csv('~/PhD_Project/Data/Graphs/gExpr_graph.csv')
DNAm_graph <- read.csv('~/PhD_Project/Data/Graphs/dnam_graph.csv')

g_gExpr <- graph_from_data_frame(gExpr_graph , directed = FALSE)
g_gExpr <- set_edge_attr(g_gExpr, "weight", value= gExpr_graph$weight)

g_DNAm <- graph_from_data_frame(DNAm_graph , directed = FALSE)
g_DNAm <- set_edge_attr(g_DNAm, "weight", value= DNAm_graph$weight)

adj_gExpr <- as_adjacency_matrix(g_gExpr , attr ="weight")
adj_DNAm <- as_adjacency_matrix(g_DNAm , attr="weight")



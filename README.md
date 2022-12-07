# PhD Project

## Python
### Custom_Dataset_Torch.ipynb
Implementation of creating a dataset compatible with pytorch. It takes a pandas dataframe as input.
It is compatible with dataframes created from iGraph using as_long_dataframe(). 

This dataset contains node features of
1. Node degree
2. Node centrality closeness

Edge Attributes :
1. Type of edge
	- Affected connected to Affected
	- Unaffected connected to Unaffected
	- Affected connected to Unaffected
2. Whether the similarity between patients is positive or negative in the network

Edge Weight : 
The strength of similarity between patients

There is an option to include a mask for splitting data into train, test and validation

This dataset is set up to perform node classification

### GNN_Node_Class.ipynb
This file performs training and testing on the created dataset. 
It implements the **GraphSAGE** ([Hamilton et al. (2017)](https://arxiv.org/abs/1706.02216)) layer and **GAT** ([Velkovic et al. (2017)](https://arxiv.org/abs/1710.10903))

### LogReg.ipynb
It is a Logistic Regression implementation used as a baseline comparison to compare the GNN results to.
Implements a L2 logistic regression with log loss used to select the value of C

## R
### geneExpr_graph_gen.R
Takes any expression matrix and creates a similarity network. The output is in the form of a dataframe.
Currently this file outputs a $\epsilon$-nearest neighbors similarity where nodes with an absolute similarity less
than 0.3 are not connected and everything else is connected. 

There is an option to perform minimum spanning tree however this has not worked well on the data so far. 

There are two main measures of similarity. The first performs PCA and takes the required number of PC's which contains
~90% of the variance in the expression dataset. This retains the variance of all genes.

The second uses the most significantly differentially expressed genes between patients affected by PD and those unaffected. 
This is more accurate but possibly only possible for gene expression datasets whereas PCA could theoretically be used for other 
biological datasets. 

### knn_graph_generation
Implements k-nearest neighbour graph generation based on the dist() R function

 

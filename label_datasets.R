# load sc-type
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

#load sc sorter
require(scSorter)
require(SCINA)
require(Seurat)
require(dplyr)

# Reads in counts matrix and marker gene lists and outputs dataframe with cells X labels from differnet tools
indices <- c("0.1","0.2","0.3","0.4")
for(i in 1:4){
  data_path <- paste("~/desktop/conradLab/thesis/scMARM/simulations/splat_",indices[i],"_de/",sep="")
  counts <- read.csv(paste(data_path,"counts.csv", sep=""), header=T, row.names = 1)
  markers_1 <- row.names(head(read.csv(paste(data_path, "group1_de.csv", sep=""), header=T, row.names = 1), 8))
  markers_2 <- row.names(head(read.csv(paste(data_path, "group2_de.csv", sep=""), header=T, row.names = 1), 8))
  markers_3 <- row.names(head(read.csv(paste(data_path, "group3_de.csv", sep=""), header=T, row.names = 1), 8))
  markers_4 <- row.names(head(read.csv(paste(data_path, "group4_de.csv", sep=""), header=T, row.names = 1), 8))
  metadata <- read.csv(paste(data_path, "meta.csv", sep=""), header=T, row.names=1)
  
  # preprocess
  data <- CreateSeuratObject(t(counts), meta.data = metadata)
  data <- NormalizeData(data)
  data <- ScaleData(data)
  data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 2000)
  data <- RunPCA(data)
  data <- FindNeighbors(data)
  data <- FindClusters(data)
  
  # run sc Type
  markers = list(Group1 = markers_1, Group2 = markers_2, Group3 = markers_3, Group4 = markers_4)
  es.max = sctype_score(scRNAseqData = as.matrix(data@assays$RNA@data), scaled = F, 
                        gs = markers, gs2 = NULL, gene_names_to_uppercase = F) 
  cL_resutls = do.call("rbind", lapply(unique(data@meta.data$seurat_clusters), function(cl){
    es.max.cl = sort(rowSums(es.max[ ,rownames(data@meta.data[data@meta.data$seurat_clusters==cl, ])]), decreasing = !0)
    head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(data@meta.data$seurat_clusters==cl)), 10)
  }))
  sctype_scores = cL_resutls %>% group_by(cluster) %>% top_n(n = 1, wt = scores)
  sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "NA"
  
  for(j in unique(sctype_scores$cluster)){
    cl_type = sctype_scores[sctype_scores$cluster==j,]; 
    data@meta.data$customclassif[data@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
  }
  
  sctype_preds = data$customclassif
  
  # run sc sort
  anno <- data.frame(Type=c(rep("Group1",8), rep("Group2",8), rep("Group3", 8), rep("Group4",8)), Marker=c(markers_1, markers_2, markers_3, markers_4))
  topgenes <- head(VariableFeatures(data), 2000)
  picked_genes <- unique(c(topgenes, anno$Marker))
  expr <- as.matrix(data@assays$RNA@data)
  expr <- expr[rownames(expr) %in% picked_genes,]
  
  rts <- scSorter(expr, anno)
  
  scsort_preds = rts$Pred_Type
  
  # run scina
  # markers same as SCTYPE
  expr = as.matrix(data@assays$RNA@data)
  results <- SCINA(expr, markers)
  scina_preds <- results$cell_labels
  
  # output predictions
  preds <- data.frame(scType=sctype_preds, scSorter=scsort_preds, SCINA=scina_preds)
  row.names(preds) <- colnames(data@assays$RNA@data)
  
  write.csv(preds, paste(data_path,"predictions.csv", sep=""))
}


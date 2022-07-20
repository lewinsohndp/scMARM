library(splatter)
library(Seurat)
library(scater)
library(ggplot2)

folder <- "~/desktop/conradLab/thesis/scMARM/simulations/"
for(i in seq(.1,.4,length.out=4)){
  full_path <- paste(folder,"splat_",i,"_de/", sep="")
  dir.create(full_path)
  
  sim.groups <- splatSimulate(batchCells=1000,group.prob = c(.423, .227, .226, .124), method = "groups", nGenes = 20000, de.facScale=i, verbose = FALSE, seed=8)
  
  sim <- CreateSeuratObject(assays(sim.groups)$counts, meta.data=as.data.frame(sim.groups@colData))
  
  features = VlnPlot(sim, features = c("nFeature_RNA", "nCount_RNA"), ncol = 2)
  ggsave(paste(full_path,"feature_plots.pdf",sep=""), plot=features, device="pdf")
  
  sim <- NormalizeData(sim)
  
  sim <- FindVariableFeatures(sim, selection.method = "vst", nfeatures = 2000)
  # Identify the 10 most highly variable genes
  top10 <- head(VariableFeatures(sim), 10)
  
  # plot variable features with and without labels
  plot1 <- VariableFeaturePlot(sim)
  plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
  variable = plot1 + plot2
  ggsave(paste(full_path,"variable.pdf",sep=""), plot = variable, device="pdf")
  
  sim <- ScaleData(sim)
  
  sim <- RunPCA(sim)
  
  pca <- DimPlot(sim, reduction="pca", group.by = "Group")
  ggsave(paste(full_path,"pca.pdf",sep=""), plot=pca, device="pdf")
  
  sim <- FindNeighbors(sim)
  sim <- FindClusters(sim)
  
  sim <- RunUMAP(sim, dims = 1:50, mid.dist=.1)
  
  umap <- DimPlot(sim, reduction="umap")
  ggsave(paste(full_path,"umap.pdf",sep=""), plot=umap, device="pdf")
  umap <- DimPlot(sim, reduction="umap", group.by = "Group")
  ggsave(paste(full_path,"umap_groups.pdf",sep=""), plot=umap, device="pdf")
  
  Idents(object = sim) <- "Group"
  group1.markers <- FindMarkers(sim, ident.1 = "Group1", ident.2 = NULL, only.pos = TRUE)[1:20,]
  group2.markers <- FindMarkers(sim, ident.1 = "Group2", ident.2 = NULL, only.pos = TRUE)[1:20,]
  group3.markers <- FindMarkers(sim, ident.1 = "Group3", ident.2 = NULL, only.pos = TRUE)[1:20,]
  group4.markers <- FindMarkers(sim, ident.1 = "Group4", ident.2 = NULL, only.pos = TRUE)[1:20,]
  write.csv(group1.markers, paste(full_path,"group1_de.csv",sep=""))
  write.csv(group2.markers, paste(full_path,"group2_de.csv",sep=""))
  write.csv(group3.markers, paste(full_path,"group3_de.csv",sep=""))
  write.csv(group4.markers, paste(full_path,"group4_de.csv",sep=""))
  
  write.csv(t(sim@assays$RNA@counts), paste(full_path,"counts.csv",sep=""))
  write.csv(sim@meta.data, paste(full_path,"meta.csv",sep=""))
}

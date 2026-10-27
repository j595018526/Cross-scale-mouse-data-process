# Cross-scale-mouse-data-process
Bridging macroscopic imaging, mesoscopic pathology, and microscopic molecular profiles: a cross-scale spatial omics analysis framework.

All file descriptions are as follows. 
1. stdata_preprocess.py: Read the raw ST data, perform standardization first, then extract the target gene expression levels and save them as a CSV table.
2. mouse_crossscale_process/liver_niche: Based on the cross-scale biological mapping, tumor-driven alterations in cross-scale metrics were first investigated, with ROIs defined as the niche in the liver tumor.
3. mouse_crossscale_process/brain_hip: Based on the cross-scale biological mapping, tumor-driven alterations in cross-scale metrics were then investigated, with ROIs defined as the hippocampus in the mouse brain.
4. mouse_crossscale_process/cross_scale_mapping_liver: A systematic cross-scale spatial multi-omics analysis was conducted across the tumor-edge, tumor-interior, and normal zones of the livers of two tumor-bearing mice.
5. mouse_crossscale_process/cross_scale_mapping_brain: To validate the generalizability of the cross-scale spatial omics analysis framework, it was applied to coronal brain sections from three mice, including one blank mouse and two liver tumor-bearing mice.

#create_segmetaion.py 
        INFO:__main__:Starting segmentation creation...
        INFO:__main__:Loading data from D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\selected_channels.zarr
        INFO:__main__:Data shape: (1, 6, 194, 688, 1363)
        INFO:__main__:
        Processing CD31...
        INFO:__main__:CD31 - Min: 0.000, Max: 32737.000
        INFO:__main__:CD31 - Otsu threshold: 0.080, Manual threshold: 0.200
        INFO:__main__:Created CD31 segmentation with 99 cells
        INFO:__main__:Creating output directory at: D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD31.zarr
        [########################################] | 100% Completed | 3.63 sms
        INFO:__main__:Segmentation saved as Zarr in D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD31.zarr
        INFO:__main__:
        Processing CD11b...
        INFO:__main__:CD11b - Min: 0.000, Max: 32282.000
        INFO:__main__:CD11b - Otsu threshold: 0.330, Manual threshold: 0.100
        INFO:__main__:Created CD11b segmentation with 84 cells
        INFO:__main__:Creating output directory at: D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD11b.zarr
        [########################################] | 100% Completed | 4.62 sms
        INFO:__main__:Segmentation saved as Zarr in D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD11b.zarr
        INFO:__main__:
        Processing CD11c...
        INFO:__main__:CD11c - Min: 0.000, Max: 32874.000
        INFO:__main__:CD11c - Otsu threshold: 0.350, Manual threshold: 0.100
        INFO:__main__:Created CD11c segmentation with 29 cells
        INFO:__main__:Creating output directory at: D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD11c.zarr
        [########################################] | 100% Completed | 3.51 sms
        INFO:__main__:Segmentation saved as Zarr in D:\VIS2025\BIoVisChallenges\SpaFGAN\backend\input\segmentation_CD11c.zarr
        INFO:__main__:Segmentation creation completed successfully
2- extract_features
        Cell_feature_CD11b.csv

                CD31 Channel:          CD11b Channel:         CD11c Channel:
        +----------------+    +----------------+    +----------------+
        |                |    |                |    |                |
        |    [Cell 1]    |    |    [Cell 1]    |    |    [Cell 1]    |
        |                |    |                |    |                |
        +----------------+    +----------------+    +----------------+
     This gives us a complete picture of:
        Where each cell type is located
        What other markers they express
        How different cell types interact
        For Cell 1 in CD31 segmentation:
    - CD31 intensity: 150.25  (High because it's a CD31 cell)
    - CD11b intensity: 30.12  (Lower, but still measured)
    - CD11c intensity: 25.89  (Lower, but still measured)
3- train_spafgan model 


        2025-06-03 16:41:47,673 - INFO - ================================================================================
        2025-06-03 16:41:47,673 - INFO - Marker     Cells      Train Acc    Val Acc      Train Loss   Val Loss
        2025-06-03 16:41:47,673 - INFO - --------------------------------------------------------------------------------
        2025-06-03 16:41:47,673 - INFO - CD31       99         0.5570      0.7500      1.2635      0.8555
        2025-06-03 16:41:47,673 - INFO - CD11b      84         0.6119      0.6471      1.2978      0.9305
        2025-06-03 16:41:47,673 - INFO - CD11c      29         0.8696      1.0000      0.8123      0.4962
        2025-06-03 16:41:47,674 - INFO - ================================================================================

4- segment.py for finding ROI with using model and selected cells
        ==================================================
        FINAL RESULTS SUMMARY
        ==================================================
        CD31 ===> 24 ROI cells
        CD11b ===> 18 ROI cells
        CD11c ===> 15 ROI cells
        ==================================================

5- build_graph.py creating graph with celss and interaction aweight with distance
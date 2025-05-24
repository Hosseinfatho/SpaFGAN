import json

# Vitessce configuration
vitessce_config = {
    "version": "1.0.0",
    "name": "SpaFGAN ROIs Visualization",
    "description": "Visualization of immune ROIs using Vitessce",
    "datasets": [
        {
            "uid": "dataset-1",
            "name": "SpaFGAN Dataset",
            "files": [
                {
                    "type": "raster",
                    "fileType": "ome-zarr",
                    "url": "backend/input/selected_channels.zarr",
                    "options": {
                        "schemaVersion": "0.4",
                        "images": [
                            {
                                "name": "CD31",
                                "channels": [0]
                            },
                            {
                                "name": "CD20",
                                "channels": [1]
                            },
                            {
                                "name": "CD11b",
                                "channels": [2]
                            },
                            {
                                "name": "CD4",
                                "channels": [3]
                            },
                            {
                                "name": "CD11c",
                                "channels": [4]
                            },
                            {
                                "name": "Catalase",
                                "channels": [5]
                            }
                        ]
                    }
                },
                    {
                    "type": "cells",
                    "fileType": "anndata-cells.zarr",
                    "url": "backend/output/cells.zarr",
                    "options": {
                        "mappings": {
                        "UMAP": ["X_umap_0", "X_umap_1"],
                        "tSNE": ["X_tsne_0", "X_tsne_1"]
                        },
                        "cellSetObservationField": "interactions"
                    }
                    }

            ]
        }
    ],
    "initStrategy": "auto",
    "coordinationSpace": {
        "spatialZoom": {
            "A": -5
        },
        "spatialTargetX": {
            "A": 0
        },
        "spatialTargetY": {
            "A": 0
        }
    },
    "layout": [
        {
            "component": "spatial",
            "x": 0,
            "y": 0,
            "w": 8,
            "h": 8,
            "coordinationScopes": {
                "spatialZoom": "A",
                "spatialTargetX": "A",
                "spatialTargetY": "A"
            }
        },
        {
            "component": "cellSets",
            "x": 8,
            "y": 0,
            "w": 4,
            "h": 4
        },
        {
            "component": "cellTypeExpression",
            "x": 8,
            "y": 4,
            "w": 4,
            "h": 4
        }
    ]
}

# Save Vitessce configuration
with open("backend/output/vitessce_config.json", "w") as f:
    json.dump(vitessce_config, f, indent=2)

print("✅ Saved Vitessce configuration to vitessce_config.json") 
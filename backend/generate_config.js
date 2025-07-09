function generateBioMedVisConfig(showROI = false) {
  // Configuration constants
  const CONFIG = {
    schemaVersion: '1.0.16',
    name: 'BioMedVis Challenge',
    description: 'ROI annotations for the BioMedVis Challenge',
    datasetUid: 'bv',
    datasetName: 'Blood Vessel',
    imageUrl: 'https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0',
    roiUrl: 'http://localhost:5000/api/roi_Segmentation.json',
    showROI: showROI,  // Control ROI visibility - passed as parameter
    channels: [
      { id: 0, name: "CD31", color: [0, 255, 0], window: [300, 20000], targetC: 19 },      // Green
      { id: 1, name: "CD20", color: [255, 255, 0], window: [1000, 7000], targetC: 27 },    // Yellow
      { id: 2, name: "CD11b", color: [255, 0, 255], window: [700, 6000], targetC: 37 },    // Magenta
      { id: 3, name: "CD4", color: [0, 255, 255], window: [1638, 10000], targetC: 25 },    // Cyan
      { id: 5, name: "CD11c", color: [128, 0, 128], window: [370, 1432], targetC: 42 } // Purple
    ],
    spatial: {
      renderingMode: "3D",
      targetX: 5454,
      targetY: 2754,
      targetZ: 100,
      targetT: 0,
      targetResolution: 3,
      zoom: -3.0
    }
  };

  // Helper functions
  function getChannelScopeName(channelId) {
    return `init_${CONFIG.datasetUid}_image_${channelId}`;
  }

  function generateDatasets() {
    // Create a mock VitessceConfig-like structure
    const config = {
      addDataset: function(name) {
        this.datasetName = name;
        this.files = [];
        return this;
      },
      addFile: function(fileConfig) {
        this.files.push(fileConfig);
        return this;
      }
    };

    const dataset = config.addDataset('My dataset')
      .addFile({
        fileType: 'image.ome-zarr',
        url: 'https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0',
      });
    
    // Add ROI file only if showROI is true
    if (CONFIG.showROI) {
      dataset.addFile({
        fileType: 'obsSegmentations.json',
        url: 'http://localhost:5000/api/roi_Segmentation.json',
        coordinationValues: {
          obsType: 'ROI',
        },
      });
    }

    // Convert to the expected format
    return [{
      uid: CONFIG.datasetUid,
      name: dataset.datasetName,
      files: dataset.files
    }];
  }

  function generateCoordinationSpace() {
    const coordinationSpace = {
      dataset: { "A": CONFIG.datasetUid },
      imageChannel: {},
      imageLayer: { "init_bv_image_0": "__dummy__" },
      metaCoordinationScopes: {
        "init_bv_image_0": {
          imageLayer: ["init_bv_image_0"],
          spatialRenderingMode: "init_bv_image_0",
          spatialTargetT: "init_bv_image_0",
          spatialTargetX: "init_bv_image_0",
          spatialTargetY: "init_bv_image_0",
          spatialTargetZ: "init_bv_image_0",
          spatialZoom: "init_bv_image_0"
        }
      },
      metaCoordinationScopesBy: {
        "init_bv_image_0": {
          imageChannel: {
            spatialChannelColor: {},
            spatialChannelOpacity: {},
            spatialChannelVisible: {},
            spatialChannelWindow: {},
            spatialTargetC: {}
          },
          imageLayer: {
            imageChannel: { "init_bv_image_0": [] },
            photometricInterpretation: { "init_bv_image_0": "init_bv_image_0" },
            spatialLayerOpacity: { "init_bv_image_0": "init_bv_image_0" },
            spatialLayerVisible: { "init_bv_image_0": "init_bv_image_0" },
            spatialTargetResolution: { "init_bv_image_0": "init_bv_image_0" }
          }
        }
      },
      photometricInterpretation: { "init_bv_image_0": "BlackIsZero" },
      spatialChannelColor: {},
      spatialChannelOpacity: {},
      spatialChannelVisible: {},
      spatialChannelWindow: {},
      spatialLayerOpacity: { "init_bv_image_0": 1.0 },  // Full opacity for image
      spatialLayerVisible: { "init_bv_image_0": true },  // Keep image layer visible
      spatialRenderingMode: { "init_bv_image_0": CONFIG.spatial.renderingMode },
      spatialTargetC: {},
      spatialTargetResolution: { "init_bv_image_0": CONFIG.spatial.targetResolution },
      spatialTargetT: { "init_bv_image_0": CONFIG.spatial.targetT },
      spatialTargetX: { "init_bv_image_0": CONFIG.spatial.targetX },
      spatialTargetY: { "init_bv_image_0": CONFIG.spatial.targetY },
      spatialTargetZ: { "init_bv_image_0": CONFIG.spatial.targetZ },
      spatialZoom: { "init_bv_image_0": CONFIG.spatial.zoom }
    };

    // Generate channel-specific coordination values
    CONFIG.channels.forEach(channel => {
      const scopeName = getChannelScopeName(channel.id);
      
      // Add to imageChannel
      coordinationSpace.imageChannel[scopeName] = "__dummy__";
      
      // Add channel colors, opacity, visibility, windows, and target channels
      coordinationSpace.spatialChannelColor[scopeName] = channel.color;
      coordinationSpace.spatialChannelOpacity[scopeName] = 1.0;
      coordinationSpace.spatialChannelVisible[scopeName] = true;
      coordinationSpace.spatialChannelWindow[scopeName] = channel.window;
      coordinationSpace.spatialTargetC[scopeName] = channel.targetC;
      
      // Add to metaCoordinationScopesBy
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageChannel.spatialChannelColor[scopeName] = scopeName;
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageChannel.spatialChannelOpacity[scopeName] = scopeName;
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageChannel.spatialChannelVisible[scopeName] = scopeName;
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageChannel.spatialChannelWindow[scopeName] = scopeName;
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageChannel.spatialTargetC[scopeName] = scopeName;
      
      // Add to imageChannel array in imageLayer
      coordinationSpace.metaCoordinationScopesBy.init_bv_image_0.imageLayer.imageChannel.init_bv_image_0.push(scopeName);
    });

    // Add ROI coordination space only if showROI is true
    if (CONFIG.showROI) {
      coordinationSpace.metaCoordinationScopes["A"] = { 
        obsType: "A",
        obsLayerVisible: true  // Show ROI layer when enabled
      };
      coordinationSpace.metaCoordinationScopesBy["A"] = {};
      coordinationSpace.obsType = { "A": "ROI" };
    }

    return coordinationSpace;
  }

  function generateLayout() {
    const baseScopes = ["init_bv_image_0"];
    const baseScopesBy = ["init_bv_image_0"];
    
    // Add ROI scopes only if showROI is true
    if (CONFIG.showROI) {
      baseScopes.push("A");
      baseScopesBy.push("A");
    }
    
    return [
      {
        component: "spatialBeta",
        coordinationScopes: {
          dataset: "A",
          metaCoordinationScopes: baseScopes,
          metaCoordinationScopesBy: baseScopesBy
        },
        x: 0, y: 0, w: 6, h: 12
      },
      {
        component: "layerControllerBeta",
        coordinationScopes: {
          dataset: "A",
          metaCoordinationScopes: baseScopes,
          metaCoordinationScopesBy: baseScopesBy
        },
        x: 6, y: 0, w: 6, h: 12
      }
    ];
  }

  // Generate the complete config
  const config = {
    version: CONFIG.schemaVersion,
    name: CONFIG.name,
    description: CONFIG.description,
    datasets: generateDatasets(),
    coordinationSpace: generateCoordinationSpace(),
    layout: generateLayout(),
    initStrategy: "auto"
  };

  return config;
}

// If running directly, output the config
if (require.main === module) {
  const config = generateBioMedVisConfig();
  console.log(JSON.stringify(config, null, 2));
}

module.exports = { generateBioMedVisConfig }; 
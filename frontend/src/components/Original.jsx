import React, { useState, useEffect, useRef } from 'react';
import { Vitessce, CoordinationType } from 'vitessce';
import ROISelector from './ROISelector';
import InteractiveCircles from './InteractiveCircles';
import Plot from 'react-plotly.js';

// Constants for Image Channels
const IMAGE_CHANNELS = {
  'CD31': { 'id': 'cd31', 'color': [0, 255, 0], 'window': [300, 20000], 'targetC': 19 },
  'CD20': { 'id': 'cd20', 'color': [255, 255, 0], 'window': [1000, 7000], 'targetC': 27 },
  'CD11b': { 'id': 'cd11b', 'color': [255, 0, 255], 'window': [700, 6000], 'targetC': 37 },
  'CD4': { 'id': 'cd4', 'color': [0, 255, 255], 'window': [1638, 10000], 'targetC': 25 },
  'CD11c': { 'id': 'cd11c', 'color': [128, 0, 128], 'window': [370, 1432], 'targetC': 42 }
};

// Constants for Interaction Type to ROI Mapping
const INTERACTION_TO_ROI = {
  'B-cell infiltration': {
    'file': 'roi_segmentation_B-cell_infiltration.json',
    'obsType': 'ROI_B-cell',
    'color': [255, 180, 180],  // Light Red
    'strokeWidth': 6
  },
  'Inflammatory zone': {
    'file': 'roi_segmentation_Inflammatory_zone.json',
    'obsType': 'ROI_Inflammatory',
    'color': [180, 255, 180],  // Light Green
    'strokeWidth': 6
  },
  'T-cell entry site': {
    'file': 'roi_segmentation_T-cell_entry_site.json',
    'obsType': 'ROI_T-cell',
    'color': [180, 180, 255],  // Light Blue
    'strokeWidth': 6
  },
  'Oxidative stress niche': {
    'file': 'roi_segmentation_Oxidative_stress_niche.json',
    'obsType': 'ROI_Oxidative',
    'color': [255, 255, 180],  // Light Yellow
    'strokeWidth': 6
  }
};

// Simple config generation function
const generateVitessceConfig = (selectedGroups = []) => { 
  // Build coordination space
  const coordination_space = {
    'dataset': { "A": "bv" },
    'imageLayer': { "image": "image" },
    'imageChannel': {},
    'spatialChannelColor': {"A": [255, 100, 100]},
    'spatialChannelOpacity': {"image": 0.2 },
    'spatialChannelVisible': {},
    'spatialChannelWindow': {},
    'spatialTargetC': {},
    'spatialLayerOpacity': { "image": 0.2 },
    'spatialLayerVisible': { "image": true },
    'spatialRenderingMode': { "image": "3D" },
    'spatialTargetX': { "A": 5454 },
    'spatialTargetY': { "A": 2754 },
    'spatialTargetZ': { "A": 0 },
    'spatialZoom': { "A": -3.5 },
    'spatialTargetResolution': { "image": 3 },
    'spatialTargetT': { "image": 0 },
    'photometricInterpretation': { "image": "BlackIsZero" },
    'spatialSegmentationFilled': {},
    'spatialSegmentationStrokeWidth': {},
    [CoordinationType.TOOLTIPS_VISIBLE]: {},
    'metaCoordinationScopes': {
      "metaA": {
        "imageLayer": ["image"],
        "spatialChannelVisible": ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
        "spatialChannelOpacity": ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
        "spatialChannelColor": [],
        "spatialSegmentationFilled": [],
        "spatialSegmentationStrokeWidth": [],
        [CoordinationType.TOOLTIPS_VISIBLE]: []
      }
    },
    'metaCoordinationScopesBy': {
      "metaA": {
        "imageLayer": {
          "imageChannel": { "image": ["CD31", "CD20", "CD11b", "CD4", "CD11c"] },
          "spatialLayerVisible": { "image": "image" },
          "spatialLayerOpacity": { "image": "image" },
          "spatialRenderingMode": { "image": "3D" },
          "spatialTargetResolution": { "image": "image" },
          "spatialTargetT": { "image": "image" },
          "photometricInterpretation": { "image": "image" }
        },
        "imageChannel": {
          "spatialTargetC": {},
          "spatialChannelColor": {},
          "spatialChannelVisible": {},
          "spatialChannelOpacity": {},
          "spatialChannelWindow": {}
        }
      }
    }
  };

  // Add image channels
  Object.entries(IMAGE_CHANNELS).forEach(([chName, chProps]) => {
    coordination_space['imageChannel'][chName] = "__dummy__";
    coordination_space['spatialChannelColor'][chName] = chProps['color'];
    coordination_space['spatialChannelOpacity'][chName] = 0.5;
    coordination_space['spatialChannelVisible'][chName] = true;
    coordination_space['spatialChannelWindow'][chName] = chProps['window'];
    coordination_space['spatialTargetC'][chName] = chProps['targetC'];
    
    // Add to meta coordination scopes
    coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialTargetC'][chName] = chName;
    coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelColor'][chName] = chName;
    coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelVisible'][chName] = chName;
    coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelOpacity'][chName] = chName;
    coordination_space['metaCoordinationScopesBy']['metaA']['imageChannel']['spatialChannelWindow'][chName] = chName;
    
    // Add to meta coordination scopes arrays
    coordination_space['metaCoordinationScopes']['metaA']['spatialChannelColor'].push(chName);
  });

  // Build files array - start with image file
  const files = [
    {
      'fileType': 'image.ome-zarr',
      'url': 'https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0',
    }
  ];

  // Add ROI segmentation files for selected groups
  selectedGroups.forEach(group => {
    if (INTERACTION_TO_ROI[group]) {
      const roi_info = INTERACTION_TO_ROI[group];
      const obs_type = roi_info['obsType'];
      
      // Add coordination settings
      coordination_space['spatialSegmentationFilled'][obs_type] = false; // ROIs are hollow
      coordination_space['spatialSegmentationStrokeWidth'][obs_type] = roi_info['strokeWidth'];
      coordination_space[CoordinationType.TOOLTIPS_VISIBLE][obs_type] = true; // Enable tooltips for ROIs
      
      // Add ROI color to spatialChannelColor (required by Vitessce)
      coordination_space['spatialChannelColor'][obs_type] = roi_info['color'];
      
      // Add to meta coordination scopes
      coordination_space['metaCoordinationScopes']['metaA']['spatialSegmentationFilled'].push(obs_type);
      coordination_space['metaCoordinationScopes']['metaA']['spatialSegmentationStrokeWidth'].push(obs_type);
      coordination_space['metaCoordinationScopes']['metaA'][CoordinationType.TOOLTIPS_VISIBLE].push(obs_type);
      coordination_space['metaCoordinationScopes']['metaA']['spatialChannelColor'].push(obs_type);
      
      files.push({
        'fileType': 'obsSegmentations.json',
        'url': `http://localhost:5000/api/${roi_info["file"]}`,
        'coordinationValues': {
          'obsType': roi_info['obsType']
        }
      });
    }
  });

  const config = {
    'version': '1.0.16',
    'name': `BioMedVis Challenge - ${selectedGroups.length > 0 ? selectedGroups.join(", ") : "Image Only"}`,
    'description': `Dynamic config with selected interaction types: ${selectedGroups.length > 0 ? selectedGroups.join(", ") : "None"}`,
    'datasets': [{
      'uid': 'bv',
      'name': 'Blood Vessel',
      'files': files
    }],
    'initStrategy': 'auto',
    'coordinationSpace': coordination_space,
    'layout': [
      {
        'component': 'spatialBeta',
        'coordinationScopes': {
          'metaCoordinationScopes': ["metaA"],
          'metaCoordinationScopesBy': ["metaA"],
          'spatialTargetX': "A",
          'spatialTargetY': "A",
          'spatialTargetZ': "A",
          'spatialZoom': "A",
          'spatialTargetResolution': "image",
          'spatialTargetT': "image",
          'spatialRenderingMode': "image",
          'spatialChannelVisible': ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
          'spatialChannelOpacity': ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
          'spatialChannelColor': Object.keys(coordination_space['spatialChannelColor']),
          'spatialSegmentationFilled': Object.keys(coordination_space['spatialSegmentationFilled']),
          'spatialSegmentationStrokeWidth': Object.keys(coordination_space['spatialSegmentationStrokeWidth']),
          [CoordinationType.TOOLTIPS_VISIBLE]: Object.keys(coordination_space[CoordinationType.TOOLTIPS_VISIBLE])
        },
        'x': 0, 'y': 0, 'w': 8, 'h': 12
      },
      {
        'component': 'layerControllerBeta',
        'coordinationScopes': {
          'metaCoordinationScopes': ["metaA"],
          'metaCoordinationScopesBy': ["metaA"],
          'spatialTargetX': "A",
          'spatialTargetY': "A",
          'spatialTargetZ': "A",
          'spatialZoom': "A",
          'spatialTargetResolution': "image",
          'spatialTargetT': "image",
          'spatialRenderingMode': "image",
          'spatialChannelVisible': ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
          'spatialChannelOpacity': ["CD31", "CD20", "CD11b", "CD4", "CD11c"],
          'spatialChannelColor': Object.keys(coordination_space['spatialChannelColor']),
          'spatialSegmentationFilled': Object.keys(coordination_space['spatialSegmentationFilled']),
          'spatialSegmentationStrokeWidth': Object.keys(coordination_space['spatialSegmentationStrokeWidth']),
          [CoordinationType.TOOLTIPS_VISIBLE]: Object.keys(coordination_space[CoordinationType.TOOLTIPS_VISIBLE])
        },
        'x': 0, 'y': 8, 'w': 4, 'h': 12
      }
    ]
  };

  return config;
};

const MainView = ({ onSetView }) => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const [prevCellSetSelection, setPrevCellSetSelection] = useState(null);

  const [heatmapResults, setHeatmapResults] = useState({});
  const [interactionHeatmapResult, setInteractionHeatmapResult] = useState(null);
  const [activeGroups, setActiveGroups] = useState({
    1: true,
    2: true,
    3: true,
    4: true
  });
  const [configKey, setConfigKey] = useState(0);
  const [rois, setRois] = useState([]);
  const [showCircles, setShowCircles] = useState(false);
  const [selectedCircle, setSelectedCircle] = useState(null);
  const [selectedGroups, setSelectedGroups] = useState([]);
  const vitessceRef = useRef(null);

  const groupColors = {
    1: '#d7191c',
    2: '#fdae61',
    3: '#abd9e9',
    4: '#2c7bb6'
  };

  const groupNames = {
    1: 'Endothelial-immune interface (CD31 + CD11b)',
    2: 'ROS detox, immune stress (CD11b + Catalase)',
    3: 'T/B cell recruitment via vessels (CD31 + CD4/CD20)',
    4: 'T-B collaboration (CD4 + CD20)'
  };

  // Generate config locally
  const generateConfig = (groups = []) => {
    const newConfig = generateVitessceConfig(groups);
    setConfig(newConfig);
    setConfigKey(prev => prev + 1); // Force re-render
    
    // Store config globally for debugging
    window.lastConfig = newConfig;
    console.log('Generated config and stored in window.lastConfig:', newConfig);
    
    // Send config to backend
    fetch('http://localhost:5000/api/updateconfig', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(newConfig)
    })
    .then(response => response.json())
    .then(data => {
      console.log('Config sent to backend:', data);
    })
    .catch(error => {
      console.error('Error sending config to backend:', error);
    });
  };

  // Generate initial config on component mount
  useEffect(() => {
    generateConfig([]);
  }, []);

  // Regenerate config when selectedGroups changes
  useEffect(() => {
    console.log('selectedGroups changed, regenerating config:', selectedGroups);
    generateConfig(selectedGroups);
  }, [selectedGroups]);

  // Handle group selection updates from ROISelector


  useEffect(() => {
    if (config) {
      console.log("Vitessce config:", config);
    }
  }, [config]);

  useEffect(() => {
    fetch("http://localhost:5000/api/roi_shapes")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        if (data.features && Array.isArray(data.features)) {
          const extracted = data.features.map((feature, index) => {
            const geometry = feature.geometry;
            if (!geometry || !geometry.coordinates) {
              return null;
            }

            let allCoords = [];
            if (geometry.type === "Polygon") {
              allCoords = geometry.coordinates;
            } else if (geometry.type === "MultiPolygon") {
              allCoords = geometry.coordinates.flat();
            } else {
              return null;
            }

            const [cx, cy] = allCoords.flat().reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
            const count = allCoords.flat().length;
            const centroid = [cx / count, cy / count];

            return {
              id: feature.properties.name || `ROI_${index}`,
              x: centroid[0],
              y: centroid[1],
              score: feature.properties.score || 0,
              interactions: feature.properties.interactions || [],
              raw: feature.properties
            };
          }).filter(Boolean);
          setRois(extracted);
        }
      })
      .catch((err) => {
        console.error("Failed to load ROI shapes:", err);
        setRois([]);
      });
  }, []);

  const handleSetView = (roiView) => {
    console.log('Mainview handleSetView:', roiView);
    
    if (roiView.hasOwnProperty('showCircles')) {
      setShowCircles(roiView.showCircles);
      console.log('Mainview: showCircles set to:', roiView.showCircles);
    }
    
    if (roiView.refreshConfig) {
      setConfigKey(prev => prev + 1);
    }

    if (roiView.selectedGroups && JSON.stringify(roiView.selectedGroups) !== JSON.stringify(selectedGroups)) {
      console.log('Updating selectedGroups:', roiView.selectedGroups);
      setSelectedGroups(roiView.selectedGroups);
    }

    // Pass the roiView to parent component
    if (onSetView) {
      onSetView(roiView);
    }
  };

  const handleHeatmapResults = (results) => {
    setHeatmapResults(results);
  };

  const handleInteractionResults = (results) => {
    setInteractionHeatmapResult(results);
  };

  const handleCircleClick = (circleId) => {
    console.log('Circle clicked:', circleId);
    setSelectedCircle(circleId);
  };

  const changeHandler = (newConfig) => {
    console.log('Config changed:', newConfig);
    setConfig(newConfig);
  };

  // Tooltip event handlers
  const handleTooltipShow = (event) => {
    console.log('🖱️ Tooltip Show Event:', event);
  };

  const handleTooltipHide = (event) => {
    console.log('🖱️ Tooltip Hide Event:', event);
  };

  const handleMouseOver = (event) => {
    console.log('🖱️ Mouse Over Event:', event);
  };

  const handleMouseOut = (event) => {
    console.log('🖱️ Mouse Out Event:', event);
  };

  const handleGroupToggle = (groupId) => {
    setActiveGroups(prev => ({
      ...prev,
      [groupId]: !prev[groupId]
    }));
  };

  const renderInteractionHeatmap = () => {
    if (!interactionHeatmapResult || !interactionHeatmapResult.heatmaps) return null;

    const activeHeatmaps = Object.entries(interactionHeatmapResult.heatmaps)
      .filter(([group]) => activeGroups[group.split('_')[1]]);

    if (activeHeatmaps.length === 0) return null;

    const combinedHeatmap = activeHeatmaps.reduce((acc, [group, data], index) => {
      const normalizedData = data.map(row => 
        row.map(val => val * (index + 1) / activeHeatmaps.length)
      );
      
      if (acc.length === 0) {
        return normalizedData;
      }
      
      return acc.map((row, i) => 
        row.map((val, j) => val + normalizedData[i][j])
      );
    }, []);

    return (
      <div style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        padding: '15px',
        borderRadius: '8px',
        zIndex: 1000,
        width: '300px'
      }}>
        <div style={{ marginBottom: '10px' }}>
          {Object.entries(groupNames).map(([id, name]) => (
            <label key={id} style={{ 
              marginRight: '10px', 
              display: 'inline-block',
              color: 'white',
              fontSize: '12px'
            }}>
              <input
                type="checkbox"
                checked={activeGroups[id]}
                onChange={() => handleGroupToggle(id)}
                style={{ marginRight: '5px' }}
              />
              <span style={{ color: groupColors[id] }}>{name}</span>
            </label>
          ))}
        </div>
        <Plot
          data={[{
            z: combinedHeatmap,
            type: 'heatmap',
            colorscale: [
              [0, 'black'],
              [0.25, groupColors[1]],
              [0.5, groupColors[2]],
              [0.75, groupColors[3]],
              [1, groupColors[4]]
            ],
            showscale: true,
            colorbar: {
              title: 'Interaction Intensity',
              titleside: 'right',
              titlefont: { color: 'white' },
              tickfont: { color: 'white' }
            }
          }]}
          layout={{
            title: {
              text: 'Combined Interactions',
              font: { color: 'white' }
            },
            width: 280,
            height: 280,
            margin: { t: 30, b: 20, l: 20, r: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
          }}
          config={{ displayModeBar: false }}
        />
      </div>
    );
  };

  if (error) {
    return <p style={{ color: 'red', padding: '10px' }}>Error generating Mainview: {error}</p>;
  }
  if (!config) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <p>Loading Vitessce configuration...</p>
        <p style={{ fontSize: '12px', color: '#666' }}>Please wait while the 3D viewer is being prepared</p>
      </div>
    );
  }

  return (
    <div className="left-panel">
      {Object.keys(heatmapResults).length > 0 && (
        <div className="heatmaps-container">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1px' }}>
            <div>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(6, 200px)',
                gap: '1px',
                padding: '1px'
              }}>
                {Object.entries(heatmapResults).map(([channel, data]) => (
                  <div key={channel} style={{ 
                    border: '1px solid #ccc',
                    borderRadius: '5px',
                    padding: '1px',
                    backgroundColor: 'rgb(0, 0, 0,0.85)'
                  }}>
                    <Plot
                      data={[{
                        z: data,
                        type: 'heatmap',
                        colorscale: 'Viridis',
                        showscale: true
                      }]}
                      layout={{
                        title: {
                          text: ` ${channel}`,
                          font: {
                            size: 16,
                            color: '#ffffff'
                          },
                          y: 0.95
                        },
                        width: 200,
                        height: 200,
                        margin: { t: 30, b: 20, l: 20, r: 1},
                        paper_bgcolor: 'rgba(0,0,0,0.0)',
                        plot_bgcolor: 'rgba(0,0,0,0.0)'
                      }}
                      config={{ displayModeBar: false }}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {interactionHeatmapResult && renderInteractionHeatmap()}

      <div className="fullscreen-vitessce" style={{ position: 'relative', width: '100%', height: '100vh' }}>

        
        <Vitessce
          ref={vitessceRef}
          key={`${configKey}-${JSON.stringify(config?.datasets?.[0]?.files?.map(f => f.url))}`}
          config={config}
          onConfigChange={changeHandler}
          onTooltipShow={handleTooltipShow}
          onTooltipHide={handleTooltipHide}
          onMouseOver={handleMouseOver}
          onMouseOut={handleMouseOut}
          theme="light"
          height={null}
          width={null}
        />
        
        {showCircles && (
          <InteractiveCircles
            rois={rois}
            showCircles={showCircles}
            onCircleClick={handleCircleClick}
            selectedCircle={selectedCircle}
            selectedInteractions={selectedGroups}
          />
        )}
        
        <div className="roi-selector-container" style={{ position: 'absolute', top: '60px', left: 0, zIndex: 10 }}>
          <ROISelector 
            onSetView={handleSetView} 
            onHeatmapResults={handleHeatmapResults}
            onInteractionResults={handleInteractionResults}

            onGroupSelection={(groups) => {
              console.log('ROISelector onGroupSelection called with:', groups);
              setSelectedGroups(groups);
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default MainView; 
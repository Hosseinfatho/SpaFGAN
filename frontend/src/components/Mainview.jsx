import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';
import ROISelector from './ROISelector';
import Plot from 'react-plotly.js';

const MainView = () => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState({
    spatialTargetZ: 0,
    spatialTargetT: 0,
    spatialZoom: -3.0,
    spatialTargetX: 5500,
    spatialTargetY: 2880,
    spatialRenderingMode: "3D",
    imageLayer: [
      {
        spatialTargetResolution: 3,
        spatialLayerOpacity: 1.0,
        spatialLayerVisible: true,
        photometricInterpretation: "BlackIsZero",
        imageChannel: [
          {
            spatialTargetC: 19,
            spatialChannelColor: [0, 255, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [300, 20000]
          },
          {
            spatialTargetC: 27,
            spatialChannelColor: [255, 255, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1000, 7000]
          },
          {
            spatialTargetC: 37,
            spatialChannelColor: [255, 0, 255],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [700, 6000]
          },
          {
            spatialTargetC: 25,
            spatialChannelColor: [0, 255, 255],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1638, 10000]
          },
          {
            spatialTargetC: 42,
            spatialChannelColor: [65, 51, 97],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [370, 1432]
          },
          {
            spatialTargetC: 59,
            spatialChannelColor: [255, 0, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1638, 7000]
          }
        ]
      }
    ]
  });

  const [heatmapResults, setHeatmapResults] = useState({});
  const [interactionHeatmapResult, setInteractionHeatmapResult] = useState(null);
  const [activeGroups, setActiveGroups] = useState({
    1: true,
    2: true,
    3: true,
    4: true
  });

  // Group colors and names
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
    4: 'T–B collaboration (CD4 + CD20)'
  };

  const fetchConfig = (stateData) => {
    fetch('http://127.0.0.1:5000/api/generate_config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(stateData)
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Mainview config generated successfully:", data);
        setConfig(data);
      })
      .catch(err => {
        console.error("Error generating Mainview config:", err);
        setError(err.message);
      });
  };

  useEffect(() => {
    fetchConfig(viewState);
  }, [viewState]);

  const handleSetView = (roiView) => {
    setViewState(prev => ({
      ...prev,
      ...roiView
    }));
  };

  // Add handlers for results from ROISelector
  const handleHeatmapResults = (results) => {
    setHeatmapResults(results);
  };

  const handleInteractionResults = (results) => {
    setInteractionHeatmapResult(results);
  };

  // Analyze heatmaps
  const analyzeHeatmaps = async () => {
    setIsAnalyzingHeatmaps(true);
    try {
      const response = await fetch('http://localhost:5000/api/analyze_heatmaps', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          roi: {
            xMin: 0,
            xMax: 400,
            yMin: 0,
            yMax: 400,
            zMin: 0,
            zMax: 193
          }
        })
      });
      const data = await response.json();
      setHeatmapResults(data);
    } catch (error) {
      console.error('Error analyzing heatmaps:', error);
    } finally {
      setIsAnalyzingHeatmaps(false);
    }
  };

  // Analyze interaction heatmap
  const analyzeInteractionHeatmap = async () => {
    setIsAnalyzingInteractionHeatmap(true);
    try {
      const response = await fetch('http://localhost:5000/api/analyze_interaction_heatmap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          roi: {
            xMin: 0,
            xMax: 400,
            yMin: 0,
            yMax: 400,
            zMin: 0,
            zMax: 193
          }
        })
      });
      const data = await response.json();
      setInteractionHeatmapResult(data);
    } catch (error) {
      console.error('Error analyzing interaction heatmap:', error);
    } finally {
      setIsAnalyzingInteractionHeatmap(false);
    }
  };

  const handleGroupToggle = (groupId) => {
    setActiveGroups(prev => ({
      ...prev,
      [groupId]: !prev[groupId]
    }));
  };

  const renderInteractionHeatmap = () => {
    if (!interactionHeatmapResult || !interactionHeatmapResult.heatmaps) return null;

    // Get active heatmaps
    const activeHeatmaps = Object.entries(interactionHeatmapResult.heatmaps)
      .filter(([group]) => activeGroups[group.split('_')[1]]);

    if (activeHeatmaps.length === 0) return null;

    // Create a combined heatmap with different colors for each group
    const combinedHeatmap = activeHeatmaps.reduce((acc, [group, data], index) => {
      const groupId = group.split('_')[1];
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
    return <p style={{ padding: '10px' }}>Generating Mainview config...</p>;
  }

  return (
    <div className="left-panel">
      {/* Regular Heatmaps */}
      {Object.keys(heatmapResults).length > 0 && (
        <div style={{
          position: 'fixed',
          bottom: 1,
          left: 300,
          right: 'auto',
          width: 'fit-content',
          backgroundColor: 'rgba(244, 239, 239, 0.1)',
          padding: '10px',
          boxShadow: '0 2px 4px rgba(241, 228, 228, 0.1)',
          zIndex: 5,
          maxHeight: '50vh',
          overflowY: 'auto'
        }}>
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

      {/* Interaction Heatmap */}
      {interactionHeatmapResult && renderInteractionHeatmap()}

      <div className="fullscreen-vitessce">
        <Vitessce
          config={config}
          theme="light"
          height={null}
          width={null}
        />
        <div className="left-panel">
          <ROISelector 
            onSetView={handleSetView} 
            onHeatmapResults={handleHeatmapResults}
            onInteractionResults={handleInteractionResults}
          />
        </div>
      </div>
    </div>
  );
};

export default MainView; 
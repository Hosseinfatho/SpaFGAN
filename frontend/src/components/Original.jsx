import React, { useState, useEffect, useRef } from 'react';
import { Vitessce } from 'vitessce';
import ROISelector from './ROISelector';
import InteractiveCircles from './InteractiveCircles';
import Plot from 'react-plotly.js';

const MainView = () => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState(null);


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

  const fetchConfig = () => {
    console.log("Fetching config from:", 'http://localhost:5000/api/config');
    fetch('http://localhost:5000/api/config')
      .then(response => {
        console.log("Response status:", response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Vitessce config loaded successfully:", data);
        console.log("Config structure:", {
          version: data.version,
          name: data.name,
          datasetsCount: data.datasets?.length,
          layoutCount: data.layout?.length
        });
        setConfig(data);
      })
      .catch(err => {
        console.error("Error loading Vitessce config:", err);
        setError(err.message);
      });
  };

  useEffect(() => {
    fetchConfig();
  }, []);

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

          if (extracted.length > 0 && selectedGroups.length === 0) {
            const allInteractions = new Set();
            extracted.forEach(roi => {
              if (Array.isArray(roi.interactions)) {
                roi.interactions.forEach(interaction => allInteractions.add(interaction));
              }
            });
            const uniqueInteractions = Array.from(allInteractions);
            if (uniqueInteractions.length > 0) {
              setSelectedGroups([uniqueInteractions[0]]);
            }
          }
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
      setTimeout(() => {
        fetchConfig();
      }, 500);
    }

    if (roiView.selectedGroups && JSON.stringify(roiView.selectedGroups) !== JSON.stringify(selectedGroups)) {
      console.log('Updating selectedGroups:', roiView.selectedGroups);
      setSelectedGroups(roiView.selectedGroups);
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
    
    // Note: View state changes are now handled by the backend config
    // Circle clicks will be handled by Vitessce's internal state management
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
        {/* Debug info */}
        {process.env.NODE_ENV === 'development' && (
          <div style={{ 
            position: 'absolute', 
            top: '10px', 
            right: '10px', 
            background: 'rgba(0,0,0,0.8)', 
            color: 'white', 
            padding: '10px', 
            borderRadius: '5px',
            fontSize: '12px',
            zIndex: 1000,
            maxWidth: '300px'
          }}>
            <div>Config loaded: {config ? '✅' : '❌'}</div>
            <div>Version: {config?.version}</div>
            <div>Name: {config?.name}</div>
            <div>Datasets: {config?.datasets?.length || 0}</div>
            <div>Layout: {config?.layout?.length || 0}</div>
          </div>
        )}
        
        <Vitessce
          ref={vitessceRef}
          key={configKey}
          config={config}
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
          />
        </div>
      </div>
    </div>
  );
};

export default MainView; 
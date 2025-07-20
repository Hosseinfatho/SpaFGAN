import React from 'react';
import Plot from 'react-plotly.js';

const HeatmapResults = ({ 
  heatmapResults, 
  interactionHeatmapResult, 
  activeGroups, 
  groupColors, 
  groupNames,
  onClose,
  onHeatmapClick,
  onGroupToggle
}) => {
  console.log('HeatmapResults render:', { heatmapResults, interactionHeatmapResult, activeGroups });
  if (!heatmapResults && !interactionHeatmapResult) return null;

  return (
    <div className="heatmap-results-container">
      <button 
        onClick={onClose}
        className="btn-close"
      >
        ×
      </button>

      {/* Regular Heatmaps - Horizontal Layout */}
      {heatmapResults && heatmapResults.heatmaps && Object.keys(heatmapResults.heatmaps).length > 0 && (
        <div className="heatmap-grid">

          
          {Object.entries(heatmapResults.heatmaps).map(([channelName, channelData]) => (
            <div key={channelName} style={{ 
              position: 'relative',
              minWidth: '280px',
              flexShrink: 0,
              background: 'transparent'
            }}>

              <Plot
                data={[{
                  z: channelData.data.slice().reverse(),
                  type: 'heatmap',
                  colorscale: 'Viridis',
                  hoverongaps: false,
                  hovertemplate: 'X: %{x}<br>Y: %{y}<br>Intensity: %{z:.3f}<extra></extra>'
                }]}
                layout={{
                  width: 280,
                  height: 200,
                  margin: { t: 25, b: 25, l: 25, r: 25 },
                  paper_bgcolor: 'rgba(0, 0, 0, 0.1)',
                                    plot_bgcolor: 'rgba(0, 0, 0, 0.1)',
                  title: {
                    text: channelName,
                    font: { color: 'white', size: 16 },
                    x: 0.5
                  },
                  xaxis: {
                    title: 'X',
                    titlefont: { color: 'white', size: 12 },
                    tickfont: { color: 'white', size: 10 }
                  },
                  yaxis: {
                    title: 'Y',
                    titlefont: { color: 'white', size: 12 },
                    tickfont: { color: 'white', size: 10 }
                  }
                }}
                config={{ 
                  displayModeBar: false,
                  responsive: true
                }}
                onClick={onHeatmapClick}
                style={{
                  background: 'transparent'
                }}
              />
            </div>
          ))}
        </div>
      )}

      {/* Interaction Heatmap - Simple Checkboxes */}
      {interactionHeatmapResult && (
        <div className="interaction-heatmap-container">
          {/* Interaction Checkboxes */}
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column',
            gap: '4px',
            background: 'transparent'
          }}>
            {Object.entries(groupNames).map(([group, name]) => (
              <label key={group} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '6px',
                color: 'white',
                background: 'transparent',
                padding: '4px 8px',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'background-color 0.2s',
                border: '1px solid rgba(255, 255, 255, 0.2)'
              }}>
                <input
                  type="checkbox"
                  checked={activeGroups[group]}
                  onChange={() => onGroupToggle(group)}
                  style={{ margin: 0, transform: 'scale(1.2)' }}
                />
                <span style={{ color: groupColors[group], fontWeight: 'bold', fontSize: '14px' }}>{name}</span>
              </label>
            ))}
          </div>

          {/* Selected Interaction Heatmaps Overlay */}
          {(() => {
            const activeGroupsList = Object.entries(activeGroups).filter(([group, isActive]) => isActive);
            if (activeGroupsList.length === 0) return null;
            
            // Create overlay data for all active groups with normalization
            const overlayData = activeGroupsList.map(([groupId]) => {
              const groupKey = `group_${groupId}`;
              const groupData = interactionHeatmapResult.heatmaps[groupKey];
              if (!groupData) return null;
              
              // Normalize the data to 0-1 range
              const flatData = groupData.flat();
              const minVal = Math.min(...flatData);
              const maxVal = Math.max(...flatData);
              const range = maxVal - minVal;
              
              const normalizedData = groupData.slice().reverse().map(row => 
                row.map(val => range > 0 ? (val - minVal) / range : 0)
              );
              
              return {
                z: normalizedData,
                type: 'heatmap',
                colorscale: [[0, 'rgba(0,0,0,0)'], [1, groupColors[groupId]]],
                showscale: false,
                opacity: 0.7,
                name: groupNames[groupId]
              };
            }).filter(Boolean);
            
            if (overlayData.length === 0) return null;
            
            return (
              <div style={{ 
                position: 'relative',
                background: 'transparent',
                display: 'flex',
                justifyContent: 'center'
              }}>
                
                
                                 <Plot
                   data={overlayData}
                   layout={{
                     width: 300,
                     height: 150,
                     margin: { t: 20, b: 20, l: 20, r: 20 },
                     paper_bgcolor: 'rgba(0, 0, 0, 0.1)',
                     plot_bgcolor: 'rgba(0, 0, 0, 0.1)',
                     xaxis: {
                       title: 'X',
                       titlefont: { color: 'white', size: 8 },
                       tickfont: { color: 'white', size: 6 },
                       showgrid: false,
                       showticklabels: false,
                       zeroline: false
                     },
                     yaxis: {
                       title: 'Y',
                       titlefont: { color: 'black', size: 8 },
                       tickfont: { color: 'black', size: 6 },
                       showgrid: false,
                       showticklabels: false,
                       zeroline: false
                     }
                   }}
                  config={{ 
                    displayModeBar: false,
                    responsive: true
                  }}
                  onClick={onHeatmapClick}
                  style={{
                    background: 'transparent'
                  }}
                />
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default HeatmapResults; 
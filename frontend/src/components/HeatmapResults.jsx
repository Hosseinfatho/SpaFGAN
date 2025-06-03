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
  if (!heatmapResults && !interactionHeatmapResult) return null;

  return (
    <div style={{ 
      position: 'fixed',
      bottom: '20px',
      left: '20px',
      right: '20px',
      backgroundColor: 'transparent',
      padding: '20px',
      borderRadius: '8px',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      <button 
        onClick={onClose}
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          background: 'none',
          border: 'none',
          color: 'white',
          fontSize: '20px',
          cursor: 'pointer',
          zIndex: 10
        }}
      >
        ×
      </button>

      {/* Regular Heatmaps */}
      {Object.keys(heatmapResults).length > 0 && (
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px',
          padding: '10px',
          background: 'transparent'
        }}>
          {Object.entries(heatmapResults).map(([channelIndex, result]) => (
            <div key={channelIndex} style={{ 
              position: 'relative',
              background: 'transparent'
            }}>
              <div style={{
                position: 'absolute',
                top: '-25px',
                left: '0',
                color: '#ffffff',
                fontSize: '14px',
                fontWeight: 'bold',
                textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
                zIndex: 2,
                backgroundColor: 'rgba(0, 0, 0, 0.5)',
                padding: '2px 8px',
                borderRadius: '4px'
              }}>
                {channelIndex}
              </div>
              <Plot
                data={[{
                  z: result.data,
                  type: 'heatmap',
                  colorscale: 'Viridis'
                }]}
                layout={{
                  width: 300,
                  height: 300,
                  margin: { t: 20, b: 20, l: 20, r: 20 },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)'
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

      {/* Interaction Heatmap */}
      {interactionHeatmapResult && (
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '20px',
          background: 'transparent'
        }}>
          <div style={{ 
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '20px',
            background: 'transparent'
          }}>
            {Object.entries(interactionHeatmapResult.heatmaps).map(([group, data]) => (
              <div key={group} style={{ 
                position: 'relative',
                background: 'transparent'
              }}>
                <div style={{
                  position: 'absolute',
                  top: '-25px',
                  left: '0',
                  color: '#ffffff',
                  fontSize: '14px',
                  fontWeight: 'bold',
                  textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)',
                  zIndex: 2,
                  backgroundColor: 'rgba(0, 0, 0, 0.5)',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  {groupNames[group.split('_')[1]]}
                </div>
                <Plot
                  data={[{
                    z: data,
                    type: 'heatmap',
                    colorscale: [[0, 'black'], [1, groupColors[group.split('_')[1]]]]
                  }]}
                  layout={{
                    width: 300,
                    height: 300,
                    margin: { t: 0, b: 20, l: 20, r: 20 },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
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

          {/* Group Toggles */}
          <div style={{ 
            display: 'flex', 
            gap: '10px', 
            flexWrap: 'wrap',
            background: 'transparent'
          }}>
            {Object.entries(activeGroups).map(([group, isActive]) => (
              <label key={group} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '5px',
                color: 'white',
                background: 'rgba(0, 0, 0, 0.5)',
                padding: '5px 10px',
                borderRadius: '4px'
              }}>
                <input
                  type="checkbox"
                  checked={isActive}
                  onChange={() => onGroupToggle(group)}
                />
                {groupNames[group]}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapResults; 
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
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
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
          cursor: 'pointer'
        }}
      >
        ×
      </button>

      {/* Regular Heatmaps */}
      {Object.keys(heatmapResults).length > 0 && (
        <div style={{ display: 'flex', gap: '10px', overflowX: 'auto', padding: '10px' }}>
          {Object.entries(heatmapResults).map(([channelIndex, result]) => (
            <div key={channelIndex} style={{ minWidth: '300px' }}>
              <Plot
                data={[{
                  z: result.data,
                  type: 'heatmap',
                  colorscale: 'Viridis'
                }]}
                layout={{
                  title: `Channel ${channelIndex}`,
                  width: 300,
                  height: 300,
                  margin: { t: 30, b: 20, l: 20, r: 20 }
                }}
                config={{ displayModeBar: false }}
                onClick={onHeatmapClick}
              />
            </div>
          ))}
        </div>
      )}

      {/* Interaction Heatmap */}
      {interactionHeatmapResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {Object.entries(interactionHeatmapResult.heatmaps).map(([group, data]) => (
              <div key={group} style={{ minWidth: '300px' }}>
                <Plot
                  data={[{
                    z: data,
                    type: 'heatmap',
                    colorscale: [[0, 'black'], [1, groupColors[group.split('_')[1]]]]
                  }]}
                  layout={{
                    title: groupNames[group.split('_')[1]],
                    width: 300,
                    height: 300,
                    margin: { t: 30, b: 20, l: 20, r: 20 }
                  }}
                  config={{ displayModeBar: false }}
                  onClick={onHeatmapClick}
                />
              </div>
            ))}
          </div>

          {/* Group Toggles */}
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {Object.entries(activeGroups).map(([group, isActive]) => (
              <label key={group} style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '5px',
                color: 'white'
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
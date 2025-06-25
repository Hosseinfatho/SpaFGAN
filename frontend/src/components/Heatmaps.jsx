import React, { useState } from 'react';

function Heatmaps({ currentROI, onHeatmapResults }) {
  const [isAnalyzingHeatmaps, setIsAnalyzingHeatmaps] = useState(false);
  const [heatmapResults, setHeatmapResults] = useState({});

  // Define specific channels to use
  const SPECIFIC_CHANNELS = ['CD31', 'CD11b', 'Catalase', 'CD4', 'CD20', 'CD11c'];

  const analyzeHeatmaps = async () => {
    setIsAnalyzingHeatmaps(true);
    try {
      // Scale factor for coordinates
      const factor = 8;
      const roiSize = 800; // Keep original ROI size

      // Get current ROI coordinates
      const x = Number(currentROI.x) || 0;
      const y = Number(currentROI.y) || 0;

      // Calculate ROI with original size, then scale for API
      const roi = {
        xMin: Math.max(0, Math.floor((x - roiSize/2) / factor)),
        xMax: Math.min(1363, Math.floor((x + roiSize/2) / factor)),
        yMin: Math.max(0, Math.floor((y - roiSize/2) / factor)),
        yMax: Math.min(688, Math.floor((y + roiSize/2) / factor)),
        zMin: 0,
        zMax: 193
      };

      // Ensure equal x and y ranges
      const xRange = roi.xMax - roi.xMin;
      const yRange = roi.yMax - roi.yMin;
      const maxRange = Math.max(xRange, yRange);

      // Center the ROI
      const centerX = (roi.xMin + roi.xMax) / 2;
      const centerY = (roi.yMin + roi.yMax) / 2;

      // Adjust ROI to have equal ranges
      roi.xMin = Math.max(0, Math.floor(centerX - maxRange/2));
      roi.xMax = Math.min(1363, Math.floor(centerX + maxRange/2));
      roi.yMin = Math.max(0, Math.floor(centerY - maxRange/2));
      roi.yMax = Math.min(688, Math.floor(centerY + maxRange/2));

      console.log('Sending ROI data:', {
        original: { x, y, size: roiSize },
        scaled: roi,
        ranges: {
          x: roi.xMax - roi.xMin,
          y: roi.yMax - roi.yMin
        }
      });

      const response = await fetch('http://localhost:5000/api/analyze_heatmaps', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          roi,
          channels: SPECIFIC_CHANNELS,
          range: {
            min: 0,
            max: 1
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze heatmaps');
      }

      const data = await response.json();
      console.log('Received heatmap data:', data);
      setHeatmapResults(data);
      onHeatmapResults(data);
    } catch (error) {
      console.error('Error analyzing heatmaps:', error);
      alert('Error analyzing heatmaps: ' + error.message);
    } finally {
      setIsAnalyzingHeatmaps(false);
    }
  };

  return (
    <div>
      <button 
        onClick={analyzeHeatmaps}
        disabled={isAnalyzingHeatmaps}
        style={{
          backgroundColor: '#ffc107',
          color: 'black',
          border: 'none',
          borderRadius: '5px',
          padding: '12px 24px',
          fontSize: '16px',
          marginRight: '10px',
          cursor: isAnalyzingHeatmaps ? 'not-allowed' : 'pointer'
        }}
      >
        {isAnalyzingHeatmaps ? 'Analyzing...' : 'Analyze Heatmaps'}
      </button>
    </div>
  );
}

export default Heatmaps; 
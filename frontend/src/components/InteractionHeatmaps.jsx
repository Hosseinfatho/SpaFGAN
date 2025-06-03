import React, { useState } from 'react';

function InteractionHeatmaps({ currentROI, onInteractionResults }) {
  const [isAnalyzingInteractionHeatmap, setIsAnalyzingInteractionHeatmap] = useState(false);
  const [interactionHeatmapResult, setInteractionHeatmapResult] = useState(null);

  const analyzeInteractionHeatmap = async () => {
    setIsAnalyzingInteractionHeatmap(true);
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

      const response = await fetch('http://localhost:5000/api/analyze_interaction_heatmap', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ roi })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze interaction heatmap');
      }

      const data = await response.json();
      console.log('Received interaction heatmap data:', data);
      setInteractionHeatmapResult(data);
      onInteractionResults(data);
    } catch (error) {
      console.error('Error analyzing interaction heatmap:', error);
      alert('Error analyzing interaction heatmap: ' + error.message);
    } finally {
      setIsAnalyzingInteractionHeatmap(false);
    }
  };

  return (
    <div>
      <button 
        onClick={analyzeInteractionHeatmap}
        disabled={isAnalyzingInteractionHeatmap}
        style={{
          backgroundColor: '#6f42c1',
          color: 'black',
          border: 'none',
          borderRadius: '4px',
          padding: '8px 96px',
          marginRight: '5px',
          marginLeft: '10px',
          cursor: isAnalyzingInteractionHeatmap ? 'not-allowed' : 'pointer'
        }}
      >
        {isAnalyzingInteractionHeatmap ? 'Analyzing...' : 'Analyze Interactions'}
      </button>
    </div>
  );
}

export default InteractionHeatmaps; 
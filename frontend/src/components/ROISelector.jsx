import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

// Define scale factors (Zarr level 3 to full-res)
const factorX = 1; //10908 / 1363;
const factorY = 1; //5508 / 688;
const fullHeight = 5508; // needed for Y flip
const getDefaultColor = (index) => {
  const colors = [
    [255, 0, 255], [0, 255, 255],[255, 255, 0], // Yellow, Cyan, Magenta
    [255, 0, 0], [0, 255, 0], [0, 0, 255], // R, G, B
    [255, 165, 0], [128, 0, 128] // Orange, Purple
    // Add more colors if needed
  ];
  return colors[index % colors.length];
};

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

function ROISelector({ onSetView, onHeatmapResults, onInteractionResults }) {
  // Move all state declarations inside the component
  const [rois, setRois] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGroups, setSelectedGroups] = useState([]);
  const [interactionGroups, setInteractionGroups] = useState([]);
  const [manualX, setManualX] = useState("");
  const [manualY, setManualY] = useState("");
  const [heatmapResults, setHeatmapResults] = useState({});
  const [isAnalyzingHeatmaps, setIsAnalyzingHeatmaps] = useState(false);
  const [interactionHeatmapResult, setInteractionHeatmapResult] = useState(null);
  const [isAnalyzingInteractionHeatmap, setIsAnalyzingInteractionHeatmap] = useState(false);
  const [hiddenChannelHeatmaps, setHiddenChannelHeatmaps] = useState([]);
  const [activeGroups, setActiveGroups] = useState({
    1: true,
    2: true,
    3: true,
    4: true
  });
  const [selectedGroup, setSelectedGroup] = useState(1);

  const computeCentroid = (allCoords) => {
    const flatCoords = allCoords.flat();
    const sum = flatCoords.reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
    return [sum[0] / flatCoords.length, sum[1] / flatCoords.length];
  };

  useEffect(() => {
    fetch("http://localhost:5000/api/roi_shapes")
      .then((res) => res.json())
      .then((data) => {
        if (!data.features || !Array.isArray(data.features)) {
          console.error("Invalid ROI data structure:", data);
          return;
        }

        const extracted = data.features.map((feature, index) => {
          const geometry = feature.geometry;
          if (!geometry || !geometry.coordinates) return null;

          let allCoords = [];
          if (geometry.type === "Polygon") {
            allCoords = geometry.coordinates;
          } else if (geometry.type === "MultiPolygon") {
            allCoords = geometry.coordinates.flat();
          } else {
            return null;
          }

          const [cx, cy] = computeCentroid(allCoords);

          return {
            id: feature.properties.name || `ROI_${index}`,
            x: cx * factorX,
            y: -(cy * factorY) + fullHeight,
            score: feature.properties.score || 0,
            interactions: feature.properties.interactions || [],
            raw: feature.properties // Store all original metadata
          };
        }).filter(Boolean);

        setRois(extracted);

        const uniqueGroups = [...new Set(extracted.flatMap(r => r.interactions))];
        setInteractionGroups(uniqueGroups);
        setSelectedGroups(uniqueGroups.slice(0, 1));
      })
      .catch((err) => console.error("Failed to load ROI shapes:", err));
  }, []);

  const filteredRois = selectedGroups.length > 0
    ? rois.filter(roi => roi.interactions.some(i => selectedGroups.includes(i)))
    : rois;

  const currentROI = filteredRois[currentIndex] || {};
  const displayX = manualX !== "" ? manualX : currentROI.x?.toFixed(0) || "";
  const displayY = manualY !== "" ? manualY : currentROI.y?.toFixed(0) || "";

  const handleSet = () => {
    if (filteredRois.length > 0 && onSetView) {
      const x = manualX !== "" ? Number(manualX) : currentROI.x;
      const y = manualY !== "" ? Number(manualY) : currentROI.y;
      onSetView({
        spatialTargetX: x,
        spatialTargetY: y,
        spatialTargetZ: 0,
        spatialZoom: -1.0,
      });
    }
  };

  const toggleGroup = (group) => {
    setSelectedGroups(prev =>
      prev.includes(group) ? prev.filter(g => g !== group) : [...prev, group]
    );
    setCurrentIndex(0);
  };

  const next = () => {
    setCurrentIndex(i => (i + 1) % filteredRois.length);
  };

  const prev = () => {
    setCurrentIndex(i => (i - 1 + filteredRois.length) % filteredRois.length);
  };

  // Add new functions for heatmap analysis
  const analyzeHeatmaps = async () => {
    setIsAnalyzingHeatmaps(true);
    try {
      // Scale factor for coordinates
      const factor = 8;
      const roiSize = 200; // Keep original ROI size

      // Get current ROI coordinates
      const x = Number(currentROI.x) || 0;
      const y = Number(currentROI.y) || 0;

      // Calculate ROI with original size, then scale for API
      const roi = {
        xMin: Math.max(0, Math.floor((x - roiSize) / factor)),
        xMax: Math.min(1363, Math.floor((x + roiSize) / factor)),
        yMin: Math.max(0, Math.floor((y - roiSize) / factor)),
        yMax: Math.min(688, Math.floor((y + roiSize) / factor)),
        zMin: 0,
        zMax: 193
      };

      console.log('Sending ROI data for heatmaps:', {
        original: { x, y, size: roiSize },
        scaled: roi
      });

      const response = await fetch('http://localhost:5000/api/analyze_heatmaps', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ roi })
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

  const analyzeInteractionHeatmap = async () => {
    setIsAnalyzingInteractionHeatmap(true);
    try {
      // Scale factor for coordinates
      const factor = 8;
      const roiSize = 200; // Keep original ROI size

      // Get current ROI coordinates
      const x = Number(currentROI.x) || 0;
      const y = Number(currentROI.y) || 0;

      // Calculate ROI with original size, then scale for API
      const roi = {
        xMin: Math.max(0, Math.floor((x - roiSize) / factor)),
        xMax: Math.min(1363, Math.floor((x + roiSize) / factor)),
        yMin: Math.max(0, Math.floor((y - roiSize) / factor)),
        yMax: Math.min(688, Math.floor((y + roiSize) / factor)),
        zMin: 0,
        zMax: 193
      };

      console.log('Sending ROI data:', {
        original: { x, y, size: roiSize },
        scaled: roi
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

  if (interactionGroups.length === 0) {
    return <p>Loading ROIs or no interactions found...</p>;
  }

  if (filteredRois.length === 0) {
    return <p>No ROIs found for selected groups: {selectedGroups.join(", ")}</p>;
  }

  return (
    <div style={{ padding: "10px", border: "1px solid #ccc", marginBottom: "10px" }}>
      <h4>ROI Navigator</h4>
      <p>Select Interaction Types:</p>
      {interactionGroups.map(group => (
        <label key={group} style={{ display: "block", marginLeft: "10px" }}>
          <input
            type="checkbox"
            checked={selectedGroups.includes(group)}
            onChange={() => toggleGroup(group)}
          />
          {group}
        </label>
      ))}

      <hr />
      <p><strong>{currentROI.id}</strong></p>
      <p>
        X: <input type="number" value={displayX} onChange={e => setManualX(e.target.value)} style={{ width: 80 }} />
        &nbsp;|&nbsp;
        Y: <input type="number" value={displayY} onChange={e => setManualY(e.target.value)} style={{ width: 80 }} />
        &nbsp;|&nbsp; Score: {currentROI.score}
      </p>
      <p>
        Interactions: {currentROI.interactions?.join(", ") || "None"}
      </p>
      {currentROI.raw && (
        <div style={{ fontSize: "small", marginTop: "5px" }}>
          <p><strong>Raw Marker Means:</strong></p>
          <ul>
            {Object.entries(currentROI.raw.marker_values || {}).map(([k, v]) => (
              <li key={k}>{k}: {v.toFixed(2)}</li>
            ))}
          </ul>
        </div>
      )}
      <button onClick={prev}>Previous</button>
      <button onClick={next}>Next</button>
      <button onClick={handleSet}>Set View</button>

      {/* Analysis Buttons */}
      <div style={{ marginTop: "10px" }}>
        <button 
          onClick={analyzeHeatmaps}
          disabled={isAnalyzingHeatmaps}
          style={{
            backgroundColor: '#ffc107',
            color: 'black',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 10px',
            marginRight: '10px',
            cursor: isAnalyzingHeatmaps ? 'not-allowed' : 'pointer'
          }}
        >
          {isAnalyzingHeatmaps ? 'Analyzing...' : 'Analyze Heatmaps'}
        </button>

        <button 
          onClick={analyzeInteractionHeatmap}
          disabled={isAnalyzingInteractionHeatmap}
          style={{
            backgroundColor: '#6f42c1',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            padding: '5px 10px',
            cursor: isAnalyzingInteractionHeatmap ? 'not-allowed' : 'pointer'
          }}
        >
          {isAnalyzingInteractionHeatmap ? 'Analyzing...' : 'Analyze Interactions'}
        </button>
      </div>
    </div>
  );
}

export default ROISelector;

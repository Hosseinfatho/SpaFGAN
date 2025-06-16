import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import Heatmaps from './Heatmaps';
import InteractionHeatmaps from './InteractionHeatmaps';

// Define scale factors (Zarr level 3 to full-res)
const factorX = 1; //10908 / 1363  multiple in vittnesse config;
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
    console.log('Fetching ROI shapes...');
    fetch("http://localhost:5000/api/roi_shapes")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log("Received ROI data:", data);
        if (!data.features || !Array.isArray(data.features)) {
          console.error("Invalid ROI data structure:", data);
          return;
        }

        const extracted = data.features.map((feature, index) => {
          const geometry = feature.geometry;
          if (!geometry || !geometry.coordinates) {
            console.warn(`Invalid geometry for feature ${index}:`, feature);
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

          const [cx, cy] = computeCentroid(allCoords);

          return {
            id: feature.properties.name || `ROI_${index}`,
            x: cx * factorX,
            y: cy * factorY,
            score: feature.properties.score || 0,
            interactions: feature.properties.interactions || [],
            raw: feature.properties // Store all original metadata
          };
        }).filter(Boolean);

        console.log("Extracted ROIs:", extracted);
        setRois(extracted);

        const uniqueGroups = [...new Set(extracted.flatMap(r => r.interactions))];
        console.log("Unique interaction groups:", uniqueGroups);
        setInteractionGroups(uniqueGroups);
        setSelectedGroups(uniqueGroups.slice(0, 1));
      })
      .catch((err) => {
        console.error("Failed to load ROI shapes:", err);
        // Add a more user-friendly error message
        setRois([]);
        setInteractionGroups([]);
      });
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

  if (interactionGroups.length === 0) {
    return <p>Loading ROIs or no interactions found...</p>;
  }

  if (filteredRois.length === 0) {
    return <p>No ROIs found for selected groups: {selectedGroups.join(", ")}</p>;
  }

  return (
    <div style={{ padding: "1px", border: "1px solid #ccc", marginBottom: "1px" ,transform: "scale(0.8)"}}>
      <h4>ROI Navigator</h4>
      <p>Select Interaction Types:</p>
      {interactionGroups.map(group => (
        <label key={group} style={{ display: "block", marginLeft: "1px" }}>
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
        X: <input type="number" value={displayX} onChange={e => setManualX(e.target.value)} style={{ width: 60 }} />
        &nbsp;|&nbsp;
        Y: <input type="number" value={displayY} onChange={e => setManualY(e.target.value)} style={{ width: 60 }} />
        &nbsp;|&nbsp; Score: {currentROI.score?.toFixed(3) || "0.000"}
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
        <Heatmaps 
          currentROI={currentROI}
          onHeatmapResults={onHeatmapResults}
        />
        <InteractionHeatmaps 
          currentROI={currentROI}
          onInteractionResults={onInteractionResults}
        />
      </div>
    </div>
  );
}

export default ROISelector;

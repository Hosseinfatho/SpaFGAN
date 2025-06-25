import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import Heatmaps from './Heatmaps';
import InteractionHeatmaps from './InteractionHeatmaps';
import './ROISelector.css';

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
  const [showCircles, setShowCircles] = useState(false);
  const [viewState, setViewState] = useState({});
  const [config, setConfig] = useState(null);
  const [configKey, setConfigKey] = useState(0);
  const [circleOverlay, setCircleOverlay] = useState({
    show: false,
    circles: [],
    selectedCircle: null
  });

  const computeCentroid = (allCoords) => {
    const flatCoords = allCoords.flat();
    const sum = flatCoords.reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
    return [sum[0] / flatCoords.length, sum[1] / flatCoords.length];
  };

  useEffect(() => {
    console.log('ROISelector: Starting to fetch ROI shapes...');
    fetch("http://localhost:5000/api/roi_shapes")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log("ROISelector: Received ROI data:", data);
        if (!data.features || !Array.isArray(data.features)) {
          console.error("ROISelector: Invalid ROI data structure:", data);
          return;
        }

        console.log("ROISelector: Processing", data.features.length, "ROI features");

        const extracted = data.features.map((feature, index) => {
          const geometry = feature.geometry;
          if (!geometry || !geometry.coordinates) {
            console.warn(`ROISelector: Invalid geometry for feature ${index}:`, feature);
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

        console.log("ROISelector: Extracted ROIs:", extracted);
        setRois(extracted);

        // Extract unique interaction types from all ROIs
        const allInteractions = new Set();
        extracted.forEach(roi => {
          if (Array.isArray(roi.interactions)) {
            roi.interactions.forEach(interaction => allInteractions.add(interaction));
          }
        });
        
        const uniqueGroups = Array.from(allInteractions);
        console.log("ROISelector: Unique interaction groups:", uniqueGroups);
        setInteractionGroups(uniqueGroups);
        
        // Select the first interaction type by default
        if (uniqueGroups.length > 0) {
          console.log("ROISelector: Setting initial selectedGroups to:", [uniqueGroups[0]]);
          setSelectedGroups([uniqueGroups[0]]);
        }
      })
      .catch((err) => {
        console.error("ROISelector: Failed to load ROI shapes:", err);
        // Add a more user-friendly error message
        setRois([]);
        setInteractionGroups([]);
      });
  }, []);

  // Notify parent component when selectedGroups changes
  useEffect(() => {
    if (selectedGroups.length > 0) {
      onSetView({
        selectedGroups: selectedGroups
      });
    }
  }, [selectedGroups, onSetView]);

  // Set initial selectedGroups when interactionGroups are loaded
  useEffect(() => {
    if (interactionGroups.length > 0 && selectedGroups.length === 0) {
      setSelectedGroups([interactionGroups[0]]);
    }
  }, [interactionGroups, selectedGroups.length]);

  const filteredRois = selectedGroups.length > 0
    ? rois.filter(roi => {
        if (!roi.interactions || !Array.isArray(roi.interactions)) {
          return false;
        }
        return roi.interactions.some(interaction => selectedGroups.includes(interaction));
      })
    : rois;

  console.log('ROISelector Debug:', {
    totalRois: rois.length,
    selectedGroups,
    filteredRois: filteredRois.length,
    sampleRoi: rois[0] ? {
      id: rois[0].id,
      interactions: rois[0].interactions,
      score: rois[0].score
    } : null
  });

  const currentROI = filteredRois[currentIndex] || {};

  const handleSetView = (roiView) => {
    setViewState(prev => ({
      ...prev,
      ...roiView
    }));

    // اگر نیاز به رفرش کانفیگ بود
    if (roiView.refreshConfig) {
      setConfig(null); // ابتدا config را خالی کن تا Vitessce مجبور به رندر مجدد شود
      setTimeout(() => {
        fetchConfig(viewState); // کانفیگ جدید را بگیر
      }, 500);
    }
  };

  const handleShowCirclesToggle = () => {
    const newShowCircles = !showCircles;
    setShowCircles(newShowCircles);
    
    // Send the toggle state to parent component
    onSetView({
      showCircles: newShowCircles
    });
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

  const handleCircleClick = (circleId) => {
    console.log('Circle clicked:', circleId);
    setCircleOverlay(prev => ({
      ...prev,
      selectedCircle: circleId
    }));
  };

  if (interactionGroups.length === 0) {
    return <p>Loading ROIs or no interactions found...</p>;
  }

  if (filteredRois.length === 0) {
    return (
      <div style={{ padding: "10px", border: "1px solid #ccc", marginBottom: "10px" }}>
        <h4>ROI Navigator</h4>
        <p>No ROIs found for selected groups: {selectedGroups.join(", ")}</p>
        <p>Available interaction types:</p>
        {interactionGroups.map(group => (
          <label key={group} style={{ display: "block", marginLeft: "10px", marginBottom: "5px" }}>
            <input
              type="checkbox"
              checked={selectedGroups.includes(group)}
              onChange={() => toggleGroup(group)}
            />
            {group}
          </label>
        ))}
        <div style={{ textAlign: "center", marginTop: "15px" }}>
          <button 
            onClick={() => setSelectedGroups(interactionGroups.slice(0, 1))}
            style={{ 
              padding: "8px 16px", 
              fontSize: "14px", 
              backgroundColor: "#007bff", 
              color: "white", 
              border: "none", 
              borderRadius: "5px", 
              cursor: "pointer"
            }}
          >
            Select First Interaction Type
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "10px", border: "1px solid #ccc", marginBottom: "10px" }}>
      <h4>ROI Navigator</h4>
      <p>Select Interaction Types:</p>
      {interactionGroups.map(group => (
        <label key={group} style={{ display: "block", marginLeft: "10px", marginBottom: "5px" }}>
          <input
            type="checkbox"
            checked={selectedGroups.includes(group)}
            onChange={() => toggleGroup(group)}
          />
          {group}
        </label>
      ))}

      <hr />
      <div style={{ textAlign: "center", marginBottom: "15px" }}>
        <h3 style={{ margin: "5px 0" }}>ROI #{currentIndex + 1}</h3>
        <p style={{ fontSize: "18px", fontWeight: "bold", margin: "5px 0" }}>
          Score: {currentROI.score?.toFixed(3) || "0.000"}
        </p>
        <p style={{ fontSize: "14px", color: "#666", margin: "5px 0" }}>
          {currentROI.interactions?.join(", ") || "None"}
        </p>
      </div>

      <div style={{ textAlign: "center", marginBottom: "15px" }}>
        <label style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
          <input
            type="checkbox"
            checked={showCircles}
            onChange={handleShowCirclesToggle}
            style={{ transform: "scale(1.2)" }}
          />
          <span style={{ fontSize: "14px" }}>Show interactive ROI circles</span>
        </label>
      </div>

      <div style={{ textAlign: "center" }}>
        <button 
          onClick={prev}
          style={{ 
            padding: "8px 12px", 
            fontSize: "14px", 
            backgroundColor: "#007bff", 
            color: "white", 
            border: "none", 
            borderRadius: "5px", 
            cursor: "pointer",
            marginRight: "10px"
          }}
        >
          ←
        </button>
        <button 
          onClick={handleSetView}
          style={{ 
            padding: "12px 24px", 
            fontSize: "16px", 
            backgroundColor: "#007bff", 
            color: "white", 
            border: "none", 
            borderRadius: "5px", 
            cursor: "pointer",
            marginRight: "10px"
          }}
        >
          Set View
        </button>
        <button 
          onClick={next}
          style={{ 
            padding: "8px 12px", 
            fontSize: "14px", 
            backgroundColor: "#007bff", 
            color: "white", 
            border: "none", 
            borderRadius: "5px", 
            cursor: "pointer"
          }}
        >
          →
        </button>
      </div>

      {/* Analysis Buttons */}
      <div style={{ marginTop: "15px", textAlign: "center" }}>
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

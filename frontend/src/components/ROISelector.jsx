import React, { useState, useEffect } from 'react';
import Heatmaps from './Heatmaps';
import InteractionHeatmaps from './InteractionHeatmaps';
import './ROISelector.css';

function ROISelector({ onSetView, onHeatmapResults, onInteractionResults }) {
  const [rois, setRois] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGroups, setSelectedGroups] = useState([]);
  const [interactionGroups, setInteractionGroups] = useState([]);
  const [showCircles, setShowCircles] = useState(false);

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
            x: cx,
            y: cy,
            score: feature.properties.score || 0,
            interactions: feature.properties.interactions || [],
            raw: feature.properties
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
      })
      .catch((err) => {
        console.error("ROISelector: Failed to load ROI shapes:", err);
        setRois([]);
        setInteractionGroups([]);
      });
  }, []);

  // Notify parent component when selectedGroups changes
  // useEffect(() => {
  //   if (selectedGroups.length > 0) {
  //     onSetView({
  //       selectedGroups: selectedGroups
  //     });
  //   }
  // }, [selectedGroups, onSetView]);

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

  const handleSetView = () => {
    if (currentROI && currentROI.x !== undefined && currentROI.y !== undefined) {
      const roiX = currentROI.x;
      const roiY = currentROI.y;
      
      const viewConfig = {
        spatialTargetX: roiX,
        spatialTargetY: roiY,
        spatialZoom: -1.0,
        refreshConfig: true
      };
      
      console.log('Setting view to ROI:', currentROI.id, 'at position:', roiX, roiY);
      onSetView(viewConfig);
    } else {
      console.warn('No valid ROI selected for Set View');
    }
  };

  const handleShowCirclesToggle = () => {
    const newShowCircles = !showCircles;
    setShowCircles(newShowCircles);
    
    onSetView({
      showCircles: newShowCircles
    });
  };

  const toggleGroup = (group) => {
    const newSelectedGroups = selectedGroups.includes(group) 
      ? selectedGroups.filter(g => g !== group) 
      : [...selectedGroups, group];
    
    setSelectedGroups(newSelectedGroups);
    setCurrentIndex(0);
    
    // Notify parent component about the change
    onSetView({
      selectedGroups: newSelectedGroups
    });
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
      
      {selectedGroups.length === 0 && (
        <div style={{ 
          marginTop: "10px", 
          padding: "8px", 
          backgroundColor: "#fff3cd", 
          border: "1px solid #ffeaa7", 
          borderRadius: "4px",
          color: "#856404"
        }}>
          <strong>Note:</strong> Please select at least one interaction type above to view ROIs.
        </div>
      )}

      <hr />
      {selectedGroups.length > 0 ? (
        <>
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
              onClick={() => handleSetView()}
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
        </>
      ) : (
        <div style={{ 
          textAlign: "center", 
          padding: "20px", 
          color: "#666",
          fontStyle: "italic"
        }}>
          Select interaction types above to view ROIs
        </div>
      )}
    </div>
  );
}

export default ROISelector;

import React, { useState, useEffect } from 'react';
import Heatmaps from './Heatmaps';
import InteractionHeatmaps from './InteractionHeatmaps';



function ROISelector({ onSetView, onHeatmapResults, onInteractionResults, onGroupSelection }) {
  const [rois, setRois] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGroups, setSelectedGroups] = useState([]);
  const [interactionGroups, setInteractionGroups] = useState([]);
  const [showCircles, setShowCircles] = useState(false);

  // Notify parent component when selectedGroups changes
  useEffect(() => {
    console.log('ROISelector: selectedGroups changed to:', selectedGroups);
    console.log('ROISelector: showCircles:', showCircles);
    
    // Notify parent component about group selection
    if (onGroupSelection) {
      onGroupSelection(selectedGroups);
    }
    
    // Config is now generated directly in frontend - no need to send to backend
  }, [selectedGroups, showCircles, onGroupSelection]);

  const computeCentroid = (allCoords) => {
    const flatCoords = allCoords.flat();
    const sum = flatCoords.reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
    return [sum[0] / flatCoords.length, sum[1] / flatCoords.length];
  };

  useEffect(() => {
    console.log('ROISelector: Starting to fetch ROI data...');
    
    // Define available interaction types and their corresponding files
    const interactionTypes = [
      'B-cell infiltration',
      'Inflammatory zone', 
      'T-cell entry site',
      'Oxidative stress niche'
    ];
    
    setInteractionGroups(interactionTypes);
    
    // Load ROI data for the first interaction type by default
    if (interactionTypes.length > 0) {
      loadROIData(interactionTypes[0]);
    }
  }, []);
  
  const loadROIData = (interactionType) => {
    console.log('ROISelector: Loading ROI data for:', interactionType);
    
    // Convert interaction type to filename format
    const filename = interactionType.replace(/\s+/g, '_').toLowerCase();
    const url = `http://localhost:5000/api/top_roi_scores_${filename}`;
    
    fetch(url)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log("ROISelector: Received ROI data for", interactionType, ":", data);
        
        if (!data.top_rois || !Array.isArray(data.top_rois)) {
          console.error("ROISelector: Invalid ROI data structure:", data);
          return;
        }

        console.log("ROISelector: Processing", data.top_rois.length, "ROI features");

        const extracted = data.top_rois.map((roi, index) => {
          return {
            id: `ROI_${roi.roi_id}`,
            x: roi.position.x,
            y: roi.position.y,
            z: roi.position.z,
            score: roi.scores.combined_score,
            interactions: [roi.interaction],
            raw: roi
          };
        });

        console.log("ROISelector: Extracted ROIs:", extracted);
        setRois(extracted);
      })
      .catch((err) => {
        console.error("ROISelector: Failed to load ROI data for", interactionType, ":", err);
        setRois([]);
      });
  };

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
      // Transform coordinates: X = x*8, Y = 5508-y*8
      const roiX = currentROI.x * 8;
      const roiY = 5508 - (currentROI.y * 8);
      
      // Find the interaction group for the current ROI
      const currentROIGroup = currentROI.interactions && currentROI.interactions.length > 0 
        ? currentROI.interactions[0] 
        : null;
      
      const viewConfig = {
        spatialTargetX: roiX,
        spatialTargetY: roiY,
        spatialZoom: -2.0,  // Moderate zoom to show ROI with range x±200, y±200
        refreshConfig: true,
        currentROIGroup: currentROIGroup // Pass the current ROI group
      };
      
      console.log('Setting view to ROI:', currentROI.id, 'at position:', roiX, roiY, 'with group:', currentROIGroup);
      console.log('Original coordinates from file:', currentROI.x, currentROI.y);
      console.log('Transformed coordinates:', roiX, roiY);
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
    let newSelectedGroups;
    
    if (selectedGroups.includes(group)) {
      // If the group is already selected, unselect it
      newSelectedGroups = selectedGroups.filter(g => g !== group);
    } else {
      // If selecting a new group, unselect all others and select only this one
      newSelectedGroups = [group];
      
      // Load ROI data for the selected interaction type
      loadROIData(group);
    }
    
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
      <div className="roi-selector-container">
        <h4 style={{ fontSize: '14px', marginBottom: '8px', fontWeight: '600', color: '#000' }}>ROI Navigator</h4>
        <p style={{ fontSize: '11px', marginBottom: '8px', color: '#000' }}>Please select an interaction type to view ROIs:</p>
        <p style={{ fontSize: '11px', marginBottom: '8px', color: '#000' }}>Available interaction types:</p>
        {interactionGroups.map(group => (
          <label key={group} className="checkbox-item" style={{ fontSize: '11px', marginBottom: '4px', color: '#000' }}>
            <input
              type="radio"
              name="interactionType"
              checked={selectedGroups.includes(group)}
              onChange={() => toggleGroup(group)}
              style={{ marginRight: '6px' }}
            />
            {group}
          </label>
        ))}
      </div>
    );
  }

  return (
    <div className="roi-selector-container">
      <h4 style={{ fontSize: '13px', marginBottom: '2px', fontWeight: '600', color: '#000' }}>ROI Navigator</h4>
      <p style={{ fontSize: '10px', marginBottom: '2px', color: '#000' }}>Select Interaction Type (only one at a time):</p>
      {interactionGroups.map(group => (
        <label key={group} className="checkbox-item" style={{ fontSize: '10px', marginBottom: '1px', color: '#000' }}>
          <input
            type="radio"
            name="interactionType"
            checked={selectedGroups.includes(group)}
            onChange={() => toggleGroup(group)}
            style={{ marginRight: '4px' }}
          />
          {group}
        </label>
      ))}
      
      {selectedGroups.length === 0 && (
        <div style={{ marginTop: "10px", padding: "8px", backgroundColor: "rgba(255, 243, 205, 0.8)", border: "1px solid rgba(255, 193, 7, 0.3)", borderRadius: "4px", color: "#856404" }}>
          <strong>Note:</strong> Please select one interaction type above to view ROIs.
        </div>
      )}

      <hr style={{ borderColor: "rgba(255, 255, 255, 0.2)" }} />
      {selectedGroups.length > 0 ? (
        <>
          <div className="text-center" style={{ marginBottom: "3px", display: "flex", justifyContent: "center", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "12px", fontWeight: "600", color: "#000" }}>ROI #{currentIndex + 1}</span>
            <span style={{ fontSize: "11px", fontWeight: "bold", color: "#000" }}>
              Score: {currentROI.score?.toFixed(3) || "0.000"}
            </span>
            <span style={{ fontSize: "9px", color: "#666" }}>
              {currentROI.interactions?.join(", ") || "None"}
            </span>
          </div>

          <div className="text-center" style={{ marginBottom: "3px" }}>
            <label className="checkbox-item" style={{ justifyContent: "center", fontSize: "10px", color: "#000" }}>
              <input
                type="checkbox"
                checked={showCircles}
                onChange={handleShowCirclesToggle}
                style={{ transform: "scale(1.0)", marginRight: "4px" }}
              />
              <span>Show interactive ROI circles</span>
            </label>
          </div>

          <div className="text-center" style={{ marginBottom: "1px" }}>
            <button 
              onClick={prev}
              className="btn"
              style={{ marginRight: "3px", padding: "3px 6px", fontSize: "10px" }}
            >
              ←
            </button>
            <button 
              onClick={() => handleSetView()}
              className="btn"
              style={{ marginRight: "3px", padding: "4px 10px", fontSize: "10px" }}
            >
              Set View
            </button>
            <button 
              onClick={next}
              className="btn"
              style={{ padding: "3px 6px", fontSize: "10px" }}
            >
              →
            </button>
          </div>

          {/* Analysis Buttons */}
          <div className="text-center" style={{ marginTop: "1px", display: "flex", justifyContent: "center", gap: "1px" }}>
            <Heatmaps 
              currentROI={currentROI}
              onHeatmapResults={onHeatmapResults}
              selectedInteractionType={selectedGroups[0]}
              selectedROIIndex={currentIndex}
            />
            <InteractionHeatmaps 
              currentROI={currentROI}
              onInteractionResults={onInteractionResults}
            />
          </div>
        </>
      ) : (
        <div className="text-center" style={{ padding: "2px", fontStyle: "italic", color: "#666" }}>
          Select one interaction type above to view ROIs
        </div>
      )}
    </div>
  );
}

export default ROISelector;

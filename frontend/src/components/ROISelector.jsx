import React, { useState, useEffect } from 'react';
const factor = 8;
function ROISelector({ onSetView }) {
  const [rois, setRois] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGroup, setSelectedGroup] = useState("");
  const [interactionGroups, setInteractionGroups] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/api/roi_shapes")
      .then((res) => res.json())
      .then((data) => {
        if (!data.features || !Array.isArray(data.features)) {
          console.error("Invalid ROI data structure:", data);
          return;
        }

        const extracted = data.features.map((feature, index) => {
          let coords = feature.geometry?.coordinates;

          // Handle both Polygon and MultiPolygon
          if (feature.geometry?.type === "Polygon") {
            coords = coords?.[0];
          } else if (feature.geometry?.type === "MultiPolygon") {
            coords = coords?.[0]?.[0]; // first polygon in multipolygon
          }

          if (!Array.isArray(coords) || coords.length === 0) return null;

          const centroid = coords.reduce(
            (acc, [x, y]) => [acc[0] + x, acc[1] + y],
            [0, 0]
          ).map(v => v / coords.length);

          return {
            id: feature.properties.name || `ROI_${index}`,
            x: centroid[0]*factor,
            y: centroid[1]*factor,
            score: feature.properties.score || 0,
            interactions: feature.properties.interactions || []
          };
        }).filter(Boolean);

        setRois(extracted);

        // Extract unique interaction labels
        const allInteractions = extracted.flatMap(r => r.interactions);
        const uniqueGroups = [...new Set(allInteractions)];
        setInteractionGroups(uniqueGroups);
        setSelectedGroup(uniqueGroups[0] || "");
      })
      .catch((err) => console.error("Failed to load ROI shapes:", err));
  }, []);

  const filteredRois = rois.filter(roi =>
    roi.interactions.includes(selectedGroup)
  );
  const currentROI = filteredRois[currentIndex] || {};

  const handleSet = () => {
    if (filteredRois.length > 0 && onSetView) {
      const roi = filteredRois[currentIndex];
      onSetView({
        spatialTargetX: roi.x,
        spatialTargetY: roi.y,
        spatialTargetZ: 0,
        spatialZoom: -1.1,
      });
    }
  };

  const next = () => {
    setCurrentIndex((i) => (i + 1) % filteredRois.length);
  };

  const prev = () => {
    setCurrentIndex((i) => (i - 1 + filteredRois.length) % filteredRois.length);
  };

  if (interactionGroups.length === 0) {
    return <p>Loading ROIs or no interactions found...</p>;
  }

  if (filteredRois.length === 0) {
    return <p>No ROIs found for selected group: {selectedGroup}</p>;
  }

  return (
    <div style={{ padding: "10px", border: "1px solid #ccc", marginBottom: "10px" }}>
      <h4>ROI Navigator</h4>

      <label htmlFor="interaction-select">Select Interaction Type:</label>
      <select
        id="interaction-select"
        value={selectedGroup}
        onChange={(e) => {
          setSelectedGroup(e.target.value);
          setCurrentIndex(0);
        }}
      >
        {interactionGroups.map(group => (
          <option key={group} value={group}>{group}</option>
        ))}
      </select>

      <p><strong>{currentROI.id}</strong></p>
      <p>
        X: {currentROI.x?.toFixed(0)} | Y: {currentROI.y?.toFixed(0)} | Score: {currentROI.score}
      </p>
      <p>
        Interactions: {currentROI.interactions.join(", ")}
      </p>

      <button onClick={prev}>Previous</button>
      <button onClick={next}>Next</button>
      <button onClick={handleSet}>Set View</button>
    </div>
  );
}

export default ROISelector;

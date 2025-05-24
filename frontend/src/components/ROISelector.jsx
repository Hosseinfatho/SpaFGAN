import React, { useState, useEffect } from 'react';

// Define scale factors (Zarr level 3 to full-res)
const factorX = 10908 / 1363;
const factorY = 5508 / 688;
const fullHeight = 5508; // needed for Y flip

function ROISelector({ onSetView }) {
  const [rois, setRois] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGroups, setSelectedGroups] = useState([]);
  const [interactionGroups, setInteractionGroups] = useState([]);
  const [manualX, setManualX] = useState("");
  const [manualY, setManualY] = useState("");

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
            y: fullHeight - (cy * factorY), // Flip Y for bottom-left origin
            score: feature.properties.score || 0,
            interactions: feature.properties.interactions || []
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

      <button onClick={prev}>Previous</button>
      <button onClick={next}>Next</button>
      <button onClick={handleSet}>Set View</button>
    </div>
  );
}

export default ROISelector;

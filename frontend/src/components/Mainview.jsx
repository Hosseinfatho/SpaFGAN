import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';

function Mainview() {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  // Define the specific view state for Mainview as a JS object
  const viewState = {
    spatialTargetZ: 0,
    spatialTargetT: 0,
    spatialZoom: -3.54,
    spatialTargetX: 5230,
    spatialTargetY: 2880,
    // spatialTargetZ repeated in input, using 0
    spatialRenderingMode: "3D",
    // Represent layers and channels as plain arrays/objects
    // The backend will wrap these with CL()
    imageLayer: [
      {
        spatialTargetResolution: 5,
        spatialLayerOpacity: 1.0,
        spatialLayerVisible: true,
        photometricInterpretation: "BlackIsZero",
        imageChannel: [
          {
            spatialTargetC: 0,
            spatialChannelColor: [0, 0, 255],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            // spatialChannelWindow: [?,?] // Window omitted as not in input
          }
        ]
      }
    ]
  };

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/generate_config', { // New endpoint
      method: 'POST', // Use POST
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(viewState) // Send the view state object
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        return response.json()
      })
      .then(data => {
        console.log("Mainview config generated successfully:", data)
        setConfig(data) // Set the complete config received from backend
      })
      .catch(err => {
        console.error("Error generating Mainview config:", err)
        setError(err.message)
      })
  }, []); // Fetch only once on mount

  if (error) {
    return <p style={{ color: 'red', padding: '10px' }}>Error generating Mainview: {error}</p>;
  }
  if (!config) {
    return <p style={{ padding: '10px' }}>Generating Mainview config...</p>;
  }
  return (
    <Vitessce
      config={config}
      theme="light"
      height={null} // Let CSS handle height
      width={null} // Let CSS handle width
    />
  );
}

export default Mainview; 
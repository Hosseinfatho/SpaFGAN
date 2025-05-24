// Mainview.jsx
import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';
import ROISelector from './ROISelector';

function Mainview() {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);
  const [viewState, setViewState] = useState({
    spatialTargetZ: 0,
    spatialTargetT: 0,
    spatialZoom: -3.0,
    spatialTargetX: 5500,
    spatialTargetY: 2700,
    spatialRenderingMode: "3D",
    imageLayer: [
      {
        spatialTargetResolution: 3,
        spatialLayerOpacity: 1.0,
        spatialLayerVisible: true,
        photometricInterpretation: "BlackIsZero",
        imageChannel: [
          {
            spatialTargetC: 19,
            spatialChannelColor: [0, 255, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [300, 20000]
          },
          {
            spatialTargetC: 27,
            spatialChannelColor: [255, 255, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1000, 7000]
          },
          {
            spatialTargetC: 37,
            spatialChannelColor: [255, 0, 255],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [700, 6000]
          },
          {
            spatialTargetC: 25,
            spatialChannelColor: [0, 255, 255],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1638, 10000]
          },
          {
            spatialTargetC: 42,
            spatialChannelColor: [65, 51, 97],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [370, 1432]
          },
          {
            spatialTargetC: 59,
            spatialChannelColor: [255, 0, 0],
            spatialChannelVisible: true,
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: [1638, 7000]
          }
        ]
      }
    ]
  });

  const fetchConfig = (stateData) => {
    fetch('http://127.0.0.1:5000/api/generate_config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(stateData)
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("Mainview config generated successfully:", data);
        setConfig(data);
      })
      .catch(err => {
        console.error("Error generating Mainview config:", err);
        setError(err.message);
      });
  };

  useEffect(() => {
    fetchConfig(viewState);
  }, [viewState]);

  const handleSetView = (roiView) => {
    setViewState(prev => ({
      ...prev,
      ...roiView
    }));
  };

  if (error) {
    return <p style={{ color: 'red', padding: '10px' }}>Error generating Mainview: {error}</p>;
  }
  if (!config) {
    return <p style={{ padding: '10px' }}>Generating Mainview config...</p>;
  }

  return (
    <div>
      <ROISelector onSetView={handleSetView} />
      <Vitessce
        config={config}
        theme="light"
        height={null}
        width={null}
      />
    </div>
  );
}

export default Mainview;

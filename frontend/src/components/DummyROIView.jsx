import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';

const DummyROIView = () => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Create a simple config with only dummy ROI rectangles
    const simpleConfig = {
      "version": "1.0.16",
      "name": "Dummy ROI Rectangles Only",
      "description": "Only showing dummy ROI rectangles",
      "datasets": [
        {
          "uid": "dummy_dataset",
          "name": "Dummy Dataset",
          "files": [
            {
              "fileType": "obsSegmentations.json",
              "url": "http://localhost:5001/api/dummy_roi_rectangles",
              "coordinationValues": {
                "obsType": "ROI"
              }
            }
          ]
        }
      ],
      "coordinationSpace": {
        "dataset": {
          "A": "dummy_dataset"
        },
        "obsType": {
          "A": "ROI"
        }
      },
      "layout": [
        {
          "component": "spatialBeta",
          "coordinationScopes": {
            "dataset": "A"
          },
          "x": 0,
          "y": 0,
          "w": 12,
          "h": 12
        }
      ],
      "initStrategy": "auto"
    };

    setConfig(simpleConfig);
  }, []);

  if (error) {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h2>Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
      <div style={{ 
        position: 'absolute', 
        top: '10px', 
        left: '10px', 
        zIndex: 1000,
        background: 'rgba(255, 255, 255, 0.9)',
        padding: '10px',
        borderRadius: '5px',
        fontSize: '12px'
      }}>
        <h3>Dummy ROI Rectangles Only</h3>
        <p>Status: {config ? '✅ Loaded' : '⏳ Loading...'}</p>
        <p>5 colored rectangles should be visible</p>
      </div>

      {config ? (
        <Vitessce
          config={config}
          theme="light"
          height={null}
          width={null}
        />
      ) : (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100%',
          fontSize: '18px'
        }}>
          Loading dummy ROI rectangles...
        </div>
      )}
    </div>
  );
};

export default DummyROIView; 
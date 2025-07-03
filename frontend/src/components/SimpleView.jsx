import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';

const SimpleView = () => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Create a Vitessce config for the dummy OME-Zarr file
    const dummyConfig = {
      "version": "1.0.16",
      "name": "Dummy OME-Zarr with 5 Channels and Random Rectangles",
      "description": "Showing dummy OME-Zarr data with 5 channels, each containing 5 random rectangles",
      "datasets": [
        {
          "uid": "dummy_ome_zarr",
          "name": "Dummy OME-Zarr Dataset",
          "files": [
            {
              "fileType": "image.ome-zarr",
              "url": "http://localhost:5001/api/dummy_ome_zarr"
            }
          ]
        }
      ],
      "coordinationSpace": {
        "dataset": {
          "A": "dummy_ome_zarr"
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
          "w": 8,
          "h": 8
        },
        {
          "component": "layerControllerBeta",
          "coordinationScopes": {
            "dataset": "A"
          },
          "x": 8,
          "y": 0,
          "w": 4,
          "h": 8
        }
      ],
      "initStrategy": "auto"
    };

    setConfig(dummyConfig);
  }, []);

  if (error) {
    return (
      <div style={{ 
        padding: '20px', 
        color: 'red',
        textAlign: 'center'
      }}>
        <h2>Error</h2>
        <p>{error}</p>
        <p>Make sure the server is running on port 5001</p>
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
        <h3>Dummy OME-Zarr with Vitessce</h3>
        <p>Status: {config ? '✅ Loaded' : '⏳ Loading...'}</p>
        <p>5 channels, each with 5 random rectangles</p>
        <p>Use layer controller to adjust channels</p>
        <button 
          onClick={() => {
            // Check if dummy OME-Zarr exists
            fetch('http://localhost:5001/api/dummy_ome_zarr')
              .then(response => response.json())
              .then(data => {
                console.log('Dummy OME-Zarr status:', data);
                alert(`Dummy OME-Zarr status: ${JSON.stringify(data, null, 2)}`);
              })
              .catch(err => {
                console.error('Error checking dummy OME-Zarr:', err);
                alert(`Error: ${err.message}`);
              });
          }}
          style={{ marginTop: '10px' }}
        >
          Check Dummy OME-Zarr Status
        </button>
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
          Loading Vitessce configuration...
        </div>
      )}
    </div>
  );
};

export default SimpleView; 
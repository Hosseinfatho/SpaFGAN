import React, { useState, useEffect } from 'react';
import { Vitessce } from 'vitessce';

const ROITestView = () => {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  const fetchTestConfig = () => {
    fetch('http://localhost:5001/api/generate_simple_config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        spatialRenderingMode: "3D",
        spatialTargetZ: 0,
        spatialTargetT: 0,
        spatialZoom: -2.5,
        spatialTargetX: 5454,
        spatialTargetY: 2754
      })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log("ROI test config generated:", data);
        setConfig(data);
      })
      .catch(err => {
        console.error("Error generating ROI test config:", err);
        setError(err.message);
      });
  };

  useEffect(() => {
    fetchTestConfig();
  }, []);

  if (error) {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={fetchTestConfig}>Retry</button>
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
        <h3>ROI Test View (Rectangles Only)</h3>
        <p>Status: {config ? '✅ Loaded' : '⏳ Loading...'}</p>
        <button onClick={fetchTestConfig}>Refresh Config</button>
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
          Loading ROI test configuration...
        </div>
      )}
    </div>
  );
};

export default ROITestView; 
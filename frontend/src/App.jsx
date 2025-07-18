import React, { useState } from 'react';
import './App.css';
import MainView from './components/Original';
import ROISelector from './components/ROISelector';

function App() {
  const [currentConfig, setCurrentConfig] = useState(null);

  const handleROISelection = (roiView) => {
    // ROISelector now handles config generation internally
    // We just need to pass the view settings to MainView
    console.log('App: Received ROI selection:', roiView);
  };

  return (
    <div className="app">
      <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <div className="view-area">
          <MainView onSetView={handleROISelection} />
        </div>
        <div className="controls-panel">
          <ROISelector onSetView={handleROISelection} />
        </div>
      </div>
    </div>
  );
}

export default App;

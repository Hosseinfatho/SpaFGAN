import React from 'react';
import './App.css';
import Mainview from './components/Mainview';

function App() {
  return (
    <div className="app">
      <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <div className="view-area">
          <Mainview />
        </div>
      </div>
    </div>
  );
}

export default App;

import React from 'react';
import './App.css';
import MainView from './components/Original';

function App() {
  return (
    <div className="app">
      <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <div className="view-area">
          <MainView />
        </div>
      </div>
    </div>
  );
}

export default App;

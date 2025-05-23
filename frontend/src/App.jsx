import React from 'react';
import './App.css';
import Mainview from './components/Mainview';
import ROIView from './components/ROIView';
import ROIBestView from './components/ROIBestView';
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";

function App() {
  return (
    <div className="app">
      <div style={{ width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <PanelGroup direction="horizontal" style={{ width: '100%', height: '100%' }}>
          <Panel defaultSize={25} minSize={20} style={{ width: '25%' }}>
            <PanelGroup direction="vertical">
              <Panel defaultSize={50} minSize={30}>
                <div className="view-area">
                  <Mainview />
                </div>
              </Panel>
              <PanelResizeHandle style={{ height: '4px', background: '#ccc' }} />
              <Panel defaultSize={50} minSize={30}>
                <div className="view-area">
                  <ROIView />
                </div>
              </Panel>
            </PanelGroup>
          </Panel>
          <PanelResizeHandle style={{ width: '4px', background: '#ccc' }} />
          <Panel defaultSize={75} minSize={25} style={{ width: '75%' }}>
            <div className="view-area">
              <ROIBestView />
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions }) => {
  const [circles, setCircles] = useState([]);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (showCircles && selectedInteractions && selectedInteractions.length > 0) {
      setLoading(true);
      
      console.log('Fetching circles for interactions:', selectedInteractions);
      
      // Fetch filtered ROIs from backend
      fetch('http://localhost:5000/api/filtered_rois', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          interactions: selectedInteractions
        })
      })
      .then(response => response.json())
      .then(data => {
        console.log('Received circles data:', data);
        if (data.success && data.circles) {
          setCircles(data.circles.map(circle => ({
            ...circle,
            selected: selectedCircle === circle.id
          })));
        } else {
          console.error('Failed to fetch filtered ROIs:', data.error);
          setCircles([]);
        }
      })
      .catch(error => {
        console.error('Error fetching filtered ROIs:', error);
        setCircles([]);
      })
      .finally(() => {
        setLoading(false);
      });
    } else {
      setCircles([]);
    }
  }, [showCircles, selectedInteractions, selectedCircle]);

  const getCircleColor = (index) => {
    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'];
    return colors[index % colors.length];
  };

  const handleCircleClick = (circleId) => {
    console.log('Circle clicked:', circleId);
    if (onCircleClick) {
      onCircleClick(circleId);
    }
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(10, prev * delta)));
  };

  const handleMouseDown = (e) => {
    if (e.button === 0) { // Left click
      const startX = e.clientX;
      const startY = e.clientY;
      const startPos = { ...position };

      const handleMouseMove = (e) => {
        const deltaX = e.clientX - startX;
        const deltaY = e.clientY - startY;
        setPosition({
          x: startPos.x + deltaX,
          y: startPos.y + deltaY
        });
      };

      const handleMouseUp = () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
  };

  if (!showCircles) {
    return null;
  }

  if (loading) {
    return (
      <div className="interactive-circles-container">
        <div className="loading-overlay">
          <div className="loading-spinner">Loading ROIs...</div>
        </div>
      </div>
    );
  }

  if (circles.length === 0) {
    return (
      <div className="interactive-circles-container">
        <div className="circle-info-panel">
          <div className="info-title">No ROIs Found</div>
          <div className="info-content">
            <p>No ROIs found for selected interactions.</p>
            <p>Try selecting different interaction types.</p>
            <p>Selected: {selectedInteractions?.join(', ') || 'None'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="interactive-circles-container"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
    >
      <div 
        className="circles-viewport"
        style={{
          transform: `translate(${position.x}px, ${position.y}px) scale(${zoom}) rotate(${rotation}deg)`
        }}
      >
        {circles.map((circle) => (
          <div
            key={circle.id}
            className={`interactive-circle ${circle.selected ? 'selected' : ''}`}
            style={{
              left: `${circle.x}px`,
              top: `${circle.y}px`,
              backgroundColor: circle.color,
              borderColor: circle.selected ? '#ffffff' : circle.color,
              transform: `translate(-50%, -50%) scale(${1/zoom})`
            }}
            onClick={() => handleCircleClick(circle.id)}
            title={`ROI: ${circle.interactions.join(', ')} - Score: ${circle.score.toFixed(3)}`}
          >
            <div className="circle-label">
              {circle.id.split('_')[1]}
            </div>
          </div>
        ))}
      </div>
      
      {/* Controls */}
      <div className="circle-controls">
        <button 
          onClick={() => setZoom(prev => Math.min(10, prev * 1.2))}
          className="control-btn"
          title="Zoom In"
        >
          +
        </button>
        <button 
          onClick={() => setZoom(prev => Math.max(0.1, prev * 0.8))}
          className="control-btn"
          title="Zoom Out"
        >
          -
        </button>
        <button 
          onClick={() => setRotation(prev => prev + 90)}
          className="control-btn"
          title="Rotate"
        >
          ↻
        </button>
        <button 
          onClick={() => {
            setPosition({ x: 0, y: 0 });
            setZoom(1);
            setRotation(0);
          }}
          className="control-btn"
          title="Reset View"
        >
          Reset
        </button>
      </div>
      
      {/* Debug Panel */}
      <div className="debug-panel">
        <div className="debug-title">Debug Info</div>
        <div className="debug-content">
          <p>Show Circles: {showCircles ? 'Yes' : 'No'}</p>
          <p>Selected Interactions: {selectedInteractions?.join(', ') || 'None'}</p>
          <p>Circles Count: {circles.length}</p>
          <p>Loading: {loading ? 'Yes' : 'No'}</p>
          <p>Zoom: {zoom.toFixed(2)}</p>
          <p>Position: ({position.x.toFixed(0)}, {position.y.toFixed(0)})</p>
          {circles.length > 0 && (
            <div>
              <p>First Circle:</p>
              <p>  ID: {circles[0].id}</p>
              <p>  Position: ({circles[0].x}, {circles[0].y})</p>
              <p>  Color: {circles[0].color}</p>
              <p>  Score: {circles[0].score.toFixed(3)}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Info Panel */}
      <div className="circle-info-panel">
        <div className="info-title">Interactive ROI Circles</div>
        <div className="info-content">
          <p>• Click circles to zoom to ROI</p>
          <p>• Use mouse wheel to zoom</p>
          <p>• Drag to pan view</p>
          <p>• Use controls to adjust view</p>
          <p>• Found {circles.length} ROIs</p>
          <p>• Type: {selectedInteractions?.join(', ')}</p>
        </div>
      </div>
    </div>
  );
};

export default InteractiveCircles; 
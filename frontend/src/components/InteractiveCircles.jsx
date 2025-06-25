import React, { useState, useEffect } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions }) => {
  const [circles, setCircles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [vitessceViewport, setVitessceViewport] = useState({
    zoom: 1,
    x: 0,
    y: 0,
    width: 1920,
    height: 1080
  });

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
          // Scale coordinates to fit the viewport
          const imageWidth = data.image_dimensions?.width || 10908;
          const imageHeight = data.image_dimensions?.height || 5508;
          
          // Get viewport dimensions (assuming Vitessce container size)
          const viewportWidth = window.innerWidth * 0.8; // Approximate Vitessce width
          const viewportHeight = window.innerHeight * 0.8; // Approximate Vitessce height
          
          const scaleX = viewportWidth / imageWidth;
          const scaleY = viewportHeight / imageHeight;
          const scale = Math.min(scaleX, scaleY);
          
          const scaledCircles = data.circles.map(circle => ({
            ...circle,
            selected: selectedCircle === circle.id,
            // Scale coordinates to viewport
            x: circle.x * scale,
            y: circle.y * scale,
            original_x: circle.x,
            original_y: circle.y
          }));
          
          console.log('Scaled circles:', scaledCircles);
          setCircles(scaledCircles);
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

  // Listen for Vitessce viewport changes
  useEffect(() => {
    const handleVitessceViewportChange = (event) => {
      // This will be called when Vitessce viewport changes
      // We'll need to get this from Vitessce's viewport state
      console.log('Vitessce viewport changed:', event);
    };

    // Add event listener for Vitessce viewport changes
    window.addEventListener('vitessce-viewport-change', handleVitessceViewportChange);
    
    return () => {
      window.removeEventListener('vitessce-viewport-change', handleVitessceViewportChange);
    };
  }, []);

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

  if (!showCircles) {
    return null;
  }

  if (loading) {
    return (
      <div className="interactive-circles-overlay">
        <div className="loading-overlay">
          <div className="loading-spinner">Loading ROIs...</div>
        </div>
      </div>
    );
  }

  if (circles.length === 0) {
    return (
      <div className="interactive-circles-overlay">
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
    <div className="interactive-circles-overlay">
      {/* ROI Circles - positioned absolutely over Vitessce */}
      {circles.map((circle) => (
        <div
          key={circle.id}
          className={`vitessce-circle ${circle.selected ? 'selected' : ''}`}
          style={{
            position: 'absolute',
            left: `${circle.x}px`,
            top: `${circle.y}px`,
            width: '20px',
            height: '20px',
            backgroundColor: circle.color,
            border: `2px solid ${circle.selected ? '#ffffff' : circle.color}`,
            borderRadius: '50%',
            cursor: 'pointer',
            zIndex: 1000,
            transform: 'translate(-50%, -50%)',
            boxShadow: circle.selected ? '0 0 10px rgba(255,255,255,0.8)' : '0 0 5px rgba(0,0,0,0.5)',
            transition: 'all 0.2s ease'
          }}
          onClick={() => handleCircleClick(circle.id)}
          title={`ROI: ${circle.interactions.join(', ')} - Score: ${circle.score.toFixed(3)}`}
        >
          <div 
            className="circle-label"
            style={{
              position: 'absolute',
              top: '-25px',
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: 'rgba(0,0,0,0.8)',
              color: 'white',
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '10px',
              whiteSpace: 'nowrap',
              pointerEvents: 'none'
            }}
          >
            {circle.id.split('_')[1]}
          </div>
        </div>
      ))}
      
      {/* Debug Panel */}
      <div className="debug-panel">
        <div className="debug-title">ROI Overlay Debug</div>
        <div className="debug-content">
          <p>Show Circles: {showCircles ? 'Yes' : 'No'}</p>
          <p>Selected Interactions: {selectedInteractions?.join(', ') || 'None'}</p>
          <p>Circles Count: {circles.length}</p>
          <p>Loading: {loading ? 'Yes' : 'No'}</p>
          {circles.length > 0 && (
            <div>
              <p>First Circle:</p>
              <p>  ID: {circles[0].id}</p>
              <p>  Position: ({circles[0].x.toFixed(0)}, {circles[0].y.toFixed(0)})</p>
              <p>  Original: ({circles[0].original_x?.toFixed(0)}, {circles[0].original_y?.toFixed(0)})</p>
              <p>  Color: {circles[0].color}</p>
              <p>  Score: {circles[0].score.toFixed(3)}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Info Panel */}
      <div className="circle-info-panel">
        <div className="info-title">ROI Overlay</div>
        <div className="info-content">
          <p>• Click circles to select ROI</p>
          <p>• Circles overlay on image</p>
          <p>• Found {circles.length} ROIs</p>
          <p>• Type: {selectedInteractions?.join(', ')}</p>
        </div>
      </div>
    </div>
  );
};

export default InteractiveCircles; 
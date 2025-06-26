import React, { useState, useEffect, useRef } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions, viewState }) => {
  const [circles, setCircles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [vitessceViewport, setVitessceViewport] = useState({
    zoom: 1,
    x: 0,
    y: 0,
    width: 1920,
    height: 1080
  });
  const abortControllerRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    console.log('InteractiveCircles useEffect triggered:', {
      showCircles,
      selectedInteractions,
      selectedCircle,
      viewState: viewState ? 'present' : 'missing'
    });
    
    // Clear previous debounce
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    
    // Debounce the fetch to prevent rapid requests
    debounceRef.current = setTimeout(() => {
      // Abort previous request if it exists
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      if (showCircles && selectedInteractions && selectedInteractions.length > 0) {
        setLoading(true);
        
        console.log('Fetching circles for interactions:', selectedInteractions);
        
        // Create new AbortController for this request
        abortControllerRef.current = new AbortController();
        
        // Fetch filtered ROIs from backend
        fetch('http://localhost:5000/api/filtered_rois', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            interactions: selectedInteractions
          }),
          signal: abortControllerRef.current.signal
        })
        .then(response => {
          console.log('Response status:', response.status);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log('Received circles data:', data);
          if (data.success && data.circles) {
            // Get image dimensions
            const imageWidth = data.image_dimensions?.width || 10908;
            const imageHeight = data.image_dimensions?.height || 5508;
            
            // Get viewport dimensions (assuming Vitessce container size)
            const viewportWidth = window.innerWidth * 0.5; // More accurate Vitessce width
            const viewportHeight = window.innerHeight * 0.5; // More accurate Vitessce height
            
            const scaledCircles = data.circles.map(circle => {
              // Use fixed positioning instead of percentage to avoid flickering
              const scaleX = viewportWidth / imageWidth;
              const scaleY = viewportHeight / imageHeight;
              const scale = Math.min(scaleX, scaleY);
              
              return {
                ...circle,
                selected: selectedCircle === circle.id,
                x: circle.x * scale,
                y: circle.y * scale,
                original_x: circle.x,
                original_y: circle.y,
                color: circle.color || getCircleColor(parseInt(circle.id.split('_')[1]) || 0)
              };
            });
            
            console.log('Scaled circles with viewport coordinates:', scaledCircles);
            setCircles(scaledCircles);
          } else {
            console.error('Failed to fetch filtered ROIs:', data.error);
            setCircles([]);
          }
        })
        .catch(error => {
          if (error.name === 'AbortError') {
            console.log('Request was aborted');
          } else {
            console.error('Error fetching filtered ROIs:', error);
            setCircles([]);
          }
        })
        .finally(() => {
          setLoading(false);
          abortControllerRef.current = null;
        });
      } else {
        console.log('Not fetching circles - showCircles:', showCircles, 'selectedInteractions:', selectedInteractions);
        setCircles([]);
        setLoading(false);
      }
    }, 300); // 300ms debounce
    
    // Cleanup function
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [showCircles, selectedInteractions]); // Removed selectedCircle and viewState from dependencies

  // Separate useEffect to handle selectedCircle updates
  useEffect(() => {
    if (circles.length > 0) {
      setCircles(prevCircles => 
        prevCircles.map(circle => ({
          ...circle,
          selected: selectedCircle === circle.id
        }))
      );
    }
  }, [selectedCircle, circles.length]);

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

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

  console.log('InteractiveCircles render:', {
    showCircles,
    loading,
    circlesCount: circles.length,
    selectedInteractions
  });

  if (!showCircles) {
    return null;
  }

  if (loading) {
    return (
      <div className="interactive-circles-overlay" style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 9999,
        overflow: 'visible'
      }}>
        <div className="loading-overlay" style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
          zIndex: 10001,
          pointerEvents: 'auto'
        }}>
          <div className="loading-spinner" style={{
            textAlign: 'center',
            fontSize: '14px'
          }}>
            Loading ROIs...
          </div>
        </div>
      </div>
    );
  }

  if (circles.length === 0 && !loading && showCircles) {
    return (
      <div className="interactive-circles-overlay" style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 9999,
        overflow: 'visible'
      }}>
        <div className="circle-info-panel" style={{ pointerEvents: 'auto' }}>
          <div className="info-title">ROI Overlay</div>
          <div className="info-content">
            <p>No ROIs found for selected interactions.</p>
            <p>Try selecting different interaction types.</p>
            <p>Selected: {selectedInteractions?.join(', ') || 'None'}</p>
            <p>Show Circles: {showCircles ? 'Yes' : 'No'}</p>
            <p>Loading: {loading ? 'Yes' : 'No'}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="interactive-circles-overlay" style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: 9999,
      overflow: 'visible'
    }}>
      {/* Circle Container */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}>
        {/* ROI Circles - positioned absolutely over Vitessce */}
        {circles.map((circle) => (
          <div
            key={circle.id}
            className={`vitessce-circle ${circle.selected ? 'selected' : ''}`}
            style={{
              position: 'absolute',
              left: `${circle.x}px`,
              top: `${circle.y}px`,
              width: '30px',
              height: '30px',
              backgroundColor: circle.color || '#ff0000',
              border: `3px solid ${circle.selected ? '#ffffff' : circle.color || '#ff0000'}`,
              borderRadius: '50%',
              cursor: 'pointer',
              zIndex: 10000,
              transform: 'translate(-50%, -50%)',
              boxShadow: circle.selected ? '0 0 15px rgba(255,255,255,0.9)' : '0 0 8px rgba(0,0,0,0.7)',
              transition: 'all 0.3s ease',
              pointerEvents: 'auto',
              opacity: 1
            }}
            onClick={() => handleCircleClick(circle.id)}
            title={`ROI: ${circle.interactions.join(', ')} - Score: ${circle.score.toFixed(3)} - Position: (${circle.original_x?.toFixed(0)}, ${circle.original_y?.toFixed(0)})`}
          >
            <div 
              className="circle-label"
              style={{
                position: 'absolute',
                top: '-30px',
                left: '50%',
                transform: 'translateX(-50%)',
                backgroundColor: 'rgba(0,0,0,0.9)',
                color: 'white',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                whiteSpace: 'nowrap',
                pointerEvents: 'none',
                zIndex: 10001,
                fontWeight: 'bold'
              }}
            >
              {circle.id.split('_')[1]}
            </div>
          </div>
        ))}
      </div>
      
      {/* Debug Panel */}
      <div className="debug-panel" style={{ pointerEvents: 'auto' }}>
        <div className="debug-title">ROI Overlay Debug</div>
        <div className="debug-content">
          <p>Show Circles: {showCircles ? 'Yes' : 'No'}</p>
          <p>Selected Interactions: {selectedInteractions?.join(', ') || 'None'}</p>
          <p>Circles Count: {circles.length}</p>
          <p>Loading: {loading ? 'Yes' : 'No'}</p>
          <p>Zoom: {viewState?.spatialZoom || 'N/A'}</p>
          <p>Target: ({viewState?.spatialTargetX || 'N/A'}, {viewState?.spatialTargetY || 'N/A'})</p>
          <p>Viewport: {window.innerWidth} x {window.innerHeight}</p>
          <p>Overlay Z-Index: 9999</p>
          <p>Circle Z-Index: 10000</p>
          {circles.length > 0 && (
            <div>
              <p>First Circle:</p>
              <p>  ID: {circles[0].id}</p>
              <p>  Position: ({circles[0].x.toFixed(0)}px, {circles[0].y.toFixed(0)}px)</p>
              <p>  Original: ({circles[0].original_x?.toFixed(0)}, {circles[0].original_y?.toFixed(0)})</p>
              <p>  Color: {circles[0].color}</p>
              <p>  Score: {circles[0].score.toFixed(3)}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Info Panel */}
      <div className="circle-info-panel" style={{ pointerEvents: 'auto' }}>
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
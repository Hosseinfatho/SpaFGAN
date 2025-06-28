import React, { useState, useEffect, useRef } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions, viewState }) => {
  const [circles, setCircles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 10908, height: 5508 });
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);
  const containerRef = useRef(null);
  const lastFetchRef = useRef('');

  console.log('InteractiveCircles render:', {
    showCircles,
    selectedInteractions,
    circlesCount: circles.length,
    loading,
    error
  });

  // Function to convert data coordinates to screen coordinates
  const dataToScreenCoords = (dataX, dataY) => {
    // Get Vitessce container
    const vitessceContainer = document.querySelector('.fullscreen-vitessce');
    if (!vitessceContainer) {
      console.log('Vitessce container not found');
      return { x: 100, y: 100 }; // Fallback position
    }
    
    const containerRect = vitessceContainer.getBoundingClientRect();
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;
    
    console.log('Container dimensions:', { containerWidth, containerHeight });
    
    // Get current zoom level
    const zoomLevel = Math.pow(2, -(viewState?.spatialZoom || -2.5));
    
    // Calculate scale factors
    const scaleX = containerWidth / imageDimensions.width;
    const scaleY = containerHeight / imageDimensions.height;
    const scale = Math.min(scaleX, scaleY);
    
    // Apply scale and zoom
    const scaledX = dataX * scale * zoomLevel;
    const scaledY = dataY * scale * zoomLevel;
    
    // Center the image in the container
    const offsetX = (containerWidth - (imageDimensions.width * scale * zoomLevel)) / 2;
    const offsetY = (containerHeight - (imageDimensions.height * scale * zoomLevel)) / 2;
    
    // Apply pan offset (if any)
    const panX = viewState?.spatialTargetX || 0;
    const panY = viewState?.spatialTargetY || 0;
    
    const finalX = scaledX + offsetX + (panX * scale * zoomLevel);
    const finalY = containerHeight - scaledY - offsetY - (panY * scale * zoomLevel);
    
    console.log('Coordinate conversion:', {
      dataX, dataY, finalX, finalY,
      zoomLevel, scale, offsetX, offsetY, panX, panY
    });
    
    return { x: finalX, y: finalY };
  };

  // Function to update circle positions
  const updateCirclePositions = (circlesData) => {
    return circlesData.map(circle => {
      const screenCoords = dataToScreenCoords(circle.original_x, circle.original_y);
      return {
        ...circle,
        x: screenCoords.x,
        y: screenCoords.y,
        selected: selectedCircle === circle.id
      };
    });
  };

  // Create test circles if no API data is available
  const createTestCircles = () => {
    const testCircles = [
      {
        id: "roi_0",
        original_x: 3000,
        original_y: 2000,
        score: 0.85,
        interactions: ["B-cell infiltration"],
        color: "#ff0000",
        x: 200, // Fixed position for testing
        y: 200
      },
      {
        id: "roi_1", 
        original_x: 5000,
        original_y: 3000,
        score: 0.72,
        interactions: ["Inflammatory zone"],
        color: "#00ff00",
        x: 400, // Fixed position for testing
        y: 200
      },
      {
        id: "roi_2",
        original_x: 7000,
        original_y: 2500,
        score: 0.68,
        interactions: ["Oxidative stress niche"],
        color: "#0000ff",
        x: 600, // Fixed position for testing
        y: 200
      }
    ];
    console.log('Created test circles with fixed positions:', testCircles);
    return testCircles;
  };

  // Main effect for fetching ROIs
  useEffect(() => {
    console.log('InteractiveCircles useEffect triggered:', {
      showCircles,
      selectedInteractions,
      selectedInteractionsLength: selectedInteractions?.length,
      loading,
      circlesCount: circles.length
    });

    // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
    if (showCircles) {
      // For now, use test circles to avoid API issues
      console.log('Using test circles for debugging');
      const testCircles = createTestCircles();
      setCircles(testCircles);
          setLoading(false);
      setError(null);
      } else {
      console.log('showCircles is false, clearing circles');
        setCircles([]);
        setLoading(false);
      setError(null);
      }
    
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [showCircles]); // Only depend on showCircles

  // Separate effect for updating circle positions when viewState changes
  useEffect(() => {
    if (circles.length > 0) {
      console.log('Updating circle positions due to viewState change');
      setCircles(updateCirclePositions(circles));
    }
  }, [viewState?.spatialZoom, viewState?.spatialTargetX, viewState?.spatialTargetY]);

  // Separate effect for updating selected state
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
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
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

  console.log('Before render check:', { showCircles, circles: circles.length, loading, error });

  if (!showCircles) {
    console.log('showCircles is false, returning null');
    return null;
  }

  if (loading) {
    console.log('Loading state, showing loading overlay');
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
            <br />
            <small>Selected: {selectedInteractions?.join(', ') || 'All interactions'}</small>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    console.log('Error state, showing error overlay');
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
        <div className="error-overlay" style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'rgba(255, 0, 0, 0.8)',
          color: 'white',
          padding: '20px',
          borderRadius: '8px',
          zIndex: 10001,
          pointerEvents: 'auto',
          maxWidth: '400px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '14px', marginBottom: '10px' }}>
            Error loading ROIs
          </div>
          <div style={{ fontSize: '12px', marginBottom: '10px' }}>
            {error}
          </div>
          <div style={{ fontSize: '11px' }}>
            Selected interactions: {selectedInteractions?.join(', ') || 'All interactions'}
          </div>
        </div>
      </div>
    );
  }

  console.log('Rendering circles:', circles.length, 'circles');

  return (
    <div 
      ref={containerRef}
      className="interactive-circles-overlay" 
      style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: 9999,
        overflow: 'visible',
        border: '2px solid red' // Debug border
      }}
    >
      {/* Test circle - always visible */}
      <div
        style={{
          position: 'absolute',
          left: '100px',
          top: '100px',
          width: '50px',
          height: '50px',
          backgroundColor: '#ff0000',
          border: '3px solid #ffffff',
          borderRadius: '50%',
          cursor: 'pointer',
          zIndex: 10000,
          transform: 'translate(-50%, -50%)',
          boxShadow: '0 0 15px rgba(255,255,255,0.9)',
          pointerEvents: 'auto'
        }}
        onClick={() => console.log('Test circle clicked!')}
        title="Test Circle"
      >
        <div style={{
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
        }}>
          TEST
        </div>
      </div>

      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        border: '2px solid blue' // Debug border
      }}>
        {circles.map((circle, index) => {
          console.log(`Rendering circle ${index}:`, circle);
          return (
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
          );
        })}
      </div>
    </div>
  );
};

export default InteractiveCircles; 
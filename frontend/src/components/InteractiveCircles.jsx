import React, { useState, useEffect, useRef } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions, viewState }) => {
  const [circles, setCircles] = useState([]);
  const [loading, setLoading] = useState(false);
  const abortControllerRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    
    debounceRef.current = setTimeout(() => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      if (showCircles && selectedInteractions && selectedInteractions.length > 0) {
        setLoading(true);
        
        abortControllerRef.current = new AbortController();
        
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
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          if (data.success && data.circles) {
            const imageWidth = data.image_dimensions?.width || 10908;
            const imageHeight = data.image_dimensions?.height || 5508;
            
            const viewportWidth = window.innerWidth * 0.5;
            const viewportHeight = window.innerHeight * 0.5;
            
            const scaledCircles = data.circles.map(circle => {
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
            
            setCircles(scaledCircles);
          } else {
            setCircles([]);
          }
        })
        .catch(error => {
          if (error.name !== 'AbortError') {
            console.error('Error fetching filtered ROIs:', error);
            setCircles([]);
          }
        })
        .finally(() => {
          setLoading(false);
          abortControllerRef.current = null;
        });
      } else {
        setCircles([]);
        setLoading(false);
      }
    }, 300);
    
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [showCircles, selectedInteractions]);

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

  const getCircleColor = (index) => {
    const colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff'];
    return colors[index % colors.length];
  };

  const handleCircleClick = (circleId) => {
    if (onCircleClick) {
      onCircleClick(circleId);
    }
  };

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
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none'
      }}>
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
    </div>
  );
};

export default InteractiveCircles; 
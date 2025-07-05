import React, { useState, useEffect, useRef } from 'react';
import './InteractiveCircles.css';

const InteractiveCircles = ({ rois, showCircles, onCircleClick, selectedCircle, selectedInteractions }) => {
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
    
    // Use default zoom level since viewState is no longer available
    const zoomLevel = Math.pow(2, -(-2.5)); // Default zoom
    
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
    
    // Use default pan values
    const panX = 5454; // Default from backend config
    const panY = 2754; // Default from backend config
    
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
      // Circles are now handled by Vitessce layer, so we don't need DOM circles
      console.log('Circles are handled by Vitessce layer, clearing DOM circles');
      setCircles([]);
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

  // Separate effect for updating circle positions when needed
  useEffect(() => {
    if (circles.length > 0) {
      console.log('Updating circle positions');
      setCircles(updateCirclePositions(circles));
    }
  }, [circles.length]);

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

  // Circles are now handled by Vitessce layer, so we don't render DOM circles
  console.log('Circles are handled by Vitessce layer, returning null');
  return null;
};

export default InteractiveCircles; 
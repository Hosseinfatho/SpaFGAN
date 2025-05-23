import React, { useState, useEffect, useCallback } from 'react';
import { Vitessce } from 'vitessce';
import Plot from 'react-plotly.js'; // Import Plotly component
import { PanelGroup, Panel } from 'react-resizable-panels';
import TopInteractionPoints from './TopInteractionPoints';

// Helper function for default colors (optional)
const getDefaultColor = (index) => {
  const colors = [
    [255, 0, 255], [0, 255, 255],[255, 255, 0], // Yellow, Cyan, Magenta
    [255, 0, 0], [0, 255, 0], [0, 0, 255], // R, G, B
    [255, 165, 0], [128, 0, 128] // Orange, Purple
    // Add more colors if needed
  ];
  return colors[index % colors.length];
};

const groupColors = {
  1: '#d7191c',
  2: '#fdae61',
  3: '#abd9e9',
  4: '#2c7bb6'
};
const groupNames = {
  1: 'Endothelial-immune interface (CD31 + CD11b)',
  2: 'ROS detox, immune stress (CD11b + Catalase)',
  3: 'T/B cell recruitment via vessels (CD31 + CD4/CD20)',
  4: 'T–B collaboration (CD4 + CD20)'
};

// Update the scale factor constant
const SCALE_FACTOR = 8; // Changed from 4 to 8 to match resolution 3

function ROIBestView() {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  // States
  const [activeChannels, setActiveChannels] = useState([]); 
  const [currentViewState, setCurrentViewState] = useState(null); // Store the dynamically generated initial state
  const [analysisResults, setAnalysisResults] = useState({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [availableChannels, setAvailableChannels] = useState({}); // Store all { index: name }
  const [showChannelSelector, setShowChannelSelector] = useState(false); // Control channel selector visibility
  const [isAddingChannel, setIsAddingChannel] = useState(false); // Indicate loading state for adding channel

  // --- State for Heatmap Results ---
  const [heatmapResults, setHeatmapResults] = useState({});
  const [isAnalyzingHeatmaps, setIsAnalyzingHeatmaps] = useState(false);
  // --- Add state for Interaction Heatmap ---
  const [interactionHeatmapResult, setInteractionHeatmapResult] = useState(null);
  const [isAnalyzingInteractionHeatmap, setIsAnalyzingInteractionHeatmap] = useState(false);
  // --- Add state for Projection Type ---
  const [projectionType, setProjectionType] = useState('mean'); // Changed default to 'mean'

  // --- State for ROI definition (Min/Max) - Will be calculated ---
  const [roiState, setRoiState] = useState({
    xMin: 1, xMax: 10908,
    yMin: 1, yMax: 5508,
    zMin: 1,  zMax: 194,
  });

  // --- State for ROI Range definition ---
  const [roiRangeState, setRoiRangeState] = useState({
    x: 200, // +/- 50 pixels around center
    y: 200, // +/- 50 pixels around center
    z: 194, // +/- 5 slices around center
  });

  // --- State for hiding individual channel heatmaps ---
  const [hiddenChannelHeatmaps, setHiddenChannelHeatmaps] = useState([]);

  // --- State for Group Toggle ---
  const [activeGroups, setActiveGroups] = useState({
    1: true,
    2: true,
    3: true,
    4: true
  });

  // Add state for selected group
  const [selectedGroup, setSelectedGroup] = useState(1);

  // Add new state for save status
  const [saveStatus, setSaveStatus] = useState(null);

  // --- Handler for ROI Range input changes ---
  const handleRoiRangeChange = (event) => {
    const { name, value } = event.target;
    setRoiRangeState(prevState => ({
        ...prevState,
        [name]: value === '' ? '' : Number(value) 
    }));
  };

  // --- Function to Calculate ROI from View Center and Range ---
  const calculateRoiFromView = () => {
    // Hardcoded image dimensions (Replace with dynamic loading if possible later)
    const imgDims = {x: 10908, y: 5508, z: 194 }; 

    const centerX = Math.round(viewControlState.targetX);
    const centerY = Math.round(viewControlState.targetY);
    const centerZ = Math.round(viewControlState.targetZ);

    const rangeX = roiRangeState.x;
    const rangeY = roiRangeState.y;
    const rangeZ = roiRangeState.z;

    if (isNaN(centerX) || isNaN(centerY) || isNaN(centerZ) || isNaN(rangeX) || isNaN(rangeY) || isNaN(rangeZ)) {
        console.error("Invalid center or range values for ROI calculation.");
        setError("Invalid numeric input for View Center or ROI Range.");
        return;
    }

    // Calculate initial min/max
    let xMin = centerX - rangeX;
    let xMax = centerX + rangeX;
    let yMin = centerY - rangeY;
    let yMax = centerY + rangeY;
    let zMin = centerZ - rangeZ;
    let zMax = centerZ + rangeZ;

    // Clamp to image boundaries and ensure integers
    xMin = Math.max(0, Math.round(xMin));
    xMax = Math.min(imgDims.x - 1, Math.round(xMax));
    yMin = Math.max(0, Math.round(yMin));
    yMax = Math.min(imgDims.y - 1, Math.round(yMax));
    zMin = Math.max(0, Math.round(zMin));
    zMax = Math.min(imgDims.z - 1, Math.round(zMax));

    // Ensure min <= max (adjust if range is too large or center is at edge)
    if (xMin >= xMax) xMax = xMin + 1; // Ensure at least 1 pixel width if possible
    if (yMin >= yMax) yMax = yMin + 1;
    if (zMin >= zMax) zMax = zMin + 1;
    // Re-clamp max values after potential adjustment
    xMax = Math.min(imgDims.x - 1, xMax);
    yMax = Math.min(imgDims.y - 1, yMax);
    zMax = Math.min(imgDims.z - 1, zMax);

    const calculatedRoi = { xMin, xMax, yMin, yMax, zMin, zMax };
    console.log("[calculateRoiFromView] Calculated ROI:", calculatedRoi);
    setRoiState(calculatedRoi);
    setError(null); // Clear previous errors
  };

 
  // --- Handler for View Control input changes ---
  const handleViewControlChange = (event) => {
    const { name, value } = event.target;
    setViewControlState(prevState => ({
        ...prevState,
        [name]: value === '' ? '' : Number(value)
    }));
  };

  // Base view state structure (without channels initially)
  const baseViewState = {
      spatialTargetZ: 0,
      spatialTargetT: 0,
      spatialZoom: -3.5,
      spatialTargetX: 5500,
      spatialTargetY: 2850,
      spatialRenderingMode: "3D",
      imageLayer: [
        {
          spatialTargetResolution: 3, // Adjust if necessary
          spatialLayerOpacity: 1.0,
          spatialLayerVisible: true,
          photometricInterpretation: "BlackIsZero",
          imageChannel: [], // Will be populated dynamically
        
        }
      ]
    };
 // --- State for View Control ---
 const [viewControlState, setViewControlState] = useState({
  targetX: 5500,
  targetY: 2800,
  targetZ: 0,
  zoom: -3.4,
});

  // Fetch channel names and then generate config on mount
  useEffect(() => {
    let isMounted = true; // Flag to prevent state updates on unmounted component

    const fetchAndConfigure = async () => {
      try {
        // 1. Fetch ALL Channel Names first
        console.log("Fetching all available channel names...");
        let allNamesData = {};
        try {
          const namesResponse = await fetch('http://127.0.0.1:5000/api/channel_names');
          if (!namesResponse.ok) {
            throw new Error(`HTTP error fetching names! status: ${namesResponse.status}`); 
          }
          allNamesData = await namesResponse.json();
          console.log("All channel names fetched:", allNamesData);
          if (!isMounted) return;
          setAvailableChannels(allNamesData); // Store all available channels
        } catch (namesError) {
            console.error("Failed to fetch channel names:", namesError);
            setError(`Failed to load channel list: ${namesError.message}`);
            // Optional: Decide if you want to proceed without channel names or stop
            // For now, we proceed, but channel selection might be limited
            if (!isMounted) return; 
        }

        // 2. Define the default channel indices to LOAD INITIALLY
        const initialVisibleChannelIndices = [19, 27, 37, 25, 59]; // <-- NEW Default Channels

        // 3. Build minimal initialViewState with only default channels

        // Define the specific settings for default channels
        const defaultChannelSettings = {
            19: { color: [0, 255, 0], window: [300, 20000] },
            27: { color: [255, 255, 0], window: [1000, 7000] },
            37: { color: [255, 0, 255], window: [700, 6000] },
            25: { color: [0, 255, 255], window: [1638, 10000] },
            59: { color: [255, 0, 0], window: [1638, 7000] }
        };

        const minimalInitialImageChannels = initialVisibleChannelIndices.map((channelIndex) => {
            // We might not have names if the fetch failed, handle gracefully
            const channelName = allNamesData[channelIndex] || `Channel ${channelIndex}`;
            // Get settings from our map, provide fallbacks if somehow missing
            const settings = defaultChannelSettings[channelIndex] || { color: getDefaultColor(0), window: null }; // Fallback color/window

            console.log(`Defining initial channel: Index=${channelIndex}, Name=${channelName}, Settings=${JSON.stringify(settings)}`);
            return {
              spatialTargetC: channelIndex,
              spatialChannelColor: settings.color, // Use specific color
              spatialChannelVisible: true, // These defaults are visible
              spatialChannelOpacity: 1.0,
              spatialChannelWindow: settings.window // Use specific window
            };
        });

        // Ensure we handle the case where default indices might not be in fetched names (though unlikely)
        if (minimalInitialImageChannels.length !== initialVisibleChannelIndices.length) {
             console.warn("Some default channel indices were not found in the available channels list!");
             // Potentially filter out channels that weren't found if needed
        }
        
        // Use minimal state for initial config generation
        const minimalInitialViewState = {
          ...baseViewState,
          imageLayer: [
            {
              ...baseViewState.imageLayer[0],
              imageChannel: minimalInitialImageChannels // Only include the defaults
            }
          ]
        };
        console.log("Generated Minimal Initial View State:", minimalInitialViewState);

        // 4. Fetch Vitessce Config using minimal state
        console.log("Fetching Vitessce config with minimal channels...");
        const configResponse = await fetch('http://127.0.0.1:5000/api/generate_config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(minimalInitialViewState) // Send the minimal state
        });
        if (!configResponse.ok) {
           // Throw error with status to be caught below
          throw new Error(`HTTP error generating config! status: ${configResponse.status}`);
        }
        const configData = await configResponse.json();
        console.log("ROIBestView config generated successfully:", configData);

        // 5. Update State if component is still mounted
        if (isMounted) {
          setConfig(configData);
          setCurrentViewState(minimalInitialViewState); // Store the minimal state we used
          updateActiveChannels(minimalInitialViewState); // Initialize activeChannels based on the minimal state
        }

      } catch (err) {
        console.error("Error during initial setup:", err); // Log the actual error object
        if (isMounted) {
          // Set the error state with the error message
          setError(err.message); 
        }
      }
    };

    fetchAndConfigure();

    // Cleanup function to set the flag on unmount
    return () => {
      isMounted = false;
    };

  }, []); // Run only once on mount

  // Re-introduced function to update active channels state
  const updateActiveChannels = (viewState) => {
    let currentActiveInfo = [];
    // Ensure viewState and nested properties exist before accessing
    if (viewState && viewState.imageLayer && viewState.imageLayer[0] && viewState.imageLayer[0].imageChannel) {
      currentActiveInfo = viewState.imageLayer[0].imageChannel
        .filter(ch => ch.spatialChannelVisible === true)
        .map(ch => ({ 
            index: ch.spatialTargetC, 
            // Use the color from the view state, fallback to gray if missing
            color: ch.spatialChannelColor || [128, 128, 128] 
        }));
    } else {
        console.warn("[updateActiveChannels] viewState or imageChannel structure not found:", viewState);
    }
    // Update the state directly
    console.log("[updateActiveChannels] Setting active channels to:", currentActiveInfo);
    setActiveChannels(currentActiveInfo);
  };

  // --- Function to Add a Channel --- 
  const handleAddChannel = async (event) => {
    const selectedIndexStr = event.target.value;
    setShowChannelSelector(false); // Hide selector immediately
    if (!selectedIndexStr) return; // Ignore if default option selected

    const selectedIndex = parseInt(selectedIndexStr, 10);
    console.log(`[handleAddChannel] User selected channel index: ${selectedIndex}`);

    if (!currentViewState || !currentViewState.imageLayer || !currentViewState.imageLayer[0]) {
        console.error("[handleAddChannel] Current view state is not ready.");
        return;
    }

    // Check if channel already exists in the view state
    const existingChannel = currentViewState.imageLayer[0].imageChannel.find(ch => ch.spatialTargetC === selectedIndex);
    if (existingChannel) {
        console.warn(`[handleAddChannel] Channel ${selectedIndex} is already added.`);
        // Optional: Maybe just make it visible if it was hidden?
        // For now, just return
        return;
    }

    setIsAddingChannel(true);
    setError(null); // Clear previous errors

    try {
        // Get info for the new channel
        const channelName = availableChannels[selectedIndex] || `Channel ${selectedIndex}`;
        const existingChannels = currentViewState.imageLayer[0].imageChannel;
        const newChannelColor = getDefaultColor(existingChannels.length); // Assign next color in sequence

        const newChannelObject = {
            spatialTargetC: selectedIndex,
            spatialChannelColor: newChannelColor,
            spatialChannelVisible: true, // Make the newly added channel visible
            spatialChannelOpacity: 1.0,
            spatialChannelWindow: null
        };
        console.log("[handleAddChannel] New channel object:", newChannelObject);

        // Create new view state
        const newImageChannelArray = [...existingChannels, newChannelObject];
        const newViewState = {
            ...currentViewState,
            imageLayer: [
                {
                    ...currentViewState.imageLayer[0],
                    imageChannel: newImageChannelArray
                }
            ]
        };
        console.log("[handleAddChannel] New view state constructed:", newViewState);

        // Fetch new config from backend
        console.log("[handleAddChannel] Fetching updated config from backend...");
        const configResponse = await fetch('http://127.0.0.1:5000/api/generate_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newViewState)
        });

        if (!configResponse.ok) {
            throw new Error(`HTTP error generating config! status: ${configResponse.status}`);
        }
        const newConfigData = await configResponse.json();
        console.log("[handleAddChannel] New config received:", newConfigData);

        // Update state with the new config and view state
        setConfig(newConfigData);
        setCurrentViewState(newViewState);
        updateActiveChannels(newViewState); // Update active channels list

    } catch (err) {
        console.error("[handleAddChannel] Error adding channel:", err);
        setError(`Failed to add channel: ${err.message}`);
    } finally {
        setIsAddingChannel(false);
    }
  };
  // --- End Function to Add a Channel ---

  // Callback for Vitessce view state changes - Updates both states
  const handleViewStateChange = useCallback((newViewState) => {
    // console.log("[handleViewStateChange] Received new view state:", JSON.parse(JSON.stringify(newViewState)));
    setCurrentViewState(newViewState);
    // Call the function to update activeChannels state as well
    updateActiveChannels(newViewState);
  }, []); // Dependency array is empty, function created once

  // Analysis Function - Reads activeChannels state
  const analyzeActiveChannels = async () => {
      // 1. Get active channels from the state variable
      if (activeChannels.length === 0) {
          console.log("No active channels in state to analyze.");
          setAnalysisResults({});
          return;
      }

      console.log("[analyzeActiveChannels] STARTING ANALYSIS FOR (from state):", activeChannels);
      setIsAnalyzing(true);
      setAnalysisResults({});

      // Use the global SCALE_FACTOR
      console.log(`[analyzeActiveChannels] Using global scale factor: ${SCALE_FACTOR}`);
      // -------------------------------------------------------------

      // Array dimensions
      const arrayDims = {
          z: 194,
          y: 688,
          x: 1363
      };

      // First downscale X and Y values
      const downscaledXMin = Math.round(Number(roiState.xMin) / SCALE_FACTOR);
      const downscaledXMax = Math.round(Number(roiState.xMax) / SCALE_FACTOR);
      const downscaledYMin = Math.round(Number(roiState.yMin) / SCALE_FACTOR);
      const downscaledYMax = Math.round(Number(roiState.yMax) / SCALE_FACTOR);

      // Now clamp the downscaled values to array dimensions
      const roiStateNumeric = {
          xMin: Math.max(0, Math.min(arrayDims.x, downscaledXMin)),
          xMax: Math.max(0, Math.min(arrayDims.x, downscaledXMax)),
          yMin: Math.max(0, Math.min(arrayDims.y, downscaledYMin)),
          yMax: Math.max(0, Math.min(arrayDims.y, downscaledYMax)),
          zMin: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMin) || 0)),
          zMax: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMax) || 0))
      };

      // Ensure min values are less than max values
      const finalXMin = Math.min(roiStateNumeric.xMin, roiStateNumeric.xMax);
      const finalXMax = Math.max(roiStateNumeric.xMin, roiStateNumeric.xMax);
      const finalYMin = Math.min(roiStateNumeric.yMin, roiStateNumeric.yMax);
      const finalYMax = Math.max(roiStateNumeric.yMin, roiStateNumeric.yMax);
      const finalZMin = Math.min(roiStateNumeric.zMin, roiStateNumeric.zMax);
      const finalZMax = Math.max(roiStateNumeric.zMin, roiStateNumeric.zMax);

      const analysisROI = {
          z: [finalZMin, finalZMax],
          y: [finalYMin, finalYMax],
          x: [finalXMin, finalXMax]
      };

      console.log("[analyzeActiveChannels] Original ROI:", roiState);
      console.log("[analyzeActiveChannels] Downscaled values before clamping:", {
          xMin: downscaledXMin, xMax: downscaledXMax,
          yMin: downscaledYMin, yMax: downscaledYMax
      });
      console.log("[analyzeActiveChannels] Final ROI:", analysisROI);
      console.log("[analyzeActiveChannels] Array dimensions:", arrayDims);

      // 4. Create promises using the activeChannels list and the DOWNSCALED ROI
      const histogramPromises = activeChannels.map(channelInfo => {
        const channelIndex = channelInfo.index;
        console.log(`[analyzeActiveChannels] Creating fetch promise for Ch ${channelIndex}`);
        return fetch('http://127.0.0.1:5000/api/histogram', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            channel_index: channelIndex,
            roi: analysisROI,
            num_bins: 50
          }),
        })
        .then(response => response.json().then(data => ({ ok: response.ok, status: response.status, data, channelIndex })))
        .catch(networkError => {
            console.error(`[analyzeActiveChannels] Network error Ch ${channelIndex}:`, networkError);
            return { ok: false, status: 0, error: networkError.message, channelIndex };
        });
      });

      // 5. Process results (same as before)
      const results = await Promise.allSettled(histogramPromises);
      console.log("[analyzeActiveChannels] Raw settled results:", results);

      const newAnalysisResults = {};
      results.forEach((result) => {
          if (result.status === 'fulfilled') {
              const responseData = result.value;
              const chIndex = responseData.channelIndex;
              if (typeof chIndex === 'undefined') {
                 console.error(` -> ERROR: channelIndex is undefined in fulfilled promise value!`, responseData);
                 return; 
              }
              if (responseData.ok) {
                 console.log(`Raw histogram data for channel ${chIndex}:`, {
                     binEdges: responseData.data.bin_edges,
                     counts: responseData.data.counts,
                     minCount: Math.min(...responseData.data.counts),
                     maxCount: Math.max(...responseData.data.counts),
                     nonZeroBins: responseData.data.counts.filter(c => c > 0).length
                 });
                 newAnalysisResults[chIndex] = { data: responseData.data, error: null };
              } else {
                 const errorMsg = responseData.data?.error || responseData.error || `Backend error (status ${responseData.status})`;
                 console.error(` -> Response NOT OK for Ch ${chIndex}:`, errorMsg);
                 newAnalysisResults[chIndex] = { data: null, error: errorMsg };
              }
          } else { 
              const reason = result.reason;
              const failedChannelIndex = reason?.channelIndex || 'unknown'; 
              console.error(` -> REJECTED promise for index ${failedChannelIndex}:`, reason);
              const errorMsg = reason?.message || 'Unknown fetch rejection';
              newAnalysisResults[failedChannelIndex] = { data: null, error: errorMsg };
          }
      });

      console.log("[analyzeActiveChannels] Updating state with results:", newAnalysisResults);
      setAnalysisResults(newAnalysisResults);
      setIsAnalyzing(false);
  };

  // --- Function to Set the View --- 
  const handleSetView = async () => {
    console.log("[handleSetView] Setting view with state:", viewControlState);
    setError(null);

    if (!currentViewState || !currentViewState.imageLayer || !currentViewState.imageLayer[0]) {
        console.error("[handleSetView] Current view state (with channels) is not ready.");
        setError("Cannot set view: Channel information not loaded yet.");
        return;
    }

    try {
        // Construct the new complete view state for the config request
        const newCompleteViewState = {
            ...currentViewState,
            spatialTargetX: viewControlState.targetX,
            spatialTargetY: viewControlState.targetY,
            spatialTargetZ: viewControlState.targetZ,
            spatialZoom: viewControlState.zoom,
            imageLayer: currentViewState.imageLayer
        };
        console.log("[handleSetView] Constructed view state for config request:", newCompleteViewState);

        // Fetch new config from backend
        console.log("[handleSetView] Fetching updated config from backend...");
        const configResponse = await fetch('http://127.0.0.1:5000/api/generate_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newCompleteViewState)
        });

        if (!configResponse.ok) {
            throw new Error(`HTTP error generating config! status: ${configResponse.status}`);
        }
        const newConfigData = await configResponse.json();
        console.log("[handleSetView] New config received:", newConfigData);

        // Update state with the new config and view state
        setConfig(newConfigData);
        setCurrentViewState(newCompleteViewState);
        updateActiveChannels(newCompleteViewState);

    } catch (err) {
        console.error("[handleSetView] Error setting view:", err);
        setError(`Failed to set view: ${err.message}`);
    }
  };

  // --- End Function to Set the View ---

  // --- Function to Analyze Heatmaps ---
  const analyzeHeatmaps = async () => {
    console.log("!!!!! analyzeHeatmaps function CALLED !!!!!");
    // 1. Get active channels from the state variable
    if (activeChannels.length === 0) {
        console.log("No active channels in state to analyze heatmaps.");
        setHeatmapResults({});
        return;
    }

    console.log("[analyzeHeatmaps] STARTING ANALYSIS FOR (from state):", activeChannels);
    setIsAnalyzingHeatmaps(true);
    setHeatmapResults({}); // Clear previous results
    setError(null); // Clear previous errors

    // Use the global SCALE_FACTOR
    console.log(`[analyzeHeatmaps] Using global scale factor: ${SCALE_FACTOR}`);
    // -------------------------------------------------------------

    // Array dimensions
    const arrayDims = {
        z: 194,
        y: 688,
        x: 1363
    };

    // First downscale X and Y values
    const downscaledXMin = Math.round(Number(roiState.xMin) / SCALE_FACTOR);
    const downscaledXMax = Math.round(Number(roiState.xMax) / SCALE_FACTOR);
    const downscaledYMin = Math.round(Number(roiState.yMin) / SCALE_FACTOR);
    const downscaledYMax = Math.round(Number(roiState.yMax) / SCALE_FACTOR);

    // Now clamp the downscaled values to array dimensions
    const roiStateNumeric = {
        xMin: Math.max(0, Math.min(arrayDims.x, downscaledXMin)),
        xMax: Math.max(0, Math.min(arrayDims.x, downscaledXMax)),
        yMin: Math.max(0, Math.min(arrayDims.y, downscaledYMin)),
        yMax: Math.max(0, Math.min(arrayDims.y, downscaledYMax)),
        zMin: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMin) || 0)),
        zMax: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMax) || 0))
    };

    // Ensure min values are less than max values
    const finalXMin = Math.min(roiStateNumeric.xMin, roiStateNumeric.xMax);
    const finalXMax = Math.max(roiStateNumeric.xMin, roiStateNumeric.xMax);
    const finalYMin = Math.min(roiStateNumeric.yMin, roiStateNumeric.yMax);
    const finalYMax = Math.max(roiStateNumeric.yMin, roiStateNumeric.yMax);
    const finalZMin = Math.min(roiStateNumeric.zMin, roiStateNumeric.zMax);
    const finalZMax = Math.max(roiStateNumeric.zMin, roiStateNumeric.zMax);

    const analysisROI = {
        z: [finalZMin, finalZMax],
        y: [finalYMin, finalYMax],
        x: [finalXMin, finalXMax]
    };

    console.log("[analyzeHeatmaps] Original ROI:", roiState);
    console.log("[analyzeHeatmaps] Downscaled values before clamping:", {
        xMin: downscaledXMin, xMax: downscaledXMax,
        yMin: downscaledYMin, yMax: downscaledYMax
    });
    console.log("[analyzeHeatmaps] Final ROI:", analysisROI);
    console.log("[analyzeHeatmaps] Array dimensions:", arrayDims);

    // 4. Create promises using the activeChannels list and the DOWNSCALED ROI
    const heatmapPromises = activeChannels.map(channelInfo => {
      const channelIndex = channelInfo.index;
      console.log(`[analyzeHeatmaps] Creating fetch promise for Ch ${channelIndex}`);
      return fetch('http://127.0.0.1:5000/api/z_projection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          channel_indices: [channelIndex],
          roi: analysisROI,
          interaction: false,
          projection_type: projectionType
        }),
      })
      .then(response => response.json().then(data => ({ ok: response.ok, status: response.status, data, channelIndex })))
      .catch(networkError => {
          console.error(`[analyzeHeatmaps] Network error Ch ${channelIndex}:`, networkError);
          return { ok: false, status: 0, error: networkError.message, channelIndex };
      });
    });

    // 5. Process results
    const results = await Promise.allSettled(heatmapPromises);
    console.log("[analyzeHeatmaps] Raw settled results:", results);

    const newHeatmapResults = {};
    results.forEach((result) => {
        if (result.status === 'fulfilled') {
            const responseData = result.value;
            const chIndex = responseData.channelIndex;
            if (typeof chIndex === 'undefined') {
               console.error(` -> ERROR: channelIndex is undefined in fulfilled promise value!`, responseData);
               return; 
            }
            if (responseData.ok) {
               const heatmapData = responseData.data.heatmap;
               // --- Log raw heatmap data ---
               if (chIndex === 19) { // Log for one channel
                   console.log(`DEBUG: Raw heatmap data received for Ch ${chIndex} (Shape: ${heatmapData?.length}x${heatmapData?.[0]?.length})`);
                   if (heatmapData && heatmapData.length > 0 && heatmapData[0].length > 0) {
                       // Check first row (corresponds to Y=0 in downscaled data)
                       console.log(`DEBUG: First row (Y=0) Ch ${chIndex}:`, heatmapData[0]?.slice(0, 10));
                       // Check row ~10% down (Y=80)
                       const rowIndex10 = Math.floor(heatmapData.length * 0.1);
                       console.log(`DEBUG: Row ~10% (Y=${rowIndex10}) Ch ${chIndex} (Row ${rowIndex10}):`, heatmapData[rowIndex10]?.slice(0, 10));
                       // Check a middle row (Y=400)
                       const midRowIndex = Math.floor(heatmapData.length / 2);
                       console.log(`DEBUG: Middle row (Y=${midRowIndex}) Ch ${chIndex} (Row ${midRowIndex}):`, heatmapData[midRowIndex]?.slice(0, 10));
                       // Check last row (Y=799)
                       console.log(`DEBUG: Last row (Y=${heatmapData.length - 1}) Ch ${chIndex}:`, heatmapData[heatmapData.length - 1]?.slice(0, 10));

                       // Check if *any* non-zero values exist in later rows (e.g. second half)
                       let foundNonZeroLater = false;
                       for(let i = midRowIndex; i < heatmapData.length; i++) {
                           if (heatmapData[i]?.some(val => val !== 0)) {
                               foundNonZeroLater = true;
                               break;
                           }
                       }
                       console.log(`DEBUG: Found non-zero in rows >= ${midRowIndex} (Y>=${midRowIndex})?`, foundNonZeroLater);
                   }
               }
               // -----------------------------

               // --- Re-introduce vertical flip ---
               const verticallyFlippedData = heatmapData.slice().reverse();

               // Check dimensions AFTER vertical flip
               if (verticallyFlippedData && verticallyFlippedData.length > 0 && Array.isArray(verticallyFlippedData[0]) && verticallyFlippedData[0].length > 0) { 
                    console.log(`[analyzeHeatmaps] Received heatmap for Ch ${chIndex}. Original Dimensions (Y, X):`, heatmapData.length, heatmapData[0].length);
                    // Store flipped data 
                    newHeatmapResults[chIndex] = { data: verticallyFlippedData, error: null }; 
               } else {
                   console.warn(`[analyzeHeatmaps] Received empty or invalid heatmap array after flip attempt for Ch ${chIndex}.`);
                   newHeatmapResults[chIndex] = { data: null, error: 'Received empty/invalid heatmap data' };
               }
            } else {
               const errorMsg = responseData.data?.error || responseData.error || `Backend error (status ${responseData.status})`;
               console.error(` -> Response NOT OK for Ch ${chIndex}:`, errorMsg);
               newHeatmapResults[chIndex] = { data: null, error: errorMsg };
            }
        } else { 
            const reason = result.reason;
            const failedChannelIndex = reason?.channelIndex || 'unknown'; 
            console.error(` -> REJECTED promise for index ${failedChannelIndex}:`, reason);
            const errorMsg = reason?.message || 'Unknown fetch rejection';
            newHeatmapResults[failedChannelIndex] = { data: null, error: errorMsg };
        }
    });

    console.log("[analyzeHeatmaps] Updating state with results:", newHeatmapResults);
    setHeatmapResults(newHeatmapResults);
    setIsAnalyzingHeatmaps(false);
  };
  // --- End Function to Analyze Heatmaps ---

  // --- MODIFIED Function to Analyze RGB Interaction Heatmap ---
  const analyzeInteractionHeatmap = async () => {
    console.log("[analyzeInteractionHeatmap] STARTING ANALYSIS");
    setIsAnalyzingInteractionHeatmap(true);
    setInteractionHeatmapResult(null); 
    setError(null); 

    // Use the global SCALE_FACTOR
    console.log(`[analyzeInteractionHeatmap] Using global scale factor: ${SCALE_FACTOR}`);
    // -------------------------------------------------------------

    // Array dimensions
    const arrayDims = {
        z: 194,
        y: 688,
        x: 1363
    };

    // First downscale X and Y values
    const downscaledXMin = Math.round(Number(roiState.xMin) / SCALE_FACTOR);
    const downscaledXMax = Math.round(Number(roiState.xMax) / SCALE_FACTOR);
    const downscaledYMin = Math.round(Number(roiState.yMin) / SCALE_FACTOR);
    const downscaledYMax = Math.round(Number(roiState.yMax) / SCALE_FACTOR);

    // Now clamp the downscaled values to array dimensions
    const roiStateNumeric = {
        xMin: Math.max(0, Math.min(arrayDims.x, downscaledXMin)),
        xMax: Math.max(0, Math.min(arrayDims.x, downscaledXMax)),
        yMin: Math.max(0, Math.min(arrayDims.y, downscaledYMin)),
        yMax: Math.max(0, Math.min(arrayDims.y, downscaledYMax)),
        zMin: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMin) || 0)),
        zMax: Math.max(0, Math.min(arrayDims.z, Number(roiState.zMax) || 0))
    };

    // Ensure min values are less than max values
    const finalXMin = Math.min(roiStateNumeric.xMin, roiStateNumeric.xMax);
    const finalXMax = Math.max(roiStateNumeric.xMin, roiStateNumeric.xMax);
    const finalYMin = Math.min(roiStateNumeric.yMin, roiStateNumeric.yMax);
    const finalYMax = Math.max(roiStateNumeric.yMin, roiStateNumeric.yMax);
    const finalZMin = Math.min(roiStateNumeric.zMin, roiStateNumeric.zMax);
    const finalZMax = Math.max(roiStateNumeric.zMin, roiStateNumeric.zMax);

    const analysisROI = {
        z: [finalZMin, finalZMax],
        y: [finalYMin, finalYMax],
        x: [finalXMin, finalXMax]
    };

    console.log("[analyzeInteractionHeatmap] Original ROI:", roiState);
    console.log("[analyzeInteractionHeatmap] Downscaled values before clamping:", {
        xMin: downscaledXMin, xMax: downscaledXMax,
        yMin: downscaledYMin, yMax: downscaledYMax
    });
    console.log("[analyzeInteractionHeatmap] Final ROI:", analysisROI);
    console.log("[analyzeInteractionHeatmap] Array dimensions:", arrayDims);
    
    try {
        const response = await fetch('http://127.0.0.1:5000/api/rgb_interaction_heatmap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                roi: analysisROI,
                projection_type: projectionType 
            }),
        });
        const responseData = await response.json();
        console.log('Interaction heatmap result:', responseData);
        if (!response.ok) {
            const errorMsg = responseData.error || `Backend error (status ${response.status})`;
            throw new Error(errorMsg);
        }
        if (responseData.heatmaps && responseData.shape) {
            setInteractionHeatmapResult({ 
                heatmaps: responseData.heatmaps,
                shape: responseData.shape,
                error: null 
            }); 
        } else {
            let errorDetail = 'Received incomplete data for interaction heatmap';
            if(responseData.error) errorDetail = responseData.error;
            setInteractionHeatmapResult({ heatmaps: null, shape: null, error: errorDetail });
        }
    } catch (networkError) {
        console.error(`[analyzeInteractionHeatmap] Network or processing error:`, networkError);
        setInteractionHeatmapResult({ heatmaps: null, shape: null, error: networkError.message || 'Network error' });
    } finally {
        setIsAnalyzingInteractionHeatmap(false);
    }
  };
  // --- End of MODIFIED Function ---

  // --- Function to Toggle Group ---
  const handleGroupToggle = (group) => {
    setActiveGroups(prev => ({ ...prev, [group]: !prev[group] }));
  };

  // --- Function to Get Filtered Interaction Heatmap ---
  const getMaxGroupHeatmap = () => {
    if (!interactionHeatmapResult?.heatmaps) return null;
    const h1 = interactionHeatmapResult.heatmaps['group_1'];
    const h2 = interactionHeatmapResult.heatmaps['group_2'];
    const h3 = interactionHeatmapResult.heatmaps['group_3'];
    const h4 = interactionHeatmapResult.heatmaps['group_4'];
    const height = h1.length;
    const width = h1[0]?.length || 0;
    const zMatrix = [];
    const valueMatrix = [];
    const groupMatrix = [];
    for (let i = 0; i < height; i++) {
      zMatrix[i] = [];
      valueMatrix[i] = [];
      groupMatrix[i] = [];
      for (let j = 0; j < width; j++) {
        const vals = [
          activeGroups[1] ? h1[i][j] : -Infinity,
          activeGroups[2] ? h2[i][j] : -Infinity,
          activeGroups[3] ? h3[i][j] : -Infinity,
          activeGroups[4] ? h4[i][j] : -Infinity
        ];
        let maxVal = Math.max(...vals);
        let groupIdx = vals.findIndex(v => v === maxVal) + 1;
        zMatrix[i][j] = groupIdx;
        valueMatrix[i][j] = maxVal === -Infinity ? 0 : maxVal;
        groupMatrix[i][j] = groupIdx;
      }
    }
    return { zMatrix, valueMatrix, groupMatrix, height, width };
  };

  const getSingleGroupHeatmap = (groupIdx) => {
    if (!interactionHeatmapResult?.heatmaps) return null;
    const h = interactionHeatmapResult.heatmaps[`group_${groupIdx}`];
    if (!h || !h.length) return null;
    
    const height = h.length;
    const width = h[0]?.length || 0;
    const zMatrix = [];
    const valueMatrix = [];
    for (let i = 0; i < height; i++) {
      zMatrix[i] = [];
      valueMatrix[i] = [];
      for (let j = 0; j < width; j++) {
        zMatrix[i][j] = groupIdx;
        valueMatrix[i][j] = h[i][j];
      }
    }
    return { zMatrix, valueMatrix, height, width };
  };

  // Add this function to convert group heatmaps to an RGB image:
  function getInteractionRGBImageMixture() {
    if (!interactionHeatmapResult?.heatmaps) return null;
    const h1 = interactionHeatmapResult.heatmaps['group_1'];
    const h2 = interactionHeatmapResult.heatmaps['group_2'];
    const h3 = interactionHeatmapResult.heatmaps['group_3'];
    const h4 = interactionHeatmapResult.heatmaps['group_4'];
    
    // Check if all heatmaps exist and have valid dimensions
    if (!h1 || !h2 || !h3 || !h4 || !h1.length || !h1[0]?.length) {
      console.error('Invalid heatmap data:', { h1, h2, h3, h4 });
      return null;
    }
    
    const height = h1.length;
    const width = h1[0].length;
    const rgb = [];
    for (let i = 0; i < height; i++) {
      rgb[i] = [];
      for (let j = 0; j < width; j++) {
        const vals = [
          activeGroups[1] ? h1[i][j] : 0,
          activeGroups[2] ? h2[i][j] : 0,
          activeGroups[3] ? h3[i][j] : 0,
          activeGroups[4] ? h4[i][j] : 0
        ];
        const colors = [
          [215, 25, 28],   // group 1
          [253, 174, 97],  // group 2
          [171, 217, 233], // group 3
          [44, 123, 182]   // group 4
        ];
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < 4; k++) {
          r += vals[k] * colors[k][0];
          g += vals[k] * colors[k][1];
          b += vals[k] * colors[k][2];
        }
        const sumVals = vals.reduce((a, b) => a + b, 0);
        if (sumVals > 1) {
          r /= sumVals;
          g /= sumVals;
          b /= sumVals;
        }
        rgb[i][j] = [Math.round(r), Math.round(g), Math.round(b)];
      }
    }
    return { rgb, height, width };
  }

  // Update handlePushToView function
  const handlePushToView = (point) => {
    setViewControlState(prev => ({
        ...prev,
        targetX: point.x * SCALE_FACTOR,
        targetY: point.y * SCALE_FACTOR,
        targetZ: point.z || 0
    }));
  };

  // Update handleHeatmapClick function
  const handleHeatmapClick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Scale the coordinates
    const scaledX = x * SCALE_FACTOR;
    const scaledY = y * SCALE_FACTOR;
    
    console.log(`Heatmap click at (${x}, ${y}) -> Scaled to (${scaledX}, ${scaledY})`);
    
    // Update view control state with scaled coordinates
    setViewControlState(prev => ({
        ...prev,
        targetX: scaledX,
        targetY: scaledY
    }));
  };

  // Add new function to handle saving default channels
  const handleSaveDefaultChannels = async () => {
    try {
      setSaveStatus({ type: 'loading', message: 'loading and saving default channels...' });
      
      const response = await fetch('http://127.0.0.1:5000/api/save_default_channels', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setSaveStatus({ 
        type: 'success', 
        message: `channels saved: ${result.message}` 
      });
    } catch (error) {
      console.error('Error saving default channels:', error);
      setSaveStatus({ 
        type: 'error', 
        message: `error saving channels: ${error.message}` 
      });
    }
  };

  if (error) {
    return <p style={{ color: 'red', padding: '10px' }}>Error: {error}</p>;
  }
  if (!config) {
    return <p style={{ padding: '10px' }}>Loading config...</p>;
  }

  // Render Vitessce normally, and the overlay separately using CSS classes
  return (
    // We need a fragment here because we return two sibling elements
    <>
      <Vitessce
        config={config}
        theme="light"
        height={null} // Let CSS handle height via .view-area > div:first-child
        width={null}  // Let CSS handle width
        onViewStateChange={handleViewStateChange} // This now updates activeChannels state too
      />

      <TopInteractionPoints
        groupId={selectedGroup}
        onPointSelect={handlePushToView}
        onGroupChange={setSelectedGroup}
      />

      {/* --- Analysis Control Button --- */}
      <button
            onClick={analyzeActiveChannels} 
            disabled={isAnalyzing || activeChannels.length === 0} // Disable based on state
            style={{
                position: 'absolute',
                top: '40px',
                left: '50px',
                zIndex: 10,
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '3px 8px',
                fontSize: '0.8em',
                cursor: 'pointer'
            }}
            title={`Analyze Histograms for ${activeChannels.length} Visible Channel${activeChannels.length > 1 ? 's' : ''}`}
        >
           {/* Optionally show count from state */} 
           {isAnalyzing ? 'Analyzing...' : `Histo (${activeChannels.length})`}
        </button>

        {/* --- Projection Type Toggle --- */}
        <button
            onClick={() => setProjectionType(prev => prev === 'mean' ? 'max' : 'mean')}
            style={{
                position: 'absolute',
                top: '80px', // Adjust position as needed
                left: '50px', // Adjust position as needed
                zIndex: 10,
                backgroundColor: '#17a2b8', // Teal color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '3px 8px',
                fontSize: '0.8em',
                cursor: 'pointer'
            }}
            title={`Current Projection: ${projectionType}. Click to switch.`}
        >
           Proj: {projectionType}
        </button>

        {/* --- Heatmap Button --- */}
        <button
            onClick={() => analyzeHeatmaps()} // 
            disabled={isAnalyzingHeatmaps || isAnalyzingInteractionHeatmap || activeChannels.length === 0}
            style={{
                position: 'absolute',
                top: '40px', // Align with Histo button for now
                left: '140px', // Position next to Histo
                zIndex: 10,
                backgroundColor: '#ffc107', // Yellow color
                color: 'black',
                border: 'none',
                borderRadius: '4px',
                padding: '3px 8px',
                fontSize: '0.8em',
                cursor: 'pointer'
            }}
            title={`Analyze Heatmaps for ${activeChannels.length} Visible Channel${activeChannels.length > 1 ? 's' : ''}`}
        >
           {isAnalyzingHeatmaps ? 'Analyzing HMaps...' : `Heatmap (${activeChannels.length})`}
        </button>

        {/* --- Interaction Heatmap Button --- */}
        <button
            onClick={analyzeInteractionHeatmap} // Call the new function
            disabled={isAnalyzingInteractionHeatmap || isAnalyzingHeatmaps || activeChannels.length < 2 || activeChannels.length > 6} // Disable based on state and channel count
            style={{
                position: 'absolute',
                top: '80px',
                left: '140px', // Position next to Heatmap button
                zIndex: 10,
                backgroundColor: '#6f42c1', // Purple color
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                padding: '3px 8px',
                fontSize: '0.8em',
                cursor: 'pointer'
            }}
            title={activeChannels.length < 2 || activeChannels.length > 6 ? 'Requires 2-6 active channels' : `Analyze Interaction Heatmap for ${activeChannels.length} Channels`}
        >
           {isAnalyzingInteractionHeatmap ? 'Analyzing Inter...' : `Interact (${activeChannels.length})`}
        </button>

        {/* --- View Control and ROI Controls --- */}
        <div style={{ 
          position: 'absolute', 
          top: '40px', 
          left: '300px', 
          zIndex: 10, 
          background: 'rgba(0,0,0,0.6)', 
          padding: '10px', 
          borderRadius: '8px', 
          fontSize: '0.75em', 
          color: 'white',
          display: 'flex',
          flexDirection: 'column',
          gap: '10px'
        }}>
          {/* View Controls */}
          <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
            <span style={{fontWeight: 'bold'}}>View:</span>
            <label>X: <input type="number" step="50" name="targetX" value={viewControlState.targetX} onChange={handleViewControlChange} style={{ width: '45px' }} /></label>
            <label>Y: <input type="number" step="50" name="targetY" value={viewControlState.targetY} onChange={handleViewControlChange} style={{ width: '45px' }} /></label>
            <label>Z: <input type="number" step="10" name="targetZ" value={viewControlState.targetZ} onChange={handleViewControlChange} style={{ width: '45px' }} /></label>
            <label>Zoom: <input type="number" step="0.5" name="zoom" value={viewControlState.zoom} onChange={handleViewControlChange} style={{ width: '45px' }} /></label>
            <button onClick={handleSetView} style={{ fontSize: '0.8em', padding: '1px 4px'}}>Set</button>
          </div>

          {/* ROI Range Controls */}
          <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
            <span style={{fontWeight: 'bold'}}>ROI Range (± Center):</span>
            <label>X: ±<input type="number" step="50" name="x" value={roiRangeState.x} onChange={handleRoiRangeChange} style={{ width: '45px' }} /></label>
            <label>Y: ±<input type="number" step="50" name="y" value={roiRangeState.y} onChange={handleRoiRangeChange} style={{ width: '45px' }} /></label>
            <label>Z: ±<input type="number" step="10" name="z" value={roiRangeState.z} onChange={handleRoiRangeChange} style={{ width: '45px' }} /></label>
            <button onClick={calculateRoiFromView} style={{ fontSize: '0.8em', padding: '1px 4px'}}>Calculate ROI</button>
          </div>

          {/* Calculated ROI Display */}
          <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
            <span style={{fontWeight: 'bold'}}>Calculated ROI:</span>
            <span>Z: [{roiState.zMin} - {roiState.zMax}]</span>
            <span>Y: [{roiState.yMin} - {roiState.yMax}]</span>
            <span>X: [{roiState.xMin} - {roiState.xMax}]</span>
          </div>
        </div>
        {/* --- End ROI Calculation and Display --- */}

        {/* --- Add Channel Button & Selector --- */}
        <div style={{ position: 'absolute', top: '2px', left: '50px', zIndex: 10 }}> 
            <button
                onClick={() => setShowChannelSelector(!showChannelSelector)}
                disabled={isAddingChannel || Object.keys(availableChannels).length === 0}
                style={{ /* Basic styling */ 
                    backgroundColor: '#28a745', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: '4px',
                    padding: '3px 8px',
                    fontSize: '0.8em',
                    cursor: 'pointer',
                    marginLeft: '5px' 
                }}
                title="Add a channel to the view"
            >
                {isAddingChannel ? 'Adding...' : 'Add Channel'}
            </button>

            {showChannelSelector && (
                <select 
                    onChange={handleAddChannel} 
                    style={{ marginLeft: '5px', fontSize: '0.8em' }}
                    defaultValue="" // Start with no value selected
                >
                    <option value="">-- Select Channel --</option>
                    {Object.entries(availableChannels)
                        // Filter out channels already present in the current view state
                        .filter(([indexStr]) => 
                            !currentViewState?.imageLayer?.[0]?.imageChannel.some(ch => ch.spatialTargetC === parseInt(indexStr, 10))
                        )
                        // Sort remaining channels maybe by index?
                        .sort(([indexA], [indexB]) => parseInt(indexA, 10) - parseInt(indexB, 10))
                        .map(([index, name]) => (
                            <option key={index} value={index}>{name} (Index: {index})</option>
                        ))
                    }
                </select>
            )}
        </div>
        {/* --- End Add Channel Button & Selector --- */}

      {/* --- Analysis Results Panel --- */}
      {!isAnalyzing && Object.keys(analysisResults).length > 0 && (
        <div
          className="analysis-overlay"
          style={{ // Top-left, small, scaled
            position: 'absolute',
            top: '120px',
            left: '15px',
            width: '300px',
            maxHeight: '1200px',
            overflowY: 'auto',
            zIndex: 10,
            transform: 'scale(0.6)',
            transformOrigin: 'top left',
            display: 'flex',
            flexDirection: 'column',
            gap: '1px'
          }}
        >
             <button
                onClick={() => setAnalysisResults({})} // Clear results
                style={{ 
                    position: 'absolute', 
                    top: '1px', 
                    right: '1px', 
                    cursor: 'pointer', 
                    border:'none', 
                    background:'rgba(0,0,0,0.9)', // Make button visible
                    borderRadius: '50%', // Circle button
                    width: '10px', 
                    height: '10px', 
                    lineHeight: '10px', // Center the 'x'
                    textAlign: 'center',
                    fontSize:'1.5em', 
                    color: '#FFF'
                }}
                title="Close Results"
             >
                &times;
             </button>

            {Object.entries(analysisResults).map(([channelIndex, result]) => {
                // Find the color from the activeChannels state
                const activeChannelInfo = activeChannels.find(ch => ch.index == channelIndex);
                const lineColor = activeChannelInfo ? `rgb(${activeChannelInfo.color.join(',')})` : '#007bff';
                
                // Get channel name from available map, fallback to index
                const channelName = availableChannels[channelIndex] || `Ch ${channelIndex}`; // Use availableChannels map

                let plotData = null;
                // Added more robust check for valid data structure
                if (result.data && result.data.bin_edges && result.data.counts && !result.error) {
                    console.log(`Processing histogram for channel ${channelIndex}:`, {
                        binEdges: result.data.bin_edges,
                        counts: result.data.counts
                    });
                    const binEdges = result.data.bin_edges;
                    const counts = result.data.counts;
                    if (counts.length === binEdges.length - 1) {
                         const binMidpoints = binEdges.slice(0, -1).map((edge, i) => (edge + binEdges[i+1]) / 2);
                         plotData = [{
                           x: binMidpoints,
                           y: counts,
                           type: 'scatter',
                           mode: 'lines',
                           line: { color: lineColor }
                         }];
                    } else {
                         const errorDetail = `counts=${counts.length}, bins=${binEdges.length}`;
                         console.error(`Invalid histogram data structure for Ch ${channelIndex}: ${errorDetail}`);
                         result.error = 'Invalid data structure'; // Set error state
                    }
                } else if (result.error) {
                     // Error already exists (from backend or previous check)
                    console.error(`Error state for Ch ${channelIndex}:`, result.error);
                } else {
                    // Catch cases where data is missing without an explicit error from backend
                    console.warn(`Missing or invalid data for Ch ${channelIndex}.`);
                    result.error = 'Missing or invalid data'; // Set error state
                }

                return (
                    <div
                       key={channelIndex}
                       style={{
                           borderBottom: '1px solid #ccc',
                           padding: '5px',
                           marginBottom: '5px',
                           background: 'rgba(0, 0, 0, 0.5)',
                           borderRadius: '3px'
                       }}
                    >
                        {/* Display Channel Name and Index */}
                        <h6 style={{textAlign: 'center', margin: '2px 0', color: lineColor, fontSize: '0.9em'}} title={`Index: ${channelIndex}`}>{channelName}</h6>
                        {result.error ? (
                            <p style={{ color: 'orange', fontSize: '0.8em', margin: '2px 0' }}>Error: {result.error}</p>
                        ) : plotData ? (
                            <Plot
                                data={plotData}
                                layout={{
                                  width: 240,
                                  height: 160,
                                  title: null,
                                  xaxis: { title: null, tickfont: {size: 8, color: 'white'} },
                                  yaxis: { title: null, type: 'log', tickfont: {size: 8, color: 'white'} },
                                  paper_bgcolor: 'rgba(0,0,0,0)',
                                  plot_bgcolor: 'rgba(0,0,0,0)',
                                  margin: { l: 25, r: 5, t: 5, b: 20 }
                                }}
                                config={{displayModeBar: false}}
                            />
                        ) : (
                            <p style={{fontSize: '0.9em'}}>No data.</p>
                        )}
                    </div>
                );
            })}
        </div>
      )}
      {/* --- End Analysis Results Panel --- */}

      {/* --- MODIFIED Heatmap Results Panel --- */}
      {(interactionHeatmapResult || Object.keys(heatmapResults).length > 0) && (
        <>
          {/* Regular Heatmaps Panel */}
          {Object.keys(heatmapResults).length > 0 && (
            <div
              className="heatmap-overlay"
              style={{ 
                position: 'absolute',
                bottom: '-50px', 
                left: '-10px', 
                right: 'auto', //
                height: '300px',
                width: 'fit-content', // 
                maxWidth: 'calc(100% + 10px)', 
                overflowX: 'auto',
                overflowY: 'auto',
                zIndex: 10,
                transform: 'scale(0.8)',
                background: 'rgba(50, 50, 50, 0.25)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                gap: '5px',
                padding: '5px',
                borderRadius: '4px',
                color: 'white'
              }}
            >
              <button
                onClick={() => {
                  setHeatmapResults({});
                  setHiddenChannelHeatmaps([]);
                }}
                style={{ 
                  position: 'absolute', 
                  top: '5px', 
                  right: '5px', 
                  cursor: 'pointer', 
                  border:'none', 
                  background:'rgba(255,255,255,0.2)',
                  borderRadius: '50%',
                  width: '10px', 
                  height: '10px', 
                  lineHeight: '2px',
                  textAlign: 'center',
                  fontSize:'20px', 
                  color: '#f00',
                  fontWeight: 'bold'
                }}
                title="Close All Heatmaps"
              >
                &times;
              </button>

              {/* Scrollable Row for Regular Heatmaps */}
              <div style={{ 
                display: 'flex', 
                flexDirection: 'row', 
                gap: '2px', 
                width: 'fit-content',
                alignItems: 'flex-start', 
                paddingBottom:'2px',
                border: '1px solid rgba(255,255,255,0.2)',
                borderRadius: '4px',
                padding: '2px 7px' 
              }}>
                {Object.entries(heatmapResults)
                  .filter(([channelIndex]) => !hiddenChannelHeatmaps.includes(Number(channelIndex)))
                  .map(([channelIndex, result]) => {
                    if (result && result.data) {
                      const chIndexNum = Number(channelIndex);
                      const channelName = availableChannels[chIndexNum] || `Channel ${chIndexNum}`;
                      const activeChannelInfo = activeChannels.find(ch => ch.index == chIndexNum);
                      const channelColor = activeChannelInfo ? activeChannelInfo.color : [128, 128, 128];
                      return (
                        <div key={chIndexNum} style={{
                          position: 'relative', 
                          border: `1px solid rgb(${channelColor.join(',')})`, 
                          borderRadius: '4px', 
                          padding: '5px', 
                          backgroundColor: 'rgba(0,0,0,0.3)',
                          flexShrink: 0
                        }}>
                          <button
                            onClick={() => setHiddenChannelHeatmaps(prev => [...prev, chIndexNum])}
                            style={{
                              position: 'absolute',
                              top: '15px',
                              right: '15px',
                              zIndex: 10,
                              cursor: 'pointer',
                              backgroundColor: 'rgba(200, 0, 0, 0.6)',
                              color: 'white',
                              border: '1px solid rgba(150, 0, 0, 0.8)',
                              borderRadius: '50%',
                              width: '22px',
                              height: '22px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              padding: '0',
                              fontSize: '12px',
                              fontWeight: 'bold',
                              lineHeight: '22px'
                            }}
                            title={`Close Heatmap for ${channelName}`}
                          >
                            X
                          </button>
                          <Plot
                            data={[{
                              z: result.data,
                              type: 'heatmap',
                              colorscale: 'Viridis',
                              colorbar: { thickness: 10, tickfont: { size: 8, color: 'white'}, len: 0.9, y: 0.5, yanchor: 'middle' }
                            }]}
                            layout={{
                              title: {text: `${channelName} (${projectionType})`, font: {color: `rgb(${channelColor.join(',')})`} },
                              width: 300, 
                              height: 220, 
                              margin: { t: 30, b: 20, l: 20, r: 20 },
                              paper_bgcolor: 'rgba(0,0,0,0)',
                              plot_bgcolor: 'rgba(0,0,0,0.1)',
                              font: { color: 'white' },
                              xaxis: { visible: false },
                              yaxis: { visible: false, scaleanchor: 'x' }
                            }}
                            config={{ responsive: true, displaylogo: false }}
                          />
                        </div>
                      );
                    }
                    return null;
                  })}
              </div>
            </div>
          )}

          {/* Interaction Heatmap Panel */}
          {interactionHeatmapResult && (
            <div
              className="interaction-heatmap-overlay"
              style={{ 
                position: 'absolute',
                bottom: '75px', 
                right: '75px',
                width: 'fit-content',
                height: '300px',
                zIndex: 10,
                transform: 'scale(1.2)',
                background: 'rgba(50, 50, 50, 0.25)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '5px',
                padding: '5px',
                borderRadius: '4px',
                color: 'white'
              }}
            >
              <button
                onClick={() => setInteractionHeatmapResult(null)}
                style={{ 
                  position: 'absolute', 
                  top: '5px', 
                  right: '5px', 
                  cursor: 'pointer', 
                  border:'none', 
                  background:'rgba(255,255,255,0.2)',
                  borderRadius: '50%',
                  width: '5px', 
                  height: '5px', 
                  lineHeight: '2px',
                  textAlign: 'center',
                  fontSize:'15px', 
                  color: '#f00',
                  fontWeight: 'bold'
                }}
                title="Close Interaction Heatmap"
              >
                &times;
              </button>

              {interactionHeatmapResult.error ? (
                <p style={{ color: 'orange', fontSize: '0.8em', margin: '2px 0' }}>
                    Error: {interactionHeatmapResult.error}
                </p>
              ) : (
                <div style={{
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '4px',
                  padding: '5px',
                  backgroundColor: 'rgba(0,0,0,0.3)'
                }}>
                  {(() => {
                    const activeGroupIndices = Object.keys(activeGroups).filter(k => activeGroups[k]);
                    if (activeGroupIndices.length === 1) {
                      const groupIdx = Number(activeGroupIndices[0]);
                      const result = getSingleGroupHeatmap(groupIdx);
                      if (!result) return <div style={{color:'gray',textAlign:'center'}}>No data available</div>;
                      
                      const { zMatrix, valueMatrix, height, width } = result;
                      const colorHex = groupColors[groupIdx];
                      return (
                        <Plot
                          data={[{
                            z: valueMatrix.slice().reverse(),
                            type: 'heatmap',
                            colorscale: [
                              [0, 'black'],
                              [1, colorHex]
                            ],
                            zmin: 0,
                            zmax: 1,
                            showscale: true,
                            customdata: valueMatrix.slice().reverse().map((row, i) => row.map((val, j) => {
                              return {
                                value: val,
                                x: j,
                                y: i,
                                z: interactionHeatmapResult.shape ? interactionHeatmapResult.shape[0] : null
                              };
                            })),
                            hovertemplate:
                              '<span style="font-size:11px">value: %{customdata.value:.1f}<br>' +
                              'x: %{customdata.x}<br>'+
                              'y: %{customdata.y}</span>' +
                              '<br><span style="color:#4CAF50;cursor:pointer">Click to push to view</span><extra></extra>'
                          }]} 
                          layout={{
                            title: {
                              text: groupNames[groupIdx],
                              font: { color: colorHex, size: 10 }
                            },
                            width: 300,
                            height: 220,
                            margin: { t: 25, b: 20, l: 20, r: 20 },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0.1)',
                            font: { color: 'white', size: 9 },
                            xaxis: { showgrid: false },
                            yaxis: { showgrid: false }
                          }}
                          config={{ displayModeBar: false }}
                          onClick={handleHeatmapClick}
                        />
                      );
                    } else if (activeGroupIndices.length > 1) {
                      const result = getInteractionRGBImageMixture();
                      if (!result) return <div style={{color:'gray',textAlign:'center'}}>No data available</div>;
                      
                      const { rgb, height, width } = result;
                      const canvas = document.createElement('canvas');
                      canvas.width = width;
                      canvas.height = height;
                      const ctx = canvas.getContext('2d');
                      const imageData = ctx.createImageData(width, height);
                      for (let y = 0; y < height; y++) {
                        for (let x = 0; x < width; x++) {
                          const idx = (y * width + x) * 4;
                          imageData.data[idx] = rgb[y][x][0];
                          imageData.data[idx + 1] = rgb[y][x][1];
                          imageData.data[idx + 2] = rgb[y][x][2];
                          imageData.data[idx + 3] = 255;
                        }
                      }
                      ctx.putImageData(imageData, 0, 0);
                      const dataUrl = canvas.toDataURL();
                      return (
                        <div style={{ position: 'relative', width: 300, height: 220 }}>
                          <img
                            src={dataUrl}
                            alt="Interaction RGB Mixture"
                            style={{ width: 300, height: 220, imageRendering: 'pixelated', borderRadius: '4px', border: '1px solid #444' }}
                          />
                          <div style={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            color: 'white',
                            fontWeight: 'bold',
                            fontSize: '0.5em',
                            textAlign: 'center',
                            textShadow: '0 0 4px #000',
                            background: 'rgba(0,0,0,0.2)',
                            padding: '2px 0'
                          }}>
                            Cellular Interactions (Mixture)
                          </div>
                        </div>
                      );
                    } else {
                      return <div style={{color:'gray',textAlign:'center'}}>Select at least one group</div>;
                    }
                  })()}
                  <div style={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    justifyContent: 'space-around',
                    marginTop: '5px',
                    fontSize: '0.7em',
                    gap: '2px'
                  }}>
                    <label style={{ color: '#d7191c', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input type="checkbox" checked={activeGroups[1]} onChange={() => handleGroupToggle(1)} style={{marginRight: '4px'}} />
                      Endothelial-immune interface (CD31 + CD11b)
                    </label>
                    <label style={{ color: '#fdae61', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input type="checkbox" checked={activeGroups[2]} onChange={() => handleGroupToggle(2)} style={{marginRight: '4px'}} />
                      ROS detox, immune stress (CD11b + Catalase)
                    </label>
                    <label style={{ color: '#abd9e9', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input type="checkbox" checked={activeGroups[3]} onChange={() => handleGroupToggle(3)} style={{marginRight: '4px'}} />
                      T/B cell recruitment via vessels (CD31 + CD4/CD20)
                    </label>
                    <label style={{ color: '#2c7bb6', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <input type="checkbox" checked={activeGroups[4]} onChange={() => handleGroupToggle(4)} style={{marginRight: '4px'}} />
                      T–B collaboration (CD4 + CD20)
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
      {/* --- End MODIFIED Heatmap Results Panel --- */}

      {/* --- DEBUGGING INFO (Optional) --- */}
      {/*
      <div style={{position: 'absolute', bottom: '5px', right: '5px', background: 'rgba(0,0,0,0.7)', color: 'white', padding: '5px', fontSize: '0.7em', zIndex: 20}}> 
        Active Channels: {JSON.stringify(activeChannels)} <br />
        Channel Names: {Object.keys(availableChannels).length > 0 ? 'Loaded' : 'Loading...'} 
      </div>
      */}

      {/* Add Save Default Channels button */}
      <div className="control-group">
        <button 
          onClick={handleSaveDefaultChannels}
          className="control-button"
          title="Save default channels in Zarr format"
        >
          save default channels
    </button>
        {saveStatus && (
          <div className={`status-message ${saveStatus.type}`}>
            {saveStatus.message}
          </div>
        )}
      </div>
    </>
  );
}

export default ROIBestView; 
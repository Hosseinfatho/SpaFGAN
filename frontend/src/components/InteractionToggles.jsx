import React, { useState } from 'react';
import './InteractionToggles.css';

// Import INTERACTION_TYPES from Original.jsx
const INTERACTION_TYPES = {
  'B-cell_infiltration': 'B-cell infiltration',
  'T-cell_maturation': 'T-cell maturation', 
  'Inflammatory_zone': 'Inflammatory zone',
  'Oxidative_stress_regulation': 'Oxidative stress regulation'
};

const InteractionToggles = ({ onConfigChange }) => {
    const [activeInteractions, setActiveInteractions] = useState(['B-cell_infiltration']);

    const handleToggleChange = (interactionKey, isChecked) => {
        let newActiveInteractions;
        
        if (isChecked) {
            // Add to active interactions
            newActiveInteractions = [...activeInteractions, interactionKey];
        } else {
            // Remove from active interactions
            newActiveInteractions = activeInteractions.filter(key => key !== interactionKey);
        }
        
        // Ensure at least one interaction is always active
        if (newActiveInteractions.length === 0) {
            newActiveInteractions = ['B-cell_infiltration'];
        }
        
        setActiveInteractions(newActiveInteractions);
        
        // Notify parent component about config change
        if (onConfigChange) {
            onConfigChange(newActiveInteractions);
        }
    };

    return (
        <div className="interaction-toggles">
            <h3>Interaction Controls</h3>
            <div className="toggles-container">
                {Object.entries(INTERACTION_TYPES).map(([key, label]) => (
                    <div key={key} className="toggle-item">
                        <label className="toggle-label">
                            <input
                                type="checkbox"
                                checked={activeInteractions.includes(key)}
                                onChange={(e) => handleToggleChange(key, e.target.checked)}
                                className="toggle-checkbox"
                            />
                            <span className="toggle-slider"></span>
                            <span className="toggle-text">{label}</span>
                        </label>
                    </div>
                ))}
            </div>
            <div className="active-interactions">
                <strong>Active:</strong> {activeInteractions.map(key => INTERACTION_TYPES[key]).join(', ')}
            </div>
        </div>
    );
};

export default InteractionToggles; 
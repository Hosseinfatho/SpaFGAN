import React from 'react';
import { Plot } from 'react-plotly.js';

const EvaluationResults = ({ results }) => {
    if (!results) return null;

    const { classifier_metrics, biological_metrics, visual_metrics, embeddings_2d } = results;

    return (
        <div style={{ padding: '20px', backgroundColor: 'rgba(0,0,0,0.05)', borderRadius: '8px' }}>
            <h2>ROI Evaluation Results</h2>
            
            {/* Classifier Metrics */}
            <div style={{ marginBottom: '20px' }}>
                <h3>Classifier Performance</h3>
                <div style={{ display: 'flex', gap: '20px' }}>
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <p>Accuracy: {(classifier_metrics.accuracy * 100).toFixed(2)}%</p>
                    </div>
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <p>AUC: {classifier_metrics.auc.toFixed(3)}</p>
                    </div>
                </div>
            </div>

            {/* Biological Metrics */}
            <div style={{ marginBottom: '20px' }}>
                <h3>Biological Plausibility</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                    {Object.entries(biological_metrics).map(([pair, metrics]) => (
                        <div key={pair} style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                            <h4>{pair.replace('_', ' ').toUpperCase()}</h4>
                            <p>High-scoring coexpression: {metrics.high_scoring_coexpression.toFixed(3)}</p>
                            <p>Low-scoring coexpression: {metrics.low_scoring_coexpression.toFixed(3)}</p>
                            <p>Difference: {metrics.difference.toFixed(3)}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Visual Metrics */}
            <div style={{ marginBottom: '20px' }}>
                <h3>Visual Interpretability</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h4>ROI Organization</h4>
                        <p>Spatial Coherence: {visual_metrics.roi_organization.spatial_coherence.toFixed(3)}</p>
                        <p>Boundary Definition: {visual_metrics.roi_organization.boundary_definition.toFixed(3)}</p>
                    </div>
                    <div style={{ padding: '10px', backgroundColor: 'white', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                        <h4>Cell Type Enrichment</h4>
                        <p>Enrichment Score: {visual_metrics.cell_type_enrichment.enrichment_score.toFixed(3)}</p>
                        <p>Specificity Score: {visual_metrics.cell_type_enrichment.specificity_score.toFixed(3)}</p>
                    </div>
                </div>
            </div>

            {/* t-SNE Visualization */}
            <div style={{ marginBottom: '20px' }}>
                <h3>t-SNE Visualization</h3>
                <div style={{ width: '100%', height: '400px' }}>
                    <Plot
                        data={[{
                            x: embeddings_2d.map(point => point[0]),
                            y: embeddings_2d.map(point => point[1]),
                            type: 'scatter',
                            mode: 'markers',
                            marker: {
                                size: 8,
                                color: embeddings_2d.map((_, i) => i),
                                colorscale: 'Viridis',
                                showscale: true
                            }
                        }]}
                        layout={{
                            title: 't-SNE Visualization of ROI Embeddings',
                            xaxis: { title: 't-SNE 1' },
                            yaxis: { title: 't-SNE 2' },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)'
                        }}
                        style={{ width: '100%', height: '100%' }}
                    />
                </div>
            </div>
        </div>
    );
};

export default EvaluationResults; 
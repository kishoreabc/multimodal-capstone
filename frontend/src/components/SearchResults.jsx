import React from 'react';
import './SearchResults.css';

const SearchResults = ({ results, isLoading, error }) => {
    if (isLoading) {
        return (
            <div className="results-container">
                <div className="loading-state">
                    <div className="loading-spinner-large"></div>
                    <p>Searching...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="results-container">
                <div className="error-state">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <circle cx="12" cy="12" r="10" strokeWidth="2" />
                        <line x1="12" y1="8" x2="12" y2="12" strokeWidth="2" />
                        <line x1="12" y1="16" x2="12.01" y2="16" strokeWidth="2" />
                    </svg>
                    <h3>Search Failed</h3>
                    <p>{error}</p>
                </div>
            </div>
        );
    }

    if (!results || results.length === 0) {
        return (
            <div className="results-container">
                <div className="empty-state">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <circle cx="11" cy="11" r="8" strokeWidth="2" />
                        <path d="m21 21-4.35-4.35" strokeWidth="2" />
                    </svg>
                    <h3>No Results</h3>
                    <p>Try a different search query</p>
                </div>
            </div>
        );
    }

    return (
        <div className="results-container">
            <div className="results-header">
                <h2>Search Results</h2>
                <span className="results-count">{results.length} results found</span>
            </div>
            <div className="results-grid">
                {results.map((result, index) => {
                    // Extract filename from path - handle both \ and / separators
                    const pathParts = result.document.replace(/\\/g, '/').split('/');
                    const filename = pathParts[pathParts.length - 1];
                    const imageUrl = `http://localhost:8000/images/${filename}`;

                    return (
                        <div key={index} className="result-card">
                            <div className="result-rank">#{index + 1}</div>
                            {(result.score !== null && result.score !== undefined) && (
                                <div className="result-score score">
                                    Score: {result.score.toFixed(4)}
                                </div>
                            )}
                            {(result.distance !== null && result.distance !== undefined) && (
                                <div className="result-score distance">
                                    Distance: {result.distance.toFixed(4)}
                                </div>
                            )}
                            <div className="result-image-container">
                                <img
                                    src={imageUrl}
                                    alt={`Result ${index + 1}`}
                                    className="result-image"
                                    onError={(e) => {
                                        e.target.style.display = 'none';
                                        e.target.nextSibling.style.display = 'block';
                                    }}
                                />
                                <div className="image-error" style={{ display: 'none' }}>
                                    Image not found
                                </div>
                            </div>
                            {result.metadata && Object.keys(result.metadata).length > 0 && (
                                <div className="result-metadata">
                                    {/* Define fixed order for metadata fields */}
                                    {['material', 'category', 'color', 'stone', 'style'].map((key) => {
                                        if (result.metadata[key] !== undefined && result.metadata[key] !== null) {
                                            return (
                                                <div key={key} className="metadata-item">
                                                    <span className="metadata-key">{key}:</span>
                                                    <span className="metadata-value">{String(result.metadata[key])}</span>
                                                </div>
                                            );
                                        }
                                        return null;
                                    })}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default SearchResults;

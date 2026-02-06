import React, { useState } from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch, isLoading }) => {
    const [query, setQuery] = useState('');
    const [topK, setTopK] = useState(5);
    const [useReranking, setUseReranking] = useState(true);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query, topK, useReranking);
        }
    };

    return (
        <div className="search-bar-container">
            <form onSubmit={handleSubmit} className="search-form">
                <div className="search-input-wrapper">
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search for jewelry... (e.g., 'gold wedding ring')"
                        className="search-input"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        className="search-button"
                        disabled={isLoading || !query.trim()}
                    >
                        {isLoading ? (
                            <span className="loading-spinner"></span>
                        ) : (
                            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                                <path d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM18 18l-4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        )}
                    </button>
                </div>

                <div className="search-controls">
                    <div className="control-group">
                        <label htmlFor="topK">Results: {topK}</label>
                        <input
                            id="topK"
                            type="range"
                            min="1"
                            max="20"
                            value={topK}
                            onChange={(e) => setTopK(parseInt(e.target.value))}
                            className="slider"
                            disabled={isLoading}
                        />
                    </div>

                    <div className="control-group">
                        <label className="checkbox-label">
                            <input
                                type="checkbox"
                                checked={useReranking}
                                onChange={(e) => setUseReranking(e.target.checked)}
                                disabled={isLoading}
                            />
                            <span>Use Reranking</span>
                        </label>
                    </div>
                </div>
            </form>
        </div>
    );
};

export default SearchBar;

import React, { useState } from 'react';
import './ImageUpload.css';

const ImageUpload = ({ onSearch, isLoading }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [topK, setTopK] = useState(5);
    const [useReranking, setUseReranking] = useState(false);
    const [queryText, setQueryText] = useState('');
    const [isDragging, setIsDragging] = useState(false);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (selectedFile) {
            onSearch(selectedFile, topK, useReranking, queryText || null);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setQueryText('');
    };

    return (
        <div className="image-upload-container">
            <form onSubmit={handleSubmit} className="upload-form">
                <div
                    className={`drop-zone ${isDragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    {preview ? (
                        <div className="preview-container">
                            <img src={preview} alt="Preview" className="preview-image" />
                            <button
                                type="button"
                                onClick={handleClear}
                                className="clear-button"
                                disabled={isLoading}
                            >
                                ✕
                            </button>
                        </div>
                    ) : (
                        <div className="drop-zone-content">
                            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" strokeWidth="2" />
                                <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" />
                                <polyline points="21 15 16 10 5 21" strokeWidth="2" />
                            </svg>
                            <p>Drag & drop an image here</p>
                            <p className="or-text">or</p>
                            <label htmlFor="file-input" className="file-input-label">
                                Choose File
                            </label>
                            <input
                                id="file-input"
                                type="file"
                                accept="image/*"
                                onChange={handleFileInput}
                                className="file-input"
                                disabled={isLoading}
                            />
                        </div>
                    )}
                </div>

                {useReranking && (
                    <div className="rerank-input-wrapper">
                        <input
                            type="text"
                            value={queryText}
                            onChange={(e) => setQueryText(e.target.value)}
                            placeholder="Optional: Enter text query for reranking"
                            className="rerank-input"
                            disabled={isLoading}
                        />
                    </div>
                )}

                <div className="upload-controls">
                    <div className="control-group">
                        <label htmlFor="image-topK">Results: {topK}</label>
                        <input
                            id="image-topK"
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

                <button
                    type="submit"
                    className="submit-button"
                    disabled={isLoading || !selectedFile}
                >
                    {isLoading ? (
                        <>
                            <span className="loading-spinner"></span>
                            Searching...
                        </>
                    ) : (
                        'Search by Image'
                    )}
                </button>
            </form>
        </div>
    );
};

export default ImageUpload;

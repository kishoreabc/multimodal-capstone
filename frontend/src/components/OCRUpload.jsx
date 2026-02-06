import React, { useState } from 'react';
import './OCRUpload.css';

const OCRUpload = ({ onSearch, isLoading }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [topK, setTopK] = useState(5);
    const [useReranking, setUseReranking] = useState(true);
    const [isDragging, setIsDragging] = useState(false);
    const [extractedText, setExtractedText] = useState('');

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith('image/')) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
            setExtractedText(''); // Clear previous extracted text
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

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (selectedFile) {
            const result = await onSearch(selectedFile, topK, useReranking);
            if (result && result.extracted_text) {
                setExtractedText(result.extracted_text);
            }
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setExtractedText('');
    };

    return (
        <div className="ocr-upload-container">
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
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeWidth="2" />
                                <polyline points="14 2 14 8 20 8" strokeWidth="2" />
                                <line x1="16" y1="13" x2="8" y2="13" strokeWidth="2" />
                                <line x1="16" y1="17" x2="8" y2="17" strokeWidth="2" />
                                <polyline points="10 9 9 9 8 9" strokeWidth="2" />
                            </svg>
                            <p>Drag & drop a handwritten image here</p>
                            <p className="or-text">or</p>
                            <label htmlFor="ocr-file-input" className="file-input-label">
                                Choose File
                            </label>
                            <input
                                id="ocr-file-input"
                                type="file"
                                accept="image/*"
                                onChange={handleFileInput}
                                className="file-input"
                                disabled={isLoading}
                            />
                        </div>
                    )}
                </div>

                {extractedText && (
                    <div className="extracted-text-container">
                        <label>Extracted Text:</label>
                        <textarea
                            value={extractedText}
                            readOnly
                            className="extracted-text"
                            rows="3"
                        />
                    </div>
                )}

                <div className="upload-controls">
                    <div className="control-group">
                        <label htmlFor="ocr-topK">Results: {topK}</label>
                        <input
                            id="ocr-topK"
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
                            Extracting & Searching...
                        </>
                    ) : (
                        'Extract Text & Search'
                    )}
                </button>
            </form>
        </div>
    );
};

export default OCRUpload;

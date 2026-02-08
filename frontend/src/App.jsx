import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import ImageUpload from './components/ImageUpload';
import OCRUpload from './components/OCRUpload';
import SearchResults from './components/SearchResults';
import { searchByText, searchByImage, searchByOCR } from './services/api';
import './App.css';

function App() {
    const [activeTab, setActiveTab] = useState('text');
    const [results, setResults] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleTextSearch = async (query, topK, useReranking) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await searchByText(query, topK, useReranking);
            setResults(response.results);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Search failed. Please try again.');
            setResults(null);
        } finally {
            setIsLoading(false);
        }
    };

    const handleImageSearch = async (file, topK, useReranking, queryText) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await searchByImage(file, topK, useReranking, queryText);
            console.log('Image search response:', response);
            console.log('First result sample:', response.results[0]);
            setResults(response.results);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Search failed. Please try again.');
            setResults(null);
        } finally {
            setIsLoading(false);
        }
    };

    const handleOCRSearch = async (file, topK, useReranking) => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await searchByOCR(file, topK, useReranking);
            setResults(response.results);
            return response; // Return response so OCRUpload can get extracted_text
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'OCR search failed. Please try again.');
            setResults(null);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app">
            <div className="background-gradient"></div>

            <div className="container">
                <header className="header">
                    <h1 className="title">
                        <span className="gradient-text">Multimodal Search</span>
                    </h1>
                    <p className="subtitle">Search jewelry using text or images with AI-powered reranking</p>
                </header>

                <div className="tabs">
                    <button
                        className={`tab ${activeTab === 'text' ? 'active' : ''}`}
                        onClick={() => {
                            setActiveTab('text');
                            setResults(null);
                            setError(null);
                        }}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M4 7h16M4 12h16M4 17h10" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        Text Search
                    </button>
                    <button
                        className={`tab ${activeTab === 'image' ? 'active' : ''}`}
                        onClick={() => {
                            setActiveTab('image');
                            setResults(null);
                            setError(null);
                        }}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" strokeWidth="2" />
                            <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" />
                            <polyline points="21 15 16 10 5 21" strokeWidth="2" />
                        </svg>
                        Image Search
                    </button>
                    <button
                        className={`tab ${activeTab === 'ocr' ? 'active' : ''}`}
                        onClick={() => {
                            setActiveTab('ocr');
                            setResults(null);
                            setError(null);
                        }}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" strokeWidth="2" />
                            <polyline points="14 2 14 8 20 8" strokeWidth="2" />
                            <line x1="16" y1="13" x2="8" y2="13" strokeWidth="2" />
                            <line x1="16" y1="17" x2="8" y2="17" strokeWidth="2" />
                        </svg>
                        OCR Search
                    </button>
                </div>

                <div className="search-section">
                    {activeTab === 'text' ? (
                        <SearchBar onSearch={handleTextSearch} isLoading={isLoading} />
                    ) : activeTab === 'image' ? (
                        <ImageUpload onSearch={handleImageSearch} isLoading={isLoading} />
                    ) : (
                        <OCRUpload onSearch={handleOCRSearch} isLoading={isLoading} />
                    )}
                </div>

                {(results || isLoading || error) && (
                    <SearchResults results={results} isLoading={isLoading} error={error} />
                )}
            </div>

            <footer className="footer">
                <p>Powered by CLIP, ChromaDB, and Cross-Encoder Reranking</p>
            </footer>
        </div>
    );
}

export default App;

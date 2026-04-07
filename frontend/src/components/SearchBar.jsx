import React, { useState, useRef } from 'react';
import { speechToText } from '../services/api';
import './SearchBar.css';

const SearchBar = ({ onSearch, isLoading }) => {
    const [query, setQuery] = useState('');
    const [topK, setTopK] = useState(5);
    const [useReranking, setUseReranking] = useState(true);
    const [isRecording, setIsRecording] = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query, topK, useReranking);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : 'audio/webm'
            });
            mediaRecorderRef.current = mediaRecorder;
            chunksRef.current = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // Stop all tracks to release the microphone
                stream.getTracks().forEach(track => track.stop());

                const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
                setIsTranscribing(true);

                try {
                    const text = await speechToText(audioBlob);
                    if (text) {
                        setQuery(text);
                    }
                } catch (err) {
                    console.error('Transcription failed:', err);
                } finally {
                    setIsTranscribing(false);
                }
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (err) {
            console.error('Microphone access denied:', err);
            alert('Microphone access is required for voice input. Please allow microphone access and try again.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleMicClick = () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
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
                        placeholder={isTranscribing ? "Transcribing your voice..." : "Search for fashion products... (e.g., 'watches, shoes, etc')"}
                        className="search-input"
                        disabled={isLoading || isTranscribing}
                    />
                    <button
                        type="button"
                        className={`mic-button ${isRecording ? 'recording' : ''} ${isTranscribing ? 'transcribing' : ''}`}
                        onClick={handleMicClick}
                        disabled={isLoading || isTranscribing}
                        title={isRecording ? 'Stop recording' : isTranscribing ? 'Transcribing...' : 'Voice search'}
                    >
                        {isTranscribing ? (
                            <span className="loading-spinner mic-spinner"></span>
                        ) : isRecording ? (
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                <rect x="6" y="6" width="12" height="12" rx="2" />
                            </svg>
                        ) : (
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="9" y="1" width="6" height="12" rx="3" />
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                <line x1="12" y1="19" x2="12" y2="23" />
                                <line x1="8" y1="23" x2="16" y2="23" />
                            </svg>
                        )}
                        {isRecording && <span className="recording-pulse"></span>}
                    </button>
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

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

/**
 * Check if the backend is healthy
 */
export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data;
    } catch (error) {
        console.error('Health check failed:', error);
        throw error;
    }
};

/**
 * Search using text query
 * @param {string} query - The search query text
 * @param {number} topK - Number of results to return
 * @param {boolean} useReranking - Whether to use reranking
 * @param {object} filters - Optional metadata filters
 */
export const searchByText = async (query, topK = 5, useReranking = true, filters = null) => {
    try {
        const response = await api.post('/api/search/text', {
            query,
            top_k: topK,
            use_reranking: useReranking,
            filters,
        });
        return response.data;
    } catch (error) {
        console.error('Text search failed:', error);
        throw error;
    }
};

/**
 * Search using image upload
 * @param {File} file - The image file to search with
 * @param {number} topK - Number of results to return
 * @param {boolean} useReranking - Whether to use reranking
 * @param {string} queryText - Optional text query for reranking
 */
export const searchByImage = async (file, topK = 5, useReranking = false, queryText = null) => {
    try {
        console.log('searchByImage called with:', { topK, useReranking, queryText });

        const formData = new FormData();
        formData.append('file', file);
        formData.append('top_k', topK);
        formData.append('use_reranking', useReranking);
        if (queryText) {
            formData.append('query_text', queryText);
        }

        console.log('FormData contents:');
        for (let [key, value] of formData.entries()) {
            console.log(`  ${key}:`, value);
        }

        const response = await api.post('/api/search/image', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Image search failed:', error);
        throw error;
    }
};

/**
 * Search using OCR - extract text from image and search
 * @param {File} file - The image file containing text to extract
 * @param {number} topK - Number of results to return
 * @param {boolean} useReranking - Whether to use reranking
 */
export const searchByOCR = async (file, topK = 5, useReranking = true) => {
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('top_k', topK);
        formData.append('use_reranking', useReranking);

        const response = await api.post('/api/search/ocr', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('OCR search failed:', error);
        throw error;
    }
};

import axios from 'axios';
import { Standard, SearchResult, ProjectContext, Recommendation } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const StandardsService = {
  async getAllStandards() {
    const response = await api.get('/api/standards');
    return response.data;
  },

  async getStandardById(id: string): Promise<Standard> {
    const response = await api.get(`/api/standards/${id}`);
    return response.data;
  },

  async searchStandards(query: string, filters?: any, limit?: number): Promise<SearchResult[]> {
    const response = await api.post('/api/search', {
      query,
      filters,
      limit
    });
    return response.data.results;
  },

  async analyzeProject(context: ProjectContext): Promise<Recommendation[]> {
    const response = await api.post('/api/analyze', context);
    return response.data.recommendations;
  },

  async getCategories() {
    const response = await api.get('/api/categories');
    return response.data;
  },

  async getTags() {
    const response = await api.get('/api/tags');
    return response.data;
  },

  async exportStandard(id: string, format: string = 'markdown'): Promise<Blob> {
    const response = await api.get(`/api/export/${id}`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  async exportBulkStandards(standardIds: string[], format: string = 'json'): Promise<Blob> {
    const response = await api.post('/api/export/bulk', {
      standards: standardIds,
      format
    }, {
      responseType: 'blob'
    });
    return response.data;
  }
};
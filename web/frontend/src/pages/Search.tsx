import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  TextField,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  IconButton,
  Collapse,
  Alert,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterListIcon,
  Visibility as VisibilityIcon,
  Clear as ClearIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { debounce } from 'lodash';
import toast from 'react-hot-toast';
import { StandardsService } from '../services/standardsService';
import { SearchResult } from '../types';
import { useStandards } from '../contexts/StandardsContext';

const Search: React.FC = () => {
  const navigate = useNavigate();
  const { categories } = useStandards();
  const [searchQuery, setSearchQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    category: '',
    tags: [] as string[],
  });
  const [savedSearches, setSavedSearches] = useState<string[]>([]);

  const performSearch = useCallback(
    debounce(async (query: string, searchFilters: any) => {
      if (!query.trim() && !searchFilters.category && searchFilters.tags.length === 0) {
        setResults([]);
        return;
      }

      try {
        setLoading(true);
        const searchResults = await StandardsService.searchStandards(
          query,
          searchFilters,
          50
        );
        setResults(searchResults);
      } catch (error) {
        toast.error('Search failed');
        console.error(error);
      } finally {
        setLoading(false);
      }
    }, 300),
    []
  );

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const query = event.target.value;
    setSearchQuery(query);
    performSearch(query, filters);
  };

  const handleFilterChange = (event: SelectChangeEvent<string>) => {
    const newFilters = { ...filters, category: event.target.value };
    setFilters(newFilters);
    performSearch(searchQuery, newFilters);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setFilters({ category: '', tags: [] });
    setResults([]);
  };

  const handleSaveSearch = () => {
    if (searchQuery.trim()) {
      setSavedSearches([...savedSearches, searchQuery]);
      toast.success('Search saved');
    }
  };

  const handleViewStandard = (standardId: string) => {
    navigate(`/standards/${standardId}`);
  };

  const highlightText = (text: string, highlights?: string[]) => {
    if (!highlights || highlights.length === 0) return text;
    
    let highlightedText = text;
    highlights.forEach((highlight) => {
      const regex = new RegExp(`(${highlight})`, 'gi');
      highlightedText = highlightedText.replace(
        regex,
        '<mark style="background-color: yellow;">$1</mark>'
      );
    });
    
    return <span dangerouslySetInnerHTML={{ __html: highlightedText }} />;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Search Standards
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Search through all standards using semantic search
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box display="flex" alignItems="center" gap={2}>
          <TextField
            fullWidth
            placeholder="Search for standards, rules, or examples..."
            value={searchQuery}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
              endAdornment: searchQuery && (
                <InputAdornment position="end">
                  <IconButton onClick={handleClearSearch}>
                    <ClearIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          <IconButton onClick={() => setShowFilters(!showFilters)}>
            <FilterListIcon />
          </IconButton>
          <Button variant="outlined" onClick={handleSaveSearch}>
            Save Search
          </Button>
        </Box>

        <Collapse in={showFilters}>
          <Box mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={filters.category}
                    onChange={handleFilterChange}
                    label="Category"
                  >
                    <MenuItem value="">All Categories</MenuItem>
                    {categories.map((cat) => (
                      <MenuItem key={cat.name} value={cat.name}>
                        {cat.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        </Collapse>
      </Paper>

      {loading && (
        <Alert severity="info">Searching...</Alert>
      )}

      {!loading && results.length === 0 && searchQuery && (
        <Alert severity="info">No results found for "{searchQuery}"</Alert>
      )}

      {!loading && results.length > 0 && (
        <Box>
          <Typography variant="h6" gutterBottom>
            {results.length} results found
          </Typography>
          <Grid container spacing={2}>
            {results.map((result) => (
              <Grid item xs={12} key={result.standard.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="start">
                      <Box flex={1}>
                        <Typography variant="h6" gutterBottom>
                          {highlightText(
                            result.standard.title,
                            result.highlights.title
                          )}
                        </Typography>
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          paragraph
                        >
                          {highlightText(
                            result.standard.description,
                            result.highlights.description
                          )}
                        </Typography>
                        <Box>
                          <Chip
                            label={result.standard.category}
                            size="small"
                            sx={{ mr: 1 }}
                          />
                          {result.standard.tags.slice(0, 3).map((tag) => (
                            <Chip
                              key={tag}
                              label={tag}
                              size="small"
                              variant="outlined"
                              sx={{ mr: 0.5 }}
                            />
                          ))}
                        </Box>
                      </Box>
                      <Box>
                        <Chip
                          label={`Score: ${(result.score * 100).toFixed(0)}%`}
                          color="primary"
                          size="small"
                        />
                      </Box>
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button
                      size="small"
                      startIcon={<VisibilityIcon />}
                      onClick={() => handleViewStandard(result.standard.id)}
                    >
                      View Details
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {savedSearches.length > 0 && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Saved Searches
          </Typography>
          <Box>
            {savedSearches.map((search, index) => (
              <Chip
                key={index}
                label={search}
                onClick={() => setSearchQuery(search)}
                onDelete={() => {
                  setSavedSearches(savedSearches.filter((_, i) => i !== index));
                }}
                sx={{ mr: 1, mb: 1 }}
              />
            ))}
          </Box>
        </Paper>
      )}
    </Box>
  );
};
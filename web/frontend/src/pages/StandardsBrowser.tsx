import React, { useState } from 'react';
import {
  Box,
  Typography,
  TreeView,
  TreeItem,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ChevronRight as ChevronRightIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useStandards } from '../contexts/StandardsContext';
import { Standard } from '../types';

const StandardsBrowser: React.FC = () => {
  const navigate = useNavigate();
  const { standards, loading } = useStandards();
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState<string[]>([]);

  const handleToggle = (event: React.SyntheticEvent, nodeIds: string[]) => {
    setExpanded(nodeIds);
  };

  const handleSelect = (event: React.SyntheticEvent, nodeId: string) => {
    if (!nodeId.includes('-')) {
      setSelectedCategory(nodeId);
    }
  };

  const filteredStandards = selectedCategory
    ? standards[selectedCategory] || []
    : Object.values(standards).flat();

  const searchFilteredStandards = filteredStandards.filter(
    (standard) =>
      standard.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      standard.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
      standard.tags.some((tag) => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const handleViewStandard = (standardId: string) => {
    navigate(`/standards/${standardId}`);
  };

  const handleExportStandard = async (standardId: string) => {
    // TODO: Implement export functionality
    console.log('Export standard:', standardId);
  };

  if (loading) {
    return <Typography>Loading standards...</Typography>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Standards Browser
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Browse and explore all available standards
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Categories
            </Typography>
            <TreeView
              defaultCollapseIcon={<ExpandMoreIcon />}
              defaultExpandIcon={<ChevronRightIcon />}
              expanded={expanded}
              selected={selectedCategory}
              onNodeToggle={handleToggle}
              onNodeSelect={handleSelect}
              sx={{ flexGrow: 1, overflowY: 'auto' }}
            >
              <TreeItem nodeId="all" label="All Standards" />
              {Object.entries(standards).map(([category, categoryStandards]) => (
                <TreeItem
                  key={category}
                  nodeId={category}
                  label={
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography>{category}</Typography>
                      <Chip label={categoryStandards.length} size="small" />
                    </Box>
                  }
                >
                  {categoryStandards.map((standard) => (
                    <TreeItem
                      key={standard.id}
                      nodeId={`${category}-${standard.id}`}
                      label={standard.title}
                      onClick={() => handleViewStandard(standard.id)}
                    />
                  ))}
                </TreeItem>
              ))}
            </TreeView>
          </Paper>
        </Grid>

        <Grid item xs={12} md={9}>
          <Box mb={2}>
            <TextField
              fullWidth
              placeholder="Search standards..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
          </Box>

          <Grid container spacing={2}>
            {searchFilteredStandards.map((standard) => (
              <Grid item xs={12} md={6} key={standard.id}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {standard.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {standard.description}
                    </Typography>
                    <Box mt={1}>
                      {standard.tags.slice(0, 3).map((tag) => (
                        <Chip key={tag} label={tag} size="small" sx={{ mr: 0.5 }} />
                      ))}
                      {standard.tags.length > 3 && (
                        <Chip label={`+${standard.tags.length - 3} more`} size="small" />
                      )}
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button
                      size="small"
                      startIcon={<VisibilityIcon />}
                      onClick={() => handleViewStandard(standard.id)}
                    >
                      View
                    </Button>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={() => handleExportStandard(standard.id)}
                    >
                      Export
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};
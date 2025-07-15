import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  TextField,
  InputAdornment,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Checkbox,
  CircularProgress,
  Alert,
  Snackbar,
} from '@mui/material';
import { SimpleTreeView, TreeItem } from '@mui/x-tree-view';
import {
  ExpandMore as ExpandMoreIcon,
  ChevronRight as ChevronRightIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useStandards } from '../contexts/StandardsContext';
import { StandardsService } from '../services/standardsService';

const StandardsBrowser: React.FC = () => {
  const navigate = useNavigate();
  const { standards, loading } = useStandards();
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState<string[]>([]);
  const [selectedStandards, setSelectedStandards] = useState<Set<string>>(new Set());
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState('json');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportSuccess, setExportSuccess] = useState(false);
  const [selectMode, setSelectMode] = useState(false);

  const handleToggle = (event: React.SyntheticEvent, itemIds: string[]) => {
    setExpanded(itemIds);
  };

  const handleSelect = (event: React.SyntheticEvent, itemIds: string | null) => {
    if (itemIds && !itemIds.includes('-')) {
      setSelectedCategory(itemIds);
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
    try {
      setExporting(true);
      const blob = await StandardsService.exportStandard(standardId, exportFormat);
      downloadBlob(blob, `${standardId}.${exportFormat}`);
      setExportSuccess(true);
    } catch (error) {
      console.error('Export failed:', error);
      setExportError('Failed to export standard');
    } finally {
      setExporting(false);
    }
  };

  const handleBulkExport = async () => {
    try {
      setExporting(true);
      setExportError(null);

      const standardsToExport = selectedStandards.size > 0 
        ? Array.from(selectedStandards)
        : Object.values(standards).flat().map(s => s.id);

      if (standardsToExport.length === 0) {
        setExportError('No standards to export');
        return;
      }

      if (exportFormat === 'json') {
        // Use bulk export endpoint for JSON
        const blob = await StandardsService.exportBulkStandards(standardsToExport, 'json');
        downloadBlob(blob, `standards-export-${Date.now()}.json`);
      } else {
        // Export individual files
        const fileFormat = exportFormat === 'json-individual' ? 'json' : exportFormat;
        for (const standardId of standardsToExport) {
          const blob = await StandardsService.exportStandard(standardId, fileFormat);
          downloadBlob(blob, `${standardId}.${fileFormat}`);
          // Add small delay to prevent overwhelming the browser
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }

      setExportSuccess(true);
      setExportDialogOpen(false);
      setSelectedStandards(new Set());
      setSelectMode(false);
    } catch (error) {
      console.error('Bulk export failed:', error);
      setExportError('Failed to export standards');
    } finally {
      setExporting(false);
    }
  };

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  };

  const toggleStandardSelection = (standardId: string) => {
    const newSelection = new Set(selectedStandards);
    if (newSelection.has(standardId)) {
      newSelection.delete(standardId);
    } else {
      newSelection.add(standardId);
    }
    setSelectedStandards(newSelection);
  };

  const selectAllInCategory = () => {
    const categoryStandards = selectedCategory
      ? standards[selectedCategory] || []
      : Object.values(standards).flat();
    
    const allIds = categoryStandards.map(s => s.id);
    setSelectedStandards(new Set(allIds));
  };

  const clearSelection = () => {
    setSelectedStandards(new Set());
  };

  if (loading) {
    return <Typography>Loading standards...</Typography>;
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Standards Browser
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Browse and explore all available standards
          </Typography>
        </Box>
        <Box>
          {selectMode && (
            <Box display="flex" gap={1} alignItems="center" mr={2} component="span">
              <Typography variant="body2" color="text.secondary">
                {selectedStandards.size} selected
              </Typography>
              <Button size="small" onClick={selectAllInCategory}>Select All</Button>
              <Button size="small" onClick={clearSelection}>Clear</Button>
            </Box>
          )}
          <Button
            variant={selectMode ? "contained" : "outlined"}
            onClick={() => setSelectMode(!selectMode)}
            sx={{ mr: 1 }}
          >
            {selectMode ? 'Cancel Selection' : 'Select Multiple'}
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={() => setExportDialogOpen(true)}
            disabled={selectMode && selectedStandards.size === 0}
          >
            Export Standards
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        <Grid size={{xs: 12, md: 3}}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Categories
            </Typography>
            <SimpleTreeView
              expandedItems={expanded}
              selectedItems={selectedCategory}
              onExpandedItemsChange={handleToggle}
              onSelectedItemsChange={handleSelect}
              slots={{
                collapseIcon: ExpandMoreIcon,
                expandIcon: ChevronRightIcon,
              }}
              sx={{ flexGrow: 1, overflowY: 'auto' }}
            >
              <TreeItem itemId="all" label="All Standards" />
              {Object.entries(standards).map(([category, categoryStandards]) => (
                <TreeItem
                  key={category}
                  itemId={category}
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
                      itemId={`${category}-${standard.id}`}
                      label={standard.title}
                      onClick={() => handleViewStandard(standard.id)}
                    />
                  ))}
                </TreeItem>
              ))}
            </SimpleTreeView>
          </Paper>
        </Grid>

        <Grid size={{xs: 12, md: 9}}>
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
              <Grid size={{xs: 12, md: 6}} key={standard.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                      <Typography variant="h6" gutterBottom>
                        {standard.title}
                      </Typography>
                      {selectMode && (
                        <Checkbox
                          checked={selectedStandards.has(standard.id)}
                          onChange={() => toggleStandardSelection(standard.id)}
                          icon={<CheckBoxOutlineBlankIcon />}
                          checkedIcon={<CheckBoxIcon />}
                        />
                      )}
                    </Box>
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
                      onClick={() => {
                        setExportFormat('markdown');
                        handleExportStandard(standard.id);
                      }}
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

      {/* Export Dialog */}
      <Dialog open={exportDialogOpen} onClose={() => setExportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export Standards</DialogTitle>
        <DialogContent>
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              {selectedStandards.size > 0 
                ? `Export ${selectedStandards.size} selected standard${selectedStandards.size > 1 ? 's' : ''}`
                : 'Export all standards'}
            </Typography>
          </Box>
          
          <FormControl component="fieldset">
            <FormLabel component="legend">Export Format</FormLabel>
            <RadioGroup
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
            >
              <FormControlLabel 
                value="json" 
                control={<Radio />} 
                label="JSON (Single file with all standards)" 
              />
              <FormControlLabel 
                value="markdown" 
                control={<Radio />} 
                label="Markdown (Individual files)" 
              />
              <FormControlLabel 
                value="json-individual" 
                control={<Radio />} 
                label="JSON (Individual files)" 
              />
            </RadioGroup>
          </FormControl>

          {exportError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {exportError}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)} disabled={exporting}>
            Cancel
          </Button>
          <Button 
            onClick={handleBulkExport} 
            variant="contained" 
            disabled={exporting}
            startIcon={exporting ? <CircularProgress size={20} /> : <DownloadIcon />}
          >
            {exporting ? 'Exporting...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success Snackbar */}
      <Snackbar
        open={exportSuccess}
        autoHideDuration={3000}
        onClose={() => setExportSuccess(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="success" onClose={() => setExportSuccess(false)}>
          Standards exported successfully!
        </Alert>
      </Snackbar>

      {/* Error Snackbar */}
      <Snackbar
        open={!!exportError && !exportDialogOpen}
        autoHideDuration={3000}
        onClose={() => setExportError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity="error" onClose={() => setExportError(null)}>
          {exportError}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default StandardsBrowser;
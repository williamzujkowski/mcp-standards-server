import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Chip,
  Button,
  Tabs,
  Tab,
  Grid,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  ContentCopy as ContentCopyIcon,
} from '@mui/icons-material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import toast from 'react-hot-toast';
import { StandardsService } from '../services/standardsService';
import { Standard } from '../types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`standard-tabpanel-${index}`}
      aria-labelledby={`standard-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const StandardDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [standard, setStandard] = useState<Standard | null>(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    if (id) {
      loadStandard(id);
    }
  }, [id]);

  const loadStandard = async (standardId: string) => {
    try {
      setLoading(true);
      const data = await StandardsService.getStandardById(standardId);
      setStandard(data);
    } catch (error) {
      toast.error('Failed to load standard');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!standard) return;
    try {
      const blob = await StandardsService.exportStandard(standard.id, 'markdown');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${standard.id}.md`;
      a.click();
      window.URL.revokeObjectURL(url);
      toast.success('Standard exported successfully');
    } catch (error) {
      toast.error('Failed to export standard');
      console.error(error);
    }
  };

  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    toast.success('Code copied to clipboard');
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" mt={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (!standard) {
    return (
      <Box>
        <Typography variant="h5">Standard not found</Typography>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/standards')}>
          Back to Standards
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/standards')}>
          Back to Standards
        </Button>
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={handleExport}
        >
          Export
        </Button>
      </Box>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          {standard.title}
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          {standard.description}
        </Typography>
        
        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid size={{xs: 12, sm: 6}}>
            <Typography variant="subtitle2" color="text.secondary">
              Category
            </Typography>
            <Typography variant="body1">{standard.category}</Typography>
          </Grid>
          <Grid size={{xs: 12, sm: 6}}>
            <Typography variant="subtitle2" color="text.secondary">
              Subcategory
            </Typography>
            <Typography variant="body1">{standard.subcategory}</Typography>
          </Grid>
          <Grid size={{xs: 12, sm: 6}}>
            <Typography variant="subtitle2" color="text.secondary">
              Priority
            </Typography>
            <Chip label={standard.priority} color="primary" size="small" />
          </Grid>
          <Grid size={{xs: 12, sm: 6}}>
            <Typography variant="subtitle2" color="text.secondary">
              Version
            </Typography>
            <Typography variant="body1">{standard.version}</Typography>
          </Grid>
        </Grid>

        <Box mt={3}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Tags
          </Typography>
          <Box>
            {standard.tags.map((tag) => (
              <Chip key={tag} label={tag} sx={{ mr: 0.5, mb: 0.5 }} />
            ))}
          </Box>
        </Box>
      </Paper>

      <Paper>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
          <Tab label="Examples" />
          <Tab label="Rules" />
          <Tab label="Metadata" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          {standard.examples.length === 0 ? (
            <Typography color="text.secondary">No examples available</Typography>
          ) : (
            <Grid container spacing={2}>
              {standard.examples.map((example, index) => (
                <Grid size={{xs: 12}} key={index}>
                  <Card>
                    <CardContent>
                      {example.title && (
                        <Typography variant="h6" gutterBottom>
                          {example.title}
                        </Typography>
                      )}
                      {example.description && (
                        <Typography variant="body2" color="text.secondary" paragraph>
                          {example.description}
                        </Typography>
                      )}
                      <Box position="relative">
                        <Button
                          size="small"
                          startIcon={<ContentCopyIcon />}
                          onClick={() => handleCopyCode(example.code)}
                          sx={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}
                        >
                          Copy
                        </Button>
                        <SyntaxHighlighter
                          language={example.language || 'text'}
                          style={vscDarkPlus}
                          customStyle={{ borderRadius: 4 }}
                        >
                          {example.code}
                        </SyntaxHighlighter>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <SyntaxHighlighter
            language="json"
            style={vscDarkPlus}
            customStyle={{ borderRadius: 4 }}
          >
            {JSON.stringify(standard.rules, null, 2)}
          </SyntaxHighlighter>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <SyntaxHighlighter
            language="json"
            style={vscDarkPlus}
            customStyle={{ borderRadius: 4 }}
          >
            {JSON.stringify(standard.metadata, null, 2)}
          </SyntaxHighlighter>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default StandardDetail;
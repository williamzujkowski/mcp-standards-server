import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Card,
  CardContent,
  Chip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import { StandardsService } from '../services/standardsService';
import { ProjectContext, Recommendation } from '../types';

const RuleTesting: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [projectContext, setProjectContext] = useState<ProjectContext>({
    languages: [],
    frameworks: [],
    project_type: '',
    deployment_target: '',
    team_size: '',
    compliance_requirements: [],
    existing_tools: [],
    performance_requirements: {},
    security_requirements: {},
    scalability_requirements: {},
  });

  const steps = [
    {
      label: 'Project Information',
      description: 'Basic project details',
    },
    {
      label: 'Technical Stack',
      description: 'Languages, frameworks, and tools',
    },
    {
      label: 'Requirements',
      description: 'Performance, security, and compliance needs',
    },
    {
      label: 'Analysis Results',
      description: 'Recommended standards',
    },
  ];

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setProjectContext({
      languages: [],
      frameworks: [],
      project_type: '',
      deployment_target: '',
      team_size: '',
      compliance_requirements: [],
      existing_tools: [],
      performance_requirements: {},
      security_requirements: {},
      scalability_requirements: {},
    });
    setRecommendations([]);
  };

  const handleAnalyze = async () => {
    try {
      setLoading(true);
      const results = await StandardsService.analyzeProject(projectContext);
      setRecommendations(results);
      handleNext();
      toast.success(`Found ${results.length} relevant standards`);
    } catch (error) {
      toast.error('Analysis failed');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const updateContext = (field: keyof ProjectContext, value: any) => {
    setProjectContext({
      ...projectContext,
      [field]: value,
    });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Rule Testing
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Test the rule engine with your project context
      </Typography>

      <Paper sx={{ p: 3 }}>
        <Stepper activeStep={activeStep} orientation="vertical">
          <Step>
            <StepLabel>{steps[0].label}</StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary" paragraph>
                {steps[0].description}
              </Typography>
              <Grid container spacing={2}>
                <Grid size={{xs: 12, sm: 6}}>
                  <FormControl fullWidth>
                    <InputLabel>Project Type</InputLabel>
                    <Select
                      value={projectContext.project_type}
                      onChange={(e) => updateContext('project_type', e.target.value)}
                      label="Project Type"
                    >
                      <MenuItem value="web">Web Application</MenuItem>
                      <MenuItem value="api">API Service</MenuItem>
                      <MenuItem value="mobile">Mobile App</MenuItem>
                      <MenuItem value="desktop">Desktop Application</MenuItem>
                      <MenuItem value="library">Library/Framework</MenuItem>
                      <MenuItem value="microservice">Microservice</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid size={{xs: 12, sm: 6}}>
                  <FormControl fullWidth>
                    <InputLabel>Deployment Target</InputLabel>
                    <Select
                      value={projectContext.deployment_target}
                      onChange={(e) => updateContext('deployment_target', e.target.value)}
                      label="Deployment Target"
                    >
                      <MenuItem value="cloud">Cloud (AWS, GCP, Azure)</MenuItem>
                      <MenuItem value="kubernetes">Kubernetes</MenuItem>
                      <MenuItem value="serverless">Serverless</MenuItem>
                      <MenuItem value="on-premise">On-Premise</MenuItem>
                      <MenuItem value="hybrid">Hybrid</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid size={{xs: 12, sm: 6}}>
                  <FormControl fullWidth>
                    <InputLabel>Team Size</InputLabel>
                    <Select
                      value={projectContext.team_size}
                      onChange={(e) => updateContext('team_size', e.target.value)}
                      label="Team Size"
                    >
                      <MenuItem value="solo">Solo Developer</MenuItem>
                      <MenuItem value="small">Small (2-5)</MenuItem>
                      <MenuItem value="medium">Medium (6-20)</MenuItem>
                      <MenuItem value="large">Large (20+)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
              <Box mt={2}>
                <Button variant="contained" onClick={handleNext}>
                  Continue
                </Button>
              </Box>
            </StepContent>
          </Step>

          <Step>
            <StepLabel>{steps[1].label}</StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary" paragraph>
                {steps[1].description}
              </Typography>
              <Grid container spacing={2}>
                <Grid size={{xs: 12}}>
                  <TextField
                    fullWidth
                    label="Programming Languages"
                    placeholder="e.g., Python, JavaScript, Go"
                    value={projectContext.languages.join(', ')}
                    onChange={(e) =>
                      updateContext(
                        'languages',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                    }
                    helperText="Comma-separated list"
                  />
                </Grid>
                <Grid size={{xs: 12}}>
                  <TextField
                    fullWidth
                    label="Frameworks"
                    placeholder="e.g., React, FastAPI, Express"
                    value={projectContext.frameworks.join(', ')}
                    onChange={(e) =>
                      updateContext(
                        'frameworks',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                    }
                    helperText="Comma-separated list"
                  />
                </Grid>
                <Grid size={{xs: 12}}>
                  <TextField
                    fullWidth
                    label="Existing Tools"
                    placeholder="e.g., Docker, Kubernetes, Jenkins"
                    value={projectContext.existing_tools.join(', ')}
                    onChange={(e) =>
                      updateContext(
                        'existing_tools',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                    }
                    helperText="Comma-separated list"
                  />
                </Grid>
              </Grid>
              <Box mt={2}>
                <Button onClick={handleBack} sx={{ mr: 1 }}>
                  Back
                </Button>
                <Button variant="contained" onClick={handleNext}>
                  Continue
                </Button>
              </Box>
            </StepContent>
          </Step>

          <Step>
            <StepLabel>{steps[2].label}</StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary" paragraph>
                {steps[2].description}
              </Typography>
              <Grid container spacing={2}>
                <Grid size={{xs: 12}}>
                  <TextField
                    fullWidth
                    label="Compliance Requirements"
                    placeholder="e.g., GDPR, HIPAA, SOC2"
                    value={projectContext.compliance_requirements.join(', ')}
                    onChange={(e) =>
                      updateContext(
                        'compliance_requirements',
                        e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
                      )
                    }
                    helperText="Comma-separated list"
                  />
                </Grid>
              </Grid>
              <Box mt={2}>
                <Button onClick={handleBack} sx={{ mr: 1 }}>
                  Back
                </Button>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  Analyze Project
                </Button>
              </Box>
              {loading && <LinearProgress sx={{ mt: 2 }} />}
            </StepContent>
          </Step>

          <Step>
            <StepLabel>{steps[3].label}</StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary" paragraph>
                {steps[3].description}
              </Typography>
              
              {recommendations.length === 0 ? (
                <Alert severity="info">No recommendations found</Alert>
              ) : (
                <Box>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Found {recommendations.length} relevant standards for your project
                  </Alert>
                  
                  {recommendations.map((rec, index) => (
                    <Accordion key={index} defaultExpanded={index === 0}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box
                          display="flex"
                          alignItems="center"
                          justifyContent="space-between"
                          width="100%"
                          pr={2}
                        >
                          <Box display="flex" alignItems="center">
                            <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                            <Typography>{rec.standard.title}</Typography>
                          </Box>
                          <Box display="flex" gap={1}>
                            <Chip
                              label={`Relevance: ${(rec.relevance_score * 100).toFixed(0)}%`}
                              color="primary"
                              size="small"
                            />
                            <Chip
                              label={`Confidence: ${(rec.confidence * 100).toFixed(0)}%`}
                              color="secondary"
                              size="small"
                            />
                          </Box>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          {rec.standard.description}
                        </Typography>
                        
                        <Box display="flex" alignItems="center" mb={2}>
                          <InfoIcon color="primary" sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Reasoning</Typography>
                        </Box>
                        <Typography variant="body2" paragraph>
                          {rec.reasoning}
                        </Typography>
                        
                        <Box display="flex" alignItems="center" mb={2}>
                          <InfoIcon color="secondary" sx={{ mr: 1 }} />
                          <Typography variant="subtitle2">Implementation Notes</Typography>
                        </Box>
                        <Typography variant="body2">
                          {rec.implementation_notes}
                        </Typography>
                        
                        <Box mt={2}>
                          <Chip label={rec.standard.category} size="small" sx={{ mr: 0.5 }} />
                          {rec.standard.tags.slice(0, 3).map((tag) => (
                            <Chip
                              key={tag}
                              label={tag}
                              size="small"
                              variant="outlined"
                              sx={{ mr: 0.5 }}
                            />
                          ))}
                        </Box>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                </Box>
              )}
              
              <Box mt={2}>
                <Button onClick={handleReset}>Start New Analysis</Button>
              </Box>
            </StepContent>
          </Step>
        </Stepper>
      </Paper>
    </Box>
  );
};

export default RuleTesting;
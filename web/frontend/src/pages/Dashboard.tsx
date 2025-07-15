import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  LinearProgress,
  Chip,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Category as CategoryIcon,
  Tag as TagIcon,
  Update as UpdateIcon,
} from '@mui/icons-material';
import { useStandards } from '../contexts/StandardsContext';

const Dashboard: React.FC = () => {
  const { standards, categories, loading } = useStandards();

  if (loading) {
    return <LinearProgress />;
  }

  const totalStandards = Object.values(standards).flat().length;
  const totalCategories = categories.length;
  const allTags = new Set<string>();
  Object.values(standards).flat().forEach(s => s.tags.forEach(t => allTags.add(t)));
  const totalTags = allTags.size;

  const stats = [
    {
      title: 'Total Standards',
      value: totalStandards,
      icon: <TrendingUpIcon />,
      color: '#1976d2',
    },
    {
      title: 'Categories',
      value: totalCategories,
      icon: <CategoryIcon />,
      color: '#388e3c',
    },
    {
      title: 'Unique Tags',
      value: totalTags,
      icon: <TagIcon />,
      color: '#f57c00',
    },
    {
      title: 'Last Updated',
      value: 'Today',
      icon: <UpdateIcon />,
      color: '#7b1fa2',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Overview of standards and compliance requirements
      </Typography>

      <Grid container spacing={3}>
        {stats.map((stat, index) => (
          <Grid size={{xs: 12, sm: 6, md: 3}} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 48,
                      height: 48,
                      borderRadius: '50%',
                      backgroundColor: `${stat.color}20`,
                      color: stat.color,
                      mr: 2,
                    }}
                  >
                    {stat.icon}
                  </Box>
                  <Box>
                    <Typography color="text.secondary" variant="body2">
                      {stat.title}
                    </Typography>
                    <Typography variant="h5">{stat.value}</Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}

        <Grid size={{xs: 12, md: 8}}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Standards by Category
            </Typography>
            <Box mt={2}>
              {Object.entries(standards).map(([category, categoryStandards]) => (
                <Box key={category} mb={2}>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Typography variant="subtitle1">{category}</Typography>
                    <Chip label={categoryStandards.length} size="small" />
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={(categoryStandards.length / totalStandards) * 100}
                    sx={{ mt: 1, height: 8, borderRadius: 4 }}
                  />
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        <Grid size={{xs: 12, md: 4}}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Popular Tags
            </Typography>
            <Box mt={2}>
              {Array.from(allTags).slice(0, 10).map((tag) => (
                <Chip
                  key={tag}
                  label={tag}
                  sx={{ m: 0.5 }}
                  onClick={() => {
                    // Navigate to search with tag filter
                  }}
                />
              ))}
            </Box>
          </Paper>
        </Grid>

        <Grid size={{xs: 12}}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <Typography variant="body2" color="text.secondary">
              No recent activity to display
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
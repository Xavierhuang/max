import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  FormControl, 
  InputLabel, 
  MenuItem, 
  Select,
  FormHelperText,
  TextField, 
  Grid, 
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InsightsIcon from '@mui/icons-material/Insights';
import AssessmentIcon from '@mui/icons-material/Assessment';
import axios from 'axios';

const ReportGenerator = ({ selectedDataset, disabled }) => {
  const [reportType, setReportType] = useState('summary');
  const [predictionColumn, setPredictionColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState([]);
  const [predictValue, setPredictValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState(null);
  const [numericColumns, setNumericColumns] = useState([]);
  const [columns, setColumns] = useState([]);
  
  useEffect(() => {
    // If we have a selected dataset, fetch its columns
    if (selectedDataset) {
      fetchDatasetColumns();
    }
  }, [selectedDataset]);
  
  const fetchDatasetColumns = async () => {
    try {
      const response = await axios.get(`/datasets/${selectedDataset}`);
      if (response.data && response.data.columns) {
        setColumns(response.data.columns);
      }
    } catch (err) {
      console.error("Error fetching dataset columns:", err);
    }
  };
  
  useEffect(() => {
    // Filter for numeric columns
    if (columns && columns.length > 0) {
      console.log("Columns received in ReportGenerator:", columns);
      const numeric = columns.filter(col => {
        // Check if it's a numeric column by name or type
        if (typeof col === 'object' && col.type) {
          return ['number', 'integer', 'float', 'Int64', 'Float64'].includes(col.type);
        }
        return typeof col === 'string'; // For simple string column names, assume it could be numeric
      });
      
      // Map columns to their proper format for the dropdown
      const mappedNumericColumns = numeric.map(col => 
        typeof col === 'object' ? col.field : col
      );
      
      setNumericColumns(mappedNumericColumns);
    }
  }, [columns]);
  
  const handleFeatureColumnChange = (event) => {
    const {
      target: { value },
    } = event;
    setFeatureColumns(
      typeof value === 'string' ? value.split(',') : value,
    );
  };
  
  const generateReport = async () => {
    if (!selectedDataset) {
      setError('No dataset selected');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(
        `/api/generate-report?dataset_id=${selectedDataset}&report_type=${reportType}${predictionColumn ? `&target_column=${predictionColumn}` : ''}`,
        {}
      );
      
      console.log('Report response:', response.data);
      if (response.data.success) {
        setReport(response.data.report);
      } else {
        setError(response.data.error || 'Failed to generate report');
      }
    } catch (err) {
      console.error('Error generating report:', err);
      const errorMessage = err.response?.data?.error || err.response?.data?.detail || err.message;
      
      // Show more specific messages for different errors
      if (errorMessage.includes('OpenAI') || errorMessage.includes('API key')) {
        setError(`OpenAI API error: ${errorMessage}. AI insights will not be available.`);
      } else if (err.response?.status === 404) {
        setError('Dataset not found. Please select another dataset or reload the application.');
      } else if (err.response?.status === 500) {
        setError(`Server error: ${errorMessage}. Please check the backend logs.`);
      } else if (!navigator.onLine) {
        setError('Network connection lost. Please check your internet connection.');
      } else {
        setError(`Error generating report: ${errorMessage}`);
      }
    } finally {
      setLoading(false);
    }
  };
  
  const exportReport = () => {
    if (!report) return;
    
    // Create a blob with the report data
    const reportText = generateReportText();
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    // Create a link and click it to trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `data-report-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  const generateReportText = () => {
    if (!report) return '';
    
    let text = `DATA REPORT - ${new Date().toLocaleString()}\n\n`;
    
    if (reportType === 'summary') {
      text += `DATASET SUMMARY\n`;
      text += `Total Rows: ${report.summary?.row_count || 'N/A'}\n`;
      text += `Total Columns: ${report.summary?.column_count || 'N/A'}\n\n`;
      
      if (report.summary?.column_stats) {
        text += `COLUMN STATISTICS\n`;
        Object.keys(report.summary.column_stats).forEach(col => {
          const stats = report.summary.column_stats[col];
          text += `\n${col}:\n`;
          text += `  Data Type: ${stats.data_type || 'N/A'}\n`;
          text += `  Missing Values: ${stats.missing_count || 0} (${stats.missing_percentage || '0'}%)\n`;
          if (stats.numeric) {
            text += `  Mean: ${stats.mean?.toFixed(4) || 'N/A'}\n`;
            text += `  Std Dev: ${stats.std?.toFixed(4) || 'N/A'}\n`;
            text += `  Min: ${stats.min?.toFixed(4) || 'N/A'}\n`;
            text += `  Max: ${stats.max?.toFixed(4) || 'N/A'}\n`;
          } else {
            text += `  Unique Values: ${stats.unique_count || 'N/A'}\n`;
            text += `  Most Common: ${stats.most_common || 'N/A'}\n`;
          }
        });
      }
    } else if (reportType === 'prediction') {
      text += `PREDICTION REPORT\n\n`;
      text += `Target Column: ${predictionColumn}\n`;
      text += `Feature Columns: ${featureColumns.join(', ')}\n\n`;
      
      if (report.prediction) {
        text += `MODEL PERFORMANCE\n`;
        text += `R-squared: ${report.prediction.r_squared?.toFixed(4) || 'N/A'}\n`;
        text += `Mean Absolute Error: ${report.prediction.mae?.toFixed(4) || 'N/A'}\n\n`;
        
        text += `FEATURE IMPORTANCE\n`;
        if (report.prediction.feature_importance) {
          Object.keys(report.prediction.feature_importance).forEach(feature => {
            text += `${feature}: ${report.prediction.feature_importance[feature].toFixed(4)}\n`;
          });
        }
        
        if (report.prediction.prediction_value !== undefined) {
          text += `\nPREDICTED VALUE: ${report.prediction.prediction_value.toFixed(4)}\n`;
        }
      }
    }
    
    if (report.insights && report.insights.length > 0) {
      text += `\nINSIGHTS\n`;
      report.insights.forEach((insight, index) => {
        text += `${index + 1}. ${insight}\n`;
      });
    }
    
    return text;
  };
  
  const renderReportOptions = () => {
    switch (reportType) {
      case 'summary':
        return (
          <Typography variant="body2" color="text.secondary">
            Generates a comprehensive summary of the dataset including statistics for each column, missing value analysis, and data distribution insights.
          </Typography>
        );
        
      case 'prediction':
        return (
          <Box>
            <FormControl fullWidth margin="normal">
              <InputLabel id="prediction-column-label">Target Column (Y)</InputLabel>
              <Select
                labelId="prediction-column-label"
                value={predictionColumn}
                label="Target Column (Y)"
                onChange={(e) => setPredictionColumn(e.target.value)}
              >
                {numericColumns && numericColumns.length > 0 ? (
                  numericColumns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>No numeric columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The column you want to predict</FormHelperText>
            </FormControl>
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="feature-columns-label">Feature Columns (X)</InputLabel>
              <Select
                labelId="feature-columns-label"
                multiple
                value={featureColumns}
                label="Feature Columns (X)"
                onChange={handleFeatureColumnChange}
              >
                {numericColumns && numericColumns.length > 0 ? (
                  numericColumns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>No numeric columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The columns to use for making predictions</FormHelperText>
            </FormControl>
            
            <TextField
              fullWidth
              margin="normal"
              label="Predict Value (Optional)"
              value={predictValue}
              onChange={(e) => setPredictValue(e.target.value)}
              helperText="Enter comma-separated values for the features to make a prediction (optional)"
            />
          </Box>
        );
        
      default:
        return null;
    }
  };
  
  const renderReport = () => {
    if (!report) return null;
    
    return (
      <Box mt={3}>
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {report.report_type === 'summary' ? 'Dataset Summary' : 'Prediction Model Results'}
            </Typography>
            
            {report.report_type === 'summary' && (
              <>
                <Typography variant="subtitle1">Dataset Information</Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Rows:</strong> {report.data_shape.rows}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Columns:</strong> {report.data_shape.columns}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2">
                      <strong>Missing Values:</strong> {Object.values(report.missing_values).reduce((a, b) => a + b, 0)}
                    </Typography>
                  </Grid>
                </Grid>
              </>
            )}
            
            {report.report_type === 'prediction' && (
              <>
                <Typography variant="subtitle1">Model Information</Typography>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Target:</strong> {report.target_column}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>Model Type:</strong> {report.model_type}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>R² Score:</strong> {report.metrics?.r2_score.toFixed(4)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      <strong>RMSE:</strong> {report.metrics?.rmse.toFixed(4)}
                    </Typography>
                  </Grid>
                </Grid>
              </>
            )}
          </CardContent>
        </Card>
        
        {/* AI Insights Section */}
        {report.ai_insights && (
          <Card variant="outlined" sx={{ mb: 2, backgroundColor: 'rgba(25, 118, 210, 0.05)' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI-Generated Insights
              </Typography>
              <Box sx={{ 
                maxHeight: '400px', 
                overflowY: 'auto',
                whiteSpace: 'pre-wrap',
                pr: 1
              }}>
                <Typography variant="body2" component="div" sx={{ whiteSpace: 'pre-line' }}>
                  {report.ai_insights}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Visualizations Section */}
        {report.visualizations && report.visualizations.length > 0 && (
          <Card variant="outlined" sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Visualizations
              </Typography>
              <Grid container spacing={3}>
                {report.visualizations.map((visualization, index) => (
                  <Grid item xs={12} md={6} key={index}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom align="center">
                          {visualization.title}
                        </Typography>
                        <Box
                          component="img"
                          src={`data:image/png;base64,${visualization.image}`}
                          alt={visualization.title}
                          sx={{
                            maxWidth: '100%',
                            height: 'auto',
                            display: 'block',
                            margin: '0 auto',
                          }}
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        )}
      </Box>
    );
  };

  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" gutterBottom>
        Report Generator
      </Typography>
      
      {!selectedDataset ? (
        <Alert severity="info">Please select a dataset to generate reports</Alert>
      ) : (
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <FormControl fullWidth margin="normal" disabled={disabled || loading}>
            <InputLabel id="report-type-label">Report Type</InputLabel>
            <Select
              labelId="report-type-label"
              value={reportType}
              label="Report Type"
              onChange={(e) => setReportType(e.target.value)}
            >
              <MenuItem value="summary">
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <DescriptionIcon sx={{ mr: 1 }} />
                  Dataset Summary
                </Box>
              </MenuItem>
              <MenuItem value="prediction">
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TrendingUpIcon sx={{ mr: 1 }} />
                  Predictive Model
                </Box>
              </MenuItem>
            </Select>
          </FormControl>
          
          {renderReportOptions()}
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          
          <Button 
            variant="contained" 
            color="primary" 
            onClick={generateReport} 
            disabled={disabled || loading || !selectedDataset || 
              (reportType === 'prediction' && (!predictionColumn || !featureColumns.length))}
            sx={{ mt: 2 }}
            startIcon={<AssessmentIcon />}
          >
            {loading ? <CircularProgress size={24} /> : 'Generate Report'}
          </Button>
          
          <Box sx={{ flexGrow: 1, overflowY: 'auto', mt: 2 }}>
            {renderReport()}
          </Box>
        </Box>
      )}
    </Paper>
  );
};

export default ReportGenerator; 
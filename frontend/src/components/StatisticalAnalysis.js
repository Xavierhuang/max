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
  TextField, 
  Grid, 
  Chip,
  Alert,
  FormHelperText,
  CircularProgress,
  Card,
  CardContent,
  CardMedia,
  Divider
} from '@mui/material';
import axios from 'axios';

const StatisticalAnalysis = ({ dataset, columns }) => {
  const [analysisType, setAnalysisType] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState([]);
  const [groupColumn, setGroupColumn] = useState('');
  const [valueColumn, setValueColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [numericColumns, setNumericColumns] = useState([]);
  
  useEffect(() => {
    // Filter for numeric columns only
    if (columns && columns.length > 0) {
      console.log("Columns received:", columns);
      const numeric = columns.filter(col => 
        (typeof col === 'object' && col.type && (
          col.type === 'number' || 
          col.type === 'integer' || 
          col.type === 'float' || 
          col.type === 'Int64' ||
          col.type === 'Float64'
        )) || (
          // Handle simple string column names as well
          typeof col === 'string' && !isNaN(Number(dataset.data[0]?.[col]))
        )
      );
      
      // Map columns to their proper format for the dropdown
      const mappedNumericColumns = numeric.map(col => 
        typeof col === 'object' ? col.field : col
      );
      
      console.log("Numeric columns:", mappedNumericColumns);
      setNumericColumns(mappedNumericColumns);
    } else {
      console.log("No columns received or columns array is empty");
    }
  }, [columns, dataset]);
  
  // Log the dataset to see what we're working with
  useEffect(() => {
    console.log("Dataset received:", dataset);
  }, [dataset]);
  
  const handleAnalysisTypeChange = (event) => {
    setAnalysisType(event.target.value);
    // Reset other fields when analysis type changes
    setTargetColumn('');
    setFeatureColumns([]);
    setGroupColumn('');
    setValueColumn('');
    setResults(null);
    setError('');
  };
  
  const handleFeatureColumnChange = (event) => {
    const {
      target: { value },
    } = event;
    setFeatureColumns(
      typeof value === 'string' ? value.split(',') : value,
    );
  };
  
  const runAnalysis = async () => {
    if (!dataset || !dataset.dataset_id) {
      setError('No dataset selected');
      console.error('No dataset or dataset_id provided');
      return;
    }
    
    console.log("Attempting analysis with dataset_id:", dataset.dataset_id);
    console.log("Full dataset object:", dataset);
    
    if (!analysisType) {
      setError('Please select an analysis type');
      return;
    }
    
    // Validate required fields based on analysis type
    if (analysisType === 'regression' && (!targetColumn || featureColumns.length === 0)) {
      setError('For regression analysis, you must select a target column and at least one feature column');
      return;
    }
    
    if (analysisType === 'comparison' && (!groupColumn || !valueColumn)) {
      setError('For comparison analysis, you must select a group column and a value column');
      return;
    }
    
    setLoading(true);
    setError('');
    
    const requestBody = {
      analysis_type: analysisType,
      target_column: targetColumn || undefined,
      feature_columns: featureColumns.length > 0 ? featureColumns : undefined,
      group_column: groupColumn || undefined,
      value_column: valueColumn || undefined
    };
    
    console.log(`Sending analysis request for dataset ${dataset.dataset_id}:`, requestBody);
    
    try {
      const response = await axios.post(
        `http://localhost:8000/analyze/${dataset.dataset_id}`,
        requestBody
      );
      
      console.log('Analysis response:', response.data);
      setResults(response.data);
      
      if (response.data.error) {
        setError(response.data.error);
      }
    } catch (err) {
      console.error('Error running analysis:', err);
      let errorMessage = 'An error occurred while running the analysis';
      
      if (err.response) {
        // The request was made and the server responded with a status code
        console.error('Error response:', err.response.data);
        errorMessage = err.response.data.detail || err.response.data.error || errorMessage;
      } else if (err.request) {
        // The request was made but no response was received
        console.error('Error request:', err.request);
        errorMessage = 'No response received from server. Is the backend running?';
      } 
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };
  
  const renderAnalysisOptions = () => {
    switch (analysisType) {
      case 'descriptive':
        return (
          <Typography variant="body2" color="text.secondary">
            Descriptive statistics will be calculated for all numeric columns in the dataset.
          </Typography>
        );
        
      case 'correlation':
        return (
          <Typography variant="body2" color="text.secondary">
            A correlation matrix will be generated for all numeric columns in the dataset.
          </Typography>
        );
        
      case 'regression':
        return (
          <Box>
            <FormControl fullWidth margin="normal">
              <InputLabel id="target-column-label">Target Column (Y)</InputLabel>
              <Select
                labelId="target-column-label"
                value={targetColumn}
                label="Target Column (Y)"
                onChange={(e) => setTargetColumn(e.target.value)}
              >
                {numericColumns && numericColumns.length > 0 ? (
                  numericColumns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>No numeric columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The dependent variable you want to predict</FormHelperText>
            </FormControl>
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="feature-columns-label">Feature Columns (X)</InputLabel>
              <Select
                labelId="feature-columns-label"
                multiple
                value={featureColumns}
                label="Feature Columns (X)"
                onChange={handleFeatureColumnChange}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {numericColumns && numericColumns.length > 0 ? (
                  numericColumns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>No numeric columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The independent variables used to make predictions</FormHelperText>
            </FormControl>
          </Box>
        );
        
      case 'comparison':
        return (
          <Box>
            <FormControl fullWidth margin="normal">
              <InputLabel id="group-column-label">Group Column</InputLabel>
              <Select
                labelId="group-column-label"
                value={groupColumn}
                label="Group Column"
                onChange={(e) => setGroupColumn(e.target.value)}
              >
                {columns && columns.length > 0 ? (
                  columns.map((col) => {
                    const colName = typeof col === 'object' ? col.field : col;
                    return <MenuItem key={colName} value={colName}>{colName}</MenuItem>;
                  })
                ) : (
                  <MenuItem disabled>No columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The column containing group labels (e.g., categories, treatment groups)</FormHelperText>
            </FormControl>
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="value-column-label">Value Column</InputLabel>
              <Select
                labelId="value-column-label"
                value={valueColumn}
                label="Value Column"
                onChange={(e) => setValueColumn(e.target.value)}
              >
                {numericColumns && numericColumns.length > 0 ? (
                  numericColumns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))
                ) : (
                  <MenuItem disabled>No numeric columns found</MenuItem>
                )}
              </Select>
              <FormHelperText>The numeric column to compare across groups</FormHelperText>
            </FormControl>
          </Box>
        );
        
      default:
        return null;
    }
  };
  
  const renderResults = () => {
    if (!results) return null;
    
    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Analysis Results
        </Typography>
        
        {results.error ? (
          <Alert severity="error">{results.error}</Alert>
        ) : (
          <Box>
            {analysisType === 'descriptive' && results.result.summary && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Summary Statistics
                </Typography>
                <Box sx={{ overflowX: 'auto' }}>
                  <table className="table table-bordered table-sm">
                    <thead>
                      <tr>
                        <th>Statistic</th>
                        {Object.keys(results.result.summary).map(col => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map(stat => (
                        <tr key={stat}>
                          <td><strong>{stat}</strong></td>
                          {Object.keys(results.result.summary).map(col => (
                            <td key={`${col}-${stat}`}>
                              {typeof results.result.summary[col][stat] === 'number' 
                                ? results.result.summary[col][stat].toFixed(4) 
                                : results.result.summary[col][stat]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              </Box>
            )}
            
            {analysisType === 'correlation' && results.result.correlation_matrix && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Correlation Matrix
                </Typography>
                <Box sx={{ overflowX: 'auto' }}>
                  <table className="table table-bordered table-sm">
                    <thead>
                      <tr>
                        <th></th>
                        {Object.keys(results.result.correlation_matrix).map(col => (
                          <th key={col}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(results.result.correlation_matrix).map(row => (
                        <tr key={row}>
                          <td><strong>{row}</strong></td>
                          {Object.keys(results.result.correlation_matrix[row]).map(col => (
                            <td key={`${row}-${col}`} 
                                style={{
                                  backgroundColor: `rgba(${results.result.correlation_matrix[row][col] >= 0 ? '0, 0, 255' : '255, 0, 0'}, ${Math.abs(results.result.correlation_matrix[row][col]) * 0.5})`,
                                  color: Math.abs(results.result.correlation_matrix[row][col]) > 0.7 ? 'white' : 'black'
                                }}>
                              {results.result.correlation_matrix[row][col].toFixed(4)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Box>
              </Box>
            )}
            
            {analysisType === 'regression' && results.result.summary && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Regression Results
                </Typography>
                {results.result.summary.data_cleaning && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Data cleaning:</strong> {results.result.summary.data_cleaning.rows_removed} rows with missing or infinite values were removed.
                      Using {results.result.summary.data_cleaning.rows_used} out of {results.result.summary.data_cleaning.total_rows} total rows.
                    </Typography>
                  </Alert>
                )}
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>Model Fit</Typography>
                        <Typography><strong>R-squared:</strong> {results.result.summary.r_squared.toFixed(4)}</Typography>
                        <Typography><strong>Adjusted R-squared:</strong> {results.result.summary.adj_r_squared.toFixed(4)}</Typography>
                        <Typography><strong>F-statistic:</strong> {results.result.summary.f_statistic.toFixed(4)}</Typography>
                        <Typography><strong>p-value:</strong> {results.result.summary.p_value.toExponential(4)}</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="h6" gutterBottom>Coefficients</Typography>
                        <Box sx={{ overflowX: 'auto' }}>
                          <table className="table table-bordered table-sm">
                            <thead>
                              <tr>
                                <th>Variable</th>
                                <th>Coefficient</th>
                                <th>Std Error</th>
                                <th>p-value</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.keys(results.result.summary.coefficients).map(variable => (
                                <tr key={variable}>
                                  <td>{variable}</td>
                                  <td>{results.result.summary.coefficients[variable].toFixed(4)}</td>
                                  <td>{results.result.summary.std_errors[variable].toFixed(4)}</td>
                                  <td>{results.result.summary.p_values[variable].toExponential(4)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {analysisType === 'comparison' && (results.result.t_test || results.result.anova) && (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Comparison Results
                </Typography>
                
                {results.result.t_test && (
                  <Card variant="outlined" sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>T-Test Results</Typography>
                      <Typography><strong>t-statistic:</strong> {results.result.t_test.t_statistic.toFixed(4)}</Typography>
                      <Typography><strong>p-value:</strong> {results.result.t_test.p_value.toExponential(4)}</Typography>
                      <Divider sx={{ my: 2 }} />
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Group 1</Typography>
                          <Typography><strong>Mean:</strong> {results.result.t_test.group1_mean.toFixed(4)}</Typography>
                          <Typography><strong>Std Dev:</strong> {results.result.t_test.group1_std.toFixed(4)}</Typography>
                          <Typography><strong>Count:</strong> {results.result.t_test.group1_count}</Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Group 2</Typography>
                          <Typography><strong>Mean:</strong> {results.result.t_test.group2_mean.toFixed(4)}</Typography>
                          <Typography><strong>Std Dev:</strong> {results.result.t_test.group2_std.toFixed(4)}</Typography>
                          <Typography><strong>Count:</strong> {results.result.t_test.group2_count}</Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                )}
                
                {results.result.anova && (
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>ANOVA Results</Typography>
                      <Typography><strong>F-statistic:</strong> {results.result.anova.f_statistic.toFixed(4)}</Typography>
                      <Typography><strong>p-value:</strong> {results.result.anova.p_value.toExponential(4)}</Typography>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2" gutterBottom>Group Statistics</Typography>
                      <Grid container spacing={2}>
                        {Object.keys(results.result.anova.group_stats).map(group => (
                          <Grid item xs={12} sm={6} md={4} key={group}>
                            <Card variant="outlined">
                              <CardContent>
                                <Typography variant="subtitle2">{group}</Typography>
                                <Typography><strong>Mean:</strong> {results.result.anova.group_stats[group].mean.toFixed(4)}</Typography>
                                <Typography><strong>Std Dev:</strong> {results.result.anova.group_stats[group].std.toFixed(4)}</Typography>
                                <Typography><strong>Count:</strong> {results.result.anova.group_stats[group].count}</Typography>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </CardContent>
                  </Card>
                )}
              </Box>
            )}
            
            {/* Display plots if available */}
            {results.plots && results.plots.length > 0 && (
              <Box sx={{ mt: 4 }}>
                <Typography variant="h6" gutterBottom>
                  Visualization
                </Typography>
                <Grid container spacing={2}>
                  {results.plots.map((plot, index) => (
                    <Grid item xs={12} md={6} key={index}>
                      <Card>
                        <CardMedia
                          component="img"
                          image={`data:image/png;base64,${plot}`}
                          alt={`Plot ${index + 1}`}
                        />
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            )}
          </Box>
        )}
      </Box>
    );
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3, mt: 2 }}>
      <Typography variant="h5" gutterBottom>
        Statistical Analysis
      </Typography>
      
      {!dataset?.dataset_id ? (
        <Alert severity="info">Please select a dataset to perform statistical analysis</Alert>
      ) : (
        <Box>
          <FormControl fullWidth margin="normal">
            <InputLabel id="analysis-type-label">Analysis Type</InputLabel>
            <Select
              labelId="analysis-type-label"
              value={analysisType}
              label="Analysis Type"
              onChange={handleAnalysisTypeChange}
            >
              <MenuItem value="descriptive">Descriptive Statistics</MenuItem>
              <MenuItem value="correlation">Correlation Analysis</MenuItem>
              <MenuItem value="regression">Regression Analysis</MenuItem>
              <MenuItem value="comparison">Group Comparison</MenuItem>
            </Select>
          </FormControl>
          
          {renderAnalysisOptions()}
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
          
          <Button 
            variant="contained" 
            color="primary" 
            onClick={runAnalysis} 
            disabled={loading || !analysisType || !dataset?.dataset_id}
            sx={{ mt: 2 }}
          >
            {loading ? <CircularProgress size={24} /> : 'Run Analysis'}
          </Button>
          
          {renderResults()}
        </Box>
      )}
    </Paper>
  );
};

export default StatisticalAnalysis; 
import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  CircularProgress,
  Snackbar,
  Alert,
  CssBaseline,
  ThemeProvider,
  createTheme,
  IconButton,
  Divider,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemButton,
  Badge,
  Switch,
  FormControlLabel
} from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import GridViewIcon from '@mui/icons-material/GridView';
import ViewSidebarIcon from '@mui/icons-material/ViewSidebar';
import CloseIcon from '@mui/icons-material/Close';
import axios from 'axios';
import DataGrid from './components/DataGrid';
import TransformationPanel from './components/TransformationPanel';
import FileUpload from './components/FileUpload';
import AIAssistant from './components/AIAssistant';
import VersionHistory from './components/VersionHistory';
import StatisticalAnalysis from './components/StatisticalAnalysis';
import ReportGenerator from './components/ReportGenerator';
import AddIcon from '@mui/icons-material/Add';
import SettingsIcon from '@mui/icons-material/Settings';
import AnalyticsIcon from '@mui/icons-material/Analytics';

function App() {
  const [darkMode, setDarkMode] = useState(localStorage.getItem('darkMode') === 'true');
  const [datasets, setDatasets] = useState([]);
  const [activeDatasetIndex, setActiveDatasetIndex] = useState(0);
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const [multiViewMode, setMultiViewMode] = useState(false);
  const [openDatasets, setOpenDatasets] = useState([]);
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [activeToolTab, setActiveToolTab] = useState(0);
  const [transformationHistory, setTransformationHistory] = useState({});
  const [transformAlert, setTransformAlert] = useState(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [isProcessingData, setIsProcessingData] = useState(false);
  const [isProcessingTransformation, setIsProcessingTransformation] = useState(false);

  // Create theme based on dark mode preference
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#f50057',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: '4px',
            boxShadow: darkMode ? '0 0 10px rgba(0, 0, 0, 0.5)' : '0 0 10px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  });

  // Save dark mode preference to localStorage
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  // Update active dataset when switching
  useEffect(() => {
    if (datasets.length > 0 && activeDatasetIndex < datasets.length) {
      setData(datasets[activeDatasetIndex].data || []);
      setColumns(datasets[activeDatasetIndex].columns || []);
    }
  }, [activeDatasetIndex, datasets]);

  // Effect to set the selectedDatasetId when active dataset changes
  useEffect(() => {
    if (datasets.length > 0 && activeDatasetIndex < datasets.length) {
      const currentDataset = datasets[activeDatasetIndex];
      if (currentDataset && currentDataset.dataset_id) {
        setSelectedDatasetId(currentDataset.dataset_id);
      }
    }
  }, [activeDatasetIndex, datasets]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const toggleMultiViewMode = () => {
    setMultiViewMode(!multiViewMode);
    
    // Initialize open datasets with the active dataset if none are open
    if (!multiViewMode && openDatasets.length === 0 && datasets.length > 0) {
      const datasetToOpen = datasets[activeDatasetIndex];
      if (datasetToOpen && datasetToOpen.status === 'success') {
        setOpenDatasets([activeDatasetIndex]);
      }
    }
  };

  const handleFileUpload = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { dataset_id, data, columns } = response.data;
      const newDataset = { 
        dataset_id, 
        data: data || [], 
        columns: columns || [],
        filename: file.name,
        status: 'success'
      };
      
      // Add new dataset and update active index
      const newIndex = datasets.length;
      setDatasets(prev => [...prev, newDataset]);
      setActiveDatasetIndex(newIndex);
      
      // Add to open datasets in multi-view mode
      if (multiViewMode) {
        const openDatasetsLength = Array.isArray(openDatasets) ? openDatasets.length : 0;
        setOpenDatasets(prev => [...(prev || []), newIndex]);
        setActiveTabIndex(openDatasetsLength);
      }
      
      setNotification({ open: true, message: `${file.name} uploaded successfully!`, severity: 'success' });
    } catch (error) {
      console.error('Error uploading file:', error);
      
      // Add failed upload to datasets list
      const newDataset = { 
        filename: file.name,
        status: 'error',
        error: error.response?.data?.detail || error.message
      };
      
      setDatasets(prev => [...prev, newDataset]);
      
      setNotification({ 
        open: true, 
        message: `Upload failed: ${error.response?.data?.detail || error.message}`, 
        severity: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTransform = async (operation) => {
    if (datasets.length === 0 || !datasets[activeDatasetIndex]?.dataset_id) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`http://localhost:8000/transform/${datasets[activeDatasetIndex].dataset_id}`, operation);
      
      const { data, columns, affected_rows } = response.data;
      
      // Add affected rows count to the operation
      const operationWithMetrics = {
        ...operation,
        affectedRows: affected_rows || 0
      };
      
      // Update transformation history
      setTransformationHistory(prev => {
        const datasetId = datasets[activeDatasetIndex].dataset_id;
        const currentHistory = prev[datasetId] || [];
        return {
          ...prev,
          [datasetId]: [...currentHistory, operationWithMetrics]
        };
      });
      
      // Show transform alert
      if (affected_rows > 0) {
        setTransformAlert({
          message: `Transformation affected ${affected_rows} rows!`,
          operation: getOperationDescription(operation)
        });
        
        // Clear alert after 5 seconds
        setTimeout(() => {
          setTransformAlert(null);
        }, 5000);
      }
      
      // Update the datasets array with new data and columns
      setDatasets(prev => {
        const updated = [...prev];
        if (updated[activeDatasetIndex]) {
          updated[activeDatasetIndex].data = data || [];
          updated[activeDatasetIndex].columns = columns || [];
        }
        return updated;
      });
      
      setData(data || []);
      setColumns(columns || []);
      setNotification({ open: true, message: 'Transformation applied successfully!', severity: 'success' });
    } catch (error) {
      console.error('Error applying transformation:', error);
      setNotification({ 
        open: true, 
        message: `Transformation failed: ${error.response?.data?.detail || error.message}`, 
        severity: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleUndoToVersion = async (versionIndex) => {
    if (datasets.length === 0 || !datasets[activeDatasetIndex]?.dataset_id) return;
    
    const datasetId = datasets[activeDatasetIndex].dataset_id;
    const history = transformationHistory[datasetId] || [];
    
    setLoading(true);
    try {
      // Clear all transformations (restore to original)
      const undoResponse = await axios.post(`http://localhost:8000/undo/${datasetId}`, { 
        restore_to_original: true 
      });
      
      // If rolling back to a specific version (not original)
      if (versionIndex >= 0 && versionIndex < history.length) {
        // Re-apply transformations up to the specified version
        for (let i = 0; i <= versionIndex; i++) {
          await axios.post(`http://localhost:8000/transform/${datasetId}`, history[i]);
        }
        
        // Update transformation history
        setTransformationHistory(prev => ({
          ...prev,
          [datasetId]: history.slice(0, versionIndex + 1)
        }));
      } else {
        // Reset transformation history for this dataset
        setTransformationHistory(prev => ({
          ...prev,
          [datasetId]: []
        }));
      }
      
      // Get the latest state of the data
      const response = await axios.get(`http://localhost:8000/datasets/${datasetId}`);
      
      // Update the datasets array with new data and columns
      setDatasets(prev => {
        const updated = [...prev];
        if (updated[activeDatasetIndex]) {
          updated[activeDatasetIndex].data = response.data.preview || [];
          updated[activeDatasetIndex].columns = response.data.columns || [];
        }
        return updated;
      });
      
      setData(response.data.preview || []);
      setColumns(response.data.columns || []);
      
      const versionName = versionIndex >= 0 ? `version ${versionIndex + 1}` : 'original dataset';
      setNotification({ 
        open: true, 
        message: `Restored to ${versionName} successfully!`, 
        severity: 'success' 
      });
    } catch (error) {
      console.error('Error restoring version:', error);
      setNotification({ 
        open: true, 
        message: `Version restore failed: ${error.response?.data?.detail || error.message}`, 
        severity: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  // This is a regular undo of just the last operation
  const handleUndo = async () => {
    if (datasets.length === 0 || !datasets[activeDatasetIndex]?.dataset_id) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`http://localhost:8000/undo/${datasets[activeDatasetIndex].dataset_id}`);
      
      const { data, columns } = response.data;
      
      // Update the datasets array with new data and columns
      setDatasets(prev => {
        const updated = [...prev];
        if (updated[activeDatasetIndex]) {
          updated[activeDatasetIndex].data = data || [];
          updated[activeDatasetIndex].columns = columns || [];
        }
        return updated;
      });
      
      // Update transformation history
      const datasetId = datasets[activeDatasetIndex].dataset_id;
      setTransformationHistory(prev => {
        const currentHistory = prev[datasetId] || [];
        if (currentHistory.length === 0) return prev;
        
        return {
          ...prev,
          [datasetId]: currentHistory.slice(0, -1)
        };
      });
      
      setData(data || []);
      setColumns(columns || []);
      setNotification({ open: true, message: 'Last operation undone successfully!', severity: 'success' });
    } catch (error) {
      console.error('Error undoing operation:', error);
      setNotification({ 
        open: true, 
        message: `Undo failed: ${error.response?.data?.detail || error.message}`, 
        severity: 'error' 
      });
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format) => {
    if (datasets.length === 0 || !datasets[activeDatasetIndex]?.dataset_id) return;
    
    try {
      window.open(`http://localhost:8000/export/${datasets[activeDatasetIndex].dataset_id}?format=${format}`, '_blank');
      setNotification({ open: true, message: `Exporting as ${format.toUpperCase()}...`, severity: 'info' });
    } catch (error) {
      console.error('Error exporting data:', error);
      setNotification({ 
        open: true, 
        message: `Export failed: ${error.message}`, 
        severity: 'error' 
      });
    }
  };

  const handleSwitchDataset = (index) => {
    setActiveDatasetIndex(index);
    
    // If in multi-view mode, add to open datasets if not already there
    if (multiViewMode) {
      const safeOpenDatasets = Array.isArray(openDatasets) ? openDatasets : [];
      if (!safeOpenDatasets.includes(index) && datasets[index]?.status === 'success') {
        setOpenDatasets(prev => [...(prev || []), index]);
        setActiveTabIndex(safeOpenDatasets.length);
      } else if (safeOpenDatasets.includes(index)) {
        // Set the active tab to this dataset
        setActiveTabIndex(safeOpenDatasets.indexOf(index));
      }
    }
  };
  
  const handleTabChange = (event, newValue) => {
    setActiveTabIndex(newValue);
    // Set the active dataset to match the tab
    const safeOpenDatasets = Array.isArray(openDatasets) ? openDatasets : [];
    if (safeOpenDatasets[newValue] !== undefined) {
      setActiveDatasetIndex(safeOpenDatasets[newValue]);
    }
  };
  
  const handleCloseTab = (index) => {
    // Remove from open datasets
    const safeOpenDatasets = Array.isArray(openDatasets) ? [...openDatasets] : [];
    safeOpenDatasets.splice(index, 1);
    setOpenDatasets(safeOpenDatasets);
    
    // Adjust active tab index if needed
    if (activeTabIndex >= safeOpenDatasets.length) {
      setActiveTabIndex(Math.max(0, safeOpenDatasets.length - 1));
    }
    
    // Update active dataset if we still have open tabs
    if (safeOpenDatasets.length > 0) {
      const newActiveTabIndex = Math.min(activeTabIndex, safeOpenDatasets.length - 1);
      setActiveDatasetIndex(safeOpenDatasets[newActiveTabIndex]);
    }
  };

  const handleCloseNotification = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setNotification({ ...notification, open: false });
  };

  const activeDataset = datasets.length > 0 ? datasets[activeDatasetIndex] : null;
  
  const renderDataGrid = (datasetIndex) => {
    const dataset = datasets[datasetIndex];
    if (!dataset || dataset.status !== 'success') {
      return (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6" color="error">
            Error loading dataset
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {dataset?.error || 'Invalid dataset'}
          </Typography>
        </Box>
      );
    }
    
    return (
      <DataGrid 
        data={datasetIndex === activeDatasetIndex ? data : (dataset.data || [])} 
        columns={datasetIndex === activeDatasetIndex ? columns : (dataset.columns || [])} 
        rowCount={datasetIndex === activeDatasetIndex ? (data || []).length : (dataset.data || []).length}
        loading={loading && datasetIndex === activeDatasetIndex} 
      />
    );
  };

  // Check if openDatasets is valid
  const safeOpenDatasets = Array.isArray(openDatasets) ? openDatasets : [];

  // Helper function to get operation description
  const getOperationDescription = (operation) => {
    const type = operation.type;
    switch(type) {
      case 'drop_columns':
        return `Dropped ${operation.columns.length} column(s)`;
      case 'fill_na':
        return `Filled missing values in "${operation.column}"`;
      case 'change_type':
        return `Changed "${operation.column}" type to ${operation.new_type}`;
      case 'filter_rows':
        return `Filtered rows by "${operation.column}"`;
      case 'remove_duplicates':
        return 'Removed duplicate rows';
      case 'rename_column':
        return `Renamed column "${operation.old_name}"`;
      default:
        return `Applied ${type} operation`;
    }
  };

  const handleApplyTransformation = async (operation) => {
    setIsProcessingTransformation(true);
    try {
      await handleTransform(operation);
    } finally {
      setIsProcessingTransformation(false);
    }
  };

  const handleAITransform = async (operation) => {
    setIsProcessingData(true);
    try {
      await handleTransform(operation);
    } finally {
      setIsProcessingData(false);
    }
  };

  const handleRevertToVersion = async (versionIndex) => {
    setIsProcessingData(true);
    try {
      await handleUndoToVersion(versionIndex);
    } finally {
      setIsProcessingData(false);
    }
  };

  const handleToolTabChange = (event, newValue) => {
    setActiveToolTab(newValue);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={false} sx={{ height: '100vh', display: 'flex', flexDirection: 'column', pt: 2, pb: 2 }}>
        {/* Header */}
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h4" component="h1">
            DataCleaner
          </Typography>
          <Box>
            <IconButton onClick={toggleDarkMode} color="inherit" sx={{ mr: 1 }}>
              {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
            <FormControlLabel
              control={
                <Switch 
                  checked={multiViewMode} 
                  onChange={toggleMultiViewMode} 
                  name="multiView" 
                  color="primary"
                />
              }
              label={
                <Box display="flex" alignItems="center">
                  <ViewSidebarIcon sx={{ mr: 0.5 }} fontSize="small" />
                  <Typography variant="body2">Multi-view</Typography>
                </Box>
              }
            />
          </Box>
        </Box>

        {/* Main Content */}
        <Box sx={{ display: 'flex', height: 'calc(100% - 40px)', gap: 2 }}>
          {/* Left sidebar - Dataset list */}
          <Paper elevation={3} sx={{ width: '200px', p: 1, overflow: 'auto' }}>
            <Typography variant="h6" gutterBottom>Datasets</Typography>
            <FileUpload onFileUpload={handleFileUpload} disabled={loading} />
            <Divider sx={{ my: 1 }} />
            <List dense>
              {datasets.map((dataset, index) => (
                <ListItem 
                  key={index} 
                  disablePadding
                  secondaryAction={
                    dataset.status === 'success' ? (
                      <Badge color="success" variant="dot">
                        <CheckCircleIcon fontSize="small" color="success" />
                      </Badge>
                    ) : (
                      <Badge color="error" variant="dot">
                        <ErrorIcon fontSize="small" color="error" />
                      </Badge>
                    )
                  }
                >
                  <ListItemButton 
                    selected={activeDatasetIndex === index}
                    onClick={() => handleSwitchDataset(index)}
                  >
                    <ListItemIcon>
                      <InsertDriveFileIcon color={dataset.status === 'success' ? 'primary' : 'disabled'} />
                    </ListItemIcon>
                    <ListItemText 
                      primary={dataset.filename} 
                      primaryTypographyProps={{ 
                        noWrap: true, 
                        style: { maxWidth: '100px' } 
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Paper>

          {/* Main Content Area - Flexible based on mode */}
          <Box sx={{ display: 'flex', flex: 1, gap: 2 }}>
            {/* Data Grid Section */}
            <Paper 
              elevation={3} 
              sx={{ 
                flex: 1, 
                display: 'flex', 
                flexDirection: 'column', 
                overflow: 'hidden'
              }}
            >
              {/* In single view mode, show data grid for active dataset */}
              {!multiViewMode && (
                renderDataGrid(activeDatasetIndex)
              )}
              
              {/* In multi-view mode, show tabs for open datasets */}
              {multiViewMode && (
                <>
                  <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs
                      value={activeTabIndex}
                      onChange={handleTabChange}
                      variant="scrollable"
                      scrollButtons="auto"
                    >
                      {safeOpenDatasets.map((datasetIndex, index) => (
                        <Tab 
                          key={index}
                          label={
                            <Box display="flex" alignItems="center">
                              <Typography variant="body2" noWrap sx={{ maxWidth: '120px' }}>
                                {datasets[datasetIndex]?.filename || `Dataset ${datasetIndex + 1}`}
                              </Typography>
                              <IconButton 
                                size="small" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleCloseTab(index);
                                }}
                                sx={{ ml: 1 }}
                              >
                                <CloseIcon fontSize="small" />
                              </IconButton>
                            </Box>
                          }
                          sx={{ minHeight: '48px' }}
                        />
                      ))}
                    </Tabs>
                  </Box>
                  
                  {safeOpenDatasets.map((datasetIndex, index) => (
                    <Box
                      key={index}
                      hidden={activeTabIndex !== index}
                      sx={{ height: '100%', display: activeTabIndex === index ? 'flex' : 'none', flexDirection: 'column' }}
                    >
                      {renderDataGrid(datasetIndex)}
                    </Box>
                  ))}
                </>
              )}
            </Paper>

            {/* Right section for tools */}
            <Paper 
              elevation={3} 
              sx={{ 
                width: '30%', 
                p: 2, 
                display: 'flex', 
                flexDirection: 'column',
                height: '100%',
                overflowY: 'auto'
              }}
            >
              <Tabs 
                value={activeToolTab} 
                onChange={handleToolTabChange} 
                variant="scrollable" 
                scrollButtons="auto"
                sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
              >
                <Tab label="Transform" />
                <Tab label="AI Assistant" />
                <Tab label="Reports" />
                <Tab label="History" />
              </Tabs>

              {activeToolTab === 0 && (
                <TransformationPanel 
                  selectedDataset={selectedDatasetId} 
                  onApplyTransformation={handleApplyTransformation}
                  isProcessing={isProcessingTransformation}
                  disabled={!selectedDatasetId || isProcessingData}
                />
              )}
              {activeToolTab === 1 && (
                <AIAssistant 
                  dataset={datasets[activeDatasetIndex]?.data} 
                  onTransform={handleAITransform}
                  selectedDataset={selectedDatasetId}
                  disabled={!selectedDatasetId || isProcessingData}
                />
              )}
              {activeToolTab === 2 && (
                <ReportGenerator 
                  selectedDataset={selectedDatasetId}
                  disabled={!selectedDatasetId || isProcessingData}
                />
              )}
              {activeToolTab === 3 && (
                <VersionHistory 
                  history={transformationHistory[selectedDatasetId] || []} 
                  onRevertToVersion={handleRevertToVersion}
                  disabled={!selectedDatasetId || isProcessingData}
                />
              )}
            </Paper>
          </Box>
        </Box>

        {/* Snackbar for notifications */}
        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
        >
          <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
            {notification.message}
          </Alert>
        </Snackbar>

        {/* Transform result alert */}
        {transformAlert && (
          <Box 
            sx={{ 
              position: 'fixed', 
              bottom: 20, 
              right: 20, 
              zIndex: 1000,
              maxWidth: '400px'
            }}
          >
            <Paper elevation={4} sx={{ p: 2, backgroundColor: 'success.light' }}>
              <Typography variant="subtitle1" sx={{ color: 'white', fontWeight: 'bold' }}>
                {transformAlert.message}
              </Typography>
              <Typography variant="body2" sx={{ color: 'white' }}>
                {transformAlert.operation}
              </Typography>
            </Paper>
          </Box>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App; 
import React, { useState, useRef } from 'react';
import {
  Box,
  Button,
  Typography,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  Stack,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import UploadFileIcon from '@mui/icons-material/UploadFile';

const FileUpload = ({ onFileUpload, loading, disabled }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const filesArray = Array.from(e.dataTransfer.files);
      setSelectedFiles(prev => [...prev, ...filesArray]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...filesArray]);
    }
  };

  const handleUpload = () => {
    if (selectedFiles.length > 0) {
      onFileUpload(selectedFiles[currentFileIndex]);
    }
  };

  const handleRemoveFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    if (currentFileIndex >= index && currentFileIndex > 0) {
      setCurrentFileIndex(prev => prev - 1);
    }
  };

  const handleSelectFile = (index) => {
    setCurrentFileIndex(index);
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 500 }}>
      <Box
        sx={{
          p: 3,
          border: '2px dashed',
          borderColor: dragActive ? 'primary.main' : 'divider',
          borderRadius: 2,
          backgroundColor: dragActive ? 'action.hover' : 'background.paper',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.2s ease',
          cursor: disabled ? 'default' : 'pointer',
          mb: 2,
          opacity: disabled ? 0.6 : 1,
        }}
        onDragEnter={disabled ? null : handleDrag}
        onDragLeave={disabled ? null : handleDrag}
        onDragOver={disabled ? null : handleDrag}
        onDrop={disabled ? null : handleDrop}
        onClick={disabled ? null : () => fileInputRef.current.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          accept=".csv,.xlsx,.json,.xls,.tsv,.txt"
          multiple
          disabled={disabled}
        />
        <CloudUploadIcon color="primary" sx={{ fontSize: 48, mb: 2 }} />
        <Typography variant="h6" gutterBottom align="center">
          Drag & Drop your data files here
        </Typography>
        <Typography variant="body2" color="text.secondary" align="center">
          —or—
        </Typography>
        <Button variant="outlined" sx={{ mt: 2 }} disabled={disabled}>
          Browse Files
        </Button>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          Supported formats: CSV, Excel, JSON, TSV, TXT
        </Typography>
      </Box>

      {selectedFiles.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Selected Files ({selectedFiles.length})
          </Typography>
          
          <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap', gap: 1 }}>
            {selectedFiles.map((file, index) => (
              <Chip 
                key={`${file.name}-${index}`}
                label={file.name}
                onClick={() => handleSelectFile(index)}
                onDelete={() => handleRemoveFile(index)}
                color={currentFileIndex === index ? "primary" : "default"}
                variant={currentFileIndex === index ? "filled" : "outlined"}
                icon={<InsertDriveFileIcon />}
                disabled={disabled || loading}
              />
            ))}
          </Stack>
          
          <List disablePadding>
            {selectedFiles[currentFileIndex] && (
              <ListItem
                secondaryAction={
                  <IconButton edge="end" onClick={() => handleRemoveFile(currentFileIndex)} disabled={loading || disabled}>
                    <DeleteIcon />
                  </IconButton>
                }
              >
                <ListItemIcon>
                  <InsertDriveFileIcon color="primary" />
                </ListItemIcon>
                <ListItemText
                  primary={selectedFiles[currentFileIndex].name}
                  secondary={`${(selectedFiles[currentFileIndex].size / 1024).toFixed(2)} KB`}
                />
              </ListItem>
            )}
          </List>
        </Paper>
      )}

      <Button
        variant="contained"
        color="primary"
        fullWidth
        startIcon={loading ? <CircularProgress size={24} color="inherit" /> : <UploadFileIcon />}
        onClick={handleUpload}
        disabled={selectedFiles.length === 0 || loading || disabled}
        sx={{ py: 1 }}
      >
        {loading ? 'Uploading...' : `Upload ${selectedFiles[currentFileIndex]?.name || 'File'}`}
      </Button>
    </Box>
  );
};

export default FileUpload; 
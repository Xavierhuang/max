import React, { useCallback } from 'react';
import { Box, Typography, Button, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const FileUploader = ({ onUpload, loading }) => {
  const onDrop = useCallback((acceptedFiles) => {
    // Only use the first file if multiple files are dropped
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
    },
    multiple: false,
  });

  return (
    <Box sx={{ textAlign: 'center' }}>
      <Typography variant="h6" gutterBottom>
        Upload Your Dataset
      </Typography>
      
      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.400',
          borderRadius: 2,
          p: 3,
          mb: 2,
          backgroundColor: isDragActive ? 'primary.50' : 'grey.50',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          height: 200,
        }}
      >
        <input {...getInputProps()} />
        
        {loading ? (
          <CircularProgress />
        ) : (
          <>
            <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
            <Typography variant="body1">
              {isDragActive
                ? 'Drop the file here...'
                : 'Drag & drop a CSV, Excel, or JSON file here, or click to select'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Supported file formats: CSV, Excel (.xlsx, .xls), JSON
            </Typography>
          </>
        )}
      </Box>
      
      <Button
        variant="contained"
        component="label"
        disabled={loading}
        sx={{ mt: 2 }}
      >
        {loading ? <CircularProgress size={24} /> : 'Select File'}
        <input
          type="file"
          hidden
          accept=".csv,.xlsx,.xls,.json"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              onUpload(e.target.files[0]);
            }
          }}
        />
      </Button>
    </Box>
  );
};

export default FileUploader; 
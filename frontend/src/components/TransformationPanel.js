import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Grid,
  Divider,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
  Paper
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DeleteIcon from '@mui/icons-material/Delete';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import UndoIcon from '@mui/icons-material/Undo';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import FilterListIcon from '@mui/icons-material/FilterList';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import TextFormatIcon from '@mui/icons-material/TextFormat';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import WarningIcon from '@mui/icons-material/Warning';

const TransformationPanel = ({ columns = [], onTransform, onUndo, onExport, loading, disabled = false }) => {
  const [transformationType, setTransformationType] = useState('');
  const [selectedColumn, setSelectedColumn] = useState('');
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [fillMethod, setFillMethod] = useState('value');
  const [fillValue, setFillValue] = useState('');
  const [newColumnName, setNewColumnName] = useState('');
  const [dataType, setDataType] = useState('');
  const [filterCondition, setFilterCondition] = useState('equals');
  const [filterValue, setFilterValue] = useState('');
  const [exportFormat, setExportFormat] = useState('csv');

  const handleApplyTransformation = () => {
    let operation = { type: transformationType };

    switch (transformationType) {
      case 'drop_columns':
        operation.columns = selectedColumns;
        break;
      case 'rename_column':
        operation.old_name = selectedColumn;
        operation.new_name = newColumnName;
        break;
      case 'fill_na':
        operation.column = selectedColumn;
        operation.method = fillMethod;
        if (fillMethod === 'value') {
          operation.value = fillValue;
        }
        break;
      case 'change_type':
        operation.column = selectedColumn;
        operation.new_type = dataType;
        break;
      case 'filter_rows':
        operation.column = selectedColumn;
        operation.condition = filterCondition;
        operation.value = filterValue;
        break;
      case 'remove_duplicates':
        operation.columns = selectedColumns.length > 0 ? selectedColumns : null;
        break;
      default:
        return;
    }

    onTransform(operation);
  };

  const handleExport = () => {
    onExport(exportFormat);
  };

  const renderTransformationOptions = () => {
    switch (transformationType) {
      case 'drop_columns':
        return (
          <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
            <InputLabel>Select Columns to Drop</InputLabel>
            <Select
              multiple
              value={selectedColumns}
              onChange={(e) => setSelectedColumns(e.target.value)}
              label="Select Columns to Drop"
            >
              {columns.map((column) => (
                <MenuItem key={column} value={column}>
                  {column}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );
      
      case 'rename_column':
        return (
          <>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Select Column to Rename</InputLabel>
              <Select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                label="Select Column to Rename"
              >
                {columns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              fullWidth
              size="small"
              label="New Column Name"
              value={newColumnName}
              onChange={(e) => setNewColumnName(e.target.value)}
              margin="normal"
              disabled={disabled || loading}
            />
          </>
        );
      
      case 'fill_na':
        return (
          <>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Select Column</InputLabel>
              <Select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                label="Select Column"
              >
                {columns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Fill Method</InputLabel>
              <Select
                value={fillMethod}
                onChange={(e) => setFillMethod(e.target.value)}
                label="Fill Method"
              >
                <MenuItem value="value">Fixed Value</MenuItem>
                <MenuItem value="mean">Mean</MenuItem>
                <MenuItem value="median">Median</MenuItem>
                <MenuItem value="mode">Mode</MenuItem>
                <MenuItem value="ffill">Forward Fill</MenuItem>
                <MenuItem value="bfill">Backward Fill</MenuItem>
              </Select>
            </FormControl>
            {fillMethod === 'value' && (
              <TextField
                fullWidth
                size="small"
                label="Fill Value"
                value={fillValue}
                onChange={(e) => setFillValue(e.target.value)}
                margin="normal"
                disabled={disabled || loading}
              />
            )}
          </>
        );
      
      case 'change_type':
        return (
          <>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Select Column</InputLabel>
              <Select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                label="Select Column"
              >
                {columns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>New Data Type</InputLabel>
              <Select
                value={dataType}
                onChange={(e) => setDataType(e.target.value)}
                label="New Data Type"
              >
                <MenuItem value="int">Integer</MenuItem>
                <MenuItem value="float">Float</MenuItem>
                <MenuItem value="string">String</MenuItem>
                <MenuItem value="datetime">Date/Time</MenuItem>
              </Select>
            </FormControl>
          </>
        );
      
      case 'filter_rows':
        return (
          <>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Select Column</InputLabel>
              <Select
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                label="Select Column"
              >
                {columns.map((column) => (
                  <MenuItem key={column} value={column}>
                    {column}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
              <InputLabel>Condition</InputLabel>
              <Select
                value={filterCondition}
                onChange={(e) => setFilterCondition(e.target.value)}
                label="Condition"
              >
                <MenuItem value="equals">Equals</MenuItem>
                <MenuItem value="not_equals">Not Equals</MenuItem>
                <MenuItem value="greater_than">Greater Than</MenuItem>
                <MenuItem value="less_than">Less Than</MenuItem>
                <MenuItem value="contains">Contains</MenuItem>
              </Select>
            </FormControl>
            <TextField
              fullWidth
              size="small"
              label="Value"
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              margin="normal"
              disabled={disabled || loading}
            />
          </>
        );
      
      case 'remove_duplicates':
        return (
          <FormControl fullWidth size="small" sx={{ mt: 1 }} disabled={disabled || loading}>
            <InputLabel>Consider Columns (optional)</InputLabel>
            <Select
              multiple
              value={selectedColumns}
              onChange={(e) => setSelectedColumns(e.target.value)}
              label="Consider Columns (optional)"
            >
              {columns.map((column) => (
                <MenuItem key={column} value={column}>
                  {column}
                </MenuItem>
              ))}
            </Select>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
              If no columns are selected, all columns will be considered for duplicate detection.
            </Typography>
          </FormControl>
        );
      
      default:
        return null;
    }
  };

  if (disabled) {
    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', p: 2 }}>
        <WarningIcon color="action" sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
        <Typography variant="body1" color="text.secondary" align="center">
          Select a valid dataset to enable transformations
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <AutoFixHighIcon sx={{ mr: 1 }} />
        Non-AI Transformations
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <FormControl fullWidth size="small" disabled={loading}>
          <InputLabel>Transformation Type</InputLabel>
          <Select
            value={transformationType}
            onChange={(e) => setTransformationType(e.target.value)}
            label="Transformation Type"
          >
            <MenuItem value="drop_columns">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
                Drop Columns
              </Box>
            </MenuItem>
            <MenuItem value="rename_column">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TextFormatIcon fontSize="small" sx={{ mr: 1 }} />
                Rename Column
              </Box>
            </MenuItem>
            <MenuItem value="fill_na">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <AutoFixHighIcon fontSize="small" sx={{ mr: 1 }} />
                Fill Missing Values
              </Box>
            </MenuItem>
            <MenuItem value="change_type">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TextFormatIcon fontSize="small" sx={{ mr: 1 }} />
                Change Data Type
              </Box>
            </MenuItem>
            <MenuItem value="filter_rows">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <FilterListIcon fontSize="small" sx={{ mr: 1 }} />
                Filter Rows
              </Box>
            </MenuItem>
            <MenuItem value="remove_duplicates">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <ContentCopyIcon fontSize="small" sx={{ mr: 1 }} />
                Remove Duplicates
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      {renderTransformationOptions()}
      
      <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
        <Tooltip title="Apply Transformation">
          <Button
            variant="contained"
            color="primary"
            size="small"
            onClick={handleApplyTransformation}
            disabled={loading || !transformationType}
            startIcon={loading ? <CircularProgress size={18} /> : <PlayArrowIcon />}
            sx={{ flexGrow: 1 }}
          >
            Apply
          </Button>
        </Tooltip>
        
        <Tooltip title="Undo Last Transformation">
          <IconButton
            color="default"
            size="small"
            onClick={onUndo}
            disabled={loading}
          >
            <UndoIcon />
          </IconButton>
        </Tooltip>
      </Box>
      
      <Divider sx={{ my: 2 }} />
      
      <Box sx={{ mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Export Data
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControl size="small" sx={{ flexGrow: 1 }} disabled={loading}>
            <InputLabel>Format</InputLabel>
            <Select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
              label="Format"
              size="small"
            >
              <MenuItem value="csv">CSV</MenuItem>
              <MenuItem value="xlsx">Excel (.xlsx)</MenuItem>
              <MenuItem value="json">JSON</MenuItem>
            </Select>
          </FormControl>
          
          <Button
            variant="outlined"
            color="primary"
            size="small"
            onClick={handleExport}
            disabled={loading}
            startIcon={<FileDownloadIcon />}
          >
            Export
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default TransformationPanel; 
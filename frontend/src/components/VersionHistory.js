import React from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemText,
  Typography,
  Divider,
  Paper,
  Chip,
  Tooltip,
  IconButton
} from '@mui/material';
import RestoreIcon from '@mui/icons-material/Restore';
import HistoryIcon from '@mui/icons-material/History';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import FunctionsIcon from '@mui/icons-material/Functions';
import ChangeCircleIcon from '@mui/icons-material/ChangeCircle';
import FilterAltIcon from '@mui/icons-material/FilterAlt';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import InfoIcon from '@mui/icons-material/Info';

const VersionHistory = ({ transformationHistory, onUndoToVersion }) => {
  if (!transformationHistory || transformationHistory.length === 0) {
    return (
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <HistoryIcon sx={{ mr: 1 }} />
          <Typography variant="subtitle1">Version History</Typography>
        </Box>
        <Paper sx={{ p: 2, mb: 2, bgcolor: 'background.paper', flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
            No transformations applied yet
          </Typography>
        </Paper>
      </Box>
    );
  }

  const getOperationIcon = (operation) => {
    const type = operation.type;
    switch(type) {
      case 'drop_columns':
        return <FormatListBulletedIcon color="error" />;
      case 'fill_na':
        return <FunctionsIcon color="info" />;
      case 'change_type':
        return <ChangeCircleIcon color="warning" />;
      case 'filter_rows':
        return <FilterAltIcon color="success" />;
      case 'remove_duplicates':
        return <DeleteSweepIcon color="secondary" />;
      default:
        return <InfoIcon color="action" />;
    }
  };

  const getOperationDescription = (operation) => {
    const type = operation.type;
    switch(type) {
      case 'drop_columns':
        return `Dropped columns: ${operation.columns.join(', ')}`;
      case 'fill_na': {
        const method = operation.method === 'value' 
          ? `value "${operation.value}"` 
          : operation.method;
        return `Filled missing values in "${operation.column}" with ${method}`;
      }
      case 'change_type':
        return `Changed "${operation.column}" to ${operation.new_type} type`;
      case 'filter_rows':
        return `Filtered rows where "${operation.column}" ${operation.condition.replace('_', ' ')} "${operation.value}"`;
      case 'remove_duplicates':
        return operation.columns 
          ? `Removed duplicates based on columns: ${operation.columns.join(', ')}` 
          : 'Removed all duplicate rows';
      case 'rename_column':
        return `Renamed column "${operation.old_name}" to "${operation.new_name}"`;
      default:
        return `Applied ${type} operation`;
    }
  };

  return (
    <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <HistoryIcon sx={{ mr: 1 }} />
        <Typography variant="subtitle1">Version History</Typography>
      </Box>
      
      <Paper sx={{ p: 0, flexGrow: 1, overflow: 'auto', bgcolor: 'background.paper' }}>
        <List disablePadding>
          <ListItem sx={{ bgcolor: 'background.default', py: 1 }}>
            <ListItemText 
              primary={
                <Typography variant="subtitle2">
                  Original Dataset
                </Typography>
              }
              secondary="Initial state before transformations"
            />
            <Tooltip title="Restore to original">
              <IconButton 
                size="small"
                onClick={() => onUndoToVersion(-1)}
              >
                <RestoreIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </ListItem>
          
          <Divider />
          
          {transformationHistory.map((operation, index) => (
            <React.Fragment key={index}>
              <ListItem sx={{ py: 1 }}>
                <Box sx={{ display: 'flex', mr: 1 }}>
                  {getOperationIcon(operation)}
                </Box>
                <ListItemText 
                  primary={
                    <Typography variant="body2" sx={{ fontWeight: 'medium' }}>
                      {getOperationDescription(operation)}
                    </Typography>
                  }
                  secondary={
                    operation.affectedRows !== undefined ? (
                      <Chip 
                        label={`${operation.affectedRows} rows affected`} 
                        size="small" 
                        color="primary" 
                        variant="outlined"
                        sx={{ mt: 0.5, height: 20 }}
                      />
                    ) : null
                  }
                />
                <Tooltip title="Restore to this version">
                  <IconButton 
                    size="small"
                    onClick={() => onUndoToVersion(index)}
                  >
                    <RestoreIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </ListItem>
              <Divider />
            </React.Fragment>
          ))}
        </List>
      </Paper>
    </Box>
  );
};

export default VersionHistory; 
import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Paper,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import TextFormatIcon from '@mui/icons-material/TextFormat';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import FilterListIcon from '@mui/icons-material/FilterList';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

const TransformationHistory = ({ transformations = [] }) => {
  const getTransformationIcon = (type) => {
    switch (type) {
      case 'drop_columns':
        return <DeleteIcon />;
      case 'rename_column':
        return <TextFormatIcon />;
      case 'fill_na':
        return <AutoFixHighIcon />;
      case 'change_type':
        return <TextFormatIcon />;
      case 'filter_rows':
        return <FilterListIcon />;
      case 'remove_duplicates':
        return <ContentCopyIcon />;
      default:
        return <AutoFixHighIcon />;
    }
  };

  const getTransformationDescription = (transformation) => {
    switch (transformation.type) {
      case 'drop_columns':
        return `Dropped columns: ${transformation.columns.join(', ')}`;
      
      case 'rename_column':
        return `Renamed column '${transformation.old_name}' to '${transformation.new_name}'`;
      
      case 'fill_na':
        if (transformation.method === 'value') {
          return `Filled missing values in '${transformation.column}' with '${transformation.value}'`;
        } else {
          return `Filled missing values in '${transformation.column}' using ${transformation.method}`;
        }
      
      case 'change_type':
        return `Changed '${transformation.column}' type to ${transformation.new_type}`;
      
      case 'filter_rows':
        return `Filtered rows where '${transformation.column}' ${transformation.condition.replace('_', ' ')} '${transformation.value}'`;
      
      case 'remove_duplicates':
        if (transformation.columns && transformation.columns.length > 0) {
          return `Removed duplicates based on columns: ${transformation.columns.join(', ')}`;
        } else {
          return 'Removed duplicates across all columns';
        }
      
      default:
        return 'Unknown transformation';
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Transformation History
      </Typography>
      
      {transformations.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          No transformations applied yet
        </Typography>
      ) : (
        <List sx={{ maxHeight: 300, overflow: 'auto' }}>
          {transformations.map((transformation, index) => (
            <React.Fragment key={index}>
              <ListItem>
                <ListItemIcon>
                  {getTransformationIcon(transformation.type)}
                </ListItemIcon>
                <ListItemText
                  primary={`#${index + 1}: ${transformation.type.replace('_', ' ')}`}
                  secondary={getTransformationDescription(transformation)}
                />
              </ListItem>
              {index < transformations.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Box>
  );
};

export default TransformationHistory; 
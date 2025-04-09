import React, { useState, useMemo } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import 'ag-grid-community/styles/ag-theme-material.css';

const DataGrid = ({ data = [], columns = [], rowCount, loading }) => {
  const [gridApi, setGridApi] = useState(null);
  const [gridColumnApi, setGridColumnApi] = useState(null);

  // Generate column definitions from column names
  const columnDefs = useMemo(() => {
    if (!columns || columns.length === 0) {
      return [];
    }
    
    return columns.map(col => ({
      field: col,
      headerName: col,
      sortable: true,
      filter: true,
      resizable: true,
      floatingFilter: true,
      editable: true,
      minWidth: 150,
      cellStyle: params => {
        // Highlight missing values
        if (params.value === null || params.value === undefined || params.value === '') {
          return { backgroundColor: 'rgba(255, 0, 0, 0.1)' };
        }
        return null;
      }
    }));
  }, [columns]);

  const defaultColDef = useMemo(() => ({
    flex: 1,
    minWidth: 100,
    filter: true,
    sortable: true,
    resizable: true,
    floatingFilter: true,
  }), []);

  const onGridReady = params => {
    setGridApi(params.api);
    setGridColumnApi(params.columnApi);
    
    // Auto-size all columns
    setTimeout(() => {
      params.columnApi.autoSizeAllColumns();
    }, 500);
    
    // Force refresh the grid to ensure data is displayed
    setTimeout(() => {
      params.api.refreshCells({ force: true });
    }, 100);
  };

  const exportToCsv = () => {
    if (gridApi) {
      gridApi.exportDataAsCsv();
    }
  };

  // Create formatted rows from data
  const rowData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0 || !columns || columns.length === 0) {
      return [];
    }

    console.log("Processing data for grid:", { dataLength: data.length, columns });
    
    try {
      // Create rows with the correct fields for each column
      return data.map((row, index) => {
        // If row is an array, convert to object using column names
        if (Array.isArray(row)) {
          return columns.reduce((obj, col, i) => {
            obj[col] = i < row.length ? row[i] : null;
            return obj;
          }, {});
        }
        
        // If row is already an object, use it directly
        return row;
      });
    } catch (error) {
      console.error("Error formatting row data:", error);
      return [];
    }
  }, [data, columns]);

  console.log("DataGrid render", { 
    hasData: Array.isArray(data) && data.length > 0, 
    dataLength: Array.isArray(data) ? data.length : 0,
    hasColumns: Array.isArray(columns) && columns.length > 0,
    rowDataLength: rowData.length,
    firstRow: rowData.length > 0 ? rowData[0] : null
  });

  return (
    <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ 
        flexGrow: 1, 
        width: '100%', 
        height: 'calc(100% - 30px)',
        '& .ag-cell-focus': { outline: 'none' },
        '& .ag-row-hover': { backgroundColor: 'rgba(33, 150, 243, 0.1)' },
        '& .ag-header-cell-label': { fontWeight: 'bold', color: '#333' },
        overflow: 'hidden'
      }} className="ag-theme-material">
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        ) : rowData.length > 0 ? (
          <AgGridReact
            rowData={rowData}
            columnDefs={columnDefs}
            defaultColDef={defaultColDef}
            animateRows={true}
            rowSelection="multiple"
            onGridReady={onGridReady}
            pagination={true}
            paginationPageSize={20}
            enableRangeSelection={false}
            enableCellChangeFlash={true}
            suppressRowClickSelection={true}
            domLayout="normal"
            suppressDragLeaveHidesColumns={true}
            suppressScrollOnNewData={false}
            suppressPropertyNamesCheck={true}
            suppressColumnVirtualisation={false}
            suppressRowVirtualisation={false}
            debugMode={false}
          />
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', flexDirection: 'column' }}>
            <Typography variant="body1" color="text.secondary">
              No rows to show
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {columns.length > 0 ? "The dataset appears to be empty" : "No columns detected in the dataset"}
            </Typography>
            <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1, maxWidth: '80%' }}>
              <Typography variant="caption" color="text.secondary">
                Debugging info: Columns: {columns.length}, Data items: {Array.isArray(data) ? data.length : 'not array'}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '30px', px: 1 }}>
        <Typography variant="caption" color="text.secondary">
          Total rows: {rowCount || (Array.isArray(data) ? data.length : 0)} | Showing: {rowData.length} rows
        </Typography>
      </Box>
    </Box>
  );
};

export default DataGrid; 
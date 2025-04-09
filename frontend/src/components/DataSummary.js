import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Button,
  Chip,
  Divider,
  CircularProgress,
  Alert,
  IconButton,
  Tooltip,
  Autocomplete
} from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart';
import MoneyIcon from '@mui/icons-material/Money';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

const DataSummary = ({ dataset, onClose }) => {
  const [summaryType, setSummaryType] = useState('financial');
  const [balanceColumn, setBalanceColumn] = useState('');
  const [companyColumn, setCompanyColumn] = useState('');
  const [dateColumn, setDateColumn] = useState('');
  const [loading, setLoading] = useState(false);
  const [summaryData, setSummaryData] = useState([]);
  const [error, setError] = useState(null);
  
  // Find potential financial columns
  useEffect(() => {
    if (!dataset || !dataset.columns) return;
    
    // Look for columns likely to contain financial data
    const possibleBalanceColumns = dataset.columns.filter(col => {
      const lowerCol = col.toLowerCase();
      return lowerCol.includes('balance') || 
             lowerCol.includes('amount') || 
             lowerCol.includes('total') ||
             lowerCol.includes('value') ||
             lowerCol.includes('sum');
    });
    
    if (possibleBalanceColumns.length > 0) {
      setBalanceColumn(possibleBalanceColumns[0]);
    }
    
    // Look for columns likely to contain company names
    const possibleCompanyColumns = dataset.columns.filter(col => {
      const lowerCol = col.toLowerCase();
      return lowerCol.includes('company') || 
             lowerCol.includes('entity') || 
             lowerCol.includes('organization') ||
             lowerCol.includes('business') ||
             lowerCol.includes('client') ||
             lowerCol.includes('customer') ||
             lowerCol.includes('name');
    });
    
    if (possibleCompanyColumns.length > 0) {
      setCompanyColumn(possibleCompanyColumns[0]);
    }
    
    // Look for columns likely to contain dates
    const possibleDateColumns = dataset.columns.filter(col => {
      const lowerCol = col.toLowerCase();
      return lowerCol.includes('date') || 
             lowerCol.includes('time') || 
             lowerCol.includes('period') ||
             lowerCol.includes('month') ||
             lowerCol.includes('year');
    });
    
    if (possibleDateColumns.length > 0) {
      setDateColumn(possibleDateColumns[0]);
    }
  }, [dataset]);

  // Generate summary based on selected columns
  const generateSummary = () => {
    if (!dataset || !dataset.data || !balanceColumn) {
      setError('Please select a balance column to analyze');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      let summary = [];
      
      if (summaryType === 'financial') {
        if (companyColumn) {
          // Group by company and calculate ending balances
          const companies = {};
          
          // First pass: collect all data for each company
          dataset.data.forEach(row => {
            const company = row[companyColumn];
            const balance = parseFloat(row[balanceColumn]) || 0;
            
            if (!company) return;
            
            if (!companies[company]) {
              companies[company] = {
                balances: [balance],
                dates: dateColumn ? [row[dateColumn]] : []
              };
            } else {
              companies[company].balances.push(balance);
              if (dateColumn) companies[company].dates.push(row[dateColumn]);
            }
          });
          
          // Second pass: calculate ending balances and other metrics
          Object.entries(companies).forEach(([company, data]) => {
            let endingBalance = 0;
            let maxBalance = Math.max(...data.balances);
            let minBalance = Math.min(...data.balances);
            
            // Sort by date if available
            if (dateColumn && data.dates.length > 0) {
              // Create paired arrays of dates and balances
              const paired = data.dates.map((date, i) => ({ date, balance: data.balances[i] }));
              
              // Sort by date (assuming date format can be compared)
              paired.sort((a, b) => {
                return new Date(a.date) - new Date(b.date);
              });
              
              // Get the last (most recent) balance
              endingBalance = paired[paired.length - 1].balance;
            } else {
              // If no date column, use the last balance in the dataset
              endingBalance = data.balances[data.balances.length - 1];
            }
            
            summary.push({
              company,
              endingBalance,
              maxBalance,
              minBalance,
              change: data.balances.length > 1 ? endingBalance - data.balances[0] : 0,
              count: data.balances.length
            });
          });
        } else {
          // No company column, just show overall total
          const balances = dataset.data.map(row => parseFloat(row[balanceColumn]) || 0);
          const endingBalance = balances[balances.length - 1];
          
          summary.push({
            company: 'Overall',
            endingBalance,
            maxBalance: Math.max(...balances),
            minBalance: Math.min(...balances),
            change: balances.length > 1 ? endingBalance - balances[0] : 0,
            count: balances.length
          });
        }
      }
      
      // Sort by ending balance (highest first)
      summary.sort((a, b) => b.endingBalance - a.endingBalance);
      
      setSummaryData(summary);
    } catch (err) {
      console.error('Error generating summary:', err);
      setError('Error analyzing data. Please check the selected columns and try again.');
    } finally {
      setLoading(false);
    }
  };

  // Format currency for display
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" gutterBottom>
          <BarChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Data Summary: {dataset?.filename}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Autocomplete
            value={balanceColumn}
            onChange={(_, newValue) => setBalanceColumn(newValue)}
            options={dataset?.columns || []}
            renderInput={(params) => <TextField {...params} label="Balance Column" size="small" fullWidth />}
            size="small"
            sx={{ flexGrow: 1 }}
          />
          
          <Autocomplete
            value={companyColumn}
            onChange={(_, newValue) => setCompanyColumn(newValue)}
            options={dataset?.columns || []}
            renderInput={(params) => <TextField {...params} label="Company Column" size="small" fullWidth />}
            size="small"
            sx={{ flexGrow: 1 }}
          />
          
          <Autocomplete
            value={dateColumn}
            onChange={(_, newValue) => setDateColumn(newValue)}
            options={dataset?.columns || []}
            renderInput={(params) => <TextField {...params} label="Date Column (optional)" size="small" fullWidth />}
            size="small"
            sx={{ flexGrow: 1 }}
          />
        </Box>
        
        <Button 
          variant="contained" 
          onClick={generateSummary}
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
          disabled={loading || !balanceColumn}
          sx={{ mr: 1 }}
        >
          Generate Summary
        </Button>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ m: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        {summaryData.length > 0 ? (
          <>
            <Typography variant="subtitle1" gutterBottom>
              <MoneyIcon sx={{ mr: 1, verticalAlign: 'middle', color: 'primary.main' }} />
              Financial Summary - Ending Balances
            </Typography>
            
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Company</strong></TableCell>
                    <TableCell align="right"><strong>Ending Balance</strong></TableCell>
                    <TableCell align="right"><strong>Change</strong></TableCell>
                    <TableCell align="right"><strong>Min</strong></TableCell>
                    <TableCell align="right"><strong>Max</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {summaryData.map((row, index) => (
                    <TableRow key={index} hover>
                      <TableCell>{row.company}</TableCell>
                      <TableCell align="right">
                        <Typography 
                          fontWeight="bold" 
                          color={row.endingBalance >= 0 ? 'success.main' : 'error.main'}
                        >
                          {formatCurrency(row.endingBalance)}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                          {row.change > 0 ? (
                            <TrendingUpIcon fontSize="small" color="success" sx={{ mr: 0.5 }} />
                          ) : row.change < 0 ? (
                            <TrendingDownIcon fontSize="small" color="error" sx={{ mr: 0.5 }} />
                          ) : null}
                          <Typography color={row.change > 0 ? 'success.main' : row.change < 0 ? 'error.main' : 'text.secondary'}>
                            {formatCurrency(row.change)}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">{formatCurrency(row.minBalance)}</TableCell>
                      <TableCell align="right">{formatCurrency(row.maxBalance)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <AutoGraphIcon sx={{ fontSize: 60, color: 'text.secondary', opacity: 0.5, mb: 2 }} />
            <Typography color="text.secondary">
              Select columns and generate a summary to see financial analysis
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default DataSummary; 
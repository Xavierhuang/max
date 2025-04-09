import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider,
  IconButton,
  Card,
  CardContent,
  Chip,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import CodeIcon from '@mui/icons-material/Code';
import { 
  PsychologyAlt as AIIcon, 
  CleaningServices as CleaningIcon,
  Analytics as InsightsIcon 
} from '@mui/icons-material';
import axios from 'axios';

const AIAssistant = ({ dataset, onTransform, selectedDataset, disabled }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Hello! I can help you analyze your data and suggest transformations. You can ask me questions like "What are the missing values?", "How should I clean this dataset?", or "What is the average income?"' 
    }
  ]);
  const [loading, setLoading] = useState(false);
  const [dataInsights, setDataInsights] = useState(null);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);
  const [analysisType, setAnalysisType] = useState('general');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (dataset) {
      analyzeData();
    }
  }, [dataset]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const analyzeData = async () => {
    if (!dataset || !dataset.data || !dataset.columns) return;
    
    setLoading(true);
    
    try {
      // Perform more thorough data analysis to identify issues and suggest intelligent actions
      const insights = {
        rowCount: dataset.data.length,
        columnCount: dataset.columns.length,
        missingValues: {},
        datatypeIssues: [],
        outliers: [],
        correlations: [],
        duplicateRows: 0,
        qualityScore: 0,
        suggestions: []
      };
      
      // 1. Check for missing values with more detail
      let totalMissingCount = 0;
      dataset.columns.forEach(column => {
        const missingCount = dataset.data.filter(row => 
          row[column] === null || row[column] === undefined || row[column] === ''
        ).length;
        
        if (missingCount > 0) {
          insights.missingValues[column] = {
            count: missingCount,
            percentage: (missingCount / dataset.data.length * 100).toFixed(2)
          };
          totalMissingCount += missingCount;
          
          // Intelligent suggestions based on missing value patterns
          if (missingCount / dataset.data.length < 0.05) {
            // Very few missing values - simple fill
            insights.suggestions.push({
              type: 'fill_missing',
              column: column,
              method: 'value',
              value: '0',
              priority: 'high',
              description: `Fill missing values in "${column}" (${missingCount} missing values, likely data entry errors)`
            });
          } else if (missingCount / dataset.data.length < 0.3) {
            // Moderate missing values - statistical fill
            const isNumeric = dataset.data.some(row => row[column] !== null && row[column] !== undefined && !isNaN(parseFloat(row[column])));
            
            if (isNumeric) {
              insights.suggestions.push({
                type: 'fill_missing',
                column: column,
                method: 'mean',
                priority: 'medium',
                description: `Fill missing values in "${column}" with mean (${missingCount} values, ${(missingCount / dataset.data.length * 100).toFixed(2)}% missing)`
              });
            } else {
              insights.suggestions.push({
                type: 'fill_missing',
                column: column,
                method: 'mode',
                priority: 'medium',
                description: `Fill missing values in "${column}" with most common value (${missingCount} values, ${(missingCount / dataset.data.length * 100).toFixed(2)}% missing)`
              });
            }
          } else if (missingCount / dataset.data.length > 0.7) {
            // Mostly missing - consider dropping
            insights.suggestions.push({
              type: 'drop_column',
              column: column,
              priority: 'high',
              description: `Consider dropping "${column}" (${(missingCount / dataset.data.length * 100).toFixed(2)}% missing, not enough data to be reliable)`
            });
          }
        }
      });
      
      // 2. Check for data type inconsistencies with more detail
      dataset.columns.forEach(column => {
        // Get non-null/undefined values for analysis
        const validValues = dataset.data
          .map(row => row[column])
          .filter(val => val !== null && val !== undefined && val !== '');
        
        if (validValues.length === 0) return;
        
        // Skip columns that appear to be date fields based on name
        const dateNamePatterns = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'birth', 'registration'];
        const isLikelyDateColumn = dateNamePatterns.some(pattern => 
          column.toLowerCase().includes(pattern)
        );
        
        // Check for date-like strings first
        const dateRegexes = [
          /^\d{1,4}[-\/]\d{1,2}[-\/]\d{1,4}$/, // yyyy-mm-dd or dd/mm/yyyy
          /^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}$/, // dd-mm-yyyy or mm/dd/yy
          /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/, // ISO datetime
          /^\w+ \d{1,2}, \d{4}$/ // Month Day, Year
        ];
        
        const dateCount = validValues.filter(val => {
          if (typeof val !== 'string') return false;
          return dateRegexes.some(regex => regex.test(val));
        }).length;
        const datePercentage = dateCount / validValues.length;
        
        // Always prioritize date detection for columns with date-like names
        if ((datePercentage > 0.5) || (isLikelyDateColumn && datePercentage > 0.2)) {
          insights.datatypeIssues.push({
            column,
            currentType: 'string',
            suggestedType: 'datetime',
            confidence: isLikelyDateColumn ? Math.max(0.9, datePercentage) : datePercentage,
            reason: isLikelyDateColumn ? 
              "Column name suggests date and values match date patterns" : 
              "Values appear to be dates"
          });
          
          insights.suggestions.push({
            type: 'change_type',
            column: column,
            new_type: 'datetime',
            priority: isLikelyDateColumn ? 'high' : 'medium',
            description: `Convert "${column}" to date/time format (${(datePercentage * 100).toFixed(0)}% match date patterns)`
          });
          
          // Skip other type checks for date columns
          return;
        }
        
        // Only check for numeric conversion if not a date-like column
        if (!isLikelyDateColumn) {
          // Check numeric columns that might be incorrectly stored as strings
          const numericCount = validValues.filter(val => !isNaN(parseFloat(val))).length;
          const numericPercentage = numericCount / validValues.length;
          
          // Add type conversion suggestions based on patterns
          if (numericPercentage > 0.9 && validValues.some(val => typeof val === 'string')) {
            // Column looks numeric but contains strings
            const hasDecimals = validValues.some(val => String(val).includes('.'));
            
            insights.datatypeIssues.push({
              column,
              currentType: 'mixed',
              suggestedType: hasDecimals ? 'float' : 'int',
              confidence: numericPercentage,
              reason: "Values are primarily numeric"
            });
            
            insights.suggestions.push({
              type: 'change_type',
              column: column,
              new_type: hasDecimals ? 'float' : 'int',
              priority: 'high',
              description: `Convert "${column}" to ${hasDecimals ? 'decimal' : 'integer'} type (${(numericPercentage * 100).toFixed(0)}% are numeric values)`
            });
          }
        }
      });
      
      // 3. Check for duplicate rows
      const stringifiedRows = dataset.data.map(row => JSON.stringify(row));
      const uniqueRows = new Set(stringifiedRows);
      insights.duplicateRows = dataset.data.length - uniqueRows.size;
      
      if (insights.duplicateRows > 0) {
        const dupePercentage = (insights.duplicateRows / dataset.data.length * 100).toFixed(2);
        insights.suggestions.push({
          type: 'remove_duplicates',
          priority: insights.duplicateRows > 5 ? 'high' : 'medium',
          description: `Remove ${insights.duplicateRows} duplicate rows (${dupePercentage}% of data)`
        });
      }
      
      // 4. Calculate an overall data quality score (0-100)
      const missingPercentage = totalMissingCount / (dataset.data.length * dataset.columns.length);
      const typeIssuesPercentage = insights.datatypeIssues.length / dataset.columns.length;
      const dupePercentage = insights.duplicateRows / dataset.data.length;
      
      insights.qualityScore = Math.round(100 * (1 - (0.5 * missingPercentage + 0.3 * typeIssuesPercentage + 0.2 * dupePercentage)));
      
      // 5. Sort suggestions by priority
      insights.suggestions.sort((a, b) => {
        const priorityOrder = { high: 0, medium: 1, low: 2 };
        return priorityOrder[a.priority] - priorityOrder[b.priority];
      });
      
      setDataInsights(insights);
      
      // Add initial analysis message with specific insights
      let initialMessage = `I've analyzed your dataset "${dataset.filename}" with ${insights.rowCount} rows and ${insights.columnCount} columns.\n\n`;
      
      if (insights.qualityScore < 70) {
        initialMessage += `Your data quality score is ${insights.qualityScore}/100. I've detected several issues that should be addressed.\n\n`;
      } else if (insights.qualityScore < 90) {
        initialMessage += `Your data quality score is ${insights.qualityScore}/100. There are a few issues you might want to fix.\n\n`;
      } else {
        initialMessage += `Your data quality score is ${insights.qualityScore}/100. Your data looks quite clean!\n\n`;
      }
      
      if (Object.keys(insights.missingValues).length > 0) {
        initialMessage += `📊 Found missing values in ${Object.keys(insights.missingValues).length} columns.\n`;
      }
      
      if (insights.datatypeIssues.length > 0) {
        initialMessage += `🔄 Detected ${insights.datatypeIssues.length} columns with inconsistent data types.\n`;
      }
      
      if (insights.duplicateRows > 0) {
        initialMessage += `🔍 Found ${insights.duplicateRows} duplicate rows.\n`;
      }
      
      if (insights.suggestions.length > 0) {
        initialMessage += `\nHere are my top recommendations:\n`;
        insights.suggestions.slice(0, 3).forEach((suggestion, idx) => {
          initialMessage += `${idx + 1}. ${suggestion.description}\n`;
        });
        
        initialMessage += `\nWould you like me to help you apply any of these transformations?`;
      }
      
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: initialMessage,
          insights: true
        }
      ]);
      
    } catch (error) {
      console.error('Error analyzing data:', error);
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: 'I encountered an error while analyzing your data. Please try again.'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!query.trim() || loading || !dataset || !dataset.dataset_id) return;
    
    const userMessage = query.trim();
    setQuery('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    
    setLoading(true);
    
    try {
      // Check if we can handle the query locally with advanced analysis
      let localResponse = null;
      let suggestedOperation = null;
      
      // Analyze request for local processing
      const lowercaseMsg = userMessage.toLowerCase();
      
      // Handle missing values questions
      if (lowercaseMsg.includes('missing') || lowercaseMsg.includes('null') || lowercaseMsg.includes('empty')) {
        if (Object.keys(dataInsights?.missingValues || {}).length > 0) {
          // Format a detailed response about missing values
          localResponse = `I found missing values in ${Object.keys(dataInsights.missingValues).length} columns:\n\n`;
          
          // Sort columns by missing percentage
          const sortedColumns = Object.entries(dataInsights.missingValues)
            .sort(([, a], [, b]) => parseFloat(b.percentage) - parseFloat(a.percentage));
          
          sortedColumns.forEach(([column, info], index) => {
            localResponse += `${index + 1}. **${column}**: ${info.count} missing values (${info.percentage}% of data)\n`;
            
            // Add recommendation for each column
            if (parseFloat(info.percentage) > 70) {
              localResponse += `   → Consider dropping this column as it's mostly empty\n`;
            } else if (parseFloat(info.percentage) < 5) {
              localResponse += `   → These appear to be rare missing entries, filling with 0 or mean should work\n`;
            } else {
              localResponse += `   → Moderate amount of missing data, statistical imputation recommended\n`;
            }
          });
          
          // Add action suggestion if there's a high priority missing data issue
          const highPriorityColumn = dataInsights.suggestions.find(
            s => s.type === 'fill_missing' && s.priority === 'high'
          );
          
          if (highPriorityColumn) {
            localResponse += `\nWould you like me to fill the missing values in "${highPriorityColumn.column}" with ${highPriorityColumn.method === 'value' ? 'zeros' : highPriorityColumn.method}?`;
            
            suggestedOperation = {
              type: 'fill_na',
              column: highPriorityColumn.column,
              method: highPriorityColumn.method,
              value: highPriorityColumn.value
            };
          }
        } else {
          localResponse = "Good news! I didn't find any missing values in your dataset.";
        }
      }
      // Handle data type questions
      else if (lowercaseMsg.includes('type') || lowercaseMsg.includes('format') || lowercaseMsg.includes('convert')) {
        if (dataInsights?.datatypeIssues?.length > 0) {
          localResponse = `I found ${dataInsights.datatypeIssues.length} columns with potential data type issues:\n\n`;
          
          dataInsights.datatypeIssues.forEach((issue, index) => {
            const confidence = Math.round(issue.confidence * 100);
            localResponse += `${index + 1}. **${issue.column}**: Currently appears as ${issue.currentType}, but ${confidence}% of values suggest it should be ${issue.suggestedType}\n`;
            if (issue.reason) {
              localResponse += `   → Reason: ${issue.reason}\n`;
            }
          });
          
          // Find highest confidence data type issue that isn't trying to convert dates to numbers
          const sortedIssues = [...dataInsights.datatypeIssues]
            .sort((a, b) => b.confidence - a.confidence);
          
          // Avoid suggesting numeric conversion for date-like field names
          const dateNamePatterns = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'birth', 'registration'];
          const highConfidenceIssue = sortedIssues.find(issue => {
            // For fields with date-like names, only suggest datetime conversion
            const isDateName = dateNamePatterns.some(pattern => 
              issue.column.toLowerCase().includes(pattern)
            );
            
            if (isDateName) {
              return issue.suggestedType === 'datetime';
            }
            
            // For other fields, consider any high-confidence suggestion
            return true;
          });
            
          if (highConfidenceIssue) {
            localResponse += `\nWould you like me to convert "${highConfidenceIssue.column}" to ${highConfidenceIssue.suggestedType} type?`;
            
            suggestedOperation = {
              type: 'change_type',
              column: highConfidenceIssue.column,
              new_type: highConfidenceIssue.suggestedType
            };
          }
        } else {
          localResponse = "I didn't detect any data type inconsistencies in your columns.";
        }
      }
      // Handle duplicate questions
      else if (lowercaseMsg.includes('duplicate') || lowercaseMsg.includes('unique')) {
        if (dataInsights?.duplicateRows > 0) {
          const percentage = ((dataInsights.duplicateRows / dataInsights.rowCount) * 100).toFixed(1);
          localResponse = `I found ${dataInsights.duplicateRows} duplicate rows in your dataset (${percentage}% of total data).\n\n`;
          
          if (dataInsights.duplicateRows > 5) {
            localResponse += "These duplicates could affect your analysis results and should be removed.";
            
            suggestedOperation = {
              type: 'remove_duplicates'
            };
          } else {
            localResponse += "This is a relatively small number of duplicates, but you might still want to clean them up for more accurate analysis.";
            
            suggestedOperation = {
              type: 'remove_duplicates'
            };
          }
        } else {
          localResponse = "Good news! I didn't find any duplicate rows in your dataset.";
        }
      }
      // Handle quality assessment
      else if (lowercaseMsg.includes('quality') || lowercaseMsg.includes('problems') || lowercaseMsg.includes('issues')) {
        if (dataInsights?.qualityScore !== undefined) {
          localResponse = `Your dataset quality score is ${dataInsights.qualityScore}/100.\n\n`;
          
          if (dataInsights.qualityScore < 70) {
            localResponse += "Issues I've identified:\n";
            if (Object.keys(dataInsights.missingValues || {}).length > 0) {
              const totalMissing = Object.values(dataInsights.missingValues).reduce((sum, col) => sum + col.count, 0);
              localResponse += `• Missing values: ${totalMissing} cells across ${Object.keys(dataInsights.missingValues).length} columns\n`;
            }
            if (dataInsights.datatypeIssues?.length > 0) {
              localResponse += `• Data type issues: ${dataInsights.datatypeIssues.length} columns with inconsistent formats\n`;
            }
            if (dataInsights.duplicateRows > 0) {
              localResponse += `• Duplicate rows: ${dataInsights.duplicateRows} rows (${((dataInsights.duplicateRows / dataInsights.rowCount) * 100).toFixed(1)}% of data)\n`;
            }
            
            // Suggest the highest priority action
            const topSuggestion = dataInsights.suggestions[0];
            if (topSuggestion) {
              localResponse += `\nMy top recommendation is: ${topSuggestion.description}`;
              
              if (topSuggestion.type === 'fill_missing') {
                suggestedOperation = {
                  type: 'fill_na',
                  column: topSuggestion.column,
                  method: topSuggestion.method || 'value',
                  value: topSuggestion.value || '0'
                };
              } else if (topSuggestion.type === 'drop_column') {
                suggestedOperation = {
                  type: 'drop_columns',
                  columns: [topSuggestion.column]
                };
              } else if (topSuggestion.type === 'change_type') {
                suggestedOperation = {
                  type: 'change_type',
                  column: topSuggestion.column,
                  new_type: topSuggestion.new_type || 'string'
                };
              } else if (topSuggestion.type === 'remove_duplicates') {
                suggestedOperation = {
                  type: 'remove_duplicates'
                };
              }
            }
          } else if (dataInsights.qualityScore < 90) {
            localResponse += "Your data has moderate quality with a few issues that could be addressed.";
            
            if (dataInsights.suggestions.length > 0) {
              localResponse += `\n\nI recommend: ${dataInsights.suggestions[0].description}`;
            }
          } else {
            localResponse += "Your data appears to be of high quality! There are few or no issues that need addressing.";
          }
        }
      }
      // Handle recommendations/suggestions
      else if (lowercaseMsg.includes('suggest') || lowercaseMsg.includes('recommend') || lowercaseMsg.includes('fix') || lowercaseMsg.includes('clean')) {
        if (dataInsights?.suggestions?.length > 0) {
          localResponse = "Here are my recommendations to improve your data quality:\n\n";
          
          // Group suggestions by priority
          const highPriority = dataInsights.suggestions.filter(s => s.priority === 'high');
          const mediumPriority = dataInsights.suggestions.filter(s => s.priority === 'medium');
          
          if (highPriority.length > 0) {
            localResponse += "🔴 High Priority:\n";
            highPriority.forEach((suggestion, idx) => {
              localResponse += `${idx + 1}. ${suggestion.description}\n`;
            });
            localResponse += "\n";
          }
          
          if (mediumPriority.length > 0) {
            localResponse += "🟠 Medium Priority:\n";
            mediumPriority.forEach((suggestion, idx) => {
              localResponse += `${idx + 1}. ${suggestion.description}\n`;
            });
          }
          
          // Suggest the highest priority action
          const topSuggestion = dataInsights.suggestions[0];
          if (topSuggestion) {
            localResponse += `\nShould I apply the first recommendation for you?`;
            
            if (topSuggestion.type === 'fill_missing') {
              suggestedOperation = {
                type: 'fill_na',
                column: topSuggestion.column,
                method: topSuggestion.method || 'value',
                value: topSuggestion.value || '0'
              };
            } else if (topSuggestion.type === 'drop_column') {
              suggestedOperation = {
                type: 'drop_columns',
                columns: [topSuggestion.column]
              };
            } else if (topSuggestion.type === 'change_type') {
              suggestedOperation = {
                type: 'change_type',
                column: topSuggestion.column,
                new_type: topSuggestion.new_type || 'string'
              };
            } else if (topSuggestion.type === 'remove_duplicates') {
              suggestedOperation = {
                type: 'remove_duplicates'
              };
            }
          }
        } else {
          localResponse = "Your data already looks clean! I don't have any specific recommendations for improvement.";
        }
      }
      
      // If we can handle locally, add the response directly
      if (localResponse) {
        setMessages(prev => [
          ...prev, 
          { 
            role: 'assistant', 
            content: localResponse,
            operation: suggestedOperation
          }
        ]);
        setLoading(false);
        return;
      }
      
      // Otherwise, call the backend API for LLM processing
      const response = await axios.post(
        `/ai-assistant/${dataset.dataset_id}`,
        { query: userMessage }
      );
      
      const assistantResponse = response.data.response;
      const operationFromBackend = response.data.suggested_operation;
      const insights = response.data.insights;
      
      // Update dataInsights with any new insights
      if (insights) {
        setDataInsights(prevInsights => ({
          ...prevInsights,
          ...(insights.rowCount && { rowCount: insights.rowCount }),
          ...(insights.columnCount && { columnCount: insights.columnCount }),
          ...(insights.missingValues && { missingValues: insights.missingValues }),
          ...(insights.suggestions && { suggestions: insights.suggestions }),
          ...(insights.numericColumns && { numericStats: insights.numericColumns }),
        }));
      }
      
      // Add assistant response to chat
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: assistantResponse,
          operation: operationFromBackend,
          insights: insights
        }
      ]);
    } catch (error) {
      console.error('Error processing query:', error);
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: 'I encountered an error while processing your request. Please try again.'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyOperation = (operation) => {
    if (onTransform && operation) {
      onTransform(operation);
      
      // Add confirmation message
      setMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          content: `I've applied the transformation. Is there anything else you'd like me to help with?`
        }
      ]);
    }
  };

  const generateInsights = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`/api/generate-ai-insights?dataset_id=${selectedDataset}&analysis_type=${analysisType}`);
      
      if (response.data.success) {
        setInsights(response.data.insights);
      } else {
        setError(response.data.error || 'Failed to generate insights');
      }
    } catch (err) {
      console.error('Error generating AI insights:', err);
      const errorMessage = err.response?.data?.error || err.message;
      setError(`Error: ${errorMessage}`);
      
      // Show specific message for OpenAI API issues
      if (errorMessage.includes('OpenAI') || errorMessage.includes('API key')) {
        setError(`OpenAI API error: ${errorMessage}. Please check your API key in the backend.`);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleAnalysisTypeChange = (event) => {
    setAnalysisType(event.target.value);
    // Clear previous insights when changing analysis type
    setInsights(null);
  };

  const renderInsightContent = () => {
    if (loading) {
      return (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      );
    }

    if (error) {
      return (
        <Box my={2} p={2} bgcolor="error.light" color="error.contrastText" borderRadius={1}>
          <Typography variant="body2">{error}</Typography>
        </Box>
      );
    }

    if (!insights) {
      return (
        <Box my={2} textAlign="center">
          <Typography color="textSecondary" variant="body2">
            Select an analysis type and click "Generate Insights" to get AI-powered recommendations for your dataset.
          </Typography>
        </Box>
      );
    }

    return (
      <Card variant="outlined" sx={{ mt: 2, backgroundColor: 'rgba(25, 118, 210, 0.05)' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {analysisType === 'general' && 'Data Analysis Insights'}
            {analysisType === 'cleaning' && 'Data Cleaning Recommendations'}
            {analysisType === 'insights' && 'Advanced Analytical Insights'}
          </Typography>
          <Box sx={{ 
            maxHeight: '400px', 
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            pr: 1
          }}>
            <Typography variant="body2" component="div" sx={{ whiteSpace: 'pre-line' }}>
              {insights}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="subtitle1" gutterBottom>
        <LightbulbIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
        AI Data Assistant
      </Typography>
      
      {/* Chat messages */}
      <Box sx={{ 
        flexGrow: 1, 
        overflow: 'auto', 
        mb: 2,
        bgcolor: 'background.default',
        borderRadius: 1,
        p: 1
      }}>
        <List>
          {messages.map((message, index) => (
            <ListItem 
              key={index}
              alignItems="flex-start"
              sx={{ 
                flexDirection: 'column',
                alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
                mb: 1
              }}
            >
              <Box 
                sx={{ 
                  maxWidth: '85%',
                  backgroundColor: message.role === 'user' ? 'primary.main' : 'background.paper',
                  color: message.role === 'user' ? 'white' : 'text.primary',
                  borderRadius: 2,
                  p: 1.5,
                  boxShadow: 1
                }}
              >
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {message.content}
                </Typography>
                
                {message.insights && dataInsights && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                      Dataset Summary
                    </Typography>
                    
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                      <Chip 
                        size="small" 
                        icon={<AnalyticsIcon />} 
                        label={`${dataInsights.rowCount} rows`} 
                        color="primary" 
                        variant="outlined" 
                      />
                      <Chip 
                        size="small" 
                        label={`${dataInsights.columnCount} columns`} 
                        color="primary" 
                        variant="outlined" 
                      />
                      {Object.keys(dataInsights.missingValues || {}).length > 0 && (
                        <Chip 
                          size="small" 
                          label={`${Object.keys(dataInsights.missingValues).length} cols with missing values`} 
                          color="warning" 
                          variant="outlined" 
                        />
                      )}
                      {dataInsights.qualityScore !== undefined && (
                        <Chip 
                          size="small" 
                          label={`Quality Score: ${dataInsights.qualityScore}/100`} 
                          color={dataInsights.qualityScore > 80 ? "success" : dataInsights.qualityScore > 60 ? "warning" : "error"} 
                          variant="outlined" 
                        />
                      )}
                    </Box>
                    
                    {dataInsights.suggestions && dataInsights.suggestions.length > 0 && (
                      <>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, mb: 0.5 }}>
                          Recommended actions:
                        </Typography>
                        {dataInsights.suggestions.slice(0, 3).map((suggestion, idx) => (
                          <Box key={idx} sx={{ mb: 1 }}>
                            <Chip 
                              size="small" 
                              icon={<AutoFixHighIcon />} 
                              label={suggestion.description.length > 60 ? suggestion.description.substring(0, 60) + '...' : suggestion.description} 
                              color={suggestion.priority === 'high' ? "error" : suggestion.priority === 'medium' ? "warning" : "success"} 
                              variant="outlined"
                              sx={{ mb: 0.5 }}
                              onClick={() => {
                                let operation = null;
                                
                                if (suggestion.type === 'fill_missing') {
                                  operation = {
                                    type: 'fill_na',
                                    column: suggestion.column,
                                    method: suggestion.method || 'value',
                                    value: suggestion.value || '0'
                                  };
                                } else if (suggestion.type === 'drop_column') {
                                  operation = {
                                    type: 'drop_columns',
                                    columns: [suggestion.column]
                                  };
                                } else if (suggestion.type === 'change_type') {
                                  operation = {
                                    type: 'change_type',
                                    column: suggestion.column,
                                    new_type: suggestion.new_type || 'string'
                                  };
                                } else if (suggestion.type === 'remove_duplicates') {
                                  operation = {
                                    type: 'remove_duplicates'
                                  };
                                }
                                
                                if (operation) {
                                  handleApplyOperation(operation);
                                }
                              }}
                            />
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 1, fontSize: '0.7rem' }}>
                              Priority: {suggestion.priority === 'high' ? "High ⚠️" : suggestion.priority === 'medium' ? "Medium" : "Low"}
                            </Typography>
                          </Box>
                        ))}
                        {dataInsights.suggestions.length > 3 && (
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, fontStyle: 'italic' }}>
                            {dataInsights.suggestions.length - 3} more suggestions available. Ask me for details.
                          </Typography>
                        )}
                      </>
                    )}
                  </Box>
                )}
                
                {message.operation && (
                  <Button 
                    variant="contained" 
                    size="small" 
                    startIcon={<AutoFixHighIcon />}
                    sx={{ mt: 1 }}
                    onClick={() => handleApplyOperation(message.operation)}
                  >
                    Apply Transformation
                  </Button>
                )}
              </Box>
              
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                {message.role === 'user' ? 'You' : 'AI Assistant'}
              </Typography>
            </ListItem>
          ))}
          <div ref={messagesEndRef} />
        </List>
      </Box>
      
      {/* Input area */}
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <TextField
          fullWidth
          variant="outlined"
          size="small"
          placeholder="Ask about your data or request transformations..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSendMessage();
            }
          }}
          disabled={loading}
          sx={{ mr: 1 }}
        />
        <IconButton 
          color="primary" 
          onClick={handleSendMessage}
          disabled={loading || !query.trim()}
        >
          {loading ? <CircularProgress size={24} /> : <SendIcon />}
        </IconButton>
      </Box>
      
      <Box mb={2}>
        <FormControl fullWidth variant="outlined" size="small" disabled={disabled || loading}>
          <InputLabel>Analysis Type</InputLabel>
          <Select
            value={analysisType}
            onChange={handleAnalysisTypeChange}
            label="Analysis Type"
          >
            <MenuItem value="general">General Analysis</MenuItem>
            <MenuItem value="cleaning">Data Cleaning</MenuItem>
            <MenuItem value="insights">Advanced Insights</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
      <Box mb={2}>
        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={generateInsights}
          disabled={disabled || loading || !selectedDataset}
          startIcon={
            analysisType === 'general' ? <AIIcon /> : 
            analysisType === 'cleaning' ? <CleaningIcon /> : 
            <InsightsIcon />
          }
        >
          Generate Insights
        </Button>
      </Box>
      
      {renderInsightContent()}
    </Box>
  );
};

export default AIAssistant; 
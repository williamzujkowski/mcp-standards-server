import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import StandardsBrowser from './pages/StandardsBrowser';
import StandardDetail from './pages/StandardDetail';
import Search from './pages/Search';
import RuleTesting from './pages/RuleTesting';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { StandardsProvider } from './contexts/StandardsContext';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <WebSocketProvider>
        <StandardsProvider>
          <Router>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/standards" element={<StandardsBrowser />} />
                <Route path="/standards/:id" element={<StandardDetail />} />
                <Route path="/search" element={<Search />} />
                <Route path="/testing" element={<RuleTesting />} />
              </Routes>
            </Layout>
          </Router>
        </StandardsProvider>
      </WebSocketProvider>
      <Toaster position="bottom-right" />
    </ThemeProvider>
  );
}

export default App;

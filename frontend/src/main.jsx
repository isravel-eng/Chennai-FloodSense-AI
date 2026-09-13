import React from 'react';
import { createRoot } from 'react-dom/client';
import 'leaflet/dist/leaflet.css';
import './styles.css';
import { WrappedApp } from './App';

// WrappedApp includes the AuthProvider so session state is available
// to all child components including the auth gate.
createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <WrappedApp />
  </React.StrictMode>
);

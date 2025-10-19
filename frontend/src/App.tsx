import { useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import AnalyticsPage from './pages/AnalyticsPage';

export default function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <div className="app">
      <nav className="navbar">
        <div className="navbar-content">
          <div className="brand">
            <svg className="brand-icon" width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path d="M16 4L4 10L16 16L28 10L16 4Z" fill="url(#gradient1)" />
              <path d="M4 16L16 22L28 16" stroke="url(#gradient2)" strokeWidth="2" strokeLinecap="round" />
              <path d="M4 22L16 28L28 22" stroke="url(#gradient2)" strokeWidth="2" strokeLinecap="round" />
              <defs>
                <linearGradient id="gradient1" x1="4" y1="4" x2="28" y2="16">
                  <stop stopColor="#6366F1" />
                  <stop offset="1" stopColor="#8B5CF6" />
                </linearGradient>
                <linearGradient id="gradient2" x1="4" y1="16" x2="28" y2="28">
                  <stop stopColor="#6366F1" />
                  <stop offset="1" stopColor="#8B5CF6" />
                </linearGradient>
              </defs>
            </svg>
            <span className="brand-text">Ikarus</span>
          </div>
          
          <button 
            className="mobile-menu-toggle" 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            <span></span>
            <span></span>
            <span></span>
          </button>

          <div className={`nav-links ${mobileMenuOpen ? 'mobile-open' : ''}`}>
            <NavLink 
              to="/" 
              end 
              className={({ isActive }) => isActive ? 'link active' : 'link'}
              onClick={() => setMobileMenuOpen(false)}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              Recommend
            </NavLink>
            <NavLink 
              to="/analytics" 
              className={({ isActive }) => isActive ? 'link active' : 'link'}
              onClick={() => setMobileMenuOpen(false)}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18h18" />
                <path d="M18 17V9" />
                <path d="M13 17V5" />
                <path d="M8 17v-3" />
              </svg>
              Analytics
            </NavLink>
          </div>
        </div>
      </nav>
      <main className="container">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="footer-logo">Ikarus</span>
            <span className="footer-tagline">AI-Powered Furniture Discovery</span>
          </div>
          <div className="footer-info">
            <a 
              href="https://0504ankitsharma-ikarus.hf.space/docs" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link"
            >
              API Docs: https://0504ankitsharma-ikarus.hf.space/docs
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
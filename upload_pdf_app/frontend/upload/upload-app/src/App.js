import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

function App() {
  const [currentStep, setCurrentStep] = useState('upload'); 
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [crawledPDFs, setCrawledPDFs] = useState([]);
  const [excludedPDFs, setExcludedPDFs] = useState(new Set());
  const [crawlProgress, setCrawlProgress] = useState(0);
  const [error, setError] = useState(null);
  const [crawlMessage, setCrawlMessage] = useState('');

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploadedFiles(files);
    setCurrentStep('crawling');
    setError(null);
    setCrawlProgress(0);
    setCrawlMessage('Uploading files...');

    try {
      await startWebCrawling(files);
    } catch (err) {
      setError('Failed to start web crawling: ' + err.message);
      setCurrentStep('upload');
    }
  };

  const startWebCrawling = async (files) => {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    try {
      setCrawlMessage('Starting web crawler...');
      setCrawlProgress(20);
      
      const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData
      });

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        throw new Error(`Upload failed: ${errorText}`);
      }

      const result = await uploadResponse.json();
      console.log('Upload result:', result);
      
      setCrawlProgress(50);
      setCrawlMessage('Processing links from PDFs...');
      
      // Wait a bit for processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setCrawlProgress(80);
      setCrawlMessage('Finalizing downloads...');
      
      // Load the crawled PDFs
      await loadCrawledPDFs();
      
    } catch (err) {
      console.error('Crawling error:', err);
      throw err;
    }
  };

  const loadCrawledPDFs = async () => {
    try {
      setCrawlProgress(90);
      setCrawlMessage('Loading results...');
      
      const response = await fetch(`${API_BASE_URL}/api/pdfs`);
      if (!response.ok) {
        throw new Error('Failed to load PDFs');
      }
      
      const pdfs = await response.json();
      console.log('Loaded PDFs:', pdfs);
      
      if (Array.isArray(pdfs)) {
        setCrawledPDFs(pdfs);
        setCrawlProgress(100);
        
        // Wait a moment then switch to review
        setTimeout(() => {
          setCurrentStep('review');
          setCrawlMessage('');
        }, 500);
      } else {
        setCrawledPDFs([]);
        setCurrentStep('review');
      }
      
    } catch (err) {
      setError('Failed to load crawled PDFs: ' + err.message);
      setCrawledPDFs([]);
      setCurrentStep('review');
    }
  };

  const togglePDFExclusion = async (pdfName) => {
    const newExcluded = new Set(excludedPDFs);
    const isExcluded = !newExcluded.has(pdfName);
    
    if (isExcluded) {
      newExcluded.add(pdfName);
    } else {
      newExcluded.delete(pdfName);
    }
    
    setExcludedPDFs(newExcluded);

    try {
      const response = await fetch(`${API_BASE_URL}/api/pdfs/${encodeURIComponent(pdfName)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          name: pdfName,
          excluded: isExcluded 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update PDF status');
      }
    } catch (err) {
      setError('Failed to update PDF: ' + err.message);
      setExcludedPDFs(excludedPDFs);
    }
  };

  const handleFinalSubmit = async () => {
    setCurrentStep('finalizing');
    setError(null);

    try {
      const moveResponse = await fetch(`${API_BASE_URL}/api/finalize`, {
        method: 'POST',
      });

      if (!moveResponse.ok) throw new Error('Failed to move PDFs');
      
      const result = await moveResponse.json();
      
      setTimeout(() => {
        setCurrentStep('upload');
        setUploadedFiles([]);
        setCrawledPDFs([]);
        setExcludedPDFs(new Set());
        setCrawlProgress(0);
        alert(result.message || 'Successfully uploaded PDFs to Knowledge Base!');
      }, 1000);
    } catch (err) {
      setError('Failed to finalize upload: ' + err.message);
      setCurrentStep('review');
    }
  };

  return (
    <div className="app-container">
      <div className="main-content">
        <header className="app-header">
          <h1>DB Upload Menu</h1>
          <p className="subtitle">Upload PDFs To Have Them Crawled</p>
        </header>

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            {error}
            <button onClick={() => setError(null)} className="close-error">×</button>
          </div>
        )}

        {currentStep === 'upload' && (
          <div className="upload-container">
            <div className="upload-box">
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <h2>Upload PDF Files</h2>
              <p>Select one or more PDF files to start web crawling</p>
              <label htmlFor="file-upload" className="upload-button">
                Choose Files
              </label>
              <input
                id="file-upload"
                type="file"
                multiple
                accept=".pdf"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </div>
          </div>
        )}

        {currentStep === 'crawling' && (
          <div className="crawling-container">
            <div className="loading-spinner"></div>
            <h2>Web Crawling in Progress</h2>
            <p>{crawlMessage || 'Extracting links and downloading PDFs from your uploaded files...'}</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${crawlProgress}%` }}></div>
            </div>
            <span className="progress-text">{crawlProgress}%</span>
            <div className="uploaded-files">
              <h3>Processing Files:</h3>
              {uploadedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  📄 {file.name}
                </div>
              ))}
            </div>
          </div>
        )}

        {currentStep === 'review' && (
          <div className="review-container">
            <h2>Review Crawled PDFs</h2>
            <p>Click the ❌ to exclude PDFs you don't want in your knowledge base</p>
            
            <div className="pdf-list">
              {crawledPDFs && crawledPDFs.length > 0 ? (
                crawledPDFs.map((pdf) => (
                  <div
                    key={pdf.name}
                    className={`pdf-item ${excludedPDFs.has(pdf.name) ? 'excluded' : ''}`}
                  >
                    <div className="pdf-info">
                      {pdf.preview_url && (
                        <img 
                          src={`${API_BASE_URL}${pdf.preview_url}`} 
                          alt={pdf.name}
                          className="pdf-thumbnail"
                        />
                      )}
                      <div className="pdf-details">
                        <span className="pdf-name">{pdf.name}</span>
                        <span className="pdf-size">{pdf.size_kb} KB</span>
                      </div>
                    </div>
                    <button
                      className="exclude-button"
                      onClick={() => togglePDFExclusion(pdf.name)}
                      title={excludedPDFs.has(pdf.name) ? 'Include' : 'Exclude'}
                    >
                      {excludedPDFs.has(pdf.name) ? '✓' : '❌'}
                    </button>
                  </div>
                ))
              ) : (
                <p>No PDFs found. Try uploading PDFs with links in them.</p>
              )}
            </div>

            <div className="action-buttons">
              <div className="selected-count">
                {crawledPDFs.length - excludedPDFs.size} of {crawledPDFs.length} PDFs selected
              </div>
              <button
                className="submit-button"
                onClick={handleFinalSubmit}
                disabled={crawledPDFs.length === 0 || crawledPDFs.length - excludedPDFs.size === 0}
              >
                Upload to Knowledge Base
              </button>
            </div>
          </div>
        )}

        {currentStep === 'finalizing' && (
          <div className="finalizing-container">
            <div className="loading-spinner"></div>
            <h2>Finalizing Upload</h2>
            <p>Moving selected PDFs to your knowledge base...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
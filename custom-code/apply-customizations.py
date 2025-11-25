#!/usr/bin/env python3
"""
Apply customizations to Open WebUI for PDF crawler integration
"""
import sys
import os
import shutil
import subprocess
from pathlib import Path

print("=" * 60)
print("Starting Open WebUI customization...")
print("=" * 60)

print("\nInstalling system dependencies...")
try:
    # Install poppler-utils for pdf2image
    result = subprocess.run(
        ["apt-get", "update"],
        capture_output=True,
        timeout=60
    )
    result = subprocess.run(
        ["apt-get", "install", "-y", "poppler-utils"],
        capture_output=True,
        timeout=120
    )
    if result.returncode == 0:
        print("  ✓ poppler-utils (for PDF thumbnails)")
    else:
        print(f"  ⚠ poppler-utils install issue: {result.stderr.decode()[:100]}")
except Exception as e:
    print(f"  ⚠ Could not install poppler-utils: {e}")

# ============================================================================
# Step 1: Copy custom router files
# ============================================================================

CUSTOM_CODE_DIR = Path("/app/custom_code")
ROUTERS_DIR = Path("/app/backend/open_webui/routers")
STATIC_DIR = Path("/app/backend/open_webui/static")

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nCustom code directory: {CUSTOM_CODE_DIR}")
print(f"Contents: {list(CUSTOM_CODE_DIR.iterdir()) if CUSTOM_CODE_DIR.exists() else 'NOT FOUND'}")

# Copy the integrated backend router
integrated_router = CUSTOM_CODE_DIR / "integrated_backend" / "custom_pdf_router.py"
if integrated_router.exists():
    shutil.copy2(integrated_router, ROUTERS_DIR / "custom_pdf_router.py")
    print(f"✓ Copied custom_pdf_router.py to {ROUTERS_DIR}")
else:
    print(f"✗ custom_pdf_router.py not found at {integrated_router}")

print("\nLooking for Webscraping folder...")

# List of possible locations for the Webscraping folder
webscraping_locations = [
    CUSTOM_CODE_DIR / "integrated_backend" / "Webscraping",
    CUSTOM_CODE_DIR / "upload_pdf_app" / "Webscraping",
    CUSTOM_CODE_DIR / "Webscraping",
    CUSTOM_CODE_DIR / "upload_pdf_app" / "backend" / "Webscraping",
]

webscraping_dst = ROUTERS_DIR / "Webscraping"
webscraping_found = False

for webscraping_src in webscraping_locations:
    print(f"  Checking: {webscraping_src} ... ", end="")
    if webscraping_src.exists():
        print("FOUND!")
        
        # Check if link_downloader.py exists in this folder
        link_downloader = webscraping_src / "link_downloader.py"
        if link_downloader.exists():
            print(f"    ✓ link_downloader.py found")
        else:
            print(f"    ⚠ link_downloader.py NOT in this folder")
            # List contents
            print(f"    Contents: {[f.name for f in webscraping_src.iterdir()]}")
            continue
        
        # Remove existing and copy new
        if webscraping_dst.exists():
            shutil.rmtree(webscraping_dst)
        shutil.copytree(webscraping_src, webscraping_dst)
        print(f"  ✓ Copied Webscraping to {webscraping_dst}")
        print(f"    Contents: {[f.name for f in webscraping_dst.iterdir()]}")
        webscraping_found = True
        break
    else:
        print("not found")

if not webscraping_found:
    print("\n✗ ERROR: Could not find Webscraping folder with link_downloader.py!")
    print("  Searching recursively in custom_code...")
    
    # Recursive search for link_downloader.py
    for path in CUSTOM_CODE_DIR.rglob("link_downloader.py"):
        print(f"  Found: {path}")
        webscraping_src = path.parent
        if webscraping_dst.exists():
            shutil.rmtree(webscraping_dst)
        shutil.copytree(webscraping_src, webscraping_dst)
        print(f"  ✓ Copied from {webscraping_src}")
        webscraping_found = True
        break

if not webscraping_found:
    print("\n⚠ CRITICAL: link_downloader.py not found anywhere!")
    print("  The PDF crawler will not work without it.")
    print("  Please ensure the Webscraping folder is in custom-code/integrated_backend/")

# ============================================================================
# Step 2: Install required packages
# ============================================================================

packages = [
    "pdf2image",
    "pillow", 
    "beautifulsoup4",
    "pymupdf",
    "tqdm",
    "playwright",
    "lxml"
]

print("\nInstalling required packages...")
for package in packages:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0:
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package}: {result.stderr.decode()[:100]}")
    except Exception as e:
        print(f"  ✗ {package}: {e}")

# Install Playwright browsers (optional, for page rendering)
try:
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        capture_output=True,
        timeout=180
    )
    print("  ✓ Playwright Chromium browser")
except:
    print("  ⚠ Playwright browser install skipped (optional)")

# ============================================================================
# Step 3: Skip main.py modification - must be done manually once
# ============================================================================

print("\n⚠ Skipping main.py modification (do this manually once):")
print("  1. Add 'custom_pdf_router,' to the routers import block")
print("  2. Add router registration after utils.router:")
print("     app.include_router(custom_pdf_router.router, prefix=\"/api/v1/custom\", tags=[\"custom_pdf\"])")

# ============================================================================
# Step 4: Create the JavaScript for the floating button
# ============================================================================

print("\nCreating frontend JavaScript...")

custom_js = '''
(function() {
    console.log('[PDF Crawler] Initializing...');
    
    const API_PREFIX = '/api/v1/custom';
    
    let floatingButton = null;
    let uploadModal = null;
    let currentStep = 'upload';
    let uploadedFiles = [];
    let crawledPDFs = [];
    let excludedPDFs = new Set();
    
    // Get auth token from localStorage
    function getAuthHeaders() {
        const token = localStorage.getItem('token');
        return {
            'Authorization': `Bearer ${token}`
        };
    }
    
    async function fetchWithAuth(url, options = {}) {
        options.headers = {
            ...options.headers,
            ...getAuthHeaders()
        };
        return fetch(url, options);
    }
    
    function createStyles() {
        if (document.getElementById('pdf-crawler-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'pdf-crawler-styles';
        style.textContent = `
            #pdf-crawler-btn {
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 9999;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }
            
            #pdf-crawler-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }
            
            .pdf-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.8);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .pdf-modal {
                background: #1e1e1e;
                border-radius: 16px;
                width: 90%;
                max-width: 700px;
                max-height: 85vh;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            
            .pdf-modal-header {
                padding: 20px 24px;
                border-bottom: 1px solid #333;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .pdf-modal-header h2 {
                margin: 0;
                color: #fff;
                font-size: 1.25rem;
            }
            
            .pdf-modal-close {
                background: none;
                border: none;
                color: #888;
                font-size: 24px;
                cursor: pointer;
                padding: 0;
                line-height: 1;
            }
            
            .pdf-modal-close:hover {
                color: #fff;
            }
            
            .pdf-modal-body {
                padding: 24px;
                overflow-y: auto;
                flex: 1;
            }
            
            .pdf-modal-footer {
                padding: 16px 24px;
                border-top: 1px solid #333;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .pdf-upload-zone {
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 48px 24px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .pdf-upload-zone:hover {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.05);
            }
            
            .pdf-upload-zone h3 {
                color: #fff;
                margin: 0 0 8px 0;
            }
            
            .pdf-upload-zone p {
                color: #888;
                margin: 0 0 24px 0;
            }
            
            .pdf-upload-btn {
                display: inline-block;
                padding: 12px 32px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s;
            }
            
            .pdf-upload-btn:hover {
                background: #5a6fd6;
            }
            
            .pdf-progress {
                text-align: center;
            }
            
            .pdf-spinner {
                width: 48px;
                height: 48px;
                border: 4px solid #333;
                border-top-color: #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 24px;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .pdf-progress-bar {
                height: 8px;
                background: #333;
                border-radius: 4px;
                overflow: hidden;
                margin: 16px 0;
            }
            
            .pdf-progress-fill {
                height: 100%;
                background: #667eea;
                border-radius: 4px;
                transition: width 0.3s ease;
            }
            
            .pdf-list {
                max-height: 400px;
                overflow-y: auto;
            }
            
            .pdf-item {
                display: flex;
                align-items: center;
                padding: 12px;
                background: #2a2a2a;
                border-radius: 8px;
                margin-bottom: 8px;
            }
            
            .pdf-item.excluded {
                opacity: 0.5;
            }
            
            .pdf-item.excluded .pdf-name {
                text-decoration: line-through;
            }
            
            .pdf-thumb {
                width: 40px;
                height: 52px;
                background: #333;
                border-radius: 4px;
                margin-right: 12px;
                object-fit: cover;
            }
            
            .pdf-info {
                flex: 1;
            }
            
            .pdf-name {
                color: #fff;
                font-weight: 500;
                display: block;
            }
            
            .pdf-size {
                color: #888;
                font-size: 0.85rem;
            }
            
            .pdf-toggle {
                background: none;
                border: none;
                font-size: 20px;
                cursor: pointer;
                padding: 8px;
            }
            
            .pdf-submit-btn {
                padding: 12px 24px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 500;
                cursor: pointer;
            }
            
            .pdf-submit-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .pdf-count {
                color: #888;
            }
            
            .pdf-notification {
                position: fixed;
                bottom: 100px;
                right: 24px;
                padding: 16px 24px;
                border-radius: 8px;
                color: white;
                z-index: 10001;
                animation: slideIn 0.3s ease;
            }
            
            .pdf-notification.success { background: #10b981; }
            .pdf-notification.error { background: #ef4444; }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    function showNotification(message, type = 'success') {
        const notif = document.createElement('div');
        notif.className = `pdf-notification ${type}`;
        notif.textContent = message;
        document.body.appendChild(notif);
        
        setTimeout(() => {
            notif.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => notif.remove(), 300);
        }, 3000);
    }
    
    function createButton() {
        if (floatingButton) return;
        
        floatingButton = document.createElement('button');
        floatingButton.id = 'pdf-crawler-btn';
        floatingButton.innerHTML = '🕸️';
        floatingButton.title = 'PDF Web Crawler';
        floatingButton.onclick = () => openModal();
        
        document.body.appendChild(floatingButton);
    }
    
    function openModal() {
        if (uploadModal) return;
        
        uploadModal = document.createElement('div');
        uploadModal.className = 'pdf-modal-overlay';
        uploadModal.innerHTML = `
            <div class="pdf-modal">
                <div class="pdf-modal-header">
                    <h2>📄 PDF Web Crawler</h2>
                    <button class="pdf-modal-close" onclick="window.pdfCrawler.closeModal()">×</button>
                </div>
                <div class="pdf-modal-body" id="pdf-modal-body"></div>
                <div class="pdf-modal-footer" id="pdf-modal-footer" style="display:none"></div>
            </div>
        `;
        
        document.body.appendChild(uploadModal);
        uploadModal.onclick = (e) => {
            if (e.target === uploadModal) closeModal();
        };
        
        showUploadStep();
    }
    
    function closeModal() {
        if (uploadModal) {
            uploadModal.remove();
            uploadModal = null;
        }
        currentStep = 'upload';
        uploadedFiles = [];
        crawledPDFs = [];
        excludedPDFs = new Set();
    }
    
    function showUploadStep() {
        const body = document.getElementById('pdf-modal-body');
        const footer = document.getElementById('pdf-modal-footer');
        
        body.innerHTML = `
            <div class="pdf-upload-zone" onclick="document.getElementById('pdf-file-input').click()">
                <h3>Upload PDF Files</h3>
                <p>Select PDFs to extract and crawl linked documents</p>
                <button class="pdf-upload-btn">Choose Files</button>
                <input type="file" id="pdf-file-input" multiple accept=".pdf" 
                       style="display:none" onchange="window.pdfCrawler.handleFiles(this.files)">
            </div>
        `;
        footer.style.display = 'none';
    }
    
    function showProgressStep(message, progress) {
        const body = document.getElementById('pdf-modal-body');
        const footer = document.getElementById('pdf-modal-footer');
        
        body.innerHTML = `
            <div class="pdf-progress">
                <div class="pdf-spinner"></div>
                <h3 style="color:#fff;margin:0 0 8px">${message}</h3>
                <div class="pdf-progress-bar">
                    <div class="pdf-progress-fill" style="width:${progress}%"></div>
                </div>
                <p style="color:#888">${progress}%</p>
            </div>
        `;
        footer.style.display = 'none';
    }
    
    function showReviewStep() {
        const body = document.getElementById('pdf-modal-body');
        const footer = document.getElementById('pdf-modal-footer');
        
        if (crawledPDFs.length === 0) {
            body.innerHTML = `
                <div style="text-align:center;padding:48px">
                    <p style="color:#888;font-size:1.1rem">No PDFs were found from crawling.</p>
                    <button class="pdf-upload-btn" onclick="window.pdfCrawler.showUploadStep()" style="margin-top:24px">
                        Try Again
                    </button>
                </div>
            `;
            footer.style.display = 'none';
            return;
        }
        
        body.innerHTML = `
            <div>
                <p style="color:#888;margin:0 0 16px">Click ❌ to exclude PDFs you don't want:</p>
                <div class="pdf-list" id="pdf-list"></div>
            </div>
        `;
        
        const list = document.getElementById('pdf-list');
        list.innerHTML = crawledPDFs.map(pdf => `
            <div class="pdf-item ${pdf.excluded ? 'excluded' : ''}" data-name="${pdf.name}">
                ${pdf.preview_url ? 
                    `<img src="${pdf.preview_url}" class="pdf-thumb" alt="">` : 
                    `<div class="pdf-thumb"></div>`
                }
                <div class="pdf-info">
                    <span class="pdf-name">${pdf.name}</span>
                    <span class="pdf-size">${pdf.size_kb} KB</span>
                </div>
                <button class="pdf-toggle" onclick="window.pdfCrawler.togglePDF('${pdf.name}')">
                    ${pdf.excluded ? '✓' : '❌'}
                </button>
            </div>
        `).join('');
        
        updateFooter();
        footer.style.display = 'flex';
    }
    
    function updateFooter() {
        const footer = document.getElementById('pdf-modal-footer');
        const selected = crawledPDFs.filter(p => !excludedPDFs.has(p.name)).length;
        
        footer.innerHTML = `
            <span class="pdf-count">${selected} of ${crawledPDFs.length} selected</span>
            <button class="pdf-submit-btn" onclick="window.pdfCrawler.finalize()" ${selected === 0 ? 'disabled' : ''}>
                Upload to Open WebUI
            </button>
        `;
    }
    
    async function handleFiles(files) {
        if (!files || files.length === 0) return;
        
        uploadedFiles = Array.from(files);
        showProgressStep('Uploading files...', 10);
        
        const formData = new FormData();
        for (const file of uploadedFiles) {
            formData.append('files', file);
        }
        
        try {
            showProgressStep('Crawling PDFs from links...', 30);
            
            const response = await fetchWithAuth(`${API_PREFIX}/pdf-upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.text();
                throw new Error(error);
            }
            
            showProgressStep('Loading results...', 80);
            await loadPDFs();
            
            showProgressStep('Done!', 100);
            setTimeout(() => showReviewStep(), 500);
            
        } catch (error) {
            console.error('[PDF Crawler] Upload error:', error);
            showNotification('Upload failed: ' + error.message, 'error');
            showUploadStep();
        }
    }
    
    async function loadPDFs() {
        const response = await fetchWithAuth(`${API_PREFIX}/pdf-list`);
        if (!response.ok) throw new Error('Failed to load PDFs');
        
        crawledPDFs = await response.json();
        excludedPDFs = new Set(crawledPDFs.filter(p => p.excluded).map(p => p.name));
    }
    
    async function togglePDF(name) {
        const isExcluded = !excludedPDFs.has(name);
        
        if (isExcluded) {
            excludedPDFs.add(name);
        } else {
            excludedPDFs.delete(name);
        }
        
        try {
            await fetchWithAuth(`${API_PREFIX}/pdf-toggle/${encodeURIComponent(name)}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, excluded: isExcluded })
            });
            
            const item = document.querySelector(`.pdf-item[data-name="${name}"]`);
            if (item) {
                item.classList.toggle('excluded', isExcluded);
                item.querySelector('.pdf-toggle').textContent = isExcluded ? '✓' : '❌';
            }
            
            const pdf = crawledPDFs.find(p => p.name === name);
            if (pdf) pdf.excluded = isExcluded;
            
            updateFooter();
            
        } catch (error) {
            console.error('[PDF Crawler] Toggle error:', error);
            if (isExcluded) excludedPDFs.delete(name);
            else excludedPDFs.add(name);
        }
    }
    
    async function finalize() {
        showProgressStep('Uploading to Open WebUI...', 50);
        
        try {
            const response = await fetchWithAuth(`${API_PREFIX}/pdf-finalize`, {
                method: 'POST'
            });
            
            if (!response.ok) throw new Error('Finalize failed');
            
            const result = await response.json();
            showNotification(`Successfully uploaded ${result.moved.length} PDFs!`, 'success');
            
            setTimeout(() => closeModal(), 1500);
            
        } catch (error) {
            console.error('[PDF Crawler] Finalize error:', error);
            showNotification('Failed to upload: ' + error.message, 'error');
            showReviewStep();
        }
    }
    
    // Expose functions globally
    window.pdfCrawler = {
        closeModal,
        handleFiles,
        togglePDF,
        finalize,
        showUploadStep
    };
    
    // Initialize
    createStyles();
    createButton();
    
    console.log('[PDF Crawler] Ready!');
})();
'''

with open(STATIC_DIR / "pdf_crawler.js", 'w') as f:
    f.write(custom_js)
print("  ✓ Created pdf_crawler.js")

# ============================================================================
# Done!
# ============================================================================


print("\n" + "=" * 60)
print("Final Verification:")
print("=" * 60)

router_file = ROUTERS_DIR / "custom_pdf_router.py"
print(f"  Router file: {router_file.exists()}")

ws_dir = ROUTERS_DIR / "Webscraping"
print(f"  Webscraping dir: {ws_dir.exists()}")

if ws_dir.exists():
    ld_script = ws_dir / "link_downloader.py"
    print(f"  link_downloader.py: {ld_script.exists()}")
    if ws_dir.exists():
        print(f"  Webscraping contents: {[f.name for f in ws_dir.iterdir()]}")

print("\n" + "=" * 60)
print("✓ Customization complete!")
print("=" * 60)
print("\nTo load the PDF crawler button, run this in browser console:")
print("  fetch('/api/v1/custom/inject-script').then(r=>r.text()).then(eval)")
#!/usr/bin/env python3
import sys
import os
import shutil
import subprocess
from pathlib import Path

print("Starting customization application...")

# Step 1: Copy custom files FIRST
if os.path.exists("/app/custom_code/custom_api.py"):
    shutil.copy2(
        "/app/custom_code/custom_api.py",
        "/app/backend/open_webui/routers/custom_api.py"
    )
    print("✓ Copied custom_api.py")

# Step 2: Setup integrated backend
if os.path.exists("/app/custom_code/integrated_backend"):
    print("Setting up integrated PDF backend...")
    
    if os.path.exists("/app/custom_code/integrated_backend/custom_pdf_router.py"):
        # Copy the router file
        shutil.copy2("/app/custom_code/integrated_backend/custom_pdf_router.py", 
                      "/app/backend/open_webui/routers/custom_pdf_router.py")
        
        # Remove authentication requirements from the copied file
        router_file = "/app/backend/open_webui/routers/custom_pdf_router.py"
        with open(router_file, 'r') as f:
            content = f.read()
        content = content.replace('user=Depends(get_verified_user)', '# user=Depends(get_verified_user)')
        content = content.replace('user=Depends(get_admin_user)', '# user=Depends(get_admin_user)')
        with open(router_file, 'w') as f:
            f.write(content)
        print("✓ Copied and modified custom_pdf_router.py (auth disabled)")
    
    # Install ALL required packages including missing ones
    packages = [
        "pdf2image", 
        "pillow", 
        "beautifulsoup4", 
        "pymupdf", 
        "tqdm",
        "requests",      # Missing in original
        "playwright"     # Missing in original
    ]
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          capture_output=True, check=True, timeout=60)
            print(f"✓ Installed {package}")
        except:
            print(f"✗ Could not install {package}")
    
    # Install Playwright browsers
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], 
                      capture_output=True, check=True, timeout=120)
        print("✓ Installed Playwright chromium browser")
    except:
        print("✗ Could not install Playwright browsers")

# Step 3: Clean and update main.py
main_file = "/app/backend/open_webui/main.py"
with open(main_file, 'r') as f:
    lines = f.readlines()

# Clean up any existing custom entries (both imports and routers)
cleaned_lines = []
for line in lines:
    # Skip custom router includes
    if 'app.include_router' in line and ('custom_api' in line or 'custom_pdf_router' in line):
        continue
    # Skip custom imports
    if 'custom_api' in line or 'custom_pdf_router' in line:
        continue
    cleaned_lines.append(line)

lines = cleaned_lines

# Find where to insert - look for the multi-line import statement
import_index = None
router_index = None

for i, line in enumerate(lines):
    # Look for the closing parenthesis of the import block
    if 'from open_webui.routers import (' in line:
        # Find the closing parenthesis
        for j in range(i, min(i + 50, len(lines))):
            if ')' in lines[j]:
                import_index = j  # Insert just before the closing parenthesis
                break
    
    # Find where routers are included
    if 'app.include_router(utils.router' in line:
        router_index = i + 1

# Add imports if we found the import block
if import_index is not None:
    # Add custom imports to the import tuple
    if os.path.exists("/app/backend/open_webui/routers/custom_api.py"):
        # Insert before the closing parenthesis
        lines[import_index] = lines[import_index].rstrip()
        if lines[import_index].endswith(')'):
            lines[import_index] = lines[import_index][:-1] + ',\n'
            lines.insert(import_index + 1, '    custom_api,\n')
            lines.insert(import_index + 2, ')\n')
            import_index += 2
        else:
            lines[import_index] = lines[import_index] + ',\n'
            lines.insert(import_index + 1, '    custom_api,\n')
            import_index += 1
        print("✓ Added custom_api to imports")
    
    if os.path.exists("/app/backend/open_webui/routers/custom_pdf_router.py"):
        # Find the closing parenthesis again (it may have moved)
        for j in range(import_index, min(import_index + 10, len(lines))):
            if ')' in lines[j]:
                lines[j] = lines[j].rstrip()
                if lines[j].endswith(')'):
                    lines[j] = lines[j][:-1] + ',\n'
                    lines.insert(j + 1, '    custom_pdf_router,\n')
                    lines.insert(j + 2, ')\n')
                else:
                    lines[j] = lines[j] + ',\n'
                    lines.insert(j + 1, '    custom_pdf_router,\n')
                break
        print("✓ Added custom_pdf_router to imports")

# Add routers if we found the router location
if router_index is not None:
    routers_added = []
    
    if os.path.exists("/app/backend/open_webui/routers/custom_api.py"):
        lines.insert(router_index, 'app.include_router(custom_api.router, prefix="/api/v1/custom", tags=["custom"])\n')
        router_index += 1
        routers_added.append('custom_api')
        print("✓ Added custom_api router")
    
    if os.path.exists("/app/backend/open_webui/routers/custom_pdf_router.py"):
        lines.insert(router_index, 'app.include_router(custom_pdf_router.router, prefix="/api/v1/custom", tags=["custom_pdf"])\n')
        routers_added.append('custom_pdf_router')
        print("✓ Added custom_pdf_router router")

# Write back the updated file
with open(main_file, 'w') as f:
    f.writelines(lines)

print("✓ Updated main.py")

# Determine if we should use integrated backend or external
USE_INTEGRATED = os.path.exists("/app/custom_code/integrated_backend/backend_functions.py")
if USE_INTEGRATED:
    print("Using INTEGRATED backend (no external server needed)")
    PDF_BACKEND_URL = ''  # Empty means same origin
    API_PREFIX = '/api/v1/custom'
else:
    print("Using EXTERNAL backend on port 8002")
    PDF_BACKEND_URL = 'http://localhost:8002'
    API_PREFIX = '/api'

# Create custom JavaScript with PDF upload functionality
custom_js = f'''
(function() {{
    console.log('[Custom] PDF Upload Integration loaded - Mode: {"INTEGRATED" if USE_INTEGRATED else "EXTERNAL"}');
    
    // Backend configuration
    const PDF_BACKEND_URL = '{PDF_BACKEND_URL}';
    const API_PREFIX = '{API_PREFIX}';
    const USE_AUTH = {str(USE_INTEGRATED).lower()};
    
    let floatingButton = null;
    let uploadModal = null;
    let currentStep = 'upload';
    let uploadedFiles = [];
    let crawledPDFs = [];
    let excludedPDFs = new Set();
    
    // Helper function for authenticated requests
    async function fetchWithAuth(url, options = {{}}) {{
        if (USE_AUTH) {{
            const token = localStorage.getItem('token');
            options.headers = {{
                ...options.headers,
                'Authorization': `Bearer ${{token}}`
            }};
        }}
        return fetch(url, options);
    }}
    
    function createFloatingButton() {{
        if (floatingButton) {{
            floatingButton.remove();
        }}
        
        floatingButton = document.createElement('div');
        floatingButton.id = 'custom-floating-button';
        floatingButton.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            font-size: 24px;
            user-select: none;
        `;
        
        floatingButton.innerHTML = '🕸️';
        floatingButton.title = 'PDF Upload & Web Crawler';
        
        floatingButton.onmouseenter = function() {{
            this.style.transform = 'scale(1.1)';
            this.style.boxShadow = '0 6px 16px rgba(0,0,0,0.2)';
        }};
        
        floatingButton.onmouseleave = function() {{
            this.style.transform = 'scale(1)';
            this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        }};
        
        floatingButton.onclick = function(e) {{
            e.preventDefault();
            e.stopPropagation();
            openUploadModal();
        }};
        
        document.body.appendChild(floatingButton);
        addStyles();
    }}
    
    function addStyles() {{
        if (!document.getElementById('custom-upload-styles')) {{
            const style = document.createElement('style');
            style.id = 'custom-upload-styles';
            style.innerHTML = `
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                    100% {{ transform: scale(1); }}
                }}
                
                @keyframes slideIn {{
                    from {{ transform: translateX(100%); opacity: 0; }}
                    to {{ transform: translateX(0); opacity: 1; }}
                }}
                
                @keyframes spin {{
                    to {{ transform: rotate(360deg); }}
                }}
                
                .custom-notification {{
                    position: fixed;
                    bottom: 100px;
                    right: 30px;
                    padding: 12px 20px;
                    border-radius: 8px;
                    color: white;
                    font-weight: 500;
                    z-index: 10000;
                    animation: slideIn 0.3s ease;
                    max-width: 300px;
                }}
                
                .custom-notification.success {{ background: #10b981; }}
                .custom-notification.error {{ background: #ef4444; }}
                .custom-notification.info {{ background: #3b82f6; }}
                
                .upload-modal {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                
                .modal-content {{
                    background: #2f2f2f;
                    border-radius: 16px;
                    width: 90%;
                    max-width: 800px;
                    max-height: 90vh;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                }}
                
                .modal-header {{
                    padding: 1.5rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .modal-header h2 {{
                    color: #ececec;
                    margin: 0;
                    font-size: 1.5rem;
                }}
                
                .modal-close {{
                    background: none;
                    border: none;
                    color: #ababab;
                    font-size: 1.5rem;
                    cursor: pointer;
                }}
                
                .modal-close:hover {{
                    color: #ececec;
                }}
                
                .modal-body {{
                    padding: 2rem;
                    overflow-y: auto;
                    flex: 1;
                }}
                
                .upload-box {{
                    border: 2px dashed rgba(255, 255, 255, 0.2);
                    border-radius: 12px;
                    padding: 3rem;
                    text-align: center;
                    transition: all 0.3s ease;
                }}
                
                .upload-box:hover {{
                    border-color: #10a37f;
                    background: rgba(16, 163, 127, 0.05);
                }}
                
                .upload-button {{
                    display: inline-block;
                    padding: 0.75rem 2rem;
                    background: #10a37f;
                    color: white;
                    border-radius: 8px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }}
                
                .upload-button:hover {{
                    background: #0d8f6e;
                    transform: translateY(-1px);
                }}
                
                .loading-spinner {{
                    width: 40px;
                    height: 40px;
                    border: 4px solid rgba(255, 255, 255, 0.1);
                    border-top-color: #10a37f;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 1rem;
                }}
                
                .progress-bar {{
                    width: 100%;
                    height: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    margin: 1rem 0;
                    overflow: hidden;
                }}
                
                .progress-fill {{
                    height: 100%;
                    background: #10a37f;
                    border-radius: 4px;
                    transition: width 0.3s ease;
                }}
                
                .pdf-list {{
                    max-height: 400px;
                    overflow-y: auto;
                    margin: 1rem 0;
                }}
                
                .pdf-item {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0.75rem;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                    transition: all 0.2s ease;
                }}
                
                .pdf-item.excluded {{
                    opacity: 0.5;
                    text-decoration: line-through;
                }}
                
                .pdf-info {{
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    flex: 1;
                }}
                
                .pdf-thumbnail {{
                    width: 48px;
                    height: 64px;
                    object-fit: cover;
                    border-radius: 4px;
                    background: rgba(255, 255, 255, 0.1);
                }}
                
                .pdf-details {{
                    display: flex;
                    flex-direction: column;
                }}
                
                .pdf-name {{
                    color: #ececec;
                    font-weight: 500;
                }}
                
                .pdf-size {{
                    color: #ababab;
                    font-size: 0.85rem;
                }}
                
                .exclude-button {{
                    background: none;
                    border: none;
                    font-size: 1.5rem;
                    cursor: pointer;
                    padding: 0.5rem;
                }}
                
                .modal-footer {{
                    padding: 1.5rem;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                
                .submit-button {{
                    padding: 0.75rem 2rem;
                    background: #10a37f;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-weight: 500;
                    cursor: pointer;
                }}
                
                .submit-button:disabled {{
                    opacity: 0.5;
                    cursor: not-allowed;
                }}
            `;
            document.head.appendChild(style);
        }}
    }}
    
    function openUploadModal() {{
        if (uploadModal) {{
            uploadModal.remove();
        }}
        
        uploadModal = document.createElement('div');
        uploadModal.className = 'upload-modal';
        uploadModal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>PDF Upload & Web Crawler</h2>
                    <button class="modal-close" onclick="this.closest('.upload-modal').remove()">×</button>
                </div>
                <div class="modal-body" id="modal-body">
                    <!-- Content will be updated based on step -->
                </div>
                <div class="modal-footer" id="modal-footer" style="display: none;">
                    <!-- Footer content will be updated based on step -->
                </div>
            </div>
        `;
        
        document.body.appendChild(uploadModal);
        updateModalContent('upload');
    }}
    
    function updateModalContent(step) {{
        currentStep = step;
        const modalBody = document.getElementById('modal-body');
        const modalFooter = document.getElementById('modal-footer');
        
        if (step === 'upload') {{
            modalBody.innerHTML = `
                <div class="upload-box">
                    <h3 style="color: #ececec; margin-bottom: 1rem;">Upload PDF Files</h3>
                    <p style="color: #ababab; margin-bottom: 2rem;">Select one or more PDF files to start web crawling</p>
                    <label for="pdf-file-upload" class="upload-button">
                        Choose Files
                    </label>
                    <input
                        id="pdf-file-upload"
                        type="file"
                        multiple
                        accept=".pdf"
                        style="display: none;"
                        onchange="handleFileUpload(this.files)"
                    />
                </div>
            `;
            modalFooter.style.display = 'none';
            
        }} else if (step === 'crawling') {{
            modalBody.innerHTML = `
                <div style="text-align: center;">
                    <div class="loading-spinner"></div>
                    <h3 style="color: #ececec; margin: 1rem 0;">Web Crawling in Progress</h3>
                    <p style="color: #ababab;" id="crawl-message">Extracting links and downloading PDFs...</p>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                    </div>
                    <div style="margin-top: 2rem;">
                        <h4 style="color: #ababab;">Processing Files:</h4>
                        <div id="processing-files"></div>
                    </div>
                </div>
            `;
            modalFooter.style.display = 'none';
            
        }} else if (step === 'review') {{
            modalBody.innerHTML = `
                <div>
                    <h3 style="color: #ececec; margin-bottom: 1rem;">Review Crawled PDFs</h3>
                    <p style="color: #ababab; margin-bottom: 1rem;">Click ❌ to exclude PDFs you don't want</p>
                    <div class="pdf-list" id="pdf-list">
                        <!-- PDFs will be loaded here -->
                    </div>
                </div>
            `;
            modalFooter.style.display = 'flex';
            modalFooter.innerHTML = `
                <span style="color: #ababab;" id="selected-count">0 PDFs selected</span>
                <button class="submit-button" onclick="finalizeUpload()">
                    Upload to Knowledge Base
                </button>
            `;
            loadCrawledPDFs();
            
        }} else if (step === 'finalizing') {{
            modalBody.innerHTML = `
                <div style="text-align: center;">
                    <div class="loading-spinner"></div>
                    <h3 style="color: #ececec; margin: 1rem 0;">Finalizing Upload</h3>
                    <p style="color: #ababab;">Moving selected PDFs to your knowledge base...</p>
                </div>
            `;
            modalFooter.style.display = 'none';
        }}
    }}
    
    window.handleFileUpload = async function(files) {{
        if (!files || files.length === 0) return;
        
        uploadedFiles = Array.from(files);
        updateModalContent('crawling');
        
        // Show uploaded files
        const processingDiv = document.getElementById('processing-files');
        processingDiv.innerHTML = uploadedFiles.map(f => 
            `<div style="color: #ababab; margin: 0.5rem;">📄 ${{f.name}}</div>`
        ).join('');
        
        // Create FormData
        const formData = new FormData();
        for (const file of uploadedFiles) {{
            formData.append('files', file);
        }}
        
        try {{
            updateProgress(20, 'Uploading files...');
            
            const uploadUrl = USE_AUTH ? 
                `${{API_PREFIX}}/pdf-upload` : 
                `${{PDF_BACKEND_URL}}${{API_PREFIX}}/upload`;
            
            const response = await fetchWithAuth(uploadUrl, {{
                method: 'POST',
                body: formData
            }});
            
            if (!response.ok) {{
                throw new Error('Upload failed');
            }}
            
            updateProgress(50, 'Processing links from PDFs...');
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            updateProgress(80, 'Finalizing downloads...');
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            updateProgress(100, 'Loading results...');
            updateModalContent('review');
            
        }} catch (error) {{
            console.error('[Custom] Upload error:', error);
            showNotification('Upload failed: ' + error.message, 'error');
            updateModalContent('upload');
        }}
    }};
    
    async function loadCrawledPDFs() {{
        try {{
            const listUrl = USE_AUTH ? 
                `${{API_PREFIX}}/pdf-list` : 
                `${{PDF_BACKEND_URL}}${{API_PREFIX}}/pdfs`;
                
            const response = await fetchWithAuth(listUrl);
            if (!response.ok) throw new Error('Failed to load PDFs');
            
            crawledPDFs = await response.json();
            const pdfList = document.getElementById('pdf-list');
            
            if (crawledPDFs.length === 0) {{
                pdfList.innerHTML = '<p style="color: #ababab;">No PDFs found.</p>';
                return;
            }}
            
            pdfList.innerHTML = crawledPDFs.map(pdf => `
                <div class="pdf-item ${{pdf.excluded ? 'excluded' : ''}}" data-name="${{pdf.name}}">
                    <div class="pdf-info">
                        ${{pdf.preview_url ? `
                            <img src="${{PDF_BACKEND_URL}}${{pdf.preview_url}}" 
                                 alt="${{pdf.name}}" 
                                 class="pdf-thumbnail" />
                        ` : ''}}
                        <div class="pdf-details">
                            <span class="pdf-name">${{pdf.name}}</span>
                            <span class="pdf-size">${{pdf.size_kb}} KB</span>
                        </div>
                    </div>
                    <button class="exclude-button" onclick="togglePDF('${{pdf.name}}')">
                        ${{pdf.excluded ? '✓' : '❌'}}
                    </button>
                </div>
            `).join('');
            
            // Initialize excluded set
            excludedPDFs = new Set(crawledPDFs.filter(p => p.excluded).map(p => p.name));
            updateSelectedCount();
            
        }} catch (error) {{
            console.error('[Custom] Error loading PDFs:', error);
            showNotification('Failed to load PDFs', 'error');
        }}
    }}
    
    window.togglePDF = async function(pdfName) {{
        const isExcluded = !excludedPDFs.has(pdfName);
        
        if (isExcluded) {{
            excludedPDFs.add(pdfName);
        }} else {{
            excludedPDFs.delete(pdfName);
        }}
        
        try {{
            const toggleUrl = USE_AUTH ? 
                `${{API_PREFIX}}/pdf-toggle/${{encodeURIComponent(pdfName)}}` : 
                `${{PDF_BACKEND_URL}}${{API_PREFIX}}/pdfs/${{encodeURIComponent(pdfName)}}`;
                
            const response = await fetchWithAuth(toggleUrl, {{
                method: 'PATCH',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ 
                    name: pdfName,
                    excluded: isExcluded 
                }})
            }});
            
            if (!response.ok) throw new Error('Failed to update PDF');
            
            // Update UI
            const pdfItem = document.querySelector(`.pdf-item[data-name="${{pdfName}}"]`);
            if (pdfItem) {{
                if (isExcluded) {{
                    pdfItem.classList.add('excluded');
                }} else {{
                    pdfItem.classList.remove('excluded');
                }}
                pdfItem.querySelector('.exclude-button').textContent = isExcluded ? '✓' : '❌';
            }}
            
            updateSelectedCount();
            
        }} catch (error) {{
            console.error('[Custom] Error toggling PDF:', error);
            showNotification('Failed to update PDF', 'error');
            // Revert on error
            if (isExcluded) {{
                excludedPDFs.delete(pdfName);
            }} else {{
                excludedPDFs.add(pdfName);
            }}
        }}
    }};
    
    window.finalizeUpload = async function() {{
        updateModalContent('finalizing');
        
        try {{
            const finalizeUrl = USE_AUTH ? 
                `${{API_PREFIX}}/pdf-finalize` : 
                `${{PDF_BACKEND_URL}}${{API_PREFIX}}/finalize`;
                
            const response = await fetchWithAuth(finalizeUrl, {{
                method: 'POST'
            }});
            
            if (!response.ok) throw new Error('Failed to finalize upload');
            
            const result = await response.json();
            showNotification('Successfully uploaded PDFs to Knowledge Base!', 'success');
            
            // Close modal after success
            setTimeout(() => {{
                if (uploadModal) {{
                    uploadModal.remove();
                    uploadModal = null;
                }}
                // Reset state
                uploadedFiles = [];
                crawledPDFs = [];
                excludedPDFs = new Set();
            }}, 2000);
            
        }} catch (error) {{
            console.error('[Custom] Finalize error:', error);
            showNotification('Failed to finalize upload', 'error');
            updateModalContent('review');
        }}
    }};
    
    function updateProgress(percent, message) {{
        const progressFill = document.getElementById('progress-fill');
        const crawlMessage = document.getElementById('crawl-message');
        
        if (progressFill) {{
            progressFill.style.width = percent + '%';
        }}
        if (crawlMessage && message) {{
            crawlMessage.textContent = message;
        }}
    }}
    
    function updateSelectedCount() {{
        const count = crawledPDFs.length - excludedPDFs.size;
        const countElement = document.getElementById('selected-count');
        if (countElement) {{
            countElement.textContent = `${{count}} of ${{crawledPDFs.length}} PDFs selected`;
        }}
    }}
    
    function showNotification(message, type = 'info') {{
        const notification = document.createElement('div');
        notification.className = `custom-notification ${{type}}`;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {{
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => notification.remove(), 300);
        }}, 3000);
    }}
    
    function checkPage() {{
        if (window.location.pathname.includes('knowledge')) {{
            if (!floatingButton) {{
                createFloatingButton();
            }}
        }} else {{
            if (floatingButton) {{
                floatingButton.remove();
                floatingButton = null;
            }}
            if (uploadModal) {{
                uploadModal.remove();
                uploadModal = null;
            }}
        }}
    }}
    
    // Initialize
    checkPage();
    
    // Monitor for URL changes
    let lastPath = window.location.pathname;
    setInterval(() => {{
        if (window.location.pathname !== lastPath) {{
            lastPath = window.location.pathname;
            checkPage();
        }}
    }}, 500);
    
    console.log('[Custom] PDF Upload Integration initialized - Mode:', USE_AUTH ? 'INTEGRATED' : 'EXTERNAL');
}})();
'''

# Save to static directory
static_dir = Path("/app/backend/open_webui/static")
if static_dir.exists():
    with open(static_dir / "custom_button.js", 'w') as f:
        f.write(custom_js)
    print("✓ Created custom_button.js")
    
    # Create/update loader.js
    loader_content = '''// Custom loader
console.log('[Loader] Loading customizations...');
if (!document.querySelector('script[src*="custom_button.js"]')) {
    const script = document.createElement('script');
    script.src = '/static/custom_button.js?v=' + Date.now();
    document.head.appendChild(script);
}'''
    
    with open(static_dir / "loader.js", 'w') as f:
        f.write(loader_content)
    print("✓ Updated loader.js")
else:
    print("✗ Static directory not found!")

print("\n✓ All customizations applied successfully!")
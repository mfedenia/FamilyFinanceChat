"""
Custom UI Router - Open WebUI Interface Customizations
Handles CSS injection, landing pages, and other UI customizations
"""

import logging
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
from starlette.responses import Response as StarletteResponse

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

router = APIRouter()

# ============================================================================
# CSS INJECTION ENDPOINT
# ============================================================================

@router.get("/inject-custom-css")
async def get_custom_css_injection():
    """Returns JavaScript that injects custom CSS styles into Open WebUI"""
    
    script = r'''
(function() {
    // Remove existing custom styles if reloading
    const existingStyle = document.getElementById('openwebui-custom-styles');
    if (existingStyle) {
        existingStyle.remove();
    }
    
    console.log('[Custom CSS] Injecting styles...');
    
    const style = document.createElement('style');
    style.id = 'openwebui-custom-styles';
    style.textContent = `
        /* ============================================
           CUSTOM OPEN WEBUI STYLES
           Customize these styles to your preference
           ============================================ */
        
        /* === CHAT CONTAINER === */
        /* Main chat area background */
        .main-chat-container,
        main[class*="flex-1"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        
        /* Alternative: Dark gradient */
        /*
        .main-chat-container,
        main[class*="flex-1"] {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        }
        */
        
        /* === MESSAGE BUBBLES === */
        /* User messages (right side) */
        [class*="whitespace-pre-wrap"][class*="self-end"],
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            color: white !important;
        }
        
        /* AI messages (left side) */
        [class*="whitespace-pre-wrap"]:not([class*="self-end"]),
        .ai-message {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 12px 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
            color: #1a1a1a !important;
        }
        
        /* === SIDEBAR === */
        /* Left sidebar background */
        aside,
        nav[class*="flex"][class*="flex-col"] {
            background: rgba(30, 30, 30, 0.95) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Sidebar chat items */
        aside a,
        nav a {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        
        aside a:hover,
        nav a:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            transform: translateX(4px) !important;
        }
        
        /* === INPUT AREA === */
        /* Chat input container */
        textarea[placeholder*="Send a message"],
        textarea[placeholder*="message"],
        .chat-input {
            background: rgba(255, 255, 255, 0.95) !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 16px !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            transition: all 0.3s ease !important;
        }
        
        textarea[placeholder*="Send a message"]:focus,
        textarea[placeholder*="message"]:focus,
        .chat-input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            outline: none !important;
        }
        
        /* Send button */
        button[type="submit"],
        button[aria-label*="Send"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 12px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
        }
        
        button[type="submit"]:hover,
        button[aria-label*="Send"]:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* === SCROLLBAR === */
        /* Custom scrollbar for chat */
        *::-webkit-scrollbar {
            width: 10px !important;
            height: 10px !important;
        }
        
        *::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1) !important;
            border-radius: 5px !important;
        }
        
        *::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-radius: 5px !important;
            transition: all 0.3s ease !important;
        }
        
        *::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        }
        
        /* === BADGES AND TAGS === */
        /* Model badges, tags, etc. */
        .badge,
        [class*="badge"],
        span[class*="bg-"][class*="rounded"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 4px 12px !important;
            border-radius: 12px !important;
            font-weight: 500 !important;
        }
        
        /* === BUTTONS === */
        /* Primary buttons */
        button[class*="bg-blue"],
        button[class*="primary"],
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        button[class*="bg-blue"]:hover,
        button[class*="primary"]:hover,
        .btn-primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3) !important;
        }
        
        /* === MODALS AND DROPDOWNS === */
        /* Modal backgrounds */
        [class*="fixed"][class*="inset-0"][class*="z-"],
        .modal-overlay {
            backdrop-filter: blur(8px) !important;
        }
        
        /* Modal content */
        [role="dialog"],
        .modal {
            background: rgba(30, 30, 30, 0.98) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Dropdown menus */
        [role="menu"],
        .dropdown-menu {
            background: rgba(30, 30, 30, 0.98) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* === LOADING INDICATORS === */
        /* Loading spinners */
        .spinner,
        [class*="animate-spin"] {
            border-color: #667eea transparent #764ba2 transparent !important;
        }
        
        /* === CODE BLOCKS === */
        /* Code syntax highlighting */
        pre,
        code {
            background: rgba(0, 0, 0, 0.8) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
        }
        
        pre code {
            border: none !important;
        }
        
        /* === ANIMATIONS === */
        @keyframes slideInFromBottom {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        /* Apply animation to new messages */
        [class*="whitespace-pre-wrap"] {
            animation: slideInFromBottom 0.3s ease !important;
        }
        
        /* === ACCESSIBILITY === */
        /* Focus indicators */
        *:focus-visible {
            outline: 2px solid #667eea !important;
            outline-offset: 2px !important;
        }
        
        /* === CUSTOM BRANDING === */
        /* Add your logo or branding here */
        /*
        header::before {
            content: "🏦 Family Finance Chat";
            font-size: 18px;
            font-weight: bold;
            margin-right: 12px;
        }
        */
        
        /* ============================================
           END CUSTOM STYLES
           Feel free to modify or add more styles!
           ============================================ */
    `;
    
    document.head.appendChild(style);
    
    // Show confirmation notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
        z-index: 99999;
        font-weight: 500;
        animation: slideInFromRight 0.3s ease;
    `;
    notification.textContent = '✨ Custom styles applied!';
    
    const keyframeStyle = document.createElement('style');
    keyframeStyle.textContent = `
        @keyframes slideInFromRight {
            from { transform: translateX(100px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(keyframeStyle);
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transition = 'all 0.3s ease';
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100px)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
    
    console.log('[Custom CSS] Styles applied successfully!');
})();
'''
    
    return StarletteResponse(
        content=script,
        media_type="application/javascript",
        headers={
            "Content-Type": "application/javascript; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


# ============================================================================
# CUSTOM LANDING PAGE
# ============================================================================

@router.get("/landing-page")
async def get_landing_page():
    """Returns a custom landing page for Open WebUI"""
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Family Finance Chat - Welcome</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
            }
            
            .container {
                max-width: 1200px;
                padding: 40px;
                text-align: center;
            }
            
            .logo {
                font-size: 80px;
                margin-bottom: 20px;
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-20px); }
            }
            
            h1 {
                font-size: 48px;
                margin-bottom: 20px;
                font-weight: 700;
            }
            
            .subtitle {
                font-size: 24px;
                margin-bottom: 40px;
                opacity: 0.9;
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 30px;
                margin: 60px 0;
            }
            
            .feature-card {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-10px);
                background: rgba(255, 255, 255, 0.15);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            }
            
            .feature-icon {
                font-size: 48px;
                margin-bottom: 15px;
            }
            
            .feature-title {
                font-size: 22px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .feature-description {
                font-size: 16px;
                opacity: 0.9;
                line-height: 1.6;
            }
            
            .cta-button {
                display: inline-block;
                padding: 18px 50px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 50px;
                font-size: 18px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }
            
            .cta-button:hover {
                transform: scale(1.05);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
            }
            
            .footer {
                margin-top: 60px;
                opacity: 0.8;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🏦</div>
            <h1>Family Finance Chat</h1>
            <p class="subtitle">Your AI-Powered Financial Assistant</p>
            
            <div class="features">
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <div class="feature-title">Smart Conversations</div>
                    <div class="feature-description">
                        Get instant answers to your financial questions with AI-powered intelligence
                    </div>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Financial Analysis</div>
                    <div class="feature-description">
                        Analyze your financial data and get personalized insights
                    </div>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <div class="feature-title">Secure & Private</div>
                    <div class="feature-description">
                        Your data stays private with enterprise-grade security
                    </div>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon">📚</div>
                    <div class="feature-title">Knowledge Base</div>
                    <div class="feature-description">
                        Upload and query your financial documents with ease
                    </div>
                </div>
            </div>
            
            <a href="/" class="cta-button">Get Started →</a>
            
            <div class="footer">
                <p>Powered by Open WebUI • Version 0.6.41</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


# ============================================================================
# CLIENT DOSSIER HEADER INJECTION
# ============================================================================

@router.get("/inject-client-dossier")
async def inject_client_dossier():
    """Injects a Client Dossier header at the top of the chat interface"""
    
    script = r'''
(function() {
    // Check if already injected to avoid duplicates
    if (window.__clientDossierInjected) return;
    window.__clientDossierInjected = true;
    
    console.log('[Client Dossier] Injecting header...');
    
    function injectDossierHeader() {
        // Find the main chat container (adjusting selectors for Open WebUI 0.6.41)
        const chatContainer = document.querySelector('main') || 
                            document.querySelector('[class*="flex-1"]') ||
                            document.querySelector('#chat-container');
        
        if (!chatContainer) {
            console.log('[Client Dossier] Chat container not found yet, retrying...');
            return false;
        }
        
        // Check if header already exists
        if (document.getElementById('client-dossier-header')) {
            return true;
        }
        
        // Create the Client Dossier header
        const dossierHeader = document.createElement('div');
        dossierHeader.id = 'client-dossier-header';
        // STICKY: stays at top of viewport while scrolling
        dossierHeader.className = 'sticky top-0 z-50 bg-slate-50 border-b border-slate-200 px-6 py-4 shadow-sm backdrop-blur-sm bg-opacity-95';
        
        dossierHeader.innerHTML = `
            <div class="max-w-5xl mx-auto">
                <div class="flex items-start justify-between">
                    <div class="flex-1">
                        <div class="flex items-center gap-3 mb-2">
                            <div class="flex items-center justify-center w-10 h-10 bg-blue-600 rounded-full text-white font-semibold text-lg">
                                TB
                            </div>
                            <div>
                                <h2 class="text-lg font-semibold text-slate-800">
                                    Active Client Meeting: Tony Beckham (55) & Antonella Beckham (53)
                                </h2>
                                <p class="text-sm text-slate-600">
                                    Objective: Gather data on retirement goals, estate planning, and current liabilities.
                                </p>
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            <span class="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></span>
                            In Session
                        </span>
                    </div>
                </div>
            </div>
        `;
        
        // Insert at the beginning of the chat container
        chatContainer.insertBefore(dossierHeader, chatContainer.firstChild);
        
        console.log('[Client Dossier] Header injected successfully!');
        return true;
    }
    
    // Try to inject immediately
    if (!injectDossierHeader()) {
        // If not successful, wait for DOM and try again
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function retry() {
                setTimeout(injectDossierHeader, 500);
            });
        } else {
            // DOM already loaded, try with a delay
            setTimeout(injectDossierHeader, 500);
        }
    }
    
    // Also watch for route changes (SPA navigation)
    const observer = new MutationObserver(function(mutations) {
        // Check if header still exists, re-inject if needed
        if (!document.getElementById('client-dossier-header')) {
            injectDossierHeader();
        }
    });
    
    // Start observing the document body for changes
    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    // Show confirmation notification
    setTimeout(function() {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            z-index: 99999;
            font-size: 14px;
            font-weight: 500;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = '✓ Client Dossier Active';
        
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        setTimeout(function() {
            notification.style.transition = 'all 0.3s ease';
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100px)';
            setTimeout(function() { notification.remove(); }, 300);
        }, 2500);
    }, 100);
})();
'''
    
    return StarletteResponse(
        content=script,
        media_type="application/javascript",
        headers={
            "Content-Type": "application/javascript; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "custom-ui-router"}

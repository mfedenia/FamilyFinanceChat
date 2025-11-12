#!/usr/bin/env python3
import sys
import os
import shutil

print("Starting customization application...")

# Add custom API import to main.py if not already there
main_file = "/app/backend/open_webui/main.py"
with open(main_file, 'r') as f:
    content = f.read()

# Check if custom_api import already exists
if 'from open_webui.routers import custom_api' not in content:
    # Add import after other router imports
    import_line = "from open_webui.routers import utils"
    if import_line in content:
        content = content.replace(
            import_line,
            f"{import_line}\nfrom open_webui.routers import custom_api"
        )
    
    # Add router registration
    router_line = 'app.include_router(utils.router, prefix="/api/v1/utils", tags=["utils"])'
    if router_line in content:
        content = content.replace(
            router_line,
            f'{router_line}\napp.include_router(custom_api.router, prefix="/api/v1/custom", tags=["custom"])'
        )
    
    with open(main_file, 'w') as f:
        f.write(content)
    print("✓ Modified main.py")

# Copy custom_api.py to routers directory
if os.path.exists("/app/custom_code/custom_api.py"):
    shutil.copy2(
        "/app/custom_code/custom_api.py",
        "/app/backend/open_webui/routers/custom_api.py"
    )
    print("✓ Copied custom_api.py")

# Create custom JavaScript
custom_js = '''
(function() {
    console.log('[Custom] Script loaded - persistent version');
    
    let customButtonAdded = false;
    let checkInterval;
    
    function addCustomButton() {
        if (!window.location.pathname.includes('knowledge')) {
            customButtonAdded = false;
            return;
        }
        
        const buttons = Array.from(document.querySelectorAll('button'));
        const uploadButton = buttons.find(btn => {
            const text = (btn.innerText || btn.textContent || '').toLowerCase();
            return (text.includes('upload') || text.includes('add file')) && 
                   btn.offsetParent !== null &&
                   !btn.id?.includes('custom');
        });
        
        if (!uploadButton || document.getElementById('custom-action-button')) {
            return;
        }
        
        console.log('[Custom] Adding custom button...');
        
        const customButton = uploadButton.cloneNode(true);
        customButton.id = 'custom-action-button';
        
        const textNodes = customButton.querySelectorAll('span, div');
        if (textNodes.length > 0) {
            textNodes[textNodes.length - 1].textContent = 'Custom Action';
        } else {
            customButton.textContent = 'Custom Action';
        }
        
        customButton.onclick = async function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Custom] Button clicked!');
            
            const token = localStorage.getItem('token');
            if (!token) {
                alert('Please login first');
                return;
            }
            
            try {
                customButton.disabled = true;
                const response = await fetch('/api/v1/custom/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        data: {
                            timestamp: new Date().toISOString(),
                            page: window.location.pathname
                        }
                    })
                });
                
                const result = await response.json();
                console.log('[Custom] Response:', result);
                
                if (response.ok) {
                    alert('Custom action completed!');
                } else {
                    alert('Error: ' + (result.detail || 'Unknown error'));
                }
            } catch (error) {
                console.error('[Custom] Error:', error);
                alert('Error: ' + error.message);
            } finally {
                customButton.disabled = false;
            }
        };
        
        uploadButton.parentNode.insertBefore(customButton, uploadButton.nextSibling);
        customButtonAdded = true;
        console.log('[Custom] Button added successfully');
    }
    
    // Check periodically for the upload button
    checkInterval = setInterval(() => {
        if (window.location.pathname.includes('knowledge')) {
            addCustomButton();
        }
    }, 1000);
    
    // Watch for clicks to detect dropdown opening
    document.addEventListener('click', function(e) {
        setTimeout(addCustomButton, 200);
        setTimeout(addCustomButton, 500);
    });
    
    console.log('[Custom] Persistent script initialized');
})();
'''

with open("/app/backend/open_webui/static/custom_button.js", 'w') as f:
    f.write(custom_js)
print("✓ Created custom_button.js")

# Update loader.js
loader_js = '''// Custom loader
console.log('[Loader] Loading customizations...');
const script = document.createElement('script');
script.src = '/static/custom_button.js?v=' + Date.now();
document.head.appendChild(script);
'''

with open("/app/backend/open_webui/static/loader.js", 'w') as f:
    f.write(loader_js)
print("✓ Updated loader.js")

print("\n✓ All customizations applied successfully!")

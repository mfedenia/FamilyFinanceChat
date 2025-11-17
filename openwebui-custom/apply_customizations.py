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
    console.log('[Custom] Script loaded - targeting + button version');
    
    let customButtonAdded = false;
    let checkInterval;
    
    function addCustomButton() {
        if (!window.location.pathname.includes('knowledge')) {
            customButtonAdded = false;
            return;
        }
        
        // Look for the + button in the collection container
        // Target button with aria-label containing "Add" or buttons in flex containers with specific classes
        const buttons = Array.from(document.querySelectorAll('button'));
        const addButton = buttons.find(btn => {
            // Check for + button characteristics
            const ariaLabel = btn.getAttribute('aria-label');
            const btnText = (btn.innerText || btn.textContent || '').trim();
            
            // Look for button with + symbol or Add aria-label
            if (btnText === '+' || (ariaLabel && ariaLabel.toLowerCase().includes('add'))) {
                // Additional check: should be in a collection container
                const parent = btn.closest('.flex.flex-col');
                if (parent && btn.offsetParent !== null) {
                    return true;
                }
            }
            
            // Alternative: Check for button with specific classes related to adding files
            if (btn.classList.contains('rounded-xl') && 
                btn.classList.contains('hover:bg-gray-100') &&
                btnText === '+') {
                return true;
            }
            
            return false;
        });
        
        if (!addButton || document.getElementById('custom-action-button')) {
            return;
        }
        
        console.log('[Custom] Found + button, adding custom button...');
        
        // Clone the + button to maintain styling
        const customButton = addButton.cloneNode(true);
        customButton.id = 'custom-action-button';
        
        // Change the content to a custom icon or text
        // Using a different icon - you can change this to any icon or emoji
        customButton.innerHTML = '⚡'; // Lightning bolt icon, or use '🔧' for wrench, '⚙️' for gear
        customButton.setAttribute('aria-label', 'Custom Action');
        customButton.title = 'Custom Action';
        
        // Remove any existing onclick handlers
        customButton.onclick = null;
        
        // Add our custom click handler
        customButton.addEventListener('click', async function(e) {
            e.preventDefault();
            e.stopPropagation();
            console.log('[Custom] Custom button clicked!');
            
            const token = localStorage.getItem('token');
            if (!token) {
                alert('Please login first');
                return;
            }
            
            try {
                customButton.disabled = true;
                
                // Get collection info if available
                const collectionContainer = customButton.closest('[class*="collection"]');
                const collectionName = collectionContainer ? 
                    collectionContainer.querySelector('input[type="text"]')?.value || 'Unknown' : 
                    'Unknown';
                
                const response = await fetch('/api/v1/custom/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        data: {
                            timestamp: new Date().toISOString(),
                            page: window.location.pathname,
                            collection: collectionName,
                            action: 'custom_button_click'
                        }
                    })
                });
                
                const result = await response.json();
                console.log('[Custom] Response:', result);
                
                if (response.ok) {
                    alert('Custom action completed successfully!');
                } else {
                    alert('Error: ' + (result.detail || 'Unknown error'));
                }
            } catch (error) {
                console.error('[Custom] Error:', error);
                alert('Error: ' + error.message);
            } finally {
                customButton.disabled = false;
            }
        });
        
        // Insert the custom button right after the + button
        addButton.parentNode.insertBefore(customButton, addButton.nextSibling);
        
        // Add a small margin to separate the buttons
        customButton.style.marginLeft = '8px';
        
        customButtonAdded = true;
        console.log('[Custom] Custom button added successfully next to + button');
    }
    
    // Check periodically for the + button
    checkInterval = setInterval(() => {
        if (window.location.pathname.includes('knowledge')) {
            addCustomButton();
        }
    }, 1000);
    
    // Also check on any DOM changes
    const observer = new MutationObserver(() => {
        if (window.location.pathname.includes('knowledge')) {
            setTimeout(addCustomButton, 100);
        }
    });
    
    // Start observing the document for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Initial check
    setTimeout(addCustomButton, 500);
    
    console.log('[Custom] Script initialized - monitoring for + button');
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
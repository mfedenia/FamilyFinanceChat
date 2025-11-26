// Custom loader
console.log('[Loader] Loading customizations...');
if (!document.querySelector('script[src*="custom_button.js"]')) {
    const script = document.createElement('script');
    script.src = '/static/custom_button.js?v=' + Date.now();
    document.head.appendChild(script);
}
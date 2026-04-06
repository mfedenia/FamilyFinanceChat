// API base URL configuration
// Automatically switches between localhost (dev) and GCP (production)

const getApiBaseUrl = () => {
  // Check if running on GCP (any non-localhost domain)
  const hostname = window.location.hostname;
  
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    // Local development: proxy to localhost:9500
    return "";  // Empty string uses relative '/api' which proxies via vite.config.js
  }
  
  // Production on GCP: use the same domain
  // Assumes backend API is on same domain (e.g., https://your-gcp-domain/api/...)
  return window.location.protocol + "//" + window.location.host;
};

export const API_BASE_URL = getApiBaseUrl();

export default API_BASE_URL;

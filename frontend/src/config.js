// Configuration for API endpoints
const config = {
  // Development environment
  development: {
    apiBaseUrl: 'http://localhost:5000'
  },
  // Production environment - replace with your actual backend URL
  production: {
    apiBaseUrl: 'https://spafgan-backend-production.up.railway.app' // TODO: Replace with actual Railway URL after deployment
  }
};

// Get current environment
const environment = process.env.NODE_ENV || 'development';

// Export the appropriate configuration
export const apiBaseUrl = config[environment].apiBaseUrl;

// Helper function to build API URLs
export const buildApiUrl = (endpoint) => {
  return `${apiBaseUrl}/api/${endpoint}`;
};

// TODO: After Railway deployment, replace the URL above with your actual Railway URL
// Example: 'https://your-app-name.up.railway.app' 
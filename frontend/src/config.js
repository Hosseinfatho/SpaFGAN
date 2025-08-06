// Configuration for API endpoints
const config = {
  // Development environment
  development: {
    apiBaseUrl: 'http://localhost:5000'
  },
  // Production environment - replace with your actual backend URL
  production: {
    apiBaseUrl: 'https://your-backend-url.herokuapp.com' // Replace with actual backend URL
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
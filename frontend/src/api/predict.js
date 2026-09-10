const API_BASE_URL = 'http://localhost:8000';

/**
 * Check health status of the FastAPI backend.
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`Health check failed with status ${res.status}`);
    return await res.json();
  } catch (err) {
    // Attempt fallback via Vite proxy if direct port access fails
    try {
      const resProxy = await fetch('/api/health');
      if (resProxy.ok) return await resProxy.json();
    } catch {
      // ignore secondary error
    }
    throw err;
  }
}

/**
 * Upload an image file to the /predict endpoint.
 * @param {File} file - Retinal fundus image file
 * @returns {Promise<Object>} JSON response containing grade, confidence, Grad-CAM overlay, etc.
 */
export async function predictSeverity(file) {
  const formData = new FormData();
  formData.append('file', file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch (err) {
    // Try fallback proxy
    try {
      response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
    } catch {
      throw new Error('Cannot connect to FastAPI backend on port 8000. Ensure the backend server is running.');
    }
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
    throw new Error(errorData.detail || `Prediction failed with status: ${response.status}`);
  }

  return await response.json();
}

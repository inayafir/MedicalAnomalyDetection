import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({ baseURL: API_BASE_URL });

/**
 * Normalizes an Axios/network error into a plain object the UI can render
 * without reaching into error.response.data.detail everywhere.
 */
function toApiError(error) {
  if (error.response) {
    const detail = error.response.data?.detail;
    return {
      status: error.response.status,
      message:
        typeof detail === "string"
          ? detail
          : detail
          ? JSON.stringify(detail)
          : `Request failed with status ${error.response.status}.`,
    };
  }
  if (error.request) {
    return {
      status: 0,
      message: `Can't reach the backend at ${API_BASE_URL}. Is it running?`,
    };
  }
  return { status: -1, message: error.message || "Unknown error." };
}

async function call(promise) {
  try {
    const res = await promise;
    return res.data;
  } catch (error) {
    throw toApiError(error);
  }
}

export const getHealth = () => call(client.get("/health"));

export const uploadImage = (file, patientId) => {
  const form = new FormData();
  form.append("file", file);
  const params = {};
  if (patientId) params.patient_id = patientId;
  return call(
    client.post("/images/upload", form, {
      params,
      headers: { "Content-Type": "multipart/form-data" },
    })
  );
};

export const listImages = ({ limit = 20, offset = 0, patientId } = {}) =>
  call(
    client.get("/images", {
      params: { limit, offset, patient_id: patientId || undefined },
    })
  );

export const getImage = (imageId) => call(client.get(`/images/${imageId}`));

export const deleteImage = (imageId) =>
  call(client.delete(`/images/${imageId}`));

export const createPrediction = (imageId) =>
  call(client.post(`/predictions/${imageId}`));

export const getPrediction = (predictionId) =>
  call(client.get(`/predictions/${predictionId}`));

export const listPredictions = ({
  limit = 20,
  offset = 0,
  imageId,
  predictedClass,
} = {}) =>
  call(
    client.get("/predictions", {
      params: {
        limit,
        offset,
        image_id: imageId || undefined,
        predicted_class: predictedClass || undefined,
      },
    })
  );

export const createReport = (predictionId) =>
  call(client.post(`/reports/${predictionId}`));

export const listPatients = ({ limit = 50, offset = 0 } = {}) =>
  call(client.get("/patients", { params: { limit, offset } }));

export const createPatient = (displayName) =>
  call(client.post("/patients", { display_name: displayName || null }));

export const fileUrl = (relativePath) =>
  `${API_BASE_URL}/files/${relativePath}`;

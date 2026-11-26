import axios from "axios";

// All backend calls go through this instance. By default requests are same-origin, under
// whatever base path the app was built for (VITE_BASE, e.g. "/" or "/grading/"), so the
// hosted build needs no API URL at all. Set VITE_API_BASE_URL at build time only if the API
// lives on a different origin.
const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || import.meta.env.BASE_URL,
});

export default api;

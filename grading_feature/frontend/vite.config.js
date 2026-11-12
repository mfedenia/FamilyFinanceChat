import process from 'node:process'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// URL path the dashboard is served under, e.g. "/" or "/grading/". Must start and end with "/".
// Read at build time; the router and API client both follow it via import.meta.env.BASE_URL.
const base = process.env.VITE_BASE || '/'

// https://vite.dev/config/
export default defineConfig({
  base,
  plugins: [react()],
  server: {
    proxy: {
      // Dev only: forward API calls to the local backend, which always serves them at /api.
      [`${base}api`]: {
        target: 'http://localhost:9500',
        rewrite: (path) => path.slice(base.length - 1),
      },
    },
  },
})

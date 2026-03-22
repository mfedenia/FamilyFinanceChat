import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/users': 'http://localhost:9500',
      '/user': 'http://localhost:9500',
      '/refresh': 'http://localhost:9500',
      '/api': 'http://localhost:9500',
    },
  },
})

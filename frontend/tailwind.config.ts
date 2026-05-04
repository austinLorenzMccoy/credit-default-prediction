import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: '#0D1B2A',
        copper: '#C4622D',
        'slate-blue': '#2E4057',
        sand: '#F0E6D3',
        sage: '#4A7C59',
        amber: '#D4A017',
        mist: '#F7F4F0',
        rule: '#D9D0C7',
        surface: {
          DEFAULT: '#fff8f6',
          dim: '#e9d6cf',
          bright: '#fff8f6',
          'container-lowest': '#ffffff',
          'container-low': '#fff1eb',
          'container': '#fdeae2',
          'container-high': '#f7e4dd',
          'container-highest': '#f2dfd7',
        }
      },
      fontFamily: {
        display: ['Playfair Display', 'serif'],
        heading: ['DM Serif Display', 'serif'],
        body: ['Lora', 'serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
      boxShadow: {
        'ambient': '0 2px 12px rgba(13,27,42,0.06)',
        'code-glow': '0 0 20px rgba(196, 98, 45, 0.15)',
      }
    },
  },
  plugins: [],
} satisfies Config

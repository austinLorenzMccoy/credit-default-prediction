/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Core brand palette
        copper:     '#C4622D',
        'copper-dk':'#A8521E',
        'slate-blue':'#2E4057',
        ink:        '#0D1B2A',
        sand:       '#F0E6D3',
        sage:       '#4A7C59',
        amber:      '#D4A017',
        mist:       '#F7F4F0',
        error:      '#C0392B',
        rule:       '#D9D0C7',
      },
      fontFamily: {
        heading: ['"Playfair Display"', 'Georgia', 'serif'],
        display: ['"DM Serif Display"', '"Playfair Display"', 'Georgia', 'serif'],
        body:    ['"Lora"', 'Georgia', 'serif'],
        mono:    ['"IBM Plex Mono"', 'monospace'],
      },
      spacing: {
        'stack-sm':  '1.5rem',
        'stack-md':  '2.5rem',
        'stack-lg':  '4rem',
        'gutter':    '1.5rem',
      },
      boxShadow: {
        'ambient': '0 4px 20px rgba(13,27,42,0.08)',
        'lifted':  '0 12px 32px rgba(13,27,42,0.14)',
      },
      borderColor: {
        rule: '#D9D0C7',
      },
    },
  },
  plugins: [],
};

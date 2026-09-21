/** @type {import('tailwindcss').Config} */

// Semantic colors live as RGB channels in src/index.css (light + dark),
// so opacity modifiers like `bg-tint/10` keep working.
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: token('bg'),
        surface: token('surface'),
        raised: token('raised'),
        fill: token('fill'),
        label: token('label'),
        'label-2': token('label-2'),
        'label-3': token('label-3'),
        separator: token('separator'),
        tint: token('tint'),
        'on-tint': token('on-tint'),
        danger: token('danger'),
        warning: token('warning'),
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

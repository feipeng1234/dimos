/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./App.{js,jsx,ts,tsx}",
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        'dimos-blue': '#0016B1',
        'dimos-yellow': '#FFF200',
      },
      fontFamily: {
        mono: ['monospace'],
      },
    },
  },
  plugins: [],
}

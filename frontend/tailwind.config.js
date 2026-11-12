/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F7F8FC",
        sidebar: "#101827",
        card: "#FFFFFF",
        primary: {
          DEFAULT: "#2563EB",
          foreground: "#FFFFFF",
        },
        teal: {
          accent: "#0D9488",
        },
        success: "#16A34A",
        ink: {
          primary: "#172033",
          secondary: "#64748B",
        },
        border: "#E2E8F0",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px 0 rgba(16, 24, 39, 0.04), 0 1px 3px 0 rgba(16, 24, 39, 0.03)",
      },
    },
  },
  plugins: [],
};


/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#E9EDFA",
        "canvas-secondary": "#EEF1FB",
        sidebar: "#17243A",
        card: "#F3F5FD",
        primary: {
          DEFAULT: "#4568D9",
          secondary: "#718BE0",
          foreground: "#FFFFFF",
        },
        aurora: {
          lavender: "#EAE1FA",
          lightblue: "#DCE5FF",
          mint: "#D9F0EA",
          teal: "#159C91",
        },
        ink: {
          primary: "#263452",
          secondary: "#64748B",
        },
        border: "#D5DDF2",
      },
      fontFamily: {
        sans: ["'Plus Jakarta Sans'", "Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        card: "0 1px 3px 0 rgba(23, 36, 58, 0.05), 0 4px 12px -2px rgba(69, 104, 217, 0.06)",
        "card-hover":
          "0 4px 12px -2px rgba(23, 36, 58, 0.08), 0 8px 20px -4px rgba(69, 104, 217, 0.1)",
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F6EFE7",
        "canvas-secondary": "#EFE6DC",
        sidebar: "#38271F",
        card: "#FFFCF7",
        primary: {
          DEFAULT: "#6B3E26",
          secondary: "#B98255",
          foreground: "#FFFFFF",
        },
        coffee: {
          brown: "#6B3E26",
          caramel: "#B98255",
          tan: "#E9D5C0",
          cream: "#FFFCF7",
          beige: "#F6EFE7",
        },
        ink: {
          primary: "#38271F",
          secondary: "#806D60",
          muted: "#A89A8F",
        },
        border: "#E6D6C6",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        card: "0 2px 4px -1px rgba(56, 39, 31, 0.06), 0 4px 12px -2px rgba(107, 62, 38, 0.08)",
        "card-hover":
          "0 4px 6px -1px rgba(56, 39, 31, 0.08), 0 10px 20px -3px rgba(107, 62, 38, 0.12)",
        header: "0 2px 8px -2px rgba(56, 39, 31, 0.06)",
      },
    },
  },
  plugins: [],
};

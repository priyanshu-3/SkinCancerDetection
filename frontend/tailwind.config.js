/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                "white-b-color": "#ffffff",
                "pale-white-50": "#ffffff",
                "pale-white-100": "#ffffff",
                "pale-white-200": "#ffffff",
                "pale-white-300": "#ffffff"
            }
        },
    },
    plugins: [],
}


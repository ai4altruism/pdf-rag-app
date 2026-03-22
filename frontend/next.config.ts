import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // API rewrites so the frontend can call /api/v1/* without CORS issues in dev
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: "http://localhost:8000/api/v1/:path*",
      },
    ];
  },
};

export default nextConfig;

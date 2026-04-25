import type { NextConfig } from "next";

const repository = process.env.GITHUB_REPOSITORY;
const [owner, repo] = repository?.split("/") ?? [];
const isProjectPage = Boolean(owner && repo && repo !== `${owner}.github.io`);
const basePath = process.env.GITHUB_ACTIONS === "true" && isProjectPage ? `/${repo}` : undefined;

const nextConfig: NextConfig = {
  assetPrefix: basePath,
  basePath,
  images: {
    unoptimized: true,
  },
  output: "export",
  reactStrictMode: true,
};

export default nextConfig;

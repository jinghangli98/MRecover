import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MRecover",
  description:
    "Landing page for MRI contrast translation from T1w contrast to T2w TSE-like contrast.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

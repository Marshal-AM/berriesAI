import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { Toaster } from "@/components/ui/toaster";
import { WalletProvider } from "@/components/providers/wallet-provider";
import { WrongNetworkAlert } from "@/components/ui/custom/wrong-network-alert";
import { EnvironmentStoreProvider } from "@/components/context";
const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "SocioBerries",
  description:
    "SocioBerries is a decentralized social media platform built on the Aptos Blockchain, where businesses and creators collaborate through promotional content.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <WalletProvider>
            <EnvironmentStoreProvider>
              <Toaster />
              <WrongNetworkAlert />
              {children}
            </EnvironmentStoreProvider>
          </WalletProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}

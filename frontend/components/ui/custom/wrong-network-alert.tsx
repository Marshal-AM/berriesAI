"use client";

import { useWallet } from "@aptos-labs/wallet-adapter-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogOverlay,
} from "@/components/ui/dialog";
import { NETWORK } from "@/lib/aptos";

export function WrongNetworkAlert() {
  const { network, connected } = useWallet();

  return !connected || network?.name === NETWORK ? (
    <></>
  ) : (
    <Dialog open={true}>
      <DialogOverlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm transition-opacity duration-300 ease-in-out" />
      <DialogContent className="fixed z-50 top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 shadow-lg rounded-lg p-6 bg-white dark:bg-gray-800 transition-transform duration-300 ease-in-out w-[90vw] max-w-md">
        <DialogHeader className="text-center">
          <DialogTitle className="text-3xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
            Wrong Network
          </DialogTitle>
          <DialogDescription className="text-lg text-gray-700 dark:text-gray-300">
            Your wallet is currently on{" "}
            <span className="font-bold">{network?.name}</span>. Please switch to{" "}
            <span className="font-bold">{NETWORK}</span> to continue using the
            app.
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
}

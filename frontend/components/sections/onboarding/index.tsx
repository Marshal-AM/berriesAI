import { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { cn } from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";
import { UserAuthForm } from "./auth";
import OnboardingCarousel from "./carousel";
import { Separator } from "@/components/ui/separator";
import { useWallet } from "@aptos-labs/wallet-adapter-react";
import NewUser from "./new-user";

export const metadata: Metadata = {
  title: "Authentication",
  description: "Authentication forms built using the components.",
};

export default function OnboardingPage() {
  const { connected } = useWallet();
  return (
    <>
      <div className="container relative hidden h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
        <div className="relative hidden h-full w-full  bg-muted  dark:border-r lg:flex">
          <OnboardingCarousel />

          {/* <div className="relative z-20 mt-auto">
            <blockquote className="space-y-2">
              <p className="text-lg">
                &ldquo;This library has saved me countless hours of work and
                helped me deliver stunning designs to my clients faster than
                ever before.&rdquo;
              </p>
              <footer className="text-sm">Sofia Davis</footer>
            </blockquote>
          </div> */}
        </div>
        <div className="lg:p-8">
          {connected ? (
            <NewUser />
          ) : (
            <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
              <div className="flex flex-col space-y-2 text-center">
                <div className="flex space-x-3 items-end justify-center">
                  <Image src={"/logo.png"} width={60} height={60} alt="Logo" />
                  <p className="font-semibold text-2xl mb-2">SocioBerries</p>
                </div>
                <div className="flex justify-center">
                  <Separator className="w-[85%]" />
                </div>
                <p className="text-muted-foreground font-medium text-sm">
                  Connecting Influencers and Businesses for the good.
                </p>
              </div>
              <UserAuthForm />
              <p className="px-8 text-center text-sm text-muted-foreground">
                By using our app, you agree to our{" "}
                <Link
                  href="/terms"
                  className="underline underline-offset-4 hover:text-primary"
                >
                  Terms of Service
                </Link>{" "}
                and{" "}
                <Link
                  href="/privacy"
                  className="underline underline-offset-4 hover:text-primary"
                >
                  Privacy Policy
                </Link>
                .
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

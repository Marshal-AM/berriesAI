import Image from "next/image";
import WalletInfo from "./wallet-info";
import { Separator } from "../separator";
import SuggestedProfile from "./suggested-profile";

export default function Suggested() {
  const profiles = [
    {
      id: 1,
      name: "Marshal",
      username: "marshal.aptos",
      image: "/onboarding/1.png",
    },
    {
      id: 2,
      name: "Tabitha",
      username: "tabitha.aptos",
      image: "/onboarding/2.jpg",
    },
  ];
  return (
    <div className="w-[25%] flex flex-col">
      <div className="flex justify-between items-center">
        <div className="flex items-center  space-x-3 py-2">
          <Image
            src={"/avatar.jpeg"}
            width={30}
            height={30}
            alt="Profile"
            className="rounded-full"
          />
          <div className="flex flex-col">
            <p className="text-white text-sm font-semibold">gabrielaxy.aptos</p>
            <p className="text-sm text-muted-foreground">Gabriel</p>
          </div>
        </div>
        <WalletInfo />
      </div>
      <Separator className="my-4 w-[95%] mx-auto" />
      <p className="text-muted-foreground font-semibold my-2">
        Sugested for you
      </p>
      {profiles.map((p, idx) => (
        <SuggestedProfile
          key={idx}
          name={p.name}
          username={p.username}
          image={p.image}
          action={1}
        />
      ))}
    </div>
  );
}

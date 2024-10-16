import {
  Bell,
  GitGraph,
  Home,
  LineChart,
  Plus,
  Search,
  User,
} from "lucide-react";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Label } from "../ui/label";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import SuggestedProfile from "../ui/custom/suggested-profile";
export default function SideBar() {
  const [showSearch, setShowSearch] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const navs = [
    {
      id: 1,
      name: "Home",
      icon: "🏠",
      href: "/home",
    },
    {
      id: 2,
      name: "Search",
      icon: "🔍",
      href: "/search",
    },
    {
      id: 3,
      name: "Create",
      icon: "🖼️",
      href: "/create",
    },
    {
      id: 4,
      name: "Notifications",
      icon: "🔔",
      href: "/notifications",
    },
    {
      id: 5,
      name: "Collabs",
      icon: "🤝",
      href: "/collab",
    },
    {
      id: 6,
      name: "Profile",
      icon: "😀",
      href: "/profile",
    },
  ];
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
    <>
      <div className="h-full border-r-[1px] border-r-secondary py-4 px-8 flex flex-col">
        <div className="flex space-x-4 items-end mb-12">
          <Image src={"/logo.png"} width={40} height={40} alt="Logo" />
          <p className="font-semibold text-lg mb-[1px]">SocioBerries</p>
        </div>
        <div className="flex-1 flex flex-col space-y-4">
          {navs.map((nav) => (
            <div
              key={nav.id}
              className={`flex items-center gap-4 p-2 cursor-pointer hover:translate-x-1 hover:scale-110 transition ease-out delay-100  ${
                pathname === nav.href
                  ? "text-primary font-semibold "
                  : "text-white font-medium"
              }`}
              onClick={() => {
                if (nav.id != 2) router.push(nav.href);
                else setShowSearch(true);
              }}
            >
              <p className="text-white font-bold text-lg">{nav.icon}</p>
              <p className="">{nav.name}</p>
            </div>
          ))}
          <div className=" flex-1 flex flex-col justify-end">
            <div className="flex items-center  space-x-3">
              <Image
                src={"/avatar.jpeg"}
                width={30}
                height={30}
                alt="Profile"
                className="rounded-full"
              />
              <p className="text-white text-sm font-semibold">
                gabrielaxy.aptos
              </p>
            </div>
          </div>
        </div>
      </div>{" "}
      <Sheet
        open={showSearch}
        onOpenChange={(op) => {
          setShowSearch(op);
        }}
      >
        <SheetContent side="left" className="p-0">
          <SheetHeader className="p-4">
            <SheetTitle>Search</SheetTitle>
          </SheetHeader>
          <Input placeholder="Search" className="w-[90%] mx-auto my-4 p-4" />
          <Separator />
          <div className="p-4">
            <p className=" font-semibold text-sm py-4">Recent</p>
            {profiles.map((p, idx) => (
              <SuggestedProfile
                key={idx}
                name={p.name}
                username={p.username}
                image={p.image}
                action={0}
              />
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}

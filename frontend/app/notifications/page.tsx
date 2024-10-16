"use client";
import SideBar from "@/components/sections/side-nav";
import WalletInfo from "@/components/ui/custom/wallet-info";
import Notification from "@/components/sections/notifications/notification";
import { Separator } from "@/components/ui/separator";
export default function NotificationsPage() {
  const notifications = [
    {
      id: 1,
      type: 1,
      user: {
        name: "Marshal",
        username: "marshal.aptos",
      },
      ref: {
        image: "/post/hi.jpg",
        url: "/post/1",
      },
    },
    {
      id: 2,
      type: 2,
      user: {
        name: "Marshal",
        username: "marshal.aptos",
      },
      ref: {
        image: "/post/hi.jpg",
        url: "/post/1",
      },
    },
    {
      id: 2,
      type: 2,
      user: {
        name: "Tabitha",
        username: "tabitha.aptos",
      },
      ref: {
        image: "/post/hi.jpg",
        url: "/post/1",
      },
    },
  ];
  return (
    <div className="flex h-screen ">
      <SideBar />
      <div className="pt-7 px-7 flex-1 flex justify-center space-x-12">
        <div className="w-[40%]">
          <div className="flex justify-between">
            <p className="font-semibold text-lg">Notifications</p>
            <WalletInfo />
          </div>
          <Separator className="my-4" />
          {notifications.map((not, idx) => (
            <Notification
              key={idx}
              type={not.type}
              user={not.user.name}
              image={not.ref.image}
              url={not.ref.url}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

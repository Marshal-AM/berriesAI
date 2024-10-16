"use client";

import SideBar from "@/components/sections/side-nav";
import Post from "@/components/ui/custom/post";
import Suggested from "@/components/ui/custom/suggested";
import { ScrollBar, ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function HomePage() {
  const feed = [
    {
      id: 1,
      name: "John Doe",
      username: "@johndoe",
      picture: "https://randomuser.me/api/portraits",
      time: "2 hours ago",
      post: "https://source.unsplash.com/random",
      caption:
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nib",
      likes: 100,
      comments: 20,
    },
    {
      id: 2,
      name: "Jane Doe",
      username: "@janedoe",
      picture: "https://randomuser.me/api/portraits",
      time: "1 hour ago",
      post: "https://source.unsplash.com/random",
      caption:
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer nec odio. Praesent libero. Sed cursus ante dapibus diam. Sed nisi. Nulla quis sem at nib",
      likes: 200,
      comments: 30,
    },
  ];
  return (
    <div className="flex h-screen">
      <SideBar />
      <div className="pt-7 px-7 flex-1 flex justify-center space-x-12">
        <div className=" w-[40%]">
          <Tabs defaultValue="suggested">
            <div className=" flex justify-between w-full">
              <p className="font-semibold text-lg">Your Feed</p>
              <TabsList>
                <TabsTrigger value="suggested" className="px-6 font-semibold">
                  Suggested
                </TabsTrigger>
                <TabsTrigger value="following" className="px-6 font-semibold">
                  Following
                </TabsTrigger>
              </TabsList>
            </div>
            <div className="flex flex-col items-center">
              <ScrollArea className="h-[90vh] mt-4 w-full">
                <Post />
                <Post />
                <ScrollBar
                  orientation="vertical"
                  className="ml-4 border-r-card"
                />
              </ScrollArea>
            </div>
          </Tabs>
        </div>
        <Suggested />
      </div>
    </div>
  );
}

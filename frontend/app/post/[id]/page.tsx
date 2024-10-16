"use client";
import Comments from "@/components/sections/post/comments";
import PostPage from "@/components/sections/post/post-page";
import SideBar from "@/components/sections/side-nav";

export default function Post({ params }: { params: { id: string } }) {
  return (
    <div className="flex h-screen">
      <SideBar />
      <div className="pt-7 px-7 flex-1 flex justify-center">
        <div className="w-[40%]">
          <PostPage />
        </div>
        <div className="w-[40%]">
          <Comments />
        </div>
      </div>
    </div>
  );
}

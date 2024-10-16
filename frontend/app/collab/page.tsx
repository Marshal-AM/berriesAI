"use client";
import SideBar from "@/components/sections/side-nav";

export default function DashboardPage() {
  return (
    <div className="flex h-screen ">
      <SideBar />
      <div className="flex-1">
        <p>Hello</p>
      </div>
    </div>
  );
}

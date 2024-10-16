import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import Image from "next/image";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { BarChart, Heart, MessageCircle, UserMinusIcon } from "lucide-react";
import { Input } from "@/components/ui/input";
export default function PostPage() {
  return (
    <Card className="mb-4 mr-4 h-[90vh]">
      <CardHeader className="p-3 m-0">
        <div className="flex justify-between">
          <div
            className="flex items-center space-x-4 cursor-pointer"
            onClick={() => {
              // TODO: Go to profile page
            }}
          >
            <Image
              src={"/avatar.jpeg"}
              width={30}
              height={30}
              alt="Avatar"
              className="rounded-full"
            />
            <div className="flex flex-col">
              <p className="font-semibold hover:scale-105 hover:-translate-y-[1px] transition duration-150 ease-in-out">
                gabrielaxy.aptos
              </p>
              <p className="text-xs text-muted-foreground">📍Singapore</p>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <p className="font-medium text-xl cursor-pointer px-2">...</p>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                onSelect={() => {
                  // TODO: Implement unfollow
                }}
                className="gap-2 cursor-pointer"
              >
                <UserMinusIcon className="h-4 w-4" /> Unfollow
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent className="py-2 px-0 m-0 ">
        <div className="flex justify-center bg-card">
          <Image src={"/post/hi.jpg"} width={500} height={500} alt="Post" />
        </div>
        <div className="flex space-x-4 p-3 text-muted-foreground">
          <div className="flex space-x-1 items-center">
            <Heart className="h-5 w-5 fill-red-500 text-red-500" />
            <p className="text-sm">100</p>
          </div>
          <div className="flex space-x-1 items-center">
            <MessageCircle className="h-5 w-5" />
            <p className="text-sm">100</p>
          </div>
          <TooltipProvider>
            <Tooltip delayDuration={50}>
              <TooltipTrigger>
                <div className="flex space-x-1 items-center">
                  <BarChart className="h-5 w-5" />
                  <p className="text-sm">??</p>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Coming Soon</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <p className="font-semibold px-3">
          gabrielaxy.aptos &nbsp;
          <span className="text-sm font-medium">too cool for this app</span>
        </p>
      </CardContent>
    </Card>
  );
}

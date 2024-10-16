import Image from "next/image";
import { Button } from "../button";
import { X } from "lucide-react";

export default function SuggestedProfile({
  name,
  username,
  image,
  action,
}: {
  name: string;
  username: string;
  image: string;
  action: number;
}) {
  return (
    <div className="flex justify-between ">
      <div
        className="flex items-center  space-x-3 py-2 hover:scale-110 cursor-pointer"
        onClick={() => {
          // TODO: navigate to user profile
        }}
      >
        <Image
          src={image}
          width={30}
          height={30}
          alt="Profile"
          className="rounded-full"
        />
        <div className="flex flex-col">
          <p className="text-white text-sm font-semibold">{username}</p>
          <p className="text-sm text-muted-foreground">{name}</p>
        </div>
      </div>
      {action == 1 ? (
        <Button
          className="bg-transparent text-white hover:bg-transparent hover:font-semibold  hover:-translate-y-1 hover:scale-110 transition ease-in-out duration-150"
          onClick={() => {
            // TODO: Follow user
          }}
        >
          Follow
        </Button>
      ) : (
        <Button variant={"ghost"}>
          <X className="w-4 h-4" />
        </Button>
      )}
    </div>
  );
}

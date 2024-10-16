import { Heart, MessageCircle } from "lucide-react";
import Image from "next/image";
import { useRouter } from "next/navigation";

export default function Notifcation({
  type,
  user,
  image,
  url,
}: {
  type: number;
  user: string;
  image: string;
  url: string;
}) {
  const router = useRouter();
  return (
    <div
      className="flex justify-between cursor-pointer hover:scale-105 hover:translate-x-2 transition duration-100 ease-in-out"
      onClick={() => {
        router.push(url);
      }}
    >
      <div className="flex items-center space-x-2">
        {type == 1 ? (
          <Heart className="fill-red-500 text-red-500" />
        ) : (
          <MessageCircle className="fill-white" />
        )}
        <p className="py-4 font-semibold">
          {type == 1
            ? `${user} liked your post`
            : `${user} commented on your post`}
        </p>
      </div>

      <Image src={image} width={50} height={50} alt="not" />
    </div>
  );
}

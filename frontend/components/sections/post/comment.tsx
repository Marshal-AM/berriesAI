import Image from "next/image";

export default function Comment({
  username,
  content,
  image,
}: {
  username: string;
  content: string;
  image: string;
}) {
  return (
    <div className="flex justify-between ">
      <div className="flex items-center  space-x-3 py-2 ">
        <Image
          src={image}
          width={30}
          height={30}
          alt="Profile"
          className="rounded-full cursor-pointer"
        />
        <p
          className="text-white text-sm font-semibold hover:scale-105 hover:-translate-y-1 transition duration-100 ease-out cursor-pointer"
          onClick={() => {
            // TODO: navigate to user profile
          }}
        >
          {username} &nbsp;
        </p>
        <p className="text-sm cursor-default font-medium">{content}</p>
      </div>
    </div>
  );
}

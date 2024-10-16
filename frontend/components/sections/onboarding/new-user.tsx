import { Button } from "@/components/ui/button";

import { useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { Trash } from "lucide-react";
import Image from "next/image";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { useRouter } from "next/navigation";
import WalletInfo from "@/components/ui/custom/wallet-info";
export default function NewUser() {
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState<string>("");
  const [username, setUsername] = useState<string>("");
  const [bio, setBio] = useState<string>("");
  const [niches, setNiches] = useState<string[]>([]);
  const [preferences, setPreferences] = useState<string[]>([]);
  const { toast } = useToast();

  const router = useRouter();
  const availableCatgegories = [
    "Technology",
    "Health & Fitness",
    "Travel",
    "Fashion",
    "Music",
    "Food & Drink",
    "Gaming",
    "Movies & TV Shows",
    "Business & Finance",
    "Sports",
    "Art & Design",
    "Photography",
    "Books & Literature",
    "Science",
    "Personal Development",
    "Politics",
    "Education",
    "Lifestyle",
    "Entrepreneurship",
    "Environment",
  ];

  return (
    <ScrollArea className="h-[80vh]">
      <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px] lg:w-[450px] xl:w-[550px]">
        <div className="flex justify-between items-center">
          <div className="flex space-x-3 items-end justify-center">
            <Image src={"/logo.png"} width={40} height={40} alt="Logo" />
            <p className="font-semibold text-lg mb-1">SocioBerries</p>
          </div>
          <WalletInfo />
        </div>
        <p className="text-lg font-medium text-center">Create Account</p>
        <div className="flex justify-center">
          {image != null ? (
            <div className="relative group">
              <Image
                src={image}
                alt="uploaded"
                width={150}
                height={150}
                className="cursor-pointer rounded-lg transition-opacity duration-300 ease-in-out group-hover:opacity-50"
                onClick={() => setImage(null)}
              />
              <div
                className="cursor-pointer absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2  rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300 ease-in-out"
                onClick={() => setImage(null)}
              >
                <Trash className="w-8 h-8 " />
              </div>
            </div>
          ) : (
            <label
              htmlFor="imageUpload"
              className="flex justify-center items-center w-[150px] h-[150px] border border-dashed border-secondary cursor-pointer"
            >
              <input
                id="imageUpload"
                type="file"
                accept="image/*"
                onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                  const file = event.target.files?.[0]; // Get the first selected file

                  if (file) {
                    if (file.size > 2 * 1024 * 1024) {
                      // Restrict size to 2MB
                      setError("File size exceeds 2MB");
                      toast({
                        variant: "destructive",
                        title: "Error",
                        description: "File size exceeds 2MB",
                      });
                      setImage(null);
                      return;
                    }

                    const reader = new FileReader();
                    reader.onloadend = () => {
                      setImage(reader.result as string); // Convert the file to a base64 string
                      setError(null);
                    };
                    reader.readAsDataURL(file); // Read the file as a data URL
                  }
                }}
                className="hidden"
              />
              <p className="text-xs text-muted-foreground text-center">
                Click here to <br /> upload
              </p>
            </label>
          )}
        </div>
        <div className="flex flex-col space-y-2">
          <p className="font-semibold text-sm">User Name</p>
          <Input
            className="ml-1"
            value={username}
            onChange={(e) => {
              setUsername(e.target.value);
            }}
          ></Input>
          <p className="text-xs text-muted-foreground">
            A unique username or id of your profile.
          </p>
        </div>
        <div className="flex flex-col space-y-2">
          <p className="font-semibold text-sm">Display Name</p>
          <Input
            className="ml-1"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
            }}
          ></Input>
          <p className="text-xs text-muted-foreground">
            The name that will be displayed on your profile.
          </p>
        </div>
        <div className="flex flex-col space-y-2">
          <p className="font-semibold text-sm">Bio</p>
          <Textarea
            className="ml-1"
            value={bio}
            onChange={(e) => {
              setBio(e.target.value);
            }}
          />
          <p className="text-xs text-muted-foreground">
            Tell something about yourself and what you do.
          </p>
        </div>
        <div className="flex flex-col space-y-2">
          <p className="font-semibold text-sm">Your Niche</p>
          <div className="">
            {availableCatgegories.map((category, idx) => (
              <Badge
                key={idx}
                className={`m-1 cursor-pointer ${
                  niches.includes(category)
                    ? "bg-primary"
                    : `bg-secondary ${
                        niches.length >= 5
                          ? "text-muted-foreground hover:bg-secondary cursor-not-allowed"
                          : "text-white"
                      }`
                }`}
                onClick={() => {
                  if (niches.includes(category)) {
                    setNiches(niches.filter((n) => n !== category));
                  } else {
                    setNiches([...niches, category]);
                  }
                }}
              >
                {category}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Choose your content niche to reach your target audience.
          </p>
        </div>
        <div className="flex flex-col space-y-2">
          <p className="font-semibold text-sm">Preferences</p>
          <div className="">
            {availableCatgegories.map((category, idx) => (
              <Badge
                key={idx}
                className={`m-1 cursor-pointer ${
                  preferences.includes(category)
                    ? "bg-primary"
                    : `, bg-secondary ${
                        preferences.length >= 5
                          ? "text-muted-foreground hover:bg-secondary cursor-not-allowed"
                          : "text-white"
                      }`
                }`}
                onClick={() => {
                  if (preferences.includes(category)) {
                    setPreferences(preferences.filter((n) => n !== category));
                  } else {
                    setPreferences([...preferences, category]);
                  }
                }}
              >
                {category}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Choose the type of content you want to see on your feed.
          </p>
        </div>
        <Button
          variant={"outline"}
          className="font-semibold"
          onClick={() => {
            // TODO: Store image to ipfs
            // TODO: Send a transaction to create a new user
            // TODO: Update zustand state
            router.push("/home");
          }}
        >
          Get Started
        </Button>
      </div>
      <ScrollBar orientation="vertical" />
    </ScrollArea>
  );
}

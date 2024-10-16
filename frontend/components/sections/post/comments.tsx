import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Comment from "./comment";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
export default function Comments() {
  const comments = [
    {
      id: 1,
      profile: {
        name: "Marshal",
        username: "marshal.aptos",
        image: "/onboarding/1.png",
      },
      content: "Giga Chad right here 🦍",
    },
    {
      id: 2,
      profile: {
        name: "Tabitha",
        username: "tabitha.aptos",
        image: "/onboarding/2.jpg",
      },
      content: "👍🔥",
    },
  ];
  return (
    <ScrollArea className="h-[90vh]">
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Comments</CardTitle>
        </CardHeader>
        <CardContent>
          {comments.map((c, idx) => (
            <Comment
              key={idx}
              username={c.profile.username}
              image={c.profile.image}
              content={c.content}
            />
          ))}
          <ScrollBar orientation="vertical" />
        </CardContent>
      </Card>
    </ScrollArea>
  );
}

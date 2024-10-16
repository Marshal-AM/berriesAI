import * as React from "react";
import Autoplay from "embla-carousel-autoplay";

import { Card, CardContent } from "@/components/ui/card";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import Image from "next/image";

export default function OnboardingCarousel() {
  const plugin = React.useRef(
    Autoplay({ delay: 2000, stopOnInteraction: true })
  );

  return (
    <Carousel
      plugins={[plugin.current]}
      className="w-full h-full"
      onMouseEnter={plugin.current.stop}
      onMouseLeave={plugin.current.reset}
    >
      <CarouselContent className="w-full h-full p-0 m-0">
        {Array.from({ length: 4 }).map((_, index) => (
          <CarouselItem key={index} className="w-full h-full p-0 m-0 ">
            <img
              src={`/onboarding/${index + 1}.${index == 0 ? "png" : "jpg"}`}
              alt="onboarding"
              className="object-cover w-full h-full"
            />
          </CarouselItem>
        ))}
      </CarouselContent>
    </Carousel>
  );
}

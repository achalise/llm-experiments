'use client'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import * as React from "react"
import { Input } from "@/components/ui/input"
import { useData } from "./useData"

type CardProps = React.ComponentProps<typeof Card>

const ClaimProcessorPage = ({ className, ...props }: CardProps) => {
  let { data, callApi } = useData();
  let [query, setQuery] = React.useState<string>("");
  const lastMessage = React.useRef<null | HTMLElement>(null);

  React.useEffect(() => {
    lastMessage.current?.scrollIntoView({ behavior: "smooth" });
  }, [data]);

  const submitQuery = () => {
    callApi(query);
    setQuery("");
  }

  return (
    <div className="flex min-h-screen bg-slate-100 items-center justify-center">
      <Card className={cn("w-[1000px] h-[800px] grid grid-rows-[min-content_1fr_min-content]  gap-4", className)} {...props}>
        <CardHeader>
          <CardTitle className={"text-2xl font-bold"}> Claim Processor Application</CardTitle>
          <CardDescription>
            Process claims for insurance companies.
          </CardDescription>
        </CardHeader>
        <ScrollArea className="h-full w-full rounded-md border">

          <CardContent className="space-y-3">
            {(data as any)?.map((message: any, index: number) => (
              <div key={message.id} ref={index + 1 === (data as any)?.length ? lastMessage : null}>
                <p><span className="mr-3 font-bold">{message.id[2]}</span><span>{message.kwargs.content}</span></p>
              </div>
            ))}
          </CardContent>
        </ScrollArea>
        <CardFooter>
          <form className="w-full space-x-2 flex justify-end items-center">
            <Input onChange={(e) => setQuery(e.target.value)} value={query}></Input>
            <Button type="button" className="btn btn-primary" onClick={submitQuery}>
              Send
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}

export default ClaimProcessorPage;
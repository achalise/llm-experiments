import { ChatOpenAI } from "@langchain/openai";

export const model = new ChatOpenAI({model: "gpt-3.5-turbo", temperature: 0});
//export const model = new ChatOpenAI({model: "gpt-4o", temperature: 0.7});

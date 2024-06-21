import { ChatOpenAI } from "@langchain/openai";

export const model = new ChatOpenAI({model: "gpt-3.5-turbo"});

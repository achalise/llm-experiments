import { Request, Response } from "express";
import { model } from "../openai/model";
import { sqlAgent } from "../datasource/agent";
import { AgentExecutor } from "langchain/dist/agents/executor";

export const chatHandler = async (request: Request, response: Response) => {
  const query = request.query["query"];
  console.log(`Executing query for ${query}`);

  // to call ther openai model
  //const answer = await model.invoke(`${query}`);


  const agent: AgentExecutor = await sqlAgent();

  const queryString = `${query}`;

  // to invoke the agent
  const result = await agent.invoke({ input: queryString });

  console.log(`Got output ${JSON.stringify(result)}`);

  response.send(
    result.output
  );
}
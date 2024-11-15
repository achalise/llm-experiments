import { HumanMessage } from "@langchain/core/messages";
import { graph } from "./agent/graph.js";
const config = { configurable: { thread_id: "thread1" } };
let streamResults = async () => { 
    //const graph = await ();
    const results = graph.withConfig(config).stream(
    {
      messages: [
        new HumanMessage({
          content: `I would like to submit a claim, my details are as below:

name: Arun Chalise
insurance type: car
address: 300 Pramatta Road, Sydney 2000
amount: 5000
description: I had a car accident under no fault

email: user@email.test
policyNumber: 1122323232

When my application is completed, please write a satire for me.
`,
        }),
      ],
    },
    { recursionLimit: 100 },
  );
  
  for await (const output of await results) {
    if (!output?.__end__) {
      console.log(output);
    }
}}

streamResults();

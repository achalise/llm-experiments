/**
 * Starter LangGraph.js Template
 * Make this code your own!
 */
import { END, MemorySaver, StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateAnnotation } from "./state.js";
import { model } from "./model.js";
import { approvePaymentTool, createOrUpdateClaimTool, fraudCheckTool, sendConfirmationEmailTool, userDetailsTool } from "./tools.js";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage } from "@langchain/core/messages";
import { systemSetUpMessage } from "../systemMessage.js";
import { validateClaimDetail } from "./validateClaimDetails.js";
import { validateApprovalRequest } from "./validateApprovalRules.js";

const ALL_TOOLS = [userDetailsTool, fraudCheckTool, createOrUpdateClaimTool, approvePaymentTool, sendConfirmationEmailTool];
const toolNode = new ToolNode(ALL_TOOLS);

/**
 * Define a node, these do the work of the graph and should have most of the logic.
 * Must return a subset of the properties set in StateAnnotation.
 * @param state The current state of the graph.
 * @param config Extra parameters passed into the state graph.
 * @returns Some subset of parameters of the graph state, used to update the state
 * for the edges and nodes executed next.
 */
const callModel = async (
  state: typeof StateAnnotation.State,
  _config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> => {
  /**
   * Do some work... (e.g. call an LLM)
   * For example, with LangChain you could do something like:
   *
   * ```bash
   * $ npm i @langchain/anthropic
   * ```
   *
   * ```ts
   * import { ChatAnthropic } from "@langchain/anthropic";
   * const model = new ChatAnthropic({
   *   model: "claude-3-5-sonnet-20240620",
   *   apiKey: process.env.ANTHROPIC_API_KEY,
   * });
   * const res = await model.invoke(state.messages);
   * ```
   *
   * Or, with an SDK directly:
   *
   * ```bash
   * $ npm i openai
   * ```
   *
   * ```ts
   * import OpenAI from "openai";
   * const openai = new OpenAI({
   *   apiKey: process.env.OPENAI_API_KEY,
   * });
   *
   * const chatCompletion = await openai.chat.completions.create({
   *   messages: [{
   *     role: state.messages[0]._getType(),
   *     content: state.messages[0].content,
   *   }],
   *   model: "gpt-4o-mini",
   * });
   * ```
   */
  console.log("Current state:", state);
  const { messages } = state;

  const systemMessage = {
    role: "system",
    content: systemSetUpMessage()
  }
  const result = await model.bindTools(ALL_TOOLS).invoke([systemMessage, ...messages]);
  return { messages: [result] };
};

/**
 * Routing function: Determines whether to continue research or end the builder.
 * This function decides if the gathered information is satisfactory or if more research is needed.
 *
 * @param state - The current state of the research builder
 * @returns Either "callModel" to continue research or END to finish the builder
 */
export const route = (
  state: typeof StateAnnotation.State,
): "__end__" | "agent" => {
  if (state.messages.length > 0) {
    return "__end__";
  }
  // Loop back
  return "agent";
};

const shouldContinue = (state: typeof StateAnnotation.State) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  // Cast here since `tool_calls` does not exist on `BaseMessage`
  const messageCastAI = lastMessage as AIMessage;
  if (messageCastAI._getType() !== "ai" || !messageCastAI.tool_calls?.length) {
    // LLM did not call any tools, or it's not an AI message, so we should end.
    return END;
  }
  const { tool_calls } = messageCastAI;
  if (!tool_calls?.length) {
    throw new Error(
      "Expected tool_calls to be an array with at least one element"
    );
  }
  return tool_calls.map((tc) => {
    if (tc.name === "create_or_update_claim") {
      // The user is trying to purchase a stock, route to the verify purchase node.
      return "prepare_claim_detail";
    } else if(tc.name === "approve_payment") {
      console.log(`agent is trying to approve the payment`);
      return "execute_approve_payment";
    } else {
      return "tools";
    }
  });
}

// Finally, create the graph itself.
const builder = new StateGraph(StateAnnotation)
  // Add the nodes to do the work.
  // Chaining the nodes together in this way
  // updates the types of the StateGraph instance
  // so you have static type checking when it comes time
  // to add the edges.
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addNode("prepare_claim_detail", validateClaimDetail)
  .addNode("execute_approve_payment", validateApprovalRequest)
  // Regular edges mean "always transition to node B after node A is done"
  // The "__start__" and "__end__" nodes are "virtual" nodes that are always present
  // and represent the beginning and end of the builder.
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addEdge("prepare_claim_detail", "tools")
  .addEdge("execute_approve_payment", "tools")
  // Conditional edges optionally route to different nodes (or end)
  .addConditionalEdges("agent", shouldContinue, ["tools", "prepare_claim_detail", "execute_approve_payment",
    END]);

export const graph = builder.compile(
  // The LangGraph Studio/Cloud API will automatically add a checkpointer
  // only uncomment if running locally
  {checkpointer: new MemorySaver(),}
);

graph.name = "New }Agent";

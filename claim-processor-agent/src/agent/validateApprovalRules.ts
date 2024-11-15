import { AIMessage } from "@langchain/core/messages";
import { StateAnnotation } from "./state.js";
import { NodeInterrupt } from "@langchain/langgraph";

export const validateApprovalRequest = async (state: typeof StateAnnotation.State) => {
    const { messages, claimDetails } = state;

    const lastMessage = messages[messages.length - 1];
    if (lastMessage.getType() !== "ai") {
        throw new Error("Expected the last message to be an AI message");
    }

    // Cast here since `tool_calls` does not exist on `BaseMessage`
  const messageCastAI = lastMessage as AIMessage;
  const approvePaymentTool = messageCastAI.tool_calls?.find(
    (tc) => tc.name === "approve_payment"
  );
  if (!approvePaymentTool) {
    throw new Error(
      "Expected the last AI message to have a approve_payment tool call"
    );
  }
  console.log(`the claim amount is  ${claimDetails.claimAmount}`);
  if (claimDetails.claimAmount > 50000) {
    console.log(`Claim amount is too large, needs human intervention`);
    throw new NodeInterrupt("Please approve the payment for amount greater than 30000");
  } 

  return {
    paymentApproved : true,
    claimDetails : {...state.claimDetails,
        claimStatus: "APPROVED"
    }
  };

}

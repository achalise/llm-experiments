import { AIMessage } from "@langchain/core/messages";
import { StateAnnotation } from "./state.js";

export const validateClaimDetail = async (state: typeof StateAnnotation.State) => {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    if (lastMessage.getType() !== "ai") {
        throw new Error("Expected the last message to be an AI message");
    }

    // Cast here since `tool_calls` does not exist on `BaseMessage`
  const messageCastAI = lastMessage as AIMessage;
  const createOrUpdateClaimTool = messageCastAI.tool_calls?.find(
    (tc) => tc.name === "create_or_update_claim"
  );
  if (!createOrUpdateClaimTool) {
    throw new Error(
      "Expected the last AI message to have a create_or_update_claim tool call"
    );
  }
  // mostly validation here... to ensure claim details is correctly formed
  return {
    claimDetails : {...state.claimDetails, 
        firstName: createOrUpdateClaimTool.args?.firstName,
        lastName: createOrUpdateClaimTool.args?.lastName,
        claimType: createOrUpdateClaimTool.args?.claimType,
        claimStatus: createOrUpdateClaimTool.args?.claimStatus,
        claimDescription: createOrUpdateClaimTool.args?.claimDescription,
        claimAmount: createOrUpdateClaimTool.args?.claimAmount,
        userEmail: createOrUpdateClaimTool.args?.userEmail,
        policyNumber: createOrUpdateClaimTool.args?.policyNumber,
        address: createOrUpdateClaimTool.args?.address
    }
  };

}

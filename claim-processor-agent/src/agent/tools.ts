import { tool } from "@langchain/core/tools";
import { claimDetailsSchema } from "./state.js";
import { z } from "zod";

export const userDetailsTool = tool((input) => {
   console.log(`Getting user details for ${JSON.stringify(input)} ...`);
    return JSON.stringify({
        userEmail: input.userEmail,
        userExists: true
    })
}, {name: "user_details", schema: z.object({
    userEmail: z.string().email().describe("Email of user"),
})
});

export const fraudCheckTool = tool((input) => {
    console.log(`Fraud check for ${JSON.stringify(input)}...`);
     return JSON.stringify({
         result: "SUCCESS"
     })
 }, {name: "fraud_Check", schema: z.object({
     firstName: z.string().describe("First name of user"),
     lastName: z.string().describe("Last name of user"),
     userEmail: z.string().email().describe("Email of user"),
     policyNumber: z.string().describe("Policy number")
 })
 });

 export const createOrUpdateClaimTool = tool((input) => {
    console.log(`Creating or updating claim...`);
    if(!input.claimId || input.claimId === "UNDEFINED") {
        let claimId = Math.random().toString(36).substr(2, 9);
        input.claimId = claimId;
        return `Successfully created claim for policy number ${input.policyNumber} and claimId ${input.claimId}.`;
    } else {
        return `Successfully updated claim with ID ${input.claimId} for policy number ${input.policyNumber} to status ${input.claimStatus}`;
    }
 }, {name: "create_or_update_claim", schema: claimDetailsSchema});

 export const approvePaymentTool = tool((input) => {
    console.log(`Approving the claim...`);
    if(input.claimStatus !== "FRAUD_CHECK_SUCCESS") {
        return `Payment is not approved as the fraud check failed for the claim`;
    } else {
        return `Approved the claim with ID ${input.claimId} for policy number ${input.policyNumber} to status APPROVED.`;
    }
 }, {name: "approve_payment", schema: claimDetailsSchema, description: "Approve payment for the claim"})

 export const sendConfirmationEmailTool = tool((input) => {
    console.log(`Sending email for the claim...`);
    return `Email sent to ${input.userEmail} for the claim with status ${input.claimStatus}`;
 }, {name: "send_confirmation_email", schema: claimDetailsSchema, description: "Send email to the customer when claim is approved"})

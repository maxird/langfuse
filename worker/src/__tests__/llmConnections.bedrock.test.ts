import { describe, test, expect } from "bun:test";
import { fetchLLMCompletion } from "@langfuse/shared/src/server";
import {
  ChatMessageType,
  LLMAdapter,
  BEDROCK_USE_DEFAULT_CREDENTIALS,
} from "@langfuse/shared";
import { z } from "zod/v3";

/**
 * Bedrock LLM Connection Tests - Profile-Based Credentials
 *
 * This test suite is specifically for testing Bedrock with AWS profile-based credentials.
 * It does NOT require Docker infrastructure - only AWS credentials and network access.
 *
 * Required environment variables:
 * - AWS_PROFILE (optional, defaults to "default")
 * - LANGFUSE_LLM_CONNECTION_BEDROCK_REGION (required, e.g., "us-east-1")
 *
 * Optional environment variables:
 * - LANGFUSE_LLM_CONNECTION_BEDROCK_MODEL (defaults to anthropic.claude-3-sonnet-20240229-v1:0)
 *
 * Setup:
 * 1. Ensure your AWS credentials are configured in ~/.aws/credentials
 * 2. Ensure you have Bedrock access in your AWS account
 * 3. Set the required environment variables
 * 4. Run: pnpm test llmConnections.bedrock.test.ts
 */

// Eval schema matching production usage
const evalOutputSchema = z.object({
  score: z.number(),
  reasoning: z.string(),
});

// Common tool definition for tool calling tests
const weatherTool = {
  name: "get_weather",
  description: "Get the current weather for a location",
  parameters: {
    type: "object",
    properties: {
      location: {
        type: "string",
        description: "The city name, e.g. 'Paris' or 'London'",
      },
    },
    required: ["location"],
  },
};

describe("Bedrock with Profile-Based Credentials", () => {
  // Allow user to override model via environment variable
  const MODEL =
    process.env.LANGFUSE_LLM_CONNECTION_BEDROCK_MODEL ||
    "anthropic.claude-3-sonnet-20240229-v1:0";

  const checkEnvVars = () => {
    if (!process.env.LANGFUSE_LLM_CONNECTION_BEDROCK_REGION) {
      throw new Error(
        "LANGFUSE_LLM_CONNECTION_BEDROCK_REGION not set. " +
          "This test requires a valid AWS region (e.g., 'us-east-1'). " +
          "Set the environment variable to run this test.",
      );
    }

    console.log("Test Configuration:");
    console.log("  AWS Profile:", process.env.AWS_PROFILE || "default");
    console.log(
      "  Region:",
      process.env.LANGFUSE_LLM_CONNECTION_BEDROCK_REGION,
    );
    console.log("  Model:", MODEL);
  };

  const getApiKey = () => {
    checkEnvVars();
    return BEDROCK_USE_DEFAULT_CREDENTIALS;
  };

  const getConfig = () => {
    return {
      region: process.env.LANGFUSE_LLM_CONNECTION_BEDROCK_REGION!,
    };
  };

  test("simple completion", async () => {
    checkEnvVars();

    const { completion } = await fetchLLMCompletion({
      streaming: false,
      messages: [
        {
          role: "user",
          content:
            "What is 2+2? Answer only with the number. You are being tested for structured output capabilities. Just do the work.",
          type: ChatMessageType.PublicAPICreated,
        },
      ],
      modelParams: {
        provider: "bedrock",
        adapter: LLMAdapter.Bedrock,
        model: MODEL,
        temperature: 0,
        max_tokens: 10,
      },
      apiKey: BEDROCK_USE_DEFAULT_CREDENTIALS,
      config: getConfig(),
      context: {
        tracing: "langfuse",
        credentials: "user",
      },
    });

    expect(typeof completion).toBe("string");
    expect(completion).toContain("4");
  }, 30_000);

  test("structured output - eval schema", async () => {
    checkEnvVars();

    const { completion } = await fetchLLMCompletion({
      streaming: false,
      messages: [
        {
          role: "user",
          content:
            "Evaluate the quality of this response: 'The answer is 42.' Provide a score from 0-100 and reasoning.",
          type: ChatMessageType.PublicAPICreated,
        },
      ],
      modelParams: {
        provider: "bedrock",
        adapter: LLMAdapter.Bedrock,
        model: MODEL,
        temperature: 0,
        max_tokens: 200,
      },
      structuredOutputSchema: evalOutputSchema,
      apiKey: getApiKey(),
      config: getConfig(),
      context: {
        tracing: "langfuse",
        credentials: "user",
      },
    });

    const parsed = evalOutputSchema.safeParse(completion);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(typeof parsed.data.score).toBe("number");
      expect(typeof parsed.data.reasoning).toBe("string");
      expect(parsed.data.reasoning.length).toBeGreaterThan(0);

      console.log("Structured Output Test Results:");
      console.log("  Score:", parsed.data.score);
      console.log("  Reasoning:", parsed.data.reasoning);
    }
  }, 30_000);

  test("tool calling", async () => {
    checkEnvVars();

    const { completion } = await fetchLLMCompletion({
      streaming: false,
      messages: [
        {
          role: "user",
          content: "What's the weather like in Paris?",
          type: ChatMessageType.PublicAPICreated,
        },
      ],
      modelParams: {
        provider: "bedrock",
        adapter: LLMAdapter.Bedrock,
        model: MODEL,
        temperature: 0,
        max_tokens: 100,
      },
      tools: [weatherTool],
      apiKey: getApiKey(),
      config: getConfig(),
      context: {
        tracing: "langfuse",
        credentials: "user",
      },
    });

    expect(completion).toHaveProperty("tool_calls");
    expect(Array.isArray(completion.tool_calls)).toBe(true);
    expect(completion.tool_calls.length).toBeGreaterThan(0);
    expect(completion.tool_calls[0].name).toBe("get_weather");
    expect(completion.tool_calls[0].args).toHaveProperty("location");

    console.log("Tool Calling Test Results:");
    console.log("  Tool called:", completion.tool_calls[0].name);
    console.log("  Arguments:", JSON.stringify(completion.tool_calls[0].args));
  }, 30_000);
});

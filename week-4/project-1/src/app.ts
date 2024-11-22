/**
 * @fileoverview Main application entry point that sets up a GitHub App webhook server
 * to process pull request events and perform automated code reviews.
 */

import { Octokit } from "@octokit/rest";
import { createNodeMiddleware } from "@octokit/webhooks";
import { WebhookEventMap } from "@octokit/webhooks-definitions/schema";
import * as http from "http";
import { App } from "octokit";
import { Review } from "./constants";
import { env } from "./env";
import { processPullRequest } from "./review-agent";
import { applyReview } from "./reviews";

/**
 * Creates a new instance of the GitHub App using environment credentials
 */
const reviewApp = new App({
  appId: env.GITHUB_APP_ID,
  privateKey: env.GITHUB_PRIVATE_KEY,
  webhooks: {
    secret: env.GITHUB_WEBHOOK_SECRET,
  },
});

/**
 * Retrieves the list of changed files for a pull request using GitHub's API
 * @param payload - The pull request webhook event payload
 * @returns Promise resolving to array of changed files or empty array on error
 */
const getChangesPerFile = async (payload: WebhookEventMap["pull_request"]) => {
  try {
    const octokit = await reviewApp.getInstallationOctokit(
      payload.installation.id
    );
    const { data: files } = await octokit.rest.pulls.listFiles({
      owner: payload.repository.owner.login,
      repo: payload.repository.name,
      pull_number: payload.pull_request.number,
    });
    console.dir({ files }, { depth: null });
    return files;
  } catch (exc) {
    console.log("exc");
    return [];
  }
};

/**
 * Handles the 'pull_request.opened' webhook event
 * Processes the pull request by:
 * 1. Getting changed files
 * 2. Running code review analysis
 * 3. Applying review comments back to GitHub
 * 
 * @param octokit - Authenticated Octokit REST client
 * @param payload - Pull request webhook event payload
 */
async function handlePullRequestOpened({
  octokit,
  payload,
}: {
  octokit: Octokit;
  payload: WebhookEventMap["pull_request"];
}) {
  console.log(
    `Received a pull request event for #${payload.pull_request.number}`
  );
  try {
    console.log("pr info", {
      id: payload.repository.id,
      fullName: payload.repository.full_name,
      url: payload.repository.html_url,
    });
    const files = await getChangesPerFile(payload);
    const review: Review = await processPullRequest(
      octokit,
      payload,
      files,
      true
    );
    await applyReview({ octokit, payload, review });
    console.log("Review Submitted");
  } catch (exc) {
    console.log(exc);
  }
}

// Register webhook handler for pull request opened events
//@ts-ignore
reviewApp.webhooks.on("pull_request.opened", handlePullRequestOpened);

/**
 * Server configuration
 */
const port = process.env.PORT || 3000;
const reviewWebhook = `/api/review`;

/**
 * Create middleware to handle GitHub webhook requests
 */
const reviewMiddleware = createNodeMiddleware(reviewApp.webhooks, {
  path: "/api/review",
});

/**
 * HTTP server that handles incoming webhook requests
 * Routes /api/review requests to the webhook middleware
 * Returns 404 for all other paths
 */
const server = http.createServer((req, res) => {
  if (req.url === reviewWebhook) {
    reviewMiddleware(req, res);
  } else {
    res.statusCode = 404;
    res.end();
  }
});

/**
 * Start the webhook server and listen for incoming requests
 */
server.listen(port, () => {
  console.log(`Server is listening for events.`);
  console.log("Press Ctrl + C to quit.");
});

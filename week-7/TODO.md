# TODO

Take a look at the TODOs throughout the repo, namely:

- [ ] src/app/page.tsx: Update the UI and handle the API response as needed
- [ ] src/app/api/chat/route.ts: Implement the chat API with Groq and web scraping with Cheerio and Puppeteer
- [ ] src/middleware.ts: Implement the code here to add rate limiting with Redis

## Project Submission Requirements

A chat interface where a user can:

- [ ] Paste in a set of URLs and get a response back with the context of all the URLs through an LLM
- [ ] Ask a question and get an answer with sources cited
- [ ] Share their conversation with others, and let them continue with their conversation

## Challenges (Attempt these after you have finished the requirements above)

- [ ] Build a comprehensive solution to extract content from any kind of URL or data source, such as YouTube videos, PDFs, CSV files, and images
- [ ] Generate visualizations from the data such as bar charts, line charts, histograms, etc.
- [ ] Implement a hierarchical web crawler that starts at a given URL and identifies all relevant links on the page (e.g., hyperlinks, embedded media links, and scrapes the content from those links as well)

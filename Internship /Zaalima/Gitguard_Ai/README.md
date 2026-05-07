# GitGuard AI

GitGuard AI is a PR-focused code review assistant that analyzes GitHub pull request diffs, posts AI review comments, and shows review history in a React dashboard.

## Features
- Webhook-based PR analysis (`opened`, `reopened`, `synchronize`)
- AI risk scoring: `Low`, `Medium`, `High`, `Critical`
- Issue-level feedback with recommendation and optional file/line
- Manual code analyzer in UI
- Repo-level review settings (`strictMode`, `ignoreStyling`)
- Dashboard with filters, search, and live refresh

## Tech Stack
- Backend: Node.js, Express, Axios, OpenAI API
- Frontend: React + Vite + Tailwind CSS
- Integration: GitHub Webhooks + GitHub REST API

## Project Structure
```text
backend/
  server.js
  routes/
    webhook.js
    analyze.js
    history.js
    settings.js
  services/
    githubService.js
    diffProcessor.js
    formatter.js
    aiService.js
  models/
    reviewHistory.js
    settings.js
  frontend/
    src/
      pages/
      components/
      api.js
```

## Prerequisites
- Node.js 18+
- npm
- GitHub Personal Access Token (repo access)
- OpenAI API key

## Environment Variables
Create `backend/.env`:

```env
PORT=5003
GITHUB_TOKEN=your_github_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
AI_TIMEOUT_MS=30000
AI_MAX_RETRIES=2
WEBHOOK_SECRET=your_webhook_secret
```

## Run Locally
### 1) Backend
```bash
cd backend
npm install
npm run dev
```
Backend runs on `http://localhost:5003` (unless `PORT` changed).

### 2) Frontend
```bash
cd backend/frontend
npm install
npm run dev
```
Frontend runs on `http://localhost:5173`.

## UI Pages
- **Dashboard**: Review history, risk stats, repo search, filters
- **Analyze**: Paste code and run instant manual AI review
- **Settings**: Save repo-specific analysis behavior

## Webhook Setup (GitHub)
1. Open your repository settings → **Webhooks** → **Add webhook**
2. Payload URL: `http://<your-public-backend-url>/webhook`
3. Content type: `application/json`
4. Secret: use `WEBHOOK_SECRET`
5. Events: **Pull requests**

When a PR is opened/updated, GitGuard AI:
1. Fetches PR diff
2. Extracts added lines
3. Sends cleaned diff to AI for analysis
4. Posts review comment on the PR
5. Stores review in history for dashboard display

## API Endpoints
- `POST /webhook` - GitHub webhook receiver
- `POST /analyze` - Manual code analysis
- `GET /history` - Review history
- `GET /settings?repo=owner/repo` - Fetch repo settings
- `POST /settings` - Update repo settings

## Notes
- Current `reviewHistory` and `settings` stores are in-memory (reset on backend restart).
- `verifySignature` middleware is currently in debug bypass mode in this codebase.

## Suggested Next Improvements
- Persist history/settings in a database (SQLite/Postgres/MongoDB)
- Re-enable strict webhook signature verification for production
- Add auth + multi-user repo management
- Add CI tests for routes and AI response validation

## License
ISC

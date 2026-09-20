# Nifty Monitor

Free, browser-only Nifty 50 forecast monitor that runs on GitHub (Actions + Pages).
Every 15 minutes in market hours it reads Nifty candles, the NSE option chain, global markets and
world news, forecasts the next 15 / 30 / 60 minutes as ranges, raises alerts, and scores itself.

## Setup (about 10 minutes)
1. Add your Gemini key: repo **Settings > Secrets and variables > Actions > New repository secret**
   Name `GEMINI_API_KEY`, value = your key. Never put the key in a file.
2. Repo **Settings > Actions > General > Workflow permissions**: choose **Read and write** and Save.
3. **Actions** tab > **Nifty Monitor** > **Run workflow** (leave "force" ticked) > wait ~2 min for the green tick.
4. Repo **Settings > Pages**: Source = *Deploy from a branch*, Branch = **data**, folder **/ (root)**, Save.
5. Open `https://<your-username>.github.io/<repo-name>/` (first publish takes 1-2 minutes).

Optional Telegram alerts: create a bot with @BotFather, then add secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Notes
- Data: Yahoo Finance (delayed, unofficial) + NSE public option chain (NSE sometimes blocks cloud IPs; the
  system keeps working without it and shows a warning) + Google News/GDELT + Gemini.
- Results live on a `data` branch that is overwritten each run, so the repo never grows.
- Change thresholds and weights in `src/config.py`.
- `python tests/simulate.py 6 /tmp/out` runs a full synthetic multi-day simulation without internet.
- Not investment advice. Check the Scoreboard tab before trusting it.

## Version 2 additions
- Time-of-day and event-day aware ranges (edit `events.json` to add dates).
- "No clear edge" zone: up/down is only called when the signals are strong and agree.
- Heavyweight stocks + Bank Nifty as a fifth signal ("market breadth").
- **Backtest** workflow (Actions > Backtest > Run workflow): scores each technical indicator on ~60 days of
  candles and nudges weights only for strong, consistent edges (max +/-50%). Results show on the Scoreboard tab.
- Daily results tab: one row per market day.

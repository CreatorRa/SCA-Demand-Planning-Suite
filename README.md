# Hafen & Home – Demand Planning Suite

A web app for planners. It lets them import weekly sales and price data, explore the history, check whether price cuts pay off, and choose the 13-week forecast for the buying meeting.

## What the app does

| Tab | Planner question |
|---|---|
| 📈 Sales history | What happened, and when? Do we really sell more before the holidays? |
| 🏷️ Promotions | Do price cuts pay off? What would we have sold at the regular price? |
| 🔮 Forecast | Which forecast do we take to the buying meeting? (models compared on the last 13 weeks by MAPE, with planned promotions as input) |
| ⬇️ Export | 13-week forecast for all products and regions, downloadable as CSV |

The sidebar holds the data import, the product/region choice and the promotion settings: the discount threshold and the unit cost as a % of the regular price.

## Project structure

```
app.py                       # User interface only (tabs, charts, inputs)
data.py                      # Data import, gap handling, regular price, holiday flags
models.py                    # Promotion regression, baseline, forecast models, backtest, promo economics
weekly_sales_price_data.xls  # Sample dataset (loaded when no file is uploaded)
requirements.txt             # Python packages for Streamlit Cloud
```

Changes to the interface go in `app.py`, and changes to calculations go in `data.py` / `models.py`. Keeping them separate lets two people work at the same time without editing the same file.

---

## One-time setup (team lead)

1. **Create the repo:** on github.com, click **New repository** and name it `hafen-home-demand-suite`. Set it to **Private** and create it.
2. **Upload the files:** click **uploading an existing file**, then drag in all files. Click **Commit changes**.
3. **Add teammates:** go to **Settings → Collaborators → Add people** and enter each teammate's GitHub username.
4. **Deploy:** go to **share.streamlit.io**, sign in with GitHub, choose **Create app → Deploy a public app from GitHub**, then pick the repo, branch `main` and file `app.py`. Click **Deploy**.
   - After about 2 minutes you get a link like `https://hafen-home-....streamlit.app`. This is the link you submit.
   - The app updates automatically after every change on GitHub.

## How teammates edit (no installation needed)

**Small change (text, labels, a number):**
1. Open the file on github.com and click the ✏️ pencil icon.
2. Make the change, then click **Commit changes…** → *Create a new branch* → **Propose changes** → **Create pull request**.
3. The team lead reviews and merges. The live app updates within a minute.

**Bigger change, made with an AI assistant:**
1. Copy the content of the file you want to change into Claude or ChatGPT and describe the change in plain words, e.g. *"Add a chart comparing North and South for the selected product to the Sales history tab."*
2. Paste the new version back via the ✏️ pencil icon and commit it as above.

**Preview before publishing (optional):** on the repo page, click **Code → Codespaces → Create codespace**. In the terminal at the bottom, run:
```
pip install -r requirements.txt
streamlit run app.py
```
then open the link that appears.

## Team rules
- Never commit directly to `main`. Always use a branch and a pull request, so the live app never breaks.
- Only one person edits a given file at a time. Agree on this in the group chat.
- If the live app shows an error, open **Manage app → Logs** on Streamlit Cloud, or roll back by reverting the last pull request.

---

## Method notes (for the report)

- **Regular price:** the rolling 13-week 75th percentile of the observed price. A week counts as a promotion when the price is at least *x* % below it (default 15 %, adjustable).
- **Price & promotion regression** (per series, on log units): discount depth + log regular price + trend + Black Friday/Cyber week flag + pre-Christmas flag (1–21 Dec).
- **Baseline** ("what would we have sold at the regular price"): the same model with discount set to 0.
- **Promotion pays off** when actual uplift > break-even uplift = discount ÷ (margin − discount).
- **Forecast selection:** each model forecasts the last 13 known weeks without seeing them, assuming no planned promotions. The lowest MAPE is marked ★.
- **Known limitations:** product 5 changes behaviour in Dec 2021 (higher price, higher volume), so all models are weak there. Only two holiday seasons are available, so holiday effects are indicative.

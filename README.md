# vibecheck

Judge a Spotify playlist: **mood**, **estimated age**, **listening personality**, **lyrical themes**, a
Pudding-style **roast**, and, with your data export, **when and how you listen** and how your mood has
moved over time. Plus two small ML models you can train and evaluate.

```
## Demo: sad hours: The 3 A.M. Drive Specialist
Verdict: Leans Punk/Emo, centred on 2019, with an angsty streak.
Mood           Angsty (valence 0.40, arousal 0.54)
Estimated age  ~27 (80% interval 24-30)
The roast      21 sad bangers detected. Crying, but make it danceable.
```

## Quick start (Windows PowerShell)

```powershell
cd vibecheck
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[notebook,dev]"         # add ,nlp for the transformer lyric model (pulls in torch)
python -m vibecheck --demo               # offline demo on synthetic data
jupyter lab notebooks/vibecheck.ipynb    # the walkthrough; DEMO = True by default
```

### Connect Spotify

1. Go to <https://developer.spotify.com/dashboard>, create an app, tick **Web API**.
2. Add the redirect URI **`http://127.0.0.1:8888/callback`** (Spotify no longer accepts `localhost`).
3. Copy `.env.example` to `.env` and paste the **Client ID**. No client secret is needed (PKCE flow).
4. Set `DEMO = False` in the notebook, or run `python -m vibecheck "https://open.spotify.com/playlist/..." --out report`.

The first call opens a browser to log in; tokens are cached in `~/.vibecheck/`.

### What Spotify allows in 2026 (and how vibecheck copes)

| Restriction | Since | Workaround here |
|---|---|---|
| No audio features / audio analysis / recommendations for new apps | Nov 2024 | Audio features from [ReccoBeats](https://reccobeats.com) (free, takes Spotify IDs; their own estimates) |
| Development Mode needs Premium, max 5 users per app | Feb/Mar 2026 | Fine for personal use and a few friends |
| Playlist tracks only for playlists you **own or collaborate on** | Feb/Mar 2026 | Copy the playlist into one of yours, or export a CSV (e.g. Chosic's analyzer) and use `load_tracks_csv()` |
| Batch `GET /tracks`, `/artists` removed; `popularity` removed | Feb 2026 | One cached request per artist/track; no popularity-based features |
| Playlist `tracks`/`track` renamed to `items`/`item` | Feb 2026 | Both shapes accepted |

Sources: [Feb 2026 migration guide](https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide),
[Spotify's announcement](https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security).

### Optional extras

- **Last.fm** (`LASTFM_API_KEY` in `.env`): crowd mood tags as an extra mood signal and a genre fallback.
- **Transformer lyric model**: `pip install -e ".[nlp]"` uses `j-hartmann/emotion-english-distilroberta-base`.
  Without it, an offline word-list + VADER model is used.
- **Listening history**: request *Extended streaming history* at spotify.com/account/privacy, then
  `python -m vibecheck --history my_spotify_data.zip --tz America/New_York --out report`.

## What it computes

| Output | How |
|---|---|
| Mood (circumplex) | Per track: audio valence/energy/tempo (ReccoBeats) blended with lyric emotion and tags; playlist mean with a bootstrap 90% interval |
| Sad bangers | Sad-reading lyrics on danceable / energetic audio |
| Estimated age | Bayesian "reminiscence bump" model on album release years (earliest of album date and ISRC year), mixture of formative-years / current / background music, tempered likelihood; calibratable with `AgeModel.fit()` |
| Personality | Genres to the MUSIC model (Mellow, Unpretentious, Sophisticated, Intense, Contemporary), a persona, loose Big Five hints from published (small) correlations |
| Lyrics | Ekman emotions + neutral, themes (heartbreak, party & flex, nostalgia...), repetition, first-person rate |
| Roast | ~35 rules; the most extreme stats win |
| Craft score | Chosic-style 0-5: artist variety, genre focus, mood coherence, length |
| Listening behaviour | Listening clock, night share, skips, same-day repeats, discovery rate, year-by-year table, binge days |
| Emotional tone | Descriptive signals plus a weekly mood timeline vs **your own** baseline |
| ML | Weak-supervision mood classifier from playlist titles (GroupKFold by playlist, baseline comparison, genre-shortcut ablation); age-model calibration |

### About the "mental health" part

Music is a weak and noisy signal of how someone is doing: sad music is common and often helps people feel
better. vibecheck therefore describes the **emotional tone of the music**, compares listening only against the
person's own past, and never labels the listener. Thresholds are readability heuristics, not validated cut-offs.
If you want to study this properly, collect consented self-reported mood alongside listening and evaluate
against that.

## Layout

```
vibecheck/
  sources/   links, Spotify PKCE auth + client, data-export and CSV loaders
  enrich/    ReccoBeats, LRCLIB, Last.fm, genres -> macro/MUSIC, release years
  nlp/       lyric emotion models, lexicons, themes
  analysis/  features, mood, age, personality, behaviour, wellbeing, roast
  ml/        weak-label mood classifier, MPD loader
  report.py  markdown + charts      pipeline.py  Vibecheck / Profile      demo.py  synthetic data
notebooks/vibecheck.ipynb           tests/  (pytest, network mocked)
```

Run the tests with `pytest`. Network code is tested against canned responses; the live APIs
(Spotify, ReccoBeats, LRCLIB, Last.fm) still need a first real run on your machine.

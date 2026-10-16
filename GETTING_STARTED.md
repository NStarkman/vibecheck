# Getting started with vibecheck

A first-run guide for Windows (PowerShell). It takes about 20 minutes, most of it spent in Spotify's developer
dashboard. Commands for macOS/Linux are noted where they differ.

**Checklist**

- [ ] Python 3.10 or newer
- [ ] A Spotify account with an active **Premium** subscription (Individual is the safe choice)
- [ ] The `vibecheck` folder unzipped somewhere permanent
- [ ] Virtual environment created and packages installed
- [ ] Offline demo runs
- [ ] Spotify developer app created, redirect URI added, Client ID copied
- [ ] `.env` file filled in
- [ ] First real login done
- [ ] *(Optional)* Last.fm key, transformer lyric model, listening-history export

---

## 1. Check Python

```powershell
py --version
```

You want **3.10 or newer**. If `py` isn't recognised, install Python from <https://www.python.org/downloads/> and
tick **"Add python.exe to PATH"** in the installer.

> macOS/Linux: use `python3 --version`.

## 2. Unzip and open the folder

Unzip `vibecheck.zip` to a folder that won't get cleaned up, for example
`C:\Users\<you>\Projects\vibecheck`. Open PowerShell there:

```powershell
cd C:\Users\<you>\Projects\vibecheck
dir   # you should see README.md, pyproject.toml, vibecheck\, notebooks\, tests\
```

Every command below is run from this folder (the **project root**) unless it says otherwise.

## 3. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. Activate it again each time you open a new terminal.

If PowerShell says *"running scripts is disabled on this system"*, allow local scripts for your user once,
then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

> Command Prompt instead of PowerShell: `.venv\Scripts\activate.bat`.
> macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## 4. Install

```powershell
python -m pip install --upgrade pip
pip install -e ".[notebook,dev]"
```

`-e` installs in editable mode, so changes you make to the code take effect without reinstalling.
`notebook` adds JupyterLab; `dev` adds pytest.

**Optional: the transformer lyric-emotion model.** Without it, vibecheck uses a smaller offline word-list model,
which is fine for a first run. To add it later, install the CPU build of PyTorch first (much smaller than the
default download), then the extra:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[nlp]"
```

The first lyric analysis then downloads the model weights (a few hundred MB) from Hugging Face once.

## 5. Check everything works offline

```powershell
pytest
python -m vibecheck --demo --out demo_report
```

Expected:

- `pytest` ends with something like **61 passed**.
- The demo prints a report (persona, verdict, mood, estimated age, roast, signals) and writes eight PNG charts plus
  `report.md` into `demo_report\`. Open `demo_report\card.png` to see the summary card.

Everything in the demo is synthetic: made-up artists, made-up lyrics, a fake listening history. If this step fails,
fix it before involving Spotify (see Troubleshooting).

## 6. Create your Spotify developer app

1. Go to <https://developer.spotify.com/dashboard> and **log in with the Spotify account that has Premium**.
   Whoever creates the app is its owner, and the app only works while the owner's Premium is active.
2. Accept the Developer Terms if asked. Spotify may ask you to verify your email first.
3. Click **Create app** and fill in:

   | Field | What to enter |
   |---|---|
   | App name | anything, e.g. `vibecheck` |
   | App description | anything, e.g. `Personal playlist analyzer` |
   | Website | leave blank |
   | Redirect URIs | `http://127.0.0.1:8765/callback`, then click **Add** |
   | APIs used | tick **Web API** |

   The redirect URI must match **exactly**: `http`, the IP `127.0.0.1` (Spotify rejects `localhost`), port `8765`,
   path `/callback`, no trailing slash.
4. Agree to the terms and **Save**.
5. Open the app, go to **Settings**, and copy the **Client ID**. You do **not** need the Client Secret; vibecheck
   uses the PKCE login flow, which doesn't use one. Never paste the secret anywhere in this project.

**Adding friends (optional, up to 5):** in the app's Settings, open **User Management** and add each person's name
and the email address of *their* Spotify account. They don't need Premium. Anyone not on the list can see the login
page, but their requests will fail with 403 errors.

## 7. Fill in `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Set your Client ID and leave the rest as is for now:

```ini
SPOTIFY_CLIENT_ID=paste_your_client_id_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8765/callback
LASTFM_API_KEY=
VIBECHECK_EMOTION_BACKEND=auto
```

Keep `.env` in the project root. vibecheck looks for it in the current folder and up to three parent folders, so
the notebook (which runs from `notebooks\`) finds it too. `.env` is in `.gitignore`; keep it out of any repo.

## 8. First real run and login

Start with your own top tracks. It's the Pudding-style "judge me" input and needs no playlist link:

```powershell
python -m vibecheck top:long_term --tz America/Toronto --out my_report
```

(Use your own time zone name, e.g. `America/New_York`.)

What happens:

1. A browser tab opens on Spotify's consent page, listing what vibecheck may read: your playlists, top items,
   recently played and saved tracks. It's read-only; vibecheck never changes your library.
2. Click **Agree**. The browser is sent to `http://127.0.0.1:8765/callback` and shows
   *"vibecheck: you're logged in"*. You can close that tab.
3. Back in the terminal, vibecheck fetches artist genres (one request per artist, so the first run on a big list
   takes a minute), audio features from ReccoBeats, and lyrics from LRCLIB, then prints the report and saves charts
   to `my_report\`.

The login token is saved to `%USERPROFILE%\.vibecheck\spotify_token.json` and refreshed automatically, so later runs
don't open the browser. If Windows Firewall asks about Python, **Allow on private networks** is enough; the login
listener only accepts connections from your own computer.

Other inputs to try:

```powershell
python -m vibecheck top:short_term                # last ~4 weeks
python -m vibecheck liked --out liked_report       # Liked Songs (first 1000)
python -m vibecheck recent                         # last 50 plays
python -m vibecheck "https://open.spotify.com/playlist/XXXXXXXX" --out playlist_report
```

To get a playlist link in Spotify: open the playlist, then **⋯ → Share → Copy link to playlist**.
It has to be a playlist **you own or collaborate on**; that's a Spotify rule for Development Mode apps. For anyone
else's playlist, either copy its tracks into a new playlist of yours (desktop: open it, `Ctrl+A`, right-click,
**Add to playlist → New playlist**) or export a CSV with a tool that can read it, then:

```powershell
python -m vibecheck path\to\export.csv --out csv_report
```

## 9. Use the notebook

```powershell
jupyter lab
```

Open `notebooks/vibecheck.ipynb`. In the first code cell:

```python
DEMO = False
PLAYLIST = "https://open.spotify.com/playlist/XXXXXXXX"   # or "top:long_term", "liked", "recent"
TZ = "America/Toronto"
```

Then **Run → Run All Cells**. Login works the same as on the command line. If a browser tab doesn't open,
the cell prints the login URL and, failing that, asks you to paste the address your browser ends up on.

The notebook walks through each part (mood, age, personality, lyrics, emotional tone, listening history, and the
two ML models) with explanations.

> Using VS Code instead: open the folder, open the notebook, and pick the `.venv` interpreter as the kernel.

## 10. Optional extras

### Last.fm tags

Adds crowd mood tags ("sad", "chill", "upbeat") as an extra mood signal and fills in genres for artists Spotify
has none for.

1. Create an API account at <https://www.last.fm/api/account/create> (a normal Last.fm login is enough).
2. Copy the **API key** (not the shared secret) into `.env` as `LASTFM_API_KEY=...`.

### Your full listening history

This unlocks the listening clock, skip and repeat rates, year-by-year stats and the weekly mood timeline.

1. Go to <https://www.spotify.com/account/privacy/>, tick **Extended streaming history**, and request it.
2. Confirm the request from the email Spotify sends.
3. Wait. Spotify says it can take up to 30 days; it's often quicker. You'll get an email with a download link.
4. Put the `my_spotify_data.zip` file in the project root (no need to unzip it) and run:

```powershell
python -m vibecheck --history my_spotify_data.zip --tz America/Toronto --out history_report
```

This analyzes your 300 most-played tracks. In the notebook, set the path in section 9 and run that section.

## 11. Where things are stored

| What | Where |
|---|---|
| Spotify login token | `%USERPROFILE%\.vibecheck\spotify_token.json` |
| API cache (genres, audio features, lyrics) | `%USERPROFILE%\.vibecheck\cache.sqlite` |
| Reports and charts | whatever folder you pass to `--out` |

Everything stays on your computer. Lyrics are cached locally for analysis and never displayed.

- **Log out / switch accounts:** delete `spotify_token.json`. If the browser keeps logging into the wrong account,
  log out at spotify.com first.
- **Clear the cache:** delete `cache.sqlite`.
- **Revoke vibecheck's access entirely:** <https://www.spotify.com/account/apps/>.

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Browser shows **INVALID_CLIENT: Invalid redirect URI** | The URI in the dashboard doesn't exactly match `.env`. Check `http` (not https), `127.0.0.1` (not localhost), port `8765`, `/callback`, no trailing slash. Save in the dashboard after editing. |
| **INVALID_CLIENT: Invalid client** | Client ID typo in `.env`, or `.env` isn't in the project root. |
| `SPOTIFY_CLIENT_ID is not set` | `.env` missing, misnamed (`.env.txt` is a common Notepad trap; turn on file extensions in Explorer), or the line is blank. |
| **HTTP 403** on every request | The app owner's Premium lapsed; or the logged-in account isn't the owner and isn't on the User Management list; or you're on a Duo/Family/Student plan that Spotify isn't treating as Premium for developer purposes. If it happens on your own account, add yourself under User Management too. |
| **PlaylistNotAccessible** | The playlist isn't yours. Copy it into one of your own playlists, or use a CSV export (step 8). |
| Login opens but the terminal never continues | Something else is using port 8765, or the redirect was blocked. Paste the full URL from the browser's address bar when prompted, or pick another port in both `.env` and the dashboard. |
| Runs slowly the first time | One Spotify request per artist and per-track lyric lookups, with automatic backoff on rate limits. Re-runs come from the cache and are fast. |
| *"Audio features were found for only X% of tracks"* | ReccoBeats doesn't have every track. Mood then leans on lyrics and tags. |
| *"Lyrics were found for only X%"* | LRCLIB is missing those songs, or they're instrumental. Non-English lyrics are skipped by the emotion model. |
| *"transformer emotion model unavailable"* | The `nlp` extra isn't installed, or the model couldn't download. The offline model is used instead; see step 4. |
| `ModuleNotFoundError: vibecheck` in Jupyter | The notebook kernel isn't your `.venv`. Run `python -m ipykernel install --user --name vibecheck` with the venv active, then choose the **vibecheck** kernel. |
| `Activate.ps1 cannot be loaded` | See the execution-policy fix in step 3. |
| Listening-time stats look shifted by hours | Pass your time zone with `--tz` (or set `TZ` in the notebook). |

## 13. Reading the results sensibly

- **Mood, age and personality are estimates.** The age shows an 80% interval and the mood a 90% interval; on small
  playlists (under ~15 tracks) treat both as rough guesses.
- **The emotional-tone section describes the music, not you.** Sad playlists are common and often how people feel
  better. The only comparison that means much is against your *own* past listening (the history timeline).
- **The roast is a joke generator** keyed to your most extreme stats, in the spirit of the Pudding project.

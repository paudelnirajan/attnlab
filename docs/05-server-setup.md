# Setting up the server Mac

Everything you do by hand to get from a MacBook Pro to `https://yourdomain.com`, in order. The
scripts in `deploy/` do the rest. Why it's built this way: [`04-self-hosting.md`](04-self-hosting.md).

**You need:** the server Mac with its admin password, your domain (at any registrar), a free
[Cloudflare](https://dash.cloudflare.com/sign-up) account, and the repo pushed to GitHub. Budget
about 2 hours the first time.

Commands marked **admin** are run once from an administrator account. Everything else runs as the
standard user, called `attnlab` below.

---

## 1. The standard user (admin)

System Settings → Users & Groups → Add User: **Standard**, name `attnlab`.

Why a standard account: the server can only touch that account's files, it can't install system
software or change settings, and nothing personal lives in it. If the site is ever compromised,
that's all an attacker gets.

## 2. Power, login and updates (admin)

Install git (Apple's command line tools), then keep the machine awake and have it restart itself:

```bash
xcode-select --install
sudo pmset -a sleep 0 disksleep 0 displaysleep 5 powernap 0
sudo pmset -a autorestart 1     # power back on after a power cut
sudo pmset -a womp 1            # wake on network access
pmset -g                        # check
```

`deploy/install.sh` also runs `caffeinate` as a service, which covers sleep without admin, but only
`pmset` can turn the Mac back on after a power cut.

**Automatic login.** The services are LaunchAgents of the `attnlab` user, so they start when that
user logs in. For the site to come back by itself after a reboot, set System Settings → Users &
Groups → *Automatically log in as* → `attnlab`.

> **FileVault decides this.** With FileVault on, macOS can't log in automatically: after any
> reboot the Mac waits at a password screen, and the site stays down until someone types it. Your
> options:
> - **FileVault off, auto-login on.** The site survives power cuts. Anyone who takes the Mac gets
>   the disk. Reasonable for a dedicated machine at home with nothing personal on it.
> - **FileVault on.** Safer disk, but you unlock it by hand after every reboot. For planned reboots,
>   `sudo fdesetup authrestart` reboots once without asking.
>
> Pick one knowingly. Most people running a home server choose the first.

**Lid.** A MacBook sleeps when the lid closes unless it has external power *and* a display. Leave it
open (the display turns off after 5 minutes), or run it closed with an HDMI dummy plug.

**Battery.** Plugged in 24/7, a battery ages fastest at 100%. Turn on System Settings → Battery →
*Optimized Battery Charging*, or use a charge limit at 80% if your macOS version has one (the
third-party AlDente app does it otherwise).

**Firewall.** System Settings → Network → Firewall: on. The server listens on 127.0.0.1 only and
the tunnel connects outward, so nothing needs to come in.

**Updates.** Leave *Install Security Responses and system files* on. Set macOS updates to download
but not install automatically, and install them when you can watch the Mac come back.

**SSH, for deploying from your dev machine** (optional but convenient). System Settings → General
→ Sharing → Remote Login: on, allowed for `attnlab` only. Then use keys, not passwords: from your
dev Mac, `ssh-copy-id attnlab@<server>.local`, and on the server:

```bash
sudo tee /etc/ssh/sshd_config.d/100-keys-only.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
```

Don't forward port 22 on your router. This is for your home network. From elsewhere, use
[Tailscale](https://tailscale.com) (free for personal use).

## 3. Tools, per user (as `attnlab`)

Log in as `attnlab`. None of these need admin:

```bash
# uv: Python and its packages (installs to ~/.local/bin)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Node, for building the frontend, through fnm
curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$HOME/.local/share/fnm" --skip-shell
~/.local/share/fnm/fnm install 22
~/.local/share/fnm/fnm default 22     # what the deploy scripts will use

# cloudflared, the tunnel client
mkdir -p ~/attnlab/bin
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz \
  | tar -xz -C ~/attnlab/bin
~/attnlab/bin/cloudflared --version
```

Open a new terminal, then check: `uv --version`, `git --version`.

**GitHub access.** A public repo needs nothing. For a private one, give the server a read-only
deploy key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/attnlab_deploy -N ""
cat ~/.ssh/attnlab_deploy.pub     # GitHub → repo → Settings → Deploy keys → Add (read-only)
printf 'Host github.com\n  IdentityFile ~/.ssh/attnlab_deploy\n' >> ~/.ssh/config
```

## 4. First release and install

On your **dev machine**, cut the first release (it runs the tests before tagging):

```bash
make release V=0.2.0
git push origin main v0.2.0
```

On the **server**, as `attnlab`:

```bash
git clone git@github.com:paudelnirajan/attnlab.git ~/attnlab/repo   # or the https URL
~/attnlab/repo/deploy/install.sh git@github.com:paudelnirajan/attnlab.git v0.2.0
```

This creates `~/attnlab/`, installs the services (API, watchdog, stay-awake), and deploys v0.2.0:
dependencies, frontend build, model downloads (~4 GB, a few minutes), tests, start. Then:

```bash
~/attnlab/repo/deploy/status.sh
open http://127.0.0.1:8000          # the site, locally
```

`status.sh` should show the API running, `"ready": true`, and three models resident.

```
~/attnlab/
├── repo/               git clone; deploy scripts run from here
├── releases/
│   └── v0.2.0-a1b2c3d/ one directory per deployed release (newest 3 kept)
├── current -> releases/v0.2.0-a1b2c3d
├── previous            the release before, for rollback.sh
├── bin/cloudflared
└── shared/
    ├── local.env       your overrides and secrets (not in git)
    └── logs/           app.log, access.log, deploy.log, watchdog.log, *.out/err.log
```

## 5. The domain and the tunnel

**Move DNS to Cloudflare.** Cloudflare dashboard → Add a domain → Free plan. It shows two
nameservers. Set those at your registrar, replacing theirs. It can take up to a day to take effect.
Cloudflare emails you when it does.

**Create the tunnel**, on the server as `attnlab`:

```bash
cd ~/attnlab/bin
./cloudflared tunnel login                   # opens a browser: pick your domain
./cloudflared tunnel create attnlab          # prints the tunnel ID; writes ~/.cloudflared/<ID>.json
./cloudflared tunnel route dns attnlab yourdomain.com
./cloudflared tunnel route dns attnlab www.yourdomain.com
```

Write `~/.cloudflared/config.yml` from `~/attnlab/repo/deploy/cloudflared-config.example.yml`,
putting in the tunnel ID and your domain. Then check it and try it in the foreground:

```bash
./cloudflared tunnel ingress validate
./cloudflared tunnel run attnlab             # https://yourdomain.com should work now; Ctrl+C
~/attnlab/repo/deploy/install.sh             # re-run: installs the tunnel as a service too
```

**Cloudflare settings** (dashboard, your domain):

| Where | Setting | Why |
|---|---|---|
| SSL/TLS → Edge Certificates | Always Use HTTPS: on | no plain-HTTP visitors |
| Security → Bots | Bot Fight Mode: on | turns away the obvious scrapers. If it ever blocks real visitors' API calls, turn it off |
| Security → WAF → Rate limiting rules | one rule: URI path starts with `/api/`, 100 requests per 10 s per IP → Block for 10 s | stops floods before they reach your Mac; the app's own limit handles the rest |
| Caching | leave the defaults. **Don't** add "Cache Everything" | API responses must never be cached |
| Rules → Redirect Rules | `www.yourdomain.com/*` → `https://yourdomain.com/${1}` (301) | one canonical address |

## 6. Monitoring

Two free checks, and you'll hear about a problem before a visitor tells you:

1. **[Healthchecks.io](https://healthchecks.io)**: create a check with period 1 minute, grace 5
   minutes. Put its ping URL in `~/attnlab/shared/local.env`:
   `HEALTHCHECK_URL=https://hc-ping.com/…`. The watchdog pings it every minute, or pings `/fail`
   with the reason (disk low, swap high, API down). If the Mac loses power or internet, the pings
   stop and you get an email.
2. **[UptimeRobot](https://uptimerobot.com)**: an HTTP monitor on
   `https://yourdomain.com/api/health` every 5 minutes. This one checks the whole path from outside:
   DNS, Cloudflare, the tunnel, the app.

## 7. Before you announce it

From your dev machine, simulate visitors over the real path, while watching `status.sh` or Activity
Monitor on the server:

```bash
uv run python scripts/loadtest.py https://yourdomain.com --users 10 --seconds 60
uv run python scripts/loadtest.py https://yourdomain.com --users 5 --model qwen3-0.6b
```

**Good:** memory pressure green, swap near 0, the footprint under 10.5 GB. Some `503`s are expected
at 10 users, because that's the queue cap working. For reference, on an M4 Pro locally, 10 visitors
on gpt2-small got 9 runs/s at a median of 0.33 s, and 10% were turned away as busy. Expect the M1 Pro
to manage somewhat less. Note your numbers in `03-decisions.md` under D20.

## 8. Day to day

| Task | Command (as `attnlab`) |
|---|---|
| Is everything OK? | `~/attnlab/repo/deploy/status.sh` |
| Deploy a release | `~/attnlab/repo/deploy/deploy.sh v0.3.0` (see [`06-releasing.md`](06-releasing.md)) |
| Go back one release | `~/attnlab/repo/deploy/rollback.sh` |
| Restart the API | `launchctl kickstart -k gui/$(id -u)/com.attnlab.api` |
| Stop the site | `launchctl bootout gui/$(id -u)/com.attnlab.api` (the tunnel then serves Cloudflare's error page) |
| Start it again | `~/attnlab/repo/deploy/install.sh` |
| Follow the logs | `tail -f ~/attnlab/shared/logs/app.log` (also `access.log`, `deploy.log`, `watchdog.log`) |
| Change a limit | edit `~/attnlab/shared/local.env`, then restart the API |

**Disk (256 GB).** What grows, and what to do about it:

| What | Size | Housekeeping |
|---|---|---|
| Model files, `~/.cache/huggingface` | ~4 GB | grows only when a release adds a model |
| Releases | ~50 MB each plus its `.venv` (uv shares files on APFS, so mostly free) | the newest 3 kept automatically |
| uv and npm caches | 1–3 GB | `uv cache prune` · `npm cache clean --force` |
| Logs | ≤ ~1 GB | rotated automatically |

Keep 30 GB+ free: macOS needs room for swap and updates. The watchdog reports under 15 GB, and
`fetch_models.py` refuses to download under 20 GB.

## 9. When something's wrong

| Symptom | Look at | Likely cause |
|---|---|---|
| Cloudflare error 1033 | `tunnel.err.log` | cloudflared isn't running or can't connect: `launchctl kickstart -k gui/$(id -u)/com.attnlab.tunnel` |
| Cloudflare 502 | `app.log`, `api.err.log` | the API is restarting (normal for ~10 s after a deploy) or crashing at startup |
| Site down after a reboot | the server's screen | FileVault is waiting for its password, or auto-login is off (§ 2) |
| `busy` errors for visitors | `status.sh` counters | more traffic than one Mac serves; see 04 § 5 |
| Deploy stopped at "tests failed" | `deploy.log` | nothing changed: the old release is still serving. Fix, tag, deploy again |
| The Mac is slow | `status.sh` swap line | something else is using memory; lower `MI_RAM_BUDGET_GB` in local.env |

# Kaggriculture Strategy Lab

A local, interactive Kaggriculture simulator for strategy development. You play Player 0 by hand
through a UI; Player 1 is a real agent; every action runs through the **installed Kaggriculture
engine**, which is the only source of truth.

> **Architecture rule:** `kaggle_environments` engine → Python backend → JSON API → React UI.
> The frontend never implements a game rule. Any number it shows either came from the engine or is
> explicitly labelled a visualization-only derived metric.

Engine behaviour is documented in **[ENGINE_NOTES.md](ENGINE_NOTES.md)** — read that before
changing anything that touches rules.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Engine mapping, `ENGINE_NOTES.md`, session adapter, manual-turn verification | **done** |
| 1b | **Hybrid mode**: autopilot + strategic interventions, opponent inference | **done** |
| 2 | React UI: LIVE + REPLAY modes, farm/town/market, time controls, event log, inspector | **done, 62/62 checks pass** |
| 3 | Experiment runner, sweep tooling, saved intervention sets | not started |

## Two modes, never mixed

| | LIVE simulation | REPLAY analysis |
|---|---|---|
| Engine | **running** — every action goes through `kaggle_environments` | **not running** — historical data only |
| Player 0 | you, manually, and/or a local autopilot agent | whatever the recording contains |
| Player 1 | a local agent: `random`, `starter`, `pass`, or any `.py` in `local_agents/` | the recorded opponent |
| Real ladder opponent | **cannot be simulated** — no source code exists | studied here, from the replay |

The app shows a coloured mode banner at all times so the two are never confused.

> There is no `opponent_main.py` dependency. The real competition opponent's agent code is not
> available and the lab never pretends otherwise. `opponent/opponent_main.py`, if present, is a
> *behavioural reconstruction from public traces* — a legitimate local sparring partner, listed as
> a local file, not as the real opponent.

## Setup

```bash
.venv/bin/pip install -r requirements.txt
npm install --prefix frontend
```

## Run

```bash
.venv/bin/uvicorn main:app --reload --port 8000 --app-dir backend
```

```bash
npm run dev --prefix frontend
```

Then open http://localhost:5173 — Vite proxies `/api` to the backend.

## Verify the engine integration

```bash
.venv/bin/python backend/test_engine.py
```

33 checks covering session creation, the competition-accurate view, manual-turn money arithmetic,
the engine recorder, undo, end-of-day timing, land/demand helpers, replay export and opponent
switching.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/rules` | every constant, cost and price curve, read live from the engine |
| `GET /api/opponents` | built-ins plus any `.py` in `opponent/` |
| `POST /api/game` | new episode — `{seed, opponent, episode_steps}` |
| `GET /api/game/{id}` | full state; `?competition_accurate=false` reveals opponent private state |
| `GET /api/game/{id}/observation/{player}` | raw observation exactly as that agent receives it |
| `POST /api/game/{id}/turn` | execute one turn: your action + opponent agent + engine step |
| `POST /api/game/{id}/undo` | step back one turn |
| `GET /api/game/{id}/history` | per-turn engine records |
| `GET /api/game/{id}/replay` | `env.toJSON()`, loadable by `analyze_replay.py` |
| `GET /api/quote` | cost calculator; walks the engine price function unit by unit |
| `GET /api/conditions` | intervention catalogue + presets, each tagged by information source |
| `GET·PUT /api/game/{id}/interventions` | read / replace the intervention rules |
| `GET /api/game/{id}/autopilot-preview` | what the autopilot would play, without executing |
| `POST /api/game/{id}/run` | **hybrid mode** — run until an intervention fires |

## Hybrid mode (the primary mode)

`autopilot` drives Player 0; `opponent` drives Player 1; the run loop advances until an
intervention fires. You then take control for as many turns as you like and resume.

```bash
curl -X POST localhost:8000/api/game -H 'content-type: application/json' -d '{
  "seed": 715343816,
  "opponent": "main.py",
  "autopilot": "main.py",
  "presets": ["shop_draw","melon_window","glut_guard","cash_health","opponent_watch"]
}'
curl -X POST localhost:8000/api/game/$SID/run -H 'content-type: application/json' -d '{"max_steps":720}'
```

### Information honesty

Every condition declares what it relies on, so you always know whether a real agent could
have detected it:

| Source | Meaning |
|---|---|
| `PUBLIC` | in the shared observation — market, town, clock, **and both farms** (opponent money, tiles, positions, quadrants and hire count are all public) |
| `OWN` | your own private state: shed, seeds, carried inventory |
| `DERIVED` | arithmetic over consecutive public observations (price deltas, inventory moves) |
| `INFERRED` | reconstructed opponent behaviour, exact only under stated preconditions |
| `LAB` | visible only because the backend runs the engine — **never used for a rule** |

**Opponent sales are inferred the way an agent would have to.** For a product that cannot be
bought back, inventory moves only by sales and by deterministic town demand, so
`opponent_sold = inventory_delta + town_demand - my_sales`. Validated against ground truth over
400 turns: MELON 30/30, MILK 42/42, WOOL 52/52 exact. It reports **unknowable** when the price is
at the $1 floor (sales stop moving inventory) and for WHEAT/FERTILIZER (buyable, so only net flow
is knowable). The lab also knows the true figure and shows both, so you can see the gap between
what happened and what your agent could have deduced.

### Trigger modes

A level condition like `cash_below` stays true for many consecutive steps, so rules are
**edge-triggered by default** — they fire on the false→true transition only.

| Mode | Behaviour |
|---|---|
| `edge` *(default)* | fire on the transition into the condition |
| `always` | fire every step it holds, honouring an optional `cooldown` (steps) |
| `once` | fire at most once per session |

### Opponents

Drop any file exposing `def agent(obs, config=None)` into `opponent/`, then pass its filename:

```bash
curl -X POST localhost:8000/api/game -H 'content-type: application/json' \
  -d '{"seed":715343816,"opponent":"opponent_main.py"}'
```

Built-ins `pass`, `random`, `starter` work by name. Switching opponents needs no code changes.

## Layout

```
backend/
  main.py              FastAPI app
  engine/rules.py      constants + derived curves, read live from the engine
  engine/recorder.py   captures per-unit trades, hires, town consumption, end-of-day
  engine/session.py    owns one episode; manual P0 + agent P1 + undo
  agents/loader.py     opponent resolution
  test_engine.py       Phase 1 verification
opponent/              drop opponent agents here
replays/  experiments/
analyze_replay.py      normalizes any replay (this repo's or Kaggle's) into analysis.json
ENGINE_NOTES.md        authoritative engine behaviour
```

## Related tooling

`analyze_replay.py` turns any Kaggriculture replay — including one exported from
`/api/game/{id}/replay` — into a normalized `analysis.json` with per-unit trade prices recovered by
re-executing the episode through the engine.


## Replay analysis API

| Endpoint | Purpose |
|---|---|
| `GET /api/replays` | library — rewards, winner, margin, verification, import date |
| `POST /api/replays/import` | normalize a replay by path on this machine |
| `POST /api/replays/upload` | normalize an uploaded file |
| `GET /api/replays/{id}/step/{n}` | full normalized step |
| `GET /api/replays/{id}/compare/{n}` | P0-vs-P1 side-by-side at one step |
| `GET /api/replays/{id}/series` | chart-ready parallel arrays |
| `GET /api/replays/{id}/summary` | revenue by product, margin decomposition |
| `GET /api/replays/{id}/events` | jumpable event index for the timeline |
| `DELETE /api/replays/{id}` | remove from the library |

Importing re-executes every recorded action through the real engine to recover the per-unit trade
prices a replay does not store, then verifies the result against the stored state. The library shows
a **verified 719/719** badge when reconstruction was exact.

## The research loop

```
BUILD → SIMULATE (live, hybrid) → SUBMIT → DOWNLOAD REPLAY
  ↑                                              ↓
IMPROVE ← FIND DECISION POINTS ← ANALYSE (replay)
```

## Frontend layout

```
frontend/src/
  App.tsx                  mode router + banner
  api/client.ts            typed fetch wrappers (the only place that talks to the backend)
  types/index.ts           shared types
  styles/theme.css         design tokens, dark + light
  components/              FarmGrid · MarketPanel · TownPanel · EventLog · StateInspector · ui
  live/                    LiveView · ActionBuilder · Interventions (modal + builder)
  replay/                  ReplayView (library · timeline · compare · derived facts)
```

No game rule is implemented in React. Costs, prices, curves, shop products and demand all come from
`GET /api/rules`, which reads them straight out of the installed engine.

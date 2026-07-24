// A single, config-driven renderer replaces the former per-URL demos: `render.sh
// --web` generates `./config.json` + `./stream.arrow` from the same Arrow
// producers and scene flags as `--cli`, and `./viewer` fetches and
// replays them (to the canvas or an offscreen texture, per the config). The only
// live URL param is `?fps=N`.
import "./viewer";

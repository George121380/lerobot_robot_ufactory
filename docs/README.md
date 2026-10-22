# GELLO / xArm lab notes

English operating notes for GELLO joint-space collection and policy rollout.

- [Online guide](https://peiqil.com/lerobot_robot_ufactory/)
- Content: `index.html`; styles: `assets/style.css`; controls: `assets/app.js`.
- Robot-code baseline: upstream `5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82`.
- Status: source-checked draft; installation and physical workflows are not yet lab-validated.

## Edit

Keep commands, required settings, key bindings, and actionable troubleshooting. Use short sentences. Avoid promotional headings, repeated warnings, workflow summaries, and explanations that do not change an operator's next step.

Keep existing section IDs or anchor aliases so shared links work. Verify commands against the pinned versions. Before marking the guide lab-tested, record the actual arm, GELLO mapping/reference posture, camera setup, software versions, and trial results.

## Preview and publish

```bash
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`; stop with Ctrl-C. Push changes to `main` to update GitHub Pages from `/docs`. No build step. Bump the stylesheet query version after CSS changes.

Check links, code copying, search, and desktop/mobile layout before pushing. Do not run robot commands to test documentation changes. Machine-specific settings belong in ignored `config/local/`.

Workflow reference: [TRI Raiden](https://tri-ml.github.io/raiden/). Commands and behavior here are specific to this repository.

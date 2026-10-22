# xArm Lab Guide

English documentation draft for lab members using UFACTORY xArm with LeRobot.

- **Online guide:** https://george121380.github.io/lerobot_robot_ufactory/
- **Content:** `index.html`
- **Styles:** `assets/style.css`
- **Navigation, search, theme and copy controls:** `assets/app.js`
- **Reviewed robot-code baseline:** upstream `5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82`

The workflow structure is inspired by [TRI's Raiden documentation](https://tri-ml.github.io/raiden/). Commands, device behavior and limitations are specific to this UFACTORY repository. No TRI images, branding or prose are copied.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. Stop the server with Ctrl-C. No build tool or dependency installation is required.

## Publish

GitHub Pages is configured to deploy from the fork's `main` branch, `/docs` folder. Changes pushed to that directory trigger the Pages build. `.nojekyll` allows the static files to be served directly.

If reproducing this setup in a different fork, set **Settings → Pages → Build and deployment → Deploy from a branch → main → /docs**. Replace the fork and Pages links in `index.html`, `assets/app.js`, this file and the root README.

## Edit the guide

Keep section IDs stable so shared links continue to work. Each major section uses a `doc-section` class and a `data-title` for local search. Add a matching sidebar link when adding a section. Keep HTML semantic and place shell examples inside `pre > code`; the copy button is added automatically.

Source links use the reviewed upstream commit. After updating the robot code, review the affected descriptions and update the snapshot links and status. Do not mark a station as validated based only on documentation or syntax checks.

Before calling this a station-validated guide, add:

1. The tested hardware, firmware, OS, CUDA, Python and peripheral SDK versions.
2. The station's verified reference-pose and coordinate-mapping procedure.
3. Camera-mount references and image preprocessing conventions.
4. Dataset naming, storage and experiment-log conventions.
5. Results of teleoperation, a short recording, dataset review, training and supervised rollout, with dates and code revisions.

Keep credentials and non-public station details out of this public documentation. Machine-specific YAML files belong in the ignored `config/local/` directory or the lab's private configuration store.

## Validation for documentation changes

- Check internal links and local assets.
- Review the page on desktop and mobile widths, in light and dark themes.
- Check search, code copying, keyboard navigation and deep links.
- Verify every command and source-specific claim against the pinned software version.
- Robot commands in the guide are instructions for a configured physical workstation; do not execute them merely to validate a documentation edit.

The first draft was checked against source, and the `continuous_rotvec` issue was reproduced in isolation. End-to-end installation, training and physical robot operation remain unverified.

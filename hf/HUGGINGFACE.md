# Deploy to Hugging Face Spaces (free, no credit card)

A public, always-on HTTPS endpoint for the API — no credit card, no server to
maintain. Ideal as a live ML portfolio demo.

## Steps (web UI, ~5 min)

1. **Create a free account:** <https://huggingface.co/join> (email only, no card).

2. **Create a Space:** <https://huggingface.co/new-space>
   - **Owner:** your username
   - **Space name:** `ppg-heart-rate` (or anything)
   - **License:** MIT
   - **SDK:** **Docker** → *Blank / custom*
   - **Hardware:** **CPU basic** (free)
   - **Visibility:** Public
   - Create.

3. **Upload the files.** Unzip `ppg-hr-hf-space.zip` (project root) on your machine.
   In the new Space → **Files** tab → **Add file → Upload files**, then drag in
   **all** of the unzipped contents, preserving folders:
   ```
   Dockerfile   README.md   requirements.txt   best.pt   src/   api/
   ```
   Drag the `src/` and `api/` **folders** too (HF keeps the structure). Commit.

   > The `README.md` carries the YAML header (`sdk: docker`, `app_port: 7860`) HF
   > needs — keep it at the Space root.

4. **Wait for the build.** HF builds the Docker image automatically (watch the
   **Building** logs; first build is a few minutes). When the badge turns
   **Running**, it's live.

## Your live endpoints

- App page: `https://huggingface.co/spaces/<user>/ppg-heart-rate`
- **Direct API host** (for curl): `https://<user>-ppg-heart-rate.hf.space`

```bash
curl https://<user>-ppg-heart-rate.hf.space/health
curl -X POST https://<user>-ppg-heart-rate.hf.space/predict \
  -H "Content-Type: application/json" -d @api/example_payload.json
# interactive docs: open .../docs in a browser
```

## Notes

- Free Spaces **sleep after ~48 h idle** and wake on the next request (a short
  cold start) — fine for a portfolio demo.
- The model (`best.pt`, ~0.5 MB) is baked into the image; no external storage.
- This is the *same* FastAPI app you run locally — identical behaviour.

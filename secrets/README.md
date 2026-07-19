# Secrets (local only)

Do **not** commit real passwords, JWT secrets, or production API keys.

| File | Purpose |
|------|---------|
| `../local.properties` | Android SDK path + optional `MEMORY_API_BASE_URL` (gitignored) |
| `android.local.properties.example` | Template for Android API URL overrides |
| `../backend/.env.example` | Backend fail-closed env template |
| `../backend/.env` | Your local backend secrets (gitignored) |

## Quick local setup

```bash
# Android emulator → host machine
cp secrets/android.local.properties.example local.properties
# edit sdk.dir if needed; MEMORY_API_BASE_URL defaults to http://10.0.2.2:8000/

# Backend
cp backend/.env.example backend/.env
# set GEO_DATABASE_URL + JWT_SECRET (required outside GEO_TEST_MODE)

# Optional demo user (local only)
# GEO_SEED_DEMO_USER=1 GEO_DEMO_USERNAME=demo GEO_DEMO_PASSWORD=demo-pass-change-me
```

Free offline use does **not** need backend credentials: capture + Train Mode write to Room only while Private Mode is on.

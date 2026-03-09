# Security Policy

## Supported versions

This project is currently maintained on the `main` branch only.

## Reporting a vulnerability

Please do not open a public issue for security-sensitive problems.

When reporting a vulnerability, include:

- a short description of the issue
- impact and affected area
- reproduction steps or proof of concept
- any suggested mitigation

Until a dedicated security contact is published, open a private channel with the maintainer or repository owner if available.

## Secret handling

- Never commit Ark API keys or other credentials.
- Keep `.env`, local databases, and generated reports out of version control.
- Rotate credentials immediately if they are exposed in logs, screenshots, or issue threads.

# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability in argus-ai, please report it responsibly.

**Do not open a public issue.**

Email: security@ambharii.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Scope

argus-ai is a scoring and monitoring library. It does not:
- Store or transmit LLM prompts/responses externally
- Make network calls (unless exporter backends are configured)
- Access filesystem beyond Python imports
- Execute arbitrary code from user input

The safety scorer performs pattern-matching only and does not replace dedicated security tooling.

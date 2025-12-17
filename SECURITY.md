# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
for this repository, or email the maintainer at contact@joelgotsch.com.

Please include a description, reproduction steps, and the affected version. We
aim to acknowledge reports within a few days.

## Scope notes

- `LLMJudge` interpolates task output into the grading prompt. ragpill escapes
  the section-boundary tags to blunt prompt injection, but this is a mitigation,
  not a guarantee — treat judge verdicts on adversarial/user-generated content
  accordingly.
- The report redaction (`redact=True`) matches a best-effort set of secret-like
  key patterns; it is not a security barrier. Review rendered reports before
  pasting them into third-party tools.

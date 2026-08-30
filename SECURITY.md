# Security policy

## Reporting a vulnerability

Report vulnerabilities privately to the repository maintainers. Do not include
live credentials, private conversations, face images, health information, or
messaging databases in an issue, log excerpt, or test fixture.

## Credential exposure

If a credential is committed or shared:

1. Revoke or rotate it immediately at the provider.
2. Remove it from the current tree and all reachable Git history.
3. Review provider audit logs and active sessions.
4. Replace examples with empty placeholders and add an automated regression
   check for the exposed credential format.

History rewriting does not invalidate a secret. Rotation is always required.

## Deployment guidance

- Run KikiFast as a dedicated unprivileged user where hardware access permits.
- Restrict the Web UI, ZMQ endpoints, model servers, and MCP bridges to trusted
  networks.
- Keep writable runtime state on a protected local volume.
- Enable shell, Python execution, messaging, and self-extension tools only for
  trusted users and deployments.

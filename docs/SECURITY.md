# Documentation Security Guide

## Overview

Metal-Q's documentation build system includes comprehensive security features to prevent common vulnerabilities during the documentation generation process.

## Security Features

### 1. Path Validation

All file paths in `mkdocs.yml` are validated to prevent:
- **Path traversal attacks** (e.g., `../../../etc/passwd`)
- **Absolute path access** (e.g., `/etc/shadow`)
- **Home directory expansion** (e.g., `~/.ssh/id_rsa`)
- **Variable expansion** (e.g., `${HOME}/.bashrc`)

### 2. Sandboxed Build Environment

Documentation is built in an isolated environment with:
- **Restricted file access**: Only repository files are accessible
- **Temporary sandbox**: Build happens in isolated temporary directory
- **Clean environment**: Dangerous environment variables are cleared
- **Non-root execution**: Runs with minimal privileges
- **Timeout protection**: Build times out after 5 minutes

### 3. Configuration Validation

The `mkdocs.yml` configuration is validated for:
- **Dangerous plugins**: Only whitelisted plugins allowed
- **Unsafe URLs**: `file://`, `javascript:`, `data:` URLs blocked
- **Malicious markdown extensions**: Extensions that can include arbitrary files blocked
- **Script injection**: HTML script tags detected and blocked

### 4. File Access Monitoring

All file access during build is monitored:
- **Access logging**: Every file read/write is logged
- **Violation detection**: Access outside repository is blocked
- **Audit trail**: Security events logged to `docs_security_audit.log`

## Usage

### Quick Build (Secure)

```bash
# Build documentation securely
./build_docs_secure.sh

# Build and serve locally
./build_docs_secure.sh --serve

# Validate configuration only
./build_docs_secure.sh --validate

# Build with custom output directory
./build_docs_secure.sh --output /tmp/docs

# Export audit log
./build_docs_secure.sh --audit security_audit.log
```

### Python API

```python
from docs.secure_build import SandboxedBuilder
from pathlib import Path

# Build documentation
builder = SandboxedBuilder(Path.cwd())
success = builder.build(output_dir=Path('site'))

if success:
    print("Documentation built successfully")
else:
    print("Build failed - check audit log")
```

### CI/CD Integration

The repository includes a secure GitHub Actions workflow (`.github/workflows/docs-secure.yml`) that:

1. **Validates** configuration for security issues
2. **Builds** in a containerized sandbox with:
   - Read-only filesystem
   - Non-root user
   - Memory limits
   - Network isolation
3. **Scans** output for sensitive data
4. **Deploys** only after all security checks pass

## Security Checks Performed

### Pre-Build

- [x] Configuration file validation
- [x] Path traversal prevention
- [x] URL scheme validation
- [x] Plugin whitelist enforcement
- [x] Markdown extension validation

### During Build

- [x] Sandboxed execution environment
- [x] File access restrictions
- [x] Environment variable sanitization
- [x] Timeout enforcement (5 minutes)
- [x] Memory limits

### Post-Build

- [x] Sensitive data scanning (private keys, passwords)
- [x] Executable file detection
- [x] JavaScript injection detection
- [x] File permission verification

## Configuration Guidelines

### Safe Configuration Example

```yaml
site_name: Metal-Q Documentation
site_url: https://example.com/metalq

# Use HTTPS URLs only
repo_url: https://github.com/user/repo
edit_uri: edit/main/docs/

# Only use trusted themes
theme:
  name: material
  # Don't use custom_dir with external paths

# Whitelist plugins explicitly
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: false  # Don't expose source code

# Safe navigation - relative paths only
nav:
  - Home: index.md
  - Guide: guide/intro.md
  # Don't use: ../../../etc/passwd

# Safe markdown extensions
markdown_extensions:
  - admonition
  - codehilite
  # Don't use: markdown_include (can include arbitrary files)
```

### Dangerous Patterns to Avoid

```yaml
# DON'T: Use absolute paths
nav:
  - Page: /etc/passwd  # BLOCKED

# DON'T: Use path traversal
nav:
  - Page: ../../sensitive/data.md  # BLOCKED

# DON'T: Use file:// URLs
repo_url: file:///local/repo  # BLOCKED

# DON'T: Use dangerous plugins
plugins:
  - exec  # BLOCKED - can execute code

# DON'T: Use dangerous extensions
markdown_extensions:
  - markdown_include  # BLOCKED - can include any file
```

## Testing Security

Run the security test suite:

```bash
# Run all security tests
pytest tests/test_docs_security.py -v

# Test path validation only
pytest tests/test_docs_security.py::TestPathValidator -v

# Test configuration validation
pytest tests/test_docs_security.py::TestMkDocsConfigValidator -v
```

## Audit Logging

Security events are logged to `docs_security_audit.log`:

```
2024-01-15 10:30:45 - SECURITY - Validating MkDocs configuration: mkdocs.yml
2024-01-15 10:30:45 - SECURITY - Configuration validation passed
2024-01-15 10:30:46 - SECURITY - Created sandbox: /tmp/metalq_docs_sandbox_abc123
2024-01-15 10:30:46 - SECURITY - Sandbox prepared successfully
2024-01-15 10:30:55 - SECURITY - Documentation built successfully
2024-01-15 10:30:55 - SECURITY - Sandbox cleaned up
```

## Security Recommendations

1. **Regular Updates**: Keep MkDocs and dependencies updated
2. **Audit Logs**: Review `docs_security_audit.log` regularly
3. **CSP Headers**: Enable Content Security Policy on hosted docs
4. **HTTPS Only**: Always serve documentation over HTTPS
5. **Access Control**: Restrict who can modify `mkdocs.yml`
6. **Dependency Scanning**: Use tools like Dependabot for vulnerability alerts

## Reporting Security Issues

If you discover a security vulnerability in the documentation system:

1. **Do NOT** create a public GitHub issue
2. **Email** security details to the maintainers
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Compliance

This secure documentation system helps meet security requirements for:

- **OWASP Top 10** prevention (A03: Injection, A08: Security Misconfiguration)
- **CWE-22**: Path Traversal
- **CWE-78**: OS Command Injection
- **CWE-94**: Code Injection
- **CWE-200**: Information Exposure

## Further Reading

- [MkDocs Security Best Practices](https://www.mkdocs.org/user-guide/configuration/#security)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices/)
- [Python Security Guidelines](https://python.readthedocs.io/en/latest/library/security_warnings.html)
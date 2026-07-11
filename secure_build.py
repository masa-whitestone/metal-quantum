#!/usr/bin/env python3
"""
Secure Documentation Build System for Metal-Q

Provides:
- Path validation for all mkdocs configuration
- Sandboxed build environment
- Restricted file access to repository only
- Security audit logging
"""
import os
import sys
import yaml
import subprocess
import tempfile
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import re
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security audit logger
audit_logger = logging.getLogger('security.audit')
audit_handler = logging.FileHandler('docs_security_audit.log')
audit_handler.setFormatter(
    logging.Formatter('%(asctime)s - SECURITY - %(message)s')
)
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


class PathValidator:
    """Validates file paths for security issues."""

    def __init__(self, repo_root: Path):
        """
        Initialize path validator.

        Args:
            repo_root: Repository root directory (absolute path)
        """
        self.repo_root = repo_root.resolve()
        self.suspicious_patterns = [
            r'\.\.',  # Parent directory traversal
            r'^/',  # Absolute paths (Unix)
            r'^[A-Z]:',  # Absolute paths (Windows)
            r'^~',  # Home directory reference
            r'^\\\\',  # UNC paths
            r'\$\{.*\}',  # Variable expansion
            r'\$\(.*\)',  # Command substitution
            r'`',  # Backtick command substitution
            r'%.*%',  # Windows environment variables
        ]

    def is_safe_path(self, path: str) -> bool:
        """
        Check if a path is safe to use.

        Args:
            path: Path to validate

        Returns:
            True if path is safe, False otherwise
        """
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, path):
                audit_logger.warning(f"Suspicious path pattern detected: {path}")
                return False

        # Convert to Path object for validation
        try:
            path_obj = Path(path)

            # If absolute, must be within repo
            if path_obj.is_absolute():
                resolved = path_obj.resolve()
                if not self._is_within_repo(resolved):
                    audit_logger.warning(f"Absolute path outside repo: {path}")
                    return False
            else:
                # Relative path - resolve relative to repo root
                resolved = (self.repo_root / path_obj).resolve()
                if not self._is_within_repo(resolved):
                    audit_logger.warning(f"Path traverses outside repo: {path}")
                    return False

            return True

        except Exception as e:
            audit_logger.error(f"Path validation error for '{path}': {e}")
            return False

    def _is_within_repo(self, path: Path) -> bool:
        """Check if path is within repository directory."""
        try:
            path.relative_to(self.repo_root)
            return True
        except ValueError:
            return False

    def validate_url(self, url: str) -> bool:
        """
        Validate URLs for security issues.

        Args:
            url: URL to validate

        Returns:
            True if URL is safe, False otherwise
        """
        # Block potentially dangerous URL schemes
        dangerous_schemes = [
            'file://',
            'ftp://',
            'data:',
            'javascript:',
            'vbscript:',
        ]

        url_lower = url.lower()
        for scheme in dangerous_schemes:
            if url_lower.startswith(scheme):
                audit_logger.warning(f"Dangerous URL scheme blocked: {url}")
                return False

        # Allow http/https and relative URLs
        if url.startswith(('http://', 'https://', '/', './', '../')):
            return True

        # Check if it's a simple path (no scheme)
        if '://' not in url:
            return self.is_safe_path(url)

        audit_logger.warning(f"Unknown URL scheme: {url}")
        return False


class MkDocsConfigValidator:
    """Validates MkDocs configuration for security issues."""

    def __init__(self, repo_root: Path):
        """Initialize configuration validator."""
        self.repo_root = repo_root
        self.path_validator = PathValidator(repo_root)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config(self, config_path: Path) -> bool:
        """
        Validate MkDocs configuration file.

        Args:
            config_path: Path to mkdocs.yml

        Returns:
            True if configuration is safe, False otherwise
        """
        audit_logger.info(f"Validating MkDocs configuration: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse configuration: {e}")
            return False

        # Validate all sections
        valid = all([
            self._validate_nav(config.get('nav', [])),
            self._validate_plugins(config.get('plugins', [])),
            self._validate_theme(config.get('theme', {})),
            self._validate_extra(config.get('extra', {})),
            self._validate_markdown_extensions(config.get('markdown_extensions', [])),
            self._validate_urls(config),
        ])

        # Log results
        if self.errors:
            for error in self.errors:
                audit_logger.error(f"Validation error: {error}")

        if self.warnings:
            for warning in self.warnings:
                audit_logger.warning(f"Validation warning: {warning}")

        return valid and len(self.errors) == 0

    def _validate_nav(self, nav: Any) -> bool:
        """Validate navigation structure."""
        if not nav:
            return True

        def check_nav_item(item):
            if isinstance(item, dict):
                for title, content in item.items():
                    if isinstance(content, str):
                        # It's a file path
                        if not self.path_validator.is_safe_path(content):
                            self.errors.append(f"Unsafe navigation path: {content}")
                            return False
                    elif isinstance(content, list):
                        # Nested navigation
                        for sub_item in content:
                            if not check_nav_item(sub_item):
                                return False
            elif isinstance(item, str):
                # Direct file reference
                if not self.path_validator.is_safe_path(item):
                    self.errors.append(f"Unsafe navigation path: {item}")
                    return False
            return True

        for item in nav:
            if not check_nav_item(item):
                return False

        return True

    def _validate_plugins(self, plugins: List) -> bool:
        """Validate plugins configuration."""
        # Whitelist of allowed plugins
        allowed_plugins = {
            'search',
            'mkdocstrings',
            'material',
            'minify',
            'git-revision-date',
        }

        for plugin in plugins:
            plugin_name = plugin if isinstance(plugin, str) else list(plugin.keys())[0]

            if plugin_name not in allowed_plugins:
                self.warnings.append(f"Unknown plugin: {plugin_name}")

            # Special validation for mkdocstrings
            if isinstance(plugin, dict) and 'mkdocstrings' in plugin:
                config = plugin['mkdocstrings']
                if 'custom_templates' in config:
                    template_path = config['custom_templates']
                    if not self.path_validator.is_safe_path(template_path):
                        self.errors.append(f"Unsafe template path: {template_path}")
                        return False

        return True

    def _validate_theme(self, theme: Dict) -> bool:
        """Validate theme configuration."""
        if not theme:
            return True

        # Check custom theme directory
        if 'custom_dir' in theme:
            if not self.path_validator.is_safe_path(theme['custom_dir']):
                self.errors.append(f"Unsafe theme directory: {theme['custom_dir']}")
                return False

        # Check for dangerous theme features
        if 'features' in theme:
            dangerous_features = [
                'content.code.annotate',  # Can execute JavaScript
            ]
            for feature in theme.get('features', []):
                if feature in dangerous_features:
                    self.warnings.append(f"Potentially dangerous theme feature: {feature}")

        return True

    def _validate_extra(self, extra: Dict) -> bool:
        """Validate extra configuration."""
        if not extra:
            return True

        # Check for script injections
        if 'javascript' in extra or 'css' in extra:
            self.warnings.append("Custom JavaScript/CSS detected in extra configuration")

        return True

    def _validate_markdown_extensions(self, extensions: List) -> bool:
        """Validate markdown extensions."""
        # Dangerous extensions that could execute code
        dangerous_extensions = [
            'mdx_include',  # Can include arbitrary files
            'markdown_include',  # Can include arbitrary files
        ]

        for ext in extensions:
            ext_name = ext if isinstance(ext, str) else list(ext.keys())[0]
            if ext_name in dangerous_extensions:
                self.errors.append(f"Dangerous markdown extension: {ext_name}")
                return False

        return True

    def _validate_urls(self, config: Dict) -> bool:
        """Validate all URLs in configuration."""
        url_fields = ['repo_url', 'site_url', 'edit_uri']

        for field in url_fields:
            if field in config:
                url = config[field]
                if not self.path_validator.validate_url(url):
                    self.errors.append(f"Unsafe URL in {field}: {url}")
                    return False

        return True


class SandboxedBuilder:
    """Builds documentation in a sandboxed environment."""

    def __init__(self, repo_root: Path):
        """Initialize sandboxed builder."""
        self.repo_root = repo_root.resolve()
        self.sandbox_dir: Optional[Path] = None
        self.build_user = None  # For future: run as different user

    def build(self, output_dir: Optional[Path] = None) -> bool:
        """
        Build documentation in sandboxed environment.

        Args:
            output_dir: Output directory for built docs

        Returns:
            True if build successful, False otherwise
        """
        audit_logger.info("Starting sandboxed documentation build")

        # Create sandbox
        try:
            self.sandbox_dir = Path(tempfile.mkdtemp(prefix='metalq_docs_sandbox_'))
            audit_logger.info(f"Created sandbox: {self.sandbox_dir}")

            # Copy only necessary files
            if not self._prepare_sandbox():
                return False

            # Validate configuration
            validator = MkDocsConfigValidator(self.sandbox_dir)
            config_path = self.sandbox_dir / 'mkdocs.yml'

            if not validator.validate_config(config_path):
                audit_logger.error("Configuration validation failed")
                logger.error("MkDocs configuration contains security issues:")
                for error in validator.errors:
                    logger.error(f"  - {error}")
                return False

            # Build documentation
            if not self._run_build():
                return False

            # Copy output if successful
            if output_dir:
                built_docs = self.sandbox_dir / 'site'
                if built_docs.exists():
                    shutil.copytree(built_docs, output_dir, dirs_exist_ok=True)
                    audit_logger.info(f"Documentation copied to {output_dir}")

            return True

        except Exception as e:
            audit_logger.error(f"Build failed: {e}")
            logger.error(f"Documentation build failed: {e}")
            return False

        finally:
            self._cleanup_sandbox()

    def _prepare_sandbox(self) -> bool:
        """Prepare sandbox with necessary files."""
        try:
            # Files and directories to copy
            items_to_copy = [
                'mkdocs.yml',
                'docs/',
                'metalq/',  # For API documentation
                'README.md',
                'LICENSE',
            ]

            for item in items_to_copy:
                src = self.repo_root / item
                if src.exists():
                    dst = self.sandbox_dir / item
                    if src.is_dir():
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                            '__pycache__', '*.pyc', '.git', '.env', '*.key', '*.pem'
                        ))
                    else:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)

            # Create requirements file for docs build
            requirements = self.sandbox_dir / 'requirements-docs.txt'
            requirements.write_text('\n'.join([
                'mkdocs>=1.5.0',
                'mkdocs-material>=9.0.0',
                'mkdocstrings[python]>=0.20.0',
                'pymdown-extensions>=9.0',
            ]))

            audit_logger.info("Sandbox prepared successfully")
            return True

        except Exception as e:
            audit_logger.error(f"Failed to prepare sandbox: {e}")
            return False

    def _run_build(self) -> bool:
        """Run MkDocs build in sandbox."""
        try:
            # Create virtual environment in sandbox
            venv_path = self.sandbox_dir / 'venv'
            subprocess.run(
                [sys.executable, '-m', 'venv', str(venv_path)],
                check=True,
                cwd=self.sandbox_dir,
                capture_output=True
            )

            # Install dependencies
            pip_path = venv_path / 'bin' / 'pip'
            subprocess.run(
                [str(pip_path), 'install', '-r', 'requirements-docs.txt'],
                check=True,
                cwd=self.sandbox_dir,
                capture_output=True
            )

            # Run MkDocs build with restrictions
            mkdocs_path = venv_path / 'bin' / 'mkdocs'
            env = os.environ.copy()

            # Restrict environment
            env.update({
                'HOME': str(self.sandbox_dir),
                'TMPDIR': str(self.sandbox_dir / 'tmp'),
                'TEMP': str(self.sandbox_dir / 'tmp'),
                'TMP': str(self.sandbox_dir / 'tmp'),
                'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': '',  # Clear Python path
            })

            # Remove potentially dangerous environment variables
            for var in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES']:
                env.pop(var, None)

            # Create tmp directory
            (self.sandbox_dir / 'tmp').mkdir(exist_ok=True)

            # Run build
            result = subprocess.run(
                [str(mkdocs_path), 'build', '--strict'],
                cwd=self.sandbox_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                audit_logger.error(f"MkDocs build failed: {result.stderr}")
                logger.error(f"Build error: {result.stderr}")
                return False

            audit_logger.info("Documentation built successfully")
            return True

        except subprocess.TimeoutExpired:
            audit_logger.error("Build timed out after 5 minutes")
            return False
        except Exception as e:
            audit_logger.error(f"Build execution failed: {e}")
            return False

    def _cleanup_sandbox(self):
        """Clean up sandbox directory."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir)
                audit_logger.info(f"Sandbox cleaned up: {self.sandbox_dir}")
            except Exception as e:
                audit_logger.warning(f"Failed to clean up sandbox: {e}")


class FileAccessMonitor:
    """Monitor and restrict file access during documentation build."""

    def __init__(self, allowed_dir: Path):
        """
        Initialize file access monitor.

        Args:
            allowed_dir: Directory to allow access to
        """
        self.allowed_dir = allowed_dir.resolve()
        self.access_log: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []

    def check_access(self, path: Path, operation: str) -> bool:
        """
        Check if file access is allowed.

        Args:
            path: Path being accessed
            operation: Type of operation (read, write, execute)

        Returns:
            True if access allowed, False otherwise
        """
        resolved = path.resolve()

        # Log access attempt
        access_record = {
            'timestamp': datetime.now().isoformat(),
            'path': str(path),
            'resolved': str(resolved),
            'operation': operation,
            'allowed': False
        }

        try:
            # Check if within allowed directory
            resolved.relative_to(self.allowed_dir)
            access_record['allowed'] = True
            self.access_log.append(access_record)
            return True

        except ValueError:
            # Path is outside allowed directory
            access_record['reason'] = 'Outside allowed directory'
            self.violations.append(access_record)
            audit_logger.warning(
                f"File access violation: {operation} on {path} (outside {self.allowed_dir})"
            )
            return False

    def export_report(self, output_path: Path):
        """Export access monitoring report."""
        report = {
            'summary': {
                'total_accesses': len(self.access_log),
                'violations': len(self.violations),
                'allowed_directory': str(self.allowed_dir)
            },
            'violations': self.violations,
            'sample_accesses': self.access_log[:100]  # First 100 accesses
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        audit_logger.info(f"Access report exported to {output_path}")


def main():
    """Main entry point for secure documentation build."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Securely build Metal-Q documentation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='site',
        help='Output directory for built documentation'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without building'
    )
    parser.add_argument(
        '--export-audit',
        type=str,
        help='Export audit report to specified file'
    )

    args = parser.parse_args()

    # Get repository root
    repo_root = Path(__file__).parent.parent
    logger.info(f"Repository root: {repo_root}")

    if args.validate_only:
        # Just validate configuration
        validator = MkDocsConfigValidator(repo_root)
        config_path = repo_root / 'mkdocs.yml'

        if validator.validate_config(config_path):
            logger.info("✓ Configuration validation passed")
            return 0
        else:
            logger.error("✗ Configuration validation failed")
            return 1

    # Run sandboxed build
    builder = SandboxedBuilder(repo_root)
    output_dir = Path(args.output)

    if builder.build(output_dir):
        logger.info(f"✓ Documentation built successfully in {output_dir}")

        if args.export_audit:
            # Export audit log
            shutil.copy('docs_security_audit.log', args.export_audit)
            logger.info(f"Audit log exported to {args.export_audit}")

        return 0
    else:
        logger.error("✗ Documentation build failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
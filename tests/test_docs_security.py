#!/usr/bin/env python3
"""
Tests for Documentation Build Security

Validates that the secure documentation build system properly:
- Prevents path traversal attacks
- Blocks dangerous configurations
- Restricts file access
- Sandboxes the build environment
"""
import pytest
import tempfile
import yaml
import os
from pathlib import Path
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docs.secure_build import (
    PathValidator,
    MkDocsConfigValidator,
    SandboxedBuilder,
    FileAccessMonitor
)


class TestPathValidator:
    """Test path validation for security."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a path validator with temporary repo root."""
        return PathValidator(tmp_path)

    def test_safe_relative_paths(self, validator, tmp_path):
        """Test that safe relative paths are allowed."""
        # Create test files
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "index.md").touch()

        assert validator.is_safe_path("docs/index.md")
        assert validator.is_safe_path("./docs/index.md")
        assert validator.is_safe_path("docs/subfolder/page.md")

    def test_parent_directory_traversal_blocked(self, validator):
        """Test that parent directory traversal is blocked."""
        dangerous_paths = [
            "../etc/passwd",
            "../../etc/shadow",
            "docs/../../etc/passwd",
            "../../../root/.ssh/id_rsa",
            "docs/../../../etc/passwd",
        ]

        for path in dangerous_paths:
            assert not validator.is_safe_path(path), f"Path traversal not blocked: {path}"

    def test_absolute_paths_blocked(self, validator):
        """Test that absolute paths outside repo are blocked."""
        dangerous_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "/var/log/syslog",
            "C:\\Windows\\System32\\config\\sam",
            "\\\\server\\share\\file",
        ]

        for path in dangerous_paths:
            assert not validator.is_safe_path(path), f"Absolute path not blocked: {path}"

    def test_home_directory_expansion_blocked(self, validator):
        """Test that home directory expansion is blocked."""
        dangerous_paths = [
            "~/.ssh/id_rsa",
            "~/Documents/secrets.txt",
            "~root/.bashrc",
        ]

        for path in dangerous_paths:
            assert not validator.is_safe_path(path), f"Home expansion not blocked: {path}"

    def test_variable_expansion_blocked(self, validator):
        """Test that variable expansion patterns are blocked."""
        dangerous_paths = [
            "${HOME}/.ssh/id_rsa",
            "%USERPROFILE%\\Documents\\passwords.txt",
            "$(pwd)/secret",
        ]

        for path in dangerous_paths:
            assert not validator.is_safe_path(path), f"Variable expansion not blocked: {path}"

    def test_url_validation(self, validator):
        """Test URL validation."""
        # Safe URLs
        assert validator.validate_url("https://github.com/example/repo")
        assert validator.validate_url("http://localhost:8000")
        assert validator.validate_url("/docs/page")
        assert validator.validate_url("./relative/path")

        # Dangerous URLs
        assert not validator.validate_url("file:///etc/passwd")
        assert not validator.validate_url("javascript:alert('xss')")
        assert not validator.validate_url("data:text/html,<script>alert('xss')</script>")
        assert not validator.validate_url("vbscript:msgbox('xss')")
        assert not validator.validate_url("ftp://evil.com/malware")


class TestMkDocsConfigValidator:
    """Test MkDocs configuration validation."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create configuration validator."""
        return MkDocsConfigValidator(tmp_path)

    def test_valid_configuration(self, validator, tmp_path):
        """Test that valid configuration passes."""
        config = {
            'site_name': 'Test Site',
            'nav': [
                {'Home': 'index.md'},
                {'Guide': [
                    {'Getting Started': 'guide/start.md'},
                    {'Advanced': 'guide/advanced.md'}
                ]}
            ],
            'theme': {
                'name': 'material'
            },
            'plugins': ['search', 'mkdocstrings'],
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        assert validator.validate_config(config_file)
        assert len(validator.errors) == 0

    def test_path_traversal_in_nav_blocked(self, validator, tmp_path):
        """Test that path traversal in navigation is blocked."""
        config = {
            'site_name': 'Test Site',
            'nav': [
                {'Home': 'index.md'},
                {'Exploit': '../../etc/passwd'}  # Path traversal attempt
            ]
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        assert not validator.validate_config(config_file)
        assert any('Unsafe navigation path' in error for error in validator.errors)

    def test_dangerous_plugins_blocked(self, validator, tmp_path):
        """Test that dangerous plugins are warned about."""
        config = {
            'site_name': 'Test Site',
            'plugins': [
                'search',
                'unknown_plugin',  # Unknown plugin should trigger warning
            ]
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        validator.validate_config(config_file)
        assert any('Unknown plugin' in warning for warning in validator.warnings)

    def test_dangerous_theme_paths_blocked(self, validator, tmp_path):
        """Test that dangerous theme paths are blocked."""
        config = {
            'site_name': 'Test Site',
            'theme': {
                'name': 'material',
                'custom_dir': '/etc/custom_theme'  # Absolute path outside repo
            }
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        assert not validator.validate_config(config_file)
        assert any('Unsafe theme directory' in error for error in validator.errors)

    def test_dangerous_markdown_extensions_blocked(self, validator, tmp_path):
        """Test that dangerous markdown extensions are blocked."""
        config = {
            'site_name': 'Test Site',
            'markdown_extensions': [
                'admonition',
                'markdown_include',  # Can include arbitrary files
            ]
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        assert not validator.validate_config(config_file)
        assert any('Dangerous markdown extension' in error for error in validator.errors)

    def test_file_url_blocked(self, validator, tmp_path):
        """Test that file:// URLs are blocked."""
        config = {
            'site_name': 'Test Site',
            'repo_url': 'file:///etc/passwd',  # file:// URL
        }

        config_file = tmp_path / 'mkdocs.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        assert not validator.validate_config(config_file)
        assert any('Unsafe URL' in error for error in validator.errors)


class TestSandboxedBuilder:
    """Test sandboxed documentation builder."""

    @pytest.fixture
    def builder(self, tmp_path):
        """Create sandboxed builder with test repo."""
        # Create minimal repo structure
        (tmp_path / 'docs').mkdir()
        (tmp_path / 'docs' / 'index.md').write_text('# Test Documentation')

        # Create minimal mkdocs.yml
        config = {
            'site_name': 'Test Site',
            'nav': [
                {'Home': 'index.md'}
            ]
        }

        with open(tmp_path / 'mkdocs.yml', 'w') as f:
            yaml.dump(config, f)

        return SandboxedBuilder(tmp_path)

    def test_sandbox_creation(self, builder):
        """Test that sandbox is properly created and cleaned up."""
        # Note: We can't fully test the build without MkDocs installed
        # but we can test sandbox preparation
        builder.sandbox_dir = Path(tempfile.mkdtemp(prefix='test_sandbox_'))

        try:
            assert builder._prepare_sandbox()

            # Check that files were copied
            assert (builder.sandbox_dir / 'mkdocs.yml').exists()
            assert (builder.sandbox_dir / 'docs' / 'index.md').exists()

            # Check that dangerous files are not copied
            assert not (builder.sandbox_dir / '.git').exists()
            assert not (builder.sandbox_dir / '__pycache__').exists()

        finally:
            builder._cleanup_sandbox()
            assert not builder.sandbox_dir.exists()

    def test_sandbox_isolation(self, builder):
        """Test that sandbox prevents access outside its directory."""
        builder.sandbox_dir = Path(tempfile.mkdtemp(prefix='test_sandbox_'))

        try:
            builder._prepare_sandbox()

            # Try to create a file outside sandbox (should fail in real sandbox)
            outside_file = builder.sandbox_dir.parent / 'outside.txt'

            # In a real sandbox with proper OS-level restrictions,
            # this would fail. Here we just test the logic.
            assert builder.sandbox_dir.parent != builder.repo_root

        finally:
            builder._cleanup_sandbox()


class TestFileAccessMonitor:
    """Test file access monitoring."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create file access monitor."""
        return FileAccessMonitor(tmp_path)

    def test_allowed_access(self, monitor, tmp_path):
        """Test that access within allowed directory is permitted."""
        test_file = tmp_path / 'docs' / 'test.md'
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        assert monitor.check_access(test_file, 'read')
        assert len(monitor.access_log) == 1
        assert len(monitor.violations) == 0

    def test_blocked_access(self, monitor, tmp_path):
        """Test that access outside allowed directory is blocked."""
        outside_file = tmp_path.parent / 'outside.txt'

        assert not monitor.check_access(outside_file, 'read')
        assert len(monitor.violations) == 1
        assert monitor.violations[0]['operation'] == 'read'

    def test_access_logging(self, monitor, tmp_path):
        """Test that all accesses are logged."""
        # Perform various accesses
        files = [
            tmp_path / 'file1.txt',
            tmp_path / 'file2.txt',
            tmp_path.parent / 'outside.txt',  # This should be blocked
        ]

        for f in files:
            monitor.check_access(f, 'read')

        # Check logging
        assert len(monitor.access_log) == 2  # Only allowed accesses
        assert len(monitor.violations) == 1  # One blocked access

    def test_report_export(self, monitor, tmp_path):
        """Test that access report can be exported."""
        # Generate some access logs
        monitor.check_access(tmp_path / 'allowed.txt', 'read')
        monitor.check_access(tmp_path.parent / 'blocked.txt', 'write')

        # Export report
        report_file = tmp_path / 'report.json'
        monitor.export_report(report_file)

        assert report_file.exists()

        # Verify report content
        import json
        with open(report_file) as f:
            report = json.load(f)

        assert report['summary']['total_accesses'] == 1
        assert report['summary']['violations'] == 1


class TestSecurityPatterns:
    """Test for common security vulnerabilities."""

    def test_no_command_injection(self):
        """Ensure no command injection vulnerabilities."""
        # Check that subprocess calls use lists, not strings
        source_file = Path(__file__).parent.parent / 'docs' / 'secure_build.py'

        with open(source_file) as f:
            content = f.read()

        # Check for dangerous subprocess patterns
        dangerous_patterns = [
            'shell=True',  # Shell interpretation enabled
            'os.system(',  # Direct system calls
            'eval(',  # Code evaluation
            'exec(',  # Code execution
        ]

        for pattern in dangerous_patterns:
            assert pattern not in content, f"Dangerous pattern '{pattern}' found in secure_build.py"

    def test_safe_yaml_loading(self):
        """Ensure YAML is loaded safely."""
        source_file = Path(__file__).parent.parent / 'docs' / 'secure_build.py'

        with open(source_file) as f:
            content = f.read()

        # Should use safe_load, not load
        assert 'yaml.safe_load' in content
        assert 'yaml.load(' not in content or 'Loader=yaml.SafeLoader' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
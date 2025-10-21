#!/usr/bin/env python3
"""
Theme Update Script for Reality Loop Website
Updates colors, fonts, and visual elements
"""

import argparse
import re
from pathlib import Path


class ThemeUpdater:
    def __init__(self, colors=None, logo=None):
        self.colors = colors.split(",") if colors else ["#00FF66", "#000000"]
        self.logo = logo
        self.site_dir = Path("site")

    def update_theme_colors(self):
        """Update CSS theme colors across website"""
        print("🎨 Updating theme colors...")
        print(f"   Primary: {self.colors[0]}")
        print(f"   Secondary: {self.colors[1]}")

        # Update all HTML files with CSS
        html_files = list(self.site_dir.glob("*.html"))

        for html_file in html_files:
            print(f"   Updating: {html_file.name}")

            with open(html_file, "r") as f:
                content = f.read()

            # Update CSS variables
            content = self.update_css_variables(content)

            with open(html_file, "w") as f:
                f.write(content)

        print(f"✅ Theme colors updated: {len(html_files)} files")
        return True

    def update_css_variables(self, content):
        """Update CSS color variables"""
        primary_color = self.colors[0]  # #00FF66

        # Update accent color (primary green)
        content = re.sub(
            r"--accent:#[0-9a-fA-F]{6}", f"--accent:{primary_color.lower()}", content
        )

        # Update dark mode accent
        content = re.sub(
            r"--accent:#[0-9a-fA-F]{6}(?=;--card)",
            f"--accent:{primary_color.lower()}",
            content,
        )

        # Ensure consistent green/black theme
        content = re.sub(
            r"--fg:#[0-9a-fA-F]{6}", "--fg:#1a1a1a", content  # Keep dark text
        )

        # Update background for contrast
        content = re.sub(
            r"--bg:#[0-9a-fA-F]{6}", "--bg:#fafafa", content  # Light background
        )

        return content

    def update_logo(self):
        """Update logo references"""
        if not self.logo:
            return True

        print("🖼️ Updating logo references...")
        print(f"   Logo: {self.logo}")

        # Update HTML files with new logo references
        html_files = list(self.site_dir.glob("*.html"))

        for html_file in html_files:
            with open(html_file, "r") as f:
                content = f.read()

            # Update logo src attributes
            content = re.sub(
                r'src="[^"]*logo[^"]*"',
                f'src="{self.logo}"',
                content,
                flags=re.IGNORECASE,
            )

            with open(html_file, "w") as f:
                f.write(content)

        print(f"✅ Logo updated: {len(html_files)} files")
        return True


def main():
    parser = argparse.ArgumentParser(description="Update Reality Loop Website Theme")
    parser.add_argument(
        "--colors", help="Comma-separated color values (e.g., #00FF66,#000000)"
    )
    parser.add_argument("--logo", help="Logo file path")

    args = parser.parse_args()

    print("🎨 Reality Loop Theme Update")
    print("=" * 28)

    try:
        updater = ThemeUpdater(args.colors, args.logo)

        success = True
        success &= updater.update_theme_colors()

        if args.logo:
            success &= updater.update_logo()

        if success:
            print("\n🎉 Theme update complete")
            return 0
        else:
            print("\n❌ Theme update failed")
            return 1

    except Exception as e:
        print(f"\n❌ Theme update error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

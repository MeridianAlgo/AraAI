#!/usr/bin/env python3
"""
Generate OpenAPI specification file

This script generates the OpenAPI/Swagger specification and saves it to a JSON file.
Useful for documentation, client generation, and API testing tools.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ara.api.app import app


def generate_openapi_spec(output_file: str = "openapi.json"):
    """
    Generate OpenAPI specification and save to file

    Args:
        output_file: Path to output JSON file
    """
    # Get OpenAPI schema from FastAPI app
    openapi_schema = app.openapi()

    # Save to file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print(f"✅ OpenAPI specification generated: {output_path.absolute()}")
    print(f"📊 Endpoints: {len(openapi_schema.get('paths', {}))}")
    print(f"🏷️  Tags: {len(openapi_schema.get('tags', []))}")
    print(f"📝 Version: {openapi_schema.get('info', {}).get('version', 'unknown')}")

    return openapi_schema


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI specification")
    parser.add_argument(
        "-o",
        "--output",
        default="docs/api/openapi.json",
        help="Output file path (default: docs/api/openapi.json)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    args = parser.parse_args()

    # Generate specification
    spec = generate_openapi_spec(args.output)

    if args.pretty:
        print("\n" + "=" * 60)
        print("OpenAPI Specification Preview")
        print("=" * 60)
        print(json.dumps(spec, indent=2)[:1000] + "...")


if __name__ == "__main__":
    main()

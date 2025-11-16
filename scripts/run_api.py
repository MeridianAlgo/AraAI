"""
ARA AI API Server Launcher
Simple script to start the FastAPI server
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Start the API server"""
    print("=" * 60)
    print("ARA AI Prediction API Server")
    print("=" * 60)
    print()
    print("Starting FastAPI server with Uvicorn...")
    print()
    print("The API will be available at:")
    print("  • API: http://localhost:8000")
    print("  • Interactive Docs: http://localhost:8000/docs")
    print("  • Alternative Docs: http://localhost:8000/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Check if uvicorn is installed
        try:
            import uvicorn
        except ImportError:
            print("ERROR: uvicorn is not installed")
            print("Please install it with: pip install uvicorn[standard]")
            sys.exit(1)
        
        # Check if fastapi is installed
        try:
            import fastapi
        except ImportError:
            print("ERROR: fastapi is not installed")
            print("Please install it with: pip install fastapi")
            sys.exit(1)
        
        # Start the server
        uvicorn.run(
            "ara.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

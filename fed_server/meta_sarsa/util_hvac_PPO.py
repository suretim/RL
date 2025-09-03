import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import flask
    print(f"Flask version: {flask.__version__}")
    print(f"Flask location: {flask.__file__}")
except ImportError:
    print("Flask is NOT installed")
except Exception as e:
    print(f"Error importing Flask: {e}")





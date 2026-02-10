import subprocess
import sys

print("🚀 Starting EcoVision Backend + Dashboard...")

subprocess.Popen(
    ["uvicorn", "backend.api:app", "--reload", "--port", "8000"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

subprocess.Popen(
    ["streamlit", "run", "dashboard/app.py", "--server.port", "8501"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

print("✅ Backend: http://127.0.0.1:8000")
print("✅ Dashboard: http://localhost:8502")

input("Press ENTER to stop everything...")

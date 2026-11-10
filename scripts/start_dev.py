"""
Script utilitário para iniciar simultaneamente a API Backend (FastAPI na porta 8000)
e o Frontend Web (Vite na porta 3000) em modo de desenvolvimento local.
"""
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    print("=" * 70)
    print("   ⚽ BRASILEIRÃO PREDICTOR AI — INICIANDO AMBIENTE LOCAL")
    print("=" * 70)

    # 1. Start Backend FastAPI
    print("\n[1/2] Iniciando Backend FastAPI em http://localhost:8000...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.api.main:app", "--reload", "--port", "8000"],
        cwd=str(BASE_DIR)
    )

    # Give backend 1 second to start
    time.sleep(1.5)

    # 2. Start Frontend Vite
    web_dir = BASE_DIR / "apps" / "web"
    print("[2/2] Iniciando Frontend React/Vite em http://localhost:3000...")
    is_windows = sys.platform.startswith("win")
    npm_cmd = "npm.cmd" if is_windows else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(web_dir)
    )

    print("\n" + "=" * 70)
    print("   🚀 AMBIENTE EM EXECUÇÃO:")
    print("   • API Swagger Docs: http://localhost:8000/docs")
    print("   • Frontend Web:     http://localhost:3000")
    print("   Pressione CTRL+C no terminal para encerrar os servidores.")
    print("=" * 70)

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nEncerrando servidores...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Servidores finalizados.")

if __name__ == "__main__":
    main()

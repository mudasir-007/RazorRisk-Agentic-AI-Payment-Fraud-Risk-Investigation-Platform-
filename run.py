import uvicorn
from config import HOST, PORT, DEBUG
from utils.logger import get_logger

logger = get_logger("runner")

def main():
    logger.info(f"Launching RazorRisk Server at http://{HOST}:{PORT}")
    logger.info(f"Interactive Dashboard available at: http://localhost:{PORT}/dashboard/")
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=DEBUG)


if __name__ == "__main__":
    main()

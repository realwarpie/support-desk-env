from env import app
import uvicorn


def main():
    uvicorn.run(
        "env:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )


if __name__ == "__main__":
    main()

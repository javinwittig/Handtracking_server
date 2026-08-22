from fastapi import FastAPI, Request
import cv2
import numpy as np
import threading
import uvicorn

from hand_analyizer import analyze_frame

app = FastAPI()

latest_frame = None
latest_result = None
frame_lock = threading.Lock()


@app.post("/upload_frame")
async def upload_frame(request: Request):
    global latest_frame, latest_result
    raw_data = await request.body()

    np_array = np.frombuffer(raw_data, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"x": None, "y": None, "found": False}

    result = analyze_frame(frame)

    with frame_lock:
        latest_frame = frame
        latest_result = result

    return result


def display_loop():
    global latest_frame, latest_result
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            result = latest_result

        if frame is not None:
            if result is not None and result.get("found"):
                cv2.circle(
                    frame,
                    (result["x_px"], result["y_px"]),
                    radius=10,
                    color=(0, 0, 255),  # BGR -> Rot
                    thickness=-1,       # gefüllt
                )

            cv2.imshow("Empfangener Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
        daemon=True
    )
    server_thread.start()

    display_loop()
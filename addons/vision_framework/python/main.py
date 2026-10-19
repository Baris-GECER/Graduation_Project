import asyncio
import json
import websockets
from modules.base_processor import BaseProcessor
from modules.hand_tracker import HandTracker

class VisionService:
    def __init__(self):
        self.processor = BaseProcessor()
        self.hand_tracker = HandTracker()
        server_cfg = self.processor.config["server"]
        self.host = server_cfg["host"]
        self.port = server_cfg["port"]

    async def handler(self, websocket):
        print(f"[VisionService] Godot client connected.")
        try:
            while True:
                frame = self.processor.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                hands_data = []
                if self.processor.config["modules"].get("hand_tracking", True):
                    hands_data = self.hand_tracker.process(frame)

                # Show what Python sees in a native desktop window
                self.processor.show_debug_window(frame, hands_data)

                payload = {
                    "type": "frame_data",
                    "hands": hands_data
                }

                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.005)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"[VisionService] Godot client disconnected.")
        except Exception as e:
            print(f"[VisionServiceError] {e}")

    async def start(self):
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"[VisionService] Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()

if __name__ == "__main__":
    service = VisionService()
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        service.processor.release()
        print("[VisionService] Shutting down gracefully.")
import cv2
import numpy as np
import socket
import time
from scipy.spatial.transform import Rotation as R

# Camera parameters
FOCAL_LENGTH = 600  # pixels
KNOWN_WIDTH = 0.2   # meters
MESH_SIZE = 5       # Size of grid cell in pixels
MIN_MATCHES = 20

class AltitudeEstimator:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp, self.prev_des = None, None
        self.prev_altitude = 1.0  # Initial altitude

    def estimate_altitude(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_des is None or des is None or len(kp) < MIN_MATCHES:
            self.prev_kp, self.prev_des = kp, des
            return frame  # Skip frame if not enough matches

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        if len(matches) < MIN_MATCHES:
            self.prev_kp, self.prev_des = kp, des
            return frame  # Skip if insufficient matches

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        scale_factor = self._compute_mesh_scale(pts_prev, pts_curr, frame.shape)
        altitude_change = self.prev_altitude * scale_factor
        current_altitude = max(0.1, altitude_change)

        self._draw_info(frame, current_altitude, scale_factor)
        self.prev_altitude = current_altitude
        self.prev_kp, self.prev_des = kp, des
        return frame

    def _compute_mesh_scale(self, pts1, pts2, shape):
        h, w = shape[:2]
        mesh_w, mesh_h = w // MESH_SIZE, h // MESH_SIZE

        scale_changes = []
        for i in range(MESH_SIZE):
            for j in range(MESH_SIZE):
                mask1 = (pts1[:, 0] > i * mesh_w) & (pts1[:, 0] < (i + 1) * mesh_w) & \
                        (pts1[:, 1] > j * mesh_h) & (pts1[:, 1] < (j + 1) * mesh_h)
                mask2 = (pts2[:, 0] > i * mesh_w) & (pts2[:, 0] < (i + 1) * mesh_w) & \
                        (pts2[:, 1] > j * mesh_h) & (pts2[:, 1] < (j + 1) * mesh_h)
                
                region_pts1 = pts1[mask1]
                region_pts2 = pts2[mask2]

                if len(region_pts1) > 2 and len(region_pts2) > 2:
                    avg_dist1 = np.mean(np.linalg.norm(region_pts1 - np.mean(region_pts1, axis=0), axis=1))
                    avg_dist2 = np.mean(np.linalg.norm(region_pts2 - np.mean(region_pts2, axis=0), axis=1))
                    scale_changes.append(avg_dist2 / avg_dist1 if avg_dist1 > 0 else 1.0)

        return np.mean(scale_changes) if scale_changes else 1.0

    def _draw_info(self, frame, altitude, scale_factor):
        cv2.putText(frame, f"Altitude: {altitude+2:.2f}cm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Scale Factor: {scale_factor:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

# --------------------------- #
#  Raspberry Pi Connection    #
# --------------------------- #
def main():
    HOST = ''  # Listen on all interfaces
    PORT = 5000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for connection...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    data = b""
    estimator = AltitudeEstimator()
    start_time = time.time()  # Start time for 5-second stream

    while time.time() - start_time < 2:  # Stop after 5 seconds
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

        while True:
            a = data.find(b'\xff\xd8')  # JPEG start
            b = data.find(b'\xff\xd9')  # JPEG end
            if a == -1 or b == -1:
                break  # Wait for full frame

            jpg = data[a:b + 2]
            data = data[b + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                processed = estimator.estimate_altitude(frame)
                cv2.imshow("Altitude Estimation", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    print("Streaming ended after 5 seconds.")
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import socket
import numpy as np
import cv2
import time
        
# Server settings
HOST = ''  # Listen on all interfaces
PORT = 5000

# Camera Intrinsics (Replace with calibrated values)
camera_matrix = np.array([[600, 0, 320],  # fx, 0, cx
                          [0, 600, 240],  # 0, fy, cy
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Create socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Waiting for connection...")

conn, addr = server_socket.accept()
print(f"Connected by {addr}")

data = b""
orb = cv2.ORB_create(nfeatures=500)

# Read first frame
first_frame = None
kp_old, des_old = None, None

start_time = time.time()  # Start timer

while time.time() - start_time < 30:  # Run for 10 seconds
    packet = conn.recv(4096)  # Receive data
    if not packet:
        break
    data += packet

    # Find JPEG frame boundaries
    start = data.find(b'\xff\xd8')  # JPEG start
    end = data.find(b'\xff\xd9')  # JPEG end
    if start != -1 and end != -1:
        jpg = data[start:end+2]  # Extract JPEG
        data = data[end+2:]  # Remove processed data
        
        # Decode frame
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        
        gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if first_frame is None:
            first_frame = gray_new
            kp_old, des_old = orb.detectAndCompute(gray_new, None)
            continue

        # ORB Feature Detection
        kp_new, des_new = orb.detectAndCompute(gray_new, None)

        # Feature Matching using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_old, des_new)
        matches = sorted(matches, key=lambda x: x.distance)[:50]  # Select top 50 matches

        if len(matches) > 10:
            pts_old = np.float32([kp_old[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_new = np.float32([kp_new[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Compute Essential Matrix & Recover Pose
            E, mask = cv2.findEssentialMat(pts_new, pts_old, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, pts_new, pts_old, camera_matrix)

            # Extract Roll, Pitch, Yaw
            roll = np.arctan2(R[2, 1], R[2, 2]) * (180 / np.pi)
            pitch = np.arcsin(-R[2, 0]) * (180 / np.pi)
            yaw = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)

            # Print values in console
            print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")

            # Display RPY values on video
            cv2.putText(frame, f"Roll: {roll:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw motion tracks
            for i, (new, old) in enumerate(zip(pts_new, pts_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        # Show Output
        cv2.imshow('RPY Estimation', frame)

        # Update for next iteration
        kp_old, des_old = kp_new, des_new

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close connection after 10 seconds
conn.close()
server_socket.close()
cv2.destroyAllWindows()

print("Live stream ended after 10 seconds.")

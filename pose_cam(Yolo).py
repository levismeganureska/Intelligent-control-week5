from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Gunakan CAP_DSHOW untuk kompatibilitas lebih baik di Windows

# Periksa apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Deteksi pose
    results = model(frame)

    # Tampilkan hasil dengan anotasi
    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame

    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
cv2.destroyAllWindows()

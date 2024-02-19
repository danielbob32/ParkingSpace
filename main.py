import cv2

liveVideo = cv2.VideoCapture("rtsp://id:pass@camera_ip:554/H.264") #edit here

while True:
    ret, frame = liveVideo.read()

    cv2.imshow("RTSP", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

liveVideo.release()
cv2.destroyAllWindows()

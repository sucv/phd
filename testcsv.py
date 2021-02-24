import cv2

video = cv2.VideoCapture("E:\\Mahnob_full\\Sessions\\2\\P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_2_combined.avi")
print(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

video = cv2.VideoCapture("E:\\Mahnob_full\\Sessions\\2\\P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_2_combined_fps64.avi")
print(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))